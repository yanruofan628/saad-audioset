#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
两阶段消融：Audio train → freeze ear → EEG-only residual.

Stage 1: AnatomicalAcousticPriority，CE(S)，早停盯 S，按折存 ear。
Stage 2: 加载并冻结 ear；B 只看 EEG；D = S + σ(α) B；只训 CE(D)。

划分 / 种子 / shuffle 与 nomargin_seeded 一致。无 margin。
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eeg_audio_residual import (
    SUBJECTS_CONFIG,
    OUTPUT_DIR,
    SAMPLING_RATE,
    load_all_mappings,
    load_pair_stereo_paths,
    align_single_subject_data,
    _select_audio_for_trials,
    _build_single_subject_fold_data,
)
from am_fm_salience import (
    build_am_fm_cache,
    align_am_fm_to_pair_names,
    batch_am_fm_from_cache,
    AM_FM_CACHE_TAG,
)
from anatomical_acoustic_priority import AnatomicalAcousticPriority, FrozenAudioEEGResidual
from linear_regression_loudness_models import load_stereo_2s
from prime_ablation_ctrl import (
    SUBJECTS_23,
    fold_seed,
    seed_everything,
    epoch_perm,
    seed_batch,
    fit_lambda_shrinkage,
    collect_sb,
)
from run_eeg_only_category_heldout_23 import (
    build_global_pairing_type_fold_map,
    build_category_heldout_folds,
    _fix01,
)

_env_subjects = os.environ.get("SUBJECTS", "").strip()
SUBJECTS = (
    [s.strip() for s in _env_subjects.split(",") if s.strip()]
    if _env_subjects
    else list(SUBJECTS_23)
)

N_FOLDS = 6
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
_s2lr = os.environ.get("STAGE2_LR", "").strip()
STAGE2_LR = float(_s2lr) if _s2lr else LR
_s2wd = os.environ.get("STAGE2_WD", "").strip()
STAGE2_WD = float(_s2wd) if _s2wd else 1e-4
EEG_MODE = "multi_scale_dual"
_gamma = os.environ.get("SHRINK_GAMMA", "").strip()
SHRINK_GAMMA = float(_gamma) if _gamma else 0.05
RUN_TAG = os.environ.get("RUN_TAG", "").strip() or "nomargin_seeded"
SKIP_AUDIO_TRAIN = os.environ.get("SKIP_AUDIO_TRAIN", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
_load_ear = os.environ.get("LOAD_EAR_DIR", "").strip()
LOAD_EAR_DIR = _load_ear or os.path.join(
    OUTPUT_DIR,
    "individual",
    "category_heldout_twostage_freeze_nomargin_seeded",
    "ear_ckpts",
)

NO_GATE = os.environ.get("NO_GATE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)

_max_folds = os.environ.get("MAX_FOLDS", "").strip()
MAX_FOLDS = int(_max_folds) if _max_folds.isdigit() else None


def _eval_logits(logits, y):
    pred = logits.argmax(1).cpu().numpy()
    prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
    }


def train_audio_fold(fold_data, am_fm, device, fold_idx: int):
    train_y = _fix01(fold_data["train_labels"])
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

    seed_everything(fold_seed(fold_idx))
    model = AnatomicalAcousticPriority().to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    _, _, mel_l_v, mel_r_v, s_l_v, s_r_v = batch_am_fm_from_cache(am_fm, va_pi, device)
    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = epoch_perm(n_train, fold_idx, epoch)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            seed_batch(fold_idx, epoch, b)
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            logits = model(mel_l, mel_r, s_l, s_r)
            crit(logits, yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(mel_l_v, mel_r_v, s_l_v, s_r_v)
            pred = logits.argmax(1).cpu().numpy()
        acc = accuracy_score(val_y, pred)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(mel_l_v, mel_r_v, s_l_v, s_r_v)
    metrics = _eval_logits(logits, val_y)
    return model, {
        "val_balanced_accuracy_audio": metrics["bacc"],
        "val_accuracy_audio": metrics["acc"],
        "val_auc_audio": metrics["auc"],
    }


def load_frozen_audio(ckpt_path, fold_data, am_fm, device, fold_idx: int):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"缺少冻结 ear，不重训音频: {ckpt_path}")
    seed_everything(fold_seed(fold_idx))
    model = AnatomicalAcousticPriority().to(device)
    model.ear.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    # Skip frozen-S eval; ear ckpts already exist.
    return model, {
        "val_balanced_accuracy_audio": float("nan"),
        "val_accuracy_audio": float("nan"),
        "val_auc_audio": float("nan"),
    }


def train_residual_fold(audio_model, fold_data, am_fm, device, fold_idx: int):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

    seed_everything(fold_seed(fold_idx) + 777)
    model = FrozenAudioEEGResidual(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
        sfreq=SAMPLING_RATE,
        eeg_mode=EEG_MODE,
        use_gate=not NO_GATE,
    ).to(device)
    model.ear.load_state_dict(audio_model.ear.state_dict())
    model.freeze_ear()
    opt = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE2_LR,
        weight_decay=STAGE2_WD,
    )
    crit = nn.CrossEntropyLoss()

    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)
    _, _, mel_l_v, mel_r_v, s_l_v, s_r_v = batch_am_fm_from_cache(am_fm, va_pi, device)
    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        model.ear.eval()
        perm = epoch_perm(n_train, fold_idx, epoch)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            seed_batch(fold_idx, epoch, b)
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            D = model(Xe, mel_l, mel_r, s_l, s_r)
            crit(D, yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            D = model(X_val, mel_l_v, mel_r_v, s_l_v, s_r_v)
            pred = D.argmax(1).cpu().numpy()
        acc = accuracy_score(val_y, pred)
        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("ear.")
            }
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    model.ear.eval()
    with torch.no_grad():
        D, S, B, g = model(
            X_val, mel_l_v, mel_r_v, s_l_v, s_r_v, return_parts=True
        )
    m_s = _eval_logits(S, val_y)
    m_d = _eval_logits(D, val_y)

    if NO_GATE:
        return {
            "val_balanced_accuracy_frozen_s": m_s["bacc"],
            "val_balanced_accuracy_twostage": m_d["bacc"],
            "val_accuracy_twostage": m_d["acc"],
            "val_auc_twostage": m_d["auc"],
            "val_balanced_accuracy_twostage_shrink": float("nan"),
            "val_accuracy_twostage_shrink": float("nan"),
            "val_auc_twostage_shrink": float("nan"),
            "val_gate_mean": 1.0,
            "lambda_star": float("nan"),
            "lambda_obj": float("nan"),
            "shrink_gamma": float("nan"),
        }

    S_tr, B_tr, y_tr = collect_sb(
        model, train_eeg, tr_pi, train_y, am_fm, device, BATCH_SIZE, joint=False
    )
    S_va, B_va, y_va = collect_sb(
        model, val_eeg, va_pi, val_y, am_fm, device, BATCH_SIZE, joint=False
    )
    lam, obj = fit_lambda_shrinkage(S_tr, B_tr, y_tr, gamma=SHRINK_GAMMA)
    D_sh = S_va + lam * B_va
    m_sh = _eval_logits(D_sh, val_y)
    return {
        "val_balanced_accuracy_frozen_s": m_s["bacc"],
        "val_balanced_accuracy_twostage": m_d["bacc"],
        "val_accuracy_twostage": m_d["acc"],
        "val_auc_twostage": m_d["auc"],
        "val_balanced_accuracy_twostage_shrink": m_sh["bacc"],
        "val_accuracy_twostage_shrink": m_sh["acc"],
        "val_auc_twostage_shrink": m_sh["auc"],
        "val_gate_mean": float(g.detach().mean().cpu()),
        "lambda_star": float(lam),
        "lambda_obj": float(obj),
        "shrink_gamma": float(SHRINK_GAMMA),
    }


def run_subject(subject_name, am_fm, pair_name_to_idx, type_map, out_dir, ckpt_dir, device):
    print(f"\n{'=' * 60}\n被试: {subject_name}\n{'=' * 60}")
    consistent_eeg, consistent_labels, consistent_trials, meta, _ = align_single_subject_data(
        subject_name, filter_consistent=False, project_root=project_root
    )
    n_pairs = len(pair_name_to_idx)
    dummy_audio = np.zeros((n_pairs, 2), dtype=np.float32)
    consistent_eeg, consistent_labels, consistent_trials, pair_indices = _select_audio_for_trials(
        consistent_eeg, consistent_labels, consistent_trials, dummy_audio, pair_name_to_idx
    )
    folds, shuffled_indices = build_category_heldout_folds(consistent_trials, type_map)

    fold_rows = []
    for fold_idx, val_idx in enumerate(folds, 1):
        if MAX_FOLDS is not None and fold_idx > MAX_FOLDS:
            break
        if len(val_idx) == 0:
            print(f"  Fold {fold_idx}: empty val, skip")
            continue
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=False)
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} train={len(train_idx)} val={len(val_idx)} ---")
        fold_data = _build_single_subject_fold_data(
            consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
        )
        if SKIP_AUDIO_TRAIN:
            ckpt_path = os.path.join(LOAD_EAR_DIR, f"{subject_name}_fold{fold_idx}_ear.pt")
            audio_model, m_audio = load_frozen_audio(
                ckpt_path, fold_data, am_fm, device, fold_idx
            )
        else:
            audio_model, m_audio = train_audio_fold(fold_data, am_fm, device, fold_idx)
            ckpt_path = os.path.join(ckpt_dir, f"{subject_name}_fold{fold_idx}_ear.pt")
            torch.save(audio_model.ear.state_dict(), ckpt_path)
        m_res = train_residual_fold(audio_model, fold_data, am_fm, device, fold_idx)
        row = {
            "subject": subject_name,
            "cv_scheme": "category_heldout",
            "fold_idx": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            **m_audio,
            **m_res,
        }
        fold_rows.append(row)
        if NO_GATE:
            print(
                f"  Fold {fold_idx}: twostage_D=S+B={m_res['val_balanced_accuracy_twostage']:.4f}"
            )
        else:
            print(
                f"  Fold {fold_idx}: audio={m_audio['val_balanced_accuracy_audio']:.4f} | "
                f"frozen_S={m_res['val_balanced_accuracy_frozen_s']:.4f} | "
                f"twostage_D={m_res['val_balanced_accuracy_twostage']:.4f} | "
                f"shrink={m_res['val_balanced_accuracy_twostage_shrink']:.4f} | "
                f"g={m_res['val_gate_mean']:.3f} | λ*={m_res['lambda_star']:.3f}"
            )

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(fold_rows)
    df.to_csv(
        os.path.join(out_dir, f"{subject_name}_twostage_cv_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    b_a = df["val_balanced_accuracy_audio"].mean()
    b_s = df["val_balanced_accuracy_frozen_s"].mean()
    b_d = df["val_balanced_accuracy_twostage"].mean()
    b_sh = df["val_balanced_accuracy_twostage_shrink"].mean()
    print(
        f"单人 category_heldout - {subject_name}\n"
        f"audio BACC: {b_a:.4f}\n"
        f"frozen S BACC: {b_s:.4f}\n"
        f"twostage D BACC: {b_d:.4f}\n"
        f"twostage shrink BACC: {b_sh:.4f}  λ*={df['lambda_star'].mean():.3f}\n"
        f"delta (D-audio): {b_d - b_a:+.4f}\n"
        f"delta (shrink-audio): {b_sh - b_a:+.4f}\n"
    )
    return {
        "subject": subject_name,
        "bacc_audio": float(b_a),
        "bacc_frozen_s": float(b_s),
        "bacc_twostage": float(b_d),
        "bacc_twostage_shrink": float(b_sh),
        "delta_d_minus_audio": float(b_d - b_a),
        "delta_shrink_minus_audio": float(b_sh - b_a),
        "gate_mean": float(df["val_gate_mean"].mean()),
        "lambda_star_mean": float(df["lambda_star"].mean()),
    }


def main():
    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("subjects:", SUBJECTS)
    print("n_subjects:", len(SUBJECTS))
    print("EEG_MODE:", EEG_MODE)
    print("RUN_TAG:", RUN_TAG)
    print("SHRINK_GAMMA:", SHRINK_GAMMA)
    print("SKIP_AUDIO_TRAIN:", SKIP_AUDIO_TRAIN)
    print("LOAD_EAR_DIR:", LOAD_EAR_DIR)
    print("stage1 LR:", LR)
    print("stage2 LR:", STAGE2_LR)
    print("stage2 WD:", STAGE2_WD)
    print("NO_GATE:", NO_GATE)
    print("AM_FM_CACHE_TAG:", AM_FM_CACHE_TAG)
    if MAX_FOLDS is not None:
        print(f"MAX_FOLDS={MAX_FOLDS}")

    mappings, _ = load_all_mappings()
    pair_names = sorted({v for m in mappings for v in m.values()})
    stereo_path_map = load_pair_stereo_paths()
    pair_name_to_idx = {pn: i for i, pn in enumerate(pair_names)}

    cache_dir = os.path.join(project_root, "audio_features_cache")
    am_fm_raw = build_am_fm_cache(
        pair_names, stereo_path_map, load_stereo_2s, cache_dir, cache_tag=AM_FM_CACHE_TAG
    )
    am_fm = align_am_fm_to_pair_names(am_fm_raw, pair_names)
    type_map, _ = build_global_pairing_type_fold_map(pair_names)

    out_dir = os.path.join(
        OUTPUT_DIR, "individual", f"category_heldout_twostage_freeze_{RUN_TAG}"
    )
    ckpt_dir = os.path.join(out_dir, "ear_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        try:
            rows.append(
                run_subject(
                    subj, am_fm, pair_name_to_idx, type_map, out_dir, ckpt_dir, device
                )
            )
        except Exception as e:
            print(f"\n[FAIL] {subj}: {e}")
            traceback.print_exc()

    if rows:
        sdf = pd.DataFrame(rows)
        summary_name = f"n{len(rows)}_twostage_summary.csv"
        sdf.to_csv(os.path.join(out_dir, summary_name), index=False, encoding="utf-8-sig")
        print(f"\n=== {len(rows)} 人汇总 ({summary_name}) ===")
        print(sdf.to_string(index=False))
        print(
            f"group audio={sdf.bacc_audio.mean():.4f} | "
            f"frozen_S={sdf.bacc_frozen_s.mean():.4f} | "
            f"twostage_D={sdf.bacc_twostage.mean():.4f} | "
            f"shrink={sdf.bacc_twostage_shrink.mean():.4f} | "
            f"λ*={sdf.lambda_star_mean.mean():.3f} | "
            f"delta_D={sdf.delta_d_minus_audio.mean():+.4f} | "
            f"delta_shrink={sdf.delta_shrink_minus_audio.mean():+.4f}"
        )


if __name__ == "__main__":
    main()
