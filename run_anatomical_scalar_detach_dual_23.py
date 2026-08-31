#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
23 人 category-heldout（6 折）：解剖锚定 + scalar-gate detach 融合 + multi_scale_dual EEG。
D = S + g·B；训练 D_train = S.detach() + g·B。

输出：.../individual/category_heldout_anat_scalar_detach_dual_v1/
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
    CV_SEED,
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
    content_priority_margin_loss,
    AM_FM_CACHE_TAG,
    DEFAULT_CONTENT_PRIO_LOSS_WEIGHT,
)
from anatomical_acoustic_priority import ResidualScalarGateDetachAnatomical
from linear_regression_loudness_models import load_stereo_2s, parse_pair_name
from prime_ablation_ctrl import (
    SUBJECTS_23,
    fold_seed,
    seed_everything,
    epoch_perm,
    seed_batch,
    fit_lambda_shrinkage,
    collect_sb,
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
LAMBDA_FUSION = 1.0
_prio_w = os.environ.get("PRIO_W", "").strip()
PRIO_W = float(_prio_w) if _prio_w else DEFAULT_CONTENT_PRIO_LOSS_WEIGHT
EEG_MODE = "multi_scale_dual"
_gamma = os.environ.get("SHRINK_GAMMA", "").strip()
SHRINK_GAMMA = float(_gamma) if _gamma else 0.05
RUN_TAG = os.environ.get("RUN_TAG", "").strip() or (
    "nomargin_seeded" if PRIO_W == 0.0 else "v1"
)

_max_folds = os.environ.get("MAX_FOLDS", "").strip()
MAX_FOLDS = int(_max_folds) if _max_folds.isdigit() else None


def get_pairing_type_key(pair_name):
    if not pair_name:
        return None
    parse_result = parse_pair_name(pair_name)
    if parse_result is None:
        return None
    left_cat, _, right_cat, _, _ = parse_result
    if left_cat is None or right_cat is None:
        return None
    return tuple(sorted([left_cat, right_cat]))


def build_global_pairing_type_fold_map(pair_names, n_folds=N_FOLDS, seed=CV_SEED):
    all_types = sorted(
        {key for pn in pair_names if (key := get_pairing_type_key(pn)) is not None}
    )
    if not all_types:
        raise ValueError("无法从 pair_names 提取 pairing type")
    rng = np.random.RandomState(seed)
    types_shuffled = list(all_types)
    rng.shuffle(types_shuffled)
    idx_groups = np.array_split(np.arange(len(types_shuffled)), n_folds)
    fold_type_groups = [[types_shuffled[int(i)] for i in idx] for idx in idx_groups]
    type_to_fold = {}
    for fold_idx, group in enumerate(fold_type_groups):
        for pairing_type in group:
            type_to_fold[pairing_type] = fold_idx
    print(f"\n=== Category-held-out pairing types: {len(all_types)} → {n_folds} folds ===")
    for fold_idx, group in enumerate(fold_type_groups, 1):
        preview = ", ".join(f"{a}×{b}" for a, b in group[:4])
        print(f"  Fold {fold_idx} held-out ({len(group)}): {preview}")
    return type_to_fold, fold_type_groups


def build_category_heldout_folds(consistent_trials, pairing_type_fold_map, n_folds=N_FOLDS):
    fold_indices = {i: [] for i in range(n_folds)}
    for idx, trial in enumerate(consistent_trials):
        pair_name = trial.get("pair_name") or trial.get("original_name")
        pairing_type = get_pairing_type_key(pair_name)
        if pairing_type is None or pairing_type not in pairing_type_fold_map:
            continue
        fold_indices[pairing_type_fold_map[pairing_type]].append(idx)
    folds = [np.array(fold_indices[i], dtype=np.int32) for i in range(n_folds)]
    shuffled_indices = np.arange(len(consistent_trials), dtype=np.int32)
    print("  fold val sizes:", [len(f) for f in folds])
    return folds, shuffled_indices


def _fix01(y):
    y = y.astype(np.int64)
    if set(np.unique(y).tolist()) <= {1, 2}:
        y = y - 1
    return y


def train_fold(fold_data, am_fm, device, fold_idx: int):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

    seed_everything(fold_seed(fold_idx))
    model = ResidualScalarGateDetachAnatomical(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
        sfreq=SAMPLING_RATE,
        eeg_mode=EEG_MODE,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)
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
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            D, S, B, g, p_l, p_r = model(
                Xe, mel_l, mel_r, s_l, s_r, train_detach=True, return_parts=True
            )
            loss_audio = crit(S, yb)
            if PRIO_W > 0:
                loss_audio = loss_audio + PRIO_W * content_priority_margin_loss(p_l, p_r, yb)
            loss = loss_audio + LAMBDA_FUSION * crit(D, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            D = model(X_val, mel_l_v, mel_r_v, s_l_v, s_r_v, train_detach=False)
            pred = D.argmax(1).cpu().numpy()
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
        D, S, B, g, p_l, p_r = model(
            X_val, mel_l_v, mel_r_v, s_l_v, s_r_v, train_detach=False, return_parts=True
        )
        pred_d = D.argmax(1).cpu().numpy()
        prob_d = F.softmax(D, dim=1)[:, 1].cpu().numpy()
        pred_s = S.argmax(1).cpu().numpy()
        prob_s = F.softmax(S, dim=1)[:, 1].cpu().numpy()

    S_tr, B_tr, y_tr = collect_sb(
        model, train_eeg, tr_pi, train_y, am_fm, device, BATCH_SIZE, joint=True
    )
    S_va, B_va, y_va = collect_sb(
        model, val_eeg, va_pi, val_y, am_fm, device, BATCH_SIZE, joint=True
    )
    lam, obj = fit_lambda_shrinkage(S_tr, B_tr, y_tr, gamma=SHRINK_GAMMA)
    D_sh = S_va + lam * B_va
    pred_sh = D_sh.argmax(1).cpu().numpy()
    prob_sh = F.softmax(D_sh, dim=1)[:, 1].cpu().numpy()
    return {
        "val_balanced_accuracy_audio_head": float(balanced_accuracy_score(val_y, pred_s)),
        "val_accuracy_audio_head": float(accuracy_score(val_y, pred_s)),
        "val_auc_audio_head": float(roc_auc_score(val_y, prob_s))
        if len(np.unique(val_y)) > 1
        else float("nan"),
        "val_balanced_accuracy_fusion": float(balanced_accuracy_score(val_y, pred_d)),
        "val_accuracy_fusion": float(accuracy_score(val_y, pred_d)),
        "val_auc_fusion": float(roc_auc_score(val_y, prob_d))
        if len(np.unique(val_y)) > 1
        else float("nan"),
        "val_balanced_accuracy_fusion_shrink": float(balanced_accuracy_score(val_y, pred_sh)),
        "val_accuracy_fusion_shrink": float(accuracy_score(val_y, pred_sh)),
        "val_auc_fusion_shrink": float(roc_auc_score(val_y, prob_sh))
        if len(np.unique(val_y)) > 1
        else float("nan"),
        "lambda_star": float(lam),
        "lambda_obj": float(obj),
        "shrink_gamma": float(SHRINK_GAMMA),
        "val_gate_mean": float(g.detach().mean().cpu()),
    }


def run_subject(subject_name, am_fm, pair_name_to_idx, type_map, out_dir, device):
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
        m = train_fold(fold_data, am_fm, device, fold_idx)
        row = {
            "subject": subject_name,
            "cv_scheme": "category_heldout",
            "fold_idx": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            **m,
        }
        fold_rows.append(row)
        print(
            f"  Fold {fold_idx}: S={m['val_balanced_accuracy_audio_head']:.4f} | "
            f"fusion D={m['val_balanced_accuracy_fusion']:.4f} | "
            f"shrink={m['val_balanced_accuracy_fusion_shrink']:.4f} | "
            f"λ*={m['lambda_star']:.3f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(fold_rows)
    df.to_csv(
        os.path.join(out_dir, f"{subject_name}_scalar_detach_dual_cv_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    b_s = df["val_balanced_accuracy_audio_head"].mean()
    b_f = df["val_balanced_accuracy_fusion"].mean()
    b_sh = df["val_balanced_accuracy_fusion_shrink"].mean()
    summary = (
        f"单人 category_heldout - {subject_name}\n"
        f"fusion S head BACC: {b_s:.4f}\n"
        f"fusion D=S+gB BACC: {b_f:.4f}\n"
        f"fusion shrink D=S+λB BACC: {b_sh:.4f}  λ*={df['lambda_star'].mean():.3f}\n"
    )
    with open(
        os.path.join(out_dir, f"{subject_name}_scalar_detach_dual_cv_summary.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary)
    print(summary)
    return {
        "subject": subject_name,
        "bacc_s": float(b_s),
        "bacc_fusion": float(b_f),
        "bacc_fusion_shrink": float(b_sh),
        "lambda_star_mean": float(df["lambda_star"].mean()),
        "gate_mean": float(df["val_gate_mean"].mean()),
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
    print("PRIO_W:", PRIO_W)
    print("SHRINK_GAMMA:", SHRINK_GAMMA)
    print("RUN_TAG:", RUN_TAG)
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
        OUTPUT_DIR, "individual", f"category_heldout_anat_scalar_detach_dual_{RUN_TAG}"
    )
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        try:
            rows.append(run_subject(subj, am_fm, pair_name_to_idx, type_map, out_dir, device))
        except Exception as e:
            print(f"\n[FAIL] {subj}: {e}")
            traceback.print_exc()

    if rows:
        sdf = pd.DataFrame(rows)
        summary_name = f"n{len(rows)}_scalar_detach_dual_summary.csv"
        sdf.to_csv(os.path.join(out_dir, summary_name), index=False, encoding="utf-8-sig")
        print(f"\n=== {len(rows)} 人汇总 ({summary_name}) ===")
        print(sdf.to_string(index=False))
        print(
            f"group S={sdf.bacc_s.mean():.4f} | "
            f"fusion={sdf.bacc_fusion.mean():.4f} | "
            f"shrink={sdf.bacc_fusion_shrink.mean():.4f} | "
            f"λ*={sdf.lambda_star_mean.mean():.3f}"
        )


if __name__ == "__main__":
    main()
