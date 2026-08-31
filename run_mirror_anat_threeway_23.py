#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mirror（pair-consistent 5 折）三种模型，默认 23 人：
  1) audio  — AnatomicalAcousticPriority
  2) fusion — ResidualScalarGateDetachAnatomical + multi_scale_dual EEG
  3) eeg    — EEG-only multi_scale_dual

输出：.../individual/mirror_anat_threeway_v1/
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
    build_pair_consistent_folds,
)
from am_fm_salience import (
    build_am_fm_cache,
    align_am_fm_to_pair_names,
    batch_am_fm_from_cache,
    content_priority_margin_loss,
    AM_FM_CACHE_TAG,
    DEFAULT_CONTENT_PRIO_LOSS_WEIGHT,
)
from anatomical_acoustic_priority import (
    AnatomicalAcousticPriority,
    ResidualScalarGateDetachAnatomical,
)
from eeg_multiview_branches import build_eegnet_extractor
from linear_regression_loudness_models import load_stereo_2s

SUBJECTS_23 = [
    "yanxingzhuo",
    "jinxiaoyue",
    "chenxianwei",
    "yeziyuan",
    "zhangzhiyao",
    "haoxiang",
    "hehaohuai",
    "qiuhaiyun",
    "zhouyu",
    "honghaokai",
    "caolulu",
    "yanyinsong",
    "huangxiaohang",
    "xufan",
    "qiusiqi",
    "machenxiang",
    "lizhuhang",
    "zhanghanglei",
    "jichengzhi",
    "liuzehao",
    "zengdexin",
    "zhangyajie",
    "zhangyufei",
]

_env_subjects = os.environ.get("SUBJECTS", "").strip()
SUBJECTS = (
    [s.strip() for s in _env_subjects.split(",") if s.strip()]
    if _env_subjects
    else list(SUBJECTS_23)
)

N_FOLDS = 5
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
LAMBDA_FUSION = 1.0
PRIO_W = DEFAULT_CONTENT_PRIO_LOSS_WEIGHT
EEG_MODE = "multi_scale_dual"

_max_folds = os.environ.get("MAX_FOLDS", "").strip()
MAX_FOLDS = int(_max_folds) if _max_folds.isdigit() else None


def _fix01(y):
    y = y.astype(np.int64)
    if set(np.unique(y).tolist()) <= {1, 2}:
        y = y - 1
    return y


class EEGOnlyClassifier(nn.Module):
    def __init__(self, eeg_channels, eeg_samples, mode=EEG_MODE, sfreq=SAMPLING_RATE):
        super().__init__()
        self.extractor = build_eegnet_extractor(
            mode, eeg_channels, eeg_samples, sfreq=sfreq
        )
        feat_dim = getattr(self.extractor, "out_dim", 160)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, eeg):
        return self.head(self.extractor(eeg))


def train_audio_fold(fold_data, am_fm, device):
    train_y = _fix01(fold_data["train_labels"])
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

    model = AnatomicalAcousticPriority().to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    _, _, mel_l_v, mel_r_v, s_l_v, s_r_v = batch_am_fm_from_cache(am_fm, va_pi, device)

    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            logits, _, _, p_l, p_r, *_ = model(
                mel_l, mel_r, s_l, s_r, return_parts=True
            )
            loss = crit(logits, yb) + PRIO_W * content_priority_margin_loss(p_l, p_r, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(mel_l_v, mel_r_v, s_l_v, s_r_v).argmax(1).cpu().numpy()
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
        pred = logits.argmax(1).cpu().numpy()
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "val_balanced_accuracy_audio": float(balanced_accuracy_score(val_y, pred)),
        "val_accuracy_audio": float(accuracy_score(val_y, pred)),
        "val_auc_audio": float(roc_auc_score(val_y, prob))
        if len(np.unique(val_y)) > 1
        else float("nan"),
    }


def train_fusion_fold(fold_data, am_fm, device):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

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
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            D, S, B, g, p_l, p_r = model(
                Xe, mel_l, mel_r, s_l, s_r, train_detach=True, return_parts=True
            )
            loss = (
                crit(S, yb)
                + PRIO_W * content_priority_margin_loss(p_l, p_r, yb)
                + LAMBDA_FUSION * crit(D, yb)
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val, mel_l_v, mel_r_v, s_l_v, s_r_v, train_detach=False).argmax(1).cpu().numpy()
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
        D, S, B, g, *_ = model(
            X_val, mel_l_v, mel_r_v, s_l_v, s_r_v, train_detach=False, return_parts=True
        )
        pred = D.argmax(1).cpu().numpy()
        prob = F.softmax(D, dim=1)[:, 1].cpu().numpy()
    return {
        "val_balanced_accuracy_fusion": float(balanced_accuracy_score(val_y, pred)),
        "val_accuracy_fusion": float(accuracy_score(val_y, pred)),
        "val_auc_fusion": float(roc_auc_score(val_y, prob))
        if len(np.unique(val_y)) > 1
        else float("nan"),
        "val_gate_mean": float(g.mean().cpu()),
    }


def train_eeg_fold(fold_data, device):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])

    model = EEGOnlyClassifier(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)

    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            loss = crit(model(Xe), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val).argmax(1).cpu().numpy()
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
        logits = model(X_val)
        pred = logits.argmax(1).cpu().numpy()
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "val_balanced_accuracy_eeg": float(balanced_accuracy_score(val_y, pred)),
        "val_accuracy_eeg": float(accuracy_score(val_y, pred)),
        "val_auc_eeg": float(roc_auc_score(val_y, prob))
        if len(np.unique(val_y)) > 1
        else float("nan"),
    }


def run_subject(subject_name, am_fm, pair_name_to_idx, out_dir, device):
    print(f"\n{'=' * 60}\n被试: {subject_name}\n{'=' * 60}")
    consistent_eeg, consistent_labels, consistent_trials, meta, _ = align_single_subject_data(
        subject_name, filter_consistent=False, project_root=project_root
    )
    n_pairs = len(pair_name_to_idx)
    dummy_audio = np.zeros((n_pairs, 2), dtype=np.float32)
    consistent_eeg, consistent_labels, consistent_trials, pair_indices = _select_audio_for_trials(
        consistent_eeg, consistent_labels, consistent_trials, dummy_audio, pair_name_to_idx
    )
    folds, shuffled_indices = build_pair_consistent_folds(
        consistent_trials, n_folds=N_FOLDS, seed=CV_SEED
    )

    fold_rows = []
    for fold_idx, val_idx in enumerate(folds, 1):
        if MAX_FOLDS is not None and fold_idx > MAX_FOLDS:
            break
        if len(val_idx) == 0:
            print(f"  Fold {fold_idx}: empty, skip")
            continue
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=False)
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} train={len(train_idx)} val={len(val_idx)} ---")
        fold_data = _build_single_subject_fold_data(
            consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
        )
        m_a = train_audio_fold(fold_data, am_fm, device)
        m_f = train_fusion_fold(fold_data, am_fm, device)
        m_e = train_eeg_fold(fold_data, device)
        row = {
            "subject": subject_name,
            "cv_scheme": "mirror",
            "fold_idx": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            **m_a,
            **m_f,
            **m_e,
        }
        fold_rows.append(row)
        print(
            f"  Fold {fold_idx}: audio={m_a['val_balanced_accuracy_audio']:.4f} | "
            f"fusion={m_f['val_balanced_accuracy_fusion']:.4f} | "
            f"eeg={m_e['val_balanced_accuracy_eeg']:.4f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(fold_rows)
    df.to_csv(
        os.path.join(out_dir, f"{subject_name}_mirror_threeway_cv_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    b_a = df["val_balanced_accuracy_audio"].mean()
    b_f = df["val_balanced_accuracy_fusion"].mean()
    b_e = df["val_balanced_accuracy_eeg"].mean()
    summary = (
        f"mirror threeway - {subject_name}\n"
        f"audio BACC: {b_a:.4f}\n"
        f"fusion BACC: {b_f:.4f}\n"
        f"eeg BACC: {b_e:.4f}\n"
    )
    with open(
        os.path.join(out_dir, f"{subject_name}_mirror_threeway_cv_summary.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary)
    print(summary)
    return {
        "subject": subject_name,
        "bacc_audio": float(b_a),
        "bacc_fusion": float(b_f),
        "bacc_eeg": float(b_e),
    }


def main():
    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("CV=mirror (pair-consistent), n_folds=", N_FOLDS)
    print("subjects:", SUBJECTS)
    print("n_subjects:", len(SUBJECTS))
    print("models: audio | fusion(detach+multi_scale_dual) | eeg(multi_scale_dual)")
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

    out_dir = os.path.join(OUTPUT_DIR, "individual", "mirror_anat_threeway_v1")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        try:
            rows.append(run_subject(subj, am_fm, pair_name_to_idx, out_dir, device))
        except Exception as e:
            print(f"\n[FAIL] {subj}: {e}")
            traceback.print_exc()

    if rows:
        sdf = pd.DataFrame(rows)
        summary_name = f"n{len(rows)}_mirror_threeway_summary.csv"
        sdf.to_csv(os.path.join(out_dir, summary_name), index=False, encoding="utf-8-sig")
        print(f"\n=== {len(rows)} 人汇总 ({summary_name}) ===")
        print(sdf.to_string(index=False))
        print(
            f"group audio={sdf.bacc_audio.mean():.4f} | "
            f"fusion={sdf.bacc_fusion.mean():.4f} | "
            f"eeg={sdf.bacc_eeg.mean():.4f}"
        )


if __name__ == "__main__":
    main()
