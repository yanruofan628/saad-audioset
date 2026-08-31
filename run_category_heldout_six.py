#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
category_heldout：仅跑 content-priority 骨架 + 门控 EEG 残差（am_fm_v2）。
声学 only 已有结果，本脚本不再训纯声学。

输出：A:/.../individual/category_heldout_prio_gated_v2/
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
    EarSalienceEncoder,
    DEFAULT_N_MELS,
    FM_BRANCH_DIM,
    build_am_fm_cache,
    align_am_fm_to_pair_names,
    batch_am_fm_from_cache,
    content_priority_margin_loss,
    AM_FM_CACHE_TAG,
    DEFAULT_CONTENT_PRIO_LOSS_WEIGHT,
)
from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM
from linear_regression_loudness_models import load_stereo_2s, parse_pair_name

# 原 17 人（与 v1 category_heldout 名单一致）
SUBJECTS_DEFAULT_17 = [
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
]

_env_subjects = os.environ.get("SUBJECTS", "").strip()
SUBJECTS = (
    [s.strip() for s in _env_subjects.split(",") if s.strip()]
    if _env_subjects
    else list(SUBJECTS_DEFAULT_17)
)

N_FOLDS = 6
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
LAMBDA_REG = 0.05
PRIO_W = DEFAULT_CONTENT_PRIO_LOSS_WEIGHT


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
    # 不要用 np.array_split 直接切 tuple list（会变成不可哈希的 ndarray）
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


class AMFMContentPriorityFM(nn.Module):
    """声学 only：p ← FM；s ← pre-zscore RMS（am_fm_v2）。"""

    def __init__(self, n_mels=DEFAULT_N_MELS, embed_dim=32, n_classes=2):
        super().__init__()
        self.ear_encoder = EarSalienceEncoder(n_mels=n_mels, embed_dim=embed_dim)
        self.priority_head = nn.Sequential(
            nn.Linear(FM_BRANCH_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        feat_dim = embed_dim * 2 + 4
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def _ear(self, am, mel, s):
        z, _, fm_h = self.ear_encoder.forward_parts(am, mel)
        p = self.priority_head(fm_h)
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        return z, p, s

    def forward(self, am_l, am_r, mel_l, mel_r, s_l, s_r, return_parts=False):
        z_l, p_l, s_l = self._ear(am_l, mel_l, s_l)
        z_r, p_r, s_r = self._ear(am_r, mel_r, s_r)
        feat = torch.cat([z_l, z_r, p_l, p_r, p_l - p_r, s_l - s_r], dim=1)
        logits = self.classifier(feat)
        if return_parts:
            return logits, p_l, p_r
        return logits


class ResidualGatedContentPriority(nn.Module):
    """与 AMFMContentPriorityFM 同一声学骨架，再加门控 EEG 残差。"""

    def __init__(self, eeg_channels, eeg_samples, n_mels=DEFAULT_N_MELS,
                 embed_dim=32, hidden_dim=64, n_classes=2, gate_hidden=64,
                 sfreq=SAMPLING_RATE):
        super().__init__()
        self.ear_encoder = EarSalienceEncoder(n_mels=n_mels, embed_dim=embed_dim)
        self.priority_head = nn.Sequential(
            nn.Linear(FM_BRANCH_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        self.feat_dim = embed_dim * 2 + 4
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )
        self.eeg_extractor = build_eegnet_extractor(
            "default", eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, "out_dim", EEGNET_OUT_DIM)
        self.fusion_stem = nn.Sequential(
            nn.Linear(eeg_dim + self.feat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden * 2),
        )
        self.channel_gate = nn.Linear(gate_hidden, gate_hidden)
        self.residual_head = nn.Sequential(
            nn.Linear(gate_hidden + self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def _ear(self, am, mel, s):
        z, _, fm_h = self.ear_encoder.forward_parts(am, mel)
        p = self.priority_head(fm_h)
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        return z, p, s

    def forward(self, eeg_data, am_l, am_r, mel_l, mel_r, s_l, s_r, return_parts=False):
        z_l, p_l, s_l = self._ear(am_l, mel_l, s_l)
        z_r, p_r, s_r = self._ear(am_r, mel_r, s_r)
        feat = torch.cat([z_l, z_r, p_l, p_r, p_l - p_r, s_l - s_r], dim=1)
        logits_audio = self.classifier(feat)
        eeg_feat = self.eeg_extractor(eeg_data)
        q, k = self.fusion_stem(torch.cat([eeg_feat, feat], dim=1)).chunk(2, dim=1)
        gated = torch.sigmoid(self.channel_gate(q)) * k
        delta = self.residual_head(torch.cat([gated, feat], dim=1))
        logits = logits_audio + delta
        if return_parts:
            return logits, logits_audio, delta, p_l, p_r
        return logits


def _fix01(y):
    y = y.astype(np.int64)
    if set(np.unique(y).tolist()) <= {1, 2}:
        y = y - 1
    return y


def train_gated_fold(fold_data, am_fm, device):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

    model = ResidualGatedContentPriority(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)
    am_l_v, am_r_v, mel_l_v, mel_r_v, s_l_v, s_r_v = batch_am_fm_from_cache(am_fm, va_pi, device)

    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            am_l, am_r, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            logits, _, delta, p_l, p_r = model(
                Xe, am_l, am_r, mel_l, mel_r, s_l, s_r, return_parts=True
            )
            loss = (
                crit(logits, yb)
                + PRIO_W * content_priority_margin_loss(p_l, p_r, yb)
                + LAMBDA_REG * (delta ** 2).mean()
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val, am_l_v, am_r_v, mel_l_v, mel_r_v, s_l_v, s_r_v)
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
        logits = model(X_val, am_l_v, am_r_v, mel_l_v, mel_r_v, s_l_v, s_r_v)
        pred = logits.argmax(1).cpu().numpy()
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "val_accuracy_residual_gated": float(accuracy_score(val_y, pred)),
        "val_balanced_accuracy_residual_gated": float(balanced_accuracy_score(val_y, pred)),
        "val_auc_residual_gated": float(roc_auc_score(val_y, prob))
        if len(np.unique(val_y)) > 1
        else float("nan"),
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
        if len(val_idx) == 0:
            print(f"  Fold {fold_idx}: empty val, skip")
            continue
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=False)
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} train={len(train_idx)} val={len(val_idx)} ---")
        fold_data = _build_single_subject_fold_data(
            consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
        )
        m_gated = train_gated_fold(fold_data, am_fm, device)
        row = {
            "subject": subject_name,
            "cv_scheme": "category_heldout",
            "fold_idx": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            **m_gated,
        }
        fold_rows.append(row)
        print(f"  Fold {fold_idx}: prio+gated BACC={m_gated['val_balanced_accuracy_residual_gated']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(fold_rows)
    csv_path = os.path.join(out_dir, f"{subject_name}_residual_cv_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    b_g = df["val_balanced_accuracy_residual_gated"].mean()
    summary = (
        f"单人 category_heldout - {subject_name}\n"
        f"content_priority + residual_gated BACC: {b_g:.4f}\n"
    )
    with open(
        os.path.join(out_dir, f"{subject_name}_residual_cv_summary.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(summary)
    print(summary)
    return {
        "subject": subject_name,
        "bacc_residual_gated": float(b_g),
    }


def main():
    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("subjects:", SUBJECTS)
    print("AM_FM_CACHE_TAG:", AM_FM_CACHE_TAG)

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

    out_dir = os.path.join(OUTPUT_DIR, "individual", "category_heldout_prio_gated_v2")
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
        summary_name = f"n{len(rows)}_prio_gated_summary.csv"
        sdf.to_csv(os.path.join(out_dir, summary_name), index=False, encoding="utf-8-sig")
        print(f"\n=== {len(rows)} 人汇总 ({summary_name}) ===")
        print(sdf.to_string(index=False))
        print(f"group content_priority+gated BACC={sdf.bacc_residual_gated.mean():.4f}")


if __name__ == "__main__":
    main()
