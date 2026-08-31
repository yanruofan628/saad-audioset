#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEG-only：仅跑 multi_scale_dual（多尺度 + signed/power 双读出）。
6 人、category-heldout 第 1 折；结束后与 default / multi_scale_temporal 汇总对比。
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
    align_single_subject_data,
    _select_audio_for_trials,
    _build_single_subject_fold_data,
)
from eeg_multiview_branches import build_eegnet_extractor
from linear_regression_loudness_models import parse_pair_name

SUBJECTS_DEFAULT_6 = [
    "yanxingzhuo",
    "jinxiaoyue",
    "chenxianwei",
    "yeziyuan",
    "zhangzhiyao",
    "haoxiang",
]

_env_subjects = os.environ.get("SUBJECTS", "").strip()
SUBJECTS = (
    [s.strip() for s in _env_subjects.split(",") if s.strip()]
    if _env_subjects
    else list(SUBJECTS_DEFAULT_6)
)

MODE = "multi_scale_dual"
N_FOLDS = 6
FOLD_ONLY = 1
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3


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


class EEGOnlyClassifier(nn.Module):
    def __init__(self, eeg_channels, eeg_samples, mode: str, sfreq=SAMPLING_RATE):
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


def train_eeg_only_fold(fold_data, mode, device):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])

    model = EEGOnlyClassifier(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
        mode=mode,
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
        "val_balanced_accuracy": float(balanced_accuracy_score(val_y, pred)),
        "val_accuracy": float(accuracy_score(val_y, pred)),
        "val_auc": float(roc_auc_score(val_y, prob))
        if len(np.unique(val_y)) > 1
        else float("nan"),
        "out_dim": int(getattr(model.extractor, "out_dim", -1)),
    }


def run_subject(subject_name, pair_name_to_idx, type_map, out_dir, device):
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

    fold_idx = FOLD_ONLY
    val_idx = folds[fold_idx - 1]
    train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=False)
    print(f"\n--- Fold {fold_idx}/{N_FOLDS} train={len(train_idx)} val={len(val_idx)} ---")
    fold_data = _build_single_subject_fold_data(
        consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
    )
    m = train_eeg_only_fold(fold_data, MODE, device)
    print(f"  {MODE}: BACC={m['val_balanced_accuracy']:.4f} (out_dim={m['out_dim']})")

    row = {
        "subject": subject_name,
        "cv_scheme": "category_heldout",
        "fold_idx": fold_idx,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "bacc_multi_scale_dual": m["val_balanced_accuracy"],
        "acc_multi_scale_dual": m["val_accuracy"],
        "auc_multi_scale_dual": m["val_auc"],
        "out_dim": m["out_dim"],
    }
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        os.path.join(out_dir, f"{subject_name}_eeg_only_dual_fold1.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return row


def main():
    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("subjects:", SUBJECTS)
    print("mode:", MODE)

    mappings, _ = load_all_mappings()
    pair_names = sorted({v for m in mappings for v in m.values()})
    pair_name_to_idx = {pn: i for i, pn in enumerate(pair_names)}
    type_map, _ = build_global_pairing_type_fold_map(pair_names)

    out_dir = os.path.join(OUTPUT_DIR, "individual", "eeg_only_multiscale_dual_fold1")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        try:
            rows.append(run_subject(subj, pair_name_to_idx, type_map, out_dir, device))
        except Exception as e:
            print(f"\n[FAIL] {subj}: {e}")
            traceback.print_exc()

    if not rows:
        return

    dual = pd.DataFrame(rows)
    dual.to_csv(
        os.path.join(out_dir, f"n{len(rows)}_eeg_only_dual_fold1_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    prev_path = os.path.join(
        OUTPUT_DIR, "individual", "eeg_only_multiscale_fold1", "n6_eeg_only_ms_fold1_summary.csv"
    )
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        cmp = prev[
            [
                "subject",
                "val_balanced_accuracy_default",
                "val_balanced_accuracy_multi_scale_temporal",
            ]
        ].merge(
            dual[["subject", "bacc_multi_scale_dual"]],
            on="subject",
            how="inner",
        )
        cmp["d_dual_vs_default"] = (
            cmp.bacc_multi_scale_dual - cmp.val_balanced_accuracy_default
        )
        cmp["d_dual_vs_ms"] = (
            cmp.bacc_multi_scale_dual - cmp.val_balanced_accuracy_multi_scale_temporal
        )
        cmp_path = os.path.join(out_dir, f"n{len(cmp)}_eeg_only_threeway_fold1.csv")
        cmp.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        print(f"\n=== 三方对比 fold1 ({os.path.basename(cmp_path)}) ===")
        print(
            cmp.rename(
                columns={
                    "val_balanced_accuracy_default": "default",
                    "val_balanced_accuracy_multi_scale_temporal": "multi_scale",
                    "bacc_multi_scale_dual": "dual",
                }
            ).to_string(index=False)
        )
        print(
            f"group default={cmp.val_balanced_accuracy_default.mean():.4f} | "
            f"multi_scale={cmp.val_balanced_accuracy_multi_scale_temporal.mean():.4f} | "
            f"dual={cmp.bacc_multi_scale_dual.mean():.4f}"
        )
        print(
            f"delta dual-default={cmp.d_dual_vs_default.mean():+.4f} | "
            f"dual-ms={cmp.d_dual_vs_ms.mean():+.4f}"
        )
    else:
        print(f"\n未找到旧结果 {prev_path}，仅输出 dual 汇总")
        print(dual.to_string(index=False))
        print(f"group dual={dual.bacc_multi_scale_dual.mean():.4f}")


if __name__ == "__main__":
    main()
