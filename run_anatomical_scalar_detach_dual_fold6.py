#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
category_heldout：解剖锚定 + 标量 gate 残差（detach 训练）+ multi_scale_dual EEG。
D = S + g·B；loss_audio 用 S，loss_fusion 用 D_train = S.detach() + g·B。

只跑 6 人 category-heldout 第 1 折。
输出：.../individual/category_heldout_6_scalar_detach_dual_fold1/
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
from linear_regression_loudness_models import parse_pair_name, load_stereo_2s

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

N_FOLDS = 6
FOLD_ONLY = 1
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
LAMBDA_FUSION = 1.0
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
    idx_groups = np.array_split(np.arange(len(types_shuffled)), n_folds)
    fold_type_groups = [[types_shuffled[int(i)] for i in idx] for idx in idx_groups]
    type_to_fold = {}
    for fold_idx, group in enumerate(fold_type_groups):
        for pairing_type in group:
            type_to_fold[pairing_type] = fold_idx
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
    return folds, shuffled_indices


def _fix01(y):
    y = y.astype(np.int64)
    if set(np.unique(y).tolist()) <= {1, 2}:
        y = y - 1
    return y


def train_fold(fold_data, am_fm, device):
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
        eeg_mode="multi_scale_dual",
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
            loss_audio = crit(S, yb) + PRIO_W * content_priority_margin_loss(p_l, p_r, yb)
            loss_fusion = crit(D, yb)
            loss = loss_audio + LAMBDA_FUSION * loss_fusion
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
        D, S, B, g, *_ = model(
            X_val, mel_l_v, mel_r_v, s_l_v, s_r_v, train_detach=False, return_parts=True
        )
        pred_d = D.argmax(1).cpu().numpy()
        pred_s = S.argmax(1).cpu().numpy()
        prob_d = F.softmax(D, dim=1)[:, 1].cpu().numpy()
        g_mean = float(g.mean().cpu())

    return {
        "val_balanced_accuracy_audio": float(balanced_accuracy_score(val_y, pred_s)),
        "val_balanced_accuracy_fusion": float(balanced_accuracy_score(val_y, pred_d)),
        "val_accuracy_fusion": float(accuracy_score(val_y, pred_d)),
        "val_auc_fusion": float(roc_auc_score(val_y, prob_d))
        if len(np.unique(val_y)) > 1
        else float("nan"),
        "val_gate_mean": g_mean,
    }


def run_subject(subject_name, am_fm, pair_name_to_idx, type_map, out_dir, device):
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

    fold_data = _build_single_subject_fold_data(
        consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
    )
    m = train_fold(fold_data, am_fm, device)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"subject": subject_name, **m}]).to_csv(
        os.path.join(out_dir, f"{subject_name}_scalar_detach_dual_fold{fold_idx}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return {"subject": subject_name, **m}


def main():
    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mappings, _ = load_all_mappings()
    pair_names = sorted({v for m in mappings for v in m.values()})
    stereo_path_map = load_pair_stereo_paths()
    pair_name_to_idx = {pn: i for i, pn in enumerate(pair_names)}
    type_map, _ = build_global_pairing_type_fold_map(pair_names)

    cache_dir = os.path.join(project_root, "audio_features_cache")
    am_fm_raw = build_am_fm_cache(
        pair_names, stereo_path_map, load_stereo_2s, cache_dir, cache_tag=AM_FM_CACHE_TAG
    )
    am_fm = align_am_fm_to_pair_names(am_fm_raw, pair_names)

    out_dir = os.path.join(
        OUTPUT_DIR,
        "individual",
        "category_heldout_6_scalar_detach_dual_fold1",
    )
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        try:
            rows.append(run_subject(subj, am_fm, pair_name_to_idx, type_map, out_dir, device))
        except Exception as e:
            print(f"[FAIL] {subj}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    summary_name = f"n{len(rows)}_scalar_detach_dual_summary_fold1.csv"
    df.to_csv(os.path.join(out_dir, summary_name), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

