#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯 EEG-only：multi_scale_dual（多尺度 + signed/power 双读出）。
23 人、受试者内 Stratified 5 折。不接 Audio。

输出：.../individual/eeg_only_multiscale_dual_23_5fold/
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
from sklearn.model_selection import StratifiedKFold

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eeg_audio_residual import (
    SUBJECTS_CONFIG,
    OUTPUT_DIR,
    CV_SEED,
    SAMPLING_RATE,
    align_single_subject_data,
)
from eeg_multiview_branches import build_eegnet_extractor

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

MODE = "multi_scale_dual"
N_FOLDS = 5
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3


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


def train_eeg_only_fold(train_eeg, train_y, val_eeg, val_y, device):
    model = EEGOnlyClassifier(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
        mode=MODE,
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
    }


def run_subject(subject_name, out_dir, device):
    print(f"\n{'=' * 60}\n被试: {subject_name}\n{'=' * 60}")
    eeg, labels, trials, meta, _ = align_single_subject_data(
        subject_name, filter_consistent=False, project_root=project_root
    )
    y = _fix01(np.asarray(labels))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} train={len(train_idx)} val={len(val_idx)} ---")
        m = train_eeg_only_fold(
            eeg[train_idx], y[train_idx], eeg[val_idx], y[val_idx], device
        )
        row = {
            "subject": subject_name,
            "cv_scheme": "stratified_5fold",
            "fold_idx": fold_idx,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            **m,
        }
        fold_rows.append(row)
        print(f"  Fold {fold_idx}: BACC={m['val_balanced_accuracy']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(fold_rows)
    df.to_csv(
        os.path.join(out_dir, f"{subject_name}_eeg_only_dual_5fold.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    b = df["val_balanced_accuracy"].mean()
    print(f"单人 mean BACC: {b:.4f}")
    return {"subject": subject_name, "bacc_eeg_only_dual": float(b)}


def main():
    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("subjects:", SUBJECTS)
    print("n_subjects:", len(SUBJECTS))
    print("mode:", MODE)
    print(f"n_folds={N_FOLDS} (StratifiedKFold)")

    out_dir = os.path.join(OUTPUT_DIR, "individual", "eeg_only_multiscale_dual_23_5fold")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        try:
            rows.append(run_subject(subj, out_dir, device))
        except Exception as e:
            print(f"\n[FAIL] {subj}: {e}")
            traceback.print_exc()

    if rows:
        sdf = pd.DataFrame(rows)
        summary_name = f"n{len(rows)}_eeg_only_dual_5fold_summary.csv"
        sdf.to_csv(os.path.join(out_dir, summary_name), index=False, encoding="utf-8-sig")
        print(f"\n=== {len(rows)} 人汇总 ({summary_name}) ===")
        print(sdf.to_string(index=False))
        print(f"group EEG-only dual BACC={sdf.bacc_eeg_only_dual.mean():.4f}")


if __name__ == "__main__":
    main()
