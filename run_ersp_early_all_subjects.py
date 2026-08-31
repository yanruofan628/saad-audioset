#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全员跑 residual_gated + EEG ERSP 早期融合（mirror / pair-consistent 5-fold）。

依赖：
- eeg_audio_residual.py（已恢复 individual CV 管线）
- am_fm_salience.py
- eeg_multiview_branches.py（EEGNetEarlyERSPExtractor）
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

from eeg_audio_residual import (
    SUBJECTS_CONFIG,
    OUTPUT_DIR,
    N_FOLDS,
    CV_SEED,
    SAMPLING_RATE,
    load_all_mappings,
    load_pair_stereo_paths,
    align_single_subject_data,
    build_pair_consistent_folds,
    _select_audio_for_trials,
    _build_single_subject_fold_data,
)
from am_fm_salience import (
    EarSalienceEncoder,
    DEFAULT_N_MELS,
    build_am_fm_cache,
    align_am_fm_to_pair_names,
    AM_FM_CACHE_TAG,
)
from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM
from linear_regression_loudness_models import load_stereo_2s

MODEL_NAME = 'residual_gated_eeg_ersp_early'
LAMBDA_REG = 0.05
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3


class ResidualGatedAMFMEarlyERSP(nn.Module):
    """AM+FM 声学先验 + SGCM 门控 EEG 残差；EEG 使用 early_ersp 提取器。"""

    def __init__(self, eeg_channels, eeg_samples, n_mels=DEFAULT_N_MELS,
                 embed_dim=32, hidden_dim=64, n_classes=2, gate_hidden=64,
                 sfreq=SAMPLING_RATE):
        super().__init__()
        self.audio_emb_dim = embed_dim * 2
        self.ear_encoder = EarSalienceEncoder(n_mels=n_mels, embed_dim=embed_dim)
        self.eeg_extractor = build_eegnet_extractor(
            'early_ersp', eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, 'out_dim', EEGNET_OUT_DIM)

        self.acoustic_head = nn.Sequential(
            nn.Linear(self.audio_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        total_dim = eeg_dim + self.audio_emb_dim
        self.fusion_stem = nn.Sequential(
            nn.Linear(total_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden * 2),
        )
        self.channel_gate = nn.Linear(gate_hidden, gate_hidden)
        self.residual_head = nn.Sequential(
            nn.Linear(gate_hidden + self.audio_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, eeg_data, am_left, am_right, mel_left, mel_right, return_parts=False):
        z_l = self.ear_encoder(am_left, mel_left)
        z_r = self.ear_encoder(am_right, mel_right)
        audio_emb = torch.cat([z_l, z_r], dim=1)
        logits_audio = self.acoustic_head(audio_emb)

        eeg_feat = self.eeg_extractor(eeg_data)
        combined = torch.cat([eeg_feat, audio_emb], dim=1)
        q, k = self.fusion_stem(combined).chunk(2, dim=1)
        gated = torch.sigmoid(self.channel_gate(q)) * k
        delta = self.residual_head(torch.cat([gated, audio_emb], dim=1))
        logits = logits_audio + delta
        if return_parts:
            return logits, logits_audio, delta
        return logits


def _list_runnable_subjects():
    ready = []
    for name, cfg in SUBJECTS_CONFIG.items():
        p = cfg['processed_data_path']
        if not os.path.isdir(p):
            continue
        if any(f.endswith('_trials.npy') for f in os.listdir(p)):
            ready.append(name)
    # unique preserve order
    return list(dict.fromkeys(ready))


def _am_fm_batch(am_fm, pair_indices, device):
    idx = pair_indices
    return (
        torch.FloatTensor(am_fm['am_left'][idx]).to(device),
        torch.FloatTensor(am_fm['am_right'][idx]).to(device),
        torch.FloatTensor(am_fm['mel_left'][idx]).to(device),
        torch.FloatTensor(am_fm['mel_right'][idx]).to(device),
    )


def train_one_fold(fold_data, am_fm, device):
    train_eeg = fold_data['train_eeg']
    train_y = fold_data['train_labels'].astype(np.int64)
    val_eeg = fold_data['val_eeg']
    val_y = fold_data['val_labels'].astype(np.int64)
    tr_pi = fold_data['train_pair_indices']
    va_pi = fold_data['val_pair_indices']

    # labels 1/2 -> 0/1 if needed
    if set(np.unique(train_y).tolist()) <= {1, 2}:
        train_y = train_y - 1
        val_y = val_y - 1

    model = ResidualGatedAMFMEarlyERSP(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)
    am_l_v, am_r_v, mel_l_v, mel_r_v = _am_fm_batch(am_fm, va_pi, device)
    y_val = torch.LongTensor(val_y).to(device)

    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, wait = 0.0, None, 0
    last_epoch = 0

    for epoch in range(1, N_EPOCHS + 1):
        last_epoch = epoch
        model.train()
        perm = np.random.permutation(n_train)
        correct = total = 0
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            am_l, am_r, mel_l, mel_r = _am_fm_batch(am_fm, tr_pi[idx], device)
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            logits, _, delta = model(Xe, am_l, am_r, mel_l, mel_r, return_parts=True)
            ce = crit(logits, yb)
            reg = (delta ** 2).mean()
            loss = ce + LAMBDA_REG * reg
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += len(idx)

        model.eval()
        with torch.no_grad():
            logits = model(X_val, am_l_v, am_r_v, mel_l_v, mel_r_v)
            val_pred = logits.argmax(1).cpu().numpy()
            val_prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        acc = accuracy_score(val_y, val_pred)
        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            print(
                f"    Epoch {epoch}/{N_EPOCHS} Train Acc={correct/max(total,1):.4f} "
                f"Val Acc={acc:.4f}"
            )
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
        logits = model(X_val, am_l_v, am_r_v, mel_l_v, mel_r_v)
        val_pred = logits.argmax(1).cpu().numpy()
        val_prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    metrics = {
        f'val_accuracy_{MODEL_NAME}': float(accuracy_score(val_y, val_pred)),
        f'val_balanced_accuracy_{MODEL_NAME}': float(balanced_accuracy_score(val_y, val_pred)),
        f'val_auc_{MODEL_NAME}': float(roc_auc_score(val_y, val_prob)) if len(np.unique(val_y)) > 1 else float('nan'),
        'epochs': last_epoch,
    }
    return metrics


def run_subject(subject_name, am_fm, pair_name_to_idx, output_dir, device):
    consistent_eeg, consistent_labels, consistent_trials, meta, _ = align_single_subject_data(
        subject_name, filter_consistent=False, project_root=project_root
    )
    # pair indices via dummy audio features row count = n_pairs
    n_pairs = len(pair_name_to_idx)
    dummy_audio = np.zeros((n_pairs, 2), dtype=np.float32)
    consistent_eeg, consistent_labels, consistent_trials, pair_indices = _select_audio_for_trials(
        consistent_eeg, consistent_labels, consistent_trials,
        dummy_audio, pair_name_to_idx,
    )

    folds, shuffled_indices = build_pair_consistent_folds(
        consistent_trials, n_folds=N_FOLDS, seed=CV_SEED
    )
    print(f"\n交叉验证: {N_FOLDS} 折 ({subject_name}), 模型: {MODEL_NAME}")

    fold_results = []
    for fold_idx, val_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=True)
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} (train {len(train_idx)}, val {len(val_idx)}) ---")
        fold_data = _build_single_subject_fold_data(
            consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
        )
        metrics = train_one_fold(fold_data, am_fm, device)
        row = {
            'subject': subject_name,
            'fold_idx': fold_idx,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            **metrics,
        }
        fold_results.append(row)
        print(
            f"  Fold {fold_idx}: ACC={metrics[f'val_accuracy_{MODEL_NAME}']:.4f}, "
            f"BACC={metrics[f'val_balanced_accuracy_{MODEL_NAME}']:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{subject_name}_residual_cv_results.csv')
    pd.DataFrame(fold_results).to_csv(csv_path, index=False, encoding='utf-8-sig')

    accs = [r[f'val_accuracy_{MODEL_NAME}'] for r in fold_results]
    baccs = [r[f'val_balanced_accuracy_{MODEL_NAME}'] for r in fold_results]
    aucs = [r[f'val_auc_{MODEL_NAME}'] for r in fold_results]
    summary = (
        f"单人 EEG 音频残差分类 - {subject_name}\n"
        f"CV 方案: pair_consistent / mirror\n"
        f"{'=' * 60}\n"
        f"模型 {MODEL_NAME}:\n"
        f"  平均验证准确率: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
        f"  平均平衡准确率: {np.mean(baccs):.4f} ± {np.std(baccs):.4f}\n"
        f"  平均AUC: {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}\n"
    )
    with open(os.path.join(output_dir, f'{subject_name}_residual_cv_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print(summary)
    return {
        'subject': subject_name,
        'cv_scheme': 'mirror',
        f'mean_acc_{MODEL_NAME}': float(np.mean(accs)),
        f'mean_balanced_acc_{MODEL_NAME}': float(np.mean(baccs)),
    }


def main():
    subjects = _list_runnable_subjects()
    print('=' * 72)
    print(f'全员 ERSP 早期融合: {MODEL_NAME}')
    print(f'被试 ({len(subjects)}): {subjects}')
    print('=' * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    mappings, _ = load_all_mappings()
    pair_names = sorted({v for m in mappings for v in m.values()})
    stereo_path_map = load_pair_stereo_paths()
    pair_name_to_idx = {pn: i for i, pn in enumerate(pair_names)}

    cache_dir = os.path.join(project_root, 'audio_features_cache')
    am_fm_raw = build_am_fm_cache(
        pair_names, stereo_path_map, load_stereo_2s, cache_dir, cache_tag=AM_FM_CACHE_TAG
    )
    am_fm = align_am_fm_to_pair_names(am_fm_raw, pair_names)
    print('AM/FM shapes:', {k: v.shape for k, v in am_fm.items() if hasattr(v, 'shape')})

    out_root = os.path.join(OUTPUT_DIR, 'individual', 'gated_eeg_ersp_early_all', MODEL_NAME, 'mirror')
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    for subj in subjects:
        done = os.path.join(out_root, f'{subj}_residual_cv_summary.txt')
        if os.path.isfile(done):
            print(f'\n跳过已完成: {subj}')
            # reload mean from csv if possible
            csv_path = os.path.join(out_root, f'{subj}_residual_cv_results.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                summary_rows.append({
                    'subject': subj,
                    'cv_scheme': 'mirror',
                    f'mean_acc_{MODEL_NAME}': float(df[f'val_accuracy_{MODEL_NAME}'].mean()),
                    f'mean_balanced_acc_{MODEL_NAME}': float(df[f'val_balanced_accuracy_{MODEL_NAME}'].mean()),
                })
            continue
        try:
            row = run_subject(subj, am_fm, pair_name_to_idx, out_root, device)
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_root, 'all_subjects_summary.csv'),
                index=False, encoding='utf-8-sig'
            )
        except Exception as e:
            print(f'\n被试 {subj} 失败: {e}')
            traceback.print_exc()

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(os.path.join(out_root, 'all_subjects_summary.csv'), index=False, encoding='utf-8-sig')
        bacc_col = f'mean_balanced_acc_{MODEL_NAME}'
        print('\n' + '=' * 72)
        print(f'全员汇总 {MODEL_NAME} BACC: {sdf[bacc_col].mean():.4f} ± {sdf[bacc_col].std():.4f} (n={len(sdf)})')
        print(f'结果目录: {out_root}')


if __name__ == '__main__':
    main()
