#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比（additive，不改旧模型）：
  residual_scalar_gate       — Priority + scalar g + 标准 EEGNet
  residual_scalar_gate_ltar  — 同上，但 EEG 用时序 attention 读出 B

环境变量：
  SUBJECTS, TRAIN_BATCH_SIZE, CV_SCHEME(默认 mirror)
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

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
    DEFAULT_N_MELS,
    DEFAULT_CONTENT_PRIO_LOSS_WEIGHT,
    build_am_fm_cache,
    align_am_fm_to_pair_names,
    AM_FM_CACHE_TAG,
    content_priority_margin_loss,
    batch_am_fm_from_cache,
)
from linear_regression_loudness_models import load_stereo_2s
from prime_scalar_gate import MODEL_BUILDERS

# 只跑新模块；基线 residual_scalar_gate 不再重复训
MODELS = [
    'residual_scalar_gate_ltar',
]

DEFAULT_SUBJECTS = [
    'yanxingzhuo',
    'jinxiaoyue',
    'chenxianwei',
    'qiuhaiyun',
    'haoxiang',
    'honghaokai',
]

LAMBDA_REG = 0.05
PRIO_W = DEFAULT_CONTENT_PRIO_LOSS_WEIGHT
N_EPOCHS = 100
PATIENCE = 15
LR = 1e-3


def train_one_model(name, fold_data, am_fm, device, batch_size):
    train_eeg = fold_data['train_eeg']
    train_y = fold_data['train_labels'].astype(np.int64)
    val_eeg = fold_data['val_eeg']
    val_y = fold_data['val_labels'].astype(np.int64)
    tr_pi = fold_data['train_pair_indices']
    va_pi = fold_data['val_pair_indices']

    if set(np.unique(train_y).tolist()) <= {1, 2}:
        train_y = train_y - 1
        val_y = val_y - 1

    model = MODEL_BUILDERS[name](
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
        n_mels=DEFAULT_N_MELS,
        sfreq=SAMPLING_RATE,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()

    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)
    am_l_v, am_r_v, mel_l_v, mel_r_v, s_l_v, s_r_v = batch_am_fm_from_cache(
        am_fm, va_pi, device
    )

    n_train = len(train_y)
    n_batches = (n_train + batch_size - 1) // batch_size
    best_acc, best_state, wait = 0.0, None, 0
    last_epoch = 0

    for epoch in range(1, N_EPOCHS + 1):
        last_epoch = epoch
        model.train()
        perm = np.random.permutation(n_train)
        correct = total = 0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            am_l, am_r, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(
                am_fm, tr_pi[idx], device
            )
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            out = model(
                Xe, am_l, am_r, mel_l, mel_r, s_l, s_r, return_parts=True
            )
            logits = out[0]
            delta = out[2]
            p_l, p_r = out[3], out[4]
            ce = crit(logits, yb)
            dl2 = torch.norm(delta, p=2, dim=1).mean()
            prio = content_priority_margin_loss(p_l, p_r, yb)
            loss = ce + LAMBDA_REG * dl2 + PRIO_W * prio
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += len(idx)

        model.eval()
        with torch.no_grad():
            logits = model(X_val, am_l_v, am_r_v, mel_l_v, mel_r_v, s_l_v, s_r_v)
            val_pred = logits.argmax(1).cpu().numpy()
        acc = accuracy_score(val_y, val_pred)
        if epoch == 1 or epoch % 10 == 0 or epoch == N_EPOCHS:
            print(
                f"    [{name}] Epoch {epoch}/{N_EPOCHS} "
                f"Train Acc={correct / max(total, 1):.4f} Val Acc={acc:.4f}",
                flush=True,
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
        logits = model(X_val, am_l_v, am_r_v, mel_l_v, mel_r_v, s_l_v, s_r_v)
        val_pred = logits.argmax(1).cpu().numpy()
        val_prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        f'val_accuracy_{name}': float(accuracy_score(val_y, val_pred)),
        f'val_balanced_accuracy_{name}': float(balanced_accuracy_score(val_y, val_pred)),
        f'val_auc_{name}': float(roc_auc_score(val_y, val_prob)) if len(np.unique(val_y)) > 1 else float('nan'),
        f'epochs_{name}': last_epoch,
    }


def run_subject(subject_name, am_fm, pair_name_to_idx, output_dir, device, batch_size):
    consistent_eeg, consistent_labels, consistent_trials, meta, _ = align_single_subject_data(
        subject_name, filter_consistent=False, project_root=project_root
    )
    n_pairs = len(pair_name_to_idx)
    dummy_audio = np.zeros((n_pairs, 2), dtype=np.float32)
    consistent_eeg, consistent_labels, consistent_trials, pair_indices = _select_audio_for_trials(
        consistent_eeg, consistent_labels, consistent_trials,
        dummy_audio, pair_name_to_idx,
    )
    folds, shuffled_indices = build_pair_consistent_folds(
        consistent_trials, n_folds=N_FOLDS, seed=CV_SEED
    )
    print(f"\n=== {subject_name} | models={MODELS} | batch={batch_size} ===", flush=True)

    fold_results = []
    for fold_idx, val_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=True)
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---", flush=True)
        fold_data = _build_single_subject_fold_data(
            consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
        )
        metrics = {}
        for name in MODELS:
            metrics.update(train_one_model(name, fold_data, am_fm, device, batch_size))
        row = {
            'subject': subject_name,
            'fold_idx': fold_idx,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            **metrics,
        }
        fold_results.append(row)
        msg = ' | '.join(
            f"{n} BACC={metrics[f'val_balanced_accuracy_{n}']:.4f}" for n in MODELS
        )
        print(f"  Fold {fold_idx}: {msg}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(
        os.path.join(output_dir, f'{subject_name}_cv_results.csv'),
        index=False, encoding='utf-8-sig',
    )
    lines = [f'subject={subject_name}', f'batch={batch_size}', '=' * 60]
    summary = {'subject': subject_name, 'batch': batch_size}
    for name in MODELS:
        baccs = [r[f'val_balanced_accuracy_{name}'] for r in fold_results]
        accs = [r[f'val_accuracy_{name}'] for r in fold_results]
        lines.append(
            f'{name}: ACC={np.mean(accs):.4f}±{np.std(accs):.4f} '
            f'BACC={np.mean(baccs):.4f}±{np.std(baccs):.4f}'
        )
        summary[f'mean_bacc_{name}'] = float(np.mean(baccs))
        summary[f'std_bacc_{name}'] = float(np.std(baccs))
        print(f"  {name} BACC: {np.mean(baccs):.4f} ± {np.std(baccs):.4f}", flush=True)
    with open(os.path.join(output_dir, f'{subject_name}_cv_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return summary


def main():
    batch_size = int(os.environ.get('TRAIN_BATCH_SIZE', '16'))
    subj_env = os.environ.get('SUBJECTS', '').strip()
    if subj_env:
        subjects = [s.strip() for s in subj_env.split(',') if s.strip()]
    else:
        subjects = [s for s in DEFAULT_SUBJECTS if s in SUBJECTS_CONFIG]

    tag = f'scalar_gate_ltar_compare_mirror_bs{batch_size}'
    out_root = os.path.join(OUTPUT_DIR, 'individual', tag, 'mirror')
    os.makedirs(out_root, exist_ok=True)

    print('=' * 72)
    print(f'CV=mirror | batch={batch_size} | subjects={subjects}')
    print(f'models={MODELS}')
    print(f'out={out_root}')
    print('=' * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, flush=True)

    mappings, _ = load_all_mappings()
    pair_names = sorted({v for m in mappings for v in m.values()})
    stereo_path_map = load_pair_stereo_paths()
    pair_name_to_idx = {pn: i for i, pn in enumerate(pair_names)}
    cache_dir = os.path.join(project_root, 'audio_features_cache')
    am_fm_raw = build_am_fm_cache(
        pair_names, stereo_path_map, load_stereo_2s, cache_dir, cache_tag=AM_FM_CACHE_TAG
    )
    am_fm = align_am_fm_to_pair_names(am_fm_raw, pair_names)

    rows = []
    for subj in subjects:
        done = os.path.join(out_root, f'{subj}_cv_summary.txt')
        if os.path.isfile(done):
            print(f'跳过已完成: {subj}', flush=True)
            continue
        try:
            rows.append(
                run_subject(subj, am_fm, pair_name_to_idx, out_root, device, batch_size)
            )
            pd.DataFrame(rows).to_csv(
                os.path.join(out_root, 'all_subjects_summary.csv'),
                index=False, encoding='utf-8-sig',
            )
        except Exception:
            traceback.print_exc()
            print(f'失败: {subj}', flush=True)

    if rows:
        print('\n全被试汇总:')
        for name in MODELS:
            vals = [r[f'mean_bacc_{name}'] for r in rows]
            print(f'  {name} BACC: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})')


if __name__ == '__main__':
    sys.exit(main() or 0)
