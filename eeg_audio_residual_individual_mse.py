#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单人 EEG+音频 残差模型消融实验：

- baseline: 只用手工音频特征 (left_globals + right_globals)
- +mse: 在 baseline 音频特征后追加一维 wav2vec 惊喜度 (mse_sum_diff) z-score

每个被试单独划分数据集，划分方式使用 build_consistent_folds，
两个模型在完全相同的划分上训练和测试，便于对比。
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from eeg_audio_residual import (
    SUBJECTS_CONFIG,
    load_all_mappings,
    build_benchmark_paths,
    load_hanglei_trials,
    extract_audio_features as extract_audio_features_all_pairs,
    EEGNetFeatureExtractor,
)
from eeg_audio_fusion_individual_pair_consistent import (
    load_single_subject_eeg,
    filter_consistent_trials,
)
from linear_regression_selected_features import (
    load_wav2vec_feature_zscore,
    build_consistent_folds,
)


class ResidualFusionClassifier(nn.Module):
    """
    单人版残差融合分类模型：
    - audio_head(audio_features) → logits_audio
    - eeg_extractor(eeg_data) → eeg_feat
    - residual_head([eeg_feat, audio_features]) → delta_eeg
    - logits = logits_audio + delta_eeg
    """

    def __init__(self, eeg_channels: int, eeg_samples: int, audio_dim: int,
                 hidden_dim: int = 64, n_classes: int = 2):
        super().__init__()

        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        total_dim = 160 + audio_dim
        self.residual_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, eeg_data: torch.Tensor, audio_features: torch.Tensor,
                return_parts: bool = False):
        """
        eeg_data: (B, 1, n_channels, n_timepoints)
        audio_features: (B, audio_dim)
        """
        eeg_feat = self.eeg_extractor(eeg_data)  # (B,160)

        logits_audio = self.audio_head(audio_features)  # (B,n_classes)

        combined = torch.cat([eeg_feat, audio_features], dim=1)
        delta_eeg = self.residual_head(combined)        # (B,n_classes)

        logits_final = logits_audio + delta_eeg

        if return_parts:
            return logits_final, logits_audio, delta_eeg
        return logits_final


def prepare_single_subject_data(
    subject_name: str,
    filter_consistent: bool = True,
    project_root=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为单个被试构建：
    - X_eeg: (N, C, T)
    - y: (N,)
    - X_audio_base: (N, D) 手工音频特征
    - X_audio_mse:  (N, D+1) 手工音频特征 + wav2vec mse_sum_diff z-score
    """
    if project_root is None:
        project_root = os.getcwd()

    if subject_name not in SUBJECTS_CONFIG:
        raise ValueError(f"未知被试: {subject_name}")

    config = SUBJECTS_CONFIG[subject_name]

    # 1. 读取单人 EEG
    eeg_data, labels, meta = load_single_subject_eeg(config['processed_data_path'])

    # 2. 加载音频映射和 benchmark，解析 all_trials / pair_names
    mappings, folder_ranges = load_all_mappings()
    benchmark_files = build_benchmark_paths(config)
    all_trials = load_hanglei_trials(benchmark_files, mappings, folder_ranges)

    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    pair_names = sorted(list(all_original_names))

    # 3. 筛选一致 trials
    if filter_consistent:
        consistent_trials = filter_consistent_trials(all_trials, pair_names)
        if len(consistent_trials) == 0:
            raise RuntimeError(f"被试 {subject_name} 没有一致 trials")
    else:
        raise NotImplementedError("当前仅支持 filter_consistent=True")

    # 4. 对齐 EEG 与一致 trials（基于 pair_name / original_name 的映射）
    all_trials_index_map: dict[str, List[int]] = {}
    for j, tr in enumerate(all_trials):
        key = tr.get('pair_name') or tr.get('original_name')
        if key:
            all_trials_index_map.setdefault(key, []).append(j)

    consistent_eeg_data = []
    consistent_labels = []
    consistent_pair_names = []

    for trial in consistent_trials:
        pair_name = trial['pair_name']
        idxs = all_trials_index_map.get(pair_name, [])
        if not idxs:
            continue
        j = idxs[0]
        if j < len(eeg_data):
            consistent_eeg_data.append(eeg_data[j])
            # choice: 1/2 → 0/1
            consistent_labels.append(trial['choice'] - 1)
            consistent_pair_names.append(pair_name)

    if len(consistent_eeg_data) == 0:
        raise RuntimeError(f"被试 {subject_name} 一致 trials 无法与 EEG 对齐")

    consistent_eeg_data = np.asarray(consistent_eeg_data, dtype=np.float32)  # (N,C,T)
    consistent_labels = np.asarray(consistent_labels, dtype=np.int64)        # (N,)

    # 5. 提所有 pair 的音频特征（手工特征，基于所有 pair_names）
    print(f"\n=== 提取所有 {len(pair_names)} 个音频对的手工特征用于单人 {subject_name} ===")
    all_audio_trials = [{'pair_name': pn} for pn in pair_names]
    left_audio_features, right_audio_features, feature_mapping = extract_audio_features_all_pairs(
        all_audio_trials, project_root
    )
    all_audio_features = np.concatenate([left_audio_features, right_audio_features], axis=1)  # (N_pairs, D)

    # 6. 加 wav2vec mse_sum_diff 惊喜度 z-score
    print("\n=== 尝试加载 wav2vec mse_sum_diff 惊喜度特征 ===")
    wav_info = load_wav2vec_feature_zscore(
        project_root, pair_names, feature_key='mse_sum_diff'
    )
    if wav_info is not None and wav_info.get('zscore') is not None:
        wav_z = np.asarray(wav_info['zscore'], dtype=float)
        if wav_z.shape[0] != len(pair_names):
            print(f"警告: wav2vec 特征长度 {wav_z.shape[0]} 与 pair 数量 {len(pair_names)} 不一致，将对齐前者长度")
            min_len = min(len(pair_names), wav_z.shape[0])
            wav_z = wav_z[:min_len]
            all_audio_features = all_audio_features[:min_len]
            pair_names = pair_names[:min_len]
        invalid_mask = ~np.isfinite(wav_z)
        if invalid_mask.any():
            print(f"警告: wav2vec mse_sum_diff 对 {invalid_mask.sum()} 个 pair 缺失/非有限值，将用 0 填充")
            wav_z[invalid_mask] = 0.0
        wav_z = wav_z.reshape(-1, 1)
        all_audio_features_with_mse = np.concatenate([all_audio_features, wav_z], axis=1)
    else:
        print("警告: 未能加载有效的 wav2vec mse_sum_diff 特征，退化为只用手工特征")
        all_audio_features_with_mse = all_audio_features

    # 7. 为一致 trials 选择对应的音频特征行
    pair_name_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    audio_indices = []
    for pn in consistent_pair_names:
        idx = pair_name_to_idx.get(pn)
        if idx is not None:
            audio_indices.append(idx)
    audio_indices = np.asarray(audio_indices, dtype=int)

    X_eeg = consistent_eeg_data[: len(audio_indices)]              # (N,C,T)
    y = consistent_labels[: len(audio_indices)]                    # (N,)
    X_audio_base = all_audio_features[audio_indices]               # (N,D)
    X_audio_mse = all_audio_features_with_mse[audio_indices]       # (N,D+1)

    return X_eeg, y, X_audio_base, X_audio_mse


def run_single_subject(
    subject_name: str,
    n_folds: int = 5,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    filter_consistent: bool = True,
    device=None,
):
    """
    在单个被试上，对比：
    - baseline: ResidualFusionClassifier + X_audio_base
    - +mse:     ResidualFusionClassifier + X_audio_mse
    使用 build_consistent_folds 保持折划分一致。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 60}")
    print(f"单人残差消融实验 - 被试 {subject_name}")
    print(f"{'=' * 60}")

    X_eeg, y, X_audio_base, X_audio_mse = prepare_single_subject_data(
        subject_name, filter_consistent=filter_consistent
    )
    n_trials, n_channels, n_samples = X_eeg.shape
    audio_dim_base = X_audio_base.shape[1]
    audio_dim_mse = X_audio_mse.shape[1]

    # 使用与线性回归脚本一致的折划分
    folds, shuffled_indices = build_consistent_folds(n_trials, k=n_folds)

    all_results = []

    for fold_idx, val_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(shuffled_indices, val_idx)

        print(f"\n--- Fold {fold_idx}/{n_folds} ---")
        print(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")

        # 构造张量
        X_eeg_train = torch.FloatTensor(X_eeg[train_idx]).unsqueeze(1).to(device)
        X_eeg_val = torch.FloatTensor(X_eeg[val_idx]).unsqueeze(1).to(device)
        y_train = torch.LongTensor(y[train_idx]).to(device)
        y_val = torch.LongTensor(y[val_idx]).to(device)

        Xa_base_train = torch.FloatTensor(X_audio_base[train_idx]).to(device)
        Xa_base_val = torch.FloatTensor(X_audio_base[val_idx]).to(device)
        Xa_mse_train = torch.FloatTensor(X_audio_mse[train_idx]).to(device)
        Xa_mse_val = torch.FloatTensor(X_audio_mse[val_idx]).to(device)

        # 两个模型：baseline / +mse
        models = {
            'baseline': ResidualFusionClassifier(
                eeg_channels=n_channels,
                eeg_samples=n_samples,
                audio_dim=audio_dim_base,
            ).to(device),
            '+mse': ResidualFusionClassifier(
                eeg_channels=n_channels,
                eeg_samples=n_samples,
                audio_dim=audio_dim_mse,
            ).to(device),
        }

        optimizers = {
            name: optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            for name, model in models.items()
        }
        criterion = nn.CrossEntropyLoss()
        lambda_reg = 0.05

        batch_size = 64
        n_train = len(train_idx)
        n_batches = (n_train + batch_size - 1) // batch_size

        for epoch in range(n_epochs):
            for name, model in models.items():
                model.train()
                epoch_loss = epoch_ce = epoch_reg = 0.0
                epoch_correct = epoch_total = 0

                for b in range(n_batches):
                    s = b * batch_size
                    e = min(s + batch_size, n_train)
                    Xe_batch = X_eeg_train[s:e]
                    if name == 'baseline':
                        Xa_batch = Xa_base_train[s:e]
                    else:
                        Xa_batch = Xa_mse_train[s:e]
                    yb = y_train[s:e]

                    optimizers[name].zero_grad()
                    logits, _, delta = model(Xe_batch, Xa_batch, return_parts=True)
                    ce_loss = criterion(logits, yb)
                    dl2 = torch.norm(delta, p=2, dim=1).mean()
                    loss = ce_loss + lambda_reg * dl2

                    loss.backward()
                    optimizers[name].step()

                    epoch_loss += loss.item()
                    epoch_ce += ce_loss.item()
                    epoch_reg += dl2.item()
                    with torch.no_grad():
                        _, pred = torch.max(logits, 1)
                    epoch_correct += (pred == yb).sum().item()
                    epoch_total += yb.size(0)

                avg_loss = epoch_loss / n_batches
                avg_ce = epoch_ce / n_batches
                avg_reg = epoch_reg / n_batches
                acc_train = epoch_correct / epoch_total if epoch_total else 0.0

            if (epoch + 1) % 10 == 0 or epoch + 1 == n_epochs:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Reg: {avg_reg:.4f}), "
                      f"Train Acc: {acc_train:.4f}")

        # 验证
        for name, model in models.items():
            model.eval()
            Xa_val = Xa_base_val if name == 'baseline' else Xa_mse_val
            with torch.no_grad():
                logits = model(X_eeg_val, Xa_val)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                _, pred = torch.max(logits, 1)
                pred_np = pred.cpu().numpy()

            y_true = y[val_idx]
            acc = accuracy_score(y_true, pred_np)
            bal_acc = balanced_accuracy_score(y_true, pred_np)
            auc = roc_auc_score(y_true, probs[:, 1])

            print(f"  {name:8s} - Val Acc={acc:.3f}, BalAcc={bal_acc:.3f}, AUC={auc:.3f}")
            all_results.append({
                'subject': subject_name,
                'fold': fold_idx,
                'model': name,
                'acc': acc,
                'bal_acc': bal_acc,
                'auc': auc,
            })

    return all_results


def main_all_subjects():
    project_root = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    all_results = []
    for subject_name in SUBJECTS_CONFIG.keys():
        try:
            res = run_single_subject(subject_name, device=device)
            all_results.extend(res)
        except Exception as e:
            print(f"\n被试 {subject_name} 运行出错: {e}")

    # 简单打印汇总
    print("\n=== 所有被试结果汇总 ===")
    for r in all_results:
        print(f"{r['subject']:12s} Fold {r['fold']} {r['model']:8s} "
              f"Acc={r['acc']:.3f} BalAcc={r['bal_acc']:.3f} AUC={r['auc']:.3f}")


if __name__ == "__main__":
    main_all_subjects()

