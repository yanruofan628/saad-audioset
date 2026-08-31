#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 5s EEG 任务对应的自然声音二选一数据上按被试逐个比较：
1) 只用手工音频特征的 audio-only 分类器
2) 在 audio-only 基础上追加一个 wav2vec MSE 惊喜度特征 (mse_sum_diff) 的分类器

数据来源：
- 复用 `eeg_audio_residual.py` 中的 5s EEG 任务数据加载逻辑：
  - 使用 `load_all_subjects_data()` 读取所有被试的 trial 列表 (包含 pair_name) 与标签 (0/1)
  - 使用与 EEG 实验相同映射文件和 `extract_audio_features` 提取得到的 pair-level 音频特征缓存

惊喜度特征来源：
- 复用 `wav2vec_surprise_regression.py` / `wav2vec_surprise_heatmap.py` 预先计算好的
  `wav2vec_regression_results/wav2vec_features.json`，只计算一次，之后本脚本直接加载。

分类标签构造：
- 对于某个被试，取该被试在 EEG 任务中的所有 trials：
  - 每个 trial 的标签为 0/1（选左/选右），直接来自 EEG 任务的行为响应
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用 wav2vec 特征加载工具
from linear_regression_selected_features import (  # noqa: E402
    load_wav2vec_feature_zscore,
    build_consistent_folds,
)

# 复用 EEG / 单人融合 的数据与模型（只读，不改原脚本）
import eeg_audio_residual as ear  # noqa: E402
import eeg_audio_fusion_individual as ind  # noqa: E402
import wav2vec_surprise_regression as wz  # noqa: E402


def load_pair_level_audio_features(project_root: str):
    """
    复用 `eeg_audio_residual.py` 中的音频特征缓存逻辑（简化版）：
    - 使用映射文件得到所有 pair_names
    - 如果存在 audio_features_cache 中的缓存，则直接加载
    - 如果不存在，则调用 ear.extract_audio_features 提取得到 pair-level 特征

    返回:
        pair_names: List[str]
        all_audio_features: np.ndarray, shape (n_pairs, audio_dim)
    """
    # 1) 从映射文件获取所有可能的 pair_names
    mappings, folder_ranges = ear.load_all_mappings()
    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    pair_names = sorted(list(all_original_names))

    print(f"从映射文件提取的音频对总数: {len(pair_names)}")

    # 2) 尝试从缓存加载 pair-level audio 特征
    pair_names_str = ",".join(sorted(pair_names))
    import hashlib

    pair_names_hash = hashlib.md5(pair_names_str.encode()).hexdigest()[:8]

    cache_dir = os.path.join(project_root, "audio_features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    features_cache_path = os.path.join(cache_dir, f"all_audio_features_{pair_names_hash}.npy")
    meta_cache_path = os.path.join(cache_dir, f"all_audio_features_{pair_names_hash}_meta.json")

    all_audio_features = None
    if os.path.exists(features_cache_path) and os.path.exists(meta_cache_path):
        try:
            with open(meta_cache_path, "r", encoding="utf-8") as f:
                cache_meta = json.load(f)
            cached_pair_names = cache_meta.get("pair_names", [])
            if cached_pair_names == pair_names:
                all_audio_features = np.load(features_cache_path)
                print(f"成功从缓存加载所有音频特征: {all_audio_features.shape}")
            else:
                print("缓存中的 pair_names 不匹配，重新提取 audio 特征...")
        except Exception as exc:
            print(f"加载 audio 特征缓存失败，将重新提取: {exc}")

    # 3) 若缓存不可用，则重新提取
    if all_audio_features is None:
        print("未发现有效音频特征缓存，开始提取所有 pair 的特征 (可能较慢)...")
        # 构造虚拟 trials，只包含 pair_name
        dummy_trials = [{"pair_name": name} for name in pair_names]
        left_globals, right_globals, feature_mapping = ear.extract_audio_features(
            dummy_trials, project_root
        )
        all_audio_features = np.concatenate([left_globals, right_globals], axis=1)

        # 保存缓存
        cache_meta = {
            "pair_names": pair_names,
            "n_pairs": len(pair_names),
            "feature_shape": all_audio_features.shape,
        }
        np.save(features_cache_path, all_audio_features)
        with open(meta_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_meta, f, indent=2, ensure_ascii=False)
        print(f"音频特征已缓存到: {features_cache_path}")

    return pair_names, all_audio_features


def ensure_wav2vec_features_for_pairs(project_root: str, pair_names: list):
    """
    确保针对给定的 pair_names 已经计算并保存了 wav2vec 特征：
    - 如果 `wav2vec_regression_results/wav2vec_features.json` 不存在或缺少这些 key，
      则调用 wz.compute_wav2vec_results_from_pair_names 针对缺失的 pair 计算惊喜度，
      并用 wz.extract_wav2vec_features 生成/更新 wav2vec_features.json。
    """
    regression_output_dir = os.path.join(project_root, "wav2vec_regression_results")
    os.makedirs(regression_output_dir, exist_ok=True)
    features_file = os.path.join(regression_output_dir, "wav2vec_features.json")

    existing_features = {}
    if os.path.exists(features_file):
        try:
            with open(features_file, "r", encoding="utf-8") as f:
                existing_features = json.load(f)
        except Exception as exc:
            print(f"警告: 读取已有 wav2vec 特征文件失败，将从头开始: {exc}")
            existing_features = {}

    existing_keys = set(existing_features.keys())
    # 直接用原始 pair_name 作为 key（wz.compute_wav2vec_results_from_pair_names 也会这样存）
    missing_pairs = [name for name in pair_names if name not in existing_keys]

    if not missing_pairs:
        print("所有 pair 已在 wav2vec_features.json 中有特征，跳过重新计算。")
        return

    print(f"\n=== 有 {len(missing_pairs)} 个新的 pair 需要计算 wav2vec 惊喜度 ===")
    print(f"  示例缺失: {missing_pairs[:5]}")
    print("  将调用 wav2vec_surprise_regression.compute_wav2vec_results_from_pair_names 进行一次性计算...")

    # 使用 EEG 实验所用的 AUDIO_BASE_DIR
    audio_base_dir = ear.AUDIO_BASE_DIR
    if not os.path.exists(audio_base_dir):
        raise RuntimeError(f"音频基础目录不存在: {audio_base_dir}，无法计算 wav2vec 惊喜度。")

    # 计算缺失 pair 的 wav2vec 原始结果
    wav2vec_results = wz.compute_wav2vec_results_from_pair_names(
        missing_pairs, audio_base_dir, regression_output_dir
    )
    if not wav2vec_results:
        raise RuntimeError("计算 wav2vec 惊喜度失败，未获得任何结果。")

    # 提取特征（包括 mse_sum_diff 等）
    new_features = wz.extract_wav2vec_features(wav2vec_results)
    if not new_features:
        raise RuntimeError("从 wav2vec 结果中未能提取任何特征。")

    # 合并到已有特征中
    for k, v in new_features.items():
        existing_features[k] = v

    # 保存合并后的 wav2vec_features.json
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                k: {
                    k2: float(v2)
                    if isinstance(v2, (np.integer, np.floating))
                    else v2
                    for k2, v2 in v.items()
                }
                for k, v in existing_features.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"已更新 wav2vec 特征文件: {features_file}")


def build_subject_dataset_with_surprise(
    project_root: str,
    subject_name: str,
    pair_names: list,
    all_audio_features: np.ndarray,
    feature_key: str = "mse_sum_diff",
):
    """
    构建某个被试在 EEG 任务中的 trial-level 数据集：
    - X_audio:        该被试所有 trial 的 audio 特征
    - X_audio_surp:   在 X_audio 基础上追加一列 wav2vec 惊喜度特征 (mse_sum_diff)
    - y:              每个 trial 的 0/1 选择标签
    """
    # 1) 加载所有被试的 EEG 任务数据（试次信息 + 行为标签）
    all_eeg_data, all_labels, all_subject_ids, all_trials, n_common_channels, meta_common = (
        ear.load_all_subjects_data()
    )

    subject_names = list(ear.SUBJECTS_CONFIG.keys())
    if subject_name not in subject_names:
        raise ValueError(f"subject_name={subject_name} 不在 SUBJECTS_CONFIG 中: {subject_names}")

    subject_id = subject_names.index(subject_name)
    # 选出该被试的 trial 索引
    mask = all_subject_ids == subject_id
    if mask.sum() == 0:
        raise RuntimeError(f"被试 {subject_name} 没有任何 trial。")

    # 对齐 trials / labels / pair_name / EEG
    labels_subj = all_labels[mask]
    trial_indices = np.where(mask)[0]
    trials_subj = [all_trials[i] for i in trial_indices]
    eeg_subj = all_eeg_data[mask]

    # 2) 构建 pair_name -> 索引 映射，用于从 pair-level audio 特征中取一行
    pair_name_to_idx = {name: idx for idx, name in enumerate(pair_names)}

    X_audio_list = []
    y_list = []
    pair_used = []

    for lbl, trial in zip(labels_subj, trials_subj):
        pair_name = trial.get("pair_name") or trial.get("original_name")
        if not pair_name:
            continue
        if pair_name not in pair_name_to_idx:
            # 该 pair 在 audio 特征中不存在，跳过
            continue
        pair_idx = pair_name_to_idx[pair_name]
        X_audio_list.append(all_audio_features[pair_idx])
        y_list.append(int(lbl))  # 假定为 0/1
        pair_used.append(pair_name)

    if not X_audio_list:
        raise RuntimeError(f"被试 {subject_name} 没有任何 trial 成功对齐到 audio 特征。")

    X_audio = np.vstack(X_audio_list)
    y = np.asarray(y_list, dtype=int)
    eeg_channels = int(eeg_subj.shape[1])
    eeg_samples = int(eeg_subj.shape[2])

    print(f"\n被试 {subject_name}: 有效 trial 数量: {len(y)}")
    print(f"  标签分布: 0 -> {(y == 0).sum()}, 1 -> {(y == 1).sum()}")

    # 3) 加载 wav2vec 惊喜度特征 (mse_sum_diff)，按 pair_names 顺序得到一维特征
    wav_info = load_wav2vec_feature_zscore(project_root, pair_names, feature_key=feature_key)
    if wav_info is None or wav_info.get("zscore") is None:
        raise RuntimeError(
            f"未能加载 Wav2Vec 惊喜度特征 {feature_key}，"
            f"请先运行 wav2vec_surprise_regression.py 生成 wav2vec_features.json。"
        )
    wav_z_all = np.asarray(wav_info["zscore"], dtype=float)
    if wav_z_all.shape[0] != len(pair_names):
        raise RuntimeError(
            f"wav2vec 特征长度 {wav_z_all.shape[0]} 与 pair 数量 {len(pair_names)} 不一致，请检查。"
        )

    # 为该被试的每个 trial 取对应 pair 的惊喜度特征
    wav_subj = []
    for name in pair_used:
        idx = pair_name_to_idx[name]
        val = wav_z_all[idx]
        if not np.isfinite(val):
            val = 0.0  # 非有限值用0填充（z-score的均值）
        wav_subj.append(val)
    wav_subj = np.asarray(wav_subj, dtype=float).reshape(-1, 1)

    # 4) 构造 audio+surprise 特征
    X_audio_surprise = np.concatenate([X_audio, wav_subj], axis=1)

    return X_audio, X_audio_surprise, y, eeg_channels, eeg_samples


def run_subject_cv_audio_only(
    X_audio: np.ndarray,
    X_audio_surprise: np.ndarray,
    y: np.ndarray,
    eeg_channels: int,
    eeg_samples: int,
    n_folds: int = 5,
    seed: int = 42,
):
    """
    使用与 `eeg_audio_fusion_individual.py` 一致的 build_consistent_folds，
    在单个被试的数据上做 5 折交叉验证，对比：
    - 只用手工音频特征的 AudioOnlyClassifier
    - 在手工音频特征基础上追加一列 wav2vec 惊喜度特征 (mse_sum_diff) 的 AudioOnlyClassifier
    """
    assert X_audio.shape[0] == X_audio_surprise.shape[0] == y.shape[0]

    folds, shuffled_indices = build_consistent_folds(len(y), seed=seed, k=n_folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    def _train_one_fold(X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray):
        audio_dim = X_tr.shape[1]
        model = ind.AudioOnlyClassifier(eeg_channels, eeg_samples, audio_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        X_te_t = torch.FloatTensor(X_te).to(device)
        y_tr_t = torch.LongTensor(y_tr).to(device)
        y_te_t = torch.LongTensor(y_te).to(device)

        n_epochs = 100
        patience = 15
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(None, X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(None, X_te_t)
                    _, val_pred = torch.max(val_logits, 1)
                    val_acc = accuracy_score(
                        y_te_t.cpu().numpy(), val_pred.cpu().numpy()
                    )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"    早停触发 (patience={patience})，停止该 fold 训练")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_logits = model(None, X_te_t)
            val_probs = F.softmax(val_logits, dim=1).cpu().numpy()
            _, val_pred = torch.max(val_logits, 1)
            val_pred = val_pred.cpu().numpy()

        acc = accuracy_score(y_te_t.cpu().numpy(), val_pred)
        bal_acc = balanced_accuracy_score(y_te_t.cpu().numpy(), val_pred)
        try:
            auc = roc_auc_score(y_te_t.cpu().numpy(), val_probs[:, 1])
        except ValueError:
            auc = np.nan
        return acc, bal_acc, auc

    metrics_audio = {"acc": [], "bal_acc": [], "auc": []}
    metrics_surprise = {"acc": [], "bal_acc": [], "auc": []}

    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")
        train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)

        X_tr_a = X_audio[train_idx]
        X_te_a = X_audio[test_idx]
        X_tr_s = X_audio_surprise[train_idx]
        X_te_s = X_audio_surprise[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        acc_b, bal_b, auc_b = _train_one_fold(X_tr_a, X_te_a, y_tr, y_te)
        acc_s, bal_s, auc_s = _train_one_fold(X_tr_s, X_te_s, y_tr, y_te)

        metrics_audio["acc"].append(acc_b)
        metrics_audio["bal_acc"].append(bal_b)
        metrics_audio["auc"].append(auc_b)

        metrics_surprise["acc"].append(acc_s)
        metrics_surprise["bal_acc"].append(bal_s)
        metrics_surprise["auc"].append(auc_s)

        print(
            f"  Audio-only: acc={acc_b:.3f}, bal_acc={bal_b:.3f}, auc={auc_b:.3f}"
            f" | Audio+MSE: acc={acc_s:.3f}, bal_acc={bal_s:.3f}, auc={auc_s:.3f}"
        )

    return metrics_audio, metrics_surprise




def run_cv_with_ear_pipeline(
    pair_names: list,
    base_audio_features: np.ndarray,
    surprise_audio_features: np.ndarray,
):
    """
    使用 `eeg_audio_residual.py` 中完全相同的多被试 pair-wise 交叉验证与
    AudioOnlyClassifier 训练流水线，对比：
    - base_audio_features: 原始音频特征
    - surprise_audio_features: 原始音频特征 + 一列 wav2vec mse_sum_diff
    """
    subject_names = list(ear.SUBJECTS_CONFIG.keys())

    # 1) 加载所有被试的 trial 与标签（与 ear.main 开头一致）
    (
        all_eeg_data,
        all_labels,
        all_subject_ids,
        all_trials,
        n_common_channels,
        meta_common,
    ) = ear.load_all_subjects_data()

    # 2) 为每个被试构建 subject_consistent_trials / subject_eeg_data / subject_labels
    subject_consistent_trials = {}
    subject_eeg_data = {}
    subject_labels = {}

    for subject_name in subject_names:
        subject_id = subject_names.index(subject_name)
        print(f"\n=== 处理被试: {subject_name} ===")

        subject_mask = all_subject_ids == subject_id
        subject_eeg = all_eeg_data[subject_mask]
        subject_label_array = all_labels[subject_mask]
        subject_trial_list = [all_trials[i] for i in range(len(all_trials)) if subject_mask[i]]

        print(f"  原始数据: {len(subject_eeg)} 个trials")

        # 这里与 ear.main 中 FILTER_CONSISTENT=False 的分支保持一致：
        consistent_trials = []
        for trial in subject_trial_list:
            original_name = trial.get("original_name")
            if not original_name:
                continue
            consistent_trials.append(
                {
                    "original_name": original_name,
                    "response": trial.get("response"),
                    "reaction_time": trial.get("reaction_time"),
                    "choice": trial.get("response"),
                    "pair_name": original_name,
                }
            )
        print(f"  所有trials: {len(consistent_trials)}")

        if len(consistent_trials) == 0:
            print(f"  警告: 被试 {subject_name} 没有有效的trials，跳过")
            continue

        # 对齐EEG数据和consistent_trials，仅保留 choice/response 为 1 或 2 的 trial
        subject_consistent_eeg = []
        subject_consistent_labels = []
        filtered_trials = []

        trial_index_map = {}
        for j, trial in enumerate(subject_trial_list):
            pair_name_key = trial.get("pair_name") or trial.get("original_name")
            if pair_name_key:
                trial_index_map[pair_name_key] = j

        for trial in consistent_trials:
            pair_name = trial["pair_name"]
            c = trial.get("choice") or trial.get("response")
            if c not in (1, 2):
                continue
            if pair_name not in trial_index_map:
                continue
            idx = trial_index_map[pair_name]
            if idx >= len(subject_eeg):
                continue
            subject_consistent_eeg.append(subject_eeg[idx])
            subject_consistent_labels.append(int(c) - 1)  # 0 或 1
            filtered_trials.append(trial)

        subject_consistent_eeg = np.array(subject_consistent_eeg)
        subject_consistent_labels = np.array(subject_consistent_labels)

        if len(subject_consistent_eeg) == 0:
            print(f"  警告: 被试 {subject_name} 过滤后无有效 trials（choice 均为 1/2），跳过")
            continue

        print(f"  对齐后数据: {len(subject_consistent_eeg)} 个trials")
        print(
            f"  标签分布: 选择A: {np.sum(subject_consistent_labels == 0)}, "
            f"选择B: {np.sum(subject_consistent_labels == 1)}"
        )

        subject_consistent_trials[subject_name] = filtered_trials
        subject_eeg_data[subject_name] = subject_consistent_eeg
        subject_labels[subject_name] = subject_consistent_labels

    # 3) 多被试 fold 划分，与 ear.build_multi_subject_pair_consistent_folds 一致
    print(f"\n=== 多被试fold划分 ===")
    folds, shuffled_indices, subject_trial_info = ear.build_multi_subject_pair_consistent_folds(
        subject_consistent_trials, n_folds=ear.N_FOLDS, seed=ear.CV_SEED
    )

    print(f"交叉验证设置: {ear.N_FOLDS}折交叉验证")

    # 4) 创建 pair_name 到索引的映射
    pair_name_to_idx = {name: idx for idx, name in enumerate(pair_names)}

    # 5) 关闭同步性特征
    all_sync_features = None

    # 6) 在完全相同的 folds 与训练流程下，分别用 base_audio_features 和 surprise_audio_features 训练
    all_fold_results_base = []
    all_fold_results_surp = []

    for fold_idx, val_global_indices in enumerate(folds, 1):
        print(f"\n=== Fold {fold_idx}/{ear.N_FOLDS} ===")

        train_global_indices = []
        for i, fold in enumerate(folds):
            if i != fold_idx - 1:
                train_global_indices.extend(fold.tolist())
        train_global_indices = np.array(train_global_indices, dtype=np.int32)
        val_global_indices = np.array(val_global_indices, dtype=np.int32)

        print(f"  训练集全局索引数: {len(train_global_indices)}")
        print(f"  验证集全局索引数: {len(val_global_indices)}")

        # 收集 fold 的数据
        fold_data = ear.collect_fold_data_from_global_indices(
            subject_names,
            subject_consistent_trials,
            subject_eeg_data,
            subject_labels,
            train_global_indices,
            val_global_indices,
            subject_trial_info,
            pair_name_to_idx,
        )

        # 用 base_audio_features 训练
        metrics_base = ear.train_fold_model(
            fold_data,
            base_audio_features,
            pair_name_to_idx,
            pair_names,
            all_sync_features=all_sync_features,
        )
        # 用 surprise_audio_features 训练（多一列 MSE）
        metrics_surp = ear.train_fold_model(
            fold_data,
            surprise_audio_features,
            pair_name_to_idx,
            pair_names,
            all_sync_features=all_sync_features,
        )

        all_fold_results_base.append(
            {
                "fold_idx": fold_idx,
                "val_accuracy_audio_only": metrics_base["val_accuracy_audio_only"],
                "val_balanced_accuracy_audio_only": metrics_base[
                    "val_balanced_accuracy_audio_only"
                ],
                "val_auc_audio_only": metrics_base["val_auc_audio_only"],
                "per_subject_audio_only": metrics_base["val_accuracy_per_subject_audio_only"],
            }
        )
        all_fold_results_surp.append(
            {
                "fold_idx": fold_idx,
                "val_accuracy_audio_only": metrics_surp["val_accuracy_audio_only"],
                "val_balanced_accuracy_audio_only": metrics_surp[
                    "val_balanced_accuracy_audio_only"
                ],
                "val_auc_audio_only": metrics_surp["val_auc_audio_only"],
                "per_subject_audio_only": metrics_surp["val_accuracy_per_subject_audio_only"],
            }
        )

    return all_fold_results_base, all_fold_results_surp


def summarize_and_save_results(
    metrics_audio,
    metrics_surprise,
    output_dir: str,
    feature_key: str,
):
    """
    汇总交叉验证结果并保存为 CSV 和简单文本。
    """
    os.makedirs(output_dir, exist_ok=True)

    def _summary(metrics_dict):
        return {
            "acc_mean": float(np.nanmean(metrics_dict["acc"])),
            "acc_std": float(np.nanstd(metrics_dict["acc"])),
            "bal_acc_mean": float(np.nanmean(metrics_dict["bal_acc"])),
            "bal_acc_std": float(np.nanstd(metrics_dict["bal_acc"])),
            "auc_mean": float(np.nanmean(metrics_dict["auc"])),
            "auc_std": float(np.nanstd(metrics_dict["auc"])),
        }

    summary_audio = _summary(metrics_audio)
    summary_surprise = _summary(metrics_surprise)

    df_rows = [
        {"model": "audio_only", **summary_audio},
        {"model": f"audio_plus_{feature_key}", **summary_surprise},
    ]
    df = pd.DataFrame(df_rows)

    csv_path = os.path.join(
        output_dir, f"audio_surprise_compare_{feature_key}.csv"
    )
    txt_path = os.path.join(
        output_dir, f"audio_surprise_compare_{feature_key}.txt"
    )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Audio-only vs Audio+surprise (wav2vec) comparison\n")
        f.write(f"Feature key: {feature_key}\n\n")
        for row in df_rows:
            f.write(f"Model: {row['model']}\n")
            f.write(
                f"  Acc: {row['acc_mean']:.4f} ± {row['acc_std']:.4f}\n"
                f"  BalAcc: {row['bal_acc_mean']:.4f} ± {row['bal_acc_std']:.4f}\n"
                f"  AUC: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}\n\n"
            )

    print(f"\n结果已保存到:\n  {csv_path}\n  {txt_path}")


def main():
    feature_key = "mse_sum_diff"
    subject_name = "yanxingzhuo"  # 可以按需修改为其它被试

    print(f"使用 wav2vec 惊喜度特征: {feature_key}")
    print(f"按单人 5 折方式评估被试: {subject_name}")

    # 先加载一次 pair-level audio 特征 (所有被试共用)
    pair_names, base_audio_features = load_pair_level_audio_features(PROJECT_ROOT)

    # 确保对这些 pair 已经计算并保存了 wav2vec 特征（首次运行会比较慢）
    ensure_wav2vec_features_for_pairs(PROJECT_ROOT, pair_names)

    # 构建该被试的 trial-level audioOnly / audio+MSE 数据集
    (
        X_audio,
        X_audio_surprise,
        y,
        eeg_channels,
        eeg_samples,
    ) = build_subject_dataset_with_surprise(
        PROJECT_ROOT, subject_name, pair_names, base_audio_features, feature_key=feature_key
    )

    print(f"\n被试 {subject_name} 最终数据形状:")
    print(f"  X_audio: {X_audio.shape}")
    print(f"  X_audio_surprise: {X_audio_surprise.shape}")
    print(f"  y: {y.shape}, 正类比例: {(y == 1).mean():.3f}")

    # 使用 individual 脚本中的折划分方式，做单人 AudioOnly 对比
    metrics_audio, metrics_surprise = run_subject_cv_audio_only(
        X_audio,
        X_audio_surprise,
        y,
        eeg_channels,
        eeg_samples,
        n_folds=5,
        seed=42,
    )

    output_dir = os.path.join(PROJECT_ROOT, f"audio_surprise_compare_results_{subject_name}")
    summarize_and_save_results(
        metrics_audio,
        metrics_surprise,
        output_dir=output_dir,
        feature_key=feature_key,
    )


if __name__ == "__main__":
    main()

