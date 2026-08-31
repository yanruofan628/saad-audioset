#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为每个被试单独训练模型进行对比分析。
当前仅使用 SIMPLE 模型（EEG + 音频特征）；可通过 MODELS_TO_USE 改为 sync / audio_only 等。

每个被试进行5折交叉验证，最后计算所有被试的平均准确率。
折划分确保同一音频对的 AB、BA 不在同一 fold，避免泄漏。

支持的被试：hanglei, yanxingzhuo, jinxiaoyue, chenxianwei, yeziyuan, zhangzhiyao, haoxiang

使用方法：
1. 确保数据路径正确：
   - EEG数据：A:/standard_data_interp_no_ica/{subject_name}
   - 音频映射文件：D:\\D\\research\\audioset下载\\audio_pairs_2s\\分组音频\\folder{1,2,3}
2. 运行脚本：python eeg_audio_fusion_individual_simple.py

输出：
- comparison_models_{subject_name}_results.csv: 每个被试各模型的5折交叉验证结果
- comparison_models_all_subjects_summary.csv: 所有被试平均准确率汇总
- comparison_models_summary.txt: 文本汇总
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的函数和类
from eegnet_classify_trials_5s import (
    find_data_files, load_meta, reshape_trials, 
    apply_baseline_correction, select_time_segment, select_last_seconds,
    exponential_moving_standardize, DataMeta
)
from linear_regression_selected_features import parse_pair_name
from linear_regression_loudness_models import (
    parse_stereo_pair_name, load_mono_5s, DEFAULT_AUDIO_BASE_DIR,
    parse_experiment_data_txt
)
from feature_stage_attention_model import (
    get_feature_configs_from_linear_regression, SR, HOP, DEFAULT_STAGE_SECONDS
)
from attention_loudness_model import FeatureTimeSeriesDataset
from feature_stage_attention_model import StageFeatureDataset

# 从eeg_audio_fusion_individual导入必要的类和函数（同时使用同步性和简单模型）
from eeg_audio_fusion_individual import (
    SUBJECTS_CONFIG,
    AUDIO_BASE_DIR,
    MAPPING_BASE_DIR,
    MAPPING_FILES,
    SAMPLING_RATE,
    EEGNetFeatureExtractor,
    SyncFusionClassifier,  # 同步性融合模型
    SimpleFusionClassifier,  # 简单融合模型
    load_single_subject_eeg,
    build_benchmark_paths,
    load_all_mappings,
    extract_audio_features,
    compute_sync_features_cache,  # 同步性特征计算
    parse_audio_mapping_custom,
    load_hanglei_trials,
    filter_consistent_trials
)

# 从eeg_audio_fusion_classify_hanglei导入缺失的函数
from eeg_audio_fusion_classify_hanglei import (
    map_audio_features_to_trials
)

class AudioOnlyClassifier(nn.Module):
    """仅使用音频特征的分类模型"""

    def __init__(self, audio_dim, hidden_dim=64, n_classes=2):
        super(AudioOnlyClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, audio_features):
        """
        audio_features: (batch, audio_dim) - 手工音频特征
        返回: (batch, n_classes)
        """
        return self.classifier(audio_features)


# 输出目录
OUTPUT_DIR = 'A:/standard_data_interp_no_ica/individual_comparison_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交叉验证配置
CV_SEED = 42
N_FOLDS = 5

# 使用的模型：仅 simple 时只训练简单融合模型（EEG+音频），不计算同步性特征
MODELS_TO_USE = ['simple']


def _get_pair_key(pair_name):
    """从 pair_name 解析出规范化键 tuple(sorted([audio_a, audio_b]))，无法解析时返回 None。"""
    if not pair_name:
        return None
    res = parse_pair_name(pair_name)
    if res is None or res[0] is None:
        return None
    left_cat, left_id, right_cat, right_id, _ = res
    audio_a = f"{left_cat}_{left_id}"
    audio_b = f"{right_cat}_{right_id}"
    return tuple(sorted([audio_a, audio_b]))


def build_folds_ab_ba_split(trials, n_folds, seed):
    """
    构建 k 折划分，确保同一音频对的 AB、BA 不在同一 fold（避免泄漏）。
    返回 folds (list of k arrays，每折的 test 索引), shuffled_indices (全部索引)。
    """
    n = len(trials)
    rng = np.random.RandomState(seed)
    print("\n=== 构建 AB/BA 分折（同一音频对 AB、BA 不在同一 fold）===")
    pair_key_to_indices = {}
    unpaired = []

    for i, t in enumerate(trials):
        key = _get_pair_key(t.get("pair_name"))
        if key is None:
            unpaired.append(i)
            continue
        if key not in pair_key_to_indices:
            pair_key_to_indices[key] = []
        pair_key_to_indices[key].append(i)

    fold_indices = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=np.int64)

    def assign_one(idx, exclude_fold=None):
        order = np.lexsort((np.arange(n_folds), fold_sizes))
        for fi in order:
            if exclude_fold is not None and fi == exclude_fold:
                continue
            fold_indices[fi].append(idx)
            fold_sizes[fi] += 1
            return
        fold_indices[0].append(idx)
        fold_sizes[0] += 1

    n_paired = sum(1 for inds in pair_key_to_indices.values() if len(inds) == 2)
    n_single = sum(1 for inds in pair_key_to_indices.values() if len(inds) == 1)
    if unpaired:
        print(f"  警告: {len(unpaired)} 个 trial 的 pair_name 无法解析，将随机分折")
    if n_single:
        print(f"  仅单版本（AB 或 BA）的音频对: {n_single} 个")
    print(f"  成对（AB+BA）的音频对: {n_paired} 个，已确保 AB/BA 不在同一 fold")

    pair_keys = list(pair_key_to_indices.keys())
    rng.shuffle(pair_keys)

    for key in pair_keys:
        inds = pair_key_to_indices[key]
        if len(inds) == 2:
            order = np.lexsort((np.arange(n_folds), fold_sizes))
            f0, f1 = int(order[0]), int(order[1])
            fold_indices[f0].append(inds[0])
            fold_indices[f1].append(inds[1])
            fold_sizes[f0] += 1
            fold_sizes[f1] += 1
        else:
            assign_one(inds[0])

    for idx in unpaired:
        assign_one(idx)

    folds = [np.array(fi, dtype=np.int64) for fi in fold_indices]
    shuffled_indices = np.concatenate(folds)
    print(f"  各 fold 大小:")
    for i, f in enumerate(folds):
        print(f"    Fold {i + 1}: {len(f)} 个 trials")
    return folds, shuffled_indices


def load_single_subject_trials(subject_name, subject_config, mappings):
    """
    加载单个被试的trial信息
    
    返回:
        all_trials: trial信息列表，包含pair_name等
    """
    print(f"\n=== 加载被试 {subject_name} 的trial信息 ===")
    
    benchmark_files = build_benchmark_paths(subject_config)
    print(f"Benchmark文件:")
    for bf in benchmark_files:
        print(f"  {bf}")
    
    # 解析trial信息
    parsed_trials_all_sessions = []
    for session_idx, benchmark_file in enumerate(benchmark_files):
        if not os.path.exists(benchmark_file):
            print(f"  警告: benchmark文件不存在: {benchmark_file}")
            continue
        
        parsed_trials = parse_experiment_data_txt(benchmark_file)
        print(f"  Session {session_idx + 1}: 解析了 {len(parsed_trials)} 个trials")
        
        # 处理hanglei缺失的trial
        if subject_name == 'hanglei' and subject_config.get('missing_trials'):
            if session_idx == 0:  # folder1
                parsed_trials = parsed_trials[:-1]
                print(f"    去掉folder1最后一个trial后: {len(parsed_trials)} 个trials")
            elif session_idx == 1:  # folder2
                parsed_trials = parsed_trials[:-1]
                print(f"    去掉folder2最后一个trial后: {len(parsed_trials)} 个trials")
        
        # 获取对应的映射文件
        mapping = mappings[session_idx] if session_idx < len(mappings) else mappings[0]
        
        for trial in parsed_trials:
            wavfile = trial.get('wavfile')
            if not wavfile:
                continue
            
            filename = os.path.basename(wavfile)
            
            # 应用映射
            if filename in mapping:
                original_name = mapping[filename]
            elif filename.replace('.wav', '') in mapping:
                original_name = mapping[filename.replace('.wav', '')]
            else:
                continue
            
            parsed_trials_all_sessions.append({
                'original_name': original_name,
                'response': trial.get('response'),
                'reaction_time': trial.get('reaction_time'),
                'session': session_idx + 1,
                'pair_name': original_name,
                'choice': trial.get('response')  # 直接使用response作为choice
            })
    
    print(f"总共解析了 {len(parsed_trials_all_sessions)} 个trials")
    return parsed_trials_all_sessions


def train_models_for_subject(subject_name, shared_audio_features, unique_pair_names, filter_consistent=False):
    """
    为单个被试训练指定模型（见 MODELS_TO_USE）。当前仅 simple。

    参数:
        subject_name: 被试名称
        shared_audio_features: 共享的音频特征字典，包含left_globals, right_globals, feature_mapping
        filter_consistent: 是否筛选一致的trials（AB和BA版本都选择同一个音频的trials）

    返回:
        result_dict: 包含各模型每折结果和平均准确率的字典
    """
    print(f"\n{'='*60}")
    print(f"处理被试: {subject_name}")
    print(f"{'='*60}")
    
    subject_config = SUBJECTS_CONFIG[subject_name]
    
    # 1. 加载EEG数据
    print(f"\n=== 加载EEG数据 ===")
    eeg_data, labels, meta = load_single_subject_eeg(subject_config['processed_data_path'])
    print(f"EEG数据形状: {eeg_data.shape}")
    print(f"标签数量: {len(labels)}")
    
    # 2. 加载音频映射文件
    mappings, folder_ranges = load_all_mappings()
    
    # 3. 加载trial信息
    all_trials = load_single_subject_trials(subject_name, subject_config, mappings)
    
    # 4. 对齐EEG数据和trials
    n_trials = min(len(eeg_data), len(all_trials))
    if len(eeg_data) != len(all_trials):
        print(f"警告: EEG数据({len(eeg_data)})和trials({len(all_trials)})数量不匹配，使用前{n_trials}个")
    
    eeg_data_aligned = eeg_data[:n_trials]
    labels_aligned = labels[:n_trials]
    trials_aligned = all_trials[:n_trials]

    # 4.5. 筛选一致的trials（AB和BA版本都选择同一个音频的trials）
    if filter_consistent:
        print(f"\n=== 筛选一致的trials ===")
        print("筛选逻辑：AB和BA版本都选择同一个音频的trials")

        # 筛选一致的trials
        consistent_trials = filter_consistent_trials(all_trials, unique_pair_names)

        if len(consistent_trials) == 0:
            print("警告: 没有找到一致的trials，使用所有trials")
        else:
            print(f"筛选结果：保留 {len(consistent_trials)}/{len(all_trials)} 个一致的trials")

            # 找到一致trials在原始数据中的索引
            consistent_indices = []
            for trial in consistent_trials:
                pair_name = trial['pair_name']
                try:
                    idx = next(i for i, t in enumerate(trials_aligned) if t['original_name'] == pair_name)
                    consistent_indices.append(idx)
                except StopIteration:
                    continue

            if len(consistent_indices) > 0:
                # 应用筛选
                eeg_data_aligned = eeg_data_aligned[consistent_indices]
                labels_aligned = labels_aligned[consistent_indices]
                trials_aligned = [trials_aligned[i] for i in consistent_indices]
                print(f"筛选后数据: {len(eeg_data_aligned)} 个trials")
            else:
                print("警告: 未能匹配到一致trials的索引，使用所有trials")

    # 转换标签：choice: 1=选左, 2=选右 -> 转换为0=选左, 1=选右
    y = np.array([trial['choice'] - 1 for trial in trials_aligned])
    
    print(f"对齐后的数据: {len(eeg_data_aligned)} 个trials")
    print(f"标签分布: 选择A: {np.sum(y == 0)}, 选择B: {np.sum(y == 1)}")
    
    # 5. 使用共享的音频特征映射到该被试的trials
    print(f"\n=== 将共享音频特征映射到trials ===")

    # 创建正确的feature_mapping字典
    feature_mapping = {
        'pair_to_idx': {pair_name: idx for idx, pair_name in enumerate(unique_pair_names)}
    }

    left_audio_features, right_audio_features, valid_indices = map_audio_features_to_trials(
        trials_aligned,
        shared_audio_features['left_globals'],
        shared_audio_features['right_globals'],
        feature_mapping
    )
    
    # 7. 对齐数据（只保留成功提取音频特征的trials）
    eeg_data_final = eeg_data_aligned[valid_indices]
    y_final = y[valid_indices]
    trials_final = [trials_aligned[i] for i in valid_indices]

    print(f"最终数据: {len(eeg_data_final)} 个trials")

    # 8. 拼接左右耳音频特征
    X_audio_manual = np.concatenate([left_audio_features, right_audio_features], axis=1)
    print(f"音频特征形状: {X_audio_manual.shape}")

    sync_features_cache = None
    if 'sync' in MODELS_TO_USE:
        # 9. 计算同步性特征（仅在使用 sync 模型时）
        print(f"\n=== 计算同步性特征 ===")
        subject_cache_dir = os.path.join(OUTPUT_DIR, f'sync_cache_{subject_name}')
        os.makedirs(subject_cache_dir, exist_ok=True)
        print(f"为被试 {subject_name} 使用缓存目录: {subject_cache_dir}")
        sync_features_cache, cache_path = compute_sync_features_cache(
            eeg_data_final, trials_final,
            eeg_sampling_rate=SAMPLING_RATE,
            audio_base_dir=AUDIO_BASE_DIR,
            output_dir=subject_cache_dir
        )
        print(f"缓存文件路径: {cache_path}")
        if os.path.exists(cache_path):
            print(f"✅ 缓存文件已存在，将被重用")
        else:
            print(f"⚠️ 缓存文件不存在，将创建新的缓存")
        print(f"同步性特征缓存完成，维度: {sync_features_cache.shape}")
    else:
        print(f"\n=== 未使用 sync 模型，跳过同步性特征 ===")

    # 10. 5折交叉验证
    print(f"\n=== 开始5折交叉验证 ===")
    print("使用模型:", ", ".join(MODELS_TO_USE))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 折划分：确保同一音频对的 AB、BA 不在同一 fold，避免泄漏
    folds, shuffled_indices = build_folds_ab_ba_split(trials_final, n_folds=N_FOLDS, seed=CV_SEED)

    fold_results = []
    eeg_channels = eeg_data_final.shape[1]
    eeg_samples = eeg_data_final.shape[2]
    audio_dim = X_audio_manual.shape[1]

    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")
        
        # 划分数据
        train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)
        
        X_eeg_train = eeg_data_final[train_idx]
        X_audio_train = X_audio_manual[train_idx]
        y_train = y_final[train_idx]
        
        X_eeg_val = eeg_data_final[test_idx]
        X_audio_val = X_audio_manual[test_idx]
        y_val = y_final[test_idx]
        
        print(f"  训练集大小: {len(y_train)}, 验证集大小: {len(y_val)}")

        # 定义模型（仅 MODELS_TO_USE 中的）
        models = {}
        if 'sync' in MODELS_TO_USE and sync_features_cache is not None:
            models['sync'] = SyncFusionClassifier(eeg_channels, eeg_samples, audio_dim, sync_features_cache=sync_features_cache)
        if 'simple' in MODELS_TO_USE:
            models['simple'] = SimpleFusionClassifier(eeg_channels, eeg_samples, audio_dim)
        if 'audio_only' in MODELS_TO_USE:
            models['audio_only'] = AudioOnlyClassifier(audio_dim)

        # 为每种模型初始化最佳状态
        best_models = {}
        best_val_accs = {}
        patience_counters = {}

        for model_name in models.keys():
            models[model_name].to(device)
            best_models[model_name] = None
            best_val_accs[model_name] = 0
            patience_counters[model_name] = 0

        n_epochs = 100
        batch_size = 16  # 根据GPU内存调整
        n_train = len(y_train)

        for epoch in range(n_epochs):
            # 训练所有模型
            for model_name, model in models.items():
                model.train()
                epoch_loss = 0.0
                n_batches = 0

                # Batch训练
                for start_idx in range(0, n_train, batch_size):
                    end_idx = min(start_idx + batch_size, n_train)
                    batch_indices = np.arange(start_idx, end_idx)

                    # 为不同模型准备不同的输入
                    if model_name == 'audio_only':
                        X_audio_batch = torch.FloatTensor(X_audio_train[batch_indices]).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                        optimizer.zero_grad()
                        outputs = model(X_audio_batch)
                    else:
                        X_eeg_batch = torch.FloatTensor(X_eeg_train[batch_indices]).unsqueeze(1).to(device)
                        X_audio_batch = torch.FloatTensor(X_audio_train[batch_indices]).to(device)
                        y_batch = torch.LongTensor(y_train[batch_indices]).to(device)

                        if model_name == 'sync':
                            batch_train_indices = torch.LongTensor(train_idx[batch_indices]).to(device)
                            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                            optimizer.zero_grad()
                            outputs = model(X_eeg_batch, X_audio_batch, batch_train_indices)
                        else:  # simple
                            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                            optimizer.zero_grad()
                            outputs = model(X_eeg_batch, X_audio_batch)

                    y_batch = torch.LongTensor(y_train[batch_indices]).to(device)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

            # 验证（每10个epoch）
            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                for model_name, model in models.items():
                    model.eval()
                    with torch.no_grad():
                        if model_name == 'audio_only':
                            X_audio_val_batch = torch.FloatTensor(X_audio_val).to(device)
                            val_outputs = model(X_audio_val_batch)
                        else:
                            X_eeg_val_batch = torch.FloatTensor(X_eeg_val).unsqueeze(1).to(device)
                            X_audio_val_batch = torch.FloatTensor(X_audio_val).to(device)

                            if model_name == 'sync':
                                val_indices = torch.LongTensor(test_idx).to(device)
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch, val_indices)
                            else:  # simple
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch)

                        y_val_batch = torch.LongTensor(y_val).to(device)
                        _, val_pred = torch.max(val_outputs, 1)
                        val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())

                        if val_acc > best_val_accs[model_name]:
                            best_val_accs[model_name] = val_acc
                            best_models[model_name] = model.state_dict().copy()
                            patience_counters[model_name] = 0
                        else:
                            patience_counters[model_name] += 1

                # 检查是否所有模型都触发早停
                if all(patience_counters[model_name] >= 15 for model_name in models.keys()):
                    print(f"    所有模型早停触发（patience=15），停止训练")
                    break

                print(f"  Epoch {epoch + 1}/{n_epochs}: ", end="")
                for model_name in models.keys():
                    print(f"{model_name}: {best_val_accs[model_name]:.4f} ", end="")
                print()

        # 记录实际训练的epoch数
        actual_epochs = epoch + 1

        # 最终评估（使用每种模型的最佳状态）
        fold_result = {'fold': fold_idx, 'actual_epochs': actual_epochs}

        for model_name, model in models.items():
            if best_models[model_name] is not None:
                model.load_state_dict(best_models[model_name])
            model.eval()
            with torch.no_grad():
                if model_name == 'audio_only':
                    X_audio_val_batch = torch.FloatTensor(X_audio_val).to(device)
                    val_outputs = model(X_audio_val_batch)
                else:
                    X_eeg_val_batch = torch.FloatTensor(X_eeg_val).unsqueeze(1).to(device)
                    X_audio_val_batch = torch.FloatTensor(X_audio_val).to(device)

                    if model_name == 'sync':
                        val_indices = torch.LongTensor(test_idx).to(device)
                        val_outputs = model(X_eeg_val_batch, X_audio_val_batch, val_indices)
                    else:  # simple
                        val_outputs = model(X_eeg_val_batch, X_audio_val_batch)

                val_probs = F.softmax(val_outputs, dim=1)
                _, val_pred = torch.max(val_outputs, 1)
                y_val_batch = torch.LongTensor(y_val).to(device)

                val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
                val_balanced_acc = balanced_accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
                val_auc = roc_auc_score(y_val_batch.cpu().numpy(), val_probs[:, 1].cpu().numpy())

                fold_result[f'{model_name}_accuracy'] = val_acc
                fold_result[f'{model_name}_balanced_accuracy'] = val_balanced_acc
                fold_result[f'{model_name}_auc'] = val_auc

        fold_results.append(fold_result)

        print(f"  Fold {fold_idx} - ", end="")
        for model_name in models.keys():
            acc = fold_result[f'{model_name}_accuracy']
            print(f"{model_name}: {acc:.4f} ", end="")
        print(f"- Epochs: {actual_epochs}")
    
    results_df = pd.DataFrame(fold_results)
    mean_epochs = results_df['actual_epochs'].mean()

    print(f"\n=== {subject_name} 模型结果汇总 ===")
    model_results = {}

    for model_name in MODELS_TO_USE:
        mean_acc = results_df[f'{model_name}_accuracy'].mean()
        mean_balanced_acc = results_df[f'{model_name}_balanced_accuracy'].mean()
        mean_auc = results_df[f'{model_name}_auc'].mean()
        std_acc = results_df[f'{model_name}_accuracy'].std()

        model_results[model_name] = {
            'mean_accuracy': mean_acc,
            'mean_balanced_accuracy': mean_balanced_acc,
            'mean_auc': mean_auc,
            'std_accuracy': std_acc
        }

        print(f"{model_name.upper()} 模型:")
        print(f"  平均验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  平均平衡准确率: {mean_balanced_acc:.4f}")
        print(f"  平均AUC: {mean_auc:.4f}")

    print(f"平均训练轮数: {mean_epochs:.1f} ± {results_df['actual_epochs'].std():.1f}")

    return {
        'subject_name': subject_name,
        'results_df': results_df,
        'mean_epochs': mean_epochs,
        'model_results': model_results
    }


def main():
    """主函数：遍历所有被试，训练指定模型（当前仅 simple）"""
    print("=" * 80)
    print("为每个被试单独训练模型进行对比")
    print("使用模型:", ", ".join(MODELS_TO_USE))
    print("每个被试进行5折交叉验证")
    print("=" * 80)
    
    # 第一步：提取所有被试共享的480个音频对特征（只提取一次）
    print(f"\n{'='*60}")
    print("第一步：提取所有被试共享的音频特征（480个音频对）")
    print(f"{'='*60}")
    
    # 收集所有被试的所有trials，提取独特的音频对名称
    mappings, folder_ranges = load_all_mappings()
    all_unique_trials = []
    
    for subject_name, subject_config in SUBJECTS_CONFIG.items():
        print(f"  收集被试 {subject_name} 的trials...")
        try:
            subject_trials = load_single_subject_trials(subject_name, subject_config, mappings)
            all_unique_trials.extend(subject_trials)
        except Exception as e:
            print(f"  警告: 无法加载被试 {subject_name} 的trials: {e}")
            continue
    
    # 提取独特的音频对名称（应该是480个）
    unique_pair_names = list(set(trial['pair_name'] for trial in all_unique_trials))
    unique_pair_names.sort()
    print(f"\n  发现 {len(unique_pair_names)} 个独特的音频对")
    
    # 创建全局缓存目录
    global_cache_dir = os.path.join(OUTPUT_DIR, 'global_audio_features_cache')
    os.makedirs(global_cache_dir, exist_ok=True)
    
    # 基于音频对列表生成hash（用于缓存文件名）
    pair_names_str = ','.join(sorted(unique_pair_names))
    cache_hash = hashlib.md5(pair_names_str.encode()).hexdigest()[:12]
    cache_path = os.path.join(global_cache_dir, f'audio_features_{cache_hash}.npy')
    cache_info_path = os.path.join(global_cache_dir, f'audio_features_{cache_hash}_info.json')
    
    # 检查缓存是否存在
    cache_loaded = False
    if os.path.exists(cache_path) and os.path.exists(cache_info_path):
        print(f"\n=== 发现音频特征缓存，从缓存加载 ===")
        try:
            # 加载缓存数据
            cache_data = np.load(cache_path, allow_pickle=True).item()
            left_globals = cache_data['left_globals']
            right_globals = cache_data['right_globals']
            cached_pair_names = cache_data['pair_names']
            
            # 加载缓存信息
            with open(cache_info_path, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)
            
            print(f"  缓存文件: {cache_path}")
            print(f"  缓存信息: {len(cached_pair_names)} 个音频对, 特征维度: {cache_info.get('feature_dim', 'N/A')}")
            print(f"  缓存时间: {cache_info.get('cache_time', 'N/A')}")
            
            # 验证缓存是否匹配当前音频对列表
            if cached_pair_names == unique_pair_names:
                print(f"  缓存验证通过，直接使用缓存")
                cache_loaded = True
            else:
                print(f"  警告: 缓存中的音频对列表与当前不匹配，重新提取")
                raise ValueError("缓存不匹配")
                
        except Exception as e:
            print(f"  缓存文件损坏或不匹配，重新提取: {e}")
            # 继续执行提取逻辑
            cache_loaded = False
    else:
        print(f"\n=== 未发现音频特征缓存，开始提取特征 ===")
        cache_loaded = False
    
    # 如果缓存不存在或验证失败，提取特征
    if not cache_loaded:
        # 创建虚拟trials列表用于提取特征
        dummy_trials = [{'pair_name': name} for name in unique_pair_names]
        
        # 提取音频特征（只提取一次）
        print(f"\n  开始提取 {len(unique_pair_names)} 个音频对的特征（所有被试共享）...")
        left_globals, right_globals, feature_mapping = extract_audio_features(dummy_trials, project_root)
        
        # 保存到缓存
        print(f"\n  保存音频特征到缓存...")
        cache_data = {
            'left_globals': left_globals,
            'right_globals': right_globals,
            'pair_names': unique_pair_names
        }
        np.save(cache_path, cache_data)
        
        # 保存缓存信息
        cache_info = {
            'n_pairs': len(unique_pair_names),
            'feature_dim': left_globals.shape[1] if len(left_globals.shape) > 1 else left_globals.shape[0],
            'cache_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pair_names_hash': cache_hash
        }
        with open(cache_info_path, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, indent=2, ensure_ascii=False)
        
        print(f"  缓存已保存: {cache_path}")
        print(f"  下次运行将直接使用缓存，无需重新提取")
    
    # 构建feature_mapping（无论从缓存加载还是新提取都需要）
    if cache_loaded:
        # 从缓存加载，需要重新构建feature_mapping
        feature_mapping = {
            'pair_to_idx': {pair_name: idx for idx, pair_name in enumerate(unique_pair_names)}
        }
    else:
        # feature_mapping已经在extract_audio_features中返回，直接使用
        pass
    
    # 保存共享的音频特征
    shared_audio_features = {
        'left_globals': left_globals,
        'right_globals': right_globals,
        'feature_mapping': feature_mapping
    }
    
    print(f"\n✓ 音频特征准备完成，所有被试将共享这些特征")
    print(f"  左耳特征形状: {left_globals.shape}")
    print(f"  右耳特征形状: {right_globals.shape}")
    
    # 第二步：为每个被试训练模型
    print(f"\n{'='*60}")
    print("第二步：为每个被试训练模型")
    print(f"{'='*60}")
    
    all_subject_results = {}
    
    # 遍历所有被试
    for subject_name in SUBJECTS_CONFIG.keys():
        try:
            subject_results = train_models_for_subject(
                subject_name,
                shared_audio_features,
                unique_pair_names,
                filter_consistent=False
            )
            all_subject_results[subject_name] = subject_results

            # 保存每个被试的结果
            results_path = os.path.join(OUTPUT_DIR, f"comparison_models_{subject_name}_results.csv")
            subject_results['results_df'].to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"\n{subject_name}的结果已保存: {results_path}")

        except Exception as e:
            print(f"\n处理被试 {subject_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有被试模型结果汇总")
    print(f"{'='*60}")

    summary_data = []
    all_model_accuracies = {m: [] for m in MODELS_TO_USE}

    for subject_name, result_dict in all_subject_results.items():
        subject_summary = {
            'subject': subject_name,
            'mean_epochs': result_dict['mean_epochs']
        }

        print(f"\n{subject_name}:")
        for model_name in MODELS_TO_USE:
            model_result = result_dict['model_results'][model_name]
            mean_acc = model_result['mean_accuracy']
            std_acc = model_result['std_accuracy']

            subject_summary[f'{model_name}_mean_accuracy'] = mean_acc
            subject_summary[f'{model_name}_std_accuracy'] = std_acc
            subject_summary[f'{model_name}_mean_balanced_accuracy'] = model_result['mean_balanced_accuracy']
            subject_summary[f'{model_name}_mean_auc'] = model_result['mean_auc']

            all_model_accuracies[model_name].append(mean_acc)
            print(f"  {model_name.upper():10s} - 准确率: {mean_acc:.4f} ± {std_acc:.4f}")

        summary_data.append(subject_summary)

    print(f"\n{'='*60}")
    print("各模型总体平均表现:")
    print(f"{'='*60}")

    overall_results = {}
    for model_name in MODELS_TO_USE:
        overall_mean = np.mean(all_model_accuracies[model_name])
        overall_std = np.std(all_model_accuracies[model_name])
        overall_results[model_name] = {'mean': overall_mean, 'std': overall_std}
        print(f"{model_name.upper():10s} - 总体准确率: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"{'='*60}")
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "comparison_models_all_subjects_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总结果已保存: {summary_path}")

    summary_txt_path = os.path.join(OUTPUT_DIR, "comparison_models_summary.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("模型对比 - 单被试训练结果汇总\n")
        f.write("使用模型: " + ", ".join(MODELS_TO_USE) + "\n")
        f.write("=" * 80 + "\n")
        f.write(f"总被试数: {len(all_subject_results)}\n")
        f.write(f"交叉验证折数: {N_FOLDS}\n\n")

        f.write("各被试结果:\n")
        for subject_name, result_dict in all_subject_results.items():
            f.write(f"\n{subject_name}:\n")
            for model_name in MODELS_TO_USE:
                model_result = result_dict['model_results'][model_name]
                f.write(f"  {model_name.upper():10s} - 准确率: {model_result['mean_accuracy']:.4f} ± {model_result['std_accuracy']:.4f}\n")
            f.write(f"  平均训练轮数: {result_dict['mean_epochs']:.1f}\n")

        f.write(f"\n{'='*80}\n")
        f.write("各模型总体平均表现:\n")
        for model_name in MODELS_TO_USE:
            overall_result = overall_results[model_name]
            f.write(f"  {model_name.upper():10s} - 总体准确率: {overall_result['mean']:.4f} ± {overall_result['std']:.4f}\n")
        f.write(f"{'='*80}\n")

    print(f"文本汇总已保存: {summary_txt_path}")


if __name__ == "__main__":
    main()
