#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人选择位置评估：使用feature_stage_attention_model.py中的所有模型
拟合个人选择（二分类），对比不同模型的效果

- 检查左右平衡一致性（AB和BA必须选择同一个音频）
- 合并成144个独立对
- 对每个模型配置提取特征
- 拟合个人选择（使用注意力模型）
- 对比所有模型的效果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from scipy import signal
from scipy.ndimage import convolve, zoom
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要的函数
from spatiotemporal_glm_model import (
    parse_experiment_data_txt,
    parse_experiment_data_csv,
    parse_audio_mapping,
    parse_pair_name
)
from linear_regression_selected_features import (
    load_or_extract_features,
    build_consistent_folds,
    merge_balanced_pairs_by_type,
)

# 导入linear_regression_loudness_models中的模型配置和特征提取函数
from linear_regression_loudness_models import (
    MODEL_CONFIGS,
    FEATURE_REGISTRY,
    DEFAULT_AUDIO_BASE_DIR,
    build_feature_matrix,
    build_transformed_features_matrix,
    parse_transformation_method,
    SELECTED_FEATURE_INDICES,
    SELECTED_FEATURE_NAMES,
    SELECTED_FEATURE_CONFIGS,
    PAIR_FEATURE_CACHE,
    STD_CACHE,
    _cache_stats
)

# 导入feature_stage_attention_model中的模型和函数
from feature_stage_attention_model import (
    FeatureWeightedModel,
    FeatureStageAttentionModel,
    SharedEarComparisonModel,
    IndependentEarComparisonModel,
    StageAwareSharedComparisonModel,
    StageAwareSharedComparisonUnified,
    StageFeatureDataset,
    _compute_stage_features,
    DEFAULT_STAGE_SECONDS,
    RNG_SEED,
    DEVICE,
    get_feature_configs_from_linear_regression,
    load_or_build_cached_dataset,
    SR,
    HOP
)

# 直接使用 attention_loudness_model 中的 FeatureTimeSeriesDataset，在需要时“现场”提取特征
from attention_loudness_model import FeatureTimeSeriesDataset

CV_SEED = 42
N_FOLDS = 5
MIN_CONSISTENT_TRIALS_WARNING = 50

DATA_COLLECTION_DIR = r"D:\D\research\数据采集"
EXPERIMENT_OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
DEFAULT_MAPPING_FILES = [
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
]

SUBJECT_CONFIGS = [
    {
        'subject_id': 'liyanchen',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "liyanchen_1_20251010_090338.mff", "benchmark_1_10-1-1.txt"),
            os.path.join(DATA_COLLECTION_DIR, "liyanchen_2_20251010_094603.mff", "benchmark_1_10-1-2.txt")
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'shimin',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "shimin_s1_20251010_115112.mff", "benchmark_1_10-1-1.txt"),
            os.path.join(DATA_COLLECTION_DIR, "shimin_s2_20251010_110715.mff", "benchmark_1_10-1-2.txt")
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'jiachen',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "jiachen1017_1_20251017_041306.mff", "benchmark_1_10-1-2.txt"),
            os.path.join(DATA_COLLECTION_DIR, "jiachen1017_2_20251017_050037.mff", "benchmark_1_10-1-2.txt")
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'aiwenkai',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "aiwenkai_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "aiwenkai2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'lironghua',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "lironghua.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "lironghua2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'mayunmiao',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "mayunmiao_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "mayunmiao2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'ShangZiyang',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "ShangZiyang.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "ShangZiyang1.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'wjy',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "wjy1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "wjy_2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'LiuYaorui',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "LiuYaorui_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "Liu Yaorui2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    }
]


def load_jiachen_trials(txt_files, mapping_files):
    """
    加载被试的两个txt文件并合并（通用函数，可用于任何被试）

    Parameters:
    -----------
    txt_files : list
        txt文件路径列表
    mapping_files : list
        mapping文件路径列表

    Returns:
    --------
    all_trials : list
        所有试次数据，包含original_name, response, reaction_time
    """
    print("\n=== 加载被试数据 ===")
    all_trials = []

    for txt_file, mapping_file in zip(txt_files, mapping_files):
        if not os.path.exists(txt_file):
            print(f"警告: TXT文件不存在: {txt_file}")
            continue

        if not os.path.exists(mapping_file):
            print(f"警告: Mapping文件不存在: {mapping_file}")
            continue

        print(f"\n处理文件: {txt_file}")
        trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)

        print(f"  加载了 {len(trials)} 个试次")
        print(f"  Mapping包含 {len(mapping)} 个映射")

        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)

            if filename in mapping:
                original_name = mapping[filename]
                trial_data = {
                    'original_name': original_name,
                    'response': trial['response'],
                    'reaction_time': trial.get('reaction_time', None)
                }
                all_trials.append(trial_data)
            else:
                print(f"  警告: 未找到映射 {filename}")

    print(f"\n总共加载 {len(all_trials)} 个试次")
    rt_count = sum(1 for t in all_trials if t['reaction_time'] is not None)
    print(f"  有RT数据的试次: {rt_count}/{len(all_trials)}")

    return all_trials


def load_subject_trials(subject_config):
    """
    根据配置加载单个被试的trial数据（支持TXT与CSV）
    """
    subject_id = subject_config['subject_id']
    print(f"\n=== 加载被试 {subject_id} 数据 ===")
    trials = []

    txt_files = subject_config.get('txt_files', []) or []
    txt_mapping_files = subject_config.get('txt_mapping_files', subject_config.get('mapping_files', [])) or []

    if txt_files and not txt_mapping_files:
        print(f"  警告: 未提供TXT映射文件，跳过TXT数据")
        txt_files = []

    for txt_file, mapping_file in zip(txt_files, txt_mapping_files):
        if not os.path.exists(txt_file):
            print(f"  TXT文件不存在: {txt_file}")
            continue
        if not os.path.exists(mapping_file):
            print(f"  Mapping文件不存在: {mapping_file}")
            continue

        print(f"  处理TXT: {txt_file}")
        parsed_trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in parsed_trials:
            filename = os.path.basename(trial['wavfile'])
            if filename not in mapping:
                continue
            record = {
                'subject_id': subject_id,
                'original_name': mapping[filename],
                'response': trial['response'],
                'reaction_time': trial.get('reaction_time')
            }
            trials.append(record)

    csv_files = subject_config.get('csv_files', []) or []
    csv_mapping_files = subject_config.get('csv_mapping_files', subject_config.get('mapping_files', [])) or []

    if csv_files and not csv_mapping_files:
        print(f"  警告: 未提供CSV映射文件，跳过CSV数据")
        csv_files = []

    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        if not os.path.exists(csv_file):
            print(f"  CSV文件不存在: {csv_file}")
            continue
        if not os.path.exists(mapping_file):
            print(f"  Mapping文件不存在: {mapping_file}")
            continue

        print(f"  处理CSV: {csv_file}")
        parsed_trials = parse_experiment_data_csv(csv_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in parsed_trials:
            filename = os.path.basename(trial['wavfile'])
            if filename not in mapping:
                continue
            record = {
                'subject_id': subject_id,
                'original_name': mapping[filename],
                'response': trial['response'],
                'reaction_time': None
            }
            trials.append(record)

    print(f"  完成: {len(trials)} 个trial")
    rt_count = sum(1 for t in trials if t.get('reaction_time') is not None)
    if trials:
        print(f"  含RT的trial: {rt_count}/{len(trials)}")

    return trials


def check_left_right_consistency(all_trials, pair_names):
    """
    检查左右平衡一致性，只保留AB和BA都选择同一个音频的试次

    Parameters:
    -----------
    all_trials : list
        所有试次数据
    pair_names : list
        所有音频对名称（288个）

    Returns:
    --------
    consistent_trials : list
        一致的试次数据，包含pair_name, choice, reaction_time
    """
    print("\n=== 检查左右平衡一致性 ===")

    # 创建试次字典：key是original_name，value是trial数据
    trial_dict = {}
    for trial in all_trials:
        pair_name = trial['original_name']
        if pair_name not in trial_dict:
            trial_dict[pair_name] = []
        trial_dict[pair_name].append(trial)

    # 创建规范化键字典：用于匹配AB和BA
    normalized_pairs = {}
    for pair_name in pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            continue

        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))

        if key not in normalized_pairs:
            normalized_pairs[key] = []
        normalized_pairs[key].append(pair_name)

    consistent_trials = []
    inconsistent_count = 0

    # 对于每个规范化键（代表一个独立对）
    for key, pair_names_list in normalized_pairs.items():
        if len(pair_names_list) != 2:
            # 应该有AB和BA两个版本
            continue

        pair_ab = pair_names_list[0]  # AB版本
        pair_ba = pair_names_list[1]  # BA版本

        # 找到对应的试次
        trials_ab = trial_dict.get(pair_ab, [])
        trials_ba = trial_dict.get(pair_ba, [])

        if len(trials_ab) == 0 or len(trials_ba) == 0:
            # 缺少某个版本的试次
            inconsistent_count += 1
            continue

        # 取第一个试次（假设每个音频对只有一个试次）
        trial_ab = trials_ab[0]
        trial_ba = trials_ba[0]

        # 解析AB版本的左右音频
        parse_result_ab = parse_pair_name(pair_ab)
        if parse_result_ab is None:
            continue

        left_cat_ab, left_id_ab, right_cat_ab, right_id_ab, _ = parse_result_ab
        audio_a = f"{left_cat_ab}_{left_id_ab}"  # AB版本的左音频
        audio_b = f"{right_cat_ab}_{right_id_ab}"  # AB版本的右音频

        # 检查一致性
        # AB版本：response=1表示选左（选A），response=2表示选右（选B）
        # BA版本：response=1表示选左（选B），response=2表示选右（选A）

        # 情况1：AB选了A（response=1）且BA选了A（response=2，因为A在BA的右边）
        if trial_ab['response'] == 1 and trial_ba['response'] == 2:
            # 一致：都选了A，保留AB版本，标记为选左（choice=1）
            consistent_trials.append({
                'pair_name': pair_ab,
                'choice': 1,  # 选左（选A）
                'reaction_time': trial_ab.get('reaction_time', None)
            })
        # 情况2：AB选了B（response=2）且BA选了B（response=1，因为B在BA的左边）
        elif trial_ab['response'] == 2 and trial_ba['response'] == 1:
            # 一致：都选了B
            # 如果保留BA版本：BA的response=1表示选左（选B），应该标记为choice=1（选左）
            # 如果保留AB版本：AB的response=2表示选右（选B），应该标记为choice=2（选右）
            # 为了统一，保留AB版本，标记为选右（choice=2）
            consistent_trials.append({
                'pair_name': pair_ab,
                'choice': 2,  # 选右（选B）
                'reaction_time': trial_ab.get('reaction_time', None)
            })
        else:
            # 不一致：去掉
            inconsistent_count += 1

    print(f"  一致的试次: {len(consistent_trials)}")
    print(f"  不一致的试次: {inconsistent_count}")
    if len(consistent_trials) + inconsistent_count > 0:
        print(f"  保留比例: {len(consistent_trials)/(len(consistent_trials)+inconsistent_count)*100:.1f}%")

    return consistent_trials


def build_global_stage_dataset(project_root, base_dir):
    """
    构建“全局” StageFeatureDataset（最多 144 个平衡 pair），供所有被试共享。

    与 feature_stage_attention_model.build_datasets 保持一致：
    - 使用 load_or_extract_features + merge_balanced_pairs_by_type 得到 all_pair_names
    - 针对 all_pair_names 只提取一次多特征时间序列
    - 后续每个被试只是在这个全局数据集里按 pair_name 做索引，不再重复算特征
    """
    print("\n=== 构建全局阶段特征数据集（所有平衡 pair 只提取一次） ===")

    # 1. 基于线性回归脚本的接口得到 144 个平衡 pair 名称
    diff_raw, diff_z, pair_names = load_or_extract_features(project_root)
    merged = merge_balanced_pairs_by_type(pair_names)
    all_pair_names = []
    for key in ["nn_main", "nn_sub", "main", "sub"]:
        if key in merged:
            all_pair_names.extend(merged[key]["pair_names"])

    all_pair_names = list(dict.fromkeys(all_pair_names))  # 去重保持顺序
    print(f"  全局平衡音频对数量: {len(all_pair_names)}")

    if len(all_pair_names) == 0:
        print("  错误: 未能构建任何平衡音频对，无法构建全局 StageFeatureDataset")
        return None

    # 2. 获取特征配置，与其他注意力脚本保持一致
    feature_configs = get_feature_configs_from_linear_regression()

    # 3. 构建 FeatureTimeSeriesDataset（只做一次）
    dummy_targets = np.zeros(len(all_pair_names), dtype=np.float32)
    print("  正在为所有平衡 pair 提取多特征时间序列（一次性）...")
    base_dataset = FeatureTimeSeriesDataset(
        all_pair_names,
        base_dir,
        dummy_targets,
        feature_configs,
        sr=SR,
        hop_length=HOP,
        window_size=5,
        use_high_res=False,
    )

    # 4. 构建 StageFeatureDataset
    stage_dataset = StageFeatureDataset(
        base_dataset,
        stage_seconds=DEFAULT_STAGE_SECONDS,
        feature_configs=feature_configs,
    )
    stage_dataset.pair_names = all_pair_names

    print(f"  阶段特征数据集大小: {len(stage_dataset)}")
    print(f"  阶段数量: {stage_dataset.n_stages}")
    print(f"  特征维度: {stage_dataset.feat_dim}")

    return stage_dataset


def _parse_pair_tokens(pair_name):
    parsed = parse_pair_name(pair_name)
    if parsed is None or parsed[0] is None:
        return None
    left_token = f"{parsed[0]}_{parsed[1]}"
    right_token = f"{parsed[2]}_{parsed[3]}"
    return left_token, right_token


def extract_features_for_attention_model(consistent_trials, stage_dataset):
    """
    为一致的试次提取注意力模型所需的所有特征
    """
    print(f"\n=== 提取注意力模型特征 ===")

    if not hasattr(stage_dataset, "pair_names") or stage_dataset.pair_names is None:
        raise RuntimeError("StageFeatureDataset 缺少 pair_names 信息，无法对齐行为数据")
    canonical_map = {}
    for idx, name in enumerate(stage_dataset.pair_names):
        tokens = _parse_pair_tokens(name)
        if not tokens:
            continue
        canonical_key = tuple(sorted(tokens))
        if canonical_key not in canonical_map:
            canonical_map[canonical_key] = {
                "idx": idx,
                "left_token": tokens[0],
                "right_token": tokens[1],
            }

    X_global_list = []
    X_stage_list = []
    left_globals_list = []
    right_globals_list = []
    left_stage_list = []
    right_stage_list = []
    y_choice_list = []
    missing_pairs = []

    for trial in consistent_trials:
        pair_name = trial['pair_name']
        tokens = _parse_pair_tokens(pair_name)
        if not tokens:
            missing_pairs.append(pair_name)
            continue

        canonical_key = tuple(sorted(tokens))
        entry = canonical_map.get(canonical_key)
        if entry is None:
            missing_pairs.append(pair_name)
            continue

        idx = entry["idx"]
        if idx >= len(stage_dataset.global_feats):
            missing_pairs.append(pair_name)
            continue

        orientation_same = tokens[0] == entry["left_token"]
        orientation_swap = tokens[0] == entry["right_token"]
        if not orientation_same and not orientation_swap:
            missing_pairs.append(pair_name)
            continue

        X_global_list.append(stage_dataset.global_feats[idx].copy())
        X_stage_list.append(stage_dataset.stage_feats[idx].copy())
        left_globals_list.append(stage_dataset.left_globals[idx].copy())
        right_globals_list.append(stage_dataset.right_globals[idx].copy())
        left_stage_list.append(stage_dataset.stage_left_feats[idx].copy())
        right_stage_list.append(stage_dataset.stage_right_feats[idx].copy())
        y_choice_list.append(trial['choice'])

        if orientation_swap:
            # 翻转全局差分特征（mean/std 差都是 left-right）
            X_global_list[-1] *= -1

            # 阶段特征的前半部分是 diff_mean，需要翻转；后半部分是 diff_std，保持不变
            stage_feat = X_stage_list[-1]
            half_dim = stage_feat.shape[1] // 2
            stage_feat[:, :half_dim] *= -1

            # 左右耳特征/序列需要对调
            left_globals_list[-1], right_globals_list[-1] = right_globals_list[-1], left_globals_list[-1]
            left_stage_list[-1], right_stage_list[-1] = right_stage_list[-1], left_stage_list[-1]

    if missing_pairs:
        unique_missing = sorted(set(missing_pairs))
        print(f"  警告: {len(unique_missing)} 个 pair 未在全局特征集中找到，将被忽略")
        preview = unique_missing[:10]
        for name in preview:
            print(f"    - {name}")
        if len(unique_missing) > len(preview):
            print(f"    ... 其余 {len(unique_missing) - len(preview)} 个省略 ...")

    if len(X_global_list) == 0:
        print("  警告: 没有提取到任何有效特征")
        return None

    subject_data = {
        "X_global": np.stack(X_global_list, axis=0),
        "X_stage": np.stack(X_stage_list, axis=0),
        "left_global": np.stack(left_globals_list, axis=0),
        "right_global": np.stack(right_globals_list, axis=0),
        "left_stage": np.stack(left_stage_list, axis=0),
        "right_stage": np.stack(right_stage_list, axis=0),
        "y_choice": np.asarray(y_choice_list, dtype=int),
    }

    print(f"  提取了 {subject_data['X_global'].shape[0]} 个有效试次的特征")
    print(f"  全局特征形状: {subject_data['X_global'].shape}")
    print(f"  阶段特征形状: {subject_data['X_stage'].shape}")
    print(f"  选左: {np.sum(subject_data['y_choice'] == 1)} ({np.mean(subject_data['y_choice'] == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(subject_data['y_choice'] == 2)} ({np.mean(subject_data['y_choice'] == 2)*100:.1f}%)")

    return subject_data


def fit_attention_choice_model(subject_data, model_name):
    """
    按 individual_model_comparison 的折划分方式评估注意力模型
    """
    print(f"\n=== 拟合注意力模型: {model_name} ===")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    y_choice = subject_data["y_choice"]
    y_binary = (y_choice == 2).astype(int)
    n_samples = len(y_binary)

    print(f"  数据统计:")
    print(f"    选左（0）: {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"    选右（1）: {np.sum(y_binary == 1)} ({np.mean(y_binary == 1)*100:.1f}%)")

    if n_samples < 10:
        print("  警告: 样本数太少，跳过拟合")
        return None

    def _init_model():
        if model_name == "FeatureWeightedModel":
            return FeatureWeightedModel(feature_dim=subject_data["X_global"].shape[1]).to(DEVICE)
        if model_name == "FeatureStageAttentionModel":
            n_stages, feat_dim = subject_data["X_stage"].shape[1:]
            return FeatureStageAttentionModel(n_stages=n_stages, feature_dim=feat_dim).to(DEVICE)
        if model_name == "SharedEarComparisonModel":
            ear_dim = subject_data["left_global"].shape[1]
            return SharedEarComparisonModel(ear_feature_dim=ear_dim).to(DEVICE)
        if model_name == "IndependentEarComparisonModel":
            ear_dim = subject_data["left_global"].shape[1]
            return IndependentEarComparisonModel(ear_feature_dim=ear_dim).to(DEVICE)
        if model_name == "StageAwareSharedComparisonModel":
            ear_dim = subject_data["left_stage"].shape[2]
            return StageAwareSharedComparisonModel(ear_feature_dim=ear_dim).to(DEVICE)
        if model_name == "StageAwareSharedComparisonUnified":
            ear_dim = subject_data["left_stage"].shape[2]
            return StageAwareSharedComparisonUnified(ear_feature_dim=ear_dim).to(DEVICE)
        raise ValueError(f"未知模型类型 {model_name}")

    def _build_dataset(indices):
        targets = torch.from_numpy(y_binary[indices]).float()
        if model_name == "FeatureWeightedModel":
            inputs = torch.from_numpy(subject_data["X_global"][indices]).float()
            return TensorDataset(inputs, targets)
        if model_name == "FeatureStageAttentionModel":
            inputs = torch.from_numpy(subject_data["X_stage"][indices]).float()
            return TensorDataset(inputs, targets)
        if model_name in ["SharedEarComparisonModel", "IndependentEarComparisonModel"]:
            left = torch.from_numpy(subject_data["left_global"][indices]).float()
            right = torch.from_numpy(subject_data["right_global"][indices]).float()
            return TensorDataset(left, right, targets)
        if model_name in ["StageAwareSharedComparisonModel", "StageAwareSharedComparisonUnified"]:
            left = torch.from_numpy(subject_data["left_stage"][indices]).float()
            right = torch.from_numpy(subject_data["right_stage"][indices]).float()
            return TensorDataset(left, right, targets)
        raise ValueError(f"未知模型类型 {model_name}")

    def _train_on_indices(indices, epochs=100):
        model = _init_model()
        dataset = _build_dataset(indices)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model.train()
        for _ in range(epochs):
            for batch in loader:
                if model_name in ["SharedEarComparisonModel", "IndependentEarComparisonModel"]:
                    left_feat, right_feat, targets = batch
                    left_feat = left_feat.to(DEVICE)
                    right_feat = right_feat.to(DEVICE)
                    targets = targets.to(DEVICE)
                    outputs, _, _ = model(left_feat, right_feat)
                elif model_name in ["StageAwareSharedComparisonModel", "StageAwareSharedComparisonUnified"]:
                    left_feat, right_feat, targets = batch
                    left_feat = left_feat.to(DEVICE)
                    right_feat = right_feat.to(DEVICE)
                    targets = targets.to(DEVICE)
                    outputs, _, _, _, _ = model(left_feat, right_feat)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    if model_name == "FeatureWeightedModel":
                        outputs, _ = model(inputs)
                    else:
                        outputs, _, _ = model(inputs)

                loss = criterion(outputs.view(-1), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    def _predict(model, indices):
        model.eval()
        with torch.no_grad():
            if model_name in ["SharedEarComparisonModel", "IndependentEarComparisonModel"]:
                left = torch.from_numpy(subject_data["left_global"][indices]).float().to(DEVICE)
                right = torch.from_numpy(subject_data["right_global"][indices]).float().to(DEVICE)
                proba, _, _ = model(left, right)
            elif model_name in ["StageAwareSharedComparisonModel", "StageAwareSharedComparisonUnified"]:
                left = torch.from_numpy(subject_data["left_stage"][indices]).float().to(DEVICE)
                right = torch.from_numpy(subject_data["right_stage"][indices]).float().to(DEVICE)
                proba, _, _, _, _ = model(left, right)
            else:
                if model_name == "FeatureWeightedModel":
                    inputs = torch.from_numpy(subject_data["X_global"][indices]).float().to(DEVICE)
                    proba, _ = model(inputs)
                else:
                    inputs = torch.from_numpy(subject_data["X_stage"][indices]).float().to(DEVICE)
                    proba, _, _ = model(inputs)
        return proba.cpu().numpy().squeeze()

    # 与 individual_model_comparison 一致的折划分
    folds, shuffled_indices = build_consistent_folds(n_samples, seed=CV_SEED, k=N_FOLDS)
    cv_accuracies = []
    cv_balanced_accuracies = []
    cv_aucs = []
    cv_pred_proba = np.zeros(n_samples)
    cv_pred_label = np.zeros(n_samples)

    for fold_idx, test_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx, assume_unique=False)
        model = _train_on_indices(train_idx)
        proba = _predict(model, test_idx)
        preds = (proba > 0.5).astype(int)
        cv_pred_proba[test_idx] = proba
        cv_pred_label[test_idx] = preds
        acc = accuracy_score(y_binary[test_idx], preds)
        balanced_acc = balanced_accuracy_score(y_binary[test_idx], preds)
        try:
            auc = roc_auc_score(y_binary[test_idx], proba)
        except:
            auc = np.nan
        cv_accuracies.append(acc)
        cv_balanced_accuracies.append(balanced_acc)
        cv_aucs.append(auc)
        print(f"    第 {fold_idx}/{N_FOLDS} 折: 准确率 {acc:.4f}, 平衡准确率 {balanced_acc:.4f}, AUC {auc if not np.isnan(auc) else float('nan'):.4f}")

    cv_accuracy_mean = float(np.mean(cv_accuracies))
    cv_accuracy_std = float(np.std(cv_accuracies))
    cv_balanced_accuracy_mean = float(np.mean(cv_balanced_accuracies))
    cv_balanced_accuracy_std = float(np.std(cv_balanced_accuracies))
    if np.any(~np.isnan(cv_aucs)):
        cv_auc_mean = float(np.nanmean(cv_aucs))
        cv_auc_std = float(np.nanstd(cv_aucs))
    else:
        cv_auc_mean = np.nan
        cv_auc_std = np.nan

    # 全量训练，输出最终预测
    final_model = _train_on_indices(np.arange(n_samples))
    final_proba = _predict(final_model, np.arange(n_samples))
    final_preds = (final_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_binary, final_preds)
    balanced_accuracy = balanced_accuracy_score(y_binary, final_preds)
    baseline_accuracy = max(np.mean(y_binary), 1 - np.mean(y_binary))
    try:
        auc = roc_auc_score(y_binary, final_proba)
    except:
        auc = np.nan
    cm = confusion_matrix(y_binary, final_preds)

    print(f"\n  全量训练准确率: {accuracy:.4f}, 平衡准确率: {balanced_accuracy:.4f} (baseline {baseline_accuracy:.4f})")
    print(f"  CV准确率: {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")
    print(f"  CV平衡准确率: {cv_balanced_accuracy_mean:.4f} ± {cv_balanced_accuracy_std:.4f}")

    return {
        "model": final_model,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "auc": auc,
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_std": cv_accuracy_std,
        "cv_balanced_accuracy_mean": cv_balanced_accuracy_mean,
        "cv_balanced_accuracy_std": cv_balanced_accuracy_std,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "confusion_matrix": cm,
        "y_true": y_binary,
        "y_pred": final_preds,
        "y_pred_proba": final_proba,
        "cv_pred_proba": cv_pred_proba,
        "cv_pred_label": cv_pred_label,
        "n_samples": n_samples,
    }


def reset_feature_caches():
    """清空共享缓存，避免被试之间的相互影响"""
    PAIR_FEATURE_CACHE.clear()
    STD_CACHE.clear()
    _cache_stats['pair_feature_hits'] = 0
    _cache_stats['pair_feature_misses'] = 0
    _cache_stats['std_cache_hits'] = 0
    _cache_stats['std_cache_misses'] = 0


# 定义注意力模型配置
ATTENTION_MODEL_CONFIGS = [
    {'model_name': 'FeatureWeightedModel', 'description': '特征加权模型'},
    {'model_name': 'FeatureStageAttentionModel', 'description': '特征阶段注意力模型'},
    {'model_name': 'SharedEarComparisonModel', 'description': '共享耳朵比较模型'},
    {'model_name': 'IndependentEarComparisonModel', 'description': '独立耳朵比较模型'},
    {'model_name': 'StageAwareSharedComparisonModel', 'description': '阶段感知共享比较模型'},
    {'model_name': 'StageAwareSharedComparisonUnified', 'description': '阶段感知共享时间比较模型'},
]


def run_attention_models_for_subject(subject_id, consistent_trials, output_root, stage_dataset):
    """针对单个被试运行全部注意力模型并保存结果"""
    print(f"\n{'=' * 60}")
    print(f"开始评估被试: {subject_id}")
    print(f"{'=' * 60}")

    reset_feature_caches()
    all_results = []

    subject_data = extract_features_for_attention_model(consistent_trials, stage_dataset)

    if subject_data is None or subject_data["X_global"].shape[0] < 10:
        count = 0 if subject_data is None else subject_data["X_global"].shape[0]
        print(f"  警告: 有效试次太少（{count}），跳过该被试")
        return []

    for model_idx, model_config in enumerate(ATTENTION_MODEL_CONFIGS, 1):
        model_name = model_config['model_name']
        print(f"\n{'-' * 60}")
        print(f"[{subject_id}] 模型 {model_idx}/{len(ATTENTION_MODEL_CONFIGS)}: {model_name}")
        print(f"{'-' * 60}")

        try:
            result = fit_attention_choice_model(subject_data, model_name)

            if result is not None:
                result['model_name'] = model_name
                result['model_idx'] = model_idx
                result['subject_id'] = subject_id
                result['n_samples'] = len(result.get('y_true', []))
                all_results.append(result)
            else:
                all_results.append(None)

        except Exception as e:
            print(f"  错误: 处理模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)

    output_dir = os.path.join(output_root, subject_id)
    os.makedirs(output_dir, exist_ok=True)

    plot_attention_model_comparison(all_results, output_dir)
    save_attention_results(all_results, consistent_trials, output_dir, subject_id=subject_id)

    valid_results = [r for r in all_results if r is not None]

    print(f"\n{'=' * 60}")
    print(f"{subject_id} 注意力模型对比总结")
    print(f"{'=' * 60}")
    if valid_results:
        # 与 individual_model_comparison 中保持一致：按 CV 准确率排序
        valid_results.sort(key=lambda x: x.get('cv_accuracy_mean', 0), reverse=True)
        print(f"\n成功拟合 {len(valid_results)} 个模型")
        top_k = min(5, len(valid_results))
        print(f"\n前{top_k}名模型（按CV准确率）:")
        for i, result in enumerate(valid_results[:top_k], 1):
            print(f"\n{i}. {result['model_name']}")
            print(f"   准确率: {result['accuracy']:.4f}")
            print(f"   平衡准确率: {result.get('balanced_accuracy', float('nan')):.4f}")
            print(f"   基线准确率: {result['baseline_accuracy']:.4f}")
            print(f"   提升: {(result['accuracy'] - result['baseline_accuracy'])*100:.2f}%")
            print(f"   AUC: {result['auc']:.4f}")
            print(f"   CV准确率: {result.get('cv_accuracy_mean', float('nan')):.4f} ± {result.get('cv_accuracy_std', float('nan')):.4f}")
            print(f"   CV平衡准确率: {result.get('cv_balanced_accuracy_mean', float('nan')):.4f} ± {result.get('cv_balanced_accuracy_std', float('nan')):.4f}")
    else:
        print("没有成功拟合的模型")

    print(f"\n结果已保存到: {output_dir}")

    return valid_results


def plot_attention_model_comparison(all_results, output_dir):
    """绘制注意力模型对比图表"""
    print("\n=== 生成注意力模型对比图表 ===")

    os.makedirs(output_dir, exist_ok=True)

    if not all_results:
        print("  没有结果可绘制")
        return

    # 提取所有模型的性能指标
    model_names = []
    accuracies = []
    aucs = []
    cv_accuracies = []

    for result in all_results:
        if result is None:
            continue
        model_names.append(result.get('model_name', 'Unknown'))
        accuracies.append(result.get('accuracy', np.nan))
        aucs.append(result.get('auc', np.nan))
        cv_accuracies.append(result.get('cv_accuracy_mean', np.nan))

    if len(model_names) == 0:
        print("  没有有效结果")
        return

    # 创建对比图（与 individual_model_comparison 中的结构一致：Accuracy / AUC / CV Accuracy）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 准确率对比
    ax = axes[0]
    sorted_idx = np.argsort(accuracies)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_acc = [accuracies[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_acc)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('准确率', fontsize=12)
    ax.set_title('注意力模型准确率对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # 2. AUC对比
    ax = axes[1]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_auc = [aucs[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_auc)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('注意力模型AUC对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # 3. CV准确率对比
    ax = axes[2]
    sorted_idx = np.argsort(cv_accuracies)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_cv_acc = [cv_accuracies[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_cv_acc)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('CV准确率', fontsize=12)
    ax.set_title('注意力模型交叉验证准确率对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  对比图表已保存到: {output_dir}/attention_model_comparison.png")


def save_attention_results(all_results, consistent_trials, output_dir, subject_id=None):
    """保存所有注意力模型的结果到CSV文件"""
    print("\n=== 保存注意力模型结果 ===")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存试次数据
    trials_df = pd.DataFrame(consistent_trials)
    trials_df.to_csv(os.path.join(output_dir, 'consistent_trials.csv'), index=False, encoding='utf-8-sig')

    # 2. 保存模型对比汇总
    summary_data = []
    for result in all_results:
        if result is None:
            continue
        summary_data.append({
            'subject_id': subject_id,
            'model_name': result.get('model_name', 'Unknown'),
            'accuracy': result.get('accuracy', np.nan),
            'balanced_accuracy': result.get('balanced_accuracy', np.nan),
            'baseline_accuracy': result.get('baseline_accuracy', np.nan),
            'improvement': result.get('accuracy', np.nan) - result.get('baseline_accuracy', np.nan),
            'auc': result.get('auc', np.nan),
            'cv_accuracy_mean': result.get('cv_accuracy_mean', np.nan),
            'cv_accuracy_std': result.get('cv_accuracy_std', np.nan),
            'cv_balanced_accuracy_mean': result.get('cv_balanced_accuracy_mean', np.nan),
            'cv_balanced_accuracy_std': result.get('cv_balanced_accuracy_std', np.nan),
            'n_samples': len(result.get('y_true', []))
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # 与 individual_model_comparison 保持一致：按 CV 准确率排序
        if 'cv_accuracy_mean' in summary_df.columns:
            summary_df = summary_df.sort_values('cv_accuracy_mean', ascending=False)
        else:
            summary_df = summary_df.sort_values('accuracy', ascending=False)
        summary_df.to_csv(os.path.join(output_dir, 'attention_model_comparison_summary.csv'), index=False, encoding='utf-8-sig')
        print(f"  注意力模型对比汇总已保存: attention_model_comparison_summary.csv")

    # 3. 为每个模型保存详细结果
    for result in all_results:
        if result is None:
            continue
        model_name = result.get('model_name', 'Unknown')
        safe_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')

        model_dir = os.path.join(output_dir, safe_name)
        os.makedirs(model_dir, exist_ok=True)

        # 保存预测结果
        pred_df = pd.DataFrame({
            'actual_choice': result.get('y_true', []),
            'predicted_choice': result.get('y_pred', []),
            'predicted_proba': result.get('y_pred_proba', [])
        })
        pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')

    print(f"  所有注意力模型结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("个人选择位置评估：使用feature_stage_attention_model中的所有模型")
    print("=" * 60)

    project_root = os.getcwd()
    output_root = os.path.join(project_root, 'individual_choice_with_position_results')
    os.makedirs(output_root, exist_ok=True)

    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR

    # 构建一次全局 StageFeatureDataset，所有被试共享，避免重复提特征
    stage_dataset = build_global_stage_dataset(project_root, base_dir)
    if stage_dataset is None:
        print("全局阶段特征数据集构建失败，程序终止。")
        return

    overall_summary_rows = []
    processed_subjects = 0

    for subject_config in SUBJECT_CONFIGS:
        subject_id = subject_config['subject_id']
        trials = load_subject_trials(subject_config)

        if len(trials) == 0:
            print(f"\n跳过 {subject_id}: 未能加载任何trial数据")
            continue

        consistent_trials = check_left_right_consistency(trials, pair_names)
        if len(consistent_trials) == 0:
            print(f"\n跳过 {subject_id}: 没有满足左右一致性的试次")
            continue

        if len(consistent_trials) < MIN_CONSISTENT_TRIALS_WARNING:
            print(f"\n警告: {subject_id} 的一致试次仅 {len(consistent_trials)} (<{MIN_CONSISTENT_TRIALS_WARNING})，结果可靠性可能较低（仍继续）")

        for trial in consistent_trials:
            trial['subject_id'] = subject_id

        subject_results = run_attention_models_for_subject(
            subject_id,
            consistent_trials,
            output_root,
            stage_dataset,
        )

        processed_subjects += 1

        for result in subject_results:
            overall_summary_rows.append({
                'subject_id': subject_id,
                'model_idx': result.get('model_idx'),
                'model_name': result.get('model_name', 'Unknown'),
                'accuracy': result.get('accuracy', np.nan),
                'balanced_accuracy': result.get('balanced_accuracy', np.nan),
                'baseline_accuracy': result.get('baseline_accuracy', np.nan),
                'improvement': result.get('accuracy', np.nan) - result.get('baseline_accuracy', np.nan),
                'auc': result.get('auc', np.nan),
                'cv_accuracy_mean': result.get('cv_accuracy_mean', np.nan),
                'cv_accuracy_std': result.get('cv_accuracy_std', np.nan),
                'cv_balanced_accuracy_mean': result.get('cv_balanced_accuracy_mean', np.nan),
                'cv_balanced_accuracy_std': result.get('cv_balanced_accuracy_std', np.nan),
                'n_samples': result.get('n_samples', len(result.get('y_true', [])))
            })

    if overall_summary_rows:
        summary_df = pd.DataFrame(overall_summary_rows)
        # 与 individual_model_comparison 保持一致：按 CV 准确率排序
        sort_cols = ['subject_id']
        if 'cv_accuracy_mean' in summary_df.columns:
            sort_cols.append('cv_accuracy_mean')
            summary_df = summary_df.sort_values(sort_cols, ascending=[True, False])
        else:
            sort_cols.append('accuracy')
            summary_df = summary_df.sort_values(sort_cols, ascending=[True, False])
        summary_path = os.path.join(output_root, 'all_subjects_attention_model_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n所有被试注意力模型对比汇总已保存: {summary_path}")
    else:
        print("\n未生成任何被试的模型结果")

    print(f"\n共处理 {processed_subjects}/{len(SUBJECT_CONFIGS)} 个被试")


if __name__ == '__main__':
    main()