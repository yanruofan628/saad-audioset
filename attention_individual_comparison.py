#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用注意力模型进行个人数据模型对比
基于individual_model_comparison.py和attention_loudness_model.py
使用attention_loudness_model.py中的模型替代逻辑回归
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

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
    calculate_selection_probability_144
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

# 导入attention_loudness_model中的模型和函数
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_loudness_model import (
    get_feature_configs_from_linear_regression,
    extract_feature_time_series,
    load_mono_5s,
    ConvPoolAttentionModel,
    ConvPoolOnlyModel,
    StatisticsOnlyModel,
    SimpleAttentionModel,
    DualStreamAttention,
    train_model,
    get_trials,
    SR,
    HOP
)


class AccumulationAttentionModel(nn.Module):
    """
    被动注意积累模型 - 基于生物机制的计算模型

    架构：
    1. accumulation_rates: 特征积累权重 (生物参数)
    2. feature_enhancer: 特征增强网络 (神经网络)
    3. temporal_accumulator: 时间积累器 (GRU + 记忆衰减)
    4. decision_net: 决策网络 (分类器)
    """

    def __init__(self, n_features=5, time_steps=10, d_model=32):
        """
        初始化积累模型

        Args:
            n_features: 特征数量 (默认5: 响度, 频谱质心, 调制, 不规则度, 时域)
            time_steps: 时间步数 (默认10, 对应5秒音频)
            d_model: 模型维度 (默认32)
        """
        super().__init__()

        # 保存参数
        self.n_features = n_features
        self.time_steps = time_steps
        self.d_model = d_model

        # ================ 第一层：特征积累参数 ================
        # 生物可解释参数：每个特征的注意积累效率
        # 形状: (n_features,) - [rate_响度, rate_频谱质心, rate_调制, ...]
        self.accumulation_rates = nn.Parameter(
            torch.ones(n_features, dtype=torch.float32),
            requires_grad=True
        )

        # ================ 第二层：特征增强器 ================
        # 输入: 单时间步的左右耳特征 (batch, 2*n_features)
        # 输出: 增强后的特征表示 (batch, d_model//2)
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2 * n_features, d_model),        # 拼接左右耳特征
            nn.ReLU(),                                  # 非线性激活
            nn.Linear(d_model, d_model // 2)           # 降维到d_model//2
        )

        # ================ 第三层：时间积累器 ================
        # GRU单元：处理时间序列积累
        # 输入: (batch, d_model//2)
        # 隐藏: (batch, d_model//2)
        self.gru_cell = nn.GRUCell(
            input_size=d_model // 2,    # 输入特征维度
            hidden_size=d_model // 2    # 隐藏状态维度
        )

        # 记忆衰减参数：控制注意信息的保持时间
        # 值在0-1之间，0.8表示80%的信息保留到下一时间步
        self.memory_decay = nn.Parameter(
            torch.tensor(0.8, dtype=torch.float32),
            requires_grad=True
        )

        # ================ 第四层：决策网络 ================
        # 输入: 最终积累状态 (batch, d_model//2)
        # 输出: 选择概率 (batch, 1)
        self.decision_net = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),    # 进一步降维
            nn.ReLU(),                                  # 非线性
            nn.Dropout(0.3),                           # 防止过拟合
            nn.Linear(d_model // 4, 1),                # 二分类输出
            nn.Sigmoid()                               # 概率输出 (0-1)
        )

        # ================ 初始化 ================
        # 使用Xavier初始化，确保训练稳定性
        for name, param in self.named_parameters():
            if param.dim() > 1:  # 只初始化权重矩阵
                nn.init.xavier_uniform_(param, gain=0.1)

        # 特殊初始化生物参数
        with torch.no_grad():
            self.accumulation_rates.fill_(1.0)  # 初始都设为1
            self.memory_decay.fill_(0.8)         # 初始衰减0.8

    def forward(self, x):
        """
        前向传播：实现完整的被动注意积累过程

        Args:
            x: 输入特征 (batch, 2*n_features, time_steps)
               例如: (32, 10, 10) - 32样本，5特征×2耳，10时间步

        Returns:
            choice_prob: 选择概率 (batch, 1) - 值在0-1之间
            info: 解释信息字典
        """
        batch_size = x.shape[0]
        n_channels = x.shape[1]  # 应该是 2*n_features
        T = x.shape[2]          # 时间步数

        # ==================== 阶段1：特征积累 ====================
        # 应用积累权重：模拟不同特征的注意捕获效率
        # x: (batch, 2*n_features, T)
        # accumulation_rates: (n_features,)
        # 需要广播：(n_features,) -> (1, n_features, 1) -> (1, 2*n_features, 1)

        # 为左右耳分别应用积累率
        rates_expanded = self.accumulation_rates.unsqueeze(0).unsqueeze(-1)  # (1, n_features, 1)
        rates_doubled = torch.cat([rates_expanded, rates_expanded], dim=1)   # (1, 2*n_features, 1)

        x_accumulated = x * rates_doubled  # (batch, 2*n_features, T)

        # ==================== 阶段2：时间转置 ====================
        # 改变维度顺序：(batch, 2*n_features, T) -> (batch, T, 2*n_features)
        # 准备按时间步处理
        x_time_first = x_accumulated.transpose(1, 2)  # (batch, T, 2*n_features)

        # ==================== 阶段3：逐时间步特征增强 ====================
        enhanced_sequence = []

        for t in range(T):
            # 取出第t个时间步
            time_slice = x_time_first[:, t, :]  # (batch, 2*n_features)

            # 特征增强：整合左右耳信息
            enhanced = self.feature_enhancer(time_slice)  # (batch, d_model//2)

            enhanced_sequence.append(enhanced)

        # 堆叠成序列：(T, batch, d_model//2)
        enhanced_seq = torch.stack(enhanced_sequence, dim=0)

        # 转置为(batch, T, d_model//2)以适应GRU
        enhanced_seq = enhanced_seq.transpose(0, 1)  # (batch, T, d_model//2)

        # ==================== 阶段4：时间积累 ====================
        # 初始化积累状态
        accumulation_state = torch.zeros(
            batch_size,
            self.d_model // 2,
            device=x.device,
            dtype=x.dtype
        )

        # 记录积累轨迹（用于分析）
        accumulation_trajectory = []

        # 逐时间步积累
        for t in range(T):
            current_input = enhanced_seq[:, t, :]  # (batch, d_model//2)

            # GRU更新
            new_state = self.gru_cell(current_input, accumulation_state)

            # 应用记忆衰减：保留部分旧信息 + 加入新信息
            # 公式: state = new_state * decay + current_input * (1-decay)
            accumulation_state = (
                new_state * self.memory_decay +
                current_input * (1 - self.memory_decay)
            )

            accumulation_trajectory.append(accumulation_state.clone())

        # ==================== 阶段5：决策 ====================
        # 基于最终积累状态做选择
        choice_prob = self.decision_net(accumulation_state)  # (batch, 1)

        # ==================== 解释信息 ====================
        info = {
            'accumulation_rates': self.accumulation_rates.detach().cpu(),
            'memory_decay': self.memory_decay.detach().cpu(),
            'accumulation_trajectory': torch.stack(accumulation_trajectory, dim=1),  # (batch, T, d_model//2)
            'final_accumulation': accumulation_state.detach().cpu(),
            'enhanced_sequence': enhanced_seq.detach().cpu()
        }

        return choice_prob, info

    def get_interpretation(self):
        """
        获取模型的可解释性参数

        Returns:
            dict: 包含生物可解释参数
        """
        return {
            'accumulation_rates': self.accumulation_rates.detach().cpu().numpy(),
            'memory_decay': self.memory_decay.detach().cpu().item(),
            'feature_importance': F.softmax(self.accumulation_rates, dim=0).detach().cpu().numpy()
        }

    def __repr__(self):
        """模型描述"""
        return (f"AccumulationAttentionModel(\n"
                f"  n_features={self.n_features},\n"
                f"  accumulation_rates={self.accumulation_rates.detach().cpu().tolist()},\n"
                f"  memory_decay={self.memory_decay.detach().cpu().item():.3f},\n"
                f"  d_model={self.d_model}\n"
                f")")

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

    Parameters:
    -----------
    subject_config : dict
        被试配置字典，包含subject_id、txt_files、csv_files等

    Returns:
    --------
    trials : list
        所有试次数据，包含subject_id, original_name, response, reaction_time
    """
    subject_id = subject_config['subject_id']
    print(f"\n=== 加载被试 {subject_id} 数据 ===")
    trials = []

    # 处理TXT文件
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

    # 处理CSV文件
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
            # 一致：都选了B，保留AB版本，标记为选右（choice=2）
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


def build_time_series_features_for_trials(consistent_trials, pair_names, base_dir, feature_configs):
    """
    为一致的试次构建时间序列特征

    Parameters:
    -----------
    consistent_trials : list
        一致的试次数据
    pair_names : list
        所有音频对名称
    base_dir : str
        音频文件基础目录
    feature_configs : list
        特征配置列表

    Returns:
    --------
    X : np.ndarray, shape (n_trials, 2*n_features, T)
        时间序列特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择标签
    """
    print(f"\n=== 构建时间序列特征 ===")

    from linear_regression_selected_features import load_or_extract_features
    diff_raw, diff_zscore, _ = load_or_extract_features(os.getcwd())

    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}

    # 提取时间序列特征
    X_list = []
    y_choice_list = []

    for i, trial in enumerate(consistent_trials):
        if (i + 1) % 10 == 0:
            print(f"  处理试次: {i+1}/{len(consistent_trials)}")

        pair_name = trial['pair_name']

        if pair_name not in pair_to_idx:
            continue

        try:
            # 解析音频对
            parse_result = parse_pair_name(pair_name)
            if parse_result is None:
                continue

            left_cat, left_id, right_cat, right_id, _ = parse_result
            left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
            right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")

            if not (os.path.exists(left_path) and os.path.exists(right_path)):
                continue

            # 加载音频
            yL, sr = load_mono_5s(left_path)
            yR, _ = load_mono_5s(right_path)

            # 提取特征
            left_features = []
            right_features = []

            for feat_type, compute_std in feature_configs:
                feat_L = extract_feature_time_series(
                    yL, sr=sr, feature_type=feat_type,
                    hop_length=HOP, compute_std=compute_std, window_size=5
                )
                feat_R = extract_feature_time_series(
                    yR, sr=sr, feature_type=feat_type,
                    hop_length=HOP, compute_std=compute_std, window_size=5
                )

                # 确保特征是一维数组
                feat_L = np.asarray(feat_L).flatten()
                feat_R = np.asarray(feat_R).flatten()

                # 处理NaN和Inf
                feat_L = np.nan_to_num(feat_L, nan=0.0, posinf=0.0, neginf=0.0)
                feat_R = np.nan_to_num(feat_R, nan=0.0, posinf=0.0, neginf=0.0)

                left_features.append(feat_L)
                right_features.append(feat_R)

            # 找到最小长度
            min_len = min(min(len(f) for f in left_features), min(len(f) for f in right_features))
            if min_len <= 0:
                continue

            # 对齐长度
            left_features = [f[:min_len] for f in left_features]
            right_features = [f[:min_len] for f in right_features]

            # 拼接为 (2*n_features, T)
            x = np.concatenate([np.stack(left_features), np.stack(right_features)], axis=0)

            X_list.append(x)
            y_choice_list.append(trial['choice'])

        except Exception as e:
            print(f"  警告: 处理 {pair_name} 失败: {e}")
            continue

    if len(X_list) == 0:
        print("  警告: 没有提取到任何有效特征")
        return np.array([]), np.array([])

    # 转换为numpy数组
    X = np.stack(X_list, axis=0)  # (n_trials, 2*n_features, T)
    y_choice = np.array(y_choice_list)

    # 标准化每个特征通道
    for ch_idx in range(X.shape[1]):
        ch_data = X[:, ch_idx, :].flatten()
        ch_mean = np.nanmean(ch_data)
        ch_std = np.nanstd(ch_data) + 1e-8
        X[:, ch_idx, :] = (X[:, ch_idx, :] - ch_mean) / ch_std

    print(f"  提取了 {len(X)} 个有效试次的特征")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  选左: {np.sum(y_choice == 1)} ({np.mean(y_choice == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(y_choice == 2)} ({np.mean(y_choice == 2)*100:.1f}%)")

    return X, y_choice


class IndividualChoiceDataset(Dataset):
    """个人选择数据集"""
    def __init__(self, X, y_choice):
        self.X = X.astype(np.float32)
        self.y = (y_choice == 2).astype(np.float32)  # 转换为二分类: 1=选左->0, 2=选右->1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def fit_attention_choice_model(X, y_choice, feature_names, model_name, model_type='conv_pool_only'):
    """
    使用注意力模型拟合个人选择（与individual_model_comparison.py完全一致的数据划分方式）

    Parameters:
    -----------
    X : np.ndarray, shape (n_trials, 2*n_features, T)
        时间序列特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    feature_names : list
        特征名称列表
    model_name : str
        模型名称
    model_type : str
        注意力模型类型

    Returns:
    --------
    results : dict
        拟合结果
    """
    print(f"\n=== 使用注意力模型拟合个人选择: {model_name} ===")

    # 转换为二分类：1=选左 -> 0, 2=选右 -> 1
    y_binary = (y_choice == 2).astype(int)

    print(f"  数据统计:")
    print(f"    选左（0）: {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"    选右（1）: {np.sum(y_binary == 1)} ({np.mean(y_binary == 1)*100:.1f}%)")

    if len(y_binary) < 10:
        print("  警告: 样本数太少，跳过拟合")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集
    dataset = IndividualChoiceDataset(X, y_choice)

    # 使用与individual_model_comparison.py完全一致的交叉验证方式
    # sklearn的cross_val_score默认使用StratifiedKFold，保证类别平衡
    print(f"  使用5折交叉验证（与individual_model_comparison.py一致）...")

    # 为了获得与individual_model_comparison.py完全一致的结果，
    # 我们需要使用相同的交叉验证设置
    cv_scores = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []

    # 使用sklearn的cross_val_score，但手动收集预测结果
    from sklearn.model_selection import StratifiedKFold

    # 创建与individual_model_comparison.py完全一致的CV设置
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_binary)):
        print(f"    折 {fold_idx + 1}/{N_FOLDS}...")

        # 创建训练和测试数据集
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        # 创建模型
        n_features = len(feature_names)
        T = X.shape[2]

        if model_type == 'conv_pool_attention':
            model = ConvPoolAttentionModel(n_features=n_features, d_model=64, dropout=0.2)
        elif model_type == 'conv_pool_only':
            model = ConvPoolOnlyModel(n_features=n_features, d_model=64, dropout=0.2)
        elif model_type == 'statistics_only':
            model = StatisticsOnlyModel(n_features=n_features, hidden_dim=64, dropout=0.2)
        elif model_type == 'dual_stream':
            model = DualStreamAttention(time_steps=T, n_features=n_features, dim=64, dropout=0.2)
        else:  # simple_attention
            time_input_dim = 2 * n_features
            model = SimpleAttentionModel(time_input_dim=time_input_dim, d_model=64, dropout=0.2)

        model = model.to(device)

        # 训练模型（使用更少的epoch，因为是CV）
        # 创建临时输出目录用于保存模型
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            train_model(
                model, train_dataset, None, epochs=50, device=device,  # 不使用验证集，早停
                output_dir=temp_dir, lr=1e-3, weight_decay=1e-4, patience=10
            )

        # 在测试集上评估
        model.eval()
        fold_preds = []
        fold_targets = []
        fold_pred_proba = []

        with torch.no_grad():
            for xb, yb in DataLoader(test_dataset, batch_size=32, shuffle=False):
                xb = xb.to(device)
                if model_type == 'dual_stream':
                    pred, _, _ = model(xb)
                else:
                    pred, _ = model(xb)
                fold_preds.append(pred.cpu().numpy())
                fold_targets.append(yb.cpu().numpy())
                fold_pred_proba.append(pred.cpu().numpy())

        fold_y_pred_proba = np.concatenate(fold_pred_proba)
        fold_y_pred = (fold_y_pred_proba > 0.5).astype(int)
        fold_y_true = np.concatenate(fold_targets)

        # 计算该折的准确率（与individual_model_comparison.py中的cross_val_score一致）
        fold_accuracy = accuracy_score(fold_y_true, fold_y_pred)
        cv_scores.append(fold_accuracy)

        # 收集所有预测结果用于总体评估
        all_y_true.extend(fold_y_true)
        all_y_pred.extend(fold_y_pred)
        all_y_pred_proba.extend(fold_y_pred_proba)

    # 转换回numpy数组
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_pred_proba = np.array(all_y_pred_proba)
    cv_scores = np.array(cv_scores)

    # 计算总体指标（使用所有CV折的预测结果）
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan

    baseline_accuracy = max(np.mean(y_true), 1 - np.mean(y_true))

    print(f"\n  预测准确率:")
    print(f"    准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    基线（多数类）: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"    提升: {(accuracy - baseline_accuracy)*100:.2f}%")
    print(f"    AUC: {auc:.4f}")
    print(f"    CV准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  混淆矩阵:")
    print(f"    TN（真负例，预测左实际左）: {cm[0,0]}")
    print(f"    FP（假正例，预测右实际左）: {cm[0,1]}")
    print(f"    FN（假负例，预测左实际右）: {cm[1,0]}")
    print(f"    TP（真正例，预测右实际右）: {cm[1,1]}")

    return {
        'model': None,  # CV中不保存最终模型
        'model_type': model_type,
        'accuracy': accuracy,
        'baseline_accuracy': baseline_accuracy,
        'auc': auc,
        'cv_accuracy_mean': np.mean(cv_scores),
        'cv_accuracy_std': np.std(cv_scores),
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': feature_names,
        'n_samples': len(y_true)
    }


def plot_model_comparison(all_results, output_dir):
    """绘制模型对比图表"""
    print("\n=== 生成模型对比图表 ===")

    os.makedirs(output_dir, exist_ok=True)

    if not all_results:
        print("  没有结果可绘制")
        return

    # 提取所有模型的性能指标
    model_names = []
    accuracies = []
    aucs = []
    cv_r2s = []

    for result in all_results:
        if result is None:
            continue
        model_names.append(result.get('model_name', 'Unknown'))
        accuracies.append(result.get('accuracy', np.nan))
        aucs.append(result.get('auc', np.nan))
        cv_r2s.append(result.get('cv_r2_mean', np.nan))

    if len(model_names) == 0:
        print("  没有有效结果")
        return

    # 创建对比图
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
    ax.set_title('模型准确率对比', fontsize=14, fontweight='bold')
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
    ax.set_title('模型AUC对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # 3. 交叉验证R2对比
    ax = axes[2]
    sorted_idx = np.argsort(cv_r2s)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_cv_r2 = [cv_r2s[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_cv_r2)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('CV R²', fontsize=12)
    ax.set_title('模型交叉验证R²对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  对比图表已保存到: {output_dir}/attention_model_comparison.png")


def save_results(all_results, consistent_trials, output_dir, subject_id=None):
    """保存所有模型的结果到CSV文件"""
    print("\n=== 保存结果 ===")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存试次数据
    if consistent_trials:
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
            'model_type': result.get('model_type', 'Unknown'),
            'accuracy': result.get('accuracy', np.nan),
            'baseline_accuracy': result.get('baseline_accuracy', np.nan),
            'improvement': result.get('accuracy', np.nan) - result.get('baseline_accuracy', np.nan),
            'auc': result.get('auc', np.nan),
            'cv_accuracy_mean': result.get('cv_accuracy_mean', np.nan),
            'cv_accuracy_std': result.get('cv_accuracy_std', np.nan),
            'n_features': len(result.get('feature_names', [])),
            'n_samples': result.get('n_samples', 0)
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('cv_accuracy_mean', ascending=False)
        summary_df.to_csv(os.path.join(output_dir, 'attention_model_comparison_summary.csv'), index=False, encoding='utf-8-sig')
        print(f"  模型对比汇总已保存: attention_model_comparison_summary.csv")

    # 3. 为每个模型保存详细结果
    for result in all_results:
        if result is None:
            continue
        model_name = result.get('model_name', 'Unknown')
        model_type = result.get('model_type', 'Unknown')
        safe_name = f"{model_name}_{model_type}".replace('/', '_').replace('\\', '_').replace(':', '_')

        model_dir = os.path.join(output_dir, safe_name)
        os.makedirs(model_dir, exist_ok=True)

        # 保存预测结果
        pred_df = pd.DataFrame({
            'actual_choice': result.get('y_true', []),
            'predicted_choice': result.get('y_pred', []),
            'predicted_proba': result.get('y_pred_proba', [])
        })
        pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')

    print(f"  所有结果已保存到: {output_dir}")


def demonstrate_attention_model_prediction(X, y_choice, model_type='conv_pool_only'):
    """
    演示单个注意力模型如何进行二分类预测

    Parameters:
    -----------
    X : np.ndarray, shape (n_trials, 2*n_features, T)
        时间序列特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    model_type : str
        注意力模型类型
    """
    print(f"\n{'='*60}")
    print(f"演示：{model_type} 模型进行二分类预测")
    print("任务：根据音频特征预测用户选择左还是右")
    print(f"{'='*60}")

    # 转换为二分类：1=选左 -> 0, 2=选右 -> 1
    y_binary = (y_choice == 2).astype(int)
    print("\n二分类标签：")
    print(f"  选择左耳音频 -> 标签 0")
    print(f"  选择右耳音频 -> 标签 1")
    print(f"  数据分布: {np.sum(y_binary == 0)} 个选左, {np.sum(y_binary == 1)} 个选右")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集
    dataset = IndividualChoiceDataset(X, y_choice)

    # 简单划分：80%训练，20%测试
    n_samples = len(dataset)
    indices = np.random.RandomState(CV_SEED).permutation(n_samples)
    train_size = int(0.8 * n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"\n数据划分：")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 创建模型
    n_features = X.shape[1] // 2  # 因为X是(2*n_features, T)
    T = X.shape[2]

    if model_type == 'conv_pool_attention':
        model = ConvPoolAttentionModel(n_features=n_features, d_model=64, dropout=0.2)
    elif model_type == 'conv_pool_only':
        model = ConvPoolOnlyModel(n_features=n_features, d_model=64, dropout=0.2)
    elif model_type == 'statistics_only':
        model = StatisticsOnlyModel(n_features=n_features, hidden_dim=64, dropout=0.2)
    elif model_type == 'dual_stream':
        model = DualStreamAttention(time_steps=T, n_features=n_features, dim=64, dropout=0.2)
    elif model_type == 'accumulation_model':
        model = AccumulationAttentionModel(n_features=n_features, time_steps=T, d_model=32)
    elif model_type == 'simple_accumulation':
        model = SimpleAccumulationModel(n_features=n_features, d_model=16)
    else:  # simple_attention
        time_input_dim = 2 * n_features
        model = SimpleAttentionModel(time_input_dim=time_input_dim, d_model=64, dropout=0.2)

    model = model.to(device)
    print(f"\n创建了 {model_type} 模型: {model.__class__.__name__}")

    # 训练模型
    print(f"\n训练模型...")
    # 创建临时输出目录用于保存模型
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        train_model(
            model, train_dataset, test_dataset, epochs=50, device=device,
            output_dir=temp_dir, lr=1e-3, weight_decay=1e-4, patience=10
        )

    # 测试预测
    print(f"\n在测试集上进行预测...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for xb, yb in DataLoader(test_dataset, batch_size=32, shuffle=False):
            xb = xb.to(device)
            if model_type == 'dual_stream':
                pred, _, _ = model(xb)
            else:
                pred, _ = model(xb)
            pred_binary = (pred > 0.5).cpu().numpy().flatten()
            predictions.extend(pred_binary)
            true_labels.extend(yb.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    baseline_acc = max(np.mean(true_labels), 1 - np.mean(true_labels))
    print("\n预测结果：")
    print(f"  测试准确率: {accuracy:.4f}")
    print(f"  基线准确率: {baseline_acc:.4f}")
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    print("\n混淆矩阵：")
    print(f"  预测左|实际左: {cm[0,0]:3d}  | 预测右|实际左: {cm[0,1]:3d}")
    print(f"  预测左|实际右: {cm[1,0]:3d}  | 预测右|实际右: {cm[1,1]:3d}")

    print("\n模型成功完成了二分类任务！")
    print(f"它根据音频的时间序列特征预测用户会选择左耳还是右耳音频。")


def run_attention_models_for_subject(subject_id, consistent_trials, pair_names, base_dir, feature_configs, output_root):
    """针对单个被试运行全部注意力模型并保存结果"""
    print(f"\n{'=' * 60}")
    print(f"开始评估被试: {subject_id}")
    print(f"{'=' * 60}")

    # 构建时间序列特征
    X, y_choice = build_time_series_features_for_trials(consistent_trials, pair_names, base_dir, feature_configs)

    if len(X) == 0:
        print(f"  警告: {subject_id} 未能构建任何特征，跳过")
        return []

    feature_names = [f"{feat_type}_{'std' if compute_std else 'mean'}" for feat_type, compute_std in feature_configs]

    # 定义要测试的注意力模型类型
    attention_model_types = ['conv_pool_only', 'conv_pool_attention', 'statistics_only', 'simple_attention', 'dual_stream']

    all_results = []

    for model_type in attention_model_types:
        print(f"\n--- [{subject_id}] 评估注意力模型: {model_type} ---")

        try:
            # 使用注意力模型进行二分类预测
            result = fit_attention_choice_model(X, y_choice, feature_names, model_type, model_type)

            if result is not None:
                result['model_name'] = model_type
                result['subject_id'] = subject_id
                all_results.append(result)
            else:
                all_results.append(None)

        except Exception as e:
            print(f"  错误: 处理模型 {model_type} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)

    # 保存该被试的结果
    output_dir = os.path.join(output_root, subject_id)
    os.makedirs(output_dir, exist_ok=True)

    plot_model_comparison(all_results, output_dir)
    save_results(all_results, consistent_trials, output_dir, subject_id=subject_id)

    valid_results = [r for r in all_results if r is not None]

    print(f"\n{'=' * 60}")
    print(f"{subject_id} 模型对比总结")
    print(f"{'=' * 60}")
    if valid_results:
        valid_results.sort(key=lambda x: x.get('cv_accuracy_mean', 0), reverse=True)
        print(f"\n成功拟合 {len(valid_results)} 个模型")
        top_k = min(3, len(valid_results))
        print(f"\n前{top_k}名模型（按CV准确率）:")
        for i, result in enumerate(valid_results[:top_k], 1):
            print(f"\n{i}. {result['model_name']}")
            print(f"   准确率: {result['accuracy']:.4f}")
            print(f"   CV准确率: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}")
            print(f"   AUC: {result['auc']:.4f}")
    else:
        print("没有成功拟合的模型")

    print(f"\n结果已保存到: {output_dir}")

    return valid_results


def main():
    """主函数"""
    print("="*60)
    print("使用注意力模型进行个人数据模型对比")
    print("测试所有9个被试，生成汇总对比CSV")
    print("="*60)

    project_root = os.getcwd()
    output_root = os.path.join(project_root, 'attention_individual_comparison_results')
    os.makedirs(output_root, exist_ok=True)

    base_dir = DEFAULT_AUDIO_BASE_DIR

    # 加载pair_names（需要知道所有288个音频对）
    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)

    # 获取特征配置
    feature_configs = get_feature_configs_from_linear_regression()
    print(f"\n使用{len(feature_configs)}个特征:")
    for i, (feat_type, compute_std) in enumerate(feature_configs):
        print(f"  {i+1}. {feat_type} ({'std' if compute_std else 'mean'})")

    # 收集所有被试的结果
    overall_summary_rows = []
    processed_subjects = 0

    # 循环处理所有被试
    for subject_config in SUBJECT_CONFIGS:
        subject_id = subject_config['subject_id']
        trials = load_subject_trials(subject_config)

        if len(trials) == 0:
            print(f"\n跳过 {subject_id}: 未能加载任何trial数据")
            continue

        # 检查左右平衡一致性
        consistent_trials = check_left_right_consistency(trials, pair_names)
        if len(consistent_trials) == 0:
            print(f"\n跳过 {subject_id}: 没有满足左右一致性的试次")
            continue

        if len(consistent_trials) < MIN_CONSISTENT_TRIALS_WARNING:
            print(f"\n警告: {subject_id} 的一致试次仅 {len(consistent_trials)} (<{MIN_CONSISTENT_TRIALS_WARNING})，结果可靠性可能较低（仍继续）")

        # 为每个trial添加subject_id
        for trial in consistent_trials:
            trial['subject_id'] = subject_id

        # 运行该被试的所有注意力模型
        subject_results = run_attention_models_for_subject(
            subject_id,
            consistent_trials,
            pair_names,
            base_dir,
            feature_configs,
            output_root
        )

        processed_subjects += 1

        # 收集结果用于汇总
        for result in subject_results:
            overall_summary_rows.append({
                'subject_id': subject_id,
                'model_name': result.get('model_name', 'Unknown'),
                'model_type': result.get('model_type', 'Unknown'),
                'accuracy': result.get('accuracy', np.nan),
                'baseline_accuracy': result.get('baseline_accuracy', np.nan),
                'improvement': result.get('accuracy', np.nan) - result.get('baseline_accuracy', np.nan),
                'auc': result.get('auc', np.nan),
                'cv_accuracy_mean': result.get('cv_accuracy_mean', np.nan),
                'cv_accuracy_std': result.get('cv_accuracy_std', np.nan),
                'n_features': len(result.get('feature_names', [])),
                'n_samples': result.get('n_samples', len(result.get('y_true', [])))
            })

    # 生成汇总CSV
    if overall_summary_rows:
        summary_df = pd.DataFrame(overall_summary_rows)
        summary_df = summary_df.sort_values(['subject_id', 'cv_accuracy_mean'], ascending=[True, False])
        summary_path = os.path.join(output_root, 'all_subjects_attention_model_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n{'='*60}")
        print(f"所有被试注意力模型对比汇总已保存: {summary_path}")
        print(f"{'='*60}")
        print(f"\n汇总统计:")
        print(f"  共处理 {processed_subjects}/{len(SUBJECT_CONFIGS)} 个被试")
        print(f"  共生成 {len(overall_summary_rows)} 条模型结果记录")
        print(f"\n按模型类型统计平均CV准确率:")
        model_summary = summary_df.groupby('model_type')['cv_accuracy_mean'].agg(['mean', 'std', 'count'])
        print(model_summary)
    else:
        print("\n未生成任何被试的模型结果")

    # 显示缓存统计
    print("\n=== 缓存统计 ===")
    print(f"时间序列缓存:")
    print(f"  命中: {_cache_stats['pair_feature_hits']} 次")
    print(f"  未命中: {_cache_stats['pair_feature_misses']} 次")
    if _cache_stats['pair_feature_hits'] + _cache_stats['pair_feature_misses'] > 0:
        hit_rate = _cache_stats['pair_feature_hits'] / (_cache_stats['pair_feature_hits'] + _cache_stats['pair_feature_misses']) * 100
        print(f"  命中率: {hit_rate:.1f}%")
    print(f"std缓存:")
    print(f"  命中: {_cache_stats['std_cache_hits']} 次")
    print(f"  未命中: {_cache_stats['std_cache_misses']} 次")
    if _cache_stats['std_cache_hits'] + _cache_stats['std_cache_misses'] > 0:
        hit_rate = _cache_stats['std_cache_hits'] / (_cache_stats['std_cache_hits'] + _cache_stats['std_cache_misses']) * 100
        print(f"  命中率: {hit_rate:.1f}%")


if __name__ == '__main__':
    main()
