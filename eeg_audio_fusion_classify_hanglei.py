#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合EEG和音频特征的多模态分类模型
使用EEGNet提取的EEG特征（poolmean后160维）和音频全局特征（X_global）进行分类
支持选择是否筛选一致的trials

支持的被试：hanglei, yanxingzhuo, jinxiaoyue, chenxianwei, yeziyuan, zhangzhiyao, haoxiang

使用方法：
1. 修改脚本中的FILTER_CONSISTENT参数选择是否筛选一致trials
2. 确保数据路径正确：
   - EEG数据：A:/standard_data_interp_no_ica/{subject_name}
   - 音频映射文件：D:\\D\\research\\audioset下载\\audio_pairs_2s\\分组音频\\folder{1,2,3}
3. 运行脚本：python eeg_audio_fusion_classify_hanglei.py

输出：
- fusion_classification_results.csv: 5折交叉验证结果
- fusion_classification_summary.txt: 结果汇总
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
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report
# 不使用sklearn的KFold，使用与individual_choice_with_position.py一致的build_consistent_folds
from sklearn.preprocessing import StandardScaler
import warnings
import hashlib
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的函数
from eegnet_classify_trials_5s import (
    find_data_files, load_meta, reshape_trials, 
    apply_baseline_correction, select_time_segment, select_last_seconds,
    exponential_moving_standardize, DataMeta
)
from individual_choice_with_position import (
    check_left_right_consistency, build_global_stage_dataset,
    extract_features_for_attention_model, parse_pair_name
)
from linear_regression_selected_features import (
    load_or_extract_features, merge_balanced_pairs_by_type,
    build_consistent_folds
)
from linear_regression_loudness_models import (
    parse_stereo_pair_name, load_mono_5s, DEFAULT_AUDIO_BASE_DIR,
    parse_experiment_data_txt, parse_experiment_data_csv
)
from feature_stage_attention_model import (
    get_feature_configs_from_linear_regression, SR, HOP, DEFAULT_STAGE_SECONDS,
    IndependentEarComparisonModel
)
from attention_loudness_model import FeatureTimeSeriesDataset
from feature_stage_attention_model import StageFeatureDataset

# 导入同步性计算所需的库和函数
import librosa
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import hilbert, butter, filtfilt
from scipy import interpolate

# 数据配置
SAMPLING_RATE = 250

# 五个被试的完整配置
SUBJECTS_CONFIG = {
    'yanxingzhuo': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/yanxingzhuo',
        'mff_folders': [
            'A:/yanxingzhuo_1_20260109_053050.mff',
            'A:/yanxingzhuo_2_20260109_060019.mff',
            'A:/yanxingzhuo_3_20260109_062611.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-1-{session}',  # session: 1,2,3
        'missing_trials': None  # 无缺失
    },
    'hanglei': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/hanglei',
        'mff_folders': [
            'A:/hanglei_1_20251224_060149.mff',
            'A:/hanglei_2_20251224_063021.mff',
            'A:/hanglei_3_20251224_065517.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-1-{session}',
        'missing_trials': {'folder1': -1, 'folder2': -1}  # 缺最后一个trial
    },
    'jinxiaoyue': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/jinxiaoyue',
        'mff_folders': [
            'A:/jinxiaoyue_1_20260106_094715.mff',
            'A:/jinxiaoyue_2_20260106_101543.mff',
            'A:/jinxiaoyue_3_20260106_104519.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-3-{session}',
        'missing_trials': None
    },
    'chenxianwei': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/chenxianwei',
        'mff_folders': [
            'A:/chenxianwei0106_1_20260106_072500.mff',
            'A:/chenxianwei0106_2_20260106_074722.mff',
            'A:/chenxianwei0106_3_20260106_081120.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-2-{session}',
        'missing_trials': None
    },
    'yeziyuan': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/yeziyuan',
        'mff_folders': [
            'A:/yeziyuan_1_20260108_085950.mff',
            'A:/yeziyuan_2_20260108_093037.mff',
            'A:/yeziyuan_3_20260108_100149.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-1-{session}',
        'missing_trials': None
    },
    'zhangzhiyao': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/zhangzhiyao',
        'mff_folders': [
            'A:/zhangzhiyao_1_20260112_080120.mff',
            'A:/zhangzhiyao_2_20260112_082955.mff',
            'A:/zhangzhiyao_3_20260112_085704.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-1-{session}',
        'missing_trials': None
    },
    'haoxiang': {
        'processed_data_path': 'A:/standard_data_interp_no_ica/haoxiang',
        'mff_folders': [
            'A:/haoxiang_1_20260112_094523.mff',
            'A:/haoxiang_2_20260112_101025.mff',
            'A:/haoxiang_3_20260112_103619.mff'
        ],
        'benchmark_pattern': 'benchmark_1_10-2-{session}',
        'missing_trials': None
    }
}

# 音频映射文件（所有被试共享）
AUDIO_BASE_DIR = DEFAULT_AUDIO_BASE_DIR
MAPPING_BASE_DIR = r"D:\D\research\audioset下载\audio_pairs_2s\分组音频"
MAPPING_FILES = [
    os.path.join(MAPPING_BASE_DIR, "folder1", "file_mapping_folder1.csv"),
    os.path.join(MAPPING_BASE_DIR, "folder2", "file_mapping_folder2.csv"),
    os.path.join(MAPPING_BASE_DIR, "folder3", "file_mapping_folder3.csv"),
]

# 输出目录（改为多被试输出）
OUTPUT_DIR = 'A:/standard_data_interp_no_ica/nme_fusion_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交叉验证配置
CV_SEED = 42
N_FOLDS = 5

# 同步性计算配置
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
SAMPLING_RATE_AUDIO = 16000  # 音频采样率

# 模型训练配置
# 设置为True以训练同步性融合模型（注意：训练速度较慢，因为需要实时加载音频文件）
TRAIN_SYNC_FUSION = True


def parse_audio_mapping_custom(file_path):
    """
    解析音频映射文件（自定义版本，适配实际的CSV格式）
    返回映射字典：{映射后的名字: 原始名字}
    
    支持两种格式：
    1. 标准格式：new_name列和original_name列
    2. 自定义格式：第一列是原始名称，第二列是映射后的文件名
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        
        # 检查是否是标准格式（有new_name和original_name列）
        if 'new_name' in df.columns and 'original_name' in df.columns:
            for _, row in df.iterrows():
                new_name = str(row['new_name']).strip()
                original_name = str(row['original_name']).strip()
                
                if pd.isna(row['new_name']) or pd.isna(row['original_name']) or not new_name or not original_name:
                    continue
                
                mapping[new_name] = original_name
                if new_name.lower().endswith('.wav'):
                    mapping[new_name[:-4]] = original_name
        elif len(df.columns) >= 2:
            # 自定义格式：第一列是原始名称，第二列是映射后的文件名
            col1 = df.columns[0]  # 原始音频对名称
            col2 = df.columns[1]  # 映射后的文件名（如a001.wav）
            
            for idx, row in df.iterrows():
                original_name = str(row[col1]).strip()
                new_name = str(row[col2]).strip()
                
                if pd.isna(row[col1]) or pd.isna(row[col2]) or not original_name or not new_name:
                    continue
                
                # 存储映射：映射后的名字 -> 原始名字
                mapping[new_name] = original_name
                if new_name.lower().endswith('.wav'):
                    mapping[new_name[:-4]] = original_name
        
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_all_mappings():
    """加载所有三个folder的映射文件"""
    all_mappings = {}
    folder_ranges = [
        (0, 159, 0),    # folder1: 前159个trials
        (159, 318, 1),  # folder2: 159-318个trials
        (318, 478, 2),  # folder3: 318-478个trials (160个)
    ]
    
    mappings = []
    for mapping_file in MAPPING_FILES:
        if os.path.exists(mapping_file):
            mapping = parse_audio_mapping_custom(mapping_file)
            mappings.append(mapping)
            print(f"加载映射文件: {mapping_file}, 映射数量: {len(mapping) // 2 if any(k.endswith('.wav') for k in mapping) else len(mapping)}")
        else:
            print(f"警告: 映射文件不存在: {mapping_file}")
            mappings.append({})
    
    return mappings, folder_ranges


def get_mapping_for_trial(trial_idx, mappings, folder_ranges):
    """根据trial索引获取对应的映射文件"""
    for start_idx, end_idx, folder_idx in folder_ranges:
        if start_idx <= trial_idx < end_idx:
            return mappings[folder_idx]
    # 默认返回第一个映射
    return mappings[0] if mappings else {}


class EEGNetFeatureExtractor(nn.Module):
    """EEGNet特征提取器，提取poolmean后的特征"""
    def __init__(self, n_channels: int, n_samples: int):
        super(EEGNetFeatureExtractor, self).__init__()
        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, 20, (1, 41), stride=(1, 1), bias=False)
        
        # 独立成分分析
        self.conv_ica = nn.Conv2d(1, 8, (n_channels, 1), stride=(1, 1), bias=False)
        
        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5)
        # 使用自适应池化
        self.poolmean = nn.AdaptiveAvgPool2d((8, 1))  # 输出: (8, 1) 对应 (通道, 时间)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回poolmean后的特征（展平为160维）
        x: (batch, 1, n_channels, n_timepoints)
        返回: (batch, 160)
        """
        # 调试：打印输入形状
        # print(f"EEGNet input shape: {x.shape}")

        x = self.conv_ica(x)
        # print(f"After conv_ica: {x.shape}")

        x = F.dropout(x, 0.15, training=self.training)
        x = torch.permute(x, (0, 2, 1, 3))
        # print(f"After permute: {x.shape}")

        x = self.conv_time(x)
        # print(f"After conv_time: {x.shape}")
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)
        
        x = torch.mul(x, x)
        x = self.poolmean(x)  # 输出: (batch, 20, 8, 1)
        
        # 展平为160维特征向量
        x = x.view(x.size(0), -1)  # (batch, 160)
        return x


class AudioFeatureExtractor(nn.Module):
    """使用IndependentEarComparisonModel提取音频特征"""
    def __init__(self, ear_feature_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.audio_model = IndependentEarComparisonModel(ear_feature_dim, hidden_dim, dropout)
    
    def forward(self, left_feat, right_feat):
        """
        left_feat: (batch, ear_feature_dim)
        right_feat: (batch, ear_feature_dim)
        返回: (batch, hidden_dim) - 使用left_score和right_score拼接作为特征
        """
        pred, left_score, right_score = self.audio_model(left_feat, right_feat)
        # 将left_score和right_score拼接作为音频特征
        audio_feat = torch.stack([left_score, right_score], dim=1)  # (batch, 2)
        return audio_feat


class SimpleFusionClassifier(nn.Module):
    """简单拼接融合分类模型（端到端版本）"""
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super(SimpleFusionClassifier, self).__init__()

        # EEG特征提取器（可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 融合分类器
        total_dim = 160 + audio_dim  # EEG 160维 + 手工音频特征维度
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, eeg_data, audio_features):
        """
        eeg_data: (batch, 1, n_channels, n_timepoints)
        audio_features: (batch, audio_dim) - 手工音频特征
        返回: (batch, n_classes)
        """
        # 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 拼接并分类
        combined = torch.cat([eeg_feat, audio_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output


class AttentionFusionClassifier(nn.Module):
    """最轻量级注意力融合（端到端版本）"""
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super(AttentionFusionClassifier, self).__init__()

        # EEG特征提取器（可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 最轻量级注意力：直接计算权重
        self.attn_weight = nn.Linear(160 + audio_dim, 2)  # 直接计算2个权重

        self.classifier = nn.Sequential(
            nn.Linear(160 + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, eeg_data, audio_features):
        """
        最轻量级注意力融合
        """
        # 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 拼接特征
        combined = torch.cat([eeg_feat, audio_features], dim=1)  # (batch, 160+audio_dim)

        # 最轻量级注意力：直接计算权重
        attn_weights = F.softmax(self.attn_weight(combined), dim=1)  # (batch, 2)

        # 加权融合（直接在原始特征上）
        weighted_eeg = attn_weights[:, 0:1] * eeg_feat
        weighted_audio = attn_weights[:, 1:2] * audio_features
        fused = torch.cat([weighted_eeg, weighted_audio], dim=1)

        return self.classifier(fused)


class GatedFusionClassifier(nn.Module):
    """最轻量级门控融合（端到端版本）"""
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super(GatedFusionClassifier, self).__init__()

        # EEG特征提取器（可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 最轻量级门控：直接计算门控信号
        self.gate = nn.Linear(160 + audio_dim, 1)  # 单个门控值 (0-1)

        self.classifier = nn.Sequential(
            nn.Linear(160 + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, eeg_data, audio_features):
        """
        最轻量级门控融合
        """
        # 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 拼接特征
        combined = torch.cat([eeg_feat, audio_features], dim=1)  # (batch, 160+audio_dim)

        # 最轻量级门控：单个门控值
        gate = torch.sigmoid(self.gate(combined))  # (batch, 1)

        # 门控融合：控制整体特征的权重
        gated_combined = combined * gate
        return self.classifier(gated_combined)


class AudioOnlyClassifier(nn.Module):
    """只用音频特征的消融实验"""
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super(AudioOnlyClassifier, self).__init__()

        # 只用音频特征，不需要EEG特征提取器
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, eeg_data, audio_features):
        """
        只用音频特征，完全忽略EEG数据
        """
        return self.classifier(audio_features)


class EndToEndFusionClassifier(nn.Module):
    """端到端融合分类模型：EEG特征提取器 + 手工音频特征 + 分类器"""
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super(EndToEndFusionClassifier, self).__init__()

        # EEG特征提取器（可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 音频特征直接用手工特征，不需要额外提取器
        # 音频维度 = left_global_dim + right_global_dim

        # 融合分类器
        total_dim = 160 + audio_dim  # EEG 160维 + 手工音频特征维度
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, eeg_data, audio_features):
        """
        eeg_data: (batch, 1, n_channels, n_timepoints)
        audio_features: (batch, audio_dim) - 手工特征，left_global + right_global拼接
        返回: (batch, n_classes)
        """
        # 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 音频特征直接使用（已经预处理好）
        # audio_features已经是拼接好的手工特征

        # 拼接并分类
        combined = torch.cat([eeg_feat, audio_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output


class NMEFusionClassifier(nn.Module):
    """
    Neural Mixed Effects (NME) 融合分类器

    架构设计：
    - 固定效应：群体共享的EEG-音频融合模式
    - 随机效应：每个被试的个性化调整参数
    - NME正则化：基于协方差矩阵的正则化项
    """

    def __init__(self, eeg_channels, eeg_samples, audio_dim, n_subjects=5, hidden_dim=64, n_classes=2):
        super(NMEFusionClassifier, self).__init__()

        # ============ 固定效应：群体通用模式 ============
        # EEG特征提取器（群体共享）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 群体融合网络
        total_dim = 160 + audio_dim  # EEG 160维 + 音频特征维度
        self.fixed_effects_fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

        )

        # 群体分类器
        self.fixed_effects_classifier = nn.Linear(hidden_dim // 2, n_classes)

        # ============ 随机效应：标准LME形式（截距+斜率） ============
        # 每个被试的随机截距（标量）
        self.random_intercepts = nn.ParameterDict({
            f'subject_{i}': nn.Parameter(torch.zeros(1))
            for i in range(n_subjects)
        })

        # 每个被试的随机斜率（向量，维度=特征维度）
        self.random_slopes = nn.ParameterDict({
            f'subject_{i}': nn.Parameter(torch.zeros(hidden_dim // 2))
            for i in range(n_subjects)
        })

        # ============ NME正则化参数 ============
        # 随机截距的方差（标量）
        self. log_intercept_variance = nn.Parameter(torch.tensor(-2.0))

        # 随机斜率的方差（向量，每个特征维度一个方差）
        self.log_slope_variance = nn.Parameter(torch.ones(hidden_dim // 2) * (-2.0))

        # 用于跟踪更新协方差矩阵的样本收集器
        self.parameter_samples = []
        self.update_counter = 0

    def forward(self, eeg_data, audio_features, subject_ids, return_components=False):
        """
        前向传播

        Args:
            eeg_data: (batch, 1, n_channels, n_timepoints)
            audio_features: (batch, audio_dim)
            subject_ids: (batch,) 或 int - 受试者ID列表或单个ID
            return_components: bool - 是否返回中间组件用于分析

        Returns:
            logits: (batch, n_classes)
            如果return_components=True，还返回中间特征
        """
        batch_size = eeg_data.shape[0]

        # 处理subject_ids（可能是张量、列表或单个int）
        if isinstance(subject_ids, torch.Tensor):
            subject_ids = subject_ids.tolist()
        elif isinstance(subject_ids, int):
            subject_ids = [subject_ids] * batch_size

        # ============ 固定效应：群体通用处理 ============
        # 1. 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 2. 群体融合
        combined = torch.cat([eeg_feat, audio_features], dim=1)  # (batch, 160+audio_dim)
        fixed_features = self.fixed_effects_fusion(combined)  # (batch, hidden_dim//2)

        # ============ 随机效应：标准LME调整 ============
        # 3. 为每个样本应用对应被试的随机效应
        # 标准LME形式：adjusted = fixed + intercept + slopes * fixed
        adjusted_features = []

        for i in range(batch_size):
            subject_id = subject_ids[i]
            subject_key = f'subject_{subject_id}'

            # 获取该被试的随机效应参数
            intercept_i = self.random_intercepts[subject_key]  # (1,)
            slopes_i = self.random_slopes[subject_key]  # (hidden_dim//2,)

            # 标准LME调整：fixed + intercept + slopes * fixed
            h_i = fixed_features[i]  # (hidden_dim//2,)
            adjusted_h_i = h_i + intercept_i + slopes_i * h_i  # (hidden_dim//2,)

            adjusted_features.append(adjusted_h_i)

        adjusted_features = torch.stack(adjusted_features)  # (batch, hidden_dim//2)

        # ============ 最终分类 ============
        # 4. 群体分类器
        logits = self.fixed_effects_classifier(adjusted_features)  # (batch, n_classes)

        if return_components:
            return logits, {
                'eeg_features': eeg_feat,
                'fixed_features': fixed_features,
                'adjusted_features': adjusted_features,
                'random_intercepts': {k: v for k, v in self.random_intercepts.items()},
                'random_slopes': {k: v for k, v in self.random_slopes.items()}
            }

        return logits

    def get_nme_regularization_loss(self):
        """
        计算标准LME正则化损失

        按照标准混合效应模型：随机效应服从N(0, σ²)
        - 随机截距：N(0, σ_intercept²)
        - 随机斜率：N(0, σ_slope²)
        """
        reg_loss = 0.0

        # 计算方差（使用clamp确保数值稳定性）
        log_var_intercept_clamped = torch.clamp(self.log_intercept_variance, min=-10, max=10)
        var_intercept = torch.exp(log_var_intercept_clamped)  # 标量

        log_var_slope_clamped = torch.clamp(self.log_slope_variance, min=-10, max=10)
        var_slope = torch.exp(log_var_slope_clamped)  # (hidden_dim//2,)
        mean_var_slope = torch.mean(var_slope)  # 标量，用于正则化

        for subject_key in self.random_intercepts.keys():
            # 随机截距的正则化：intercept² / σ_intercept²
            intercept_i = self.random_intercepts[subject_key]
            reg_loss += (intercept_i ** 2) / (var_intercept + 1e-8)

            # 随机斜率的正则化：slopes² / σ_slope²
            slopes_i = self.random_slopes[subject_key]
            reg_loss += torch.sum(slopes_i ** 2) / (mean_var_slope + 1e-8)

        return reg_loss

    def collect_random_effects_sample(self):
        """
        收集当前随机效应参数作为样本（用于协方差估计）
        """
        for subject_key in self.random_intercepts.keys():
            intercept_i = self.random_intercepts[subject_key]
            slopes_i = self.random_slopes[subject_key]

            # 展平并存储
            sample = torch.cat([intercept_i.flatten(), slopes_i.flatten()])
            self.parameter_samples.append(sample.detach())

    def update_covariance_estimate(self):
        """
        更新协方差矩阵估计

        按照原始NME论文：在每个epoch后更新Σ
        这里简化为更新对角线方差
        """
        if len(self.parameter_samples) > 0:
            # 计算经验协方差
            all_params = torch.cat([
                param.flatten() for param in self.parameter_samples
            ])

            # 更新对角线方差（简化的协方差估计）
            # 计算标量方差，然后更新所有维度为相同的值
            if len(all_params) > 1:
                empirical_var = torch.var(all_params, unbiased=False)
            else:
                empirical_var = torch.tensor(0.01, device=all_params.device)

            # 更新所有维度为相同的值（简化版本）
            new_log_var = torch.log(empirical_var + 1e-6)

            # 记录更新前后的方差（用于打印）
            with torch.no_grad():
                old_log_var = self.log_random_effects_variance.data.mean().item()
                old_variance = torch.exp(torch.clamp(self.log_random_effects_variance, min=-10, max=10)).mean().item()

                # 使用fill_确保所有维度都被更新
                if isinstance(new_log_var, torch.Tensor):
                    new_log_var_value = new_log_var.item()
                    self.log_random_effects_variance.data.fill_(new_log_var_value)
                else:
                    new_log_var_value = float(new_log_var)
                    self.log_random_effects_variance.data.fill_(new_log_var_value)

                new_variance = torch.exp(torch.clamp(self.log_random_effects_variance, min=-10, max=10)).mean().item()

            # 打印协方差更新信息
            print(f"    协方差更新: 旧log_var={old_log_var:.4f} (variance={old_variance:.6f}) → "
                  f"新log_var={new_log_var_value:.4f} (variance={new_variance:.6f}), "
                  f"经验方差={empirical_var.item():.6f}")

            # 清空样本缓存
            self.parameter_samples = []


# ==================== 同步性特征提取相关函数 ====================

def extract_audio_envelope(audio_signal, sr_audio, sr_target, target_duration=None):
    """
    提取音频包络并降采样到目标采样率

    参数:
        audio_signal: 原始音频信号
        sr_audio: 音频采样率
        sr_target: 目标采样率（EEG采样率）
        target_duration: 目标时长（秒），如果为None则使用音频实际时长

    返回:
        包络信号（降采样到sr_target）
    """
    # 使用希尔伯特变换提取包络
    analytic_signal = hilbert(audio_signal)
    envelope = np.abs(analytic_signal)

    # 降采样到EEG采样率
    if target_duration is None:
        num_samples = int(len(envelope) * sr_target / sr_audio)
    else:
        num_samples = int(target_duration * sr_target)

    envelope_downsampled = signal.resample(envelope, num_samples)

    return envelope_downsampled


def extract_audio_rms(audio_signal, sr_audio, sr_target):
    """
    提取音频RMS能量并降采样到目标采样率

    参数:
        audio_signal: 原始音频信号
        sr_audio: 音频采样率
        sr_target: 目标采样率

    返回:
        RMS能量时间序列（降采样到sr_target）
    """
    # 计算RMS能量
    hop_length = 256
    rms = librosa.feature.rms(y=audio_signal, hop_length=hop_length)[0]

    # 插值到目标采样率
    t_original = np.linspace(0, len(audio_signal)/sr_audio, len(rms))
    num_samples_target = int(len(audio_signal) * sr_target / sr_audio)
    t_target = np.linspace(0, len(audio_signal)/sr_audio, num_samples_target)

    f = interpolate.interp1d(t_original, rms, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    rms_resampled = f(t_target)

    return rms_resampled


def bandpass_filter(data, fs, fmin, fmax, order=4):
    """带通滤波"""
    nyquist = fs / 2.0
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data


def extract_phase(data):
    """使用希尔伯特变换提取相位"""
    if data.ndim == 1:
        analytic_signal = hilbert(data)
    else:
        analytic_signal = hilbert(data, axis=-1)
    phase = np.angle(analytic_signal)
    return phase


def compute_plv(phase1, phase2):
    """
    计算相位锁定值 (Phase Locking Value, PLV)

    参数:
        phase1, phase2: 相位数组，形状为 (n_timepoints,)

    返回:
        PLV值，范围 [0, 1]
    """
    # 确保是1维数组
    if phase1.ndim > 1:
        phase1 = phase1.flatten()
    if phase2.ndim > 1:
        phase2 = phase2.flatten()

    # 确保长度一致
    min_len = min(len(phase1), len(phase2))
    if min_len < 3:
        return 0.0

    phase1 = phase1[:min_len]
    phase2 = phase2[:min_len]

    # 计算相位差
    phase_diff = phase1 - phase2

    # 计算PLV：|mean(exp(i*phase_diff))|
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def compute_eeg_psd_time_series(eeg_data, sfreq, fmin=0.5, fmax=45.0,
                                window_size=1.0, overlap=0.5):
    """
    计算EEG功率谱密度时间序列（所有通道的平均）

    参数:
        eeg_data: EEG数据 (n_channels, n_samples)
        sfreq: 采样率
        fmin, fmax: 频率范围
        window_size: 窗口大小（秒）
        overlap: 重叠比例

    返回:
        PSD时间序列
    """
    n_channels, n_samples = eeg_data.shape
    window_samples = int(window_size * sfreq)
    hop_samples = int(window_size * sfreq * (1 - overlap))

    psd_series = []

    for start_idx in range(0, n_samples - window_samples + 1, hop_samples):
        end_idx = start_idx + window_samples
        window_data = eeg_data[:, start_idx:end_idx]

        # 计算所有通道的平均PSD
        all_psds = []
        for ch_idx, ch_data in enumerate(window_data):
            # 检查通道数据是否全为0
            if np.allclose(ch_data, 0):
                all_psds.append(0.0)
                continue

            try:
                freqs, psd = signal.welch(ch_data, sfreq, nperseg=min(window_samples, len(ch_data)))
                # 选择指定频率范围
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                psd_filtered = psd[freq_mask]
                freqs_filtered = freqs[freq_mask]

                # 对频率维度积分，得到总功率
                if len(freqs_filtered) > 0:
                    total_power = np.trapz(psd_filtered, freqs_filtered)
                else:
                    total_power = 0.0
                all_psds.append(total_power)
            except Exception as e:
                all_psds.append(0.0)

        # 所有通道的平均功率
        mean_power = np.mean(all_psds) if len(all_psds) > 0 else 0.0
        psd_series.append(mean_power)

    return np.array(psd_series)


def compute_correlation_with_delay(series1, series2, fs, max_delay_ms=200):
    """
    计算考虑延迟的相关性

    参数:
        series1, series2: 两个时间序列
        fs: 采样率
        max_delay_ms: 最大延迟（毫秒）

    返回:
        (最佳相关系数, p值, 最佳延迟ms)
    """
    max_delay_samples = int(max_delay_ms * fs / 1000)

    best_corr = -np.inf
    best_pval = np.nan
    best_delay_ms = 0

    # 检查输入数据
    if len(series1) < 3 or len(series2) < 3:
        return best_corr, best_pval, best_delay_ms

    # 尝试不同的延迟
    for delay in range(-max_delay_samples, max_delay_samples + 1):
        if delay > 0:
            s1_shifted = series1[delay:]
            s2_aligned = series2[:-delay] if delay < len(series2) else series2
        elif delay < 0:
            s1_shifted = series1[:delay] if -delay < len(series1) else series1
            s2_aligned = series2[-delay:]
        else:
            s1_shifted = series1
            s2_aligned = series2

        # 确保长度一致
        min_len = min(len(s1_shifted), len(s2_aligned))
        if min_len < 3:
            continue

        s1_final = s1_shifted[:min_len]
        s2_final = s2_aligned[:min_len]

        # 去除NaN和Inf
        mask = np.isfinite(s1_final) & np.isfinite(s2_final)
        valid_count = np.sum(mask)
        if valid_count < 3:
            continue

        s1_clean = s1_final[mask]
        s2_clean = s2_final[mask]

        # 检查数据方差（如果方差为0，相关性无法计算）
        std1 = np.std(s1_clean)
        std2 = np.std(s2_clean)
        if std1 == 0 or std2 == 0:
            continue

        # 计算相关性
        try:
            corr, pval = pearsonr(s1_clean, s2_clean)
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_pval = pval
                best_delay_ms = delay / fs * 1000
        except Exception:
            continue

    return best_corr, best_pval, best_delay_ms


def load_trial_audio_pair(wavfile, base_dir):
    """
    加载trial对应的左右音频

    参数:
        wavfile: 音频对文件名（如 "main_cat1_id1+main_cat2_id2.wav"）
        base_dir: 音频文件基础目录

    返回:
        (左音频信号, 右音频信号, 采样率)
    """
    try:
        # 解析音频对名称，得到类别和文件ID
        left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(wavfile, base_dir)

        # 构建路径：base_dir/类别名/文件ID.wav
        left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
        right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            return None, None, None

        yL, sr = load_mono_5s(left_path)
        yR, _ = load_mono_5s(right_path)

        return yL, yR, sr
    except Exception:
        return None, None, None


class SynchronizationFeatureExtractor:
    """
    同步性特征提取器
    计算EEG与左右音频的同步性特征向量（每个6维：相关性+5个频段PLV）
    """

    def __init__(self, eeg_sampling_rate=250, audio_sampling_rate=16000):
        self.eeg_sampling_rate = eeg_sampling_rate
        self.audio_sampling_rate = audio_sampling_rate
        self.freq_bands = FREQ_BANDS

    def extract_sync_features(self, eeg_data, pair_name, audio_base_dir, verbose=False):
        """
        提取同步性特征

        参数:
            eeg_data: EEG数据 (n_channels, n_samples)
            pair_name: 音频对名称
            audio_base_dir: 音频文件基础目录
            verbose: 是否输出详细信息

        返回:
            (left_sync_vector, right_sync_vector) - 每个都是6维向量
        """
        # 默认返回零向量
        zero_vector = np.zeros(6, dtype=np.float32)

        # 加载音频
        yL, yR, sr = load_trial_audio_pair(pair_name, audio_base_dir)
        if yL is None or yR is None:
            if verbose:
                print(f"      警告: 无法加载音频文件 {pair_name}")
            return zero_vector, zero_vector

        try:
            # 计算音频时长（对应EEG的时间窗口）
            audio_duration = eeg_data.shape[1] / self.eeg_sampling_rate

            # 提取音频包络和RMS
            envelope_L = extract_audio_envelope(yL, sr, self.eeg_sampling_rate, target_duration=audio_duration)
            envelope_R = extract_audio_envelope(yR, sr, self.eeg_sampling_rate, target_duration=audio_duration)
            rms_L = extract_audio_rms(yL, sr, self.eeg_sampling_rate)
            rms_R = extract_audio_rms(yR, sr, self.eeg_sampling_rate)

            # 调整RMS长度以匹配包络
            if len(rms_L) != len(envelope_L):
                t_original = np.linspace(0, len(rms_L)/self.eeg_sampling_rate, len(rms_L))
                t_target = np.linspace(0, audio_duration, len(envelope_L))
                f_L = interpolate.interp1d(t_original, rms_L, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
                f_R = interpolate.interp1d(t_original, rms_R, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
                rms_L = f_L(t_target)
                rms_R = f_R(t_target)

            # 计算EEG PSD时间序列
            eeg_psd = compute_eeg_psd_time_series(eeg_data, self.eeg_sampling_rate)

            # 对齐长度
            min_len = min(len(eeg_psd), len(rms_L))
            if min_len < 3:
                return zero_vector, zero_vector

            eeg_psd_aligned = eeg_psd[:min_len]
            rms_L_aligned = rms_L[:min_len]
            rms_R_aligned = rms_R[:min_len]

            # 计算相关性
            corr_left, _, _ = compute_correlation_with_delay(eeg_psd_aligned, rms_L_aligned, self.eeg_sampling_rate)
            corr_right, _, _ = compute_correlation_with_delay(eeg_psd_aligned, rms_R_aligned, self.eeg_sampling_rate)

            # 如果相关性计算失败，设为0
            if not np.isfinite(corr_left):
                corr_left = 0.0
            if not np.isfinite(corr_right):
                corr_right = 0.0

            # 提取音频包络相位
            phase_envelope_L = extract_phase(envelope_L)
            phase_envelope_R = extract_phase(envelope_R)

            # 计算各频段的PLV
            left_plvs = []
            right_plvs = []

            for band_name, (fmin, fmax) in self.freq_bands.items():
                # EEG在该频段的相位
                eeg_filtered = bandpass_filter(eeg_data, self.eeg_sampling_rate, fmin, fmax)
                eeg_phase = extract_phase(eeg_filtered)  # (n_channels, n_samples)

                # 计算所有通道的平均相位
                eeg_phase_mean = np.mean(eeg_phase, axis=0)  # (n_samples,)

                # 对齐长度
                min_len_phase = min(len(eeg_phase_mean), len(phase_envelope_L))
                if min_len_phase < 3:
                    left_plvs.append(0.0)
                    right_plvs.append(0.0)
                    continue

                eeg_phase_aligned = eeg_phase_mean[:min_len_phase]
                phase_L_aligned = phase_envelope_L[:min_len_phase]
                phase_R_aligned = phase_envelope_R[:min_len_phase]

                # 计算PLV
                plv_left = compute_plv(eeg_phase_aligned, phase_L_aligned)
                plv_right = compute_plv(eeg_phase_aligned, phase_R_aligned)

                left_plvs.append(plv_left)
                right_plvs.append(plv_right)

            # 组合成6维向量：[相关性, plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma]
            left_sync_vector = np.array([corr_left] + left_plvs, dtype=np.float32)
            right_sync_vector = np.array([corr_right] + right_plvs, dtype=np.float32)

            return left_sync_vector, right_sync_vector

        except Exception as e:
            # 如果计算失败，返回零向量
            return zero_vector, zero_vector


def compute_sync_features_cache(X_eeg_raw, consistent_trials, eeg_sampling_rate=250,
                                audio_base_dir=None, output_dir=None):
    """
    预计算所有trial的同步性特征并缓存

    参数:
        X_eeg_raw: 原始EEG数据 (n_trials, n_channels, n_samples)
        consistent_trials: 包含pair_name信息的trial列表
        eeg_sampling_rate: EEG采样率
        audio_base_dir: 音频文件基础目录
        output_dir: 输出目录

    返回:
        sync_features: 同步性特征数组 (n_trials, 12) - 左6维+右6维
        cache_path: 缓存文件路径
    """
    if audio_base_dir is None:
        audio_base_dir = AUDIO_BASE_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # 创建缓存目录
    cache_dir = os.path.join(output_dir, 'sync_features_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # 生成缓存文件名（基于pair_names的hash）
    pair_names_str = ','.join(sorted([trial['pair_name'] for trial in consistent_trials]))
    cache_hash = hashlib.md5(pair_names_str.encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f'sync_features_{cache_hash}.npy')
    cache_info_path = os.path.join(cache_dir, f'sync_features_{cache_hash}_info.json')

    # 检查缓存是否存在
    if os.path.exists(cache_path) and os.path.exists(cache_info_path):
        print(f"\n=== 发现同步性特征缓存，从缓存加载 ===")
        try:
            sync_features = np.load(cache_path)
            with open(cache_info_path, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)

            print(f"  缓存文件: {cache_path}")
            print(f"  缓存信息: {cache_info['n_trials']} 个trials, 特征维度: {cache_info['feature_dim']}")
            print(f"  缓存时间: {cache_info['cache_time']}")

            # 验证缓存是否匹配当前数据
            if sync_features.shape[0] == len(consistent_trials) and sync_features.shape[1] == 12:
                print(f"  缓存验证通过，直接使用缓存")
                return sync_features, cache_path
            else:
                print(f"  警告: 缓存维度不匹配，重新计算")
        except Exception as e:
            print(f"  缓存文件损坏，重新计算: {e}")

    # 预计算同步性特征
    print(f"\n=== 预计算同步性特征（只需计算一次，之后会缓存） ===")
    print(f"  总trials数: {len(consistent_trials)}")
    print(f"  每个trial需要：加载音频文件 + 计算PSD + 计算相关性 + 计算5个频段PLV")
    print(f"  预计耗时：约 {len(consistent_trials) * 0.5:.1f} - {len(consistent_trials) * 2:.1f} 秒")

    sync_extractor = SynchronizationFeatureExtractor(
        eeg_sampling_rate=eeg_sampling_rate,
        audio_sampling_rate=SAMPLING_RATE_AUDIO
    )

    sync_features_list = []
    failed_count = 0

    for i, trial in enumerate(consistent_trials):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  进度: {i+1}/{len(consistent_trials)} ({100*(i+1)/len(consistent_trials):.1f}%)")

        # 获取对应的EEG数据
        eeg_sample = X_eeg_raw[i]  # (n_channels, n_samples)
        pair_name = trial['pair_name']

        # 计算同步性特征
        left_sync, right_sync = sync_extractor.extract_sync_features(
            eeg_sample, pair_name, audio_base_dir
        )

        # 检查是否计算成功（如果全为0可能是失败）
        if np.allclose(left_sync, 0) and np.allclose(right_sync, 0):
            failed_count += 1

        # 拼接左右同步性特征 (12维)
        sync_feat = np.concatenate([left_sync, right_sync])
        sync_features_list.append(sync_feat)

    # 转换为numpy数组
    sync_features = np.stack(sync_features_list, axis=0)  # (n_trials, 12)

    # 保存缓存
    np.save(cache_path, sync_features)

    # 保存缓存信息
    cache_info = {
        'n_trials': len(consistent_trials),
        'feature_dim': 12,
        'cache_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'failed_count': failed_count,
        'pair_names_hash': cache_hash
    }
    with open(cache_info_path, 'w', encoding='utf-8') as f:
        json.dump(cache_info, f, indent=2, ensure_ascii=False)

    print(f"\n  同步性特征预计算完成！")
    print(f"  成功: {len(consistent_trials) - failed_count}/{len(consistent_trials)}")
    if failed_count > 0:
        print(f"  失败: {failed_count} (返回零向量)")
    print(f"  缓存已保存: {cache_path}")
    print(f"  下次运行将直接使用缓存，无需重新计算")

    return sync_features, cache_path


class SyncFusionClassifier(nn.Module):
    """
    同步性融合分类模型：EEG特征 + 音频特征 + 同步性特征（从缓存加载）
    """
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2,
                 sync_features_cache=None):
        super(SyncFusionClassifier, self).__init__()

        # EEG特征提取器（可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 同步性特征缓存（预计算好的）
        if sync_features_cache is not None:
            self.register_buffer('sync_features_cache', torch.FloatTensor(sync_features_cache))
        else:
            self.sync_features_cache = None

        # 融合分类器：EEG(160) + 音频(audio_dim) + 同步性(12: 左6+右6)
        total_dim = 160 + audio_dim + 12
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, eeg_data, audio_features, indices):
        """
        eeg_data: (batch, 1, n_channels, n_timepoints)
        audio_features: (batch, audio_dim) - 手工音频特征
        indices: (batch,) - trial的索引列表，用于从缓存中获取同步性特征
        返回: (batch, n_classes)
        """
        batch_size = eeg_data.shape[0]

        # 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 从缓存加载同步性特征
        if self.sync_features_cache is not None:
            # indices是batch中每个样本对应的trial索引
            sync_features = self.sync_features_cache[indices].to(eeg_data.device)  # (batch, 12)
        else:
            # 如果没有缓存，返回零向量（不应该发生）
            sync_features = torch.zeros(batch_size, 12, device=eeg_data.device)

        # 拼接所有特征
        combined = torch.cat([eeg_feat, audio_features, sync_features], dim=1)  # (batch, 160+audio_dim+12)
        fused = self.fusion(combined)
        output = self.classifier(fused)

        return output


def build_benchmark_paths(subject_config):
    """构建某个被试的3个benchmark文件路径"""
    benchmark_files = []
    for session in [1, 2, 3]:
        mff_folder = subject_config['mff_folders'][session - 1]
        benchmark_name = subject_config['benchmark_pattern'].format(session=session)
        # 确保添加.txt扩展名
        if not benchmark_name.endswith('.txt'):
            benchmark_name = benchmark_name + '.txt'
        benchmark_path = os.path.join(mff_folder, benchmark_name)
        benchmark_files.append(benchmark_path)
    return benchmark_files


def load_single_subject_eeg(processed_data_path):
    """加载单个被试的EEG数据"""
    data_dir = Path(processed_data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {processed_data_path}")

    trials_path, labels_path, info_path = find_data_files(data_dir)

    # 加载元信息
    meta = load_meta(info_path, SAMPLING_RATE, None)

    # 加载数据
    trials = np.load(trials_path)
    labels_df = pd.read_csv(labels_path)

    # 获取标签
    if 'Label' in labels_df.columns:
        labels = labels_df['Label'].to_numpy()
    else:
        if labels_df.shape[1] >= 2:
            labels = labels_df.iloc[:, 1].to_numpy()
        else:
            labels = labels_df.iloc[:, 0].to_numpy()

    # 形状处理
    trials_3d = reshape_trials(trials, meta.num_channels)

    # 获取trial时长
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    trial_duration = info.get("trial_duration")
    if trial_duration is None:
        trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz

    # 截取完整trial数据
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)

    # 基线矫正：使用前3秒作为基线
    data_baseline_corrected = apply_baseline_correction(
        data_full,
        fs=meta.sampling_rate_hz,
        baseline_start=0.0,
        baseline_end=3.0
    )

    # 提取3-5秒用于特征提取（刺激期）
    if trial_duration < 5.0:
        raise ValueError(f"数据时长{trial_duration}秒不足以提取3-5秒的刺激期")

    stimulus_start_seconds = 3.0
    stimulus_duration = 2.0
    stimulus_start_idx = int(round(stimulus_start_seconds * meta.sampling_rate_hz))
    stimulus_end_idx = int(round((stimulus_start_seconds + stimulus_duration) * meta.sampling_rate_hz))
    data = data_baseline_corrected[:, :, stimulus_start_idx:stimulus_end_idx]

    # 指数移动标准化（与eegnet_classify_trials_5s.py保持一致）
    data = exponential_moving_standardize(
        data,
        alpha=0.01,  # 与eegnet_classify_trials_5s.py保持一致
        eps=1e-8,
        init_block_size=None  # 使用第一个时间点初始化
    )

    # 确保数据和标签数量匹配
    if data.shape[0] != labels.shape[0]:
        min_n = min(data.shape[0], labels.shape[0])
        data = data[:min_n]
        labels = labels[:min_n]

    return data, labels, meta


def standardize_channel_names(channel_names):
    """
    标准化通道名称格式，将不同格式统一为EEG001、EEG002等格式

    参数:
        channel_names: 通道名称列表

    返回:
        标准化的通道名称列表
    """
    standardized_names = []
    for name in channel_names:
        name_str = str(name).strip()
        # 如果是E1、E2等格式，转换为EEG001、EEG002等格式
        if name_str.startswith('E') and len(name_str) >= 2:
            try:
                # 提取数字部分
                num_part = name_str[1:]  # 去掉'E'
                # 如果是纯数字，转换为3位格式
                if num_part.isdigit():
                    num = int(num_part)
                    standardized_names.append(f'EEG{num:03d}')
                else:
                    # 如果不是纯数字，保持原样
                    standardized_names.append(name_str)
            except (ValueError, IndexError):
                standardized_names.append(name_str)
        # 如果已经是EEG001等格式，直接添加
        elif name_str.startswith('EEG') and len(name_str) >= 5:
            standardized_names.append(name_str)
        else:
            # 其他格式保持原样
            standardized_names.append(name_str)

    return standardized_names


def load_all_subjects_data():
    """
    加载所有5个被试的数据

    返回:
        all_eeg_data: (n_total_trials, n_channels_common, n_timepoints) numpy数组
        all_labels: (n_total_trials,) numpy数组
        all_subject_ids: (n_total_trials,) numpy数组，标识每个trial属于哪个被试
        all_trials: trial信息列表，包含pair_name等
        n_common_channels: 公共通道数
        meta_common: 统一的元信息
    """
    print("\n=== 加载所有被试的数据 ===")

    # 第一步：找到所有被试的通道名称和公共通道
    print("步骤1: 检查所有被试的通道信息...")
    all_channel_names = {}
    all_metas = {}
    all_channel_counts = []

    for subject_name, config in SUBJECTS_CONFIG.items():
        try:
            # 加载元信息
            data_dir = Path(config['processed_data_path'])
            if not data_dir.exists():
                print(f"  警告: 数据目录不存在: {data_dir}")
                continue
            
            trials_path, labels_path, info_path = find_data_files(data_dir)
            
            # 读取通道名称
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            
            channel_names = info.get('channels', [])
            if isinstance(channel_names, list) and len(channel_names) > 0:
                # 标准化通道名称格式
                channel_names = standardize_channel_names(channel_names)
                all_channel_names[subject_name] = channel_names
            else:
                # 如果没有通道名称，生成默认名称（需要先获取通道数）
                # 从info中获取通道数，避免重复加载数据
                num_channels = info.get('num_channels') or info.get('n_channels')
                if isinstance(num_channels, list):
                    num_channels = len(num_channels)
                if num_channels is None:
                    # 如果info中没有通道数，才加载数据
                    _, _, meta = load_single_subject_eeg(config['processed_data_path'])
                    num_channels = meta.num_channels
                all_channel_names[subject_name] = [f'EEG{i+1:03d}' for i in range(num_channels)]
            
            # 加载meta信息（用于获取通道数等）
            _, _, meta = load_single_subject_eeg(config['processed_data_path'])
            all_metas[subject_name] = meta
            all_channel_counts.append(meta.num_channels)
            print(f"  {subject_name}: {meta.num_channels} 通道")
            if len(all_channel_names[subject_name]) <= 10:
                print(f"    通道名称: {all_channel_names[subject_name]}")
            else:
                print(f"    前10个通道: {all_channel_names[subject_name][:10]}")
        except Exception as e:
            print(f"  警告: 无法加载 {subject_name} 的元信息: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_channel_counts) == 0:
        raise ValueError("无法加载任何被试的数据")

    # 找到公共通道（所有被试都有的通道）
    if len(all_channel_names) > 0:
        # 从第一个被试开始，逐步求交集
        common_channels = set(all_channel_names[list(all_channel_names.keys())[0]])
        for subject_name, ch_names in all_channel_names.items():
            common_channels = common_channels & set(ch_names)
            print(f"  {subject_name} 与公共通道的交集: {len(common_channels)} 个通道")
        
        common_channels = sorted(list(common_channels))  # 排序以保证顺序一致
        n_common_channels = len(common_channels)
        print(f"\n  公共通道数量: {n_common_channels}")
        if n_common_channels <= 20:
            print(f"  公共通道: {common_channels}")
        else:
            print(f"  前20个公共通道: {common_channels[:20]}")
            print(f"  后10个公共通道: {common_channels[-10:]}")
        
        if n_common_channels == 0:
            raise ValueError("未找到公共通道！所有被试的通道名称都不匹配。")
        
        # 检查每个被试是否有公共通道
        for subject_name, ch_names in all_channel_names.items():
            missing_channels = set(common_channels) - set(ch_names)
            if missing_channels:
                print(f"  警告: {subject_name} 缺少以下公共通道: {missing_channels}")
    else:
        # 如果没有通道名称信息，回退到最小通道数方法
        n_common_channels = min(all_channel_counts)
        common_channels = None
        print(f"\n  警告: 无法读取通道名称，使用最小通道数: {n_common_channels}")

    # 第二步：加载所有被试的数据并统一通道数
    print("\n步骤2: 加载所有被试的数据...")
    all_eeg_data_list = []
    all_labels_list = []
    all_subject_ids_list = []
    all_trials_list = []

    # 加载音频映射文件
    mappings, folder_ranges = load_all_mappings()

    # 创建subject_id映射
    subject_id_map = {name: idx for idx, name in enumerate(SUBJECTS_CONFIG.keys())}

    for subject_name, config in SUBJECTS_CONFIG.items():
        subject_id = subject_id_map[subject_name]
        print(f"\n  加载被试: {subject_name} (ID: {subject_id})")

        try:
            # 加载EEG数据
            eeg_data, labels, meta = load_single_subject_eeg(config['processed_data_path'])
            print(f"    原始EEG数据形状: {eeg_data.shape}")

            # 统一通道：使用公共通道（根据通道名称匹配）
            if common_channels is not None:
                # 根据通道名称匹配
                subject_channel_names = all_channel_names[subject_name]
                channel_indices = []
                missing_channels = []
                
                for ch_name in common_channels:
                    if ch_name in subject_channel_names:
                        channel_indices.append(subject_channel_names.index(ch_name))
                    else:
                        missing_channels.append(ch_name)
                
                if missing_channels:
                    print(f"      警告: {subject_name} 缺少以下公共通道: {missing_channels}")
                
                if len(channel_indices) == len(common_channels):
                    eeg_data = eeg_data[:, channel_indices, :]
                    print(f"    使用公共通道后EEG数据形状: {eeg_data.shape}")
                else:
                    raise ValueError(f"{subject_name} 无法匹配所有公共通道！")
            else:
                # 回退到最小通道数方法
                if eeg_data.shape[1] > n_common_channels:
                    eeg_data = eeg_data[:, :n_common_channels, :]
                    print(f"    统一后EEG数据形状: {eeg_data.shape}")

            # 加载trial信息
            benchmark_files = build_benchmark_paths(config)
            print(f"    Benchmark文件:")
            for bf in benchmark_files:
                print(f"      {bf}")

            # 解析trial信息
            parsed_trials_all_sessions = []
            for session_idx, benchmark_file in enumerate(benchmark_files):
                if not os.path.exists(benchmark_file):
                    print(f"      警告: benchmark文件不存在: {benchmark_file}")
                    continue

                parsed_trials = parse_experiment_data_txt(benchmark_file)
                print(f"      Session {session_idx + 1}: 解析了 {len(parsed_trials)} 个trials")

                # 处理hanglei缺失的trial
                if subject_name == 'hanglei' and config['missing_trials']:
                    if session_idx == 0:  # folder1
                        parsed_trials = parsed_trials[:-1]  # 去掉最后一个
                        print(f"        去掉folder1最后一个trial后: {len(parsed_trials)} 个trials")
                    elif session_idx == 1:  # folder2
                        parsed_trials = parsed_trials[:-1]  # 去掉最后一个
                        print(f"        去掉folder2最后一个trial后: {len(parsed_trials)} 个trials")

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
                        'pair_name': original_name  # 用于后续匹配
                    })

            print(f"    总共解析了 {len(parsed_trials_all_sessions)} 个trials")

            # 筛选一致的trials（需要pair_names，这里先保存，后面统一筛选）
            # 暂时保存所有trials，在main函数中统一筛选

            # 对齐EEG数据和trials
            # 注意：这里需要确保EEG数据的顺序和trials的顺序一致
            # 由于EEG数据是按trial顺序加载的，我们需要确保trials也是按顺序的
            n_trials = min(len(eeg_data), len(parsed_trials_all_sessions))
            if len(eeg_data) != len(parsed_trials_all_sessions):
                print(f"    警告: EEG数据({len(eeg_data)})和trials({len(parsed_trials_all_sessions)})数量不匹配，使用前{n_trials}个")

            eeg_data_aligned = eeg_data[:n_trials]
            labels_aligned = labels[:n_trials]
            trials_aligned = parsed_trials_all_sessions[:n_trials]

            # 添加subject_id
            subject_ids = np.full(n_trials, subject_id)

            # 累积数据
            all_eeg_data_list.append(eeg_data_aligned)
            all_labels_list.append(labels_aligned)
            all_subject_ids_list.append(subject_ids)
            all_trials_list.extend(trials_aligned)

            print(f"    成功加载: {n_trials} 个trials")

        except Exception as e:
            print(f"    错误: 加载 {subject_name} 的数据失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 合并所有数据
    if len(all_eeg_data_list) == 0:
        raise ValueError("未能加载任何被试的数据")

    all_eeg_data = np.concatenate(all_eeg_data_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    all_subject_ids = np.concatenate(all_subject_ids_list, axis=0)

    print(f"\n步骤3: 数据汇总")
    print(f"  总trials数: {len(all_eeg_data)}")
    print(f"  EEG数据形状: {all_eeg_data.shape}")
    print(f"  标签数量: {len(all_labels)}")
    print(f"  被试ID分布:")
    for subject_name, subject_id in subject_id_map.items():
        count = np.sum(all_subject_ids == subject_id)
        print(f"    {subject_name} (ID {subject_id}): {count} 个trials")

    # 创建统一的元信息
    meta_common = DataMeta(
        sampling_rate_hz=SAMPLING_RATE,
        num_channels=n_common_channels,
        samples_per_trial=all_eeg_data.shape[2]
    )

    return all_eeg_data, all_labels, all_subject_ids, all_trials_list, n_common_channels, meta_common


def load_eeg_data():
    """保留原函数用于向后兼容（已废弃，使用load_all_subjects_data）"""
    print("警告: load_eeg_data()已废弃，请使用load_all_subjects_data()")
    return load_all_subjects_data()


def load_hanglei_trials(benchmark_files=None, mappings=None, folder_ranges=None):
    """
    加载hanglei的trial信息（三个session分别对应三个folder）

    参数:
        benchmark_files: benchmark文件路径列表（3个文件）
        mappings: 三个folder的映射文件列表
        folder_ranges: folder范围列表

    返回:
        all_trials: 包含original_name和response的trial列表
    """
    print("\n=== 加载hanglei的trial信息 ===")
    all_trials = []

    if benchmark_files is None or len(benchmark_files) != 3:
        raise ValueError("需要提供3个benchmark文件路径")

    # 累计trial索引，用于确定使用哪个folder的映射
    cumulative_trial_idx = 0

    # 依次加载三个session的benchmark文件
    for session_idx, benchmark_file in enumerate(benchmark_files):
        if not os.path.exists(benchmark_file):
            print(f"警告: benchmark文件不存在: {benchmark_file}")
            continue

        print(f"\n加载session {session_idx + 1}: {benchmark_file}")
        parsed_trials = parse_experiment_data_txt(benchmark_file)
        print(f"  解析了 {len(parsed_trials)} 个trials")

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
                # 如果找不到映射，跳过这个trial
                continue

            all_trials.append({
                'original_name': original_name,
                'response': trial.get('response'),
                'reaction_time': trial.get('reaction_time'),
                'session': session_idx + 1  # 记录是哪个session
            })

            cumulative_trial_idx += 1

    print(f"\n总共加载了 {len(all_trials)} 个trials")
    print(f"  Session 1: {sum(1 for t in all_trials if t.get('session') == 1)} 个trials")
    print(f"  Session 2: {sum(1 for t in all_trials if t.get('session') == 2)} 个trials")
    print(f"  Session 3: {sum(1 for t in all_trials if t.get('session') == 3)} 个trials")

    return all_trials


def extract_eeg_features(eeg_data, meta, device='cpu'):
    """使用EEGNet提取EEG特征"""
    print("\n=== 提取EEG特征 ===")

    n_trials, n_channels, n_timepoints = eeg_data.shape

    # 创建特征提取器
    feature_extractor = EEGNetFeatureExtractor(n_channels, n_timepoints)
    feature_extractor.to(device)
    feature_extractor.eval()

    # 准备数据：添加batch和channel维度
    eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(1)  # (n_trials, 1, n_channels, n_timepoints)
    eeg_tensor = eeg_tensor.to(device)

    # 提取特征
    with torch.no_grad():
        eeg_features = feature_extractor(eeg_tensor)  # (n_trials, 160)

    eeg_features = eeg_features.cpu().numpy()
    print(f"EEG特征形状: {eeg_features.shape}")

    return eeg_features


def extract_audio_features(consistent_trials, project_root):
    """
    使用IndependentEarComparisonModel提取音频特征
    提取left_global和right_global特征，用于IndependentEarComparisonModel

    注意：只提取独特的音频对特征（480个），而不是为每个trial提取
    """
    print("\n=== 提取独特的音频对特征（使用IndependentEarComparisonModel） ===")

    # 获取所有独特的音频对名称
    unique_pair_names = list(set(trial['pair_name'] for trial in consistent_trials))
    unique_pair_names.sort()  # 确保顺序一致
    print(f"发现 {len(unique_pair_names)} 个独特的音频对")

    # 获取特征配置
    feature_configs = get_feature_configs_from_linear_regression()
    print(f"使用 {len(feature_configs)} 个特征配置")

    # 为独特的音频对提取特征
    pair_names_to_extract = unique_pair_names
    print(f"需要提取特征的音频对数量: {len(pair_names_to_extract)}")

    # 构建FeatureTimeSeriesDataset来提取特征
    dummy_targets = np.zeros(len(pair_names_to_extract), dtype=np.float32)
    print("正在构建FeatureTimeSeriesDataset并提取特征...")
    base_dataset = FeatureTimeSeriesDataset(
        pair_names_to_extract,
        AUDIO_BASE_DIR,
        dummy_targets,
        feature_configs,
        sr=SR,
        hop_length=HOP,
        window_size=5,
        use_high_res=False,
    )

    if len(base_dataset) == 0:
        raise RuntimeError("无法构建FeatureTimeSeriesDataset，可能音频文件不存在")

    print(f"成功构建FeatureTimeSeriesDataset，包含 {len(base_dataset)} 个样本")

    # 构建StageFeatureDataset来计算全局特征
    stage_dataset = StageFeatureDataset(
        base_dataset,
        stage_seconds=DEFAULT_STAGE_SECONDS,
        feature_configs=feature_configs,
    )
    stage_dataset.pair_names = pair_names_to_extract

    print(f"阶段特征数据集大小: {len(stage_dataset)}")
    print(f"  阶段数量: {stage_dataset.n_stages}")
    print(f"  特征维度: {stage_dataset.feat_dim}")
    print(f"  左耳特征维度: {stage_dataset.left_globals.shape[1] if len(stage_dataset.left_globals) > 0 else 'N/A'}")

    # 提取独特音频对的特征
    print(f"成功提取 {len(unique_pair_names)} 个独特音频对的特征:")
    print(f"  左耳全局特征形状: {stage_dataset.left_globals.shape}")
    print(f"  右耳全局特征形状: {stage_dataset.right_globals.shape}")

    # 创建pair_name到特征索引的映射
    pair_to_feature_idx = {pair_name: idx for idx, pair_name in enumerate(unique_pair_names)}

    # 保存映射以便后续使用
    feature_mapping = {
        'pair_names': unique_pair_names,
        'pair_to_idx': pair_to_feature_idx
    }

    return stage_dataset.left_globals, stage_dataset.right_globals, feature_mapping


def map_audio_features_to_trials(consistent_trials, left_globals, right_globals, feature_mapping):
    """
    根据trial的pair_name将音频特征映射到每个trial

    参数:
        consistent_trials: trial列表
        left_globals: (n_unique_pairs, feat_dim) 独特音频对的左耳特征
        right_globals: (n_unique_pairs, feat_dim) 独特音频对的右耳特征
        feature_mapping: 包含pair_name到索引映射的字典

    返回:
        left_features: (n_trials, feat_dim) 每个trial对应的左耳特征
        right_features: (n_trials, feat_dim) 每个trial对应的右耳特征
        valid_indices: 成功映射的trial索引
    """
    print(f"\n=== 将音频特征映射到trials ===")

    left_features = []
    right_features = []
    valid_indices = []
    pair_to_idx = feature_mapping['pair_to_idx']

    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']

        if pair_name in pair_to_idx:
            idx = pair_to_idx[pair_name]
            left_features.append(left_globals[idx])
            right_features.append(right_globals[idx])
            valid_indices.append(i)
        else:
            print(f"警告: trial {i} 的音频对 '{pair_name}' 找不到对应特征")

    left_features = np.array(left_features)
    right_features = np.array(right_features)

    print(f"成功映射 {len(valid_indices)}/{len(consistent_trials)} 个trials的音频特征")
    print(f"  左耳特征形状: {left_features.shape}")
    print(f"  右耳特征形状: {right_features.shape}")

    return left_features, right_features, valid_indices


def filter_consistent_trials(all_trials, pair_names):
    """
    筛选两次选择都一致的trials
    需要确保pair_names包含所有288个音频对（包括AB和BA版本）
    """
    print("\n=== 筛选一致的trials ===")

    # 调试信息：检查all_trials中的original_name格式
    if len(all_trials) > 0:
        print(f"  前5个trial的original_name示例:")
        for i, trial in enumerate(all_trials[:5]):
            print(f"    {i+1}. {trial.get('original_name', 'N/A')}")

    # 调试信息：检查pair_names
    print(f"  pair_names数量: {len(pair_names)}")
    if len(pair_names) > 0:
        print(f"  前5个pair_names示例:")
        for i, name in enumerate(pair_names[:5]):
            print(f"    {i+1}. {name}")

    # 检查all_trials中的original_name是否在pair_names中
    trial_names_set = set(t.get('original_name') for t in all_trials)
    pair_names_set = set(pair_names)
    matched = trial_names_set & pair_names_set
    print(f"  all_trials中的original_name数量: {len(trial_names_set)}")
    print(f"  pair_names数量: {len(pair_names_set)}")
    print(f"  匹配的数量: {len(matched)}")
    if len(matched) < len(trial_names_set):
        unmatched = trial_names_set - pair_names_set
        print(f"  未匹配的original_name示例（前10个）:")
        for i, name in enumerate(list(unmatched)[:10]):
            print(f"    {i+1}. {name}")

    consistent_trials = check_left_right_consistency(all_trials, pair_names)
    return consistent_trials


def build_balanced_trials(consistent_trials, pair_names):
    """
    构建平衡的标签数据集
    对于每个一致的trial，我们知道选择了哪个音频（A或B）
    通过选择保留AB版本还是BA版本来平衡标签，但不能改变实际选择

    策略：
    - 如果选择了A：可以用AB版本（choice=1）或BA版本（choice=2）
    - 如果选择了B：可以用AB版本（choice=2）或BA版本（choice=1）
    - 目标是让choice=1和choice=2的数量平衡
    """
    print(f"\n=== 构建平衡标签数据集 ===")
    print(f"  原始一致的trials数量: {len(consistent_trials)}")

    # 首先构建规范化键到AB和BA版本pair_name的映射
    normalized_to_pairs = {}

    # 按规范化键分组所有pair_names
    for pair_name in pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue

        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))

        if key not in normalized_to_pairs:
            normalized_to_pairs[key] = {'ab': None, 'ba': None}

        # 判断是AB还是BA版本
        left_audio = f"{left_cat}_{left_id}"
        right_audio = f"{right_cat}_{right_id}"
        audio_a_str, audio_b_str = key

        if left_audio == audio_a_str and right_audio == audio_b_str:
            normalized_to_pairs[key]['ab'] = pair_name
        elif left_audio == audio_b_str and right_audio == audio_a_str:
            normalized_to_pairs[key]['ba'] = pair_name

    # 对于每个一致的trial，确定它选择了哪个音频，以及应该用AB还是BA版本
    trials_select_a = []  # 选择了A的trials
    trials_select_b = []   # 选择了B的trials

    for trial in consistent_trials:
        pair_name = trial['pair_name']
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue

        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))

        if key not in normalized_to_pairs:
            continue

        ab_pair = normalized_to_pairs[key]['ab']
        ba_pair = normalized_to_pairs[key]['ba']

        if ab_pair is None or ba_pair is None:
            continue

        # 判断当前pair_name是AB还是BA版本
        current_left = f"{left_cat}_{left_id}"
        audio_a_str, audio_b_str = key
        is_ab_version = (current_left == audio_a_str)

        # 确定选择了哪个音频
        # 如果当前是AB版本：choice=1选A，choice=2选B
        # 如果当前是BA版本：choice=1选B，choice=2选A
        if is_ab_version:
            if trial['choice'] == 1:
                selected_audio = 'A'
            elif trial['choice'] == 2:
                selected_audio = 'B'
            else:
                continue
        else:  # BA版本
            if trial['choice'] == 1:
                selected_audio = 'B'
            elif trial['choice'] == 2:
                selected_audio = 'A'
            else:
                continue

        trial_info = {
            'original_trial': trial,
            'key': key,
            'ab_pair': ab_pair,
            'ba_pair': ba_pair,
            'selected_audio': selected_audio
        }

        if selected_audio == 'A':
            trials_select_a.append(trial_info)
        else:
            trials_select_b.append(trial_info)

    print(f"  选择A的trials数量: {len(trials_select_a)}")
    print(f"  选择B的trials数量: {len(trials_select_b)}")

    # 统计当前choice=1和choice=2的数量（如果都用AB版本）
    # 选择A的用AB版本→choice=1，选择B的用AB版本→choice=2
    current_choice1 = len(trials_select_a)
    current_choice2 = len(trials_select_b)

    print(f"  如果都用AB版本：choice=1有{current_choice1}个，choice=2有{current_choice2}个")

    # 目标：让choice=1和choice=2的数量平衡
    target_count = (len(trials_select_a) + len(trials_select_b)) // 2

    # 调整策略：
    # 如果choice=1太多，将一些选择A的trial改为BA版本（这样choice变成2）
    # 如果choice=2太多，将一些选择B的trial改为BA版本（这样choice变成1）

    balanced_trials = []

    if current_choice1 > current_choice2:
        # choice=1太多，需要将一些选择A的trial改为BA版本
        need_change = current_choice1 - target_count
        np.random.seed(CV_SEED)
        change_indices = np.random.choice(len(trials_select_a), need_change, replace=False)

        for i, trial_info in enumerate(trials_select_a):
            if i in change_indices:
                # 使用BA版本，选择A→choice=2（因为BA版本中A在右边）
                balanced_trials.append({
                    'pair_name': trial_info['ba_pair'],
                    'choice': 2,  # BA版本中选A（右边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })
            else:
                # 使用AB版本，选择A→choice=1
                balanced_trials.append({
                    'pair_name': trial_info['ab_pair'],
                    'choice': 1,  # AB版本中选A（左边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })

        # 选择B的trials都用AB版本，选择B→choice=2
        for trial_info in trials_select_b:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 2,  # AB版本中选B（右边）
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })

    elif current_choice2 > current_choice1:
        # choice=2太多，需要将一些选择B的trial改为BA版本
        need_change = current_choice2 - target_count
        np.random.seed(CV_SEED)
        change_indices = np.random.choice(len(trials_select_b), need_change, replace=False)

        # 选择A的trials都用AB版本，选择A→choice=1
        for trial_info in trials_select_a:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 1,  # AB版本中选A（左边）
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })

        for i, trial_info in enumerate(trials_select_b):
            if i in change_indices:
                # 使用BA版本，选择B→choice=1（因为BA版本中B在左边）
                balanced_trials.append({
                    'pair_name': trial_info['ba_pair'],
                    'choice': 1,  # BA版本中选B（左边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })
            else:
                # 使用AB版本，选择B→choice=2
                balanced_trials.append({
                    'pair_name': trial_info['ab_pair'],
                    'choice': 2,  # AB版本中选B（右边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })

    else:
        # 已经平衡，都用AB版本
        for trial_info in trials_select_a:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 1,
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })
        for trial_info in trials_select_b:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 2,
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })

    # 打乱顺序
    np.random.shuffle(balanced_trials)

    # 统计最终结果
    final_choice1 = sum(1 for t in balanced_trials if t['choice'] == 1)
    final_choice2 = sum(1 for t in balanced_trials if t['choice'] == 2)

    print(f"  平衡后的总trials数量: {len(balanced_trials)}")
    print(f"  choice=1: {final_choice1} ({final_choice1/len(balanced_trials)*100:.1f}%)")
    print(f"  choice=2: {final_choice2} ({final_choice2/len(balanced_trials)*100:.1f}%)")

    return balanced_trials


def build_pair_consistent_folds(consistent_trials, n_folds=5, seed=42):
    """
    构建确保同一音频对的AB/BA版本在同一fold的交叉验证划分
    
    参数:
        consistent_trials: 包含pair_name信息的trial列表
        n_folds: 折数
        seed: 随机种子
    
    返回:
        folds: list[np.ndarray]，每个元素是一个fold的trial索引
        shuffled_indices: np.ndarray，打乱后的全索引
    """
    print(f"\n=== 构建音频对一致的交叉验证划分（确保AB/BA在同一fold）===")
    print(f"  总trials数: {len(consistent_trials)}")
    
    # 1. 构建规范化键到trial索引的映射
    # 规范化键：tuple(sorted([audio_a, audio_b]))，用于标识同一个音频对
    pair_key_to_indices = {}
    
    for idx, trial in enumerate(consistent_trials):
        pair_name = trial.get('pair_name')
        if not pair_name:
            continue
        
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))
        
        if key not in pair_key_to_indices:
            pair_key_to_indices[key] = []
        pair_key_to_indices[key].append(idx)
    
    print(f"  唯一音频对数量: {len(pair_key_to_indices)}")
    
    # 2. 将音频对分组到不同的fold
    # 确保同一个音频对的所有trials（AB和BA版本）都在同一个fold
    pair_keys = list(pair_key_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(pair_keys)
    
    # 将音频对分配到不同的fold
    folds_dict = {i: [] for i in range(n_folds)}
    for fold_idx, key in enumerate(pair_keys):
        fold_id = fold_idx % n_folds
        folds_dict[fold_id].extend(pair_key_to_indices[key])
    
    # 3. 转换为numpy数组格式
    folds = []
    all_indices = []
    for fold_idx in range(n_folds):
        fold_indices = np.array(folds_dict[fold_idx], dtype=np.int32)
        folds.append(fold_indices)
        all_indices.extend(fold_indices)
    
    shuffled_indices = np.array(all_indices, dtype=np.int32)
    
    # 4. 验证：检查每个音频对的trials是否都在同一个fold
    verification_passed = True
    for key, indices in pair_key_to_indices.items():
        fold_ids = set()
        for idx in indices:
            for fold_idx, fold in enumerate(folds):
                if idx in fold:
                    fold_ids.add(fold_idx)
                    break
        if len(fold_ids) > 1:
            verification_passed = False
            print(f"  警告: 音频对 {key} 的trials分布在多个fold: {fold_ids}")
    
    if verification_passed:
        print(f"  验证通过: 所有音频对的AB/BA版本都在同一fold")
    
    # 打印每个fold的大小
    print(f"  各fold大小:")
    for fold_idx, fold in enumerate(folds):
        print(f"    Fold {fold_idx + 1}: {len(fold)} 个trials")
    
    return folds, shuffled_indices


def build_multi_subject_pair_consistent_folds(subject_consistent_trials, n_folds=5, seed=42):
    """
    为多个被试构建fold划分：
    1. 先为每个被试单独划分fold（确保AB/BA在同一fold）
    2. 然后合并所有人的相同fold（所有人的fold1合并成fold1，...）
    
    参数:
        subject_consistent_trials: dict，键为subject_name，值为该被试的consistent_trials列表
        n_folds: 折数
        seed: 随机种子
    
    返回:
        folds: list[np.ndarray]，每个元素是一个fold的全局trial索引
        shuffled_indices: np.ndarray，打乱后的全索引
        subject_trial_info: dict，包含每个被试的信息：
            - 'offset': 该被试的trial在全局索引中的起始偏移量
            - 'n_trials': 该被试的trial数量
            - 'subject_folds': 该被试的局部fold划分（用于验证）
    """
    print(f"\n=== 构建多被试音频对一致的交叉验证划分 ===")
    print(f"  被试数量: {len(subject_consistent_trials)}")
    print(f"  折数: {n_folds}")
    
    # 1. 为每个被试单独划分fold
    subject_folds_dict = {}  # 存储每个被试的fold划分
    subject_trial_info = {}  # 存储每个被试的偏移量和trial数量
    global_offset = 0
    
    for subject_name, consistent_trials in subject_consistent_trials.items():
        n_trials = len(consistent_trials)
        if n_trials == 0:
            print(f"  警告: 被试 {subject_name} 没有trials，跳过")
            continue
        
        print(f"\n  处理被试: {subject_name} ({n_trials} 个trials)")
        
        # 为该被试单独划分fold（确保AB/BA在同一fold）
        subject_folds, subject_shuffled_indices = build_pair_consistent_folds(
            consistent_trials, n_folds=n_folds, seed=seed
        )
        
        # 记录该被试的信息
        subject_trial_info[subject_name] = {
            'offset': global_offset,
            'n_trials': n_trials,
            'subject_folds': subject_folds
        }
        
        # 将局部索引转换为全局索引
        subject_global_folds = []
        for fold in subject_folds:
            global_fold = fold + global_offset
            subject_global_folds.append(global_fold)
        
        subject_folds_dict[subject_name] = subject_global_folds
        
        print(f"    全局索引偏移: {global_offset}")
        print(f"    各fold大小: {[len(fold) for fold in subject_folds]}")
        
        global_offset += n_trials
    
    total_trials = global_offset
    print(f"\n  总trials数: {total_trials}")
    
    # 2. 合并所有被试的相同fold
    print(f"\n  合并所有被试的相同fold...")
    merged_folds = {i: [] for i in range(n_folds)}
    
    for subject_name, global_folds in subject_folds_dict.items():
        for fold_idx in range(n_folds):
            if fold_idx < len(global_folds):
                merged_folds[fold_idx].extend(global_folds[fold_idx].tolist())
    
    # 3. 转换为numpy数组格式
    folds = []
    all_indices = []
    for fold_idx in range(n_folds):
        fold_indices = np.array(merged_folds[fold_idx], dtype=np.int32)
        folds.append(fold_indices)
        all_indices.extend(merged_folds[fold_idx])
    
    shuffled_indices = np.array(all_indices, dtype=np.int32)
    
    # 4. 验证：检查每个被试的AB/BA是否在同一fold
    print(f"\n  验证fold划分结果...")
    verification_passed = True
    
    for subject_name, info in subject_trial_info.items():
        subject_folds = info['subject_folds']
        offset = info['offset']
        
        # 检查该被试的每个音频对的AB/BA是否在同一fold
        consistent_trials = subject_consistent_trials[subject_name]
        pair_key_to_indices = {}
        
        for idx, trial in enumerate(consistent_trials):
            pair_name = trial.get('pair_name')
            if not pair_name:
                continue
            
            parse_result = parse_pair_name(pair_name)
            if parse_result is None:
                continue
            
            left_cat, left_id, right_cat, right_id, _ = parse_result
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            key = tuple(sorted([audio_a, audio_b]))
            
            if key not in pair_key_to_indices:
                pair_key_to_indices[key] = []
            pair_key_to_indices[key].append(idx)
        
        # 检查每个音频对的trials是否都在同一个fold
        for key, indices in pair_key_to_indices.items():
            fold_ids = set()
            for idx in indices:
                for fold_idx, fold in enumerate(subject_folds):
                    if idx in fold:
                        fold_ids.add(fold_idx)
                        break
            if len(fold_ids) > 1:
                verification_passed = False
                print(f"    警告: 被试 {subject_name} 的音频对 {key} 的trials分布在多个fold: {fold_ids}")
    
    if verification_passed:
        print(f"  验证通过: 所有被试的AB/BA版本都在同一fold")
    
    # 打印每个fold的大小
    print(f"\n  合并后各fold大小:")
    for fold_idx, fold in enumerate(folds):
        print(f"    Fold {fold_idx + 1}: {len(fold)} 个trials")
    
    # 打印每个fold中各被试的分布
    print(f"\n  各fold中各被试的trials分布:")
    for fold_idx, fold in enumerate(folds):
        print(f"    Fold {fold_idx + 1}:")
        for subject_name, info in subject_trial_info.items():
            offset = info['offset']
            n_trials = info['n_trials']
            # 计算该fold中属于该被试的trials数量
            subject_trials_in_fold = np.sum((fold >= offset) & (fold < offset + n_trials))
            if subject_trials_in_fold > 0:
                print(f"      {subject_name}: {subject_trials_in_fold} 个trials")
    
    return folds, shuffled_indices, subject_trial_info


def train_and_evaluate(X_eeg, X_left_audio, X_right_audio, y, n_folds=5, seed=42):
    """
    训练和评估四种融合模型：
    - simple: SimpleFusionClassifier（预提取特征）
    - attention: AttentionFusionClassifier（预提取特征）
    - gated: GatedFusionClassifier（预提取特征）
    - end_to_end: EndToEndFusionClassifier（端到端训练）
    """
    print("\n=== 训练和评估四种融合模型 ===")

    # 数据标准化
    scaler_eeg = StandardScaler()
    scaler_left_audio = StandardScaler()
    scaler_right_audio = StandardScaler()

    X_eeg_scaled = scaler_eeg.fit_transform(X_eeg)
    X_left_audio_scaled = scaler_left_audio.fit_transform(X_left_audio)
    X_right_audio_scaled = scaler_right_audio.fit_transform(X_right_audio)

    # 转换为tensor
    X_eeg_tensor = torch.FloatTensor(X_eeg_scaled)
    X_left_audio_tensor = torch.FloatTensor(X_left_audio_scaled)
    X_right_audio_tensor = torch.FloatTensor(X_right_audio_scaled)
    y_tensor = torch.LongTensor(y)

    # 使用与individual_choice_with_position.py完全一致的折划分方法
    folds, shuffled_indices = build_consistent_folds(len(y), seed=seed, k=n_folds)

    # 存储四种模型的结果
    all_fold_results = {
        'simple': [],
        'attention': [],
        'gated': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取音频特征维度
    ear_feature_dim = X_left_audio_tensor.shape[1]

    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")

        # 划分数据：与individual_choice_with_position.py保持一致
        train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)

        X_eeg_train = X_eeg_tensor[train_idx]
        X_left_audio_train = X_left_audio_tensor[train_idx]
        X_right_audio_train = X_right_audio_tensor[train_idx]
        y_train = y_tensor[train_idx]

        X_eeg_val = X_eeg_tensor[test_idx]
        X_left_audio_val = X_left_audio_tensor[test_idx]
        X_right_audio_val = X_right_audio_tensor[test_idx]
        y_val = y_tensor[test_idx]

        # 创建音频特征提取器（使用IndependentEarComparisonModel）
        audio_extractor = AudioFeatureExtractor(ear_feature_dim, hidden_dim=64, dropout=0.2)
        audio_extractor.to(device)
        audio_extractor.eval()  # 只用于特征提取，不训练

        # 提取音频特征（使用IndependentEarComparisonModel）
        with torch.no_grad():
            X_left_train_t = X_left_audio_train.to(device)
            X_right_train_t = X_right_audio_train.to(device)
            audio_feat_train = audio_extractor(X_left_train_t, X_right_train_t)  # (batch, 2)
            audio_feat_train = audio_feat_train.cpu()  # 移到CPU，避免梯度计算

            X_left_val_t = X_left_audio_val.to(device)
            X_right_val_t = X_right_audio_val.to(device)
            audio_feat_val = audio_extractor(X_left_val_t, X_right_val_t)  # (batch, 2)
            audio_feat_val = audio_feat_val.cpu()  # 移到CPU，避免梯度计算

        audio_dim = audio_feat_train.shape[1]  # 应该是2

        # 训练三种预提取特征的模型
        models = {
            'simple': SimpleFusionClassifier(eeg_dim=160, audio_dim=audio_dim, hidden_dim=64, n_classes=2),
            'attention': AttentionFusionClassifier(eeg_dim=160, audio_dim=audio_dim, hidden_dim=64, n_classes=2),
            'gated': GatedFusionClassifier(eeg_dim=160, audio_dim=audio_dim, hidden_dim=64, n_classes=2)
        }

        for model_name, model in models.items():
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

            n_epochs = 100
            best_val_acc = 0
            best_model_state = None  # 保存最佳模型状态
            patience = 15
            patience_counter = 0

            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()

                X_eeg_batch = X_eeg_train.to(device)
                audio_feat_batch = audio_feat_train.to(device)
                y_batch = y_train.to(device)

                outputs = model(X_eeg_batch, audio_feat_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                # 验证
                if (epoch + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        X_eeg_val_batch = X_eeg_val.to(device)
                        audio_feat_val_batch = audio_feat_val.to(device)
                        y_val_batch = y_val.to(device)

                        val_outputs = model(X_eeg_val_batch, audio_feat_val_batch)
                        _, val_pred = torch.max(val_outputs, 1)
                        val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model_state = model.state_dict().copy()  # 保存最佳模型状态
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            break

            # 最终评估
            if best_model_state is not None:
                model.load_state_dict(best_model_state)  # 加载最佳模型状态
            model.eval()
            with torch.no_grad():
                X_eeg_val_batch = X_eeg_val.to(device)
                audio_feat_val_batch = audio_feat_val.to(device)
                y_val_batch = y_val.to(device)

                val_outputs = model(X_eeg_val_batch, audio_feat_val_batch)
                val_probs = F.softmax(val_outputs, dim=1)
                _, val_pred = torch.max(val_outputs, 1)

                val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
                val_balanced_acc = balanced_accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
                val_auc = roc_auc_score(y_val_batch.cpu().numpy(), val_probs[:, 1].cpu().numpy())

            all_fold_results[model_name].append({
                'fold': fold_idx,
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_balanced_acc,
                'val_auc': val_auc,
                'actual_epochs': actual_epochs
            })

            print(f"  {model_name:12s} - 准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}, Epochs: {actual_epochs}")

    # 汇总结果
    all_results = {}
    for model_name, fold_results in all_fold_results.items():
        results_df = pd.DataFrame(fold_results)
        mean_acc = results_df['val_accuracy'].mean()
        mean_balanced_acc = results_df['val_balanced_accuracy'].mean()
        mean_auc = results_df['val_auc'].mean()

        all_results[model_name] = {
            'results_df': results_df,
            'mean_acc': mean_acc,
            'mean_balanced_acc': mean_balanced_acc,
            'mean_auc': mean_auc
        }

        print(f"\n=== {model_name.upper()} 交叉验证结果汇总 ===")
        print(f"平均验证准确率: {mean_acc:.4f} ± {results_df['val_accuracy'].std():.4f}")
        print(f"平均平衡准确率: {mean_balanced_acc:.4f} ± {results_df['val_balanced_accuracy'].std():.4f}")
        print(f"平均AUC: {mean_auc:.4f} ± {results_df['val_auc'].std():.4f}")

    return all_results


def build_stratified_folds_by_subject(y, subject_ids, n_folds=5, seed=42):
    """
    按被试分层的交叉验证划分

    确保每个被试的数据都分布在训练集和测试集中，避免某个被试的所有trials
    都在训练集或都在测试集的情况（这对NME模型很重要）

    参数:
        y: 标签数组
        subject_ids: 被试ID数组
        n_folds: 折数
        seed: 随机种子

    返回:
        folds: 每折的测试集索引列表
        shuffled_indices: 打乱后的全索引
    """
    from sklearn.model_selection import StratifiedKFold

    # 使用StratifiedKFold按被试ID分层划分
    # 这样可以确保每个被试的数据都分布在训练集和测试集中
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    all_indices = np.arange(len(y))

    for train_idx, test_idx in skf.split(all_indices, subject_ids):
        folds.append(test_idx)

    # 为了保持与build_consistent_folds的接口一致，也返回shuffled_indices
    shuffled_indices = np.random.RandomState(seed).permutation(len(y))

    return folds, shuffled_indices


def train_and_evaluate_all_models(X_eeg_raw, X_audio_manual, y, subject_ids,
                                 eeg_channels, eeg_samples, n_folds=5, seed=42,
                                 consistent_trials=None, use_stratified_cv=True):
    """
    训练和评估所有端到端融合模型：
    - simple: 简单拼接融合（端到端）
    - attention: 注意力融合（端到端）
    - gated: 门控融合（端到端）
    - end_to_end: 直接融合（端到端）
    - audio_only: 只用音频特征（消融实验）
    - sync_fusion: 同步性融合（新增，使用预计算缓存）
    - nme_fusion: NME融合（群体+个体）

    参数:
        X_eeg_raw: 原始EEG数据
        X_audio_manual: 手工音频特征
        y: 标签
        subject_ids: 被试ID数组（用于NME模型和分层交叉验证）
        eeg_channels: EEG通道数
        eeg_samples: EEG时间点数
        n_folds: 交叉验证折数
        seed: 随机种子
        consistent_trials: 包含pair_name信息的trial列表，用于同步性模型
        use_stratified_cv: 是否使用按被试分层的交叉验证（推荐True，特别是对于NME模型）
    """
    print("\n=== 训练和评估所有端到端融合模型 ===")

    # 检查被试数据分布
    unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    print(f"\n被试数据分布:")
    for subj_id, count in zip(unique_subjects, subject_counts):
        subject_name = list(SUBJECTS_CONFIG.keys())[subj_id]
        print(f"  {subject_name} (ID {subj_id}): {count} 个trials")

    # 选择交叉验证方法
    # 优先：如果提供了consistent_trials，使用build_pair_consistent_folds确保AB/BA在同一fold
    if consistent_trials is not None and len(consistent_trials) == len(y):
        print(f"\n使用音频对一致的交叉验证划分（确保AB/BA在同一fold）")
        folds, shuffled_indices = build_pair_consistent_folds(consistent_trials, n_folds=n_folds, seed=seed)
        
        # 验证分层效果（虽然不保证按被试分层，但可以显示分布）
        print(f"\n验证每折中每个被试的trials分布:")
        for fold_idx, test_idx in enumerate(folds, 1):
            train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)
            test_subjects = subject_ids[test_idx]
            train_subjects = subject_ids[train_idx]

            print(f"  Fold {fold_idx}:")
            for subj_id in unique_subjects:
                train_count = np.sum(train_subjects == subj_id)
                test_count = np.sum(test_subjects == subj_id)
                subject_name = list(SUBJECTS_CONFIG.keys())[subj_id]
                print(f"    {subject_name}: 训练集 {train_count}, 测试集 {test_count}")
    elif use_stratified_cv:
        print(f"\n使用按被试分层的交叉验证（推荐，特别是对于NME模型）")
        folds, shuffled_indices = build_stratified_folds_by_subject(y, subject_ids, n_folds=n_folds, seed=seed)

        # 验证分层效果
        print(f"\n验证分层效果（每折中每个被试的trials分布）:")
        for fold_idx, test_idx in enumerate(folds, 1):
            train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)
            test_subjects = subject_ids[test_idx]
            train_subjects = subject_ids[train_idx]

            print(f"  Fold {fold_idx}:")
            for subj_id in unique_subjects:
                train_count = np.sum(train_subjects == subj_id)
                test_count = np.sum(test_subjects == subj_id)
                subject_name = list(SUBJECTS_CONFIG.keys())[subj_id]
                print(f"    {subject_name}: 训练集 {train_count}, 测试集 {test_count}")
    else:
        print(f"\n使用标准交叉验证（注意：可能导致某些被试的数据分布不均）")
        folds, shuffled_indices = build_consistent_folds(len(y), seed=seed, k=n_folds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    all_fold_results = {
        'simple': [],
        'attention': [],
        'gated': [],
        'end_to_end': [],
        'audio_only': [],
        'nme_fusion': []
    }

    # 根据配置决定是否包含同步性融合模型
    sync_features_cache = None
    if TRAIN_SYNC_FUSION:
        all_fold_results['sync_fusion'] = []

        # 预计算同步性特征（只需计算一次，之后会缓存）
        if consistent_trials is not None:
            sync_features_cache, cache_path = compute_sync_features_cache(
                X_eeg_raw, consistent_trials,
                eeg_sampling_rate=SAMPLING_RATE,
                audio_base_dir=AUDIO_BASE_DIR,
                output_dir=OUTPUT_DIR
            )
            print(f"  同步性特征缓存已准备，维度: {sync_features_cache.shape}")

    audio_dim = X_audio_manual.shape[1]

    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")

        # 划分数据
        train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)

        X_eeg_train = X_eeg_raw[train_idx]
        X_audio_train = X_audio_manual[train_idx]
        y_train = y[train_idx]
        y_val = y[test_idx]

        print(f"  训练集大小: {len(y_train)}, 验证集大小: {len(y_val)}")

        # 创建模型（根据配置决定是否包含同步性融合模型和NME模型）
        # 动态计算被试数量
        n_subjects_actual = len(set(subject_ids))

        models = {
            # 'simple': SimpleFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            # 'attention': AttentionFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            # 'gated': GatedFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            # 'end_to_end': EndToEndFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            # 'audio_only': AudioOnlyClassifier(eeg_channels, eeg_samples, audio_dim),
            'nme_fusion': NMEFusionClassifier(eeg_channels, eeg_samples, audio_dim, n_subjects=n_subjects_actual),
        }

        print(f"  创建NME模型：支持 {n_subjects_actual} 个被试（subject_0 到 subject_{n_subjects_actual-1}）")

        # 根据配置决定是否添加同步性融合模型
        # if TRAIN_SYNC_FUSION and sync_features_cache is not None:
        #     models['sync_fusion'] = SyncFusionClassifier(
        #         eeg_channels, eeg_samples, audio_dim,
        #         sync_features_cache=sync_features_cache
        #     )
        #     print(f"  将训练 {len(models)} 个模型（包括同步性融合模型，使用预计算缓存）")
        # else:
        #     if TRAIN_SYNC_FUSION:
        #         print(f"  警告: 同步性融合模型需要consistent_trials参数，跳过")
        #     else:
        print(f"  将训练 {len(models)} 个模型（只训练NME模型）")

        for model_name, model in models.items():
            print(f"\n  训练模型: {model_name}")
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

            n_epochs = 100
            best_val_acc = 0
            best_model_state = None  # 保存最佳模型状态
            patience = 15
            patience_counter = 0

            # Batch处理配置
            batch_size = 16 if model_name in ['sync_fusion', 'nme_fusion'] else 32  # 同步性模型和NME模型用更小的batch
            n_train = len(y_train)
            n_batches = (n_train + batch_size - 1) // batch_size

            # NME特定参数
            lambda_reg = 0.1  # 正则化权重
            covariance_update_freq = 10  # 每10个epoch更新一次协方差

            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                # NME模型需要分别统计分类损失和正则化损失
                if model_name == 'nme_fusion':
                    epoch_classification_loss = 0.0
                    epoch_regularization_loss = 0.0

                # Batch训练
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_train)

                    batch_train_idx = np.arange(start_idx, end_idx)

                    X_eeg_batch = torch.FloatTensor(X_eeg_train[batch_train_idx]).unsqueeze(1).to(device)
                    X_audio_batch = torch.FloatTensor(X_audio_train[batch_train_idx]).to(device)
                    y_batch = torch.LongTensor(y_train[batch_train_idx]).to(device)

                    # 获取batch对应的subject_ids（用于NME模型）
                    batch_subject_ids = subject_ids[train_idx[batch_train_idx]] if model_name == 'nme_fusion' else None

                    # 前向传播
                    if model_name == 'sync_fusion':
                        # 同步性模型需要indices参数（用于从缓存加载）
                        batch_indices = train_idx[batch_train_idx]
                        outputs = model(X_eeg_batch, X_audio_batch, batch_indices)
                        loss = criterion(outputs, y_batch)
                    elif model_name == 'nme_fusion':
                        # NME模型需要subject_ids参数
                        outputs = model(X_eeg_batch, X_audio_batch, batch_subject_ids)
                        classification_loss = criterion(outputs, y_batch)
                        regularization_loss = model.get_nme_regularization_loss()
                        loss = classification_loss + lambda_reg * regularization_loss

                        # 分别累计两种损失
                        epoch_classification_loss += classification_loss.item()
                        epoch_regularization_loss += regularization_loss.item()

                        # 收集样本用于协方差更新（已禁用经验更新，保留但不使用）
                        # model.collect_random_effects_sample()
                    else:
                        outputs = model(X_eeg_batch, X_audio_batch)
                        loss = criterion(outputs, y_batch)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # 统计
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_total += y_batch.size(0)
                    epoch_correct += (predicted == y_batch).sum().item()

                # 定期更新协方差（仅NME模型）
                # 禁用经验更新，让梯度学习来决定方差
                # if model_name == 'nme_fusion' and epoch % covariance_update_freq == 0:
                #     model.update_covariance_estimate()

                # 计算平均loss和accuracy
                avg_loss = epoch_loss / n_batches
                train_acc = epoch_correct / epoch_total

                # 对于NME模型，计算平均的分类损失和正则化损失
                if model_name == 'nme_fusion':
                    avg_classification_loss = epoch_classification_loss / n_batches
                    avg_regularization_loss = epoch_regularization_loss / n_batches

                # 验证（每10个epoch或最后一个epoch）
                if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        # 验证也使用batch处理（特别是同步性模型）
                        val_correct = 0
                        val_total = 0
                        val_batch_size = 16 if model_name in ['sync_fusion', 'nme_fusion'] else 32
                        n_val = len(y_val)
                        n_val_batches = (n_val + val_batch_size - 1) // val_batch_size

                        # NME模型在验证时也计算两种损失
                        if model_name == 'nme_fusion':
                            val_classification_loss = 0.0
                            val_regularization_loss = 0.0

                        for val_batch_idx in range(n_val_batches):
                            val_start_idx = val_batch_idx * val_batch_size
                            val_end_idx = min(val_start_idx + val_batch_size, n_val)
                            val_batch_indices = np.arange(val_start_idx, val_end_idx)

                            X_eeg_val_batch = torch.FloatTensor(X_eeg_raw[test_idx[val_batch_indices]]).unsqueeze(1).to(device)
                            X_audio_val_batch = torch.FloatTensor(X_audio_manual[test_idx[val_batch_indices]]).to(device)
                            y_val_batch = torch.LongTensor(y_val[val_batch_indices]).to(device)

                            # 同步性模型和NME模型需要额外参数
                            if model_name == 'sync_fusion':
                                val_batch_trial_indices = test_idx[val_batch_indices]
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch, val_batch_trial_indices)
                            elif model_name == 'nme_fusion':
                                val_batch_subject_ids = subject_ids[test_idx[val_batch_indices]]
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch, val_batch_subject_ids)

                                # 计算验证集上的两种损失
                                val_batch_classification_loss = criterion(val_outputs, y_val_batch)
                                val_batch_regularization_loss = model.get_nme_regularization_loss()
                                val_classification_loss += val_batch_classification_loss.item()
                                val_regularization_loss += val_batch_regularization_loss.item()
                            else:
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch)

                            _, val_pred = torch.max(val_outputs, 1)
                            val_total += y_val_batch.size(0)
                            val_correct += (val_pred == y_val_batch).sum().item()

                        val_acc = val_correct / val_total

                        # 计算验证集上的平均损失（仅NME模型）
                        if model_name == 'nme_fusion':
                            avg_val_classification_loss = val_classification_loss / n_val_batches
                            avg_val_regularization_loss = val_regularization_loss / n_val_batches

                            # 计算当前方差信息
                            with torch.no_grad():
                                log_var_intercept_clamped = torch.clamp(model.log_intercept_variance, min=-10, max=10)
                                log_var_slope_clamped = torch.clamp(model.log_slope_variance, min=-10, max=10)
                                var_intercept = torch.exp(log_var_intercept_clamped).item()
                                var_slope_mean = torch.exp(log_var_slope_clamped).mean().item()
                                variance_mean = (var_intercept + var_slope_mean) / 2
                                log_var_mean = (model.log_intercept_variance.data.item() + model.log_slope_variance.data.mean().item()) / 2

                        # 打印进度
                        if model_name == 'nme_fusion':
                            print(f"    Epoch {epoch+1}/{n_epochs}: "
                                  f"Train Loss: {avg_loss:.4f} "
                                  f"(Cls: {avg_classification_loss:.4f}, Reg: {avg_regularization_loss:.4f}, λ×Reg: {lambda_reg * avg_regularization_loss:.4f}), "
                                  f"Variance: {variance_mean:.6f} (log_var: {log_var_mean:.4f}), "
                                  f"Train Acc: {train_acc:.4f}, "
                                  f"Val Loss: {avg_val_classification_loss + lambda_reg * avg_val_regularization_loss:.4f} "
                                  f"(Cls: {avg_val_classification_loss:.4f}, Reg: {avg_val_regularization_loss:.4f}), "
                                  f"Val Acc: {val_acc:.4f}")
                        else:
                            print(f"    Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model_state = model.state_dict().copy()  # 保存最佳模型状态
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            print(f"    早停触发（patience={patience}），停止训练")
                            break

            # 记录实际训练的epoch数
            actual_epochs = epoch + 1
            print(f"    实际训练了 {actual_epochs} 个epoch")

            # 最终评估
            if best_model_state is not None:
                model.load_state_dict(best_model_state)  # 加载最佳模型状态
            model.eval()
            with torch.no_grad():
                X_eeg_val_batch = torch.FloatTensor(X_eeg_raw[test_idx]).unsqueeze(1).to(device)
                X_audio_val_batch = torch.FloatTensor(X_audio_manual[test_idx]).to(device)
                y_val_batch = torch.LongTensor(y_val).to(device)

                # 同步性模型和NME模型需要额外参数
                if model_name == 'sync_fusion':
                    val_outputs = model(X_eeg_val_batch, X_audio_val_batch, test_idx)
                elif model_name == 'nme_fusion':
                    val_subject_ids = subject_ids[test_idx]
                    val_outputs = model(X_eeg_val_batch, X_audio_val_batch, val_subject_ids)
                else:
                    val_outputs = model(X_eeg_val_batch, X_audio_val_batch)

                val_probs = F.softmax(val_outputs, dim=1)
                _, val_pred = torch.max(val_outputs, 1)

                val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
                val_balanced_acc = balanced_accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
                val_auc = roc_auc_score(y_val_batch.cpu().numpy(), val_probs[:, 1].cpu().numpy())

            all_fold_results[model_name].append({
                'fold': fold_idx,
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_balanced_acc,
                'val_auc': val_auc,
                'actual_epochs': actual_epochs
            })

            print(f"  {model_name:12s} - 准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}, Epochs: {actual_epochs}")

    # 汇总结果
    all_results = {}
    for model_name, fold_results in all_fold_results.items():
        results_df = pd.DataFrame(fold_results)
        mean_acc = results_df['val_accuracy'].mean()
        mean_balanced_acc = results_df['val_balanced_accuracy'].mean()
        mean_auc = results_df['val_auc'].mean()
        mean_epochs = results_df['actual_epochs'].mean()

        # 对于NME模型，计算每个被试的平均准确率
        per_subject_mean_acc = {}
        if model_name == 'nme_fusion' and 'per_subject_acc' in results_df.columns:
            # 收集所有折中每个被试的准确率
            subject_acc_dict = {}
            for _, row in results_df.iterrows():
                if row['per_subject_acc'] is not None:
                    for subj_name, acc in row['per_subject_acc'].items():
                        if subj_name not in subject_acc_dict:
                            subject_acc_dict[subj_name] = []
                        subject_acc_dict[subj_name].append(acc)

            # 计算每个被试的平均准确率
            for subj_name, acc_list in subject_acc_dict.items():
                per_subject_mean_acc[subj_name] = np.mean(acc_list)

        all_results[model_name] = {
            'results_df': results_df,
            'mean_acc': mean_acc,
            'mean_balanced_acc': mean_balanced_acc,
            'mean_auc': mean_auc,
            'mean_epochs': mean_epochs,
            'per_subject_mean_acc': per_subject_mean_acc if model_name == 'nme_fusion' else None
        }

        model_display_name = {
            'simple': 'Simple拼接',
            'attention': 'Attention融合',
            'gated': 'Gated融合',
            'end_to_end': '端到端融合',
            'audio_only': '仅音频特征',
            'sync_fusion': '同步性融合',
            'nme_fusion': 'NME融合（群体+个体）'
        }.get(model_name, model_name)

        print(f"\n=== {model_display_name} 交叉验证结果汇总 ===")
        print(f"平均验证准确率（所有被试混合）: {mean_acc:.4f} ± {results_df['val_accuracy'].std():.4f}")
        print(f"平均平衡准确率: {mean_balanced_acc:.4f} ± {results_df['val_balanced_accuracy'].std():.4f}")
        print(f"平均AUC: {mean_auc:.4f} ± {results_df['val_auc'].std():.4f}")
        print(f"平均训练轮数: {mean_epochs:.1f} ± {results_df['actual_epochs'].std():.1f}")

        # 对于NME模型，打印每个被试的平均准确率
        if model_name == 'nme_fusion' and per_subject_mean_acc:
            print(f"\n各被试平均准确率（跨5折）:")
            for subj_name in sorted(per_subject_mean_acc.keys()):
                print(f"  {subj_name}: {per_subject_mean_acc[subj_name]:.4f}")

    # 打印最终对比
    print("\n" + "="*80)
    print("NME模型结果：")
    print("="*80)
    print(f"{'模型':<20} {'准确率':<10} {'平衡准确率':<12} {'AUC':<10} {'训练轮数':<8}")
    print("-" * 80)

    for model_name, result_dict in all_results.items():
        model_display_name = {
            # 'simple': 'Simple拼接',
            # 'attention': 'Attention融合',
            # 'gated': 'Gated融合',
            # 'end_to_end': '端到端融合',
            # 'audio_only': '仅音频特征',
            # 'sync_fusion': '同步性融合',
            'nme_fusion': 'NME融合（群体+个体）'
        }.get(model_name, model_name)

        mean_epochs = result_dict['results_df']['actual_epochs'].mean()
        print(f"{model_display_name:<20}{result_dict['mean_acc']:<10.4f}{result_dict['mean_balanced_acc']:<12.4f}{result_dict['mean_auc']:<10.4f}{mean_epochs:<8.1f}")

    return all_results


def train_and_evaluate_end_to_end(X_eeg_raw, X_audio_manual, y,
                                 eeg_channels, eeg_samples, n_folds=5, seed=42):
    """
    端到端训练融合模型
    X_eeg_raw: 原始EEG数据 (n_trials, n_channels, n_timepoints)
    X_audio_manual: 手工音频特征 (n_trials, audio_dim) - left_global + right_global拼接
    """
    print("\n=== 端到端训练融合模型 ===")

    # 数据标准化
    scaler_audio = StandardScaler()
    X_audio_scaled = scaler_audio.fit_transform(X_audio_manual)

    # 转换为tensor
    X_eeg_tensor = torch.FloatTensor(X_eeg_raw).unsqueeze(1)  # (n_trials, 1, n_channels, n_timepoints)
    X_audio_tensor = torch.FloatTensor(X_audio_scaled)
    y_tensor = torch.LongTensor(y)

    # 使用与individual_choice_with_position.py完全一致的折划分方法
    folds, shuffled_indices = build_consistent_folds(len(y), seed=seed, k=n_folds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    all_fold_results = []
    audio_dim = X_audio_tensor.shape[1]
    
    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")

        # 划分数据
        train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)

        X_eeg_train = X_eeg_tensor[train_idx]
        X_audio_train = X_audio_tensor[train_idx]
        y_train = y_tensor[train_idx]

        X_eeg_val = X_eeg_tensor[test_idx]
        X_audio_val = X_audio_tensor[test_idx]
        y_val = y_tensor[test_idx]
        
        # 创建端到端模型
        model = EndToEndFusionClassifier(eeg_channels, eeg_samples, audio_dim)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        n_epochs = 100
        best_val_acc = 0
        best_model_state = None  # 保存最佳模型状态
        patience = 15
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            X_eeg_batch = X_eeg_train.to(device)
            X_audio_batch = X_audio_train.to(device)
            y_batch = y_train.to(device)

            outputs = model(X_eeg_batch, X_audio_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # 验证
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    X_eeg_val_batch = X_eeg_val.to(device)
                    X_audio_val_batch = X_audio_val.to(device)
                    y_val_batch = y_val.to(device)

                    val_outputs = model(X_eeg_val_batch, X_audio_val_batch)
                    _, val_pred = torch.max(val_outputs, 1)
                    val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = model.state_dict().copy()  # 保存最佳模型状态
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

        # 最终评估
        if best_model_state is not None:
            model.load_state_dict(best_model_state)  # 加载最佳模型状态
        model.eval()
        with torch.no_grad():
            X_eeg_val_batch = X_eeg_val.to(device)
            X_audio_val_batch = X_audio_val.to(device)
            y_val_batch = y_val.to(device)

            val_outputs = model(X_eeg_val_batch, X_audio_val_batch)
            val_probs = F.softmax(val_outputs, dim=1)
            _, val_pred = torch.max(val_outputs, 1)

            val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
            val_balanced_acc = balanced_accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
            val_auc = roc_auc_score(y_val_batch.cpu().numpy(), val_probs[:, 1].cpu().numpy())

        all_fold_results.append({
            'fold': fold_idx,
            'val_accuracy': val_acc,
            'val_balanced_accuracy': val_balanced_acc,
            'val_auc': val_auc
        })

        print(f"  端到端模型 - 准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}")
    
    # 汇总结果
    results_df = pd.DataFrame(all_fold_results)
    mean_acc = results_df['val_accuracy'].mean()
    mean_balanced_acc = results_df['val_balanced_accuracy'].mean()
    mean_auc = results_df['val_auc'].mean()

    print(f"\n=== 端到端模型交叉验证结果汇总 ===")
    print(f"平均验证准确率: {mean_acc:.4f} ± {results_df['val_accuracy'].std():.4f}")
    print(f"平均平衡准确率: {mean_balanced_acc:.4f} ± {results_df['val_balanced_accuracy'].std():.4f}")
    print(f"平均AUC: {mean_auc:.4f} ± {results_df['val_auc'].std():.4f}")

    return {
        'results_df': results_df,
        'mean_acc': mean_acc,
        'mean_balanced_acc': mean_balanced_acc,
        'mean_auc': mean_auc
    }


def analyze_individual_differences(nme_results, output_dir, subject_ids):
    """
    分析NME模型的个体差异
    
    参数:
        nme_results: NME模型的交叉验证结果
        output_dir: 输出目录
        subject_ids: 被试ID数组
    """
    print("\n=== NME个体差异分析 ===")
    
    subject_names = list(SUBJECTS_CONFIG.keys())
    
    # 统计每个被试的trial数量
    subject_trial_counts = {}
    for subject_id in range(5):
        count = np.sum(subject_ids == subject_id)
        subject_trial_counts[subject_names[subject_id]] = count
    
    print("各被试的trial数量:")
    for subject_name, count in subject_trial_counts.items():
        print(f"  {subject_name}: {count} 个trials")
    
    # 保存个体差异分析结果
    analysis_path = os.path.join(output_dir, "nme_individual_analysis.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("NME个体差异分析\n")
        f.write("=" * 60 + "\n")
        f.write(f"总被试数: 5\n")
        f.write(f"总trials数: {len(subject_ids)}\n\n")
        
        f.write("各被试的trial数量:\n")
        for subject_name, count in subject_trial_counts.items():
            f.write(f"  {subject_name}: {count} 个trials\n")
        
        f.write("\n注意: 详细的随机效应参数分析需要在训练后从模型对象中提取\n")
        f.write("可以通过保存模型checkpoint来进一步分析个体差异\n")
    
    print(f"个体差异分析结果已保存: {analysis_path}")


def collect_fold_data_from_global_indices(subject_names, subject_consistent_trials, 
                                         subject_eeg_data, subject_labels, 
                                         train_global_indices, val_global_indices,
                                         subject_trial_info, pair_name_to_idx):
    """
    从全局索引收集fold的训练和验证数据
    
    参数:
        subject_names: 被试名称列表
        subject_consistent_trials: dict，键为subject_name，值为该被试的consistent_trials列表
        subject_eeg_data: dict，键为subject_name，值为该被试的EEG数据
        subject_labels: dict，键为subject_name，值为该被试的标签
        train_global_indices: np.ndarray，训练集的全局索引
        val_global_indices: np.ndarray，验证集的全局索引
        subject_trial_info: dict，包含每个被试的偏移量等信息
        pair_name_to_idx: dict，pair_name到音频特征索引的映射
    
    返回:
        包含训练和验证数据的字典
    """
    train_eeg_all = []
    train_labels_all = []
    train_subjects_all = []
    train_pair_indices = []

    val_eeg_all = []
    val_labels_all = []
    val_subjects_all = []
    val_pair_indices = []

    # 为每个被试处理全局索引
    for subject_name in subject_names:
        if subject_name not in subject_consistent_trials:
            continue

        subject_id = subject_names.index(subject_name)
        info = subject_trial_info[subject_name]
        offset = info['offset']
        n_trials = info['n_trials']
        
        trials = subject_consistent_trials[subject_name]
        eeg_data = subject_eeg_data[subject_name]
        labels = subject_labels[subject_name]

        # 提取属于该被试的训练集全局索引
        subject_train_mask = (train_global_indices >= offset) & (train_global_indices < offset + n_trials)
        subject_train_global_indices = train_global_indices[subject_train_mask]
        # 转换为局部索引
        subject_train_local_indices = subject_train_global_indices - offset

        # 提取属于该被试的验证集全局索引
        subject_val_mask = (val_global_indices >= offset) & (val_global_indices < offset + n_trials)
        subject_val_global_indices = val_global_indices[subject_val_mask]
        # 转换为局部索引
        subject_val_local_indices = subject_val_global_indices - offset

        # 收集训练集数据
        for local_idx in subject_train_local_indices:
            if local_idx < len(trials) and local_idx < len(eeg_data):
                trial = trials[local_idx]
                pair_name = trial['pair_name']
                if pair_name in pair_name_to_idx:
                    train_eeg_all.append(eeg_data[local_idx])
                    train_labels_all.append(labels[local_idx])
                    train_subjects_all.append(subject_id)
                    train_pair_indices.append(pair_name_to_idx[pair_name])

        # 收集验证集数据
        for local_idx in subject_val_local_indices:
            if local_idx < len(trials) and local_idx < len(eeg_data):
                trial = trials[local_idx]
                pair_name = trial['pair_name']
                if pair_name in pair_name_to_idx:
                    val_eeg_all.append(eeg_data[local_idx])
                    val_labels_all.append(labels[local_idx])
                    val_subjects_all.append(subject_id)
                    val_pair_indices.append(pair_name_to_idx[pair_name])

    # 转换为numpy数组
    train_eeg_all = np.array(train_eeg_all)
    train_labels_all = np.array(train_labels_all)
    train_subjects_all = np.array(train_subjects_all)
    train_pair_indices = np.array(train_pair_indices)

    val_eeg_all = np.array(val_eeg_all)
    val_labels_all = np.array(val_labels_all)
    val_subjects_all = np.array(val_subjects_all)
    val_pair_indices = np.array(val_pair_indices)

    return {
        'train_eeg': train_eeg_all,
        'train_labels': train_labels_all,
        'train_subjects': train_subjects_all,
        'train_pair_indices': train_pair_indices,
        'val_eeg': val_eeg_all,
        'val_labels': val_labels_all,
        'val_subjects': val_subjects_all,
        'val_pair_indices': val_pair_indices
    }


def collect_fold_data(subject_names, subject_consistent_trials, subject_eeg_data,
                     subject_labels, train_pair_names, val_pair_names, pair_name_to_idx):
    """
    收集单个fold的训练和验证数据
    """
    train_eeg_all = []
    train_labels_all = []
    train_subjects_all = []
    train_pair_indices = []

    val_eeg_all = []
    val_labels_all = []
    val_subjects_all = []
    val_pair_indices = []

    for subject_name in subject_names:
        if subject_name not in subject_consistent_trials:
            continue

        subject_id = subject_names.index(subject_name)
        trials = subject_consistent_trials[subject_name]
        eeg_data = subject_eeg_data[subject_name]
        labels = subject_labels[subject_name]

        # 创建pair_name到trial索引的映射
        pair_to_trial_idx = {}
        for i, trial in enumerate(trials):
            pair_to_trial_idx[trial['pair_name']] = i

        # 分配到训练集
        for pair_name in train_pair_names:
            if pair_name in pair_to_trial_idx:
                idx = pair_to_trial_idx[pair_name]
                pair_idx = pair_name_to_idx[pair_name]
                train_eeg_all.append(eeg_data[idx])
                train_labels_all.append(labels[idx])
                train_subjects_all.append(subject_id)
                train_pair_indices.append(pair_idx)

        # 分配到验证集
        for pair_name in val_pair_names:
            if pair_name in pair_to_trial_idx:
                idx = pair_to_trial_idx[pair_name]
                pair_idx = pair_name_to_idx[pair_name]
                val_eeg_all.append(eeg_data[idx])
                val_labels_all.append(labels[idx])
                val_subjects_all.append(subject_id)
                val_pair_indices.append(pair_idx)

    # 转换为numpy数组
    train_eeg_all = np.array(train_eeg_all)
    train_labels_all = np.array(train_labels_all)
    train_subjects_all = np.array(train_subjects_all)
    train_pair_indices = np.array(train_pair_indices)

    val_eeg_all = np.array(val_eeg_all)
    val_labels_all = np.array(val_labels_all)
    val_subjects_all = np.array(val_subjects_all)
    val_pair_indices = np.array(val_pair_indices)

    return {
        'train_eeg': train_eeg_all,
        'train_labels': train_labels_all,
        'train_subjects': train_subjects_all,
        'train_pair_indices': train_pair_indices,
        'val_eeg': val_eeg_all,
        'val_labels': val_labels_all,
        'val_subjects': val_subjects_all,
        'val_pair_indices': val_pair_indices
    }


def train_fold_model(fold_data, all_audio_features, pair_name_to_idx, pair_names):
    """
    在单个fold上训练模型并返回评估结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据
    train_eeg_all = fold_data['train_eeg']
    train_labels_all = fold_data['train_labels']
    train_pair_indices = fold_data['train_pair_indices']

    val_eeg_all = fold_data['val_eeg']
    val_labels_all = fold_data['val_labels']
    val_pair_indices = fold_data['val_pair_indices']

    # 选择音频特征
    X_audio_train = all_audio_features[train_pair_indices]
    X_audio_val = all_audio_features[val_pair_indices]

    # 创建模型
    model = SimpleFusionClassifier(
        eeg_channels=train_eeg_all.shape[1],
        eeg_samples=train_eeg_all.shape[2],
        audio_dim=X_audio_train.shape[1]
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 准备数据
    X_eeg_train = torch.FloatTensor(train_eeg_all).unsqueeze(1).to(device)
    X_audio_train_tensor = torch.FloatTensor(X_audio_train).to(device)
    y_train = torch.LongTensor(train_labels_all).to(device)

    X_eeg_val = torch.FloatTensor(val_eeg_all).unsqueeze(1).to(device)
    X_audio_val_tensor = torch.FloatTensor(X_audio_val).to(device)
    y_val = torch.LongTensor(val_labels_all).to(device)

    n_epochs = 100
    best_val_acc = 0
    patience = 15
    patience_counter = 0

    # 训练循环
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_eeg_train, X_audio_train_tensor)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # 验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_eeg_val, X_audio_val_tensor)
                _, val_pred = torch.max(val_outputs, 1)
                val_acc = accuracy_score(val_labels_all, val_pred.cpu().numpy())

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                break

    # 最终评估
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_eeg_val, X_audio_val_tensor)
        val_probs = F.softmax(val_outputs, dim=1)
        _, val_pred = torch.max(val_outputs, 1)

        val_acc = accuracy_score(val_labels_all, val_pred.cpu().numpy())
        val_balanced_acc = balanced_accuracy_score(val_labels_all, val_pred.cpu().numpy())
        val_auc = roc_auc_score(val_labels_all, val_probs[:, 1].cpu().numpy())

    return {
        'val_accuracy': val_acc,
        'val_balanced_accuracy': val_balanced_acc,
        'val_auc': val_auc
    }


def save_cross_validation_results(all_fold_results, subject_names):
    """
    保存交叉验证结果
    """
    results_df = pd.DataFrame(all_fold_results)
    results_path = os.path.join(OUTPUT_DIR, f"fusion_classification_cross_validation_results.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n交叉验证结果已保存: {results_path}")

    # 保存汇总信息
    summary_path = os.path.join(OUTPUT_DIR, f"fusion_classification_cross_validation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多被试EEG音频融合分类 - 交叉验证结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"被试数量: {len(subject_names)}\n")
        f.write(f"被试列表: {', '.join(subject_names)}\n")
        f.write(f"交叉验证折数: {len(all_fold_results)}\n\n")

        accuracies = [result['val_accuracy'] for result in all_fold_results]
        balanced_accuracies = [result['val_balanced_accuracy'] for result in all_fold_results]
        aucs = [result['val_auc'] for result in all_fold_results]

        f.write("各折结果:\n")
        for result in all_fold_results:
            f.write(f"  Fold {result['fold_idx']}: 准确率={result['val_accuracy']:.4f}, "
                   f"平衡准确率={result['val_balanced_accuracy']:.4f}, AUC={result['val_auc']:.4f}\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"平均验证准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"平均平衡准确率: {np.mean(balanced_accuracies):.4f} ± {np.std(balanced_accuracies):.4f}\n")
        f.write(f"平均AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")
        f.write(f"{'='*60}\n")

    print(f"汇总信息已保存: {summary_path}")


def main(filter_consistent=True):
    """
    主函数 - 多被试pair-wise交叉验证

    策略:
        1. 每个被试单独筛选一致的trials
        2. 对所有被试都使用相同的fold划分（确保AB/BA在同一fold）
        3. 合并所有被试的训练/验证/测试数据
        4. 在合并的数据上训练最终模型

    参数:
        filter_consistent: 是否筛选两次选择都一致的trials (默认: True)
    """
    print("=" * 60)
    print("EEG音频融合分类 - 多被试pair-wise交叉验证")
    if filter_consistent:
        print("每个被试单独筛选一致的trials，然后合并")
    else:
        print("使用所有trials（不筛选一致性）")
    print("=" * 60)

    project_root = os.getcwd()

    # 1. 加载所有被试的数据
    all_eeg_data, all_labels, all_subject_ids, all_trials, n_common_channels, meta_common = load_all_subjects_data()

    print(f"\n数据汇总:")
    print(f"  总trials数: {len(all_eeg_data)}")
    print(f"  EEG数据形状: {all_eeg_data.shape}")
    print(f"  公共通道数: {n_common_channels}")

    subject_names = list(SUBJECTS_CONFIG.keys())
    print(f"\n被试列表: {subject_names}")

    # 2. 加载音频映射文件，确定所有可能的音频对
    mappings, folder_ranges = load_all_mappings()
    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    pair_names = sorted(list(all_original_names))
    print(f"\n从映射文件提取的音频对总数: {len(pair_names)}")

    # 3. 对每个被试分别筛选一致的trials
    subject_consistent_trials = {}
    subject_eeg_data = {}
    subject_labels = {}

    for subject_name in subject_names:
        subject_id = subject_names.index(subject_name)
        print(f"\n=== 处理被试: {subject_name} ===")

        # 提取当前被试的数据
        subject_mask = all_subject_ids == subject_id
        subject_eeg = all_eeg_data[subject_mask]
        subject_label_array = all_labels[subject_mask]
        subject_trial_list = [all_trials[i] for i in range(len(all_trials)) if subject_mask[i]]

        print(f"  原始数据: {len(subject_eeg)} 个trials")

        # 筛选一致的trials
        if filter_consistent:
            consistent_trials = filter_consistent_trials(subject_trial_list, pair_names)
            print(f"  筛选后一致trials: {len(consistent_trials)}")
        else:
            consistent_trials = []
            for trial in subject_trial_list:
                original_name = trial.get('original_name')
                if not original_name:
                    continue
                consistent_trials.append({
                    'original_name': original_name,
                    'response': trial.get('response'),
                    'reaction_time': trial.get('reaction_time'),
                    'choice': trial.get('response'),
                    'pair_name': original_name
                })
            print(f"  所有trials: {len(consistent_trials)}")

        if len(consistent_trials) == 0:
            print(f"  警告: 被试 {subject_name} 没有有效的trials，跳过")
            continue

        # 对齐EEG数据和consistent_trials
        subject_consistent_eeg = []
        subject_consistent_labels = []

        trial_index_map = {}
        for j, trial in enumerate(subject_trial_list):
            pair_name_key = trial.get('pair_name') or trial.get('original_name')
            if pair_name_key:
                trial_index_map[pair_name_key] = j

        for trial in consistent_trials:
            pair_name = trial['pair_name']
            if pair_name in trial_index_map:
                idx = trial_index_map[pair_name]
                if idx < len(subject_eeg):
                    subject_consistent_eeg.append(subject_eeg[idx])
                    subject_consistent_labels.append(trial['choice'] - 1)

        subject_consistent_eeg = np.array(subject_consistent_eeg)
        subject_consistent_labels = np.array(subject_consistent_labels)

        print(f"  对齐后数据: {len(subject_consistent_eeg)} 个trials")
        print(f"  标签分布: 选择A: {np.sum(subject_consistent_labels == 0)}, 选择B: {np.sum(subject_consistent_labels == 1)}")

        subject_consistent_trials[subject_name] = consistent_trials
        subject_eeg_data[subject_name] = subject_consistent_eeg
        subject_labels[subject_name] = subject_consistent_labels

    # 4. 使用多被试pair-wise fold划分策略
    print(f"\n=== 多被试fold划分 ===")
    print("先为每个被试单独划分fold（确保AB/BA在同一fold），然后合并相同fold")

    # 使用新的多被试fold划分函数
    folds, shuffled_indices, subject_trial_info = build_multi_subject_pair_consistent_folds(
        subject_consistent_trials, n_folds=N_FOLDS, seed=CV_SEED
    )

    print(f"交叉验证设置: {N_FOLDS}折交叉验证")

    # 存储所有fold的结果
    all_fold_results = []

    # 创建pair_name到索引的映射
    pair_name_to_idx = {pair_name: idx for idx, pair_name in enumerate(pair_names)}

    # 4. 提取或加载所有音频对的特征
    print(f"\n=== 提取音频特征 ===")
    
    # 改进后的缓存逻辑：使用hash验证和元数据保存
    # 1. 计算pair_names的hash
    pair_names_str = ','.join(sorted(pair_names))
    pair_names_hash = hashlib.md5(pair_names_str.encode()).hexdigest()[:8]

    # 2. 构建缓存文件路径
    cache_dir = os.path.join(project_root, 'audio_features_cache')
    os.makedirs(cache_dir, exist_ok=True)
    features_cache_path = os.path.join(cache_dir, f'all_audio_features_{pair_names_hash}.npy')
    meta_cache_path = os.path.join(cache_dir, f'all_audio_features_{pair_names_hash}_meta.json')

    # 3. 尝试加载缓存
    all_audio_features = None
    if os.path.exists(features_cache_path) and os.path.exists(meta_cache_path):
        try:
            # 加载元数据验证
            with open(meta_cache_path, 'r', encoding='utf-8') as f:
                cache_meta = json.load(f)
            
            cached_pair_names = cache_meta['pair_names']
            
            # 验证pair_names是否匹配
            if cached_pair_names == pair_names:
                all_audio_features = np.load(features_cache_path)
                print(f"成功从缓存加载所有音频特征: {all_audio_features.shape}")
                print(f"缓存hash: {pair_names_hash}")
            else:
                print(f"缓存中的pair_names不匹配，重新提取...")
                all_audio_features = None
        except Exception as e:
            print(f"缓存文件损坏，重新提取: {e}")
            all_audio_features = None

    # 4. 如果需要重新提取
    if all_audio_features is None:
        print(f"未发现有效音频特征缓存，开始提取所有 {len(pair_names)} 个音频对的特征...")
        # 创建包含所有音频对的虚拟trials
        all_audio_trials = [{'pair_name': pair_name} for pair_name in pair_names]
        left_audio_features, right_audio_features, feature_mapping = extract_audio_features(all_audio_trials, project_root)

        # 检查是否有所有音频对的特征
        if len(left_audio_features) != len(pair_names):
            print(f"警告: 只有 {len(left_audio_features)}/{len(pair_names)} 个音频对成功提取特征")
            # 只保留成功提取的音频对
            successful_pair_names = feature_mapping['pair_names']
            pair_names = successful_pair_names
            all_audio_features = np.concatenate([left_audio_features, right_audio_features], axis=1)

            # 重新创建映射（注意：不应该重新创建folds，因为folds已经基于consistent_trials划分好了）
            pair_name_to_idx = {pair_name: idx for idx, pair_name in enumerate(pair_names)}
            print(f"更新后音频对数: {len(pair_names)}")
            print(f"注意: fold划分保持不变，基于consistent_trials的pair_name")
        else:
            all_audio_features = np.concatenate([left_audio_features, right_audio_features], axis=1)

        # 保存缓存
        np.save(features_cache_path, all_audio_features)
        cache_meta = {
            'pair_names': pair_names,
            'n_pairs': len(pair_names),
            'feature_shape': all_audio_features.shape,
            'cache_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(meta_cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_meta, f, indent=2, ensure_ascii=False)
        print(f"音频特征已缓存到: {features_cache_path}")

    print(f"最终音频特征形状: {all_audio_features.shape}")

    # k-fold交叉验证
    for fold_idx, val_global_indices in enumerate(folds, 1):
        print(f"\n=== Fold {fold_idx}/{N_FOLDS} ===")

        # 当前fold作为验证集，其他所有fold作为训练集
        train_global_indices = []
        for i, fold in enumerate(folds):
            if i != fold_idx - 1:  # fold_idx从1开始，folds索引从0开始
                train_global_indices.extend(fold.tolist())
        train_global_indices = np.array(train_global_indices, dtype=np.int32)
        val_global_indices = np.array(val_global_indices, dtype=np.int32)

        print(f"  训练集全局索引数: {len(train_global_indices)}")
        print(f"  验证集全局索引数: {len(val_global_indices)}")

        # 从全局索引收集训练和验证数据
        fold_data = collect_fold_data_from_global_indices(
            subject_names, subject_consistent_trials, subject_eeg_data,
            subject_labels, train_global_indices, val_global_indices,
            subject_trial_info, pair_name_to_idx
        )

        # 在这个fold上训练模型
        fold_metrics = train_fold_model(fold_data, all_audio_features, pair_name_to_idx, pair_names)

        fold_result = {
            'fold_idx': fold_idx,
            'train_samples': len(fold_data['train_eeg']),
            'val_samples': len(fold_data['val_eeg']),
            **fold_metrics
        }

        all_fold_results.append(fold_result)
        print(f"  Fold {fold_idx} 结果: 准确率={fold_metrics['val_accuracy']:.4f}")

    # 汇总所有fold的结果
    print(f"\n{'='*60}")
    print("交叉验证结果汇总:")
    print(f"{'='*60}")

    accuracies = [result['val_accuracy'] for result in all_fold_results]
    balanced_accuracies = [result['val_balanced_accuracy'] for result in all_fold_results]
    aucs = [result['val_auc'] for result in all_fold_results]

    print(f"平均验证准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"平均平衡准确率: {np.mean(balanced_accuracies):.4f} ± {np.std(balanced_accuracies):.4f}")
    print(f"平均AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # 保存结果
    save_cross_validation_results(all_fold_results, subject_names)


if __name__ == "__main__":
    # ===== 设置参数 =====
    # 修改下面的变量来选择处理模式
    FILTER_CONSISTENT = False # True: 只使用两次选择一致的trials; False: 使用所有trials

    print("=" * 60)
    print("EEG音频融合分类 - 多被试pair-wise交叉验证")
    print(f"筛选一致trials: {FILTER_CONSISTENT}")
    print("可以直接点击运行此脚本")
    print("=" * 60)

    # 运行主程序
    main(filter_consistent=FILTER_CONSISTENT)

