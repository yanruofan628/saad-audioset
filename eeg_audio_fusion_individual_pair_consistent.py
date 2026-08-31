#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单人EEG和音频特征融合分类模型（音频对一致版本）
使用EEGNet提取的EEG特征（poolmean后160维）和音频全局特征（X_global）进行分类
只选择两次选择都一致的trials

关键特性：
- 交叉验证时确保同一音频对的AB和BA版本都在同一个fold中，避免数据泄露

使用方法：
1. 设置BENCHMARK_FILE或CSV_FILE路径（单人trial信息文件）
2. 确保数据路径正确：
   - EEG数据：A:/standard_data_interp_no_ica/[subject_name]
   - 音频映射文件：D:\\D\\research\\audioset下载\\audio_pairs_2s\\分组音频\\folder{1,2,3}
3. 运行脚本：python eeg_audio_fusion_individual_pair_consistent.py

输出：
- fusion_classification_results_[model].csv: 各模型5折交叉验证结果
- fusion_classification_summary.txt: 结果汇总
- fusion_classification_comparison.csv: 模型对比结果
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, \
    classification_report
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

# 七个被试的完整配置
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

# 输出目录
OUTPUT_DIR = f'A:/standard_data_interp_no_ica/fusion_results'
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
        (0, 159, 0),  # folder1: 前159个trials
        (159, 318, 1),  # folder2: 159-318个trials
        (318, 478, 2),  # folder3: 318-478个trials (160个)
    ]

    mappings = []
    for mapping_file in MAPPING_FILES:
        if os.path.exists(mapping_file):
            mapping = parse_audio_mapping_custom(mapping_file)
            mappings.append(mapping)
            print(
                f"加载映射文件: {mapping_file}, 映射数量: {len(mapping) // 2 if any(k.endswith('.wav') for k in mapping) else len(mapping)}")
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
        x = self.conv_ica(x)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.conv_time(x)
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
    t_original = np.linspace(0, len(audio_signal) / sr_audio, len(rms))
    num_samples_target = int(len(audio_signal) * sr_target / sr_audio)
    t_target = np.linspace(0, len(audio_signal) / sr_audio, num_samples_target)

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
                t_original = np.linspace(0, len(rms_L) / self.eeg_sampling_rate, len(rms_L))
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
            print(f"  进度: {i + 1}/{len(consistent_trials)} ({100 * (i + 1) / len(consistent_trials):.1f}%)")

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
    带注意力机制的同步性融合分类模型：EEG特征 + 音频特征 + 同步性特征
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

        # 添加注意力机制：为3个特征组计算权重
        total_dim = 160 + audio_dim + 12
        self.attn_weight = nn.Linear(total_dim, 3)  # 3个权重：EEG、音频、同步性

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )

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

        # 拼接所有特征（用于计算注意力权重）
        combined = torch.cat([eeg_feat, audio_features, sync_features], dim=1)  # (batch, total_dim)

        # 计算注意力权重
        attn_weights = F.softmax(self.attn_weight(combined), dim=1)  # (batch, 3)

        # 注意力加权融合
        weighted_eeg = attn_weights[:, 0:1] * eeg_feat        # EEG权重
        weighted_audio = attn_weights[:, 1:2] * audio_features  # 音频权重
        weighted_sync = attn_weights[:, 2:3] * sync_features    # 同步性权重

        # 拼接加权后的特征
        fused = torch.cat([weighted_eeg, weighted_audio, weighted_sync], dim=1)

        output = self.classifier(fused)

        return output


def build_benchmark_paths(subject_config):
    """构建某个被试的3个benchmark文件路径"""
    benchmark_files = []
    for session in [1, 2, 3]:
        mff_folder = subject_config['mff_folders'][session - 1]
        benchmark_name = subject_config['benchmark_pattern'].format(session=session)
        benchmark_path = os.path.join(mff_folder, f"{benchmark_name}.txt")
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

    # 确保数据和标签数量匹配
    if data.shape[0] != labels.shape[0]:
        min_n = min(data.shape[0], labels.shape[0])
        data = data[:min_n]
        labels = labels[:min_n]

    return data, labels, meta


def load_all_subjects_data():
    """
    加载所有被试的数据

    返回:
        all_eeg_data: (n_total_trials, n_channels_min, n_timepoints) numpy数组
        all_labels: (n_total_trials,) numpy数组
        all_subject_ids: (n_total_trials,) numpy数组，标识每个trial属于哪个被试
        all_trials: trial信息列表，包含pair_name等
        min_channels: 最小通道数
        meta_common: 统一的元信息
    """
    print("\n=== 加载所有被试的数据 ===")

    # 第一步：找到所有被试的最小通道数
    print("步骤1: 检查所有被试的通道数...")
    all_channel_counts = []
    all_metas = {}

    for subject_name, config in SUBJECTS_CONFIG.items():
        try:
            _, _, meta = load_single_subject_eeg(config['processed_data_path'])
            all_channel_counts.append(meta.num_channels)
            all_metas[subject_name] = meta
            print(f"  {subject_name}: {meta.num_channels} 通道")
        except Exception as e:
            print(f"  警告: 无法加载 {subject_name} 的元信息: {e}")
            continue

    if len(all_channel_counts) == 0:
        raise ValueError("无法加载任何被试的数据")

    min_channels = min(all_channel_counts)
    print(f"  最小通道数: {min_channels}")

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

            # 统一通道数（截取前min_channels个通道）
            if eeg_data.shape[1] > min_channels:
                eeg_data = eeg_data[:, :min_channels, :]
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
                print(
                    f"    警告: EEG数据({len(eeg_data)})和trials({len(parsed_trials_all_sessions)})数量不匹配，使用前{n_trials}个")

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
        num_channels=min_channels,
        samples_per_trial=all_eeg_data.shape[2]
    )

    return all_eeg_data, all_labels, all_subject_ids, all_trials_list, min_channels, meta_common


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
    """
    print("\n=== 现场提取音频特征（使用IndependentEarComparisonModel） ===")

    # 获取特征配置
    feature_configs = get_feature_configs_from_linear_regression()
    print(f"使用 {len(feature_configs)} 个特征配置")

    # 提取所有consistent_trials的pair_name
    pair_names_to_extract = [trial['pair_name'] for trial in consistent_trials]
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

    # 提取left_global和right_global特征（用于IndependentEarComparisonModel）
    left_global_list = []
    right_global_list = []
    valid_indices = []

    # 创建pair_name到索引的映射（在base_dataset中）
    pair_to_base_idx = {}
    for idx, name in enumerate(pair_names_to_extract):
        if name not in pair_to_base_idx:
            pair_to_base_idx[name] = idx

    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']

        # 在base_dataset中查找对应的索引
        if pair_name in pair_to_base_idx:
            base_idx = pair_to_base_idx[pair_name]
            # base_dataset和stage_dataset的索引应该是对应的
            if base_idx < len(stage_dataset.left_globals):
                left_feat = stage_dataset.left_globals[base_idx].copy()
                right_feat = stage_dataset.right_globals[base_idx].copy()
                left_global_list.append(left_feat)
                right_global_list.append(right_feat)
                valid_indices.append(i)
            else:
                print(f"警告: 索引超出范围 {base_idx} >= {len(stage_dataset.left_globals)}")
        else:
            print(f"警告: 未找到音频对 {pair_name} 在base_dataset中")

    if len(left_global_list) == 0:
        raise RuntimeError("未能提取任何有效特征")

    # 转换为numpy数组
    left_global_features = np.stack(left_global_list, axis=0)  # (n_trials, ear_feat_dim)
    right_global_features = np.stack(right_global_list, axis=0)  # (n_trials, ear_feat_dim)
    print(f"成功提取音频特征:")
    print(f"  左耳特征形状: {left_global_features.shape}")
    print(f"  右耳特征形状: {right_global_features.shape}")
    print(f"有效特征数量: {len(valid_indices)}/{len(consistent_trials)}")

    return left_global_features, right_global_features, valid_indices


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
            print(f"    {i + 1}. {trial.get('original_name', 'N/A')}")

    # 调试信息：检查pair_names
    print(f"  pair_names数量: {len(pair_names)}")
    if len(pair_names) > 0:
        print(f"  前5个pair_names示例:")
        for i, name in enumerate(pair_names[:5]):
            print(f"    {i + 1}. {name}")

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
            print(f"    {i + 1}. {name}")

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
    trials_select_b = []  # 选择了B的trials

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
    print(f"  choice=1: {final_choice1} ({final_choice1 / len(balanced_trials) * 100:.1f}%)")
    print(f"  choice=2: {final_choice2} ({final_choice2 / len(balanced_trials) * 100:.1f}%)")

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
                            print(f"    早停触发（patience={patience}），停止训练")
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
                'val_auc': val_auc
            })

            print(f"  {model_name:12s} - 准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}")

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


def train_and_evaluate_individual_models(X_eeg_raw, X_audio_manual, y,
                                        eeg_channels, eeg_samples, n_folds=5, seed=42,
                                        consistent_trials=None):
    """
    训练和评估单人EEG音频融合模型：
    - simple: 简单拼接融合（端到端）
    - attention: 注意力融合（端到端）
    - gated: 门控融合（端到端）
    - end_to_end: 直接融合（端到端）
    - audio_only: 只用音频特征（消融实验）
    - sync_fusion: 同步性融合（可选，使用预计算缓存）

    参数:
        consistent_trials: 包含pair_name信息的trial列表，用于同步性模型
    """
    print("\n=== 训练和评估所有端到端融合模型 ===")

    # 使用确保同一音频对的AB/BA版本在同一fold的划分方法
    if consistent_trials is not None:
        folds, shuffled_indices = build_pair_consistent_folds(consistent_trials, n_folds=n_folds, seed=seed)
    else:
        # 如果没有提供consistent_trials，回退到标准划分
        print("  警告: 未提供consistent_trials，使用标准交叉验证划分")
        folds, shuffled_indices = build_consistent_folds(len(y), seed=seed, k=n_folds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    all_fold_results = {
        'simple': [],
        'attention': [],
        'gated': [],
        'end_to_end': [],
        'audio_only': []
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

        # 创建模型（根据配置决定是否包含同步性融合模型）
        models = {
            'simple': SimpleFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            'attention': AttentionFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            'gated': GatedFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            'end_to_end': EndToEndFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            'audio_only': AudioOnlyClassifier(eeg_channels, eeg_samples, audio_dim),
        }

        # 根据配置决定是否添加同步性融合模型
        if TRAIN_SYNC_FUSION and sync_features_cache is not None:
            models['sync_fusion'] = SyncFusionClassifier(
                eeg_channels, eeg_samples, audio_dim,
                sync_features_cache=sync_features_cache
            )
            print(f"  将训练 {len(models)} 个模型（包括同步性融合模型，使用预计算缓存）")
        else:
            if TRAIN_SYNC_FUSION:
                print(f"  警告: 同步性融合模型需要consistent_trials参数，跳过")
            else:
                print(f"  将训练 {len(models)} 个模型（跳过同步性融合模型，设置TRAIN_SYNC_FUSION=True以启用）")

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
            batch_size = 16 if model_name in ['sync_fusion'] else 32  # 同步性模型用更小的batch
            n_train = len(y_train)
            n_batches = (n_train + batch_size - 1) // batch_size


            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                # Batch训练
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_train)

                    batch_train_idx = np.arange(start_idx, end_idx)

                    X_eeg_batch = torch.FloatTensor(X_eeg_train[batch_train_idx]).unsqueeze(1).to(device)
                    X_audio_batch = torch.FloatTensor(X_audio_train[batch_train_idx]).to(device)
                    y_batch = torch.LongTensor(y_train[batch_train_idx]).to(device)

                    # 前向传播
                    if model_name == 'sync_fusion':
                        # 同步性模型需要indices参数（用于从缓存加载）
                        batch_indices = train_idx[batch_train_idx]
                        outputs = model(X_eeg_batch, X_audio_batch, batch_indices)
                        loss = criterion(outputs, y_batch)
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


                # 计算平均loss和accuracy
                avg_loss = epoch_loss / n_batches
                train_acc = epoch_correct / epoch_total

                # 验证（每10个epoch或最后一个epoch）
                if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        # 验证也使用batch处理（特别是同步性模型）
                        val_correct = 0
                        val_total = 0
                        val_batch_size = 16 if model_name in ['sync_fusion'] else 32
                        n_val = len(y_val)
                        n_val_batches = (n_val + val_batch_size - 1) // val_batch_size

                        for val_batch_idx in range(n_val_batches):
                            val_start_idx = val_batch_idx * val_batch_size
                            val_end_idx = min(val_start_idx + val_batch_size, n_val)
                            val_batch_indices = np.arange(val_start_idx, val_end_idx)

                            X_eeg_val_batch = torch.FloatTensor(X_eeg_raw[test_idx[val_batch_indices]]).unsqueeze(1).to(
                                device)
                            X_audio_val_batch = torch.FloatTensor(X_audio_manual[test_idx[val_batch_indices]]).to(
                                device)
                            y_val_batch = torch.LongTensor(y_val[val_batch_indices]).to(device)

                            # 同步性模型需要额外参数
                            if model_name == 'sync_fusion':
                                val_batch_trial_indices = test_idx[val_batch_indices]
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch, val_batch_trial_indices)
                            else:
                                val_outputs = model(X_eeg_val_batch, X_audio_val_batch)

                            _, val_pred = torch.max(val_outputs, 1)
                            val_total += y_val_batch.size(0)
                            val_correct += (val_pred == y_val_batch).sum().item()

                        val_acc = val_correct / val_total

                        # 打印进度
                        print(
                            f"    Epoch {epoch + 1}/{n_epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model_state = model.state_dict().copy()  # 保存最佳模型状态
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            print(f"    早停触发（patience={patience}），停止训练")
                            break

            # 最终评估
            if best_model_state is not None:
                model.load_state_dict(best_model_state)  # 加载最佳模型状态
            model.eval()
            with torch.no_grad():
                X_eeg_val_batch = torch.FloatTensor(X_eeg_raw[test_idx]).unsqueeze(1).to(device)
                X_audio_val_batch = torch.FloatTensor(X_audio_manual[test_idx]).to(device)
                y_val_batch = torch.LongTensor(y_val).to(device)

                # 同步性模型需要额外参数
                if model_name == 'sync_fusion':
                    val_outputs = model(X_eeg_val_batch, X_audio_val_batch, test_idx)
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
                'val_auc': val_auc
            })

            print(f"  {model_name:12s} - 准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}")

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

        model_display_name = {
            'simple': 'Simple拼接',
            'attention': 'Attention融合',
            'gated': 'Gated融合',
            'end_to_end': '端到端融合',
            'audio_only': '仅音频特征',
            'sync_fusion': '同步性融合'
        }.get(model_name, model_name)

        print(f"\n=== {model_display_name} 交叉验证结果汇总 ===")
        print(f"平均验证准确率: {mean_acc:.4f} ± {results_df['val_accuracy'].std():.4f}")
        print(f"平均平衡准确率: {mean_balanced_acc:.4f} ± {results_df['val_balanced_accuracy'].std():.4f}")
        print(f"平均AUC: {mean_auc:.4f} ± {results_df['val_auc'].std():.4f}")

    # 打印最终对比
    print("\n" + "=" * 80)
    print("五种模型对比结果：")
    print("=" * 80)
    print(f"{'模型':<20} {'准确率':<10} {'平衡准确率':<12} {'AUC':<10}")
    print("-" * 80)

    for model_name, result_dict in all_results.items():
        model_display_name = {
            'simple': 'Simple拼接',
            'attention': 'Attention融合',
            'gated': 'Gated融合',
            'end_to_end': '端到端融合',
            'audio_only': '仅音频特征',
            'sync_fusion': '同步性融合'
        }.get(model_name, model_name)

        print(
            f"{model_display_name:<20}{result_dict['mean_acc']:<10.4f}{result_dict['mean_balanced_acc']:<12.4f}{result_dict['mean_auc']:<10.4f}")

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
                        print(f"    早停触发（patience={patience}），停止训练")
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


def main(subject_name='hanglei', filter_consistent=True):
    """
    主函数 - 单人EEG音频融合分类

    参数:
        subject_name: 被试名称 (默认: 'hanglei')
        filter_consistent: 是否筛选两次选择都一致的trials (默认: True)
                           如果为False，则使用所有trials
    """
    print("=" * 60)
    print(f"单人EEG和音频特征融合分类模型 - {subject_name}")
    if filter_consistent:
        print("只选择两次选择都一致的trials")
    else:
        print("使用所有trials（不筛选一致性）")
    print("=" * 60)

    project_root = os.getcwd()

    # 1. 加载单个被试的数据
    if subject_name not in SUBJECTS_CONFIG:
        raise ValueError(f"未知的被试名称: {subject_name}. 可用的被试: {list(SUBJECTS_CONFIG.keys())}")

    config = SUBJECTS_CONFIG[subject_name]
    print(f"加载被试: {subject_name}")
    print(f"数据路径: {config['processed_data_path']}")

    eeg_data, labels, meta = load_single_subject_eeg(config['processed_data_path'])
    print(f"EEG数据形状: {eeg_data.shape}, 标签数量: {len(labels)}")
    print(f"EEG通道数: {meta.num_channels}")

    # 2. 加载音频映射文件
    mappings, folder_ranges = load_all_mappings()

    # 3. 加载该被试的benchmark文件
    benchmark_files = build_benchmark_paths(config)
    print(f"\nBenchmark文件:")
    for i, bf in enumerate(benchmark_files, 1):
        print(f"  Session {i}: {bf}")

    # 解析trial信息
    all_trials = load_hanglei_trials(benchmark_files, mappings, folder_ranges)
    print(f"加载了 {len(all_trials)} 个原始trials")

    # 从映射文件中提取所有音频对名称
    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    pair_names = sorted(list(all_original_names))
    print(f"从映射文件提取的音频对总数: {len(pair_names)}")

    # 4. 根据参数决定是否筛选一致的trials
    if filter_consistent:
        consistent_trials = filter_consistent_trials(all_trials, pair_names)
        if len(consistent_trials) == 0:
            print("\n错误: 没有一致的trials")
            return
        print(f"\n筛选后一致的trials数量: {len(consistent_trials)}")
    else:
        # 不筛选一致性，直接使用所有trials
        # all_trials中的trials已经包含了original_name，不需要重新映射
        consistent_trials = []
        for trial in all_trials:
            # all_trials中的trial已经包含了original_name，直接使用
            original_name = trial.get('original_name')
            if not original_name:
                continue

            consistent_trials.append({
                'original_name': original_name,
                'response': trial.get('response'),
                'reaction_time': trial.get('reaction_time'),
                'choice': trial.get('response'),  # 直接使用response作为choice
                'pair_name': original_name
            })
        print(f"\n使用所有trials（不筛选一致性）: {len(consistent_trials)} 个trials")

    # 5. 对齐EEG数据和一致的trials
    # 注意：EEG数据和trials的顺序需要对应
    # 这里假设EEG数据按trial顺序存储，与benchmark文件顺序一致
    consistent_eeg_data = []
    consistent_labels = []
    consistent_indices = []

    # 5. 对齐EEG数据和一致的trials
    # 创建trial索引映射（基于pair_name）
    consistent_eeg_data = []
    consistent_labels = []
    consistent_subject_ids = []
    consistent_indices = []

    # 创建all_trials的索引映射（基于pair_name或original_name）
    all_trials_index_map = {}
    for j, all_trial in enumerate(all_trials):
        pair_name_key = all_trial.get('pair_name') or all_trial.get('original_name')
        if pair_name_key:
            if pair_name_key not in all_trials_index_map:
                all_trials_index_map[pair_name_key] = []
            all_trials_index_map[pair_name_key].append(j)

    for idx, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']
        # 在all_trials中查找匹配的trial
        matching_trial_idx = None

        # 优先使用索引映射
        if pair_name in all_trials_index_map:
            # 如果有多个匹配，取第一个
            matching_trial_idx = all_trials_index_map[pair_name][0]
        else:
            # 回退到线性搜索
            for j, all_trial in enumerate(all_trials):
                if all_trial.get('pair_name') == pair_name or all_trial.get('original_name') == pair_name:
                    matching_trial_idx = j
                    break

        if matching_trial_idx is not None and matching_trial_idx < len(eeg_data):
            consistent_eeg_data.append(eeg_data[matching_trial_idx])
            # choice: 1=选左, 2=选右 -> 转换为0=选左, 1=选右
            # consistent_trials中的choice来自filter_consistent_trials的结果
            consistent_labels.append(trial['choice'] - 1)
            consistent_indices.append(matching_trial_idx)

    if len(consistent_eeg_data) == 0:
        print("\n错误: 未能对齐EEG数据和一致的trials")
        return

    consistent_eeg_data = np.array(consistent_eeg_data)
    consistent_labels = np.array(consistent_labels)
    print(f"对齐后的数据: {len(consistent_eeg_data)} 个trials, EEG形状: {consistent_eeg_data.shape}")
    print(f"标签分布: 选择A: {np.sum(consistent_labels == 0)}, 选择B: {np.sum(consistent_labels == 1)}")

    # 6. 提取音频特征（基于所有音频对，固定缓存）
    print(f"\n=== 提取音频特征（基于所有 {len(pair_names)} 个音频对）===")

    # 创建音频特征缓存路径（基于所有音频对，不依赖筛选条件）
    audio_feature_cache_dir = os.path.join(OUTPUT_DIR, f'audio_feature_cache_{subject_name}')
    os.makedirs(audio_feature_cache_dir, exist_ok=True)
    all_audio_cache_path = os.path.join(audio_feature_cache_dir, f'all_audio_features_{len(pair_names)}.npy')

    # 检查是否存在所有音频对的特征缓存
    if os.path.exists(all_audio_cache_path):
        print(f"发现所有音频对的特征缓存，从缓存加载...")
        try:
            all_audio_features = np.load(all_audio_cache_path)
            print(f"成功从缓存加载所有音频特征: {all_audio_features.shape}")
        except Exception as e:
            print(f"音频特征缓存损坏，重新提取: {e}")
            all_audio_features = None
    else:
        print(f"未发现音频特征缓存，开始提取所有 {len(pair_names)} 个音频对的特征...")
        # 创建包含所有音频对的虚拟trials
        all_audio_trials = [{'pair_name': pair_name} for pair_name in pair_names]
        left_audio_features, right_audio_features, valid_audio_indices = extract_audio_features(all_audio_trials, project_root)

        if len(valid_audio_indices) != len(pair_names):
            print(f"警告: 只有 {len(valid_audio_indices)}/{len(pair_names)} 个音频对成功提取特征")
            # 只保留成功提取的音频对
            valid_pair_names = [pair_names[i] for i in valid_audio_indices]
            pair_names = valid_pair_names
            all_audio_features = np.concatenate([left_audio_features, right_audio_features], axis=1)
        else:
            all_audio_features = np.concatenate([left_audio_features, right_audio_features], axis=1)

        # 保存缓存
        np.save(all_audio_cache_path, all_audio_features)
        print(f"音频特征已缓存到: {all_audio_cache_path}")
        print(f"  特征形状: {all_audio_features.shape}")

    # 7. 根据筛选的trials选择对应的音频特征
    print(f"\n=== 根据筛选条件选择音频特征 ===")
    # 创建pair_name到索引的映射
    pair_name_to_idx = {pair_name: idx for idx, pair_name in enumerate(pair_names)}

    selected_audio_indices = []
    selected_eeg_indices = []

    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']
        if pair_name in pair_name_to_idx:
            audio_idx = pair_name_to_idx[pair_name]
            selected_audio_indices.append(audio_idx)
            selected_eeg_indices.append(i)

    # 选择对应的音频特征和EEG数据
    X_audio_manual = all_audio_features[selected_audio_indices]
    consistent_eeg_data = consistent_eeg_data[selected_eeg_indices]
    consistent_labels = consistent_labels[selected_eeg_indices]
    consistent_trials = [consistent_trials[i] for i in selected_eeg_indices]

    print(f"筛选后数据对齐:")
    print(f"  EEG数据形状: {consistent_eeg_data.shape}")
    print(f"  音频特征形状: {X_audio_manual.shape}")
    print(f"  标签数量: {len(consistent_labels)}")
    print(f"  trials数量: {len(consistent_trials)}")

    # 8. 保存筛选后的EEG数据缓存（用于同步性特征）
    feature_cache_dir = os.path.join(OUTPUT_DIR, f'feature_cache_{subject_name}')
    os.makedirs(feature_cache_dir, exist_ok=True)

    eeg_cache_path = os.path.join(feature_cache_dir, 'consistent_eeg_data.npy')
    labels_cache_path = os.path.join(feature_cache_dir, 'consistent_labels.npy')
    consistent_trials_cache_path = os.path.join(feature_cache_dir, 'consistent_trials.pkl')

    # 保存筛选后的EEG数据缓存（用于同步性特征）
    np.save(eeg_cache_path, consistent_eeg_data)
    np.save(labels_cache_path, consistent_labels)

    # 保存consistent_trials缓存
    import pickle
    with open(consistent_trials_cache_path, 'wb') as f:
        pickle.dump(consistent_trials, f)

    print(f"筛选后的EEG数据缓存已保存到: {feature_cache_dir}")
    print(f"  EEG数据: {eeg_cache_path}")
    print(f"  标签: {labels_cache_path}")
    print(f"  consistent_trials: {consistent_trials_cache_path}")

    # 确保consistent_subject_ids存在
    if 'consistent_subject_ids' not in locals() or consistent_subject_ids is None:
        print("警告: consistent_subject_ids不存在，将重新生成")
        # 从consistent_trials中重新生成subject_ids（如果可能）
        # 否则使用默认值（所有为0）
        consistent_subject_ids = np.zeros(len(consistent_labels), dtype=np.int32)

    # 8. 训练所有端到端融合模型
    print(f"\n=== 开始训练单人EEG音频融合模型 ===")

    all_results = train_and_evaluate_individual_models(
        consistent_eeg_data,  # 原始EEG数据（用于端到端）
        X_audio_manual,  # 手工音频特征（left+right拼接）
        consistent_labels,
        eeg_channels=meta.num_channels,  # 使用通道数
        eeg_samples=consistent_eeg_data.shape[2],  # 时间点数
        n_folds=N_FOLDS,
        seed=CV_SEED,
        consistent_trials=consistent_trials  # 传递trial信息用于同步性模型
    )

    # 10. 保存结果
    for model_name, result_dict in all_results.items():
        results_df = result_dict['results_df']
        model_name_safe = model_name.replace('_', '_')
        results_path = os.path.join(OUTPUT_DIR, f"fusion_classification_results_{subject_name}_{model_name_safe}.csv")
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n{model_name}模型结果已保存: {results_path}")

    # 保存汇总信息
    summary_path = os.path.join(OUTPUT_DIR, f"fusion_classification_summary_{subject_name}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"单人EEG音频融合分类结果汇总（{subject_name}，五种模型对比）\n")
        f.write("=" * 60 + "\n")
        f.write(f"subject: {subject_name}\n")
        f.write(f"n_trials: {len(consistent_eeg_data)}\n")
        f.write(f"eeg_feature_dim: 160 (CNN提取用于端到端)\n")
        f.write(f"eeg_channels: {min_channels}\n")
        f.write(f"audio_feature_dim (left+right): {X_audio_manual.shape[1]}\n\n")

        for model_name, result_dict in all_results.items():
            model_display_name = {
                'simple': 'Simple拼接',
                'attention': 'Attention融合',
                'gated': 'Gated融合',
                'end_to_end': '端到端融合',
                'audio_only': '仅音频特征',
                'sync_fusion': '同步性融合'
            }.get(model_name, model_name)

            f.write(f"{model_display_name}:\n")
            f.write(
                f"  平均验证准确率: {result_dict['mean_acc']:.4f} ± {result_dict['results_df']['val_accuracy'].std():.4f}\n")
            f.write(
                f"  平均平衡准确率: {result_dict['mean_balanced_acc']:.4f} ± {result_dict['results_df']['val_balanced_accuracy'].std():.4f}\n")
            f.write(f"  平均AUC: {result_dict['mean_auc']:.4f} ± {result_dict['results_df']['val_auc'].std():.4f}\n\n")

    print(f"\n汇总信息已保存: {summary_path}")

    # 创建对比结果CSV
    comparison_data = []
    for model_name, result_dict in all_results.items():
        model_display_name = {
            'simple': 'Simple拼接',
            'attention': 'Attention融合',
            'gated': 'Gated融合',
            'end_to_end': '端到端融合',
            'audio_only': '仅音频特征',
            'sync_fusion': '同步性融合'
        }.get(model_name, model_name)

        comparison_data.append({
            'model': model_display_name,
            'mean_accuracy': result_dict['mean_acc'],
            'std_accuracy': result_dict['results_df']['val_accuracy'].std(),
            'mean_balanced_accuracy': result_dict['mean_balanced_acc'],
            'std_balanced_accuracy': result_dict['results_df']['val_balanced_accuracy'].std(),
            'mean_auc': result_dict['mean_auc'],
            'std_auc': result_dict['results_df']['val_auc'].std()
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(OUTPUT_DIR, f"fusion_classification_comparison_{subject_name}.csv")
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"模型对比结果已保存: {comparison_path}")



if __name__ == "__main__":
    # ===== 设置要处理的被试和参数 =====
    # 修改下面的变量来选择不同的被试和处理模式
    SUBJECT_TO_PROCESS = 'hanglei'  # 可选: 'hanglei', 'yanxingzhuo', 'jinxiaoyue', 'chenxianwei', 'yeziyuan', 'zhangzhiyao', 'haoxiang'
    FILTER_CONSISTENT = False  # True: 只使用两次选择一致的trials; False: 使用所有trials

    print("=" * 60)
    print("EEG音频融合分类 - 单人处理")
    print(f"当前设置处理被试: {SUBJECT_TO_PROCESS}")
    print(f"筛选一致trials: {FILTER_CONSISTENT}")
    print("可以直接点击运行此脚本")
    print("=" * 60)

    # 运行主程序
    main(subject_name=SUBJECT_TO_PROCESS, filter_consistent=FILTER_CONSISTENT)

