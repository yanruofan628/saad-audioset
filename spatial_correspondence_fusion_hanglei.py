#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极轻量级左右脑空间对应融合模型
基于电极坐标的左右脑空间对应建模，左右脑都包含中线电极
参数数量：~2,753（比AttentionFusionClassifier少75%）

使用方法：
1. 确保数据路径正确：
   - EEG数据：A:/standard_data_interp_no_ica/hanglei
   - 电极坐标：从coordinates.xml读取
2. 运行脚本：python spatial_correspondence_fusion_hanglei.py

输出：
- spatial_correspondence_results.csv: 5折交叉验证结果
- spatial_correspondence_summary.txt: 结果汇总
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import xml.etree.ElementTree as ET
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
    check_left_right_consistency, parse_pair_name
)
from linear_regression_selected_features import (
    build_consistent_folds
)
from feature_stage_attention_model import (
    get_feature_configs_from_linear_regression, SR, HOP, DEFAULT_STAGE_SECONDS,
    IndependentEarComparisonModel
)
from attention_loudness_model import FeatureTimeSeriesDataset
from feature_stage_attention_model import StageFeatureDataset
from eeg_audio_fusion_classify_hanglei import (
    EEGNetFeatureExtractor, AudioFeatureExtractor,
    load_all_mappings, load_hanglei_trials, extract_audio_features,
    BENCHMARK_FILES, MAPPING_FILES, PROCESSED_DATA_PATH, OUTPUT_DIR,
    CV_SEED, N_FOLDS, SAMPLING_RATE
)

# 输出目录
SPATIAL_OUTPUT_DIR = os.path.join(PROCESSED_DATA_PATH, "spatial_correspondence_results")
os.makedirs(SPATIAL_OUTPUT_DIR, exist_ok=True)

# 中线阈值（x坐标绝对值小于此值视为中线）
MIDLINE_THRESHOLD = 0.05


def load_electrode_coordinates(coordinates_xml_path):
    """
    从coordinates.xml读取电极坐标

    参数:
    coordinates_xml_path: coordinates.xml文件路径

    返回:
    electrode_coords: dict, {channel_name: (x, y, z)}
    """
    if coordinates_xml_path is None:
        print("坐标文件路径为None，跳过坐标读取")
        return None

    if not os.path.exists(coordinates_xml_path):
        print(f"警告: 坐标文件不存在: {coordinates_xml_path}")
        return None

    try:
        tree = ET.parse(coordinates_xml_path)
        root = tree.getroot()

        # 调试：打印XML结构
        print(f"XML根节点名称: {root.tag}")
        print(f"根节点属性: {root.attrib}")

        electrode_coords = {}

        # 定义命名空间
        ns = {'egi': 'http://www.egi.com/coordinates_mff'}

        # 查找所有sensor节点（使用命名空间）
        sensors = root.findall('.//egi:sensor', ns)
        print(f"找到 {len(sensors)} 个sensor节点")

        # 如果没找到，尝试不使用命名空间
        if len(sensors) == 0:
            sensors = root.findall('.//sensor')
            print(f"不使用命名空间找到 {len(sensors)} 个sensor节点")

        for sensor in sensors:
            # 获取number标签的值（使用命名空间）
            number_elem = sensor.find('egi:number', ns)
            if number_elem is None:
                # 尝试不使用命名空间
                number_elem = sensor.find('number')
            if number_elem is None or number_elem.text is None:
                continue

            try:
                number = int(number_elem.text.strip())
            except (ValueError, TypeError):
                continue

            # 构造通道名称：优先使用EEG{number}格式
            channel_name = f'EEG{number:03d}'  # EEG001, EEG002, etc.

            # 获取坐标（使用命名空间）
            x_elem = sensor.find('egi:x', ns)
            y_elem = sensor.find('egi:y', ns)
            z_elem = sensor.find('egi:z', ns)

            # 如果没找到，尝试不使用命名空间
            if x_elem is None:
                x_elem = sensor.find('x')
            if y_elem is None:
                y_elem = sensor.find('y')
            if z_elem is None:
                z_elem = sensor.find('z')

            if x_elem is not None and y_elem is not None and z_elem is not None:
                try:
                    x = float(x_elem.text.strip())
                    y = float(y_elem.text.strip())
                    z = float(z_elem.text.strip())
                    electrode_coords[channel_name] = (x, y, z)

                    # 也添加E{number}格式作为备选
                    alt_name = f'E{number}'
                    electrode_coords[alt_name] = (x, y, z)

                except (ValueError, AttributeError, TypeError):
                    continue

        print(f"成功读取 {len(electrode_coords)//2} 个电极坐标")  # 除以2因为每个电极有两种命名方式
        if len(electrode_coords) > 0:
            print(f"前5个电极坐标示例: {list(electrode_coords.items())[:5]}")
        else:
            print("警告: 未能读取任何电极坐标，请检查XML文件结构")
        return electrode_coords

    except Exception as e:
        print(f"读取坐标文件错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_electrode_hemisphere_mapping(coordinates_xml_path, channel_names):
    """
    根据电极坐标判断左右脑，左右脑都包含中线电极
    
    参数:
    coordinates_xml_path: coordinates.xml文件路径
    channel_names: 通道名称列表
    
    返回:
    left_channels: 左脑通道索引列表（包含中线）
    right_channels: 右脑通道索引列表（包含中线）
    midline_channels: 中线通道索引列表
    """
    electrode_coords = load_electrode_coordinates(coordinates_xml_path)
    
    if electrode_coords is None:
        # 如果无法读取坐标，使用通道名称规则（10-20系统）
        print("使用通道名称规则判断左右脑（10-20系统）")
        print(f"前10个通道名称: {channel_names[:10]}")
        
        left_channels = []
        right_channels = []
        midline_channels = []
        
        # 10-20系统通道名称规则
        # 中线通道：名称包含'z'（不区分大小写）
        # 左脑通道：名称以奇数结尾（1,3,5,7,9）或包含'L'
        # 右脑通道：名称以偶数结尾（2,4,6,8,10）或包含'R'
        
        for idx, ch_name in enumerate(channel_names):
            ch_upper = ch_name.upper().strip()
            
            # 检查是否是中线通道（名称包含'z'）
            if 'Z' in ch_upper:
                midline_channels.append(idx)
            # 检查是否是左脑通道
            elif (ch_upper.endswith('1') or ch_upper.endswith('3') or 
                  ch_upper.endswith('5') or ch_upper.endswith('7') or 
                  ch_upper.endswith('9') or 'L' in ch_upper or
                  ch_upper.startswith('FP1') or ch_upper.startswith('F1') or
                  ch_upper.startswith('C1') or ch_upper.startswith('P1') or
                  ch_upper.startswith('O1') or ch_upper.startswith('F7') or
                  ch_upper.startswith('T7') or ch_upper.startswith('P7') or
                  ch_upper.startswith('FC1') or ch_upper.startswith('FC3') or
                  ch_upper.startswith('FC5') or ch_upper.startswith('CP1') or
                  ch_upper.startswith('CP3') or ch_upper.startswith('CP5') or
                  ch_upper.startswith('TP7') or ch_upper.startswith('TP9') or
                  ch_upper.startswith('AF3') or ch_upper.startswith('AF7')):
                left_channels.append(idx)
            # 检查是否是右脑通道
            elif (ch_upper.endswith('2') or ch_upper.endswith('4') or 
                  ch_upper.endswith('6') or ch_upper.endswith('8') or 
                  ch_upper.endswith('10') or 'R' in ch_upper or
                  ch_upper.startswith('FP2') or ch_upper.startswith('F2') or
                  ch_upper.startswith('C2') or ch_upper.startswith('P2') or
                  ch_upper.startswith('O2') or ch_upper.startswith('F8') or
                  ch_upper.startswith('T8') or ch_upper.startswith('P8') or
                  ch_upper.startswith('FC2') or ch_upper.startswith('FC4') or
                  ch_upper.startswith('FC6') or ch_upper.startswith('CP2') or
                  ch_upper.startswith('CP4') or ch_upper.startswith('CP6') or
                  ch_upper.startswith('TP8') or ch_upper.startswith('TP10') or
                  ch_upper.startswith('AF4') or ch_upper.startswith('AF8')):
                right_channels.append(idx)
            else:
                # 如果无法判断，尝试从通道编号判断（EEG001, EEG002等）
                # 提取数字部分
                import re
                numbers = re.findall(r'\d+', ch_name)
                if numbers:
                    last_num = int(numbers[-1])
                    if last_num % 2 == 1:  # 奇数
                        left_channels.append(idx)
                    else:  # 偶数
                        right_channels.append(idx)
                else:
                    # 默认分配到左脑（保守策略）
                    left_channels.append(idx)
        
        # 中线通道同时加入左右脑
        left_channels_with_midline = sorted(set(left_channels + midline_channels))
        right_channels_with_midline = sorted(set(right_channels + midline_channels))
        
        print(f"左脑通道（含中线）: {len(left_channels_with_midline)} 个")
        print(f"右脑通道（含中线）: {len(right_channels_with_midline)} 个")
        print(f"中线通道: {len(midline_channels)} 个")
        print(f"纯左脑通道: {len(left_channels)} 个")
        print(f"纯右脑通道: {len(right_channels)} 个")
        
        # 打印一些示例通道分配
        print("\n通道分配示例（前20个）:")
        for idx in range(min(20, len(channel_names))):
            ch_name = channel_names[idx]
            if idx in midline_channels:
                side = "中线"
            elif idx in left_channels:
                side = "左脑"
            elif idx in right_channels:
                side = "右脑"
            else:
                side = "未知"
            print(f"  {ch_name}: {side}")
        
        return left_channels_with_midline, right_channels_with_midline, midline_channels
    
    # 根据坐标判断
    left_channels = []
    right_channels = []
    midline_channels = []

    for idx, ch_name in enumerate(channel_names):
        # 尝试多种通道名称格式匹配坐标
        possible_names = [
            ch_name,                                    # 原始名称，如EEG001
            ch_name.replace('EEG', 'E'),               # E001
            ch_name.replace('EEG', ''),                # 001
            f'E{idx+1}',                               # E1, E2, etc.
            f'EEG{idx+1:03d}',                         # EEG001, EEG002, etc.
            f'EEG{idx+1}',                             # EEG1, EEG2, etc.
        ]

        coord = None
        matched_name = None
        for name in possible_names:
            if name in electrode_coords:
                coord = electrode_coords[name]
                matched_name = name
                break

        if coord is None:
            # 如果找不到坐标，使用通道名称规则
            print(f"警告: 通道 {ch_name} (索引{idx}) 找不到坐标，使用名称规则")
            ch_upper = ch_name.upper()
            if 'Z' in ch_upper or ch_upper.endswith('Z'):
                midline_channels.append(idx)
            elif any(suffix in ch_upper for suffix in ['1', '3', '5', '7', '9', 'L']):
                left_channels.append(idx)
            else:
                right_channels.append(idx)
            continue

        x, y, z = coord

        # 判断左右脑（x坐标）
        if abs(x) < MIDLINE_THRESHOLD:
            # 中线
            midline_channels.append(idx)
            print(f"通道 {ch_name} -> 中线 (x={x:.3f}, 使用坐标名称: {matched_name})")
        elif x < 0:
            # 左脑
            left_channels.append(idx)
            print(f"通道 {ch_name} -> 左脑 (x={x:.3f}, 使用坐标名称: {matched_name})")
        else:
            # 右脑
            right_channels.append(idx)
            print(f"通道 {ch_name} -> 右脑 (x={x:.3f}, 使用坐标名称: {matched_name})")
    
    # 中线通道同时加入左右脑
    left_channels_with_midline = sorted(set(left_channels + midline_channels))
    right_channels_with_midline = sorted(set(right_channels + midline_channels))
    
    print(f"左脑通道（含中线）: {len(left_channels_with_midline)} 个")
    print(f"右脑通道（含中线）: {len(right_channels_with_midline)} 个")
    print(f"中线通道: {len(midline_channels)} 个")
    
    return left_channels_with_midline, right_channels_with_midline, midline_channels


def split_eeg_by_hemisphere(eeg_features, left_channels, right_channels):
    """
    将EEGNet输出的160维特征分解为左右脑特征
    
    参数:
    eeg_features: (n_trials, 160) - EEGNet poolmean输出
    left_channels: 左脑通道索引列表（原始EEG通道索引）
    right_channels: 右脑通道索引列表（原始EEG通道索引）
    
    返回:
    left_feat: (n_trials, left_dim) - 左脑特征
    right_feat: (n_trials, right_dim) - 右脑特征
    """
    # EEGNet输出160维 = 8通道 × 20特征
    # poolmean输出形状: (batch, 20, 8, 1) -> 展平为 (batch, 160)
    # 展平顺序：先通道维度，再特征维度
    # 即：[ch0_feat0, ch0_feat1, ..., ch0_feat19, ch1_feat0, ..., ch7_feat19]
    # 或者：[ch0_feat0, ch1_feat0, ..., ch7_feat0, ch0_feat1, ..., ch7_feat19]
    
    # 根据EEGNet的实现，poolmean输出是(batch, 20, 8, 1)，展平为(batch, 160)
    # 展平顺序应该是: view(batch, -1)，即按行展平
    # 所以顺序是: [ch0的所有特征, ch1的所有特征, ..., ch7的所有特征]
    # 即: [ch0_feat0...ch0_feat19, ch1_feat0...ch1_feat19, ..., ch7_feat0...ch7_feat19]
    
    n_channels = 8
    n_features_per_channel = 20
    
    # 将160维重塑为(n_trials, n_channels, n_features_per_channel)
    eeg_reshaped = eeg_features.reshape(-1, n_channels, n_features_per_channel)
    
    # 映射原始通道索引到8个通道索引
    # 简化处理：假设前n_channels个原始通道对应EEGNet的8个通道
    # 如果left_channels或right_channels中的索引超出n_channels，则忽略
    
    left_channel_mask = np.zeros(n_channels, dtype=bool)
    right_channel_mask = np.zeros(n_channels, dtype=bool)
    
    for ch_idx in left_channels:
        if ch_idx < n_channels:
            left_channel_mask[ch_idx] = True
    
    for ch_idx in right_channels:
        if ch_idx < n_channels:
            right_channel_mask[ch_idx] = True
    
    # 提取左右脑特征
    left_feat = eeg_reshaped[:, left_channel_mask, :].reshape(eeg_features.shape[0], -1)
    right_feat = eeg_reshaped[:, right_channel_mask, :].reshape(eeg_features.shape[0], -1)
    
    return left_feat, right_feat


class UltraLightSpatialCorrespondenceModel(nn.Module):
    """
    超轻量级左右脑空间对应模型（端到端版本）

    参数数量：~EEGNet参数 + ~2,753（比AttentionFusionClassifier少75%）

    架构：
    1. EEGNet特征提取（端到端）
    2. EEG特征分解（左+中线，右+中线）
    3. 投影到低维（12维）
    4. 共享融合层（对侧+同侧）
    5. 左右脑交互
    6. 轻量级注意力（3路径）
    7. 直接分类
    """
    def __init__(self, eeg_channels, eeg_samples, left_channels, right_channels, hidden_dim=12, n_classes=2):
        super(UltraLightSpatialCorrespondenceModel, self).__init__()

        # EEG特征提取器（端到端可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 计算左右脑特征维度
        n_channels = 8  # EEGNet输出8个通道
        n_features_per_channel = 20  # 每个通道20个特征
        left_channel_mask = np.zeros(n_channels, dtype=bool)
        right_channel_mask = np.zeros(n_channels, dtype=bool)

        for ch_idx in left_channels:
            if ch_idx < n_channels:
                left_channel_mask[ch_idx] = True

        for ch_idx in right_channels:
            if ch_idx < n_channels:
                right_channel_mask[ch_idx] = True

        left_dim = left_channel_mask.sum() * n_features_per_channel
        right_dim = right_channel_mask.sum() * n_features_per_channel

        self.left_dim = left_dim
        self.right_dim = right_dim
        self.hidden_dim = hidden_dim
        self.left_channel_mask = left_channel_mask
        self.right_channel_mask = right_channel_mask

        # 左右脑投影（共享hidden_dim）
        self.left_proj = nn.Linear(left_dim, hidden_dim)
        self.right_proj = nn.Linear(right_dim, hidden_dim)

        # 共享融合层（所有路径共享）
        self.fusion = nn.Linear(hidden_dim + 1, hidden_dim)  # +1是音频score

        # 左右脑交互
        self.interhemispheric = nn.Linear(hidden_dim * 2, hidden_dim)

        # 轻量级注意力（3路径：对侧×2合并，同侧×2合并，交互×1）
        self.attention = nn.Linear(hidden_dim * 3, 3)

        # 直接分类（不要中间层）
        self.classifier = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, eeg_data, audio_feat):
        """
        前向传播（端到端版本）

        参数:
        eeg_data: (batch, 1, n_channels, n_timepoints) - 原始EEG数据
        audio_feat: (batch, 2) - [left_score, right_score]

        返回:
        output: (batch, n_classes)
        """
        batch_size = eeg_data.shape[0]

        # 1. 使用EEGNet提取特征（端到端）
        eeg_features = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 2. 分解为左右脑特征
        n_channels = 8
        n_features_per_channel = 20
        eeg_reshaped = eeg_features.reshape(-1, n_channels, n_features_per_channel)

        left_eeg_feat = eeg_reshaped[:, self.left_channel_mask, :].reshape(batch_size, -1)
        right_eeg_feat = eeg_reshaped[:, self.right_channel_mask, :].reshape(batch_size, -1)
        
        # 1. 左右脑投影
        left_proj = self.left_proj(left_eeg_feat)  # (batch, hidden_dim)
        right_proj = self.right_proj(right_eeg_feat)  # (batch, hidden_dim)
        
        # 2. 音频特征
        left_audio_score = audio_feat[:, 0:1]  # (batch, 1)
        right_audio_score = audio_feat[:, 1:2]  # (batch, 1)
        
        # 3. 空间对应融合（共享权重）
        # 对侧：左耳-右脑，右耳-左脑
        contralateral_LR = self.fusion(torch.cat([right_proj, left_audio_score], dim=1))  # 左耳+右脑
        contralateral_RL = self.fusion(torch.cat([left_proj, right_audio_score], dim=1))  # 右耳+左脑
        
        # 同侧：左耳-左脑，右耳-右脑
        ipsilateral_LL = self.fusion(torch.cat([left_proj, left_audio_score], dim=1))  # 左耳+左脑
        ipsilateral_RR = self.fusion(torch.cat([right_proj, right_audio_score], dim=1))  # 右耳+右脑
        
        # 合并对侧和同侧路径
        contralateral_combined = (contralateral_LR + contralateral_RL) / 2  # (batch, hidden_dim)
        ipsilateral_combined = (ipsilateral_LL + ipsilateral_RR) / 2  # (batch, hidden_dim)
        
        # 4. 左右脑交互
        interhemispheric_feat = self.interhemispheric(torch.cat([left_proj, right_proj], dim=1))  # (batch, hidden_dim)
        
        # 5. 轻量级注意力（3路径）
        attention_input = torch.cat([contralateral_combined, ipsilateral_combined, interhemispheric_feat], dim=1)  # (batch, hidden_dim*3)
        attention_weights = F.softmax(self.attention(attention_input), dim=1)  # (batch, 3)
        
        # 加权融合
        fused_feat = (
            attention_weights[:, 0:1] * contralateral_combined +
            attention_weights[:, 1:2] * ipsilateral_combined +
            attention_weights[:, 2:3] * interhemispheric_feat
        )  # (batch, hidden_dim)
        
        # 6. 直接分类
        output = self.classifier(fused_feat)
        
        return output
    
    def count_parameters(self):
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_and_evaluate(X_eeg_raw, left_channels, right_channels, X_audio, y, n_folds=5, seed=42):
    """
    训练和评估超轻量级左右脑空间对应模型（端到端版本）

    参数:
    X_eeg_raw: (n_trials, n_channels, n_timepoints) - 原始EEG数据
    left_channels: 左脑通道索引列表
    right_channels: 右脑通道索引列表
    X_audio: 音频特征
    y: 标签
    """
    print("\n=== 训练和评估超轻量级左右脑空间对应模型（端到端）===")

    # 数据标准化
    scaler_eeg = StandardScaler()  # 对整个EEG数据做标准化
    scaler_audio = StandardScaler()

    # 对EEG数据进行标准化（保持时空结构）
    n_trials, n_channels, n_timepoints = X_eeg_raw.shape
    X_eeg_reshaped = X_eeg_raw.reshape(n_trials, -1)  # (n_trials, n_channels * n_timepoints)
    X_eeg_scaled = scaler_eeg.fit_transform(X_eeg_reshaped)
    X_eeg_scaled = X_eeg_scaled.reshape(n_trials, n_channels, n_timepoints)  # 恢复原始形状

    X_audio_scaled = scaler_audio.fit_transform(X_audio)

    # 转换为tensor
    X_eeg_tensor = torch.FloatTensor(X_eeg_scaled)  # (n_trials, n_channels, n_timepoints)
    X_audio_tensor = torch.FloatTensor(X_audio_scaled)
    y_tensor = torch.LongTensor(y)

    # 使用与individual_choice_with_position.py完全一致的折划分方法
    folds, shuffled_indices = build_consistent_folds(len(y), seed=seed, k=n_folds)

    # 存储结果
    fold_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取数据维度信息
    audio_dim = X_audio_tensor.shape[1]
    print(f"EEG数据形状: {X_eeg_tensor.shape}, 音频特征维度: {audio_dim}")
    
    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")
        
        # 划分数据
        train_idx = np.setdiff1d(shuffled_indices, test_idx, assume_unique=True)

        X_eeg_train = X_eeg_tensor[train_idx]  # (batch, n_channels, n_timepoints)
        X_audio_train = X_audio_tensor[train_idx]
        y_train = y_tensor[train_idx]

        X_eeg_val = X_eeg_tensor[test_idx]  # (batch, n_channels, n_timepoints)
        X_audio_val = X_audio_tensor[test_idx]
        y_val = y_tensor[test_idx]
        
        # 创建模型（端到端版本）
        model = UltraLightSpatialCorrespondenceModel(
            eeg_channels=X_eeg_train.shape[1],  # n_channels
            eeg_samples=X_eeg_train.shape[2],   # n_timepoints
            left_channels=left_channels,
            right_channels=right_channels,
            hidden_dim=24,  # 适中隐藏层维度
            n_classes=2
        )
        model.to(device)
        
        # 打印参数数量
        if fold_idx == 1:
            n_params = model.count_parameters()
            print(f"模型参数数量: {n_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)  # 适中学习率
        
        n_epochs = 200  # 增加训练轮数
        best_val_acc = 0
        patience = 20  # 增加耐心值
        patience_counter = 0
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            # 使用原始EEG数据（端到端训练）
            X_eeg_batch = X_eeg_train.to(device)  # (batch, n_channels, n_timepoints)
            X_eeg_batch = X_eeg_batch.unsqueeze(1)  # 添加channel维度: (batch, 1, n_channels, n_timepoints)
            X_audio_batch = X_audio_train.to(device)
            y_batch = y_train.to(device)

            outputs = model(X_eeg_batch, X_audio_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, train_pred = torch.max(outputs, 1)
            train_acc = accuracy_score(y_batch.cpu().numpy(), train_pred.cpu().numpy())

            # 验证
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    # 使用原始EEG数据（端到端训练）
                    X_eeg_val_batch = X_eeg_val.to(device)  # (batch, n_channels, n_timepoints)
                    X_eeg_val_batch = X_eeg_val_batch.unsqueeze(1)  # 添加channel维度
                    X_audio_val_batch = X_audio_val.to(device)
                    y_val_batch = y_val.to(device)

                    val_outputs = model(X_eeg_val_batch, X_audio_val_batch)
                    _, val_pred = torch.max(val_outputs, 1)
                    val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())

                    print(f"    Epoch {epoch+1:3d}: Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"    早停: {patience} 个epoch内验证准确率没有提升")
                        break
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            # 使用原始EEG数据（端到端训练）
            X_eeg_val_batch = X_eeg_val.to(device)  # (batch, n_channels, n_timepoints)
            X_eeg_val_batch = X_eeg_val_batch.unsqueeze(1)  # 添加channel维度
            X_audio_val_batch = X_audio_val.to(device)
            y_val_batch = y_val.to(device)

            val_outputs = model(X_eeg_val_batch, X_audio_val_batch)
            val_probs = F.softmax(val_outputs, dim=1)
            _, val_pred = torch.max(val_outputs, 1)
            
            val_acc = accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
            val_balanced_acc = balanced_accuracy_score(y_val_batch.cpu().numpy(), val_pred.cpu().numpy())
            try:
                val_auc = roc_auc_score(y_val_batch.cpu().numpy(), val_probs[:, 1].cpu().numpy())
            except:
                val_auc = np.nan
        
        fold_results.append({
            'fold': fold_idx,
            'val_accuracy': val_acc,
            'val_balanced_accuracy': val_balanced_acc,
            'val_auc': val_auc
        })
        
        print(f"  准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}")
    
    # 计算平均结果
    results_df = pd.DataFrame(fold_results)
    mean_acc = results_df['val_accuracy'].mean()
    mean_balanced_acc = results_df['val_balanced_accuracy'].mean()
    mean_auc = results_df['val_auc'].mean()
    std_acc = results_df['val_accuracy'].std()
    std_balanced_acc = results_df['val_balanced_accuracy'].std()
    std_auc = results_df['val_auc'].std()
    
    print(f"\n平均结果:")
    print(f"  准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  平衡准确率: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")
    print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return results_df, mean_acc, mean_balanced_acc, mean_auc


def main():
    """主函数"""
    print("=" * 60)
    print("极轻量级左右脑空间对应融合模型")
    print("=" * 60)
    
    # 1. 加载EEG数据
    print("\n=== 加载EEG数据 ===")
    data_dir = Path(PROCESSED_DATA_PATH)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {PROCESSED_DATA_PATH}")
    
    trials_path, labels_path, info_path = find_data_files(data_dir)
    
    # 加载元信息
    meta = load_meta(info_path, SAMPLING_RATE, None)
    print(f"采样率: {meta.sampling_rate_hz} Hz, 通道数: {meta.num_channels}")
    
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
    print(f"原始trials形状: {trials_3d.shape}")
    
    # 获取通道名称
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    
    channel_names = info.get('channels', [])
    if not channel_names:
        channel_names = [f'EEG{i+1:03d}' for i in range(meta.num_channels)]
    
    print(f"通道数量: {len(channel_names)}")
    
    # 2. 读取电极坐标并判断左右脑
    print("\n=== 读取电极坐标并判断左右脑 ===")
    # 查找coordinates.xml文件
    coordinates_xml_path = None
    base_dirs = ["A:\\", "A:\\standard_data_interp_no_ica"]
    for base_dir in base_dirs:
        # 尝试查找hanglei相关的坐标文件
        for mff_name in ["hanglei_1_20251224_060149.mff", "zhangyufei0106_1_20260106_054423.mff"]:
            candidate = os.path.join(base_dir, mff_name, "coordinates.xml")
            if os.path.exists(candidate):
                coordinates_xml_path = candidate
                print(f"找到坐标文件: {coordinates_xml_path}")
                break
        if coordinates_xml_path:
            break
    
    if coordinates_xml_path is None:
        print("警告: 未找到coordinates.xml文件，将使用通道名称规则")
    
    left_channels, right_channels, midline_channels = load_electrode_hemisphere_mapping(
        coordinates_xml_path, channel_names
    )
    
    # 3. 数据预处理
    print("\n=== 数据预处理 ===")
    trial_duration = info.get("trial_duration")
    if trial_duration is None:
        trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz
    
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    
    # 基线矫正
    print("进行基线矫正...")
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
    print(f"提取3-5秒（刺激期）后形状: {data.shape}")
    
    # 4. 提取EEG特征（使用EEGNet）
    print("\n=== 提取EEG特征 ===")
    eeg_extractor = EEGNetFeatureExtractor(meta.num_channels, data.shape[2])
    eeg_extractor.eval()
    
    # 转换为tensor并提取特征
    data_tensor = torch.FloatTensor(data).unsqueeze(1)  # (n_trials, 1, n_channels, n_samples)
    
    with torch.no_grad():
        eeg_features = eeg_extractor(data_tensor).numpy()  # (n_trials, 160)
    
    print(f"EEG特征形状: {eeg_features.shape}")
    
    # 5. 加载一致的trials
    print("\n=== 加载一致的trials ===")
    mappings, folder_ranges = load_all_mappings()
    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    pair_names = sorted(list(all_original_names))
    
    all_trials = load_hanglei_trials(
        benchmark_files=BENCHMARK_FILES,
        mappings=mappings,
        folder_ranges=folder_ranges
    )
    
    if len(all_trials) == 0:
        print("错误: 未能加载任何trial信息")
        return
    
    consistent_trials = check_left_right_consistency(all_trials, pair_names)
    
    if len(consistent_trials) == 0:
        print("错误: 没有一致的trials")
        return
    
    print(f"一致的trials数量: {len(consistent_trials)}")
    
    # 7. 对齐EEG数据和一致的trials
    print("\n=== 对齐EEG数据和一致的trials ===")
    trial_name_to_idx = {}
    for idx, trial in enumerate(all_trials):
        pair_name = trial['original_name']
        if pair_name not in trial_name_to_idx:
            trial_name_to_idx[pair_name] = []
        trial_name_to_idx[pair_name].append(idx)
    
    consistent_eeg_indices = []
    consistent_labels = []

    for trial in consistent_trials:
        pair_name = trial['pair_name']
        matching_trial_idx = None
        for idx, all_trial in enumerate(all_trials):
            if all_trial['original_name'] == pair_name:
                matching_trial_idx = idx
                break

        if matching_trial_idx is not None and matching_trial_idx < len(data):
            consistent_eeg_indices.append(matching_trial_idx)
            consistent_labels.append(trial['choice'] - 1)  # choice: 1->0, 2->1
    
    if len(consistent_eeg_indices) == 0:
        print("错误: 未能对齐EEG数据和一致的trials")
        return
    
    # 筛选原始EEG数据（用于端到端训练）
    consistent_eeg_data = data[consistent_eeg_indices]  # 原始EEG数据 (n_trials, n_channels, n_timepoints)
    y = np.array(consistent_labels)

    print(f"对齐后的原始EEG数据形状: {consistent_eeg_data.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    print(f"对齐后的数据: {len(consistent_eeg_data)} 个trials")
    print(f"标签分布: {np.bincount(y)}")
    
    # 8. 提取音频特征
    print("\n=== 提取音频特征 ===")
    left_audio_features, right_audio_features, valid_audio_indices = extract_audio_features(
        consistent_trials, project_root
    )
    
    # 对齐音频特征：valid_audio_indices是consistent_trials中的有效索引
    # 需要映射到consistent_eeg_indices
    aligned_audio_indices = []
    aligned_eeg_indices = []
    aligned_labels = []
    
    for audio_idx in valid_audio_indices:
        if audio_idx < len(consistent_eeg_indices):
            aligned_audio_indices.append(audio_idx)
            aligned_eeg_indices.append(consistent_eeg_indices[audio_idx])
            aligned_labels.append(consistent_labels[audio_idx])
    
    if len(aligned_audio_indices) == 0:
        print("错误: 未能对齐音频特征和EEG数据")
        return
    
    # 筛选对齐后的数据
    X_left_audio = left_audio_features[aligned_audio_indices]
    X_right_audio = right_audio_features[aligned_audio_indices]
    
    # 使用IndependentEarComparisonModel提取音频特征
    ear_feature_dim = X_left_audio.shape[1]
    audio_extractor = AudioFeatureExtractor(ear_feature_dim, hidden_dim=64, dropout=0.2)
    audio_extractor.eval()
    
    with torch.no_grad():
        X_left_audio_t = torch.FloatTensor(X_left_audio)
        X_right_audio_t = torch.FloatTensor(X_right_audio)
        X_audio = audio_extractor(X_left_audio_t, X_right_audio_t).numpy()  # (n_trials, 2)
    
    print(f"音频特征形状: {X_audio.shape}")
    
    y = np.array(aligned_labels)
    
    print(f"最终对齐后的数据长度: {len(y)}")
    print(f"原始EEG数据形状: {consistent_eeg_data.shape}")
    print(f"音频特征形状: {X_audio.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 9. 训练和评估
    print("\n=== 训练和评估模型（端到端）===")
    results_df, mean_acc, mean_balanced_acc, mean_auc = train_and_evaluate(
        consistent_eeg_data, left_channels, right_channels, X_audio, y, n_folds=N_FOLDS, seed=CV_SEED
    )
    
    # 10. 保存结果
    print("\n=== 保存结果 ===")
    results_path = os.path.join(SPATIAL_OUTPUT_DIR, "spatial_correspondence_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存到: {results_path}")
    
    summary_path = os.path.join(SPATIAL_OUTPUT_DIR, "spatial_correspondence_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("极轻量级左右脑空间对应融合模型结果汇总\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据: {PROCESSED_DATA_PATH}\n")
        f.write(f"一致的trials数量: {len(consistent_trials)}\n")
        f.write(f"EEG数据形状: {consistent_eeg_data.shape} (端到端)\n")
        f.write(f"音频特征维度: {X_audio.shape[1]}\n")
        f.write(f"5折交叉验证结果:\n")
        f.write(f"  准确率: {mean_acc:.4f} ± {results_df['val_accuracy'].std():.4f}\n")
        f.write(f"  平衡准确率: {mean_balanced_acc:.4f} ± {results_df['val_balanced_accuracy'].std():.4f}\n")
        f.write(f"  AUC: {mean_auc:.4f} ± {results_df['val_auc'].std():.4f}\n")
        f.write("\n各折详细结果:\n")
        f.write(results_df.to_string(index=False))
    
    print(f"汇总已保存到: {summary_path}")
    print("\n完成！")


if __name__ == "__main__":
    main()

