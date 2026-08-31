#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合EEG和音频特征的多模态分类模型（多头注意力调制版本）
使用EEG提取的低维特征通过多头注意力机制调制音频特征

使用方法：
1. 设置BENCHMARK_FILE或CSV_FILE路径（hanglei的trial信息文件）
2. 确保数据路径正确：
   - EEG数据：A:/standard_data_interp_no_ica/hanglei
   - 音频映射文件：D:\\D\\research\\audioset下载\\audio_pairs_2s\\分组音频\\folder{1,2,3}
3. 运行脚本：python eeg_audio_fusion_attention_modulation.py

输出：
- attention_modulation_results.csv: 5折交叉验证结果
- attention_modulation_summary.txt: 结果汇总
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
from sklearn.preprocessing import StandardScaler
import warnings
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

# 数据配置
DATA_NAME = "yanxingzhuo"
SAMPLING_RATE = 250
PROCESSED_DATA_PATH = 'A:/standard_data_interp_no_ica/yanxingzhuo'

# 音频路径配置
AUDIO_BASE_DIR = DEFAULT_AUDIO_BASE_DIR
MAPPING_BASE_DIR = r"D:\D\research\audioset下载\audio_pairs_2s\分组音频"
MAPPING_FILES = [
    os.path.join(MAPPING_BASE_DIR, "folder1", "file_mapping_folder1.csv"),
    os.path.join(MAPPING_BASE_DIR, "folder2", "file_mapping_folder2.csv"),
    os.path.join(MAPPING_BASE_DIR, "folder3", "file_mapping_folder3.csv"),
]

# Benchmark文件路径
BENCHMARK_FILES = [
    r"A:\yanxingzhuo_1_20260109_053050.mff\benchmark_1_10-1-1.txt",
    r"A:\yanxingzhuo_2_20260109_060019.mff\benchmark_1_10-1-2.txt",
    r"A:\yanxingzhuo_3_20260109_062611.mff\benchmark_1_10-1-3.txt",
]

# 输出目录
OUTPUT_DIR = os.path.join(PROCESSED_DATA_PATH, "eeg_audio_fusion_attention_modulation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交叉验证配置
CV_SEED = 42
N_FOLDS = 5


class EEGNetFeatureExtractor(nn.Module):
    """EEGNet特征提取器，提取poolmean后的特征"""
    def __init__(self, n_channels: int, n_samples: int):
        super(EEGNetFeatureExtractor, self).__init__()
        self.conv_time = nn.Conv2d(1, 20, (1, 41), stride=(1, 1), bias=False)
        self.conv_ica = nn.Conv2d(1, 8, (n_channels, 1), stride=(1, 1), bias=False)
        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5)
        self.poolmean = nn.AdaptiveAvgPool2d((8, 1))
    
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
        x = self.poolmean(x)
        x = x.view(x.size(0), -1)
        return x


class EndToEndFusionClassifier(nn.Module):
    """端到端融合分类模型：EEG特征提取器 + 手工音频特征 + 分类器"""
    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super(EndToEndFusionClassifier, self).__init__()

        # EEG特征提取器（可训练）
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 融合分类器
        total_dim = 160 + audio_dim
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

        # 拼接并分类
        combined = torch.cat([eeg_feat, audio_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output


class MultiHeadAttentionModulation(nn.Module):
    """
    多头注意力调制模块
    用EEG提取的低维特征作为query，去调制（modulate）音频特征
    
    注意：这里使用简化的多头注意力，因为EEG和音频都是单个向量（不是序列）
    我们计算EEG query对音频特征的注意力权重，然后调制音频特征
    """
    def __init__(self, eeg_dim=160, audio_dim=None, proj_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.audio_dim = audio_dim
        
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        
        # EEG特征投影到低维（作为query）
        self.eeg_query_proj = nn.Linear(eeg_dim, proj_dim)
        
        # 音频特征投影（作为key和value）
        self.audio_key_proj = nn.Linear(audio_dim, proj_dim)
        self.audio_value_proj = nn.Linear(audio_dim, proj_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(proj_dim, audio_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, eeg_feat, audio_feat):
        """
        eeg_feat: (batch, eeg_dim) - EEG特征
        audio_feat: (batch, audio_dim) - 音频特征
        
        返回: (batch, audio_dim) - 调制后的音频特征
        """
        batch_size = eeg_feat.shape[0]
        
        # 投影
        Q = self.eeg_query_proj(eeg_feat)  # (batch, proj_dim)
        K = self.audio_key_proj(audio_feat)  # (batch, proj_dim)
        V = self.audio_value_proj(audio_feat)  # (batch, proj_dim)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)
        
        # 计算注意力分数：Q @ K^T / sqrt(head_dim)
        # Q: (batch, num_heads, head_dim), K: (batch, num_heads, head_dim)
        # 需要计算每个样本每个head的Q和K的点积
        # 使用einsum或直接矩阵乘法
        # Q @ K^T: (batch, num_heads, head_dim) @ (batch, num_heads, head_dim)^T
        # = (batch, num_heads, head_dim) @ (batch, num_heads, head_dim)
        # 实际上应该是: (batch, num_heads, 1, head_dim) @ (batch, num_heads, head_dim, 1)
        
        # 计算注意力分数（每个head独立计算）
        Q_expanded = Q.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
        K_expanded = K.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
        
        # 计算点积注意力分数
        attn_scores = torch.sum(Q_expanded * K_expanded, dim=-1, keepdim=True)  # (batch, num_heads, 1, 1)
        attn_scores = attn_scores * self.scale
        attn_weights = F.softmax(attn_scores.squeeze(-1), dim=-1)  # (batch, num_heads, 1)
        
        # 应用注意力权重到value
        # V: (batch, num_heads, head_dim)
        # attn_weights: (batch, num_heads, 1)
        attn_output = V * attn_weights  # (batch, num_heads, head_dim)
        
        # 合并多头
        attn_output = attn_output.view(batch_size, self.proj_dim)  # (batch, proj_dim)
        
        # 输出投影回音频维度
        modulated_audio = self.output_proj(attn_output)  # (batch, audio_dim)
        
        # 残差连接：原始音频特征 + 调制后的特征
        output = audio_feat + self.dropout(modulated_audio)
        
        return output


class AttentionModulationFusionClassifier(nn.Module):
    """
    多头注意力调制融合分类模型
    
    架构：
    1. EEGNet提取EEG特征（160维）
    2. EEG特征投影到低维（作为query）
    3. 音频特征作为key和value
    4. 多头注意力机制调制音频特征
    5. 调制后的音频特征 + EEG特征融合分类
    """
    def __init__(self, eeg_channels, eeg_samples, audio_dim, 
                 proj_dim=32, num_heads=4, hidden_dim=64, n_classes=2, dropout=0.2):
        super(AttentionModulationFusionClassifier, self).__init__()

        # EEG特征提取器
        self.eeg_extractor = EEGNetFeatureExtractor(eeg_channels, eeg_samples)

        # 多头注意力调制模块
        self.attention_modulation = MultiHeadAttentionModulation(
            eeg_dim=160,
            audio_dim=audio_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 融合分类器
        total_dim = 160 + audio_dim  # EEG特征 + 调制后的音频特征
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, eeg_data, audio_features):
        """
        eeg_data: (batch, 1, n_channels, n_timepoints)
        audio_features: (batch, audio_dim) - 手工特征
        返回: (batch, n_classes)
        """
        # 提取EEG特征
        eeg_feat = self.eeg_extractor(eeg_data)  # (batch, 160)

        # 用EEG特征调制音频特征
        modulated_audio = self.attention_modulation(eeg_feat, audio_features)  # (batch, audio_dim)

        # 融合并分类
        combined = torch.cat([eeg_feat, modulated_audio], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output


def parse_audio_mapping_custom(file_path):
    """解析音频映射文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        
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
            col1 = df.columns[0]
            col2 = df.columns[1]
            for idx, row in df.iterrows():
                original_name = str(row[col1]).strip()
                new_name = str(row[col2]).strip()
                if pd.isna(row[col1]) or pd.isna(row[col2]) or not original_name or not new_name:
                    continue
                mapping[new_name] = original_name
                if new_name.lower().endswith('.wav'):
                    mapping[new_name[:-4]] = original_name
        
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        return {}


def load_all_mappings():
    """加载所有三个folder的映射文件"""
    mappings = []
    for mapping_file in MAPPING_FILES:
        if os.path.exists(mapping_file):
            mapping = parse_audio_mapping_custom(mapping_file)
            mappings.append(mapping)
            print(f"加载映射文件: {mapping_file}, 映射数量: {len(mapping) // 2 if any(k.endswith('.wav') for k in mapping) else len(mapping)}")
        else:
            print(f"警告: 映射文件不存在: {mapping_file}")
            mappings.append({})
    
    return mappings


def load_eeg_data():
    """加载hanglei的EEG数据"""
    print("\n=== 加载EEG数据 ===")
    data_dir = Path(PROCESSED_DATA_PATH)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {PROCESSED_DATA_PATH}")
    
    trials_path, labels_path, info_path = find_data_files(data_dir)
    
    meta = load_meta(info_path, SAMPLING_RATE, None)
    print(f"采样率: {meta.sampling_rate_hz} Hz, 通道数: {meta.num_channels}")
    
    trials = np.load(trials_path)
    labels_df = pd.read_csv(labels_path)
    
    if 'Label' in labels_df.columns:
        labels = labels_df['Label'].to_numpy()
    else:
        if labels_df.shape[1] >= 2:
            labels = labels_df.iloc[:, 1].to_numpy()
        else:
            labels = labels_df.iloc[:, 0].to_numpy()
    
    trials_3d = reshape_trials(trials, meta.num_channels)
    print(f"原始trials形状: {trials_3d.shape}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    trial_duration = info.get("trial_duration")
    if trial_duration is None:
        trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz
    print(f"Trial时长: {trial_duration:.2f}秒")
    
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    
    print("进行基线矫正...")
    data_baseline_corrected = apply_baseline_correction(
        data_full,
        fs=meta.sampling_rate_hz,
        baseline_start=0.0,
        baseline_end=3.0
    )
    
    if trial_duration < 5.0:
        raise ValueError(f"数据时长{trial_duration}秒不足以提取3-5秒的刺激期")
    
    stimulus_start_seconds = 3.0
    stimulus_duration = 2.0
    stimulus_start_idx = int(round(stimulus_start_seconds * meta.sampling_rate_hz))
    stimulus_end_idx = int(round((stimulus_start_seconds + stimulus_duration) * meta.sampling_rate_hz))
    data = data_baseline_corrected[:, :, stimulus_start_idx:stimulus_end_idx]
    print(f"提取3-5秒（刺激期）后形状: {data.shape}")
    
    if data.shape[0] != labels.shape[0]:
        min_n = min(data.shape[0], labels.shape[0])
        print(f"警告: trial数量不匹配，使用前{min_n}个")
        data = data[:min_n]
        labels = labels[:min_n]
    
    return data, labels, meta


def load_hanglei_trials(benchmark_files=None, mappings=None):
    """加载hanglei的trial信息"""
    print("\n=== 加载hanglei的trial信息 ===")
    all_trials = []
    
    if benchmark_files is None or len(benchmark_files) != 3:
        raise ValueError("需要提供3个benchmark文件路径")
    
    for session_idx, benchmark_file in enumerate(benchmark_files):
        if not os.path.exists(benchmark_file):
            print(f"警告: benchmark文件不存在: {benchmark_file}")
            continue
        
        print(f"\n加载session {session_idx + 1}: {benchmark_file}")
        parsed_trials = parse_experiment_data_txt(benchmark_file)
        print(f"  解析了 {len(parsed_trials)} 个trials")
        
        mapping = mappings[session_idx] if session_idx < len(mappings) else mappings[0]
        
        for trial in parsed_trials:
            wavfile = trial.get('wavfile')
            if not wavfile:
                continue
            
            filename = os.path.basename(wavfile)
            
            if filename in mapping:
                original_name = mapping[filename]
            elif filename.replace('.wav', '') in mapping:
                original_name = mapping[filename.replace('.wav', '')]
            else:
                continue
            
            all_trials.append({
                'original_name': original_name,
                'response': trial.get('response'),
                'reaction_time': trial.get('reaction_time'),
                'session': session_idx + 1
            })
    
    print(f"\n总共加载了 {len(all_trials)} 个trials")
    return all_trials


def extract_audio_features(consistent_trials, project_root):
    """提取音频特征"""
    print("\n=== 提取音频特征 ===")
    
    feature_configs = get_feature_configs_from_linear_regression()
    print(f"使用 {len(feature_configs)} 个特征配置")
    
    pair_names_to_extract = [trial['pair_name'] for trial in consistent_trials]
    print(f"需要提取特征的音频对数量: {len(pair_names_to_extract)}")
    
    dummy_targets = np.zeros(len(pair_names_to_extract), dtype=np.float32)
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
        raise RuntimeError("无法构建FeatureTimeSeriesDataset")
    
    print(f"成功构建FeatureTimeSeriesDataset，包含 {len(base_dataset)} 个样本")
    
    stage_dataset = StageFeatureDataset(
        base_dataset,
        stage_seconds=DEFAULT_STAGE_SECONDS,
        feature_configs=feature_configs,
    )
    stage_dataset.pair_names = pair_names_to_extract
    
    print(f"阶段特征数据集大小: {len(stage_dataset)}")
    
    left_global_list = []
    right_global_list = []
    valid_indices = []
    
    pair_to_base_idx = {}
    for idx, name in enumerate(pair_names_to_extract):
        if name not in pair_to_base_idx:
            pair_to_base_idx[name] = idx
    
    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']
        
        if pair_name in pair_to_base_idx:
            base_idx = pair_to_base_idx[pair_name]
            if base_idx < len(stage_dataset.left_globals):
                left_feat = stage_dataset.left_globals[base_idx].copy()
                right_feat = stage_dataset.right_globals[base_idx].copy()
                left_global_list.append(left_feat)
                right_global_list.append(right_feat)
                valid_indices.append(i)
    
    if len(left_global_list) == 0:
        raise RuntimeError("未能提取任何有效特征")
    
    left_global_features = np.stack(left_global_list, axis=0)
    right_global_features = np.stack(right_global_list, axis=0)
    print(f"成功提取音频特征:")
    print(f"  左耳特征形状: {left_global_features.shape}")
    print(f"  右耳特征形状: {right_global_features.shape}")
    
    return left_global_features, right_global_features, valid_indices


def filter_consistent_trials(all_trials, pair_names):
    """筛选两次选择都一致的trials"""
    print("\n=== 筛选一致的trials ===")
    consistent_trials = check_left_right_consistency(all_trials, pair_names)
    return consistent_trials


def train_and_evaluate(X_eeg_raw, X_audio_manual, y,
                      eeg_channels, eeg_samples, n_folds=5, seed=42):
    """
    训练和评估两个模型：
    1. EndToEndFusionClassifier（基线）
    2. AttentionModulationFusionClassifier（多头注意力调制）
    """
    print("\n=== 训练和评估模型 ===")

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

    all_fold_results = {
        'baseline': [],
        'attention_modulation': []
    }

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

        # 训练两个模型
        models = {
            'baseline': EndToEndFusionClassifier(eeg_channels, eeg_samples, audio_dim),
            'attention_modulation': AttentionModulationFusionClassifier(
                eeg_channels, eeg_samples, audio_dim,
                proj_dim=32, num_heads=4, hidden_dim=64
            )
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
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            break

            # 最终评估
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

            all_fold_results[model_name].append({
                'fold': fold_idx,
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_balanced_acc,
                'val_auc': val_auc
            })

            print(f"  {model_name:20s} - 准确率: {val_acc:.4f}, 平衡准确率: {val_balanced_acc:.4f}, AUC: {val_auc:.4f}")

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
            'baseline': '基线模型（端到端）',
            'attention_modulation': '多头注意力调制模型'
        }.get(model_name, model_name)

        print(f"\n=== {model_display_name} 交叉验证结果汇总 ===")
        print(f"平均验证准确率: {mean_acc:.4f} ± {results_df['val_accuracy'].std():.4f}")
        print(f"平均平衡准确率: {mean_balanced_acc:.4f} ± {results_df['val_balanced_accuracy'].std():.4f}")
        print(f"平均AUC: {mean_auc:.4f} ± {results_df['val_auc'].std():.4f}")

    return all_results


def main():
    """主函数"""
    print("=" * 60)
    print("融合EEG和音频特征的多模态分类模型（多头注意力调制版本）")
    print("=" * 60)
    
    project_root = os.getcwd()
    
    # 1. 加载EEG数据
    eeg_data, labels, meta = load_eeg_data()
    
    # 2. 加载音频映射文件
    mappings = load_all_mappings()
    
    # 3. 从映射文件中提取所有音频对名称
    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    
    pair_names = sorted(list(all_original_names))
    print(f"从映射文件提取的音频对总数: {len(pair_names)}")
    
    # 4. 加载hanglei的trial信息
    all_trials = load_hanglei_trials(
        benchmark_files=BENCHMARK_FILES,
        mappings=mappings
    )
    
    if len(all_trials) == 0:
        print("\n错误: 未能加载任何trial信息")
        return
    
    # 5. 筛选一致的trials
    consistent_trials = filter_consistent_trials(all_trials, pair_names)
    
    if len(consistent_trials) == 0:
        print("\n错误: 没有一致的trials")
        return
    
    print(f"一致的trials数量: {len(consistent_trials)}")
    
    # 6. 对齐原始EEG数据和一致的trials
    consistent_eeg_data = []
    consistent_labels = []
    
    for trial in consistent_trials:
        pair_name = trial['pair_name']
        matching_trial_idx = None
        for idx, all_trial in enumerate(all_trials):
            if all_trial['original_name'] == pair_name:
                matching_trial_idx = idx
                break
        
        if matching_trial_idx is not None and matching_trial_idx < len(eeg_data):
            consistent_eeg_data.append(eeg_data[matching_trial_idx])
            consistent_labels.append(trial['choice'] - 1)
    
    if len(consistent_eeg_data) == 0:
        print("\n错误: 未能对齐EEG数据和一致的trials")
        return

    consistent_eeg_data = np.array(consistent_eeg_data)
    consistent_labels = np.array(consistent_labels)
    print(f"对齐后的数据: {len(consistent_eeg_data)} 个trials, EEG形状: {consistent_eeg_data.shape}")

    # 7. 提取音频特征
    left_audio_features, right_audio_features, valid_audio_indices = extract_audio_features(consistent_trials, project_root)

    # 拼接左右耳手工特征
    X_audio_manual = np.concatenate([left_audio_features, right_audio_features], axis=1)
    print(f"手工音频特征形状: {X_audio_manual.shape}")

    # 使用valid_indices来对齐数据
    if len(valid_audio_indices) != len(consistent_trials):
        print(f"警告: 只有 {len(valid_audio_indices)}/{len(consistent_trials)} 个trials成功提取了音频特征")
        consistent_eeg_data = consistent_eeg_data[valid_audio_indices]
        consistent_labels = consistent_labels[valid_audio_indices]
        X_audio_manual = X_audio_manual[valid_audio_indices]

    # 确保音频特征和EEG数据数量一致
    min_len = min(len(X_audio_manual), len(consistent_eeg_data))
    if len(X_audio_manual) != len(consistent_eeg_data):
        print(f"警告: 音频特征({len(X_audio_manual)})和EEG数据({len(consistent_eeg_data)})数量不一致，使用前{min_len}个")
        X_audio_manual = X_audio_manual[:min_len]
        consistent_eeg_data = consistent_eeg_data[:min_len]
        consistent_labels = consistent_labels[:min_len]

    print(f"最终数据: {len(consistent_eeg_data)} 个trials")
    print(f"EEG数据形状: {consistent_eeg_data.shape}")
    print(f"音频特征形状: {X_audio_manual.shape}")
    print(f"标签分布: {np.bincount(consistent_labels)}")

    # 8. 训练和评估
    print("\n=== 开始训练和评估模型 ===")
    all_results = train_and_evaluate(
        consistent_eeg_data,
        X_audio_manual,
        consistent_labels,
        eeg_channels=meta.num_channels,
        eeg_samples=consistent_eeg_data.shape[2],
        n_folds=N_FOLDS,
        seed=CV_SEED
    )
    
    # 9. 保存结果
    for model_name, result_dict in all_results.items():
        results_df = result_dict['results_df']
        model_name_safe = model_name.replace('_', '_')
        results_path = os.path.join(OUTPUT_DIR, f"attention_modulation_results_{model_name_safe}.csv")
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n{model_name}模型结果已保存: {results_path}")

    # 保存汇总信息
    summary_path = os.path.join(OUTPUT_DIR, "attention_modulation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多头注意力调制融合分类结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"n_trials: {len(consistent_eeg_data)}\n")
        f.write(f"eeg_feature_dim: 160 (CNN提取用于端到端)\n")
        f.write(f"audio_feature_dim (left+right): {X_audio_manual.shape[1]}\n\n")

        for model_name, result_dict in all_results.items():
            model_display_name = {
                'baseline': '基线模型（端到端）',
                'attention_modulation': '多头注意力调制模型'
            }.get(model_name, model_name)

            f.write(f"{model_display_name}:\n")
            f.write(f"  平均验证准确率: {result_dict['mean_acc']:.4f} ± {result_dict['results_df']['val_accuracy'].std():.4f}\n")
            f.write(f"  平均平衡准确率: {result_dict['mean_balanced_acc']:.4f} ± {result_dict['results_df']['val_balanced_accuracy'].std():.4f}\n")
            f.write(f"  平均AUC: {result_dict['mean_auc']:.4f} ± {result_dict['results_df']['val_auc'].std():.4f}\n\n")

    print(f"\n汇总信息已保存: {summary_path}")


if __name__ == "__main__":
    main()
