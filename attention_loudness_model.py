#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多特征注意力模型框架：支持不同特征类型，训练单耳和左右耳模型，并执行完整分析流程。
输出到 attention_all/{feature}/{model_type}/ 目录结构。
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from scipy import stats
import scipy.ndimage as ndimage
from scipy.ndimage import uniform_filter1d

# 复用已有函数
# 注意：为了与 linear_regression_selected_features.py 保持一致，从该文件导入核心数据预处理函数
from linear_regression_selected_features import (
    load_or_extract_features,
    merge_balanced_pairs_by_type,
    load_all_human_decisions,
    calculate_selection_probability_144,
)
# 其他函数仍从原文件导入
from linear_regression_loudness_models import (
    parse_stereo_pair_name,
    load_mono_5s,
    load_stereo_2s,
    extract_loudness_time_series,
)
from pair_feature_diff_extraction import extract_time_series_features

SR = 16000
HOP = 8000  # 500ms/帧 (0.5 * 16000)


# ============== 1. 特征提取函数 ==============
# 
# 支持的特征类型列表（从 pair_feature_diff_extraction.py 的 extract_time_series_features 提取）：
# 1. 'loudness' - 响度（对数压缩的梅尔能量和）
# 2. 'spectral_centroid' - 频谱质心
# 3. 'spectral_bandwidth' - 频谱带宽
# 4. 'spectral_flatness' - 频谱平坦度（帧级几何均值/算术均值）
# 5. 'spectral_irregularity' - 频谱不规则度（相邻滤波器能量差绝对和）
# 6. 'f0' - 基频（可能含NaN）
# 7. 'harmonicity' - 谐波度（HPSS能量比）
# 8. 'temporal_mod' - 时序调制（每帧系数的平均绝对delta）
# 9. 'spec_mod_centroid' - 频谱调制质心（每帧：以|MFCC|为权的系数索引质心）
#
# 在主函数的 features 列表中添加需要训练的特征类型即可

def compute_statistics_time_series(high_res_series, high_res_hop_length, target_hop_length, sr, stat_type='std'):
    """
    将高分辨率时间序列转换为统计量时间序列（每500ms窗口的mean/std）
    
    参数:
        high_res_series: 高分辨率时间序列 (T_high,)
        high_res_hop_length: 高分辨率hop_length（如256）
        target_hop_length: 目标hop_length（如8000）
        sr: 采样率
        stat_type: 统计量类型，'mean' 或 'std'
    
    返回:
        统计量时间序列 (T_target,)
    """
    high_res_series = np.asarray(high_res_series, dtype=np.float32)
    high_res_series = np.nan_to_num(high_res_series, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(high_res_series) == 0:
        return np.array([0.0], dtype=np.float32)
    
    # 计算时间窗口大小
    # 高分辨率：每帧时间 = high_res_hop_length / sr
    # 目标分辨率：每帧时间 = target_hop_length / sr（如500ms）
    high_res_frame_time = high_res_hop_length / sr  # 秒
    target_frame_time = target_hop_length / sr  # 秒（如0.5秒）
    
    # 计算每个目标帧对应的高分辨率帧数
    frames_per_target_frame = target_frame_time / high_res_frame_time  # 如 0.5 / 0.016 ≈ 31.25
    frames_per_window = max(1, int(round(frames_per_target_frame)))  # 取整为31
    
    # 计算目标帧数（基于5秒音频）
    audio_duration = len(high_res_series) * high_res_frame_time  # 秒
    target_frames = max(1, int(round(audio_duration / target_frame_time)))  # 如 5 / 0.5 = 10
    
    # 将高分辨率序列分成窗口，计算每个窗口的统计量
    stat_series = np.zeros(target_frames, dtype=np.float32)
    
    for i in range(target_frames):
        # 计算窗口的起始和结束索引
        start_idx = i * frames_per_window
        end_idx = min(start_idx + frames_per_window, len(high_res_series))
        
        if end_idx > start_idx:
            window_data = high_res_series[start_idx:end_idx]
            if stat_type == 'std':
                stat_value = np.std(window_data) if len(window_data) > 1 else 0.0
            else:  # mean
                stat_value = np.mean(window_data)
            stat_series[i] = stat_value
        else:
            # 如果窗口为空，使用最后一个值或0
            if len(high_res_series) > 0:
                stat_series[i] = high_res_series[-1] if stat_type == 'mean' else 0.0
            else:
                stat_series[i] = 0.0
    
    return stat_series


def compute_rolling_std(series, window_size=5):
    """
    计算滑动窗口标准差时间序列（保留用于向后兼容）
    
    参数:
        series: 输入时间序列 (T,)
        window_size: 滑动窗口大小
    
    返回:
        滑动std时间序列 (T,)
    """
    # 处理NaN值
    series = np.asarray(series, dtype=float)
    if series.ndim > 1:
        series = series.flatten()
    elif series.ndim == 0:
        series = np.array([series])
    series = np.nan_to_num(series, nan=0.0)
    
    if len(series) == 0:
        return np.array([0.0], dtype=np.float32)
    
    if len(series) < window_size:
        # 如果序列太短，返回全局std的重复
        global_std = np.std(series) if len(series) > 1 else 0.0
        return np.full(len(series), global_std, dtype=np.float32)
    
    # 确保window_size不超过序列长度，且至少为1
    window_size = max(1, min(window_size, len(series)))
    
    try:
        # 使用uniform_filter计算滑动方差，然后开方得到std
        # 先计算滑动均值
        # 注意：uniform_filter1d的size参数必须是整数且不超过数组长度
        # 确保输入是一维数组且是连续的
        series_contiguous = np.ascontiguousarray(series.astype(np.float64))
        rolling_mean = uniform_filter1d(series_contiguous, size=int(window_size), mode='nearest')
        # 计算滑动方差：E[(X - E[X])^2] = E[X^2] - E[X]^2
        series_sq = series_contiguous ** 2
        rolling_mean_sq = uniform_filter1d(series_sq, size=int(window_size), mode='nearest')
        rolling_var = rolling_mean_sq - rolling_mean**2
        rolling_std = np.sqrt(np.maximum(rolling_var, 0))  # 确保非负
        
        # 确保输出长度与输入一致
        if len(rolling_std) != len(series):
            # 如果长度不一致，使用全局std
            global_std = float(np.std(series)) if len(series) > 1 else 0.0
            return np.full(len(series), global_std, dtype=np.float32)
        
        return rolling_std.astype(np.float32)
    except Exception as e:
        # 如果uniform_filter失败，返回全局std
        print(f"  警告: 滑动std计算失败 (len={len(series)}, window={window_size})，使用全局std: {e}")
        global_std = float(np.std(series)) if len(series) > 1 else 0.0
        return np.full(len(series), global_std, dtype=np.float32)


def extract_feature_time_series_high_res(y, sr, feature_type, compute_std=False):
    """
    提取高分辨率特征时间序列（16ms/帧，不下采样）
    
    参数:
        y: 音频信号
        sr: 采样率
        feature_type: 特征类型
        compute_std: 是否计算滑动窗口std（而不是原始时间序列）
    
    返回:
        特征时间序列 (T,)，其中T基于hop_length=256计算（16ms/帧）
    """
    high_res_hop_length = 256  # 16ms/帧
    
    if feature_type == 'loudness':
        try:
            feat_series = extract_loudness_time_series(y, sr=sr, hop_length=high_res_hop_length)
        except Exception as e:
            print(f"  错误: extract_loudness_time_series失败: {e}")
            raise
    else:
        try:
            if feature_type in ['f0', 'harmonicity']:
                import time
                start_time = time.time()
            
            feat_dict = extract_time_series_features(y, sr=sr, hop_length=high_res_hop_length)
            
            if feature_type in ['f0', 'harmonicity']:
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    print(f"    警告: 特征 {feature_type} 提取耗时 {elapsed:.2f}秒")
            
            if feature_type in feat_dict:
                feat_series = feat_dict[feature_type]
            else:
                raise ValueError(f"不支持的特征类型: {feature_type}")
        except Exception as e:
            print(f"  错误: extract_time_series_features失败 (hop_length={high_res_hop_length}, feature={feature_type}): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 确保特征是一维数组
    feat_series = np.asarray(feat_series, dtype=np.float32)
    if feat_series.ndim > 1:
        feat_series = feat_series.flatten()
    elif feat_series.ndim == 0:
        feat_series = np.array([feat_series], dtype=np.float32)
    
    # 处理NaN和Inf
    feat_series = np.nan_to_num(feat_series, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 如果需要计算std，使用滑动窗口std
    if compute_std:
        try:
            feat_series = compute_rolling_std(feat_series, window_size=5)
        except Exception as e:
            print(f"  警告: std计算失败，使用原始特征: {e}")
    
    return feat_series.astype(np.float32)


def extract_feature_time_series(y, sr, feature_type, hop_length=8000, compute_std=False, window_size=5):
    """
    提取指定特征的时间序列
    
    参数:
        y: 音频信号
        sr: 采样率
        feature_type: 特征类型，支持以下选项：
            - 'loudness': 响度（使用专门的提取函数）
            - 'spectral_centroid': 频谱质心
            - 'spectral_bandwidth': 频谱带宽
            - 'spectral_flatness': 频谱平坦度
            - 'spectral_irregularity': 频谱不规则度
            - 'f0': 基频
            - 'harmonicity': 谐波度
            - 'temporal_mod': 时序调制
            - 'spec_mod_centroid': 频谱调制质心
        hop_length: 目标帧移（如8000，对应500ms/帧）
        compute_std: 是否计算统计量时间序列
            - True: 使用高分辨率（HOP=256）提取特征，然后转换为每500ms窗口的std统计量时间序列
            - False: 使用高分辨率提取特征，然后对齐到目标分辨率（mean）
        window_size: 滑动窗口大小（仅用于回退方法，正常情况下不使用）
    
    返回:
        特征时间序列 (T,)，其中T基于hop_length计算（如5秒音频，hop_length=8000时，T=10）
    """
    from scipy import interpolate
    
    # 先计算mel_spec以获取正确的时间维度（所有特征都应该基于这个）
    import librosa
    stft = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    magnitude = np.abs(stft)
    safe_fmax = min(12500, int(sr / 2) - 1)
    mel_filters = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=28, fmin=250, fmax=safe_fmax)
    mel_spec = np.dot(mel_filters, magnitude) + 1e-10
    mel_spec_len = mel_spec.shape[1]  # mel_spec的时间维度
    
    # 统一使用高分辨率（256）提取特征，以获得更好的统计量计算
    high_res_hop_length = 256  # 高分辨率hop_length
    
    if feature_type == 'loudness':
        # 使用extract_loudness_time_series提取响度特征（高分辨率）
        try:
            high_res_feat_series = extract_loudness_time_series(y, sr=sr, hop_length=high_res_hop_length)
        except Exception as e:
            print(f"  错误: extract_loudness_time_series失败: {e}")
            raise
    else:
        # 从extract_time_series_features提取（高分辨率）
        try:
            # 添加调试信息：某些特征（如f0）提取很慢
            if feature_type in ['f0', 'harmonicity']:
                import time
                start_time = time.time()
            
            feat_dict = extract_time_series_features(y, sr=sr, hop_length=high_res_hop_length)
            
            if feature_type in ['f0', 'harmonicity']:
                elapsed = time.time() - start_time
                if elapsed > 1.0:  # 如果超过1秒，打印警告
                    print(f"    警告: 特征 {feature_type} 提取耗时 {elapsed:.2f}秒")
            
            if feature_type in feat_dict:
                high_res_feat_series = feat_dict[feature_type]
            else:
                raise ValueError(f"不支持的特征类型: {feature_type}")
        except Exception as e:
            print(f"  错误: extract_time_series_features失败 (hop_length={high_res_hop_length}, feature={feature_type}): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 确保特征是一维数组
    high_res_feat_series = np.asarray(high_res_feat_series, dtype=np.float32)
    if high_res_feat_series.ndim > 1:
        high_res_feat_series = high_res_feat_series.flatten()
    elif high_res_feat_series.ndim == 0:
        high_res_feat_series = np.array([high_res_feat_series], dtype=np.float32)
    
    # 处理NaN和Inf
    high_res_feat_series = np.nan_to_num(high_res_feat_series, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 如果需要计算std，使用统计量时间序列方法（方案2）
    if compute_std:
        try:
            # 使用高分辨率特征，转换为每500ms窗口的std统计量时间序列
            feat_series = compute_statistics_time_series(
                high_res_feat_series, 
                high_res_hop_length=high_res_hop_length,
                target_hop_length=hop_length,
                sr=sr,
                stat_type='std'
            )
            # 确保长度与目标一致
            if len(feat_series) != mel_spec_len:
                if len(feat_series) > mel_spec_len:
                    feat_series = feat_series[:mel_spec_len].copy()
                elif len(feat_series) < mel_spec_len:
                    pad_len = mel_spec_len - len(feat_series)
                    if len(feat_series) > 0:
                        last_val = float(feat_series[-1])
                    else:
                        last_val = 0.0
                    padding = np.full(pad_len, last_val, dtype=np.float32)
                    feat_series = np.concatenate([feat_series, padding])
        except Exception as e:
            # 如果统计量计算失败，回退到原始方法
            print(f"  警告: 统计量时间序列计算失败，回退到原始方法: {e}")
            import traceback
            traceback.print_exc()
            # 回退：先对齐到目标分辨率，再计算滑动std
            feat_series = high_res_feat_series.copy()
            original_len = len(feat_series)
            # 对齐长度到mel_spec_len（使用平均池化）
            if original_len > mel_spec_len:
                if mel_spec_len > 1:
                    indices = np.linspace(0, original_len, mel_spec_len + 1)
                    indices = np.round(indices).astype(int)
                    indices = np.clip(indices, 0, original_len)
                    feat_series_resampled = np.zeros(mel_spec_len, dtype=np.float32)
                    for i in range(mel_spec_len):
                        start_idx = indices[i]
                        end_idx = indices[i + 1]
                        if end_idx > start_idx:
                            feat_series_resampled[i] = np.mean(feat_series[start_idx:end_idx])
                        else:
                            feat_series_resampled[i] = feat_series[min(start_idx, original_len - 1)]
                    feat_series = feat_series_resampled
                else:
                    feat_series = np.array([np.mean(feat_series)], dtype=np.float32)
            elif original_len < mel_spec_len:
                if original_len <= 0:
                    feat_series = np.zeros(mel_spec_len, dtype=np.float32)
                elif original_len == 1:
                    feat_series = np.full(mel_spec_len, feat_series[0], dtype=np.float32)
                else:
                    from scipy import interpolate
                    x_old = np.linspace(0, 1, original_len)
                    x_new = np.linspace(0, 1, mel_spec_len)
                    f_interp = interpolate.interp1d(x_old, feat_series, kind='linear', 
                                                     bounds_error=False, fill_value='extrapolate')
                    feat_series = f_interp(x_new).astype(np.float32)
            # 计算滑动std
            feat_series = compute_rolling_std(feat_series, window_size=window_size)
    else:
        # compute_std=False：使用mean，先对齐到目标分辨率
        feat_series = high_res_feat_series.copy()
        original_len = len(feat_series)
        
        # 对齐长度到mel_spec_len（使用平均池化）
        if original_len != mel_spec_len:
            if original_len > mel_spec_len:
                # 下采样：使用平均池化
                if mel_spec_len <= 0:
                    raise ValueError(f"mel_spec_len必须大于0: {mel_spec_len}")
                
                try:
                    # 方法1：使用uniform_filter1d进行下采样（最快）
                    if mel_spec_len > 1:
                        downsample_factor = original_len / mel_spec_len
                        kernel_size = max(1, int(downsample_factor))
                        smoothed = uniform_filter1d(feat_series.astype(np.float64), size=kernel_size, mode='nearest')
                        indices = np.linspace(0, original_len - 1, mel_spec_len).astype(int)
                        feat_series = smoothed[indices].astype(np.float32)
                    else:
                        feat_series = np.array([np.mean(feat_series)], dtype=np.float32)
                except Exception:
                    # 备用方法：平均池化
                    indices = np.linspace(0, original_len, mel_spec_len + 1)
                    indices = np.round(indices).astype(int)
                    indices = np.clip(indices, 0, original_len)
                    feat_series_resampled = np.zeros(mel_spec_len, dtype=np.float32)
                    for i in range(mel_spec_len):
                        start_idx = indices[i]
                        end_idx = indices[i + 1]
                        if end_idx > start_idx:
                            feat_series_resampled[i] = np.mean(feat_series[start_idx:end_idx])
                        else:
                            feat_series_resampled[i] = feat_series[min(start_idx, original_len - 1)]
                    feat_series = feat_series_resampled
            elif original_len < mel_spec_len:
                # 上采样：使用线性插值
                if original_len <= 0:
                    feat_series = np.zeros(mel_spec_len, dtype=np.float32)
                elif original_len == 1:
                    feat_series = np.full(mel_spec_len, feat_series[0], dtype=np.float32)
                else:
                    from scipy import interpolate
                    x_old = np.linspace(0, 1, original_len)
                    x_new = np.linspace(0, 1, mel_spec_len)
                    f_interp = interpolate.interp1d(x_old, feat_series, kind='linear', 
                                                     bounds_error=False, fill_value='extrapolate')
                    feat_series = f_interp(x_new).astype(np.float32)
    
    # 最终检查：确保长度正确
    if len(feat_series) != mel_spec_len:
        if len(feat_series) > mel_spec_len:
            feat_series = feat_series[:mel_spec_len].copy()
        elif len(feat_series) < mel_spec_len:
            pad_len = mel_spec_len - len(feat_series)
            if len(feat_series) > 0:
                last_val = float(feat_series[-1])
            else:
                last_val = 0.0
            padding = np.full(pad_len, last_val, dtype=np.float32)
            feat_series = np.concatenate([feat_series, padding])
    
    # 确保是float32类型且长度正确
    feat_series = feat_series.astype(np.float32)
    assert len(feat_series) == mel_spec_len, f"长度对齐失败: {len(feat_series)} != {mel_spec_len}"
    
    return feat_series


# ============== 2. 特征配置 ==============

def get_feature_configs_from_linear_regression():
    """
    根据linear_regression_selected_features.py使用的特征，返回特征配置
    返回: list of tuples (feature_type, compute_std)
    
    注意：如果使用ConvPoolAttentionModel，建议所有特征都使用原始时间序列（compute_std=False），
    让网络自己通过卷积和池化学习std信息。
    """
    # 从extract_significant_features_from_table()得到的7个特征（已移除f0和harmonicity）：
    # 索引2: Spectral Centroid (std) -> spectral_centroid
    # 索引7: Spectral Modulation Centroid (mean) -> spec_mod_centroid
    # 索引8: Spectral Modulation Centroid (std) -> spec_mod_centroid
    # 索引9: Spectral Irregularity (mean) -> spectral_irregularity
    # 索引10: Spectral Irregularity (std) -> spectral_irregularity
    # 索引12: Loudness (std) -> loudness
    # 索引17: Temporal Modulation (mean) -> temporal_mod
    
    # 方案1：使用原始配置（混合mean和std）
    # feature_configs = [
    #     ('spectral_centroid', True),   # 索引2: std
    #     ('spec_mod_centroid', False),  # 索引7: mean
    #     ('spec_mod_centroid', True),   # 索引8: std (重复特征类型，但需要std)
    #     ('spectral_irregularity', False),  # 索引9: mean
    #     ('spectral_irregularity', True),   # 索引10: std (重复特征类型，但需要std)
    #     ('loudness', True),            # 索引12: std
    #     ('temporal_mod', False),       # 索引17: mean
    # ]
    
    # 方案2：所有特征使用原始时间序列（推荐用于ConvPoolAttentionModel）
    # 让网络自己通过卷积和池化学习std信息
    feature_configs = [
        ('spectral_centroid', False),   # 使用原始时间序列
        ('spec_mod_centroid', False),   # 使用原始时间序列
        ('spectral_irregularity', False),  # 使用原始时间序列
        ('loudness', False),            # 使用原始时间序列
        ('temporal_mod', False),       # 使用原始时间序列
    ]
    # 注意：去掉了重复的特征（spec_mod_centroid和spectral_irregularity的std版本）
    # 因为网络可以通过卷积自己学习std
    
    return feature_configs


# ============== 3. Dataset类 ==============

class FeatureTimeSeriesDataset(Dataset):
    """多特征时间序列Dataset，支持多个特征类型，输出形状 (N, 2*n_features, T)"""
    def __init__(self, pair_names, base_dir, targets, feature_configs, sr=16000, hop_length=8000, window_size=5, use_high_res=False, stereo_path_map=None):
        """
        参数:
            pair_names: 音频对名称列表
            base_dir: 音频文件基础目录
            targets: 目标值（选择概率）
            feature_configs: 特征配置列表，每个元素为 (feature_type, compute_std)
            sr: 采样率
            hop_length: 帧移（当use_high_res=False时使用）
            window_size: 滑动窗口大小（用于std计算）
            use_high_res: 是否使用高分辨率（16ms/帧），如果True，忽略hop_length，直接使用16ms特征
        """
        self.pair_names = pair_names
        self.base_dir = base_dir
        self.targets = targets.astype(np.float32)
        self.feature_configs = feature_configs
        self.n_features = len(feature_configs)
        self.sr = sr
        self.hop_length = hop_length
        self.window_size = window_size
        self.use_high_res = use_high_res
        self.stereo_path_map = stereo_path_map or {}
        # 预提取全部样本的(2*n_features, T)
        self.X, self.y = self._build()

    def _build(self):
        # 存储每个样本的多特征数据: list of (left_features_list, right_features_list)
        # 其中left_features_list和right_features_list都是长度为n_features的列表
        multi_feat_list = []
        targets_list = []
        T_ref = None
        total = len(self.pair_names)
        print(f"  正在提取{self.n_features}个特征，共 {total} 个音频对...")
        
        for i, pair_name in enumerate(self.pair_names):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  进度: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
            try:
                stereo_path = self.stereo_path_map.get(pair_name)
                if stereo_path and os.path.exists(stereo_path):
                    yL, yR, sr = load_stereo_2s(stereo_path)
                else:
                    left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, self.base_dir)
                    left_path = os.path.join(self.base_dir, left_cat, f"{left_id}.wav")
                    right_path = os.path.join(self.base_dir, right_cat, f"{right_id}.wav")
                    if not (os.path.exists(left_path) and os.path.exists(right_path)):
                        continue

                    yL, sr = load_mono_5s(left_path)
                    yR, _ = load_mono_5s(right_path)
                
                # 提取所有特征
                left_features = []
                right_features = []
                for feat_idx, (feat_type, compute_std) in enumerate(self.feature_configs):
                    try:
                        # 添加特征提取的调试信息
                        if i == 0 and feat_idx == 0:
                            print(f"    开始提取特征: {feat_type} (compute_std={compute_std})")
                        
                        if self.use_high_res:
                            # 使用高分辨率特征（16ms/帧，不下采样）
                            feat_L = extract_feature_time_series_high_res(
                                yL, sr=sr, feature_type=feat_type, 
                                compute_std=compute_std
                            )
                            feat_R = extract_feature_time_series_high_res(
                                yR, sr=sr, feature_type=feat_type,
                                compute_std=compute_std
                            )
                        else:
                            # 使用标准分辨率特征（500ms/帧）
                            feat_L = extract_feature_time_series(
                                yL, sr=sr, feature_type=feat_type, 
                                hop_length=self.hop_length, 
                                compute_std=compute_std, 
                                window_size=self.window_size
                            )
                            feat_R = extract_feature_time_series(
                                yR, sr=sr, feature_type=feat_type,
                                hop_length=self.hop_length,
                                compute_std=compute_std,
                                window_size=self.window_size
                            )
                        
                        # 确保特征是一维数组
                        feat_L = np.asarray(feat_L).flatten()
                        feat_R = np.asarray(feat_R).flatten()
                        # 处理NaN和Inf
                        feat_L = np.nan_to_num(feat_L, nan=0.0, posinf=0.0, neginf=0.0)
                        feat_R = np.nan_to_num(feat_R, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        if len(feat_L) == 0 or len(feat_R) == 0:
                            raise ValueError(f"特征 {feat_type} 长度为0")
                        
                        if i == 0 and feat_idx == 0:
                            print(f"    特征 {feat_type} 提取完成: 长度={len(feat_L)}")
                        
                        left_features.append(feat_L)
                        right_features.append(feat_R)
                    except Exception as e:
                        print(f"  警告: 提取特征 {feat_type} (compute_std={compute_std}) 失败: {e}")
                        import traceback
                        traceback.print_exc()
                        raise  # 重新抛出异常，让外层catch处理
                
                # 确定时间长度（所有特征应该长度相同，但以防万一取最小值）
                if not left_features or not right_features:
                    raise ValueError("特征列表为空")
                
                # 找到所有特征的最小长度
                min_left_len = min(len(f) for f in left_features)
                min_right_len = min(len(f) for f in right_features)
                T_cur = min(min_left_len, min_right_len)
                
                if T_cur <= 0:
                    raise ValueError(f"特征长度为0或负数: T_cur={T_cur}, left_lens={[len(f) for f in left_features]}, right_lens={[len(f) for f in right_features]}")
                if T_ref is None:
                    T_ref = T_cur
                else:
                    T_ref = min(T_ref, T_cur)
                
                multi_feat_list.append((left_features, right_features))
                targets_list.append(self.targets[i])
            except Exception as e:
                print(f"  警告: 处理 {pair_name} 失败: {e}")
                continue
        
        print(f"  完成: 成功提取 {len(multi_feat_list)} 个音频对的特征")
        
        # 构建多特征矩阵: (N, 2*n_features, T)
        # 格式: [left_feat1, left_feat2, ..., left_featN, right_feat1, right_feat2, ..., right_featN]
        X_raw = []
        for left_features, right_features in multi_feat_list:
            # 对齐到T_ref并处理NaN
            feat_channels = []
            for feat_L, feat_R in zip(left_features, right_features):
                # 确保长度一致
                L = np.asarray(feat_L[:T_ref], dtype=np.float32).flatten()
                R = np.asarray(feat_R[:T_ref], dtype=np.float32).flatten()
                
                # 如果长度不足T_ref，用最后一个值填充
                if len(L) < T_ref:
                    L = np.pad(L, (0, T_ref - len(L)), mode='constant', constant_values=(L[-1] if len(L) > 0 else 0.0))
                if len(R) < T_ref:
                    R = np.pad(R, (0, T_ref - len(R)), mode='constant', constant_values=(R[-1] if len(R) > 0 else 0.0))
                
                # 截断到T_ref（以防万一）
                L = L[:T_ref]
                R = R[:T_ref]
                
                # 处理NaN和Inf
                L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
                R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                
                feat_channels.append(L)
                feat_channels.append(R)
            
            # 形状 (2*n_features, T)
            if len(feat_channels) == 0:
                continue  # 跳过空的特征列表
            x = np.stack(feat_channels, axis=0)
            X_raw.append(x)
        
        if len(X_raw) == 0:
            raise ValueError("没有成功提取任何特征，请检查特征提取函数和音频文件")
        
        X_raw = np.stack(X_raw, axis=0)  # (N, 2*n_features, T)
        
        # 按特征通道标准化：对每个特征通道分别计算全局统计
        X = X_raw.copy()
        for ch_idx in range(2 * self.n_features):
            ch_data = X_raw[:, ch_idx, :].flatten()  # (N*T,)
            ch_mean = np.nanmean(ch_data)
            ch_std = np.nanstd(ch_data) + 1e-8
            X[:, ch_idx, :] = (X[:, ch_idx, :] - ch_mean) / ch_std
            if ch_idx == 0 or ch_idx == self.n_features:
                print(f"  特征通道{ch_idx}标准化: mean={ch_mean:.4f} std={ch_std:.4f}")
        
        y = np.asarray(targets_list, dtype=np.float32)
        return X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_or_build_cached_dataset(feature_configs, project_root: str, base_dir: str, sr: int, hop_length: int, window_size: int = 5, use_high_res: bool = False) -> FeatureTimeSeriesDataset:
    """
    构建/缓存多特征数据集：
    - 优先从 project_root/transformer_cache/multi_feature/dataset_cache_<feature_key>.npz 读取
    - 若不存在，则按当前逻辑构建并写入同一路径，供后续脚本共享
    """
    import hashlib
    # 生成缓存key（基于特征配置和分辨率）
    res_suffix = '_highres' if use_high_res else '_lowres'
    feature_key = '_'.join([f"{ft}_{'std' if cs else 'mean'}" for ft, cs in feature_configs]) + res_suffix
    # 如果文件名太长，使用hash缩短（Windows路径限制约260字符）
    if len(feature_key) > 100:
        feature_key_hash = hashlib.md5(feature_key.encode('utf-8')).hexdigest()[:16]
        cache_filename = f'dataset_cache_{feature_key_hash}.npz'
    else:
        cache_filename = f'dataset_cache_{feature_key}.npz'
    
    cache_dir = os.path.join(project_root, 'transformer_cache', 'multi_feature')
    # 确保目录创建成功
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # 验证目录是否真的存在
        if not os.path.exists(cache_dir):
            raise OSError(f"无法创建缓存目录: {cache_dir}")
    except Exception as e:
        print(f"  错误: 创建缓存目录失败: {e}")
        raise
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"  找到缓存文件: {cache_path}")
        print(f"  文件大小: {os.path.getsize(cache_path) / (1024*1024):.2f} MB")
        data = np.load(cache_path)
        print(f"  载入缓存特征成功")
        X_raw = data['X']  # (N, 2*n_features, T)
        
        # 按特征通道标准化：对每个特征通道分别计算全局统计
        X = X_raw.copy()
        n_features = len(feature_configs)
        for ch_idx in range(2 * n_features):
            ch_data = X_raw[:, ch_idx, :].flatten()  # (N*T,)
            ch_mean = np.nanmean(ch_data)
            ch_std = np.nanstd(ch_data) + 1e-8
            X[:, ch_idx, :] = (X[:, ch_idx, :] - ch_mean) / ch_std
        
        ds = FeatureTimeSeriesDataset.__new__(FeatureTimeSeriesDataset)  # 构造不触发_build
        ds.pair_names = None
        ds.base_dir = base_dir
        ds.targets = None
        ds.feature_configs = feature_configs
        ds.n_features = n_features
        ds.sr = sr
        ds.hop_length = hop_length
        ds.window_size = window_size
        ds.use_high_res = use_high_res
        ds.X = X
        ds.y = data['y']
        return ds
    else:
        print("  未找到缓存，按音频构建并缓存...")
        # 需要 all_pair_names 与 probs（使用与 linear_regression_selected_features.py 相同的函数）
        diff_raw, diff_z, pair_names = load_or_extract_features(project_root)
        merged = merge_balanced_pairs_by_type(pair_names)
        all_pair_names = []
        for t in ['nn_main', 'nn_sub', 'main', 'sub']:
            if t in merged:
                all_pair_names.extend(merged[t]['pair_names'])
        trials = get_trials()
        probs = calculate_selection_probability_144(all_pair_names, trials).astype(np.float32)
        ds = FeatureTimeSeriesDataset(all_pair_names, base_dir, probs, feature_configs, sr=sr, hop_length=hop_length, window_size=window_size, use_high_res=use_high_res)
        
        # 保存缓存文件，添加错误处理
        try:
            # 再次确保目录存在
            os.makedirs(cache_dir, exist_ok=True)
                # 保存文件
            np.savez_compressed(cache_path, X=ds.X, y=ds.y)
                # 验证文件是否保存成功
            if not os.path.exists(cache_path):
                raise OSError(f"缓存文件保存后不存在: {cache_path}")
            print(f"  特征缓存已写入: {cache_path}")
            print(f"  文件大小: {os.path.getsize(cache_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"  警告: 保存缓存文件失败: {e}")
            print(f"  将继续运行，但下次仍需要重新提取特征")
            import traceback
            traceback.print_exc()
        
        return ds

class IndexedViewDataset(Dataset):
    """对 FeatureTimeSeriesDataset 的索引视图以做折切片"""
    def __init__(self, base: FeatureTimeSeriesDataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)
    def __len__(self):
        return self.indices.shape[0]
    def __getitem__(self, i):
        idx = self.indices[i]
        return self.base.X[idx], self.base.y[idx]

# ============== 3. 模型定义 ==============

class AdditiveTimeAttention(nn.Module):
    """时间维度上的加性注意力：score_t = v^T tanh(W h_t + b)，对t做softmax。"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        # H: (batch, T, input_dim)
        u = torch.tanh(self.proj(H))  # (batch, T, hidden_dim)
        scores = self.v(u).squeeze(-1)  # (batch, T)
        attn = torch.softmax(scores, dim=1)  # (batch, T)
        context = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # (batch, input_dim)
        return context, attn  # 返回上下文向量与注意力权重


class ConvPoolAttentionModel(nn.Module):
    """使用卷积+池化+注意力让网络自己学习std的模型"""
    def __init__(self, n_features, d_model=64, dropout=0.2):
        """
        参数:
            n_features: 特征数量（每个耳朵的特征数）
            d_model: 模型维度
        """
        super().__init__()
        self.n_features = n_features
        
        # 1. 卷积层：提取局部特征（包括局部std）
        # 使用多个不同kernel size的卷积来捕获不同尺度的模式
        self.conv1 = nn.Conv1d(2 * n_features, d_model, kernel_size=3, padding=1)  # 局部模式
        self.conv2 = nn.Conv1d(2 * n_features, d_model, kernel_size=5, padding=2)  # 稍长模式
        self.conv3 = nn.Conv1d(2 * n_features, d_model, kernel_size=7, padding=3)  # 更长模式
        
        # 2. 池化层：提取多尺度统计量
        # AdaptiveAvgPool1d和AdaptiveMaxPool1d可以提取mean和max统计量
        # 我们还可以通过卷积的输出来学习std
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均
        self.max_pool = nn.AdaptiveMaxPool1d(1)   # 全局最大
        
        # 3. 局部统计量提取：通过卷积学习局部std
        # 使用1x1卷积来学习如何组合特征以提取std
        self.local_std_conv = nn.Conv1d(d_model * 3, d_model, kernel_size=1)  # 融合多尺度特征
        
        # 4. 注意力机制：关注重要时间点
        self.attn = AdditiveTimeAttention(d_model, d_model)
        
        # 5. 输出层：融合所有信息
        # 输入：attention context (d_model) + 全局统计 (d_model*6 + n_features*4)
        # 注意：n_features在运行时确定，所以输出层维度需要在forward中动态计算
        # 或者使用一个足够大的维度，这里先设为None，在forward中动态创建
        self.out = None  # 将在forward中动态创建
        self.d_model = d_model
        self.dropout_rate = dropout
        
        # BatchNorm和Dropout层
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)
        
    def _get_output_layer(self, n_features):
        """动态创建输出层（因为n_features在运行时确定）"""
        input_dim = self.d_model + self.d_model * 6 + n_features * 4
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(input_dim, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        ).to(next(self.parameters()).device)

    def forward(self, x):
        # x: (batch, 2*n_features, T)
        batch_size, n_channels, T = x.shape
        
        # 1. 多尺度卷积提取局部特征
        h1 = F.relu(self.bn1(self.conv1(x)))  # (batch, d_model, T)
        h2 = F.relu(self.bn2(self.conv2(x)))  # (batch, d_model, T)
        h3 = F.relu(self.bn3(self.conv3(x)))  # (batch, d_model, T)
        
        # 融合多尺度特征
        h_multi = torch.cat([h1, h2, h3], dim=1)  # (batch, d_model*3, T)
        h_fused = F.relu(self.local_std_conv(h_multi))  # (batch, d_model, T)
        h_fused = self.dropout(h_fused)
        
        # 2. 池化提取全局统计量
        # 分离左右耳
        left_feat = x[:, :self.n_features, :]  # (batch, n_features, T)
        right_feat = x[:, self.n_features:, :]  # (batch, n_features, T)
        
        # 全局池化（mean和max）
        left_avg = self.avg_pool(left_feat).squeeze(-1)  # (batch, n_features)
        right_avg = self.avg_pool(right_feat).squeeze(-1)  # (batch, n_features)
        left_max = self.max_pool(left_feat).squeeze(-1)  # (batch, n_features)
        right_max = self.max_pool(right_feat).squeeze(-1)  # (batch, n_features)
        
        # 计算左右差异（保留所有特征的差异信息）
        diff_avg = left_avg - right_avg  # (batch, n_features)
        diff_max = left_max - right_max  # (batch, n_features)
        
        # 计算更多原始特征统计量
        left_std = left_feat.std(dim=2)  # (batch, n_features) - 时间维度的std
        right_std = right_feat.std(dim=2)  # (batch, n_features)
        left_min = left_feat.min(dim=2)[0]  # (batch, n_features)
        right_min = right_feat.min(dim=2)[0]  # (batch, n_features)
        
        diff_std = left_std - right_std  # (batch, n_features)
        diff_min = left_min - right_min  # (batch, n_features)
        
        # 3. 通过卷积特征计算局部std（网络自己学习）
        # 对卷积输出进行池化，提取统计量
        h1_avg = self.avg_pool(h1).squeeze(-1)  # (batch, d_model)
        h1_max = self.max_pool(h1).squeeze(-1)  # (batch, d_model)
        h2_avg = self.avg_pool(h2).squeeze(-1)
        h2_max = self.max_pool(h2).squeeze(-1)
        h3_avg = self.avg_pool(h3).squeeze(-1)
        h3_max = self.max_pool(h3).squeeze(-1)
        
        # 4. 注意力机制
        h_fused_t = h_fused.transpose(1, 2)  # (batch, T, d_model)
        context, attn = self.attn(h_fused_t)  # (batch, d_model), (batch, T)
        
        # 5. 融合所有特征
        # 改进：保留所有原始特征差异，而不是只用mean
        # 拼接：attention context + 多尺度卷积统计量 + 完整原始特征差异
        global_stats = torch.cat([
            h1_avg, h1_max, h2_avg, h2_max, h3_avg, h3_max,  # 卷积统计量 (batch, d_model*6)
            diff_avg,  # 原始特征均值差异 (batch, n_features) - 保留所有特征
            diff_std,  # 原始特征std差异 (batch, n_features) - 新增！
            diff_max,  # 原始特征最大值差异 (batch, n_features) - 保留所有特征
            diff_min,  # 原始特征最小值差异 (batch, n_features) - 新增！
        ], dim=1)  # (batch, d_model*6 + n_features*4)
        
        combined = torch.cat([context, global_stats], dim=1)  # (batch, d_model + d_model*6 + n_features*4)
        
        # 动态创建输出层（如果还没有创建）
        if self.out is None:
            self.out = self._get_output_layer(self.n_features)
        
        y = self.out(combined).squeeze(-1)  # (batch,)
        return y, attn
    
    def get_score(self, x):
        """获取左右得分用于分组"""
        # 分离左右耳
        left_ear = x[:, :self.n_features, :]  # (batch, n_features, T)
        right_ear = x[:, self.n_features:, :]  # (batch, n_features, T)
        
        # 投影
        left_ear_t = left_ear.transpose(1, 2)  # (batch, T, n_features)
        right_ear_t = right_ear.transpose(1, 2)  # (batch, T, n_features)
        left_projected = torch.tanh(self.left_proj(left_ear_t))  # (batch, T, d_model)
        right_projected = torch.tanh(self.right_proj(right_ear_t))  # (batch, T, d_model)
        left_projected = self.dropout_proj(left_projected)
        right_projected = self.dropout_proj(right_projected)
        
        # 卷积
        left_proj_conv = left_projected.transpose(1, 2)  # (batch, d_model, T)
        right_proj_conv = right_projected.transpose(1, 2)  # (batch, d_model, T)
        h_left = F.relu(self.bn_left(self.conv_left(left_proj_conv)))  # (batch, d_model, T)
        h_right = F.relu(self.bn_right(self.conv_right(right_proj_conv)))  # (batch, d_model, T)
        
        # 注意力
        h_left_t = h_left.transpose(1, 2)  # (batch, T, d_model)
        h_right_t = h_right.transpose(1, 2)  # (batch, T, d_model)
        left_context, left_attn = self.attn(h_left_t)
        right_context, right_attn = self.attn(h_right_t)
        
        # 使用context的L2范数作为得分
        left_score = torch.norm(left_context, dim=1)  # (batch,)
        right_score = torch.norm(right_context, dim=1)  # (batch,)
        return left_score, right_score


class ConvPoolOnlyModel(nn.Module):
    """使用卷积+池化+注意力，分离双耳处理，但不使用原始特征统计量的模型
    
    改进：为每个特征通道使用独立的卷积层，确保不同特征学到不同的权重
    """
    def __init__(self, n_features, d_model=64, dropout=0.2, use_feature_specific=False, use_batchnorm=True):
        """
        参数:
            n_features: 特征数量（每个耳朵的特征数）
            d_model: 模型维度
            use_feature_specific: 是否使用特征特定的卷积层（确保不同特征学到不同权重）
            use_batchnorm: 是否使用BatchNorm（False时禁用，可能有助于保持不同特征通道的差异）
        """
        super().__init__()
        self.n_features = n_features
        self.use_feature_specific = use_feature_specific
        self.use_batchnorm = use_batchnorm
        
        if use_feature_specific:
            # 改进方案：为每个特征通道使用独立的卷积层
            # 这样可以确保不同特征学到不同的权重
            self.conv_left_list = nn.ModuleList([
                nn.Conv1d(1, d_model // n_features, kernel_size=3, padding=1) 
                for _ in range(n_features)
            ])
            self.conv_right_list = nn.ModuleList([
                nn.Conv1d(1, d_model // n_features, kernel_size=3, padding=1) 
                for _ in range(n_features)
            ])
            # 如果 d_model 不能被 n_features 整除，需要额外的融合层
            if d_model % n_features != 0:
                self.fusion_left = nn.Conv1d(d_model, d_model, kernel_size=1)
                self.fusion_right = nn.Conv1d(d_model, d_model, kernel_size=1)
            else:
                self.fusion_left = None
                self.fusion_right = None
        else:
            # 原始方案：单一卷积层（可能学到相似权重）
            # 1. 卷积层：提取局部特征（只用一个小的卷积核）
            # 直接从n_features投影到d_model，不使用tanh
            self.conv_left = nn.Conv1d(n_features, d_model, kernel_size=3, padding=1)
            self.conv_right = nn.Conv1d(n_features, d_model, kernel_size=3, padding=1)
        
        # 2. 池化层：提取统计量
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 3. 注意力机制：关注重要时间点
        self.attn = AdditiveTimeAttention(d_model, d_model)
        
        # 4. 输出层：使用卷积统计量和注意力上下文
        # 输入：left_context (d_model) + right_context (d_model) + 
        #       left_conv_stats (d_model*2) + right_conv_stats (d_model*2) +
        #       diff_stats (d_model*2) = d_model*8
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 8, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(dropout)
        # BatchNorm：可选，禁用时可能有助于保持不同特征通道的差异
        if use_batchnorm:
            self.bn_left = nn.BatchNorm1d(d_model)
            self.bn_right = nn.BatchNorm1d(d_model)
        else:
            self.bn_left = None
            self.bn_right = None

    def forward(self, x):
        # x: (batch, 2*n_features, T)
        batch_size, n_channels, T = x.shape

        # 0. 分离左右耳
        left_ear = x[:, :self.n_features, :]  # (batch, n_features, T)
        right_ear = x[:, self.n_features:, :]  # (batch, n_features, T)
        
        # 1. 卷积提取局部特征
        if self.use_feature_specific:
            # 改进方案：为每个特征通道使用独立的卷积层
            left_feat_list = []
            right_feat_list = []
            for feat_idx in range(self.n_features):
                # 提取单个特征通道 (batch, 1, T)
                left_feat = left_ear[:, feat_idx:feat_idx+1, :]
                right_feat = right_ear[:, feat_idx:feat_idx+1, :]
                # 使用特征特定的卷积层
                left_conv_out = F.relu(self.conv_left_list[feat_idx](left_feat))  # (batch, d_model//n_features, T)
                right_conv_out = F.relu(self.conv_right_list[feat_idx](right_feat))  # (batch, d_model//n_features, T)
                left_feat_list.append(left_conv_out)
                right_feat_list.append(right_conv_out)
            # 拼接所有特征通道的输出
            h_left = torch.cat(left_feat_list, dim=1)  # (batch, d_model, T)
            h_right = torch.cat(right_feat_list, dim=1)  # (batch, d_model, T)
            # 如果需要融合层（当d_model不能被n_features整除时）
            if self.fusion_left is not None:
                h_left = self.fusion_left(h_left)
            if self.fusion_right is not None:
                h_right = self.fusion_right(h_right)
            # BatchNorm（可选）
            if self.use_batchnorm:
                h_left = self.bn_left(h_left)
                h_right = self.bn_right(h_right)
        else:
            # 原始方案：单一卷积层（可能学到相似权重）
            # 直接从n_features卷积到d_model，不使用tanh投影
            conv_left_out = self.conv_left(left_ear)
            conv_right_out = self.conv_right(right_ear)
            # BatchNorm（可选）
            if self.use_batchnorm:
                h_left = F.relu(self.bn_left(conv_left_out))  # (batch, d_model, T)
                h_right = F.relu(self.bn_right(conv_right_out))  # (batch, d_model, T)
            else:
                h_left = F.relu(conv_left_out)  # (batch, d_model, T)
                h_right = F.relu(conv_right_out)  # (batch, d_model, T)
        
        # 2. 通过卷积特征计算统计量（在dropout之前，保证统计量的稳定性）
        left_avg = self.avg_pool(h_left).squeeze(-1)  # (batch, d_model)
        left_max = self.max_pool(h_left).squeeze(-1)  # (batch, d_model)
        right_avg = self.avg_pool(h_right).squeeze(-1)  # (batch, d_model)
        right_max = self.max_pool(h_right).squeeze(-1)  # (batch, d_model)
        
        # 计算左右差异（重要：显式建模左右耳关系）
        diff_avg = left_avg - right_avg  # (batch, d_model)
        diff_max = left_max - right_max  # (batch, d_model)
        
        # 3. 注意力机制（对左右耳分别计算注意力）
        # 注意：对dropout后的特征计算注意力，增加正则化
        h_left_drop = self.dropout(h_left)
        h_right_drop = self.dropout(h_right)
        h_left_t = h_left_drop.transpose(1, 2)  # (batch, T, d_model)
        h_right_t = h_right_drop.transpose(1, 2)  # (batch, T, d_model)
        left_context, left_attn = self.attn(h_left_t)  # (batch, d_model), (batch, T)
        right_context, right_attn = self.attn(h_right_t)  # (batch, d_model), (batch, T)
        
        # 4. 融合特征：左右context + 左右卷积统计量 + 左右差异
        global_stats = torch.cat([
            left_context, right_context,  # 注意力上下文 (batch, d_model*2)
            left_avg, left_max,  # 左耳卷积统计量 (batch, d_model*2)
            right_avg, right_max,  # 右耳卷积统计量 (batch, d_model*2)
            diff_avg, diff_max,  # 左右差异（重要！） (batch, d_model*2)
        ], dim=1)  # (batch, d_model*8)
        
        y = self.out(global_stats).squeeze(-1)  # (batch,)
        
        # 返回平均注意力权重（用于可视化）
        attn = (left_attn + right_attn) / 2  # (batch, T)
        return y, attn
    
    def get_score(self, x):
        """获取左右得分用于分组"""
        # 分离左右耳
        left_ear = x[:, :self.n_features, :]  # (batch, n_features, T)
        right_ear = x[:, self.n_features:, :]  # (batch, n_features, T)
        
        # 卷积（与forward方法保持一致）
        if self.use_feature_specific:
            left_feat_list = []
            right_feat_list = []
            for feat_idx in range(self.n_features):
                left_feat = left_ear[:, feat_idx:feat_idx+1, :]
                right_feat = right_ear[:, feat_idx:feat_idx+1, :]
                left_conv_out = F.relu(self.conv_left_list[feat_idx](left_feat))
                right_conv_out = F.relu(self.conv_right_list[feat_idx](right_feat))
                left_feat_list.append(left_conv_out)
                right_feat_list.append(right_conv_out)
            h_left = torch.cat(left_feat_list, dim=1)
            h_right = torch.cat(right_feat_list, dim=1)
            if self.fusion_left is not None:
                h_left = self.fusion_left(h_left)
            if self.fusion_right is not None:
                h_right = self.fusion_right(h_right)
            # BatchNorm（可选）
            if self.use_batchnorm:
                h_left = self.bn_left(h_left)
                h_right = self.bn_right(h_right)
        else:
            conv_left_out = self.conv_left(left_ear)
            conv_right_out = self.conv_right(right_ear)
            # BatchNorm（可选）
            if self.use_batchnorm:
                h_left = F.relu(self.bn_left(conv_left_out))  # (batch, d_model, T)
                h_right = F.relu(self.bn_right(conv_right_out))  # (batch, d_model, T)
            else:
                h_left = F.relu(conv_left_out)  # (batch, d_model, T)
                h_right = F.relu(conv_right_out)  # (batch, d_model, T)
        
        # 注意力
        h_left_t = h_left.transpose(1, 2)  # (batch, T, d_model)
        h_right_t = h_right.transpose(1, 2)  # (batch, T, d_model)
        left_context, left_attn = self.attn(h_left_t)
        right_context, right_attn = self.attn(h_right_t)
        
        # 使用context的L2范数作为得分
        left_score = torch.norm(left_context, dim=1)  # (batch,)
        right_score = torch.norm(right_context, dim=1)  # (batch,)
        return left_score, right_score


class StatisticsOnlyModel(nn.Module):
    """只用原始特征统计量的简化模型（类似线性回归，但用神经网络）"""
    def __init__(self, n_features, hidden_dim=64, dropout=0.2):
        """
        参数:
            n_features: 特征数量（每个耳朵的特征数）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.n_features = n_features
        
        # 输入维度：4个统计量 * n_features = 4*n_features
        # (diff_mean, diff_std, diff_max, diff_min)
        input_dim = 4 * n_features
        
        # 简单的全连接网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # 用于返回注意力权重（为了兼容性，返回均匀权重）
        self.dummy_attn = None

    def forward(self, x):
        # x: (batch, 2*n_features, T)
        # 分离左右耳特征
        left_feat = x[:, :self.n_features, :]  # (batch, n_features, T)
        right_feat = x[:, self.n_features:, :]  # (batch, n_features, T)
        
        # 计算原始特征统计量（时间维度的统计）
        left_mean = left_feat.mean(dim=2)  # (batch, n_features)
        right_mean = right_feat.mean(dim=2)  # (batch, n_features)
        left_std = left_feat.std(dim=2)  # (batch, n_features)
        right_std = right_feat.std(dim=2)  # (batch, n_features)
        left_max = left_feat.max(dim=2)[0]  # (batch, n_features)
        right_max = right_feat.max(dim=2)[0]  # (batch, n_features)
        left_min = left_feat.min(dim=2)[0]  # (batch, n_features)
        right_min = right_feat.min(dim=2)[0]  # (batch, n_features)
        
        # 计算左右差异
        diff_mean = left_mean - right_mean  # (batch, n_features)
        diff_std = left_std - right_std  # (batch, n_features)
        diff_max = left_max - right_max  # (batch, n_features)
        diff_min = left_min - right_min  # (batch, n_features)
        
        # 拼接所有统计量
        features = torch.cat([diff_mean, diff_std, diff_max, diff_min], dim=1)  # (batch, 4*n_features)
        
        # 通过全连接网络
        y = self.net(features).squeeze(-1)  # (batch,)
        
        # 返回均匀注意力权重（为了兼容性）
        T = x.shape[2]
        if self.dummy_attn is None or self.dummy_attn.shape[1] != T:
            # 创建均匀注意力权重
            device = x.device
            self.dummy_attn = torch.ones(1, T, device=device) / T
        
        # 扩展到batch维度
        attn = self.dummy_attn.expand(x.shape[0], -1)  # (batch, T)
        
        return y, attn
    
    def get_score(self, x):
        """获取左右得分用于分组"""
        # 使用统计量计算得分
        left_feat = x[:, :self.n_features, :]
        right_feat = x[:, self.n_features:, :]
        
        # 计算左右耳的均值
        left_mean = left_feat.mean(dim=2).mean(dim=1)  # (batch,)
        right_mean = right_feat.mean(dim=2).mean(dim=1)  # (batch,)
        
        return left_mean, right_mean


class SimpleAttentionModel(nn.Module):
    """(2*n_features, T) → 注意力加权时间上下文 + 全局统计 → 预测概率（单耳模型）"""
    def __init__(self, time_input_dim, d_model=64, attn_hidden=64, dropout=0.2):
        """
        参数:
            time_input_dim: 时间步输入维度，应该是 2*n_features
        """
        super().__init__()
        # 先对通道做线性投影，得到每个时间步的特征
        self.feat = nn.Linear(time_input_dim, d_model)
        self.attn = AdditiveTimeAttention(d_model, attn_hidden)
        self.dropout = nn.Dropout(dropout)
        # 改进输出层：context + 全局统计特征 -> 中间层 -> 输出
        # 全局统计：mean, std, max, min差异 (4个特征) + context (d_model) = d_model + 4
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model + 4, d_model // 2),  # 中间层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, 2*n_features, T)
        x = x.transpose(1, 2)  # (batch, T, 2*n_features)
        h = self.feat(x)       # (batch, T, d_model)
        h = self.dropout(h)
        context, attn = self.attn(h)  # (batch, d_model), (batch, T)
        
        # 计算全局统计特征（对原始输入x，按时间维度）
        # 将x重新组织为左右耳特征: (batch, T, 2*n_features) -> 分离左右
        n_features = x.shape[-1] // 2
        left_feat = x[:, :, :n_features]  # (batch, T, n_features)
        right_feat = x[:, :, n_features:]  # (batch, T, n_features)
        
        # 计算左右耳的统计量
        left_mean = left_feat.mean(dim=1)  # (batch, n_features)
        right_mean = right_feat.mean(dim=1)  # (batch, n_features)
        left_std = left_feat.std(dim=1)    # (batch, n_features)
        right_std = right_feat.std(dim=1)  # (batch, n_features)
        left_max = left_feat.max(dim=1)[0] # (batch, n_features)
        right_max = right_feat.max(dim=1)[0] # (batch, n_features)
        left_min = left_feat.min(dim=1)[0] # (batch, n_features)
        right_min = right_feat.min(dim=1)[0] # (batch, n_features)
        
        # 合并为4个全局特征（左右耳的差异）
        global_feat = torch.stack([
            (left_mean - right_mean).mean(dim=1),  # 左右均值差的平均
            (left_std - right_std).mean(dim=1),    # 左右标准差差的平均
            (left_max - right_max).mean(dim=1),    # 左右最大值差的平均
            (left_min - right_min).mean(dim=1),    # 左右最小值差的平均
        ], dim=1)  # (batch, 4)
        
        # 拼接context和全局特征
        combined = torch.cat([context, global_feat], dim=1)  # (batch, d_model + 4)
        y = self.out(combined).squeeze(-1)  # (batch,)
        return y, attn
    
    def get_score(self, x):
        """获取左右得分用于分组（单耳模型需要特殊处理）"""
        # 对于多特征单耳模型，计算左右特征通道的加权和作为得分
        x = x.transpose(1, 2)  # (batch, T, 2*n_features)
        h = self.feat(x)
        context, attn = self.attn(h)
        # 分离左右特征
        n_features = x.shape[-1] // 2
        left_feat = x[:, :, :n_features]  # (batch, T, n_features)
        right_feat = x[:, :, n_features:]  # (batch, T, n_features)
        # 计算左右得分（对所有特征维度求和）
        left_score = (left_feat * attn.unsqueeze(-1)).sum(dim=(1, 2))  # (batch,)
        right_score = (right_feat * attn.unsqueeze(-1)).sum(dim=(1, 2))  # (batch,)
        return left_score, right_score


class DualStreamAttention(nn.Module):
    """左右耳独立注意力模型 + 全局统计特征，支持多特征输入"""
    def __init__(self, time_steps, n_features, dim=64, dropout=0.2):
        """
        参数:
            time_steps: 时间步数 T
            n_features: 特征数量（每个耳朵的特征数）
            dim: 投影维度
        """
        super().__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        # 左右耳分别投影：输入是n_features维，输出是dim维
        self.left_proj = nn.Linear(n_features, dim)
        self.right_proj = nn.Linear(n_features, dim)
        self.left_attention = nn.Linear(dim, 1)
        self.right_attention = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        # 改进分类器：context + 全局统计 -> 中间层 -> 输出
        # 全局统计：左右耳的mean, std, max, min差异 (4个特征) + context (dim*2) = dim*2 + 4
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2 + 4, dim),  # 中间层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        # x: (batch, 2*n_features, T)
        batch_size = x.shape[0]
        # 分离左右耳特征
        left_ear = x[:, :self.n_features, :].transpose(1, 2)  # (batch, T, n_features)
        right_ear = x[:, self.n_features:, :].transpose(1, 2)  # (batch, T, n_features)
        
        # 投影到维度 dim（逐时间步）
        left_projected = torch.tanh(self.left_proj(left_ear))   # (batch, T, dim)
        right_projected = torch.tanh(self.right_proj(right_ear))# (batch, T, dim)
        # 添加dropout
        left_projected = self.dropout(left_projected)
        right_projected = self.dropout(right_projected)
        # 计算注意力权重 (batch, T)
        left_weights = F.softmax(self.left_attention(left_projected).squeeze(-1), dim=-1)
        right_weights = F.softmax(self.right_attention(right_projected).squeeze(-1), dim=-1)
        # 加权汇总得到上下文 (batch, dim)
        left_context = torch.sum(left_projected * left_weights.unsqueeze(-1), dim=1)
        right_context = torch.sum(right_projected * right_weights.unsqueeze(-1), dim=1)
        
        # 计算全局统计特征（对原始输入x，按特征维度聚合）
        left_raw = left_ear  # (batch, T, n_features)
        right_raw = right_ear  # (batch, T, n_features)
        left_mean = left_raw.mean(dim=1).mean(dim=1)  # (batch,) - 先按时间平均，再按特征平均
        right_mean = right_raw.mean(dim=1).mean(dim=1)  # (batch,)
        left_std = left_raw.std(dim=1).mean(dim=1)  # (batch,)
        right_std = right_raw.std(dim=1).mean(dim=1)  # (batch,)
        left_max = left_raw.max(dim=1)[0].mean(dim=1)  # (batch,)
        right_max = right_raw.max(dim=1)[0].mean(dim=1)  # (batch,)
        left_min = left_raw.min(dim=1)[0].mean(dim=1)  # (batch,)
        right_min = right_raw.min(dim=1)[0].mean(dim=1)  # (batch,)
        # 合并为4个全局特征（左右耳的差异）
        global_feat = torch.stack([
            left_mean - right_mean,  # 左右均值差
            left_std - right_std,    # 左右标准差差
            left_max - right_max,    # 左右最大值差
            left_min - right_min,    # 左右最小值差
        ], dim=1)  # (batch, 4)
        
        # 合并context和全局特征
        combined = torch.cat([left_context, right_context, global_feat], dim=-1)  # (batch, 2*dim + 4)
        output = torch.sigmoid(self.classifier(combined)).squeeze(-1)  # (batch,)
        return output, left_weights, right_weights
    
    def get_score(self, x):
        """获取左右得分用于分组（简化版本：使用context的L2范数）"""
        # x: (batch, 2*n_features, T)
        left_ear = x[:, :self.n_features, :].transpose(1, 2)  # (batch, T, n_features)
        right_ear = x[:, self.n_features:, :].transpose(1, 2)  # (batch, T, n_features)
        left_proj = torch.tanh(self.left_proj(left_ear))
        right_proj = torch.tanh(self.right_proj(right_ear))
        left_weights = F.softmax(self.left_attention(left_proj).squeeze(-1), dim=-1)
        right_weights = F.softmax(self.right_attention(right_proj).squeeze(-1), dim=-1)
        left_context = torch.sum(left_proj * left_weights.unsqueeze(-1), dim=1)
        right_context = torch.sum(right_proj * right_weights.unsqueeze(-1), dim=1)
        # 简化：使用context的L2范数作为得分
        left_score = torch.norm(left_context, dim=1)  # (batch,)
        right_score = torch.norm(right_context, dim=1)  # (batch,)
        return left_score, right_score


# ============== 4. 训练函数 ==============

def train_model(model, train_dataset, val_dataset, epochs, device, output_dir, 
                lr=1e-3, weight_decay=1e-4, patience=20):
    """
    训练模型，支持早停和正则化
    
    参数:
        model: 模型
        train_dataset: 训练集
        val_dataset: 验证集（可为None，则不做早停）
        epochs: 最大训练轮数
        device: 设备
        output_dir: 输出目录
        lr: 学习率
        weight_decay: 权重衰减（L2正则化）
        patience: 早停耐心值（验证集损失不下降的轮数）
    """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False) if val_dataset is not None else None
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if isinstance(model, DualStreamAttention):
                pred, _, _ = model(xb)
            elif isinstance(model, (ConvPoolAttentionModel, ConvPoolOnlyModel, StatisticsOnlyModel, SimpleAttentionModel)):
                pred, _ = model(xb)
            else:
                pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_n += xb.size(0)
        avg_train_loss = train_loss / max(train_n, 1)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    if isinstance(model, DualStreamAttention):
                        pred, _, _ = model(xb)
                    elif isinstance(model, (ConvPoolAttentionModel, ConvPoolOnlyModel, StatisticsOnlyModel, SimpleAttentionModel)):
                        pred, _ = model(xb)
                    else:
                        pred, _ = model(xb)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item() * xb.size(0)
                    val_n += xb.size(0)
            avg_val_loss = val_loss / max(val_n, 1)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} (best={best_val_loss:.6f}, patience={patience_counter}/{patience})")
            
            # 早停
            if patience_counter >= patience:
                print(f"  早停触发（patience={patience}），恢复最佳模型（epoch {epoch - patience}）")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        else:
            # 无验证集，只打印训练损失
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.6f}")
    
    # 保存模型（如果有最佳模型状态则使用，否则使用当前状态）
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(output_dir, 'attention_model.pth'))
    print(f"  模型已保存: {os.path.join(output_dir, 'attention_model.pth')}")


def evaluate_model(model, dataset, device):
    """在给定数据集上评估 MSE/MAE/R^2"""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            if isinstance(out, tuple):
                yhat = out[0]
            else:
                yhat = out
            preds.append(yhat.detach().cpu())
            targets.append(yb.detach().cpu())
    import json
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float('nan') if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {'mse': mse, 'mae': mae, 'r2': r2}

# ============== 5. 注意力权重提取和分析 ==============

def normalize(v):
    """归一化向量"""
    v = np.clip(np.asarray(v, dtype=float), 0, None)
    s = v.sum()
    return (v / s) if s > 0 else np.zeros_like(v)


def normalized_entropy(p):
    """归一化熵"""
    p = normalize(p) + 1e-12
    H = -(p * np.log(p)).sum()
    Hmax = np.log(len(p)) if len(p) > 0 else 1.0
    return H / Hmax


def gini_coefficient(x):
    """基尼系数"""
    x = normalize(x)
    if len(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x)
    gini = 1.0 - 2.0 * np.sum(cumx) / (n * np.sum(sorted_x)) + 1.0 / n
    return max(0.0, min(1.0, gini))


def peak_ratio(x):
    """峰值比率"""
    x = np.asarray(x, dtype=float)
    m = np.mean(x) if len(x) else 1.0
    return (np.max(x) / m) if m > 0 else 0.0


def topk_share(x, k_ratio=0.05):
    """Top K占比"""
    x = normalize(x)
    n = len(x)
    k = max(1, int(round(n * k_ratio)))
    return np.sum(np.sort(x)[-k:])


def linear_trend_slope(x):
    """线性趋势斜率"""
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x))
    slope, intercept, r, p, stderr = stats.linregress(t, x)
    return slope


def front_back_share(x, seconds=1.0):
    """前后权重占比"""
    x = normalize(x)
    frames = int(round(seconds * SR / HOP))
    if frames <= 0 or frames > len(x):
        return 0.0, 0.0
    front = x[:frames].sum()
    back = x[-frames:].sum()
    return front, back


def compute_metrics(name, v):
    """计算指标"""
    metrics = {
        'series': name,
        'T': len(v),
        'entropy_norm': normalized_entropy(v),
        'gini': gini_coefficient(v),
        'peak_ratio': peak_ratio(v),
        'top5pct_share': topk_share(v, 0.05),
        'slope_per_frame': linear_trend_slope(v),
    }
    f1, b1 = front_back_share(v, 1.0)
    metrics['front1s_share'] = f1
    metrics['back1s_share'] = b1
    metrics['back_minus_front'] = b1 - f1
    return metrics


def smooth_and_save(name, v, output_dir, kernel_sizes=(5, 10, 20)):
    """多尺度平滑并保存"""
    results = {}
    plt.figure(figsize=(12, 3))
    for k in kernel_sizes:
        sigma = k / 3.0
        smoothed = ndimage.gaussian_filter1d(v, sigma=sigma)
        results[k] = smoothed
        np.save(os.path.join(output_dir, f'attention_weights_{name}_smooth_k{k}.npy'), smoothed)
        plt.plot(smoothed, label=f'Smooth Window={k} frames')
        plt.plot(v, alpha=0.3, color='gray', label='Raw (500ms)')
        plt.legend()
        plt.xlabel('Time (500ms frames)')
    plt.ylabel('Smoothed Attention Weight')
    plt.title(f'Multi-scale Smoothing ({name})')
    out_png = os.path.join(output_dir, f'attention_smoothing_{name}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    return results


def extract_attention_weights(model, dataset, device, model_type):
    """提取注意力权重（优化内存使用）"""
    model.eval()
    weights_dict = {}
    
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        if model_type == 'dual_stream':
            left_sum = None
            right_sum = None
            count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                _, lw, rw = model(xb)
                # 立即转移到CPU并累加，避免存储所有中间结果
                lw_cpu = lw.cpu().numpy().sum(axis=0)  # (T,)
                rw_cpu = rw.cpu().numpy().sum(axis=0)  # (T,)
                if left_sum is None:
                    left_sum = lw_cpu
                    right_sum = rw_cpu
                else:
                    left_sum += lw_cpu
                    right_sum += rw_cpu
                count += xb.size(0)
                # 清理GPU内存
                del xb, yb, lw, rw
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            weights_dict['left'] = left_sum / count
            weights_dict['right'] = right_sum / count
            del left_sum, right_sum
        else:  # single_attention
            single_sum = None
            count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                _, attn = model(xb)
                attn_cpu = attn.cpu().numpy().sum(axis=0)  # (T,)
                if single_sum is None:
                    single_sum = attn_cpu
                else:
                    single_sum += attn_cpu
                count += xb.size(0)
                del xb, yb, attn
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            weights_dict['single'] = single_sum / count
            del single_sum
    
    # 清理
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return weights_dict


def analyze_attention_metrics(weights_dict, output_dir, model_type):
    """分析注意力指标"""
    os.makedirs(output_dir, exist_ok=True)

    series = []
    labels = []
    
    if model_type == 'dual_stream':
        if 'left' in weights_dict:
            series.append(weights_dict['left'])
            labels.append('left')
        if 'right' in weights_dict:
            series.append(weights_dict['right'])
            labels.append('right')
    else:
        if 'single' in weights_dict:
            series.append(weights_dict['single'])
            labels.append('single')
    
    if not series:
        return
    
    # 计算原始指标
    rows = [compute_metrics(lbl, v) for lbl, v in zip(labels, series)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'attention_metrics.csv'), index=False, encoding='utf-8-sig')
    print(f"  指标已保存: attention_metrics.csv")
    
    # 绘制曲线对比
    T = len(series[0])
    # 根据模型类型和T的长度推断时间分辨率
    # conv_pool_only使用16ms/帧，其他模型使用500ms/帧
    if model_type == 'conv_pool_only':
        # 16ms/帧 = 256 samples / 16000 Hz
        hop_length_actual = 256
    else:
        # 500ms/帧 = 8000 samples / 16000 Hz
        hop_length_actual = HOP
    
    time_axis = np.arange(T) * (hop_length_actual / SR)
    print(f"  注意力权重长度: T={T}, 时间分辨率: {hop_length_actual/SR*1000:.1f}ms/帧, 总时长: {T*hop_length_actual/SR:.2f}秒")
    plt.figure(figsize=(12, 4))
    for v, lbl in zip(series, labels):
        plt.plot(time_axis, v, linewidth=2, label=lbl)
    # 分区：前1s / 中段 / 后1s
    f = int(round(1.0 * SR / hop_length_actual))
    if T > 2 * f:
        plt.axvspan(0, f * (hop_length_actual / SR), color='gray', alpha=0.05)
        plt.axvspan((T - f) * (hop_length_actual / SR), T * (hop_length_actual / SR), color='gray', alpha=0.05)
    plt.xlabel('Time (s)')
    plt.ylabel('Attention')
    plt.title('Attention Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_curve_with_bands.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 多尺度平滑和k=20指标
    smooth_rows = []
    for v, lbl in zip(series, labels):
        res = smooth_and_save(lbl, v, output_dir)
        if 20 in res:
            smooth_rows.append(compute_metrics(lbl + '_k20', res[20]))
            np.save(os.path.join(output_dir, f'attention_weights_{lbl}.npy'), v)
    
    if smooth_rows:
        df2 = pd.DataFrame(smooth_rows)
        df2.to_csv(os.path.join(output_dir, 'attention_metrics_smooth_k20.csv'), index=False, encoding='utf-8-sig')
        print(f"  平滑指标已保存: attention_metrics_smooth_k20.csv")
    
    # 对于dual_stream，绘制左右对比曲线
    if model_type == 'dual_stream' and 'left' in weights_dict and 'right' in weights_dict:
        left_weights = weights_dict['left']
        right_weights = weights_dict['right']
        
        # 1. 原始注意力权重（各自归一化，看各自趋势和绝对大小）
        plt.figure(figsize=(12, 3.2))
        plt.plot(time_axis, left_weights, color='red', linewidth=2, label='Left')
        plt.plot(time_axis, right_weights, color='blue', linewidth=2, label='Right')
        plt.xlabel('Time (s)')
        plt.ylabel('Avg Attention')
        plt.title('Average Attention over Time (Left vs Right) - Original')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_curve_left_right.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 统一归一化版本（量纲统一，看相对重要性）
        total = left_weights.sum() + right_weights.sum()
        if total > 0:
            left_norm = left_weights / total
            right_norm = right_weights / total
            plt.figure(figsize=(12, 3.2))
            plt.plot(time_axis, left_norm, color='red', linewidth=2, label='Left (unified norm)')
            plt.plot(time_axis, right_norm, color='blue', linewidth=2, label='Right (unified norm)')
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized Attention')
            plt.title('Average Attention over Time (Left vs Right) - Unified Normalization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'attention_curve_left_right_unified.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存归一化后的权重
            np.save(os.path.join(output_dir, 'attention_weights_left_unified.npy'), left_norm)
            np.save(os.path.join(output_dir, 'attention_weights_right_unified.npy'), right_norm)


# ============== 6. 分组注意力分析 ==============

def get_trials():
    """获取所有trial数据"""
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt",
    ]
    csv_files = [
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai2.csv",
        r"D:\D\research\audioset下载\experiment_output2\lironghua.csv",
        r"D:\D\research\audioset下载\experiment_output2\lironghua2.csv",
        r"D:\D\research\audioset下载\experiment_output2\mayunmiao_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\mayunmiao2.csv",
        r"D:\D\research\audioset下载\experiment_output2\ShangZiyang.csv",
        r"D:\D\research\audioset下载\experiment_output2\ShangZiyang1.csv",
        r"D:\D\research\audioset下载\experiment_output2\wjy1.csv",
        r"D:\D\research\audioset下载\experiment_output2\wjy_2.csv",
        r"D:\D\research\audioset下载\experiment_output2\LiuYaorui_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\Liu Yaorui2.csv",
    ]
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
    ] * 9
    return load_all_human_decisions(txt_files, csv_files, mapping_files)


def build_time_series_for_trial(pair_name, base_dir, feature_configs, window_size=5):
    """为单个trial构建多特征时间序列"""
    left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, base_dir)
    left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
    right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        raise FileNotFoundError(f"找不到音频: {left_path} 或 {right_path}")
    yL, sr = load_mono_5s(left_path)
    yR, _ = load_mono_5s(right_path)
    
    # 提取所有特征
    feat_channels = []
    for feat_type, compute_std in feature_configs:
        feat_L = extract_feature_time_series(
            yL, sr=sr, feature_type=feat_type, 
            hop_length=HOP, compute_std=compute_std, window_size=window_size
        )
        feat_R = extract_feature_time_series(
            yR, sr=sr, feature_type=feat_type,
            hop_length=HOP, compute_std=compute_std, window_size=window_size
        )
        T = min(len(feat_L), len(feat_R))
        feat_channels.append(feat_L[:T])
        feat_channels.append(feat_R[:T])
    
    # 堆叠所有特征通道: (2*n_features, T)
    x = np.stack(feat_channels, axis=0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    # 按通道标准化
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return x  # (2*n_features, T)


def compute_context_and_attn_dual(model, x_np, T_ref, device):
    """计算dual_stream模型的上下文和注意力（支持多特征输入）"""
    # x_np: (2*n_features, T)
    x_np = x_np[:, :T_ref]
    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, 2*n_features, T)
    # 手动复现前向，得到上下文
    # 分离左右耳特征
    n_features = model.n_features
    left_ear = xb[:, :n_features, :].transpose(1, 2)   # (1, T, n_features)
    right_ear = xb[:, n_features:, :].transpose(1, 2)  # (1, T, n_features)
    # 投影
    left_proj = torch.tanh(model.left_proj(left_ear))     # (1,T,dim)
    right_proj = torch.tanh(model.right_proj(right_ear))  # (1,T,dim)
    # 注意力权重
    left_scores = model.left_attention(left_proj).squeeze(-1)   # (1,T)
    right_scores = model.right_attention(right_proj).squeeze(-1) # (1,T)
    left_weights = torch.softmax(left_scores, dim=-1)    # (1,T)
    right_weights = torch.softmax(right_scores, dim=-1)  # (1,T)
    # 上下文
    left_context = torch.sum(left_proj * left_weights.unsqueeze(-1), dim=1)   # (1,dim)
    right_context = torch.sum(right_proj * right_weights.unsqueeze(-1), dim=1) # (1,dim)
    # 简化：使用context的L2范数作为得分（与get_score方法一致）
    left_score = torch.norm(left_context, dim=1)   # (1,)
    right_score = torch.norm(right_context, dim=1) # (1,)
    # 同时返回注意力权重
    return (
        left_context.squeeze(0).cpu().numpy(),
        right_context.squeeze(0).cpu().numpy(),
        left_weights.squeeze(0).cpu().numpy(),
        right_weights.squeeze(0).cpu().numpy(),
        left_score.item(),
        right_score.item(),
    )


def compute_context_and_attn_single(model, x_np, T_ref, device):
    """计算single_attention模型的上下文和注意力（支持多特征输入）"""
    # x_np: (2*n_features, T)
    x_np = x_np[:, :T_ref]
    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, 2*n_features, T)
    
    # 检查模型类型
    if isinstance(model, (ConvPoolOnlyModel, ConvPoolAttentionModel)):
        # ConvPoolOnlyModel 和 ConvPoolAttentionModel 使用 get_score 方法
        left_score, right_score = model.get_score(xb)
        # 需要手动计算注意力权重（与 get_score 中的逻辑一致）
        n_features = model.n_features
        left_ear = xb[:, :n_features, :]  # (1, n_features, T)
        right_ear = xb[:, n_features:, :]  # (1, n_features, T)
        
        if isinstance(model, ConvPoolOnlyModel):
            # ConvPoolOnlyModel: 卷积 -> 注意力
            h_left = F.relu(model.bn_left(model.conv_left(left_ear)))  # (1, d_model, T)
            h_right = F.relu(model.bn_right(model.conv_right(right_ear)))  # (1, d_model, T)
            h_left_t = h_left.transpose(1, 2)  # (1, T, d_model)
            h_right_t = h_right.transpose(1, 2)  # (1, T, d_model)
            left_context, left_attn = model.attn(h_left_t)  # (1, d_model), (1, T)
            right_context, right_attn = model.attn(h_right_t)  # (1, d_model), (1, T)
        else:  # ConvPoolAttentionModel
            # ConvPoolAttentionModel: 多尺度卷积 -> 融合 -> 注意力
            h1 = F.relu(model.bn1(model.conv1(xb)))
            h2 = F.relu(model.bn2(model.conv2(xb)))
            h3 = F.relu(model.bn3(model.conv3(xb)))
            h_multi = torch.cat([h1, h2, h3], dim=1)
            h_fused = F.relu(model.local_std_conv(h_multi))
            h_fused_t = h_fused.transpose(1, 2)
            left_context, left_attn = model.attn(h_fused_t)
            right_context = left_context  # ConvPoolAttentionModel 是单耳模型
            right_attn = left_attn
        
        return (
            left_context.squeeze(0).cpu().numpy(),
            right_context.squeeze(0).cpu().numpy(),
            left_attn.squeeze(0).cpu().numpy(),
            right_attn.squeeze(0).cpu().numpy(),
            left_score.item(),
            right_score.item(),
        )
    else:
        # SimpleAttentionModel 或其他单耳模型
        x = xb.transpose(1, 2)  # (1, T, 2*n_features)
        h = model.feat(x)       # (1,T,d_model)
        context, attn = model.attn(h)  # (1,d_model), (1,T)
        
        # 计算左右得分：分离左右特征，计算加权和
        n_features = x.shape[-1] // 2
        left_feat = x[:, :, :n_features]  # (1, T, n_features)
        right_feat = x[:, :, n_features:]  # (1, T, n_features)
        left_score = (left_feat * attn.unsqueeze(-1)).sum(dim=(1, 2)).item()   # 左特征加权和
        right_score = (right_feat * attn.unsqueeze(-1)).sum(dim=(1, 2)).item()  # 右特征加权和
        
        # 单耳模型只返回一条注意力，左右耳相同（都是这条注意力）
        return (
            context.squeeze(0).cpu().numpy(),
            context.squeeze(0).cpu().numpy(),  # 右上下文也用同一个
            attn.squeeze(0).cpu().numpy(),
            attn.squeeze(0).cpu().numpy(),  # 右注意力也用同一个
            left_score,
            right_score,
        )


def analyze_score_groups(model, trials, base_dir, output_dir, model_type, feature_configs, T_ref, window_size=5):
    """分析基于模型打分分组的注意力（完全按照analyze_trial_level_attention.py的逻辑）"""
    device = next(model.parameters()).device
    model.eval()
    
    # 按"模型打分左右更大"分组累积注意力（完全按照原始逻辑）
    Lb_left_list, Lb_right_list = [], []   # 左比分组：左耳/右耳
    Rb_left_list, Rb_right_list = [], []   # 右比分组：左耳/右耳
    
    with torch.no_grad():
        for idx, t in enumerate(trials):
            if (idx + 1) % 100 == 0:
                print(f"    处理trial: {idx+1}/{len(trials)}")
            try:
                x = build_time_series_for_trial(t['original_name'], base_dir, feature_configs, window_size)
                if model_type == 'dual_stream':
                    (cL, cR, wL, wR, sL, sR) = compute_context_and_attn_dual(model, x, T_ref, device)
                else:
                    (cL, cR, wL, wR, sL, sR) = compute_context_and_attn_single(model, x, T_ref, device)
                
                # 分组逻辑：完全按照原始代码
                if sL >= sR:
                    Lb_left_list.append(wL)
                    Lb_right_list.append(wR)
                else:
                    Rb_left_list.append(wL)
                    Rb_right_list.append(wR)
                
                # 定期清理
                if (idx + 1) % 500 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception:
                continue
    
    def mean_or_nan(lst):
        return np.nanmean(np.stack(lst, axis=0), axis=0) if len(lst) else np.full(T_ref, np.nan)
    
    # 保存分组注意力（完全按照原始命名）
    Lb_left = mean_or_nan(Lb_left_list)
    Lb_right = mean_or_nan(Lb_right_list)
    Rb_left = mean_or_nan(Rb_left_list)
    Rb_right = mean_or_nan(Rb_right_list)
    
    np.save(os.path.join(output_dir, 'attn_mean_scoreLeft_leftEar.npy'), Lb_left)
    np.save(os.path.join(output_dir, 'attn_mean_scoreLeft_rightEar.npy'), Lb_right)
    np.save(os.path.join(output_dir, 'attn_mean_scoreRight_leftEar.npy'), Rb_left)
    np.save(os.path.join(output_dir, 'attn_mean_scoreRight_rightEar.npy'), Rb_right)
    
    series = [Lb_left, Lb_right, Rb_left, Rb_right]
    labels = ['scoreLeft_leftEar', 'scoreLeft_rightEar', 'scoreRight_leftEar', 'scoreRight_rightEar']
    
    # 计算指标
    rows = [compute_metrics(lbl, v) for lbl, v in zip(labels, series)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'attention_score_groups_metrics.csv'), index=False, encoding='utf-8-sig')
    print(f"  分组指标已保存: attention_score_groups_metrics.csv")
    
    # 绘制曲线
    time_axis = np.arange(T_ref) * (HOP / SR)
    plt.figure(figsize=(12, 3.2))
    for v, lbl in zip(series, labels):
        plt.plot(time_axis, v, linewidth=2, label=lbl)
    plt.xlabel('Time (s)')
    plt.ylabel('Attention')
    plt.title('Attention Curves by Model Score Groups')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attn_score_groups_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # k=20平滑和指标
    rows_k20 = []
    for v, lbl in zip(series, labels):
        v20 = ndimage.gaussian_filter1d(v, sigma=20/3.0)
        rows_k20.append(compute_metrics(lbl + '_k20', v20))
        plt.figure(figsize=(12, 3))
        for k in (5, 10, 20):
            smoothed = ndimage.gaussian_filter1d(v, sigma=k/3.0)
            plt.plot(smoothed, label=f'k={k}')
        plt.plot(v, alpha=0.3, color='gray', label='raw')
        plt.legend()
        plt.xlabel('Time (500ms frames)')
        plt.ylabel('Attention')
        plt.title(f'Multi-scale Smoothing ({lbl})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attn_score_groups_smoothing_{lbl}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    df2 = pd.DataFrame(rows_k20)
    df2.to_csv(os.path.join(output_dir, 'attention_score_groups_metrics_k20.csv'), index=False, encoding='utf-8-sig')
    print(f"  分组平滑指标已保存: attention_score_groups_metrics_k20.csv")


# ============== 7. 贡献曲线分析（仅dual_stream） ==============

def per_trial_contrib(model, x_np, T_ref, device):
    """计算单个trial的贡献曲线（支持多特征输入）"""
    x_np = x_np[:, :T_ref]
    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, 2*n_features, T)

    # 分离左右耳特征到 (1,T,n_features)
    n_features = model.n_features
    left_ear = xb[:, :n_features, :].transpose(1, 2)  # (1, T, n_features)
    right_ear = xb[:, n_features:, :].transpose(1, 2)  # (1, T, n_features)

    # 投影 (1,T,dim)
    left_proj = torch.tanh(model.left_proj(left_ear))
    right_proj = torch.tanh(model.right_proj(right_ear))

    # 注意力权重 (1,T)
    left_scores = model.left_attention(left_proj).squeeze(-1)
    right_scores = model.right_attention(right_proj).squeeze(-1)
    wL = torch.softmax(left_scores, dim=-1)
    wR = torch.softmax(right_scores, dim=-1)

    # 分类器权重拆分
    W = model.classifier.weight  # (1,2*dim+4)，但前2*dim是左右context
    dim = left_proj.shape[-1]
    wL_vec = W[:, :dim].transpose(0,1)   # (dim,1)
    wR_vec = W[:, dim:2*dim].transpose(0,1)   # (dim,1)

    # 每帧线性贡献 (1,T)
    # (1,T,dim) · (dim,1) -> (1,T,1) -> squeeze -> (T,)
    left_lin = torch.matmul(left_proj, wL_vec).squeeze(-1)
    right_lin = torch.matmul(right_proj, wR_vec).squeeze(-1)
    left_contrib = (wL * left_lin).squeeze(0).cpu().numpy()
    right_contrib = (wR * right_lin).squeeze(0).cpu().numpy()

    # 总分用于分组
    left_score = left_lin.mul(wL).sum().item()
    right_score = right_lin.mul(wR).sum().item()

    # 归一化（按总绝对贡献）
    denom = np.sum(np.abs(left_contrib)) + np.sum(np.abs(right_contrib))
    if denom <= 0:
        left_norm = np.zeros_like(left_contrib)
        right_norm = np.zeros_like(right_contrib)
    else:
        left_norm = left_contrib / denom
        right_norm = right_contrib / denom

    return left_contrib, right_contrib, left_norm, right_norm, left_score, right_score


def analyze_contribution(model, trials, base_dir, output_dir, feature_configs, T_ref, window_size=5):
    """分析贡献曲线（仅dual_stream模型，完全按照analyze_trial_contrib_attention.py的逻辑）"""
    device = next(model.parameters()).device
    model.eval()
    
    # 累积四个分组（raw与norm）- 完全按照原始代码
    SL_L_raw, SL_R_raw, SR_L_raw, SR_R_raw = [], [], [], []
    SL_L_norm, SL_R_norm, SR_L_norm, SR_R_norm = [], [], [], []
    
    with torch.no_grad():
        for idx, t in enumerate(trials):
            if (idx + 1) % 100 == 0:
                print(f"    处理trial: {idx+1}/{len(trials)}")
            try:
                x = build_time_series_for_trial(t['original_name'], base_dir, feature_configs, window_size)
                L_raw, R_raw, L_norm, R_norm, sL, sR = per_trial_contrib(model, x, T_ref, device)
                # 分组逻辑：完全按照原始代码
                if sL >= sR:
                    SL_L_raw.append(L_raw)
                    SL_R_raw.append(R_raw)
                    SL_L_norm.append(L_norm)
                    SL_R_norm.append(R_norm)
                else:
                    SR_L_raw.append(L_raw)
                    SR_R_raw.append(R_raw)
                    SR_L_norm.append(L_norm)
                    SR_R_norm.append(R_norm)
                
                # 定期清理
                if (idx + 1) % 500 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception:
                continue
    
    def mean_list(lst):
        return np.nanmean(np.stack(lst, axis=0), axis=0) if lst else np.full(T_ref, np.nan)
    
    # 组平均（完全按照原始代码）
    sl_l_raw = mean_list(SL_L_raw)
    sl_r_raw = mean_list(SL_R_raw)
    sr_l_raw = mean_list(SR_L_raw)
    sr_r_raw = mean_list(SR_R_raw)
    sl_l_norm = mean_list(SL_L_norm)
    sl_r_norm = mean_list(SL_R_norm)
    sr_l_norm = mean_list(SR_L_norm)
    sr_r_norm = mean_list(SR_R_norm)
    
    # 保存npy（完全按照原始命名）
    np.save(os.path.join(output_dir, 'contrib_scoreLeft_left_raw.npy'), sl_l_raw)
    np.save(os.path.join(output_dir, 'contrib_scoreLeft_right_raw.npy'), sl_r_raw)
    np.save(os.path.join(output_dir, 'contrib_scoreRight_left_raw.npy'), sr_l_raw)
    np.save(os.path.join(output_dir, 'contrib_scoreRight_right_raw.npy'), sr_r_raw)
    np.save(os.path.join(output_dir, 'contrib_scoreLeft_left_norm.npy'), sl_l_norm)
    np.save(os.path.join(output_dir, 'contrib_scoreLeft_right_norm.npy'), sl_r_norm)
    np.save(os.path.join(output_dir, 'contrib_scoreRight_left_norm.npy'), sr_l_norm)
    np.save(os.path.join(output_dir, 'contrib_scoreRight_right_norm.npy'), sr_r_norm)
    
    # 计算指标（归一化版本）
    def contrib_metrics_row(name, v):
        f1, b1 = front_back_share(v, 1.0)
        return {
            'series': name,
            'T': len(v),
            'entropy_norm': normalized_entropy(np.abs(v)),
            'gini': gini_coefficient(np.abs(v)),
            'peak_ratio': peak_ratio(np.abs(v)),
            'top5pct_share': topk_share(np.abs(v), 0.05),
            'slope_per_frame': linear_trend_slope(v),
            'front1s_share(|v|)': f1,
            'back1s_share(|v|)': b1,
            'back_minus_front(|v|)': b1 - f1,
        }
    
    series_norm = [sl_l_norm, sl_r_norm, sr_l_norm, sr_r_norm]
    labels_norm = ['scoreLeft_left_norm', 'scoreLeft_right_norm', 'scoreRight_left_norm', 'scoreRight_right_norm']
    series_k20_norm = [ndimage.gaussian_filter1d(v, sigma=20/3.0) for v in series_norm]
    rows_norm = [contrib_metrics_row(lbl + '_k20', v) for lbl, v in zip(labels_norm, series_k20_norm)]
    pd.DataFrame(rows_norm).to_csv(os.path.join(output_dir, 'contrib_metrics_k20_norm.csv'), index=False, encoding='utf-8-sig')
    
    series_raw = [sl_l_raw, sl_r_raw, sr_l_raw, sr_r_raw]
    labels_raw = ['scoreLeft_left_raw', 'scoreLeft_right_raw', 'scoreRight_left_raw', 'scoreRight_right_raw']
    series_k20_raw = [ndimage.gaussian_filter1d(v, sigma=20/3.0) for v in series_raw]
    rows_raw = [contrib_metrics_row(lbl + '_k20', v) for lbl, v in zip(labels_raw, series_k20_raw)]
    pd.DataFrame(rows_raw).to_csv(os.path.join(output_dir, 'contrib_metrics_k20_raw.csv'), index=False, encoding='utf-8-sig')
    
    # 绘制曲线
    time_axis = np.arange(T_ref) * (HOP / SR)
    def plot_contrib_curves(a, b, c, d, title, fname):
        plt.figure(figsize=(12, 3.2))
        plt.plot(time_axis, a, label='Score-Left / LeftEar', color='red')
        plt.plot(time_axis, b, label='Score-Left / RightEar', color='orange')
        plt.plot(time_axis, c, label='Score-Right / LeftEar', color='blue')
        plt.plot(time_axis, d, label='Score-Right / RightEar', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Contribution (k=20 smoothed)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()
    
    plot_contrib_curves(series_k20_norm[0], series_k20_norm[1], series_k20_norm[2], series_k20_norm[3],
                       'Contribution Curves (Normalized, k=20)', 'contrib_curves_k20_norm.png')
    plot_contrib_curves(series_k20_raw[0], series_k20_raw[1], series_k20_raw[2], series_k20_raw[3],
                       'Contribution Curves (Raw, k=20)', 'contrib_curves_k20_raw.png')
    
    print(f"  贡献分析完成: contrib_metrics_k20_norm.csv, contrib_metrics_k20_raw.csv")


# ============== 8. 断点续传检查 ==============

def check_task_completed(output_dir, model_type):
    """
    检查任务是否已完成
    
    返回:
        (is_completed, missing_files): 是否完成，缺失的文件列表
    """
    required_files = []
    
    # 所有模型类型都需要
    required_files.append('attention_model.pth')
    required_files.append('attention_metrics.csv')
    required_files.append('attention_metrics_smooth_k20.csv')
    required_files.append('attention_curve_with_bands.png')
    
    if model_type == 'dual_stream':
        required_files.extend([
            'attention_weights_left.npy',
            'attention_weights_right.npy',
            'attention_curve_left_right.png',
            'attention_curve_left_right_unified.png',
            'attn_mean_scoreLeft_leftEar.npy',
            'attn_mean_scoreLeft_rightEar.npy',
            'attn_mean_scoreRight_leftEar.npy',
            'attn_mean_scoreRight_rightEar.npy',
            'attention_score_groups_metrics.csv',
            'attention_score_groups_metrics_k20.csv',
            'attn_score_groups_curves.png',
            'contrib_scoreLeft_left_norm.npy',
            'contrib_scoreLeft_right_norm.npy',
            'contrib_scoreRight_left_norm.npy',
            'contrib_scoreRight_right_norm.npy',
            'contrib_metrics_k20_norm.csv',
            'contrib_curves_k20_norm.png',
        ])
    else:  # single_attention
        required_files.extend([
            'attention_weights_single.npy',
            'attn_mean_scoreLeft.npy',
            'attn_mean_scoreRight.npy',
            'attention_score_groups_metrics.csv',
            'attention_score_groups_metrics_k20.csv',
            'attn_score_groups_curves.png',
        ])
    
    missing_files = []
    for fname in required_files:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            missing_files.append(fname)
    
    is_completed = len(missing_files) == 0
    return is_completed, missing_files


# ============== 9. 主函数 ==============

def main():
    print("=== 多特征注意力模型框架 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.getcwd()
    
    # 设置音频文件基础目录（需要根据实际情况修改）
    # 实际文件在: D:\D\research\audioset下载\clap_select
    # 当前工作目录是: D:\D\research\audioset下载\download_audioset-master\download_audioset-master
    base_dir = r"D:\D\research\audioset下载\clap_select"
    
    # 如果上面的路径不存在，尝试其他可能的路径
    if not os.path.exists(base_dir):
        possible_dirs = [
            r"D:\D\research\audioset下载\clap_select",  # 绝对路径
            os.path.join(project_root, '..', '..', 'clap_select'),  # 向上两级
            os.path.join(project_root, '..', 'clap_select'),  # 向上一级
            os.path.join(project_root, 'clap_select'),  # 当前目录
            os.path.join(project_root, '音频配对合成_新版', '分组音频', 'folder1'),
            os.path.join(project_root, '音频配对合成_新版', '分组音频', 'folder2'),
        ]
        for pd in possible_dirs:
            pd_normalized = os.path.normpath(pd)
            if os.path.exists(pd_normalized):
                base_dir = pd_normalized
                break
        else:
            print(f"警告: 未找到音频文件目录，请检查base_dir设置")
            print(f"  当前base_dir: {base_dir}")
            print(f"  尝试的路径:")
            for pd in possible_dirs:
                print(f"    - {os.path.normpath(pd)}")
    
    print(f"\n音频文件基础目录: {base_dir}")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"音频文件目录不存在: {base_dir}")
    
    # ========== 定义特征配置 ==========
    # 使用与linear_regression_selected_features.py相同的9个特征
    feature_configs = get_feature_configs_from_linear_regression()
    print(f"\n使用{len(feature_configs)}个特征:")
    for i, (feat_type, compute_std) in enumerate(feature_configs):
        print(f"  {i+1}. {feat_type} ({'std' if compute_std else 'mean'})")
    
    # 加载数据
    print("\n步骤1: 加载数据与目标")
    diff_raw, diff_z, pair_names = load_or_extract_features(project_root)
    merged = merge_balanced_pairs_by_type(pair_names)
    
    all_pair_names = []
    for t in ['nn_main', 'nn_sub', 'main', 'sub']:
        if t in merged:
            all_pair_names.extend(merged[t]['pair_names'])
    
    all_trials = get_trials()
    probs = calculate_selection_probability_144(all_pair_names, all_trials).astype(np.float32)
    
    print(f"\n{'='*60}")
    print(f"处理多特征数据集")
    print(f"{'='*60}")
    
    # 构建数据集（为不同模型类型准备不同的数据集）
    print(f"\n步骤2: 构建多特征数据集")
    window_size = 5
    
    # 为conv_pool_only模型准备高分辨率数据集（16ms）
    print(f"  构建高分辨率数据集（16ms/帧）...")
    dataset_high_res = load_or_build_cached_dataset(feature_configs, project_root, base_dir, sr=SR, hop_length=HOP, window_size=window_size, use_high_res=True)
    
    # 为其他模型准备标准分辨率数据集（500ms）
    print(f"  构建标准分辨率数据集（500ms/帧）...")
    dataset = load_or_build_cached_dataset(feature_configs, project_root, base_dir, sr=SR, hop_length=HOP, window_size=window_size, use_high_res=False)
    
    if len(dataset) == 0:
        print(f"  警告: 数据集为空，退出")
        return
    
    Xshape = dataset[0][0].shape
    T = Xshape[1]
    n_features = len(feature_configs)
    print(f"  标准分辨率单样本形状: {Xshape}, T={T}, n_features={n_features}")
    
    if len(dataset_high_res) > 0:
        Xshape_high = dataset_high_res[0][0].shape
        T_high = Xshape_high[1]
        print(f"  高分辨率单样本形状: {Xshape_high}, T={T_high}, n_features={n_features}")
    
    # 5折划分（与其它脚本一致）：固定随机种子，打乱后 array_split
    num_pairs = len(dataset)
    CV_SEED = 42  # 与其他脚本保持一致
    rng_indices = np.random.RandomState(CV_SEED).permutation(num_pairs)
    folds = np.array_split(rng_indices, 5)
    
    # 在训练集中划分：72%训练，8%验证（训练集总共80%，所以72%/80%=90%，8%/80%=10%）
    train_val_split_ratio = 0.9  # 训练集占训练+验证的90%，即72%/(72%+8%)=90%

    # 对每个模型类型与每折进行训练与测试（在测试集上评估与提取注意力）
    model_types = ['single_attention', 'dual_stream', 'conv_pool_attention', 'conv_pool_only', 'statistics_only']
    for model_type in model_types:
        print(f"\n  --- 模型类型: {model_type} ---")
        for fold_idx, test_idx in enumerate(folds, start=1):
            print(f"    -> 折 {fold_idx}/5")
            
            fold_out = os.path.join(project_root, 'attention_5fold', 'multi_feature', model_type, f'fold_{fold_idx}')
            os.makedirs(fold_out, exist_ok=True)
            
            # 选择数据集（conv_pool_only使用高分辨率数据集）
            if model_type == 'conv_pool_only':
                current_dataset = dataset_high_res
                # 需要重新划分高分辨率数据集（使用相同的随机种子和划分方式）
                num_pairs_high = len(dataset_high_res)
                rng_indices_high = np.random.RandomState(CV_SEED).permutation(num_pairs_high)
                folds_high = np.array_split(rng_indices_high, 5)
                test_idx_high = folds_high[fold_idx - 1]
                train_idx_high = np.setdiff1d(rng_indices_high, test_idx_high, assume_unique=True)
            else:
                current_dataset = dataset
                train_idx = np.setdiff1d(rng_indices, test_idx, assume_unique=True)
                test_idx_high = None  # 不使用高分辨率
            
            # 在训练集中划分训练集和验证集（72%训练，8%验证）
            if model_type == 'conv_pool_only':
                train_size_high = len(train_idx_high)
                train_size_final_high = int(train_size_high * train_val_split_ratio)
                val_size_high = train_size_high - train_size_final_high
                # 随机划分（固定随机种子保证可复现）
                train_val_indices_high = np.random.RandomState(CV_SEED + fold_idx).permutation(train_size_high)
                train_indices_final_high = train_idx_high[train_val_indices_high[:train_size_final_high]]
                val_indices_final_high = train_idx_high[train_val_indices_high[train_size_final_high:]]
                train_ds_final = IndexedViewDataset(current_dataset, train_indices_final_high)
                val_ds_final = IndexedViewDataset(current_dataset, val_indices_final_high)
                test_ds = IndexedViewDataset(current_dataset, test_idx_high)
            else:
                train_size = len(train_idx)
                train_size_final = int(train_size * train_val_split_ratio)
                val_size = train_size - train_size_final
                # 随机划分（固定随机种子保证可复现）
                train_val_indices = np.random.RandomState(CV_SEED + fold_idx).permutation(train_size)
                train_indices_final = train_idx[train_val_indices[:train_size_final]]
                val_indices_final = train_idx[train_val_indices[train_size_final:]]
                train_ds_final = IndexedViewDataset(current_dataset, train_indices_final)
                val_ds_final = IndexedViewDataset(current_dataset, val_indices_final)
                test_ds = IndexedViewDataset(current_dataset, test_idx)
            
            # 创建并训练模型（针对过拟合问题，增加正则化，减少模型复杂度）
            Xshape = current_dataset[0][0].shape
            T = Xshape[1]
            time_input_dim = Xshape[0]  # 2*n_features
            dropout = 0.4  # 增加dropout：0.2 -> 0.4，减少过拟合
            dim = 48  # 减少模型维度：64 -> 48，降低模型复杂度
            if model_type == 'dual_stream':
                model = DualStreamAttention(time_steps=T, n_features=n_features, dim=dim, dropout=dropout).to(device)
            elif model_type == 'conv_pool_attention':
                model = ConvPoolAttentionModel(n_features=n_features, d_model=dim, dropout=dropout).to(device)
            elif model_type == 'conv_pool_only':
                # 禁用BatchNorm，观察不同特征通道的权重差异
                model = ConvPoolOnlyModel(n_features=n_features, d_model=dim, dropout=dropout, use_batchnorm=False).to(device)
            elif model_type == 'statistics_only':
                model = StatisticsOnlyModel(n_features=n_features, hidden_dim=dim, dropout=dropout).to(device)
            else:
                model = SimpleAttentionModel(time_input_dim=time_input_dim, d_model=dim, attn_hidden=dim, dropout=dropout).to(device)

            val_info = f"{len(val_ds_final)}" if val_ds_final is not None else "0 (不使用)"
            print(f"      训练集: {len(train_ds_final)} 验证集: {val_info} 测试集: {len(test_ds)}")
            # 针对过拟合：增加weight_decay，减少训练轮数，降低学习率
            train_model(model, train_ds_final, val_ds_final, epochs=50, device=device, output_dir=fold_out,
                       lr=3e-4, weight_decay=5e-4, patience=20)  # 增加weight_decay：1e-4 -> 5e-4，减少epochs：80 -> 50

            # 评估测试集
            metrics = evaluate_model(model, test_ds, device)
            with open(os.path.join(fold_out, 'metrics.json'), 'w', encoding='utf-8') as f:
                import json
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"      测试集指标: MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} R^2={metrics['r2']:.6f}")

            # 在测试集上提取并分析注意力权重（更接近泛化表现）
            weights_dict = extract_attention_weights(model, test_ds, device, model_type)
            analyze_attention_metrics(weights_dict, fold_out, model_type)
            del weights_dict

            # 释放
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    # 清理数据集
    del dataset
    print(f"\n多特征处理完成")
    # 强制垃圾回收
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*60}")
    print("所有特征处理完成！")
    print(f"结果保存在: {os.path.join(project_root, 'attention_5fold', 'multi_feature')}")


if __name__ == '__main__':
    main()
