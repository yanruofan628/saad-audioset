#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多特征注意力模型框架（std时间序列版本）：
- 对于标有 std 的特征，使用滑动窗口std时间序列
- 对于标有 mean 的特征，使用原始时间序列
- 参照 spatiotemporal_glm_model.py 的 std 时间序列提取方式
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
from linear_regression_selected_features import (
    load_or_extract_features,
    merge_balanced_pairs_by_type,
    load_all_human_decisions,
    calculate_selection_probability_144,
)
from linear_regression_loudness_models import (
    parse_stereo_pair_name,
    load_mono_5s,
    extract_loudness_time_series,
)
from pair_feature_diff_extraction import extract_time_series_features

SR = 16000
HOP = 8000  # 500ms/帧 (0.5 * 16000)


# ============== 1. 特征提取函数 ==============

def compute_rolling_std(series, window_size=5):
    """
    计算滑动窗口标准差时间序列（参照 spatiotemporal_glm_model.py）
    
    参数:
        series: 输入时间序列 (T,)
        window_size: 滑动窗口大小（帧数）
    
    返回:
        滑动std时间序列 (T,)
    """
    series = np.asarray(series, dtype=float)
    if series.ndim > 1:
        series = series.flatten()
    elif series.ndim == 0:
        series = np.array([series])
    series = np.nan_to_num(series, nan=0.0)
    
    if len(series) == 0:
        return np.array([0.0], dtype=np.float32)
    
    if len(series) < window_size:
        global_std = np.std(series) if len(series) > 1 else 0.0
        return np.full(len(series), global_std, dtype=np.float32)
    
    window_size = max(1, min(window_size, len(series)))
    
    try:
        series_contiguous = np.ascontiguousarray(series.astype(np.float64))
        rolling_mean = uniform_filter1d(series_contiguous, size=int(window_size), mode='nearest')
        series_sq = series_contiguous ** 2
        rolling_mean_sq = uniform_filter1d(series_sq, size=int(window_size), mode='nearest')
        rolling_var = rolling_mean_sq - rolling_mean**2
        rolling_std = np.sqrt(np.maximum(rolling_var, 0))
        
        if len(rolling_std) != len(series):
            global_std = float(np.std(series)) if len(series) > 1 else 0.0
            return np.full(len(series), global_std, dtype=np.float32)
        
        return rolling_std.astype(np.float32)
    except Exception as e:
        global_std = float(np.std(series)) if len(series) > 1 else 0.0
        return np.full(len(series), global_std, dtype=np.float32)


def extract_feature_time_series_with_std_option(y, sr, feature_type, hop_length=8000, use_std=False, window_size=5):
    """
    提取指定特征的时间序列，支持选择使用原始序列或std序列
    
    参数:
        y: 音频信号
        sr: 采样率
        feature_type: 特征类型
        hop_length: 帧移
        use_std: 是否使用std时间序列（True=std, False=原始）
        window_size: 滑动窗口大小（当use_std=True时使用）
    
    返回:
        特征时间序列 (T,)
    """
    from scipy import interpolate
    import librosa
    
    # 先计算mel_spec以获取正确的时间维度
    stft = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    magnitude = np.abs(stft)
    safe_fmax = min(12500, int(sr / 2) - 1)
    mel_filters = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=28, fmin=250, fmax=safe_fmax)
    mel_spec = np.dot(mel_filters, magnitude) + 1e-10
    mel_spec_len = mel_spec.shape[1]  # mel_spec的时间维度
    
    if feature_type == 'loudness':
        # 使用extract_loudness_time_series提取响度特征
        try:
            feat_series = extract_loudness_time_series(y, sr=sr, hop_length=hop_length)
        except Exception as e:
            print(f"  错误: extract_loudness_time_series失败: {e}")
            raise
    else:
        # 从extract_time_series_features提取
        effective_hop_length = 256  # 统一使用256，避免大hop_length导致的问题
        
        try:
            feat_dict = extract_time_series_features(y, sr=sr, hop_length=effective_hop_length)
            
            if feature_type in feat_dict:
                feat_series = feat_dict[feature_type]
            else:
                raise ValueError(f"不支持的特征类型: {feature_type}")
        except Exception as e:
            print(f"  错误: extract_time_series_features失败 (hop_length={effective_hop_length}, feature={feature_type}): {e}")
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
    
    # 如果使用std，先计算std时间序列（在长度对齐之前）
    # 参照 spatiotemporal_glm_model.py 的方式：先提取原始序列，再计算std
    if use_std:
        # 使用较小的hop_length提取的原始序列计算std
        # 注意：这里使用effective_hop_length提取的序列，然后计算std
        feat_series = compute_rolling_std(feat_series, window_size=window_size)
    
    # 确保长度正确
    original_len = len(feat_series)
    
    # 对齐长度到mel_spec_len
    if original_len != mel_spec_len:
        if original_len > mel_spec_len:
            # 下采样
            if mel_spec_len <= 0:
                raise ValueError(f"mel_spec_len必须大于0: {mel_spec_len}")
            
            try:
                if mel_spec_len > 1:
                    downsample_factor = original_len / mel_spec_len
                    kernel_size = max(1, int(downsample_factor))
                    smoothed = uniform_filter1d(feat_series.astype(np.float64), size=kernel_size, mode='nearest')
                    indices = np.linspace(0, original_len - 1, mel_spec_len).astype(int)
                    feat_series = smoothed[indices].astype(np.float32)
                else:
                    feat_series = np.array([np.mean(feat_series)], dtype=np.float32)
            except Exception:
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

def get_feature_configs_std_timeseries():
    """
    返回特征配置，标有 std 的特征使用 std 时间序列
    
    返回: list of tuples (feature_type, use_std, display_name)
    """
    feature_configs = [
        ('spectral_centroid', True, 'Spectral Centroid (std)'),      # std -> 使用std时间序列
        ('spec_mod_centroid', False, 'Spectral Modulation Centroid (mean)'),  # mean -> 使用原始时间序列
        ('spec_mod_centroid', True, 'Spectral Modulation Centroid (std)'),   # std -> 使用std时间序列
        ('spectral_irregularity', False, 'Spectral Irregularity (mean)'),     # mean -> 使用原始时间序列
        ('spectral_irregularity', True, 'Spectral Irregularity (std)'),      # std -> 使用std时间序列
        ('loudness', True, 'Loudness (std)'),                        # std -> 使用std时间序列
        ('temporal_mod', False, 'Temporal Modulation (mean)'),       # mean -> 使用原始时间序列
    ]
    
    return feature_configs


# ============== 3. Dataset类 ==============

class FeatureTimeSeriesDataset(Dataset):
    """多特征时间序列Dataset，支持std时间序列选项"""
    def __init__(self, pair_names, base_dir, targets, feature_configs, sr=16000, hop_length=8000, window_size=5):
        """
        参数:
            pair_names: 音频对名称列表
            base_dir: 音频文件基础目录
            targets: 目标值（选择概率）
            feature_configs: 特征配置列表，每个元素为 (feature_type, use_std, display_name)
            sr: 采样率
            hop_length: 帧移
            window_size: 滑动窗口大小（用于std计算）
        """
        self.pair_names = pair_names
        self.base_dir = base_dir
        self.targets = targets.astype(np.float32)
        self.feature_configs = feature_configs
        self.n_features = len(feature_configs)
        self.sr = sr
        self.hop_length = hop_length
        self.window_size = window_size
        # 预提取全部样本的(2*n_features, T)
        self.X, self.y = self._build()

    def _build(self):
        multi_feat_list = []
        targets_list = []
        T_ref = None
        total = len(self.pair_names)
        print(f"  正在提取{self.n_features}个特征，共 {total} 个音频对...")
        
        for i, pair_name in enumerate(self.pair_names):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  进度: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
            try:
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
                for feat_idx, (feat_type, use_std, display_name) in enumerate(self.feature_configs):
                    try:
                        if i == 0 and feat_idx == 0:
                            print(f"    开始提取特征: {display_name} (use_std={use_std})")
                        
                        feat_L = extract_feature_time_series_with_std_option(
                            yL, sr=sr, feature_type=feat_type, 
                            hop_length=self.hop_length, 
                            use_std=use_std, 
                            window_size=self.window_size
                        )
                        feat_R = extract_feature_time_series_with_std_option(
                            yR, sr=sr, feature_type=feat_type,
                            hop_length=self.hop_length,
                            use_std=use_std,
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
                            print(f"    特征 {display_name} 提取完成: 长度={len(feat_L)}")
                        
                        left_features.append(feat_L)
                        right_features.append(feat_R)
                    except Exception as e:
                        print(f"  警告: 提取特征 {display_name} (use_std={use_std}) 失败: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # 确定时间长度
                if not left_features or not right_features:
                    raise ValueError("特征列表为空")
                
                min_left_len = min(len(f) for f in left_features)
                min_right_len = min(len(f) for f in right_features)
                T_cur = min(min_left_len, min_right_len)
                
                if T_cur <= 0:
                    raise ValueError(f"特征长度为0或负数: T_cur={T_cur}")
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
        X_raw = []
        for left_features, right_features in multi_feat_list:
            feat_channels = []
            for feat_L, feat_R in zip(left_features, right_features):
                L = np.asarray(feat_L[:T_ref], dtype=np.float32).flatten()
                R = np.asarray(feat_R[:T_ref], dtype=np.float32).flatten()
                
                if len(L) < T_ref:
                    L = np.pad(L, (0, T_ref - len(L)), mode='constant', constant_values=(L[-1] if len(L) > 0 else 0.0))
                if len(R) < T_ref:
                    R = np.pad(R, (0, T_ref - len(R)), mode='constant', constant_values=(R[-1] if len(R) > 0 else 0.0))
                
                L = L[:T_ref]
                R = R[:T_ref]
                
                L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
                R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                
                feat_channels.append(L)
                feat_channels.append(R)
            
            if len(feat_channels) == 0:
                continue
            x = np.stack(feat_channels, axis=0)
            X_raw.append(x)
        
        if len(X_raw) == 0:
            raise ValueError("没有成功提取任何特征，请检查特征提取函数和音频文件")
        
        X_raw = np.stack(X_raw, axis=0)  # (N, 2*n_features, T)
        
        # 按特征通道标准化
        X = X_raw.copy()
        for ch_idx in range(2 * self.n_features):
            ch_data = X_raw[:, ch_idx, :].flatten()
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


def load_or_build_cached_dataset(feature_configs, project_root: str, base_dir: str, sr: int, hop_length: int, window_size: int = 5) -> FeatureTimeSeriesDataset:
    """
    构建/缓存多特征数据集（支持std时间序列）
    """
    import hashlib
    # 生成缓存key（基于特征配置）
    feature_key = '_'.join([f"{ft}_{'std' if us else 'mean'}" for ft, us, _ in feature_configs])
    # 如果文件名太长，使用hash缩短
    if len(feature_key) > 100:
        feature_key_hash = hashlib.md5(feature_key.encode('utf-8')).hexdigest()[:16]
        cache_filename = f'dataset_cache_std_ts_{feature_key_hash}.npz'
    else:
        cache_filename = f'dataset_cache_std_ts_{feature_key}.npz'
    
    cache_dir = os.path.join(project_root, 'transformer_cache', 'std_timeseries')
    # 确保目录创建成功
    try:
        os.makedirs(cache_dir, exist_ok=True)
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
        X_raw = data['X']
        
        # 按特征通道标准化
        X = X_raw.copy()
        n_features = len(feature_configs)
        for ch_idx in range(2 * n_features):
            ch_data = X_raw[:, ch_idx, :].flatten()
            ch_mean = np.nanmean(ch_data)
            ch_std = np.nanstd(ch_data) + 1e-8
            X[:, ch_idx, :] = (X[:, ch_idx, :] - ch_mean) / ch_std
        
        ds = FeatureTimeSeriesDataset.__new__(FeatureTimeSeriesDataset)
        ds.pair_names = None
        ds.base_dir = base_dir
        ds.targets = None
        ds.feature_configs = feature_configs
        ds.n_features = n_features
        ds.sr = sr
        ds.hop_length = hop_length
        ds.window_size = window_size
        ds.X = X
        ds.y = data['y']
        return ds
    else:
        print("  未找到缓存，按音频构建并缓存...")
        # 需要 all_pair_names 与 probs
        diff_raw, diff_z, pair_names = load_or_extract_features(project_root)
        merged = merge_balanced_pairs_by_type(pair_names)
        all_pair_names = []
        for t in ['nn_main', 'nn_sub', 'main', 'sub']:
            if t in merged:
                all_pair_names.extend(merged[t]['pair_names'])
        trials = get_trials()
        probs = calculate_selection_probability_144(all_pair_names, trials).astype(np.float32)
        ds = FeatureTimeSeriesDataset(all_pair_names, base_dir, probs, feature_configs, sr=sr, hop_length=hop_length, window_size=window_size)
        
        # 保存缓存文件
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, X=ds.X, y=ds.y)
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


# ============== 4. 模型定义（复用原脚本） ==============

class AdditiveTimeAttention(nn.Module):
    """时间维度上的加性注意力"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        # H: (batch, T, input_dim)
        u = torch.tanh(self.proj(H))
        scores = self.v(u).squeeze(-1)
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), H).squeeze(1)
        return context, attn


class SimpleAttentionModel(nn.Module):
    """单耳注意力模型"""
    def __init__(self, time_input_dim, d_model=64, attn_hidden=64, dropout=0.2):
        super().__init__()
        self.feat = nn.Linear(time_input_dim, d_model)
        self.attn = AdditiveTimeAttention(d_model, attn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model + 4, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, 2*n_features, T)
        x = x.transpose(1, 2)
        h = self.feat(x)
        h = self.dropout(h)
        context, attn = self.attn(h)
        
        n_features = x.shape[-1] // 2
        left_feat = x[:, :, :n_features]
        right_feat = x[:, :, n_features:]
        
        left_mean = left_feat.mean(dim=1)
        right_mean = right_feat.mean(dim=1)
        left_std = left_feat.std(dim=1)
        right_std = right_feat.std(dim=1)
        left_max = left_feat.max(dim=1)[0]
        right_max = right_feat.max(dim=1)[0]
        left_min = left_feat.min(dim=1)[0]
        right_min = right_feat.min(dim=1)[0]
        
        global_feat = torch.stack([
            (left_mean - right_mean).mean(dim=1),
            (left_std - right_std).mean(dim=1),
            (left_max - right_max).mean(dim=1),
            (left_min - right_min).mean(dim=1),
        ], dim=1)
        
        combined = torch.cat([context, global_feat], dim=1)
        y = self.out(combined).squeeze(-1)
        return y, attn
    
    def get_score(self, x):
        x = x.transpose(1, 2)
        h = self.feat(x)
        context, attn = self.attn(h)
        n_features = x.shape[-1] // 2
        left_feat = x[:, :, :n_features]
        right_feat = x[:, :, n_features:]
        left_score = (left_feat * attn.unsqueeze(-1)).sum(dim=(1, 2))
        right_score = (right_feat * attn.unsqueeze(-1)).sum(dim=(1, 2))
        return left_score, right_score


class DualStreamAttention(nn.Module):
    """左右耳独立注意力模型"""
    def __init__(self, time_steps, n_features, dim=64, dropout=0.2):
        super().__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.left_proj = nn.Linear(n_features, dim)
        self.right_proj = nn.Linear(n_features, dim)
        self.left_attention = nn.Linear(dim, 1)
        self.right_attention = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2 + 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        left_ear = x[:, :self.n_features, :].transpose(1, 2)
        right_ear = x[:, self.n_features:, :].transpose(1, 2)
        
        left_projected = torch.tanh(self.left_proj(left_ear))
        right_projected = torch.tanh(self.right_proj(right_ear))
        left_projected = self.dropout(left_projected)
        right_projected = self.dropout(right_projected)
        
        left_weights = F.softmax(self.left_attention(left_projected).squeeze(-1), dim=-1)
        right_weights = F.softmax(self.right_attention(right_projected).squeeze(-1), dim=-1)
        
        left_context = torch.sum(left_projected * left_weights.unsqueeze(-1), dim=1)
        right_context = torch.sum(right_projected * right_weights.unsqueeze(-1), dim=1)
        
        left_raw = left_ear
        right_raw = right_ear
        left_mean = left_raw.mean(dim=1).mean(dim=1)
        right_mean = right_raw.mean(dim=1).mean(dim=1)
        left_std = left_raw.std(dim=1).mean(dim=1)
        right_std = right_raw.std(dim=1).mean(dim=1)
        left_max = left_raw.max(dim=1)[0].mean(dim=1)
        right_max = right_raw.max(dim=1)[0].mean(dim=1)
        left_min = left_raw.min(dim=1)[0].mean(dim=1)
        right_min = right_raw.min(dim=1)[0].mean(dim=1)
        
        global_feat = torch.stack([
            left_mean - right_mean,
            left_std - right_std,
            left_max - right_max,
            left_min - right_min,
        ], dim=1)
        
        combined = torch.cat([left_context, right_context, global_feat], dim=-1)
        output = torch.sigmoid(self.classifier(combined)).squeeze(-1)
        return output, left_weights, right_weights
    
    def get_score(self, x):
        left_ear = x[:, :self.n_features, :].transpose(1, 2)
        right_ear = x[:, self.n_features:, :].transpose(1, 2)
        left_proj = torch.tanh(self.left_proj(left_ear))
        right_proj = torch.tanh(self.right_proj(right_ear))
        left_weights = F.softmax(self.left_attention(left_proj).squeeze(-1), dim=-1)
        right_weights = F.softmax(self.right_attention(right_proj).squeeze(-1), dim=-1)
        left_context = torch.sum(left_proj * left_weights.unsqueeze(-1), dim=1)
        right_context = torch.sum(right_proj * right_weights.unsqueeze(-1), dim=1)
        left_score = torch.norm(left_context, dim=1)
        right_score = torch.norm(right_context, dim=1)
        return left_score, right_score


# ============== 5. 训练和评估函数（复用原脚本） ==============

def train_model(model, train_dataset, val_dataset, epochs, device, output_dir, 
                lr=1e-3, weight_decay=1e-4, patience=20):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False) if val_dataset is not None else None
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if isinstance(model, DualStreamAttention):
                pred, _, _ = model(xb)
            else:
                pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_n += xb.size(0)
        avg_train_loss = train_loss / max(train_n, 1)
        
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
                    else:
                        pred, _ = model(xb)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item() * xb.size(0)
                    val_n += xb.size(0)
            avg_val_loss = val_loss / max(val_n, 1)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} (best={best_val_loss:.6f}, patience={patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"  早停触发（patience={patience}），恢复最佳模型")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.6f}")
    
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
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float('nan') if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {'mse': mse, 'mae': mae, 'r2': r2}


# ============== 6. 主函数 ==============

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


def main():
    print("=== 多特征注意力模型框架（std时间序列版本） ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.getcwd()
    
    # 设置音频文件基础目录
    base_dir = r"D:\D\research\audioset下载\clap_select"
    
    if not os.path.exists(base_dir):
        possible_dirs = [
            r"D:\D\research\audioset下载\clap_select",
            os.path.join(project_root, '..', '..', 'clap_select'),
            os.path.join(project_root, '..', 'clap_select'),
            os.path.join(project_root, 'clap_select'),
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
    
    print(f"\n音频文件基础目录: {base_dir}")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"音频文件目录不存在: {base_dir}")
    
    # 定义特征配置
    feature_configs = get_feature_configs_std_timeseries()
    print(f"\n使用{len(feature_configs)}个特征:")
    for i, (feat_type, use_std, display_name) in enumerate(feature_configs):
        print(f"  {i+1}. {display_name} (use_std={use_std})")
    
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
    print(f"处理多特征数据集（std时间序列版本）")
    print(f"{'='*60}")
    
    # 构建数据集
    print(f"\n步骤2: 构建多特征数据集")
    window_size = 5
    dataset = load_or_build_cached_dataset(feature_configs, project_root, base_dir, sr=SR, hop_length=HOP, window_size=window_size)
    if len(dataset) == 0:
        print(f"  警告: 数据集为空，退出")
        return
    
    Xshape = dataset[0][0].shape
    T = Xshape[1]
    n_features = len(feature_configs)
    print(f"  单样本形状: {Xshape}, T={T}, n_features={n_features}")
    
    # 5折划分
    num_pairs = len(dataset)
    rng_indices = np.random.RandomState(42).permutation(num_pairs)
    folds = np.array_split(rng_indices, 5)

    # 对每个模型类型与每折进行训练与测试
    model_types = ['single_attention', 'dual_stream']
    for model_type in model_types:
        print(f"\n  --- 模型类型: {model_type} ---")
        for fold_idx, test_idx in enumerate(folds, start=1):
            print(f"    -> 折 {fold_idx}/5")
            train_idx = np.setdiff1d(rng_indices, test_idx, assume_unique=True)
            train_ds = IndexedViewDataset(dataset, train_idx)
            test_ds = IndexedViewDataset(dataset, test_idx)

            fold_out = os.path.join(project_root, 'attention_5fold', 'std_timeseries', model_type, f'fold_{fold_idx}')
            os.makedirs(fold_out, exist_ok=True)

            # 从训练集中划分验证集（10%）
            val_ratio = 0.1
            train_size = len(train_ds)
            val_size = max(1, int(train_size * val_ratio))
            train_size_actual = train_size - val_size
            train_val_indices = np.random.RandomState(42 + fold_idx).permutation(train_size)
            train_indices_final = train_val_indices[:train_size_actual]
            val_indices_final = train_val_indices[train_size_actual:]
            train_indices_mapped = train_ds.indices[train_indices_final]
            val_indices_mapped = train_ds.indices[val_indices_final]
            train_ds_final = IndexedViewDataset(dataset, train_indices_mapped)
            val_ds_final = IndexedViewDataset(dataset, val_indices_mapped)

            # 创建并训练模型
            Xshape = dataset[0][0].shape
            T = Xshape[1]
            time_input_dim = Xshape[0]
            dropout = 0.2
            dim = 64
            if model_type == 'dual_stream':
                model = DualStreamAttention(time_steps=T, n_features=n_features, dim=dim, dropout=dropout).to(device)
            else:
                model = SimpleAttentionModel(time_input_dim=time_input_dim, d_model=dim, attn_hidden=dim, dropout=dropout).to(device)

            print(f"      训练集: {len(train_ds_final)} 验证集: {len(val_ds_final)} 测试集: {len(test_ds)}")
            train_model(model, train_ds_final, val_ds_final, epochs=80, device=device, output_dir=fold_out,
                       lr=5e-4, weight_decay=1e-4, patience=20)

            # 评估测试集
            metrics = evaluate_model(model, test_ds, device)
            with open(os.path.join(fold_out, 'metrics.json'), 'w', encoding='utf-8') as f:
                import json
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"      测试集指标: MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} R^2={metrics['r2']:.6f}")

            # 释放
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    # 清理数据集
    del dataset
    print(f"\n多特征处理完成（std时间序列版本）")
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*60}")
    print("所有特征处理完成！")
    print(f"结果保存在: {os.path.join(project_root, 'attention_5fold', 'std_timeseries')}")


if __name__ == '__main__':
    main()

