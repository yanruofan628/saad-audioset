import os
import json
import numpy as np
import pandas as pd
import librosa
import time


def read_pair_list(csv_paths):
    pair_names = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if 'original_name' not in df.columns:
            raise ValueError(f"CSV缺少original_name列: {csv_path}")
        pair_names.extend(df['original_name'].astype(str).tolist())
    return pair_names


def parse_stereo_pair_name(filename, base_dir):
    # 去掉扩展名
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    # 切分左右
    if '+' not in filename:
        raise ValueError(f"无法解析左右两侧: {filename}")

    left_part, right_part = filename.split('+', 1)

    # 去掉前缀
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break

    # 现在形如 "Helicopter_W3u2hj1x7gY_2" 和 "Baby cry, infant cry_s_Le6PwhssI_2"
    # 规则：第一个下划线前面是类别名，第一个下划线后面是文件ID
    def split_category_and_id_fs(part):
        # 找到第一个下划线的位置
        first_underscore_pos = part.find('_')
        if first_underscore_pos == -1:
            raise ValueError(f"未找到下划线分隔符: {part}")
        
        # 第一个下划线前面是类别名，后面是文件ID
        category = part[:first_underscore_pos]
        file_id = part[first_underscore_pos + 1:]
        
        # 验证文件是否存在
        candidate = os.path.join(base_dir, category, f"{file_id}.wav")
        if os.path.exists(candidate):
            return category, file_id
        else:
            raise ValueError(f"文件不存在: {candidate} (类别: {category}, 文件ID: {file_id})")

    left_cat, left_id = split_category_and_id_fs(left_part)
    right_cat, right_id = split_category_and_id_fs(right_part)
    return left_cat, left_id, right_cat, right_id


def load_mono_5s(audio_path, target_sr=16000, duration_s=5.0):
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    target_len = int(duration_s * target_sr)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode='constant')
    return y, target_sr


def extract_time_series_features(y, sr=16000, hop_length=256, n_fft=2048):
    # STFT magnitude
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    # 28梅尔滤波器
    # fmax 不得超过奈奎斯特频率，否则会产生空滤波器
    safe_fmax = min(12500, int(sr / 2) - 1)
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=28, fmin=250, fmax=safe_fmax)
    mel_spec = np.dot(mel_filters, magnitude) + 1e-10
    
    # 目标时间维度（以mel_spec为准，因为它是所有特征的基础）
    T = mel_spec.shape[1]

    # 统一所有特征的时间维度函数（参考extract_features.py）
    def ensure_time_dimension(feature_array, target_length):
        """确保特征数组具有目标时间长度"""
        feat_arr = np.asarray(feature_array)
        if feat_arr.ndim > 1:
            feat_arr = feat_arr.flatten()
        elif feat_arr.ndim == 0:
            feat_arr = np.array([feat_arr])
        
        current_len = len(feat_arr)
        if current_len == target_length:
            return feat_arr
        elif current_len > target_length:
            return feat_arr[:target_length]
        else:
            # 如果长度不足，用最后一个值填充（确保fill_value是标量）
            if current_len > 0:
                last_val = float(feat_arr[-1]) if not isinstance(feat_arr[-1], np.ndarray) else float(feat_arr[-1].item())
            else:
                last_val = 0.0
            padding = np.full(target_length - current_len, last_val, dtype=feat_arr.dtype)
            return np.concatenate([feat_arr, padding])

    # 特征时间序列
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).flatten()
    spectral_centroid = ensure_time_dimension(spectral_centroid, T)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).flatten()
    spectral_bandwidth = ensure_time_dimension(spectral_bandwidth, T)

    # 频谱平坦度（帧级几何均值/算术均值）
    spec_flat = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        gmean = np.exp(np.mean(np.log(frame)))
        amean = np.mean(frame)
        spec_flat.append(gmean / amean if amean > 0 else 0.0)
    spec_flat = np.asarray(spec_flat)
    spec_flat = ensure_time_dimension(spec_flat, T)

    # 频谱不规则度（相邻滤波器能量差绝对和）
    spec_irreg = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        spec_irreg.append(np.sum(np.abs(np.diff(frame))))
    spec_irreg = np.asarray(spec_irreg)
    spec_irreg = ensure_time_dimension(spec_irreg, T)

    # F0 和 Harmonicity 特征已完全移除（短时间提取意义不大）
    # 不再提取这两个特征

    # MFCC与delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)

    # 时序调制（每帧系数的平均绝对delta）
    temporal_mod = np.mean(np.abs(delta_mfcc), axis=0)
    temporal_mod = ensure_time_dimension(temporal_mod, T)

    # 频谱调制质心（每帧：以|MFCC|为权的系数索引质心）
    coeff_idx = np.arange(mfcc.shape[0], dtype=float)  # 0..12
    spec_mod_centroid = []
    abs_mfcc = np.abs(mfcc)
    for t in range(mfcc.shape[1]):
        w = abs_mfcc[:, t]
        s = np.sum(w)
        spec_mod_centroid.append(float(np.sum(coeff_idx * w) / s) if s > 0 else 0.0)
    spec_mod_centroid = np.asarray(spec_mod_centroid)
    spec_mod_centroid = ensure_time_dimension(spec_mod_centroid, T)

    # 响度（对数压缩的梅尔能量和）
    loudness = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        loudness.append(np.sum(20 * np.log10(frame)))
    loudness = np.asarray(loudness)
    loudness = ensure_time_dimension(loudness, T)

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spec_flat,
        'spectral_irregularity': spec_irreg,
        # 'f0' 和 'harmonicity' 已移除（短时间提取意义不大）
        'temporal_mod': temporal_mod,
        'spec_mod_centroid': spec_mod_centroid,
        'loudness': loudness,
    }


def aggregate_features(feat_ts):
    # 聚合规则：所有特征都计算mean和std
    # 每个特征先mean后std，共9个特征 × 2 = 18个聚合特征
    def safe_mean(x):
        if x is None:
            return np.nan
        if np.issubdtype(x.dtype, np.floating):
            return float(np.nanmean(x))
        return float(np.mean(x))

    def safe_std(x):
        if x is None:
            return np.nan
        if np.issubdtype(x.dtype, np.floating):
            return float(np.nanstd(x))
        return float(np.std(x))

    return np.array([
        # 1. Spectral Centroid
        safe_mean(feat_ts['spectral_centroid']),     # mean
        safe_std(feat_ts['spectral_centroid']),      # std
        # 2. F0 - 已移除（短时间提取意义不大）
        # 3. Harmonicity - 已移除（短时间提取意义不大）
        # 4. Spectral Modulation Centroid
        safe_mean(feat_ts['spec_mod_centroid']),     # mean
        safe_std(feat_ts['spec_mod_centroid']),      # std
        # 5. Spectral Irregularity
        safe_mean(feat_ts['spectral_irregularity']), # mean
        safe_std(feat_ts['spectral_irregularity']),  # std
        # 6. Loudness
        safe_mean(feat_ts['loudness']),              # mean
        safe_std(feat_ts['loudness']),               # std
        # 7. Spectral Flatness
        safe_mean(feat_ts['spectral_flatness']),     # mean
        safe_std(feat_ts['spectral_flatness']),      # std
        # 8. Spectral Bandwidth
        safe_mean(feat_ts['spectral_bandwidth']),    # mean
        safe_std(feat_ts['spectral_bandwidth']),     # std
        # 9. Temporal Modulation
        safe_mean(feat_ts['temporal_mod']),           # mean
        safe_std(feat_ts['temporal_mod']),           # std
    ], dtype=float)


def zscore_per_feature(matrix):
    # matrix: F x N
    out = matrix.copy().astype(float)
    for i in range(out.shape[0]):
        row = out[i]
        mask = ~np.isnan(row)
        if not np.any(mask):
            continue
        m = np.mean(row[mask])
        s = np.std(row[mask])
        if s == 0:
            out[i, mask] = 0.0
        else:
            out[i, mask] = (row[mask] - m) / s
    return out


def extract_time_window_features(feat_ts, sr=16000, hop_length=256, n_time_windows=10, window_duration=0.5):
    """
    将时间序列特征划分为时间窗，对每个时间窗计算mean和std
    
    参数:
        feat_ts: 时间序列特征字典
        sr: 采样率
        hop_length: hop length (用于计算时间帧)
        n_time_windows: 时间窗数量（默认10个）
        window_duration: 每个时间窗的时长（秒，默认0.5秒）
    
    返回:
        timewindow_features: (18 * n_time_windows,) 数组
        特征顺序：对于每个特征f，按时间窗顺序排列 [f_mean_t0, f_std_t0, f_mean_t1, f_std_t1, ...]
    """
    # 计算每帧的时间（秒）
    frame_duration = hop_length / sr  # 每帧的时长（秒）
    
    # 计算每个时间窗包含的帧数
    frames_per_window = int(window_duration / frame_duration)
    
    # 特征名称列表（与aggregate_features中的顺序一致）
    feature_names = [
        'spectral_centroid',
        # 'f0' - 已移除（短时间提取意义不大）
        # 'harmonicity' - 已移除（短时间提取意义不大）
        'spec_mod_centroid',
        'spectral_irregularity',
        'loudness',
        'spectral_flatness',
        'spectral_bandwidth',
        'temporal_mod'
    ]
    
    def safe_mean(x, fallback_value=0.0):
        """计算均值，如果没有有效值则返回fallback_value"""
        if x is None or len(x) == 0:
            return fallback_value
        if np.issubdtype(x.dtype, np.floating):
            valid = x[~np.isnan(x)]
            if len(valid) == 0:
                return fallback_value
            return float(np.nanmean(valid))
        return float(np.mean(x))
    
    def safe_std(x, fallback_value=0.0):
        """计算标准差，如果没有有效值则返回fallback_value"""
        if x is None or len(x) == 0:
            return fallback_value
        if np.issubdtype(x.dtype, np.floating):
            valid = x[~np.isnan(x)]
            if len(valid) == 0:
                return fallback_value
            return float(np.nanstd(valid))
        return float(np.std(x))
    
    # 先计算每个特征的全局均值（用于填充缺失值）
    feature_global_means = {}
    feature_global_stds = {}
    
    for feat_name in feature_names:
        if feat_name not in feat_ts:
            feature_global_means[feat_name] = 0.0
            feature_global_stds[feat_name] = 0.0
            continue
        
        feat_series = feat_ts[feat_name]
        if len(feat_series) == 0:
            feature_global_means[feat_name] = 0.0
            feature_global_stds[feat_name] = 0.0
            continue
        
        # 计算全局均值和标准差（跨所有帧）
        valid_values = feat_series[~np.isnan(feat_series)] if np.issubdtype(feat_series.dtype, np.floating) else feat_series
        if len(valid_values) > 0:
            feature_global_means[feat_name] = float(np.nanmean(feat_series))
            feature_global_stds[feat_name] = float(np.nanstd(feat_series))
        else:
            feature_global_means[feat_name] = 0.0
            feature_global_stds[feat_name] = 0.0
    
    # 存储所有时间窗的特征
    timewindow_features = []
    
    # 对每个特征
    for feat_name in feature_names:
        if feat_name not in feat_ts:
            # 如果特征不存在，用0填充
            global_mean = feature_global_means.get(feat_name, 0.0)
            global_std = feature_global_stds.get(feat_name, 0.0)
            for t in range(n_time_windows):
                timewindow_features.append(global_mean)  # mean
                timewindow_features.append(global_std)  # std
            continue
        
        feat_series = feat_ts[feat_name]
        total_frames = len(feat_series)
        global_mean = feature_global_means.get(feat_name, 0.0)
        global_std = feature_global_stds.get(feat_name, 0.0)
        
        # 对每个时间窗
        for t in range(n_time_windows):
            start_frame = t * frames_per_window
            end_frame = min((t + 1) * frames_per_window, total_frames)
            
            if start_frame >= total_frames:
                # 超出范围，使用全局均值填充
                timewindow_features.append(global_mean)  # mean
                timewindow_features.append(global_std)   # std
            else:
                window_data = feat_series[start_frame:end_frame]
                # 如果窗口内没有有效值，使用全局均值填充
                window_mean = safe_mean(window_data, fallback_value=global_mean)
                window_std = safe_std(window_data, fallback_value=global_std)
                timewindow_features.append(window_mean)
                timewindow_features.append(window_std)
    
    return np.array(timewindow_features, dtype=float)


def create_timewindow_metadata(n_features=9, n_time_windows=10):
    """
    创建时间窗元数据，用于索引特征-时间窗组合
    
    返回:
        metadata: 字典，包含特征名称、时间窗索引等信息
    """
    feature_names = [
        'Spectral Centroid',
        # 'F0' - 已移除（短时间提取意义不大）
        # 'Harmonicity' - 已移除（短时间提取意义不大）
        'Spectral Modulation Centroid',
        'Spectral Irregularity',
        'Loudness',
        'Spectral Flatness',
        'Spectral Bandwidth',
        'Temporal Modulation'
    ]
    
    metadata = {
        'feature_names': feature_names,
        'n_features': n_features,
        'n_time_windows': n_time_windows,
        'window_duration': 0.5,  # 秒
        'feature_timewindow_index': {}
    }
    
    # 创建索引映射：feature_idx_timewindow -> (feature_idx, timewindow_idx, stat_type)
    idx = 0
    for f_idx, feat_name in enumerate(feature_names):
        for t_idx in range(n_time_windows):
            # mean
            metadata['feature_timewindow_index'][idx] = {
                'feature_idx': f_idx,
                'feature_name': feat_name,
                'timewindow_idx': t_idx,
                'stat_type': 'mean',
                'time_range': (t_idx * 0.5, (t_idx + 1) * 0.5)
            }
            idx += 1
            # std
            metadata['feature_timewindow_index'][idx] = {
                'feature_idx': f_idx,
                'feature_name': feat_name,
                'timewindow_idx': t_idx,
                'stat_type': 'std',
                'time_range': (t_idx * 0.5, (t_idx + 1) * 0.5)
            }
            idx += 1
    
    return metadata


def main():
    project_root = os.getcwd()
    csv1 = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv"
    csv2 = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    base_dir = r"D:\D\research\audioset下载\clap_select"

    print(f"正在读取CSV文件...")
    print(f"CSV1: {csv1}")
    print(f"CSV2: {csv2}")
    pair_names = read_pair_list([csv1, csv2])
    print(f"共读取到 {len(pair_names)} 个音频对")

    F = 18  # 9个特征 × 2 (mean + std) = 18个聚合特征
    N = len(pair_names)
    diff_raw = np.full((F, N), np.nan, dtype=float)

    # 时间窗特征：18个特征 × 10个时间窗 = 180维
    N_TIME_WINDOWS = 10
    F_TIMEWINDOW = 18 * N_TIME_WINDOWS  # 180
    diff_timewindow_raw = np.full((F_TIMEWINDOW, N), np.nan, dtype=float)

    print(f"开始处理 {N} 个音频对...")
    print(f"聚合特征维度: {F}, 时间窗特征维度: {F_TIMEWINDOW}")
    start_time = time.time()
    for idx, pair_name in enumerate(pair_names):
        processed = idx + 1
        elapsed = time.time() - start_time
        avg_per_item = elapsed / processed if processed > 0 else 0.0
        remaining = (N - processed) * avg_per_item
        # 简洁ETA显示
        def fmt(sec):
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        print(f"[{processed:03d}/{N}] ETA {fmt(remaining)} -> {pair_name}")
        try:
            left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, base_dir)
            left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
            right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")

            if not os.path.exists(left_path):
                raise FileNotFoundError(f"未找到左侧音频: {left_path}")
            if not os.path.exists(right_path):
                raise FileNotFoundError(f"未找到右侧音频: {right_path}")

            yL, sr = load_mono_5s(left_path)
            yR, _ = load_mono_5s(right_path)

            featL_ts = extract_time_series_features(yL, sr=sr, hop_length=256)
            featR_ts = extract_time_series_features(yR, sr=sr, hop_length=256)

            # 聚合特征（原有功能）
            aggL = aggregate_features(featL_ts)  # (18,) - 9个特征 × 2 (mean + std)
            aggR = aggregate_features(featR_ts)  # (18,) - 9个特征 × 2 (mean + std)
            diff_raw[:, idx] = aggL - aggR

            # 时间窗特征（新功能）
            twL = extract_time_window_features(featL_ts, sr=sr, hop_length=256, 
                                               n_time_windows=N_TIME_WINDOWS, window_duration=0.5)
            twR = extract_time_window_features(featR_ts, sr=sr, hop_length=256,
                                               n_time_windows=N_TIME_WINDOWS, window_duration=0.5)
            diff_timewindow_raw[:, idx] = twL - twR

        except Exception as e:
            # 直接报错并结束
            raise RuntimeError(f"处理音频对失败: {pair_name} -> {e}")

    total_time = time.time() - start_time
    diff_z = zscore_per_feature(diff_raw)
    diff_timewindow_z = zscore_per_feature(diff_timewindow_raw)

    # 保存聚合特征（原有功能）
    np.save(os.path.join(project_root, 'pair_feature_diff_raw.npy'), diff_raw)
    np.save(os.path.join(project_root, 'pair_feature_diff_zscore.npy'), diff_z)
    np.save(os.path.join(project_root, 'pair_list.npy'), np.array(pair_names, dtype=object))

    # 保存时间窗特征（新功能）
    np.save(os.path.join(project_root, 'pair_feature_diff_timewindows_raw.npy'), diff_timewindow_raw)
    np.save(os.path.join(project_root, 'pair_feature_diff_timewindows_zscore.npy'), diff_timewindow_z)
    
    # 保存时间窗元数据
    metadata = create_timewindow_metadata(n_features=9, n_time_windows=N_TIME_WINDOWS)
    # 将numpy类型转换为Python原生类型以便JSON序列化
    metadata_serializable = {
        'feature_names': metadata['feature_names'],
        'n_features': int(metadata['n_features']),
        'n_time_windows': int(metadata['n_time_windows']),
        'window_duration': float(metadata['window_duration']),
        'feature_timewindow_index': {
            str(k): {
                'feature_idx': int(v['feature_idx']),
                'feature_name': v['feature_name'],
                'timewindow_idx': int(v['timewindow_idx']),
                'stat_type': v['stat_type'],
                'time_range': [float(v['time_range'][0]), float(v['time_range'][1])]
            }
            for k, v in metadata['feature_timewindow_index'].items()
        }
    }
    with open(os.path.join(project_root, 'timewindow_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)

    # 打印尺寸与部分内容
    print(f"\n聚合特征:")
    print(f"  原始矩阵(raw)形状: {diff_raw.shape}, 归一化矩阵(zscore)形状: {diff_z.shape}")
    print("  前5列raw(前3行示例):")
    cols = min(5, diff_raw.shape[1])
    print(np.round(diff_raw[:3, :cols], 4))
    print("  前5列zscore(前3行示例):")
    print(np.round(diff_z[:3, :cols], 4))
    
    print(f"\n时间窗特征:")
    print(f"  原始矩阵(raw)形状: {diff_timewindow_raw.shape}, 归一化矩阵(zscore)形状: {diff_timewindow_z.shape}")
    print("  前5列raw(前3行示例):")
    cols = min(5, diff_timewindow_raw.shape[1])
    print(np.round(diff_timewindow_raw[:3, :cols], 4))
    print("  前5列zscore(前3行示例):")
    print(np.round(diff_timewindow_z[:3, :cols], 4))
    
    print(f"\n总耗时: {int(total_time)} 秒 (~{total_time/60:.1f} 分钟)")
    print(f"时间窗元数据已保存: timewindow_metadata.json")


if __name__ == '__main__':
    main()


