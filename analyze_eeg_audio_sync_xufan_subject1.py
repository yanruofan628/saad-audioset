#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算xufan subject1的EEG与左右音频的同步性
包括：延迟相关性和包络相位同步性（PLV）
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 复用linear_regression_loudness_models.py中的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from linear_regression_loudness_models import (
        parse_stereo_pair_name,
        load_mono_5s,
        parse_audio_mapping,
        parse_experiment_data_txt
    )
except ImportError:
    print("警告: 无法导入linear_regression_loudness_models.py中的函数")
    sys.exit(1)


def parse_audio_mapping_custom(file_path):
    """
    解析音频映射文件（自定义版本，适配实际的CSV格式）
    返回映射字典：{映射后的名字: 原始名字}
    
    CSV格式：第一列是原始音频对名称，第二列是映射后的文件名（如a001.wav）
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        
        print(f"\n解析映射文件: {file_path}")
        print(f"  文件形状: {df.shape}")
        print(f"  列名: {list(df.columns)}")
        
        if len(df.columns) >= 2:
            col1 = df.columns[0]  # 原始音频对名称
            col2 = df.columns[1]  # 映射后的文件名（如a001.wav）
            
            for idx, row in df.iterrows():
                original_name = str(row[col1]).strip()
                new_name = str(row[col2]).strip()
                
                if pd.isna(row[col1]) or pd.isna(row[col2]) or not original_name or not new_name:
                    continue
                
                # 存储映射：映射后的名字 -> 原始名字
                # 同时存储带扩展名和不带扩展名的版本
                mapping[new_name] = original_name
                if new_name.lower().endswith('.wav'):
                    mapping[new_name[:-4]] = original_name
        
        print(f"  成功解析 {len(mapping) // 2 if any(k.endswith('.wav') for k in mapping) else len(mapping)} 个映射条目")
        if len(mapping) > 0:
            # 显示前3个示例
            sample_count = 0
            for k, v in mapping.items():
                if k.endswith('.wav'):
                    print(f"    示例: {k} -> {v}")
                    sample_count += 1
                    if sample_count >= 3:
                        break
        
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}

# 配置参数
DATA_NAME = "subject1"
SAMPLING_RATE_EEG = 250
SAMPLING_RATE_AUDIO = 16000

# 数据路径
PROCESSED_DATA_PATH = 'A:/standard_data_noica/subject1'
NPY_PATH = os.path.join(PROCESSED_DATA_PATH, f"{DATA_NAME}_trials.npy")
INFO_JSON_PATH = os.path.join(PROCESSED_DATA_PATH, f"{DATA_NAME}_trial_info.json")
LABELS_CSV_PATH = os.path.join(PROCESSED_DATA_PATH, f"{DATA_NAME}_labels.csv")

# Benchmark文件路径（用于获取音频文件名）
MFF_DIRECTORY = r"A:\xufan1217_20251217_081811.mff"
BENCHMARK_FILE = os.path.join(MFF_DIRECTORY, "benchmark_1_10-1-1.txt")

# 音频路径
AUDIO_BASE_DIR = r"D:\D\research\audioset下载\clap_select_2s"
MAPPING_FILE = os.path.join(r"D:\D\research\audioset下载\audio_pairs_2s\分组音频\folder1", "file_mapping_folder1.csv")

# 输出目录
OUTPUT_DIR = os.path.join(PROCESSED_DATA_PATH, "eeg_audio_sync_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 频段定义
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# 基线窗口（0-4秒，用于基线矫正）
BASELINE_START = 0.0
BASELINE_END = 4.0

# 分析时间窗口（4-6秒，刺激期）
# 注意：EEG trial总长度是6秒，有效窗口是4-6秒
# 音频文件是2秒，对应EEG的4-6秒窗口
TIME_WINDOW_START = 4.0
TIME_WINDOW_END = 6.0


def load_eeg_data():
    """加载EEG数据"""
    if not os.path.exists(NPY_PATH):
        raise FileNotFoundError(f"未找到npy数据: {NPY_PATH}")
    if not os.path.exists(INFO_JSON_PATH):
        raise FileNotFoundError(f"未找到trial信息: {INFO_JSON_PATH}")
    if not os.path.exists(LABELS_CSV_PATH):
        raise FileNotFoundError(f"未找到标签文件: {LABELS_CSV_PATH}")

    # 加载数据
    trials = np.load(NPY_PATH, allow_pickle=True)
    with open(INFO_JSON_PATH, 'r', encoding='utf-8') as f:
        trial_info = json.load(f)
    
    # 加载标签
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    
    print(f"载入EEG数据:")
    print(f"  Trials形状: {trials.shape}")
    print(f"  标签数: {len(labels_df)}")
    
    # 检查数据形状
    if trials.ndim == 2:
        n_trials = trials.shape[0]
        flat_len = trials.shape[1]
        ch_labels = trial_info['channels']
        n_channels = len(ch_labels)
        samples_per_trial = flat_len // n_channels
        trials = trials.reshape(n_trials, n_channels, samples_per_trial)
        print(f"Reshape后的trials形状: {trials.shape}")
    
    return trials, trial_info, labels_df


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
    
    from scipy import interpolate
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
    analytic_signal = hilbert(data, axis=-1)
    phase = np.angle(analytic_signal)
    return phase


def compute_plv(phase1, phase2):
    """
    计算相位锁定值 (Phase Locking Value, PLV)
    
    参数:
        phase1, phase2: 相位数组，形状为 (n_timepoints,) 或 (n_trials, n_timepoints)
    
    返回:
        PLV值，范围 [0, 1]
    """
    if phase1.ndim == 1:
        phase1 = phase1[np.newaxis, :]
        phase2 = phase2[np.newaxis, :]
    
    n_trials, n_timepoints = phase1.shape
    
    # 计算相位差
    phase_diff = phase1 - phase2
    
    # 计算PLV：|mean(exp(i*phase_diff))|
    plv_per_trial = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
    
    # 返回所有trials的平均PLV
    return np.mean(plv_per_trial)


def compute_eeg_psd_time_series(eeg_data, sfreq, fmin=0.5, fmax=45.0,
                                window_size=1.0, overlap=0.5, debug=False):
    """
    计算EEG功率谱密度时间序列（所有通道的平均）
    
    参数:
        eeg_data: EEG数据 (n_channels, n_samples)
        sfreq: 采样率
        fmin, fmax: 频率范围
        window_size: 窗口大小（秒）
        overlap: 重叠比例
        debug: 是否输出调试信息
    
    返回:
        PSD时间序列
    """
    n_channels, n_samples = eeg_data.shape
    window_samples = int(window_size * sfreq)
    hop_samples = int(window_size * sfreq * (1 - overlap))
    
    if debug:
        print(f"        PSD计算参数: window_size={window_size}s ({window_samples}样本), overlap={overlap}, hop={hop_samples}样本")
        print(f"        数据长度: {n_samples}样本 ({n_samples/sfreq:.2f}秒)")
    
    psd_series = []
    
    for start_idx in range(0, n_samples - window_samples + 1, hop_samples):
        end_idx = start_idx + window_samples
        window_data = eeg_data[:, start_idx:end_idx]
        
        if debug and len(psd_series) == 0:
            print(f"        第1个窗口: {start_idx}-{end_idx}, 数据形状: {window_data.shape}")
            print(f"        窗口数据: min={np.nanmin(window_data):.6f}, max={np.nanmax(window_data):.6f}, std={np.nanstd(window_data):.6f}")
        
        # 计算所有通道的平均PSD
        all_psds = []
        for ch_idx, ch_data in enumerate(window_data):
            # 检查通道数据是否全为0
            if np.allclose(ch_data, 0):
                if debug and len(psd_series) == 0 and ch_idx == 0:
                    print(f"        警告: 通道{ch_idx}数据全为0")
                all_psds.append(0.0)
                continue
            
            try:
                freqs, psd = signal.welch(ch_data, sfreq, nperseg=min(window_samples, len(ch_data)))
                # 选择指定频率范围
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                psd_filtered = psd[freq_mask]
                freqs_filtered = freqs[freq_mask]
                
                if debug and len(psd_series) == 0 and ch_idx == 0:
                    print(f"        通道{ch_idx} PSD: 频率范围{fmin}-{fmax}Hz, 有效频率点数={len(freqs_filtered)}")
                    print(f"        PSD值范围: min={np.min(psd_filtered):.6e}, max={np.max(psd_filtered):.6e}")
                
                # 对频率维度积分，得到总功率
                if len(freqs_filtered) > 0:
                    total_power = np.trapz(psd_filtered, freqs_filtered)
                else:
                    total_power = 0.0
                all_psds.append(total_power)
            except Exception as e:
                if debug and len(psd_series) == 0:
                    print(f"        通道{ch_idx}计算PSD失败: {e}")
                all_psds.append(0.0)
        
        # 所有通道的平均功率
        mean_power = np.mean(all_psds) if len(all_psds) > 0 else 0.0
        psd_series.append(mean_power)
        
        if debug and len(psd_series) == 1:
            print(f"        第1个窗口的平均功率: {mean_power:.6e}")
    
    return np.array(psd_series)


def compute_correlation_with_delay(series1, series2, fs, max_delay_ms=200, debug=False):
    """
    计算考虑延迟的相关性
    
    参数:
        series1, series2: 两个时间序列
        fs: 采样率
        max_delay_ms: 最大延迟（毫秒）
        debug: 是否输出调试信息
    
    返回:
        (最佳相关系数, p值, 最佳延迟ms)
    """
    max_delay_samples = int(max_delay_ms * fs / 1000)
    
    best_corr = -np.inf
    best_pval = np.nan
    best_delay_ms = 0
    
    # 调试信息：输入数据检查
    if debug:
        print(f"      [相关性计算] 输入数据检查:")
        print(f"        series1长度: {len(series1)}, series2长度: {len(series2)}")
        print(f"        series1: min={np.nanmin(series1):.6e}, max={np.nanmax(series1):.6e}, mean={np.nanmean(series1):.6e}, std={np.nanstd(series1):.6e}")
        print(f"        series2: min={np.nanmin(series2):.6e}, max={np.nanmax(series2):.6e}, mean={np.nanmean(series2):.6e}, std={np.nanstd(series2):.6e}")
        print(f"        series1中NaN/Inf数量: {np.sum(~np.isfinite(series1))}")
        print(f"        series2中NaN/Inf数量: {np.sum(~np.isfinite(series2))}")
        print(f"        最大延迟样本数: {max_delay_samples} (对应{max_delay_ms}ms)")
    
    # 检查输入数据
    if len(series1) < 3 or len(series2) < 3:
        if debug:
            print(f"      [相关性计算] 错误: 数据长度不足: series1={len(series1)}, series2={len(series2)}")
        return best_corr, best_pval, best_delay_ms
    
    valid_delays = 0
    failed_reasons = {
        'length_too_short': 0,
        'not_enough_valid_points': 0,
        'zero_variance': 0,
        'pearsonr_exception': 0
    }
    
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
            failed_reasons['length_too_short'] += 1
            continue
        
        s1_final = s1_shifted[:min_len]
        s2_final = s2_aligned[:min_len]
        
        # 去除NaN和Inf
        mask = np.isfinite(s1_final) & np.isfinite(s2_final)
        valid_count = np.sum(mask)
        if valid_count < 3:
            failed_reasons['not_enough_valid_points'] += 1
            if debug and delay == 0:  # 只显示delay=0时的详细信息
                print(f"      [相关性计算] delay=0时: 总长度={min_len}, 有效点数={valid_count}")
            continue
        
        s1_clean = s1_final[mask]
        s2_clean = s2_final[mask]
        
        # 检查数据方差（如果方差为0，相关性无法计算）
        std1 = np.std(s1_clean)
        std2 = np.std(s2_clean)
        if std1 == 0 or std2 == 0:
            failed_reasons['zero_variance'] += 1
            if debug and delay == 0:  # 只显示delay=0时的详细信息
                print(f"      [相关性计算] delay=0时: series1方差={std1:.6e}, series2方差={std2:.6e}")
            continue
        
        # 计算相关性
        try:
            corr, pval = pearsonr(s1_clean, s2_clean)
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_pval = pval
                best_delay_ms = delay / fs * 1000
                valid_delays += 1
        except Exception as e:
            failed_reasons['pearsonr_exception'] += 1
            if debug and delay == 0:  # 只显示delay=0时的异常
                print(f"      [相关性计算] delay=0时pearsonr异常: {e}")
            continue
    
    if debug:
        print(f"      [相关性计算] 结果统计:")
        print(f"        有效延迟尝试数: {valid_delays}/{2*max_delay_samples+1}")
        print(f"        失败原因统计:")
        print(f"          长度太短: {failed_reasons['length_too_short']}")
        print(f"          有效点数不足: {failed_reasons['not_enough_valid_points']}")
        print(f"          方差为0: {failed_reasons['zero_variance']}")
        print(f"          pearsonr异常: {failed_reasons['pearsonr_exception']}")
        if best_corr == -np.inf:
            print(f"        警告: 所有延迟尝试都失败，返回-inf")
            print(f"        可能的原因:")
            if failed_reasons['zero_variance'] > 0:
                print(f"          - 数据方差为0（数据可能是常数）")
            if failed_reasons['not_enough_valid_points'] > 0:
                print(f"          - 有效数据点不足（可能包含太多NaN/Inf）")
            if failed_reasons['length_too_short'] > 0:
                print(f"          - 对齐后数据长度太短")
    
    return best_corr, best_pval, best_delay_ms


def apply_baseline_correction(data, fs, baseline_start=0.0, baseline_end=4.0):
    """
    对EEG数据进行基线矫正
    
    使用指定时间窗口的平均值作为基线，从整个trial中减去该基线。
    每个trial、每个通道独立进行基线矫正。
    
    参数:
        data: 形状为 (n_trials, n_channels, n_timepoints) 的数组
        fs: 采样率 (Hz)
        baseline_start: 基线窗口开始时间（秒），默认0.0
        baseline_end: 基线窗口结束时间（秒），默认4.0
    
    返回:
        基线矫正后的数据，形状与输入相同
    """
    n_trials, n_channels, n_timepoints = data.shape
    corrected_data = np.zeros_like(data)
    
    # 计算基线窗口的样本索引
    baseline_start_idx = int(round(baseline_start * fs))
    baseline_end_idx = int(round(baseline_end * fs))
    
    if baseline_start_idx < 0 or baseline_end_idx > n_timepoints:
        raise ValueError(
            f"基线窗口 [{baseline_start}, {baseline_end}] 超出数据范围 [0, {n_timepoints/fs:.2f}]"
        )
    if baseline_start_idx >= baseline_end_idx:
        raise ValueError(f"基线窗口开始时间必须小于结束时间")
    
    # 对每个trial、每个通道独立进行基线矫正
    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            # 提取基线窗口
            baseline_window = data[trial_idx, ch_idx, baseline_start_idx:baseline_end_idx]
            # 计算基线平均值
            baseline_mean = np.mean(baseline_window)
            # 从整个trial中减去基线平均值
            corrected_data[trial_idx, ch_idx, :] = data[trial_idx, ch_idx, :] - baseline_mean
    
    return corrected_data


def extract_time_window(data, fs, start_time, end_time):
    """提取指定时间窗口的数据"""
    start_idx = int(round(start_time * fs))
    end_idx = int(round(end_time * fs))
    return data[:, :, start_idx:end_idx]


def load_trial_audio_pair(wavfile, base_dir):
    """
    加载trial对应的左右音频
    
    参数:
        wavfile: 音频对文件名（如 "main_cat1_id1+main_cat2_id2.wav"）
        base_dir: 音频文件基础目录（如 "D:\D\research\audioset下载\clap_select_2s"）
    
    返回:
        (左音频信号, 右音频信号, 采样率)
    """
    try:
        # 解析音频对名称，得到类别和文件ID
        left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(wavfile, base_dir)
        
        # 构建路径：base_dir/类别名/文件ID.wav
        left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
        right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")
        
        # 调试信息：显示解析结果和路径
        if not hasattr(load_trial_audio_pair, '_debug_count'):
            load_trial_audio_pair._debug_count = 0
        
        if load_trial_audio_pair._debug_count < 3:
            print(f"    解析音频对: {wavfile}")
            print(f"      左: 类别={left_cat}, ID={left_id} -> {left_path}")
            print(f"      右: 类别={right_cat}, ID={right_id} -> {right_path}")
            print(f"      左文件存在: {os.path.exists(left_path)}")
            print(f"      右文件存在: {os.path.exists(right_path)}")
            load_trial_audio_pair._debug_count += 1
        
        if not os.path.exists(left_path):
            print(f"  警告: 左音频文件不存在: {left_path}")
            return None, None, None
        
        if not os.path.exists(right_path):
            print(f"  警告: 右音频文件不存在: {right_path}")
            return None, None, None
        
        yL, sr = load_mono_5s(left_path)
        yR, _ = load_mono_5s(right_path)
        
        return yL, yR, sr
    except Exception as e:
        print(f"  加载音频失败 {wavfile}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def plot_correlation_heatmap(results_df, save_path):
    """绘制相关性热图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 与左音频的相关性
    ax = axes[0, 0]
    corr_left = results_df['corr_left'].values
    im = ax.imshow(corr_left.reshape(-1, 1).T, cmap='RdBu_r', aspect='auto', 
                   vmin=-1, vmax=1)
    ax.set_title('Correlation with Left Audio', fontsize=12)
    ax.set_xlabel('Trial')
    ax.set_ylabel('')
    plt.colorbar(im, ax=ax)
    
    # 2. 与右音频的相关性
    ax = axes[0, 1]
    corr_right = results_df['corr_right'].values
    im = ax.imshow(corr_right.reshape(-1, 1).T, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1)
    ax.set_title('Correlation with Right Audio', fontsize=12)
    ax.set_xlabel('Trial')
    ax.set_ylabel('')
    plt.colorbar(im, ax=ax)
    
    # 3. 延迟分布
    ax = axes[1, 0]
    delay_left = results_df['delay_left_ms'].values
    delay_right = results_df['delay_right_ms'].values
    ax.hist(delay_left, bins=20, alpha=0.5, label='Left Audio', color='blue')
    ax.hist(delay_right, bins=20, alpha=0.5, label='Right Audio', color='red')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Delay Distribution')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # 4. 相关性对比（按标签）
    ax = axes[1, 1]
    label_0_mask = results_df['label'] == 0
    label_1_mask = results_df['label'] == 1
    
    # 选左时与左音频的相关性 vs 选右时与右音频的相关性
    corr_label_audio = []
    corr_nonlabel_audio = []
    
    for idx, row in results_df.iterrows():
        if row['label'] == 0:  # 选左
            corr_label_audio.append(row['corr_left'])
            corr_nonlabel_audio.append(row['corr_right'])
        else:  # 选右
            corr_label_audio.append(row['corr_right'])
            corr_nonlabel_audio.append(row['corr_left'])
    
    data_to_plot = [corr_label_audio, corr_nonlabel_audio]
    bp = ax.boxplot(data_to_plot, labels=['Label Audio', 'Non-Label Audio'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def plot_plv_results(plv_results, save_path):
    """绘制PLV结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. PLV矩阵（不同频段）
    ax = axes[0, 0]
    bands = list(FREQ_BANDS.keys())
    plv_left_mean = [np.mean(plv_results[f'plv_left_{band}']) for band in bands]
    plv_right_mean = [np.mean(plv_results[f'plv_right_{band}']) for band in bands]
    
    x = np.arange(len(bands))
    width = 0.35
    ax.bar(x - width/2, plv_left_mean, width, label='Left Audio', alpha=0.8)
    ax.bar(x + width/2, plv_right_mean, width, label='Right Audio', alpha=0.8)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Mean PLV')
    ax.set_title('PLV by Frequency Band')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PLV对比（按标签）
    ax = axes[0, 1]
    plv_label_audio = []
    plv_nonlabel_audio = []
    
    for band in bands:
        for idx, row in plv_results.iterrows():
            if row['label'] == 0:  # 选左
                plv_label_audio.append(row[f'plv_left_{band}'])
                plv_nonlabel_audio.append(row[f'plv_right_{band}'])
            else:  # 选右
                plv_label_audio.append(row[f'plv_right_{band}'])
                plv_nonlabel_audio.append(row[f'plv_left_{band}'])
    
    data_to_plot = [plv_label_audio, plv_nonlabel_audio]
    bp = ax.boxplot(data_to_plot, labels=['Label Audio', 'Non-Label Audio'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('PLV')
    ax.set_title('PLV Comparison (All Bands)')
    ax.grid(True, alpha=0.3)
    
    # 3. 每个频段的PLV对比
    ax = axes[1, 0]
    n_bands = len(bands)
    x_pos = np.arange(n_bands)
    width = 0.35
    
    plv_label_by_band = []
    plv_nonlabel_by_band = []
    for band in bands:
        label_vals = []
        nonlabel_vals = []
        for idx, row in plv_results.iterrows():
            if row['label'] == 0:
                label_vals.append(row[f'plv_left_{band}'])
                nonlabel_vals.append(row[f'plv_right_{band}'])
            else:
                label_vals.append(row[f'plv_right_{band}'])
                nonlabel_vals.append(row[f'plv_left_{band}'])
        plv_label_by_band.append(np.mean(label_vals))
        plv_nonlabel_by_band.append(np.mean(nonlabel_vals))
    
    ax.bar(x_pos - width/2, plv_label_by_band, width, label='Label Audio', alpha=0.8)
    ax.bar(x_pos + width/2, plv_nonlabel_by_band, width, label='Non-Label Audio', alpha=0.8)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Mean PLV')
    ax.set_title('PLV Comparison by Band')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. PLV热图（每个trial，每个频段）
    ax = axes[1, 1]
    plv_matrix = []
    for band in bands:
        plv_row = []
        for idx, row in plv_results.iterrows():
            if row['label'] == 0:
                plv_row.append(row[f'plv_left_{band}'])
            else:
                plv_row.append(row[f'plv_right_{band}'])
        plv_matrix.append(plv_row)
    
    im = ax.imshow(plv_matrix, cmap='hot', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Frequency Band')
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels(bands)
    ax.set_title('PLV Heatmap (Label Audio)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def plot_time_series_examples(results_df, trials_window, trials_info_from_benchmark, labels_df, 
                              audio_base_dir, mapping, save_path, n_examples=3):
    """
    绘制几个代表性trial的时间序列对比
    
    参数:
        results_df: 相关性结果DataFrame
        trials_window: 已经提取了4-6秒窗口的EEG数据 (n_trials, n_channels, n_samples)
        trials_info_from_benchmark: benchmark文件中的trial信息
        labels_df: 标签DataFrame
        audio_base_dir: 音频文件基础目录
        mapping: 音频文件名映射
        save_path: 保存路径
        n_examples: 每个标签绘制的示例数量
    """
    # 选择几个代表性的trial
    label_0_indices = results_df[results_df['label'] == 0].index[:n_examples]
    label_1_indices = results_df[results_df['label'] == 1].index[:n_examples]
    
    fig, axes = plt.subplots(n_examples * 2, 2, figsize=(16, 4 * n_examples))
    
    example_count = 0
    
    # 选左的trials
    for idx in label_0_indices:
        if example_count >= n_examples:
            break
        
        trial_idx = idx
        # trials_window已经是4-6秒窗口了，直接使用
        trial_data = trials_window[trial_idx]  # (n_channels, n_samples)，已经是4-6秒窗口
        
        # 获取音频文件名
        wavfile = None
        if trial_idx < len(trials_info_from_benchmark):
            wavfile = trials_info_from_benchmark[trial_idx].get('wavfile')
        
        if not wavfile:
            continue
        
        # 应用映射
        filename = os.path.basename(wavfile)
        if filename in mapping:
            wavfile = mapping[filename]
        elif filename.replace('.wav', '') in mapping:
            wavfile = mapping[filename.replace('.wav', '')]
        
        # 加载音频
        yL, yR, sr = load_trial_audio_pair(wavfile, audio_base_dir)
        if yL is None:
            continue
        
        # 提取包络（音频是2秒，对应EEG的4-6秒窗口）
        audio_duration = TIME_WINDOW_END - TIME_WINDOW_START  # 2秒
        envelope_L = extract_audio_envelope(yL, sr, SAMPLING_RATE_EEG, target_duration=audio_duration)
        envelope_R = extract_audio_envelope(yR, sr, SAMPLING_RATE_EEG, target_duration=audio_duration)
        
        # EEG数据已经是4-6秒窗口，直接平均所有通道
        eeg_window = np.mean(trial_data, axis=0)  # (n_samples,)
        envelope_L_window = envelope_L  # 音频已经是2秒长度
        envelope_R_window = envelope_R
        
        # 确保所有数组长度一致
        min_len = min(len(eeg_window), len(envelope_L_window), len(envelope_R_window))
        if min_len == 0:
            continue
        
        eeg_window = eeg_window[:min_len]
        envelope_L_window = envelope_L_window[:min_len]
        envelope_R_window = envelope_R_window[:min_len]
        
        time_axis = np.linspace(TIME_WINDOW_START, TIME_WINDOW_START + min_len/SAMPLING_RATE_EEG, min_len)
        
        # 绘制
        ax1 = axes[example_count * 2, 0]
        ax1.plot(time_axis, eeg_window, label='EEG (mean)', linewidth=1.5)
        ax1.plot(time_axis, envelope_L_window, label='Left Audio Envelope', 
                linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Trial {trial_idx+1} (Label=0, Left Choice)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[example_count * 2, 1]
        ax2.plot(time_axis, eeg_window, label='EEG (mean)', linewidth=1.5)
        ax2.plot(time_axis, envelope_R_window, label='Right Audio Envelope',
                linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Trial {trial_idx+1} (Label=0, Right Audio)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        example_count += 1
    
    # 选右的trials
    example_count = 0
    for idx in label_1_indices:
        if example_count >= n_examples:
            break
        
        trial_idx = idx
        # trials_window已经是4-6秒窗口了，直接使用
        trial_data = trials_window[trial_idx]  # (n_channels, n_samples)，已经是4-6秒窗口
        
        # 获取音频文件名
        wavfile = None
        if trial_idx < len(trials_info_from_benchmark):
            wavfile = trials_info_from_benchmark[trial_idx].get('wavfile')
        
        if not wavfile:
            continue
        
        # 应用映射
        filename = os.path.basename(wavfile)
        if filename in mapping:
            wavfile = mapping[filename]
        elif filename.replace('.wav', '') in mapping:
            wavfile = mapping[filename.replace('.wav', '')]
        
        yL, yR, sr = load_trial_audio_pair(wavfile, audio_base_dir)
        if yL is None:
            continue
        
        # 提取包络（音频是2秒，对应EEG的4-6秒窗口）
        audio_duration = TIME_WINDOW_END - TIME_WINDOW_START  # 2秒
        envelope_L = extract_audio_envelope(yL, sr, SAMPLING_RATE_EEG, target_duration=audio_duration)
        envelope_R = extract_audio_envelope(yR, sr, SAMPLING_RATE_EEG, target_duration=audio_duration)
        
        # EEG数据已经是4-6秒窗口，直接平均所有通道
        eeg_window = np.mean(trial_data, axis=0)  # (n_samples,)
        envelope_L_window = envelope_L  # 音频已经是2秒长度
        envelope_R_window = envelope_R
        
        # 确保所有数组长度一致
        min_len = min(len(eeg_window), len(envelope_L_window), len(envelope_R_window))
        if min_len == 0:
            continue
        
        eeg_window = eeg_window[:min_len]
        envelope_L_window = envelope_L_window[:min_len]
        envelope_R_window = envelope_R_window[:min_len]
        
        time_axis = np.linspace(TIME_WINDOW_START, TIME_WINDOW_START + min_len/SAMPLING_RATE_EEG, min_len)
        
        ax1 = axes[example_count * 2 + 1, 0]
        ax1.plot(time_axis, eeg_window, label='EEG (mean)', linewidth=1.5)
        ax1.plot(time_axis, envelope_L_window, label='Left Audio Envelope',
                linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Trial {trial_idx+1} (Label=1, Left Audio)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[example_count * 2 + 1, 1]
        ax2.plot(time_axis, eeg_window, label='EEG (mean)', linewidth=1.5)
        ax2.plot(time_axis, envelope_R_window, label='Right Audio Envelope',
                linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Trial {trial_idx+1} (Label=1, Right Choice)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        example_count += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def analyze_audio_category_effect(plv_diff_df, correlation_df, output_dir):
    """
    分析不同音频类别对PLV差异的影响
    
    参数:
        plv_diff_df: PLV差异方向DataFrame
        correlation_df: 相关性结果DataFrame
        output_dir: 输出目录
    """
    from collections import defaultdict
    
    # 解析每个trial的音频类别
    category_stats = defaultdict(lambda: {
        'positive_count': 0,
        'negative_count': 0,
        'positive_trials': [],
        'negative_trials': [],
        'differences': []
    })
    
    # 统计每个频段
    category_stats_by_band = defaultdict(lambda: defaultdict(lambda: {
        'positive_count': 0,
        'negative_count': 0,
        'differences': []
    }))
    
    for idx, row in plv_diff_df.iterrows():
        wavfile = row['wavfile']
        band = row['band']
        direction = row['direction']
        diff = row['difference']
        
        try:
            # 解析音频对名称，提取左右类别
            left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(wavfile, AUDIO_BASE_DIR)
            
            # 根据label确定哪个是"对应标签"的音频
            if row['label'] == 0:
                # 选左，左音频是对应标签
                corresponding_cat = left_cat
                noncorresponding_cat = right_cat
            else:
                # 选右，右音频是对应标签
                corresponding_cat = right_cat
                noncorresponding_cat = left_cat
            
            # 统计对应标签音频类别
            if direction == 'positive':
                category_stats[corresponding_cat]['positive_count'] += 1
                category_stats[corresponding_cat]['positive_trials'].append(row['trial_idx'])
            else:
                category_stats[corresponding_cat]['negative_count'] += 1
                category_stats[corresponding_cat]['negative_trials'].append(row['trial_idx'])
            
            category_stats[corresponding_cat]['differences'].append(diff)
            
            # 按频段统计
            if direction == 'positive':
                category_stats_by_band[band][corresponding_cat]['positive_count'] += 1
            else:
                category_stats_by_band[band][corresponding_cat]['negative_count'] += 1
            category_stats_by_band[band][corresponding_cat]['differences'].append(diff)
            
        except Exception as e:
            # 解析失败，跳过
            continue
    
    # 生成统计报告
    print("\n  音频类别对PLV差异的影响统计:")
    print("  " + "="*80)
    
    # 按总出现次数排序
    sorted_categories = sorted(category_stats.items(), 
                               key=lambda x: x[1]['positive_count'] + x[1]['negative_count'], 
                               reverse=True)
    
    category_summary = []
    for cat, stats in sorted_categories:
        total = stats['positive_count'] + stats['negative_count']
        if total == 0:
            continue
        
        pos_rate = stats['positive_count'] / total * 100
        neg_rate = stats['negative_count'] / total * 100
        mean_diff = np.mean(stats['differences']) if stats['differences'] else 0
        
        category_summary.append({
            'category': cat,
            'total_trials': total,
            'positive_count': stats['positive_count'],
            'negative_count': stats['negative_count'],
            'positive_rate': pos_rate,
            'negative_rate': neg_rate,
            'mean_difference': mean_diff,
            'std_difference': np.std(stats['differences']) if stats['differences'] else 0
        })
        
        print(f"\n  {cat}:")
        print(f"    总trials数: {total}")
        print(f"    正差异: {stats['positive_count']} ({pos_rate:.1f}%)")
        print(f"    负差异: {stats['negative_count']} ({neg_rate:.1f}%)")
        print(f"    平均差异: {mean_diff:.6f} ± {np.std(stats['differences']):.6f}")
        
        # 特别标注键盘声
        if 'keyboard' in cat.lower() or '键盘' in cat.lower():
            print(f"    ⚠️  注意: 此类别负差异比例较高!")
    
    # 保存到CSV
    category_summary_df = pd.DataFrame(category_summary)
    category_summary_path = os.path.join(output_dir, "audio_category_plv_effect.csv")
    category_summary_df.to_csv(category_summary_path, index=False, encoding='utf-8-sig')
    print(f"\n  类别统计已保存: {category_summary_path}")
    
    # 按频段统计
    print("\n  按频段统计:")
    for band_name in FREQ_BANDS.keys():
        print(f"\n  {band_name}频段:")
        band_stats = category_stats_by_band[band_name]
        
        band_summary = []
        for cat, stats in sorted(band_stats.items(), 
                                key=lambda x: x[1]['positive_count'] + x[1]['negative_count'], 
                                reverse=True):
            total = stats['positive_count'] + stats['negative_count']
            if total == 0:
                continue
            
            pos_rate = stats['positive_count'] / total * 100
            neg_rate = stats['negative_count'] / total * 100
            mean_diff = np.mean(stats['differences']) if stats['differences'] else 0
            
            band_summary.append({
                'band': band_name,
                'category': cat,
                'total_trials': total,
                'positive_count': stats['positive_count'],
                'negative_count': stats['negative_count'],
                'positive_rate': pos_rate,
                'negative_rate': neg_rate,
                'mean_difference': mean_diff
            })
            
            print(f"    {cat}: 正差异={stats['positive_count']} ({pos_rate:.1f}%), "
                  f"负差异={stats['negative_count']} ({neg_rate:.1f}%), "
                  f"平均差异={mean_diff:.6f}")
        
        # 保存按频段的统计
        if band_summary:
            band_summary_df = pd.DataFrame(band_summary)
            band_summary_path = os.path.join(output_dir, f"audio_category_plv_effect_{band_name}.csv")
            band_summary_df.to_csv(band_summary_path, index=False, encoding='utf-8-sig')
    
    print(f"\n  按频段的类别统计已保存到: {output_dir}")
    
    # 特别分析：当某个类别作为注意目标 vs 非注意目标时的PLV差异
    print("\n  " + "="*80)
    print("  🔍 深度分析：音频类别作为注意目标 vs 非注意目标时的PLV差异")
    print("  " + "="*80)
    
    # 统计每个类别在作为注意目标和非注意目标时的表现
    category_as_target = defaultdict(lambda: {'plv_values': [], 'trial_indices': []})
    category_as_nontarget = defaultdict(lambda: {'plv_values': [], 'trial_indices': []})
    
    for idx, row in plv_diff_df.iterrows():
        wavfile = row['wavfile']
        band = row['band']
        plv_corr = row['plv_corresponding']
        plv_noncorr = row['plv_noncorresponding']
        
        try:
            left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(wavfile, AUDIO_BASE_DIR)
            
            if row['label'] == 0:
                # 选左，左音频是注意目标，右音频是非注意目标
                target_cat = left_cat
                nontarget_cat = right_cat
            else:
                # 选右，右音频是注意目标，左音频是非注意目标
                target_cat = right_cat
                nontarget_cat = left_cat
            
            # 记录作为注意目标时的PLV
            category_as_target[target_cat]['plv_values'].append(plv_corr)
            category_as_target[target_cat]['trial_indices'].append(row['trial_idx'])
            
            # 记录作为非注意目标时的PLV
            category_as_nontarget[nontarget_cat]['plv_values'].append(plv_noncorr)
            category_as_nontarget[nontarget_cat]['trial_indices'].append(row['trial_idx'])
            
        except Exception as e:
            continue
    
    # 分析关键类别（样本量足够大的）
    print("\n  关键发现（带统计检验）：")
    from scipy.stats import ttest_rel, wilcoxon, ttest_ind
    
    category_comparison_results = []
    
    for cat in sorted(set(list(category_as_target.keys()) + list(category_as_nontarget.keys()))):
        target_plvs = category_as_target[cat]['plv_values']
        nontarget_plvs = category_as_nontarget[cat]['plv_values']
        
        if len(target_plvs) >= 5 and len(nontarget_plvs) >= 5:  # 至少5个样本
            mean_target = np.mean(target_plvs)
            mean_nontarget = np.mean(nontarget_plvs)
            std_target = np.std(target_plvs)
            std_nontarget = np.std(nontarget_plvs)
            diff = mean_target - mean_nontarget
            
            print(f"\n  {cat}:")
            print(f"    作为注意目标时的平均PLV: {mean_target:.6f} ± {std_target:.6f} (n={len(target_plvs)})")
            print(f"    作为非注意目标时的平均PLV: {mean_nontarget:.6f} ± {std_nontarget:.6f} (n={len(nontarget_plvs)})")
            print(f"    差异（注意目标 - 非注意目标）: {diff:.6f}")
            
            # 配对t检验（如果两个列表长度相同，说明是配对的）
            if len(target_plvs) == len(nontarget_plvs):
                # 配对检验
                try:
                    t_stat_paired, p_val_paired = ttest_rel(target_plvs, nontarget_plvs)
                    print(f"    配对t检验: t={t_stat_paired:.4f}, p={p_val_paired:.4f}", end="")
                    if p_val_paired < 0.001:
                        print(" ***")
                    elif p_val_paired < 0.01:
                        print(" **")
                    elif p_val_paired < 0.05:
                        print(" *")
                    else:
                        print(" (n.s.)")
                except Exception as e:
                    print(f"    配对t检验: 无法计算 ({e})")
                    t_stat_paired, p_val_paired = np.nan, np.nan
                
                # Wilcoxon符号秩检验
                try:
                    w_stat, p_val_w = wilcoxon(target_plvs, nontarget_plvs, alternative='two-sided')
                    print(f"    Wilcoxon检验: W={w_stat:.4f}, p={p_val_w:.4f}", end="")
                    if p_val_w < 0.001:
                        print(" ***")
                    elif p_val_w < 0.01:
                        print(" **")
                    elif p_val_w < 0.05:
                        print(" *")
                    else:
                        print(" (n.s.)")
                except Exception as e:
                    print(f"    Wilcoxon检验: 无法计算 ({e})")
                    w_stat, p_val_w = np.nan, np.nan
                
                # Cohen's d (配对样本)
                try:
                    diff_std = np.std(np.array(target_plvs) - np.array(nontarget_plvs))
                    if diff_std > 0:
                        cohens_d = diff / diff_std
                        print(f"    Cohen's d: {cohens_d:.4f}", end="")
                        if abs(cohens_d) < 0.2:
                            print(" (很小)")
                        elif abs(cohens_d) < 0.5:
                            print(" (小)")
                        elif abs(cohens_d) < 0.8:
                            print(" (中等)")
                        else:
                            print(" (大)")
                    else:
                        print(f"    Cohen's d: 无法计算（差异标准差为0）")
                        cohens_d = np.nan
                except Exception as e:
                    print(f"    Cohen's d: 无法计算 ({e})")
                    cohens_d = np.nan
                
                category_comparison_results.append({
                    'category': cat,
                    'n_trials': len(target_plvs),
                    'mean_as_target': mean_target,
                    'std_as_target': std_target,
                    'mean_as_nontarget': mean_nontarget,
                    'std_as_nontarget': std_nontarget,
                    'difference': diff,
                    't_stat_paired': t_stat_paired if 't_stat_paired' in locals() else np.nan,
                    'p_val_paired': p_val_paired if 'p_val_paired' in locals() else np.nan,
                    'w_stat': w_stat if 'w_stat' in locals() else np.nan,
                    'p_val_wilcoxon': p_val_w if 'p_val_w' in locals() else np.nan,
                    'cohens_d': cohens_d if 'cohens_d' in locals() else np.nan
                })
            else:
                # 非配对检验（独立样本t检验）
                try:
                    t_stat_ind, p_val_ind = ttest_ind(target_plvs, nontarget_plvs)
                    print(f"    独立样本t检验: t={t_stat_ind:.4f}, p={p_val_ind:.4f}", end="")
                    if p_val_ind < 0.001:
                        print(" ***")
                    elif p_val_ind < 0.01:
                        print(" **")
                    elif p_val_ind < 0.05:
                        print(" *")
                    else:
                        print(" (n.s.)")
                except Exception as e:
                    print(f"    独立样本t检验: 无法计算 ({e})")
                    t_stat_ind, p_val_ind = np.nan, np.nan
                
                category_comparison_results.append({
                    'category': cat,
                    'n_target': len(target_plvs),
                    'n_nontarget': len(nontarget_plvs),
                    'mean_as_target': mean_target,
                    'std_as_target': std_target,
                    'mean_as_nontarget': mean_nontarget,
                    'std_as_nontarget': std_nontarget,
                    'difference': diff,
                    't_stat_independent': t_stat_ind if 't_stat_ind' in locals() else np.nan,
                    'p_val_independent': p_val_ind if 'p_val_ind' in locals() else np.nan
                })
            
            # 解释结果
            if diff < -0.01:
                sig_text = ""
                if 'p_val_paired' in locals() and not np.isnan(p_val_paired):
                    if p_val_paired < 0.05:
                        sig_text = "（统计显著）"
                    else:
                        sig_text = "（统计不显著）"
                elif 'p_val_ind' in locals() and not np.isnan(p_val_ind):
                    if p_val_ind < 0.05:
                        sig_text = "（统计显著）"
                    else:
                        sig_text = "（统计不显著）"
                
                print(f"    ⚠️  注意：作为注意目标时PLV反而更低{sig_text}！这可能表示：")
                print(f"        - 该类别作为注意目标时被抑制")
                print(f"        - 该类别的声学特征难以与EEG建立稳定的相位同步")
                print(f"        - 可能存在注意力反转效应")
            elif diff > 0.01:
                sig_text = ""
                if 'p_val_paired' in locals() and not np.isnan(p_val_paired):
                    if p_val_paired < 0.05:
                        sig_text = "（统计显著）"
                    else:
                        sig_text = "（统计不显著）"
                elif 'p_val_ind' in locals() and not np.isnan(p_val_ind):
                    if p_val_ind < 0.05:
                        sig_text = "（统计显著）"
                    else:
                        sig_text = "（统计不显著）"
                print(f"    ✅ 作为注意目标时PLV更高{sig_text}，符合预期")
            else:
                print(f"    ➡️  作为注意目标和非注意目标时PLV相近")
    
    # 保存统计结果到CSV
    if category_comparison_results:
        comparison_df = pd.DataFrame(category_comparison_results)
        comparison_path = os.path.join(output_dir, "category_target_vs_nontarget_statistics.csv")
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"\n  类别对比统计结果已保存: {comparison_path}")
    
    # 特别关注键盘声
    keyboard_target = category_as_target.get('Computer keyboard', {'plv_values': []})['plv_values']
    keyboard_nontarget = category_as_nontarget.get('Computer keyboard', {'plv_values': []})['plv_values']
    
    if len(keyboard_target) > 0 and len(keyboard_nontarget) > 0:
        print("\n  " + "="*80)
        print("  ⌨️  键盘声特别分析（带统计检验）:")
        print("  " + "="*80)
        print(f"  键盘声作为注意目标时的平均PLV: {np.mean(keyboard_target):.6f} ± {np.std(keyboard_target):.6f} (n={len(keyboard_target)})")
        print(f"  键盘声作为非注意目标时的平均PLV: {np.mean(keyboard_nontarget):.6f} ± {np.std(keyboard_nontarget):.6f} (n={len(keyboard_nontarget)})")
        keyboard_diff = np.mean(keyboard_target) - np.mean(keyboard_nontarget)
        print(f"  差异: {keyboard_diff:.6f}")
        
        # 统计检验
        if len(keyboard_target) == len(keyboard_nontarget):
            # 配对检验
            try:
                t_stat_kb, p_val_kb = ttest_rel(keyboard_target, keyboard_nontarget)
                print(f"  配对t检验: t={t_stat_kb:.4f}, p={p_val_kb:.4f}", end="")
                if p_val_kb < 0.001:
                    print(" ***")
                elif p_val_kb < 0.01:
                    print(" **")
                elif p_val_kb < 0.05:
                    print(" *")
                else:
                    print(" (n.s.)")
            except Exception as e:
                print(f"  配对t检验: 无法计算 ({e})")
                t_stat_kb, p_val_kb = np.nan, np.nan
            
            try:
                w_stat_kb, p_val_w_kb = wilcoxon(keyboard_target, keyboard_nontarget, alternative='two-sided')
                print(f"  Wilcoxon检验: W={w_stat_kb:.4f}, p={p_val_w_kb:.4f}", end="")
                if p_val_w_kb < 0.001:
                    print(" ***")
                elif p_val_w_kb < 0.01:
                    print(" **")
                elif p_val_w_kb < 0.05:
                    print(" *")
                else:
                    print(" (n.s.)")
            except Exception as e:
                print(f"  Wilcoxon检验: 无法计算 ({e})")
            
            try:
                diff_std_kb = np.std(np.array(keyboard_target) - np.array(keyboard_nontarget))
                if diff_std_kb > 0:
                    cohens_d_kb = keyboard_diff / diff_std_kb
                    print(f"  Cohen's d: {cohens_d_kb:.4f}", end="")
                    if abs(cohens_d_kb) < 0.2:
                        print(" (很小)")
                    elif abs(cohens_d_kb) < 0.5:
                        print(" (小)")
                    elif abs(cohens_d_kb) < 0.8:
                        print(" (中等)")
                    else:
                        print(" (大)")
            except Exception as e:
                print(f"  Cohen's d: 无法计算 ({e})")
        else:
            # 独立样本检验
            try:
                t_stat_kb, p_val_kb = ttest_ind(keyboard_target, keyboard_nontarget)
                print(f"  独立样本t检验: t={t_stat_kb:.4f}, p={p_val_kb:.4f}", end="")
                if p_val_kb < 0.001:
                    print(" ***")
                elif p_val_kb < 0.01:
                    print(" **")
                elif p_val_kb < 0.05:
                    print(" *")
                else:
                    print(" (n.s.)")
            except Exception as e:
                print(f"  独立样本t检验: 无法计算 ({e})")
        
        if keyboard_diff < 0:
            sig_status = ""
            if 'p_val_kb' in locals() and not np.isnan(p_val_kb):
                if p_val_kb < 0.05:
                    sig_status = "（统计显著，p < 0.05）"
                else:
                    sig_status = f"（统计不显著，p = {p_val_kb:.4f}）"
            
            print(f"\n  💡 解释：")
            print(f"    当键盘声是注意目标时，PLV反而比非注意目标时低 {abs(keyboard_diff):.6f} {sig_status}")
            if 'p_val_kb' in locals() and not np.isnan(p_val_kb) and p_val_kb < 0.05:
                print(f"    这可能意味着：")
                print(f"    1. 键盘声作为注意目标时，可能触发了某种抑制机制")
                print(f"    2. 键盘声的瞬态、不规则特征使其难以与EEG建立稳定的相位同步")
                print(f"    3. 当需要主动注意键盘声时，可能反而抑制了其与EEG的同步性")
                print(f"    4. 这是一种'注意力反转效应'：注意目标反而同步性更低")
            else:
                print(f"    注意：虽然差异方向为负，但统计检验不显著，需要更多样本验证")


def main():
    """主函数"""
    print("="*60)
    print("开始计算xufan subject1的EEG-音频同步性")
    print("="*60)
    
    # 1. 加载EEG数据
    print("\n1. 加载EEG数据...")
    trials, trial_info, labels_df = load_eeg_data()
    
    # 2. 加载benchmark文件获取音频信息
    print("\n2. 加载benchmark文件获取音频信息...")
    if not os.path.exists(BENCHMARK_FILE):
        print(f"警告: benchmark文件不存在: {BENCHMARK_FILE}")
        trials_info_from_benchmark = []
    else:
        trials_info_from_benchmark = parse_experiment_data_txt(BENCHMARK_FILE)
        print(f"从benchmark文件读取了 {len(trials_info_from_benchmark)} 个trial信息")
    
    # 3. 加载音频映射文件
    print("\n3. 加载音频映射文件...")
    if not os.path.exists(MAPPING_FILE):
        print(f"警告: 映射文件不存在: {MAPPING_FILE}")
        mapping = {}
    else:
        # 使用自定义的映射解析函数
        mapping = parse_audio_mapping_custom(MAPPING_FILE)
        print(f"加载了 {len(mapping)} 个映射")
        
        # 应用映射转换音频文件名
        mapped_count = 0
        for trial_info_item in trials_info_from_benchmark:
            if 'wavfile' in trial_info_item:
                original_wavfile = trial_info_item['wavfile']
                filename = os.path.basename(original_wavfile)
                # 尝试匹配（带扩展名和不带扩展名）
                if filename in mapping:
                    trial_info_item['wavfile'] = mapping[filename]
                    mapped_count += 1
                    if mapped_count <= 3:  # 显示前3个映射示例
                        print(f"    映射: {filename} -> {mapping[filename]}")
                elif filename.replace('.wav', '') in mapping:
                    trial_info_item['wavfile'] = mapping[filename.replace('.wav', '')]
                    mapped_count += 1
                    if mapped_count <= 3:  # 显示前3个映射示例
                        print(f"    映射: {filename} -> {mapping[filename.replace('.wav', '')]}")
        
        print(f"  成功映射 {mapped_count} 个音频文件名")
    
    # 3.5. 基线矫正
    print(f"\n3.5. 基线矫正 (基线窗口: {BASELINE_START}-{BASELINE_END}s)...")
    trials = apply_baseline_correction(trials, SAMPLING_RATE_EEG, 
                                       BASELINE_START, BASELINE_END)
    print(f"基线矫正完成")
    
    # 4. 提取时间窗口
    print(f"\n4. 提取时间窗口 ({TIME_WINDOW_START}-{TIME_WINDOW_END}s)...")
    trials_window = extract_time_window(trials, SAMPLING_RATE_EEG, 
                                        TIME_WINDOW_START, TIME_WINDOW_END)
    print(f"时间窗口数据形状: {trials_window.shape}")
    
    # 确保trials_info_from_benchmark和EEG数据数量匹配
    min_n = min(len(trials_info_from_benchmark), trials_window.shape[0])
    if len(trials_info_from_benchmark) != trials_window.shape[0]:
        print(f"警告: benchmark trial数量({len(trials_info_from_benchmark)})与EEG trial数量({trials_window.shape[0]})不匹配，将使用前{min_n}个")
        trials_info_from_benchmark = trials_info_from_benchmark[:min_n]
        trials_window = trials_window[:min_n]
        labels_df = labels_df.iloc[:min_n]
    
    # 5. 存储结果
    correlation_results = []
    plv_results = []
    
    # 6. 处理每个trial
    print("\n5. 处理每个trial...")
    n_trials = trials_window.shape[0]
    
    skipped_no_wavfile = 0
    skipped_no_audio = 0
    skipped_no_label = 0
    
    for trial_idx in range(n_trials):
        if trial_idx % 20 == 0:
            print(f"  处理trial {trial_idx+1}/{n_trials}...")
        
        # 获取标签
        label_row = labels_df.iloc[trial_idx]
        label = label_row['Label']
        
        if pd.isna(label):
            skipped_no_label += 1
            continue
        
        label = int(label)
        
        # 获取音频文件名（优先从benchmark文件）
        wavfile = None
        if trial_idx < len(trials_info_from_benchmark):
            wavfile = trials_info_from_benchmark[trial_idx].get('wavfile')
        
        # 如果benchmark中没有，尝试从trial_info获取
        if not wavfile:
            trial_times = trial_info.get('trial_times', [])
            if trial_idx < len(trial_times):
                wavfile = trial_times[trial_idx].get('wavfile')
        
        if not wavfile:
            skipped_no_wavfile += 1
            if trial_idx < 5:  # 只打印前5个作为示例
                print(f"    警告: Trial {trial_idx+1} 没有音频文件名")
            continue
        
        # wavfile应该已经从映射中转换过了（在主函数中已经应用）
        # 但为了安全，如果还没有映射，这里再检查一次
        filename = os.path.basename(wavfile)
        if filename.endswith('.wav') and len(filename) <= 10:  # 类似a001.wav的短文件名
            # 可能还没有映射，尝试映射
            if filename in mapping:
                wavfile = mapping[filename]
                if trial_idx < 5:  # 调试信息
                    print(f"    Trial {trial_idx+1}: 应用映射 {filename} -> {wavfile}")
            elif filename.replace('.wav', '') in mapping:
                wavfile = mapping[filename.replace('.wav', '')]
                if trial_idx < 5:  # 调试信息
                    print(f"    Trial {trial_idx+1}: 应用映射 {filename} -> {wavfile}")
        
        # 调试信息：显示前几个trial的音频文件名
        if trial_idx < 5:
            print(f"    Trial {trial_idx+1}: 最终音频文件名 = {wavfile}")
        
        # 加载左右音频（使用原始名称）
        yL, yR, sr = load_trial_audio_pair(wavfile, AUDIO_BASE_DIR)
        if yL is None or yR is None:
            skipped_no_audio += 1
            if trial_idx < 5:  # 只打印前5个作为示例
                print(f"    警告: Trial {trial_idx+1} 无法加载音频文件: {wavfile}")
            continue
        
        # 提取音频包络和RMS
        # 音频是2秒，对应EEG的4-6秒窗口，所以直接提取2秒的包络
        audio_duration = TIME_WINDOW_END - TIME_WINDOW_START  # 2秒
        envelope_L = extract_audio_envelope(yL, sr, SAMPLING_RATE_EEG, target_duration=audio_duration)
        envelope_R = extract_audio_envelope(yR, sr, SAMPLING_RATE_EEG, target_duration=audio_duration)
        rms_L = extract_audio_rms(yL, sr, SAMPLING_RATE_EEG)
        rms_R = extract_audio_rms(yR, sr, SAMPLING_RATE_EEG)
        
        # RMS也需要调整为2秒长度
        if len(rms_L) != len(envelope_L):
            # 插值到相同长度（2秒）
            from scipy import interpolate
            t_original = np.linspace(0, len(rms_L)/SAMPLING_RATE_EEG, len(rms_L))
            t_target = np.linspace(0, audio_duration, len(envelope_L))
            f_L = interpolate.interp1d(t_original, rms_L, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
            f_R = interpolate.interp1d(t_original, rms_R, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
            rms_L = f_L(t_target)
            rms_R = f_R(t_target)
        
        # trials_window已经是4-6秒的窗口了，直接使用全部数据
        trial_data = trials_window[trial_idx]  # (n_channels, n_samples)，已经是4-6秒窗口
        
        # 调试信息：检查trial_data的原始值（只显示前3个trial）
        if trial_idx < 3:
            print(f"      Trial数据形状: {trial_data.shape}")
            print(f"      Trial数据: min={np.nanmin(trial_data):.6f}, max={np.nanmax(trial_data):.6f}, mean={np.nanmean(trial_data):.6f}, std={np.nanstd(trial_data):.6f}")
            print(f"      Trial数据中NaN/Inf数量: {np.sum(~np.isfinite(trial_data))}")
            # 检查是否全为0
            if np.allclose(trial_data, 0):
                print(f"      警告: Trial数据全为0！")
            # 检查每个通道的std
            channel_stds = np.std(trial_data, axis=1)
            print(f"      各通道std范围: min={np.min(channel_stds):.6f}, max={np.max(channel_stds):.6f}, mean={np.mean(channel_stds):.6f}")
        
        # 音频数据已经是2秒长度，直接使用
        envelope_L_window = envelope_L
        envelope_R_window = envelope_R
        rms_L_window = rms_L
        rms_R_window = rms_R
        
        # 计算EEG PSD时间序列（所有通道的平均）
        # 使用整个窗口数据（4-6秒）
        debug_psd = (trial_idx < 3)  # 前3个trial输出详细调试信息
        eeg_psd = compute_eeg_psd_time_series(trial_data, SAMPLING_RATE_EEG, debug=debug_psd)
        
        # 对齐长度
        min_len = min(len(eeg_psd), len(rms_L_window))
        eeg_psd_aligned = eeg_psd[:min_len]
        rms_L_aligned = rms_L_window[:min_len]
        rms_R_aligned = rms_R_window[:min_len]
        
        # 调试信息：检查数据质量（只显示前3个trial）
        if trial_idx < 3:
            print(f"    Trial {trial_idx+1} 数据检查:")
            print(f"      EEG PSD长度: {len(eeg_psd)}, RMS_L长度: {len(rms_L_window)}, RMS_R长度: {len(rms_R_window)}")
            print(f"      对齐后长度: {min_len}")
            print(f"      EEG PSD: min={np.nanmin(eeg_psd_aligned):.6f}, max={np.nanmax(eeg_psd_aligned):.6f}, mean={np.nanmean(eeg_psd_aligned):.6f}, std={np.nanstd(eeg_psd_aligned):.6f}")
            print(f"      RMS_L: min={np.nanmin(rms_L_aligned):.6f}, max={np.nanmax(rms_L_aligned):.6f}, mean={np.nanmean(rms_L_aligned):.6f}, std={np.nanstd(rms_L_aligned):.6f}")
            print(f"      RMS_R: min={np.nanmin(rms_R_aligned):.6f}, max={np.nanmax(rms_R_aligned):.6f}, mean={np.nanmean(rms_R_aligned):.6f}, std={np.nanstd(rms_R_aligned):.6f}")
            print(f"      EEG PSD中NaN/Inf数量: {np.sum(~np.isfinite(eeg_psd_aligned))}")
            print(f"      RMS_L中NaN/Inf数量: {np.sum(~np.isfinite(rms_L_aligned))}")
            print(f"      RMS_R中NaN/Inf数量: {np.sum(~np.isfinite(rms_R_aligned))}")
        
        # 计算延迟相关性
        debug_mode = (trial_idx < 3)  # 前3个trial输出详细调试信息
        
        # 先检查数据，如果可能返回-inf，也输出调试信息
        if trial_idx < 3 or len(eeg_psd_aligned) < 5 or np.nanstd(eeg_psd_aligned) == 0 or np.nanstd(rms_L_aligned) == 0 or np.nanstd(rms_R_aligned) == 0:
            debug_mode = True
        
        corr_left, pval_left, delay_left = compute_correlation_with_delay(
            eeg_psd_aligned, rms_L_aligned, SAMPLING_RATE_EEG, debug=debug_mode)
        corr_right, pval_right, delay_right = compute_correlation_with_delay(
            eeg_psd_aligned, rms_R_aligned, SAMPLING_RATE_EEG, debug=debug_mode)
        
        # 调试信息：显示相关性结果（只显示前3个trial）
        if trial_idx < 3:
            print(f"      相关性结果: corr_left={corr_left:.6f}, corr_right={corr_right:.6f}")
        
        # 如果相关性是-inf，输出额外调试信息
        if corr_left == -np.inf or corr_right == -np.inf:
            # 统计-inf的trial数量（使用函数属性）
            if not hasattr(compute_correlation_with_delay, '_inf_count'):
                compute_correlation_with_delay._inf_count = 0
            
            compute_correlation_with_delay._inf_count += 1
            inf_count = compute_correlation_with_delay._inf_count
            
            if inf_count <= 10:  # 前10个-inf的trial输出详细信息
                print(f"      [警告] Trial {trial_idx+1} 相关性为-inf (第{inf_count}个-inf的trial):")
                if corr_left == -np.inf:
                    print(f"        corr_left=-inf, 数据长度: EEG_PSD={len(eeg_psd_aligned)}, RMS_L={len(rms_L_aligned)}")
                    print(f"        EEG_PSD std: {np.nanstd(eeg_psd_aligned):.6e}")
                    print(f"        RMS_L std: {np.nanstd(rms_L_aligned):.6e}")
                    # 重新计算一次以获取详细调试信息
                    print(f"        [重新计算corr_left以获取详细调试信息]")
                    compute_correlation_with_delay(eeg_psd_aligned, rms_L_aligned, SAMPLING_RATE_EEG, debug=True)
                if corr_right == -np.inf:
                    print(f"        corr_right=-inf, 数据长度: EEG_PSD={len(eeg_psd_aligned)}, RMS_R={len(rms_R_aligned)}")
                    print(f"        EEG_PSD std: {np.nanstd(eeg_psd_aligned):.6e}")
                    print(f"        RMS_R std: {np.nanstd(rms_R_aligned):.6e}")
                    # 重新计算一次以获取详细调试信息
                    print(f"        [重新计算corr_right以获取详细调试信息]")
                    compute_correlation_with_delay(eeg_psd_aligned, rms_R_aligned, SAMPLING_RATE_EEG, debug=True)
        
        correlation_results.append({
            'trial_idx': trial_idx + 1,
            'label': label,
            'wavfile': wavfile,
            'corr_left': corr_left,
            'pval_left': pval_left,
            'delay_left_ms': delay_left,
            'corr_right': corr_right,
            'pval_right': pval_right,
            'delay_right_ms': delay_right,
        })
        
        # 计算包络相位同步性（PLV）
        # 提取音频包络的相位
        phase_envelope_L = extract_phase(envelope_L_window)
        phase_envelope_R = extract_phase(envelope_R_window)
        
        # 对每个频段计算PLV
        plv_result = {
            'trial_idx': trial_idx + 1,
            'label': label,
            'wavfile': wavfile,
        }
        
        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            # EEG在该频段的相位
            eeg_filtered = bandpass_filter(trial_data, SAMPLING_RATE_EEG, fmin, fmax)
            eeg_phase = extract_phase(eeg_filtered)  # (n_channels, n_samples)
            
            # 计算所有通道的平均相位
            eeg_phase_mean = np.mean(eeg_phase, axis=0)  # (n_samples,)
            
            # 对齐长度
            min_len_phase = min(len(eeg_phase_mean), len(phase_envelope_L))
            eeg_phase_aligned = eeg_phase_mean[:min_len_phase]
            phase_L_aligned = phase_envelope_L[:min_len_phase]
            phase_R_aligned = phase_envelope_R[:min_len_phase]
            
            # 计算PLV
            plv_left = compute_plv(eeg_phase_aligned, phase_L_aligned)
            plv_right = compute_plv(eeg_phase_aligned, phase_R_aligned)
            
            plv_result[f'plv_left_{band_name}'] = plv_left
            plv_result[f'plv_right_{band_name}'] = plv_right
        
        plv_results.append(plv_result)
    
    # 7. 保存结果
    print("\n6. 保存结果...")
    print(f"成功处理的trials: {len(correlation_results)}")
    print(f"跳过的trials统计:")
    print(f"  无标签: {skipped_no_label}")
    print(f"  无音频文件名: {skipped_no_wavfile}")
    print(f"  无法加载音频: {skipped_no_audio}")
    
    if len(correlation_results) == 0:
        print("\n错误: 没有成功计算任何相关性结果")
        print("可能的原因:")
        print("  1. 无法加载音频文件")
        print("  2. 音频文件名不匹配")
        print("  3. benchmark文件中没有音频信息")
        print(f"\n调试信息:")
        print(f"  Benchmark文件trials数: {len(trials_info_from_benchmark)}")
        print(f"  EEG trials数: {n_trials}")
        if len(trials_info_from_benchmark) > 0:
            print(f"  第一个benchmark trial的wavfile: {trials_info_from_benchmark[0].get('wavfile', 'None')}")
        return
    
    if len(plv_results) == 0:
        print("错误: 没有成功计算任何PLV结果")
        return
    
    correlation_df = pd.DataFrame(correlation_results)
    plv_df = pd.DataFrame(plv_results)
    
    correlation_path = os.path.join(OUTPUT_DIR, "eeg_audio_correlation_results.csv")
    plv_path = os.path.join(OUTPUT_DIR, "eeg_audio_plv_results.csv")
    
    correlation_df.to_csv(correlation_path, index=False, encoding='utf-8-sig')
    plv_df.to_csv(plv_path, index=False, encoding='utf-8-sig')
    
    print(f"相关性结果已保存: {correlation_path} ({len(correlation_results)} 个trials)")
    print(f"PLV结果已保存: {plv_path} ({len(plv_results)} 个trials)")
    
    # 8. 统计分析
    print("\n7. 统计分析...")
    
    # 检查-inf值
    inf_left_mask = np.isinf(correlation_df['corr_left']) & (correlation_df['corr_left'] < 0)
    inf_right_mask = np.isinf(correlation_df['corr_right']) & (correlation_df['corr_right'] < 0)
    inf_trials_left = correlation_df[inf_left_mask]['trial_idx'].tolist()
    inf_trials_right = correlation_df[inf_right_mask]['trial_idx'].tolist()
    
    if len(inf_trials_left) > 0 or len(inf_trials_right) > 0:
        print("\n[警告] 发现相关性为-inf的trials:")
        if len(inf_trials_left) > 0:
            print(f"  与左音频相关性为-inf的trials: {inf_trials_left[:20]}{'...' if len(inf_trials_left) > 20 else ''} (共{len(inf_trials_left)}个)")
        if len(inf_trials_right) > 0:
            print(f"  与右音频相关性为-inf的trials: {inf_trials_right[:20]}{'...' if len(inf_trials_right) > 20 else ''} (共{len(inf_trials_right)}个)")
        print("  这些trials在计算平均值时将被排除")
    
    print("\n相关性统计:")
    
    # 按标签分组：计算对应标签和非对应标签的相关性
    # 标签0（选左）：对应标签=左音频，非对应标签=右音频
    # 标签1（选右）：对应标签=右音频，非对应标签=左音频
    label_0_mask = correlation_df['label'] == 0
    label_1_mask = correlation_df['label'] == 1
    
    # 标签0（选左）的trials
    corr_label0_corresponding = []  # 选左时与左音频的相关性（对应标签）
    corr_label0_noncorresponding = []  # 选左时与右音频的相关性（非对应标签）
    
    for idx, row in correlation_df[label_0_mask].iterrows():
        if not (np.isinf(row['corr_left']) and row['corr_left'] < 0):
            corr_label0_corresponding.append(row['corr_left'])
        if not (np.isinf(row['corr_right']) and row['corr_right'] < 0):
            corr_label0_noncorresponding.append(row['corr_right'])
    
    # 标签1（选右）的trials
    corr_label1_corresponding = []  # 选右时与右音频的相关性（对应标签）
    corr_label1_noncorresponding = []  # 选右时与左音频的相关性（非对应标签）
    
    for idx, row in correlation_df[label_1_mask].iterrows():
        if not (np.isinf(row['corr_right']) and row['corr_right'] < 0):
            corr_label1_corresponding.append(row['corr_right'])
        if not (np.isinf(row['corr_left']) and row['corr_left'] < 0):
            corr_label1_noncorresponding.append(row['corr_left'])
    
    # 按trial配对收集数据（确保长度一致）
    corr_pairs = []  # 每个元素是(corresponding, noncorresponding)
    for idx, row in correlation_df.iterrows():
        if row['label'] == 0:
            corr_corr = row['corr_left']
            corr_noncorr = row['corr_right']
        else:
            corr_corr = row['corr_right']
            corr_noncorr = row['corr_left']
        
        # 只保留两个值都有效的trial（排除-inf）
        if not (np.isinf(corr_corr) and corr_corr < 0) and not (np.isinf(corr_noncorr) and corr_noncorr < 0):
            corr_pairs.append((corr_corr, corr_noncorr))
    
    # 分离成两个列表
    corr_corresponding = [pair[0] for pair in corr_pairs]
    corr_noncorresponding = [pair[1] for pair in corr_pairs]
    
    print(f"  标签0（选左）的trials数: {label_0_mask.sum()}")
    corr_label0_corr = [row['corr_left'] for idx, row in correlation_df[label_0_mask].iterrows() 
                        if not (np.isinf(row['corr_left']) and row['corr_left'] < 0)]
    corr_label0_noncorr = [row['corr_right'] for idx, row in correlation_df[label_0_mask].iterrows() 
                           if not (np.isinf(row['corr_right']) and row['corr_right'] < 0)]
    
    if len(corr_label0_corr) > 0:
        print(f"    与左音频的相关性（对应标签）: {np.mean(corr_label0_corr):.4f} (有效trials: {len(corr_label0_corr)})")
    else:
        print(f"    与左音频的相关性（对应标签）: 无有效数据")
    
    if len(corr_label0_noncorr) > 0:
        print(f"    与右音频的相关性（非对应标签）: {np.mean(corr_label0_noncorr):.4f} (有效trials: {len(corr_label0_noncorr)})")
    else:
        print(f"    与右音频的相关性（非对应标签）: 无有效数据")
    
    print(f"  标签1（选右）的trials数: {label_1_mask.sum()}")
    corr_label1_corr = [row['corr_right'] for idx, row in correlation_df[label_1_mask].iterrows() 
                        if not (np.isinf(row['corr_right']) and row['corr_right'] < 0)]
    corr_label1_noncorr = [row['corr_left'] for idx, row in correlation_df[label_1_mask].iterrows() 
                           if not (np.isinf(row['corr_left']) and row['corr_left'] < 0)]
    
    if len(corr_label1_corr) > 0:
        print(f"    与右音频的相关性（对应标签）: {np.mean(corr_label1_corr):.4f} (有效trials: {len(corr_label1_corr)})")
    else:
        print(f"    与右音频的相关性（对应标签）: 无有效数据")
    
    if len(corr_label1_noncorr) > 0:
        print(f"    与左音频的相关性（非对应标签）: {np.mean(corr_label1_noncorr):.4f} (有效trials: {len(corr_label1_noncorr)})")
    else:
        print(f"    与左音频的相关性（非对应标签）: 无有效数据")
    
    # 总体统计：对应标签 vs 非对应标签（使用配对数据）
    print(f"\n  总体统计（所有trials，配对数据）:")
    if len(corr_corresponding) > 0:
        print(f"    与对应标签音频的相关性: {np.mean(corr_corresponding):.4f} ± {np.std(corr_corresponding):.4f} (有效trials: {len(corr_corresponding)})")
    else:
        print(f"    与对应标签音频的相关性: 无有效数据")
    
    if len(corr_noncorresponding) > 0:
        print(f"    与非对应标签音频的相关性: {np.mean(corr_noncorresponding):.4f} ± {np.std(corr_noncorresponding):.4f} (有效trials: {len(corr_noncorresponding)})")
    else:
        print(f"    与非对应标签音频的相关性: 无有效数据")
    
    if len(corr_corresponding) > 0 and len(corr_noncorresponding) > 0:
        corr_corresponding = np.array(corr_corresponding)
        corr_noncorresponding = np.array(corr_noncorresponding)
        
        # 计算差异（现在长度一致）
        diff = corr_corresponding - corr_noncorresponding
        mean_diff = np.mean(diff)
        
        print(f"    差异（对应标签 - 非对应标签）: {mean_diff:.4f} ± {np.std(diff):.4f}")
        
        # 配对t检验（paired t-test）- 因为每个trial都有两个值
        from scipy.stats import ttest_rel, wilcoxon
        try:
            t_stat, p_val = ttest_rel(corr_corresponding, corr_noncorresponding)
            print(f"    配对t检验: t={t_stat:.4f}, p={p_val:.4f}", end="")
            if p_val < 0.001:
                print(" ***")
            elif p_val < 0.01:
                print(" **")
            elif p_val < 0.05:
                print(" *")
            else:
                print(" (n.s.)")
        except Exception as e:
            print(f"    配对t检验: 无法计算 ({e})")
        
        # Wilcoxon符号秩检验（非参数检验）
        try:
            w_stat, p_val_w = wilcoxon(corr_corresponding, corr_noncorresponding, alternative='two-sided')
            print(f"    Wilcoxon检验: W={w_stat:.4f}, p={p_val_w:.4f}", end="")
            if p_val_w < 0.001:
                print(" ***")
            elif p_val_w < 0.01:
                print(" **")
            elif p_val_w < 0.05:
                print(" *")
            else:
                print(" (n.s.)")
        except Exception as e:
            print(f"    Wilcoxon检验: 无法计算 ({e})")
        
        # 效应量（Cohen's d for paired samples）
        try:
            diff_std = np.std(diff)
            if diff_std > 0:
                cohens_d = mean_diff / diff_std
                print(f"    Cohen's d (配对): {cohens_d:.4f}", end="")
                if abs(cohens_d) < 0.2:
                    print(" (很小)")
                elif abs(cohens_d) < 0.5:
                    print(" (小)")
                elif abs(cohens_d) < 0.8:
                    print(" (中等)")
                else:
                    print(" (大)")
            else:
                print(f"    Cohen's d: 无法计算（差异标准差为0）")
        except Exception as e:
            print(f"    Cohen's d: 无法计算 ({e})")
        
        # 效应方向统计
        positive_diff_count = np.sum(diff > 0)
        negative_diff_count = np.sum(diff < 0)
        print(f"    效应方向: {positive_diff_count}个trial为正差异, {negative_diff_count}个trial为负差异")
        
        # 保存差异方向到CSV
        diff_direction_data = []
        pair_idx = 0
        for idx, row in correlation_df.iterrows():
            if row['label'] == 0:
                corr_corr = row['corr_left']
                corr_noncorr = row['corr_right']
            else:
                corr_corr = row['corr_right']
                corr_noncorr = row['corr_left']
            
            # 只记录两个值都有效的trial
            if not (np.isinf(corr_corr) and corr_corr < 0) and not (np.isinf(corr_noncorr) and corr_noncorr < 0):
                trial_diff = corr_corr - corr_noncorr
                diff_direction_data.append({
                    'trial_idx': row['trial_idx'],
                    'label': row['label'],
                    'wavfile': row['wavfile'],
                    'corr_corresponding': corr_corr,
                    'corr_noncorresponding': corr_noncorr,
                    'difference': trial_diff,
                    'direction': 'positive' if trial_diff > 0 else 'negative',
                    'abs_difference': abs(trial_diff)
                })
                pair_idx += 1
        
        diff_direction_df = pd.DataFrame(diff_direction_data)
        diff_direction_path = os.path.join(OUTPUT_DIR, "correlation_difference_direction.csv")
        diff_direction_df.to_csv(diff_direction_path, index=False, encoding='utf-8-sig')
        print(f"\n  差异方向详情已保存: {diff_direction_path} ({len(diff_direction_data)} 个trials)")
    else:
        print(f"    差异: 无法计算（数据不足）")
    
    print("\nPLV统计:")
    from scipy.stats import ttest_rel, wilcoxon
    
    for band_name in FREQ_BANDS.keys():
        plv_label = []
        plv_nonlabel = []
        for idx, row in plv_df.iterrows():
            if row['label'] == 0:
                plv_label.append(row[f'plv_left_{band_name}'])
                plv_nonlabel.append(row[f'plv_right_{band_name}'])
            else:
                plv_label.append(row[f'plv_right_{band_name}'])
                plv_nonlabel.append(row[f'plv_left_{band_name}'])
        
        plv_label = np.array(plv_label)
        plv_nonlabel = np.array(plv_nonlabel)
        
        # 计算差异
        diff = plv_label - plv_nonlabel
        mean_diff = np.mean(diff)
        
        print(f"  {band_name}频段:")
        print(f"    标签音频PLV: {np.mean(plv_label):.4f} ± {np.std(plv_label):.4f} (n={len(plv_label)})")
        print(f"    非标签音频PLV: {np.mean(plv_nonlabel):.4f} ± {np.std(plv_nonlabel):.4f} (n={len(plv_nonlabel)})")
        print(f"    差异: {mean_diff:.4f} ± {np.std(diff):.4f}")
        
        # 配对t检验（paired t-test）
        try:
            t_stat, p_val = ttest_rel(plv_label, plv_nonlabel)
            print(f"    配对t检验: t={t_stat:.4f}, p={p_val:.4f}", end="")
            if p_val < 0.001:
                print(" ***")
            elif p_val < 0.01:
                print(" **")
            elif p_val < 0.05:
                print(" *")
            else:
                print(" (n.s.)")
        except Exception as e:
            print(f"    配对t检验: 无法计算 ({e})")
        
        # Wilcoxon符号秩检验（非参数检验）
        try:
            w_stat, p_val_w = wilcoxon(plv_label, plv_nonlabel, alternative='two-sided')
            print(f"    Wilcoxon检验: W={w_stat:.4f}, p={p_val_w:.4f}", end="")
            if p_val_w < 0.001:
                print(" ***")
            elif p_val_w < 0.01:
                print(" **")
            elif p_val_w < 0.05:
                print(" *")
            else:
                print(" (n.s.)")
        except Exception as e:
            print(f"    Wilcoxon检验: 无法计算 ({e})")
        
        # 效应量（Cohen's d for paired samples）
        try:
            diff_std = np.std(diff)
            if diff_std > 0:
                cohens_d = mean_diff / diff_std
                print(f"    Cohen's d (配对): {cohens_d:.4f}", end="")
                if abs(cohens_d) < 0.2:
                    print(" (很小)")
                elif abs(cohens_d) < 0.5:
                    print(" (小)")
                elif abs(cohens_d) < 0.8:
                    print(" (中等)")
                else:
                    print(" (大)")
            else:
                print(f"    Cohen's d: 无法计算（差异标准差为0）")
        except Exception as e:
            print(f"    Cohen's d: 无法计算 ({e})")
        
        # 效应方向统计
        positive_diff_count = np.sum(diff > 0)
        negative_diff_count = np.sum(diff < 0)
        print(f"    效应方向: {positive_diff_count}个trial为正差异, {negative_diff_count}个trial为负差异")
        
        print()  # 空行
    
    # 保存PLV差异方向到CSV（所有频段）
    plv_diff_direction_data = []
    for idx, row in plv_df.iterrows():
        for band_name in FREQ_BANDS.keys():
            if row['label'] == 0:
                plv_corr = row[f'plv_left_{band_name}']
                plv_noncorr = row[f'plv_right_{band_name}']
            else:
                plv_corr = row[f'plv_right_{band_name}']
                plv_noncorr = row[f'plv_left_{band_name}']
            
            plv_diff = plv_corr - plv_noncorr
            plv_diff_direction_data.append({
                'trial_idx': row['trial_idx'],
                'label': row['label'],
                'wavfile': row['wavfile'],
                'band': band_name,
                'plv_corresponding': plv_corr,
                'plv_noncorresponding': plv_noncorr,
                'difference': plv_diff,
                'direction': 'positive' if plv_diff > 0 else 'negative',
                'abs_difference': abs(plv_diff)
            })
    
    plv_diff_direction_df = pd.DataFrame(plv_diff_direction_data)
    plv_diff_direction_path = os.path.join(OUTPUT_DIR, "plv_difference_direction.csv")
    plv_diff_direction_df.to_csv(plv_diff_direction_path, index=False, encoding='utf-8-sig')
    print(f"PLV差异方向详情已保存: {plv_diff_direction_path} ({len(plv_df)} 个trials × {len(FREQ_BANDS)} 个频段)")
    
    # 分析不同音频类别对PLV差异的影响
    print("\n分析音频类别对PLV差异的影响...")
    analyze_audio_category_effect(plv_diff_direction_df, correlation_df, OUTPUT_DIR)
    
    # 9. 绘制图表
    print("\n8. 绘制图表...")
    
    # 相关性热图
    correlation_heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plot_correlation_heatmap(correlation_df, correlation_heatmap_path)
    
    # PLV结果图
    plv_plot_path = os.path.join(OUTPUT_DIR, "plv_comparison.png")
    plot_plv_results(plv_df, plv_plot_path)
    
    # 时间序列示例
    time_series_path = os.path.join(OUTPUT_DIR, "time_series_examples.png")
    plot_time_series_examples(correlation_df, trials_window, trials_info_from_benchmark, labels_df,
                             AUDIO_BASE_DIR, mapping, time_series_path)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

