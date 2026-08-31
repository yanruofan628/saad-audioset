# -*- coding: utf-8 -*-
"""
Kaya & Elhilali (2012) 时域听觉显著性模型复现

目标：
- 复用 individual_model_comparison.py 的数据加载/一致性检查/建模与输出流程
- 严格按照论文与附带 MATLAB 代码的逻辑，实现 5 个特征（包络、音高、频率、速率、尺度）的
  多尺度中心-周围差分与 DoG 归一化，得到一维时间显著性曲线
- 在 AB 选择任务中，用显著性统计特征预测受试者的左右选择
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom
from scipy.signal import butter, convolve2d, hilbert, sosfiltfilt, resample

from individual_model_comparison import (
    load_subject_trials,
    check_left_right_consistency,
    fit_choice_model,
    plot_model_comparison,
    save_results,
)
from linear_regression_selected_features import load_or_extract_features
from linear_regression_loudness_models import (
    DEFAULT_AUDIO_BASE_DIR,
    parse_stereo_pair_name,
    load_mono_5s,
    PAIR_FEATURE_CACHE,
    STD_CACHE,
    _cache_stats,
)


# ============================================================
# 配置
# ============================================================
SALIENCY_STAT_KEYS = [
    "sum",
    "mean",
    "max",
    "std",
    "median",
    "p90",
    "p95",
    "energy",
    "active_ratio",
]

FEATURE_KEYS = ["wave", "pitch", "freq", "rate", "scale"]
SALIENCY_FEATURE_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
SALIENCY_FEATURE_NAMES: List[str] = []

DATA_COLLECTION_DIR = r"D:\D\research\数据采集"
EXPERIMENT_OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
DEFAULT_MAPPING_FILES = [
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
]

SUBJECT_CONFIGS = [
    {
        'subject_id': 'liyanchen',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "liyanchen_1_20251010_090338.mff", "benchmark_1_10-1-1.txt"),
            os.path.join(DATA_COLLECTION_DIR, "liyanchen_2_20251010_094603.mff", "benchmark_1_10-1-2.txt"),
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': True,
    },
    {
        'subject_id': 'shimin',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "shimin_s1_20251010_115112.mff", "benchmark_1_10-1-1.txt"),
            os.path.join(DATA_COLLECTION_DIR, "shimin_s2_20251010_110715.mff", "benchmark_1_10-1-2.txt"),
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': True,
    },
    {
        'subject_id': 'jiachen',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "jiachen1017_1_20251017_041306.mff", "benchmark_1_10-1-2.txt"),
            os.path.join(DATA_COLLECTION_DIR, "jiachen1017_2_20251017_050037.mff", "benchmark_1_10-1-2.txt"),
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': True,
    },
    {
        'subject_id': 'aiwenkai',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "aiwenkai_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "aiwenkai2.csv"),
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': False,
    },
    {
        'subject_id': 'lironghua',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "lironghua.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "lironghua2.csv"),
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': False,
    },
    {
        'subject_id': 'mayunmiao',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "mayunmiao_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "mayunmiao2.csv"),
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': False,
    },
    {
        'subject_id': 'ShangZiyang',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "ShangZiyang.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "ShangZiyang1.csv"),
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': False,
    },
    {
        'subject_id': 'wjy',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "wjy1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "wjy_2.csv"),
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': False,
    },
    {
        'subject_id': 'LiuYaorui',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "LiuYaorui_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "Liu Yaorui2.csv"),
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES),
        'use_consistency_filter': False,
    },
]


# ============================================================
# 数据结构
# ============================================================
@dataclass
class KayaSaliencyMaps:
    wave: np.ndarray
    pitch: np.ndarray
    freq: np.ndarray
    rate: np.ndarray
    scale: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "wave": self.wave,
            "pitch": self.pitch,
            "freq": self.freq,
            "rate": self.rate,
            "scale": self.scale,
        }


# ============================================================
# 工具函数
# ============================================================
def _normalize_audio(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def waveform_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    analytic = hilbert(y)
    envelope = np.abs(analytic)

    # 6 阶 60 Hz 低通（Hilbert 包络后去除高频颤动）
    sos = butter(6, 60 / (sr / 2), btype="low", output="sos")
    envelope = sosfiltfilt(sos, envelope)
    envelope = np.maximum(envelope, 0)

    # 高斯平滑（与 MATLAB 中的 Gaussianfilter 接近）
    smoothing_hz = 1.0 / 0.02  # 20ms
    sigma = sr / smoothing_hz
    sigma = max(1.0, sigma)
    envelope = gaussian_filter1d(envelope, sigma=sigma)

    # 20ms 帧平均
    frame = int(0.02 * sr)
    if frame <= 0:
        frame = 1
    pad = (-len(envelope)) % frame
    if pad:
        envelope = np.pad(envelope, (0, pad))
    env_frames = envelope.reshape(-1, frame)
    env = env_frames.mean(axis=1)

    if np.max(env) > 0:
        env = env / np.max(env)
    return env.astype(np.float32)


def pitch_feature(y: np.ndarray, sr: int) -> np.ndarray:
    # 带通 60-2000 Hz
    sos = butter(3, [60 / (sr / 2), 2000 / (sr / 2)], btype="band", output="sos")
    y_filt = sosfiltfilt(sos, y)

    frame_size = int(0.02 * sr)
    if frame_size < 64:
        frame_size = 64
    hop_size = frame_size // 2
    frames = librosa.util.frame(y_filt, frame_length=frame_size, hop_length=hop_size).T

    min_pitch = 60
    max_pitch = 2000
    min_lag = int(sr / max_pitch)
    max_lag = int(sr / min_pitch)
    if max_lag <= min_lag:
        max_lag = min_lag + 1

    energy = np.sum(frames ** 2, axis=1)
    energy /= np.max(energy) + 1e-6

    pitch_series = np.zeros(frames.shape[0], dtype=np.float32)
    for idx, frame in enumerate(frames):
        if energy[idx] < 0.1:
            continue
        ac = np.correlate(frame, frame, mode="full")
        ac = ac[frame_size - 1 :]
        window = ac[min_lag:max_lag]
        if window.size == 0:
            continue
        peak_index = np.argmax(window)
        lag = min_lag + peak_index
        pitch_hz = sr / lag
        pitch_series[idx] = pitch_hz

    # pitch saliency：log pitch 的一阶差分
    valid = pitch_series > 0
    if np.any(valid):
        log_pitch = np.zeros_like(pitch_series)
        log_pitch[valid] = np.log(pitch_series[valid])
        diff_log = np.abs(np.diff(log_pitch, prepend=log_pitch[0]))
        if np.max(diff_log) > 0:
            diff_log /= np.max(diff_log)
        return diff_log
    return pitch_series


def auditory_spectrogram(y: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=int(0.002 * sr),  # 2ms
        win_length=int(0.002 * sr),
        n_mels=n_mels,
        fmin=60,
        fmax=4000,
        power=1.0,
    )
    mel = np.maximum(mel, 1e-9)
    log_mel = np.log(mel)
    log_mel = gaussian_filter(log_mel, sigma=1.0)
    return log_mel.astype(np.float32)


def rate_scale_features(spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    spec: (freq, time)
    返回：
    - rate_map: (num_rates, time)
    - scale_map: (num_scales, time)
    """
    freq_bins, time_bins = spec.shape
    time_axis = spec

    # Rate filters（沿时间的 DoG）
    rate_values = 2.0 ** np.arange(0, 5, 0.5)  # 1 到 ~32 Hz
    rate_maps = []
    for rate in rate_values:
        sigma = max(1.0, int(0.5 * (time_bins / (rate + 1e-6))))
        dog = gaussian_filter1d(time_axis, sigma=sigma, axis=1) - gaussian_filter1d(
            time_axis, sigma=2 * sigma, axis=1
        )
        rate_maps.append(np.maximum(dog, 0.0))
    rate_maps = np.stack(rate_maps, axis=0)  # (Nr, F, T)
    rate_maps = rate_maps.mean(axis=1)  # 平均掉频率 → (Nr, T)

    # Scale filters（沿频率的 DoG）
    scale_values = 2.0 ** np.arange(-2, 4.5, 0.5)
    scale_maps = []
    for scale in scale_values:
        sigma = max(1.0, int(freq_bins / (scale + 1e-6)))
        dog = gaussian_filter1d(spec, sigma=sigma, axis=0) - gaussian_filter1d(
            spec, sigma=2 * sigma, axis=0
        )
        scale_maps.append(np.maximum(dog, 0.0))
    scale_maps = np.stack(scale_maps, axis=0)  # (Ns, F, T)
    scale_maps = scale_maps.mean(axis=1)  # (Ns, T)

    return rate_maps.astype(np.float32), scale_maps.astype(np.float32)


def _match_shape(data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    通过裁剪/填充的方式将数据调整为 target_shape
    """
    result = data
    for axis, size in enumerate(target_shape):
        curr = result.shape[axis]
        if curr == size:
            continue
        if curr < size:
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, size - curr)
            result = np.pad(result, pad_width=pad_width, mode="edge")
        else:
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(0, size)
            result = result[tuple(slicer)]
    return result


def resize_like(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    将输入数组放缩/裁剪为 target_shape
    """
    arr = np.asarray(arr, dtype=np.float32)
    target_shape = tuple(int(max(1, dim)) for dim in target_shape)
    if arr.shape == target_shape:
        return arr

    print(f"[resize_like] 输入shape={arr.shape}, 目标={target_shape}")

    src = arr
    if src.ndim < len(target_shape):
        expand_dims = len(target_shape) - src.ndim
        src = src.reshape(src.shape + (1,) * expand_dims)

    zoom_factors = []
    for axis, size in enumerate(target_shape):
        curr = src.shape[axis]
        if curr <= 0:
            zoom_factors.append(1.0)
        else:
            zoom_factors.append(size / curr)

    resized = zoom(src, zoom=zoom_factors, order=1, mode="nearest", prefilter=False)
    if resized.shape != target_shape:
        resized = _match_shape(resized, target_shape)

    if resized.shape != target_shape:
        resized = np.resize(resized, target_shape)

    print(f"[resize_like] 输出shape={resized.shape}")
    return resized.astype(np.float32)


def build_pyramid(base_map: np.ndarray, num_levels: int) -> List[np.ndarray]:
    pyramid = []
    current = base_map
    for lvl in range(num_levels):
        pyramid.append(current.astype(np.float32))
        if min(current.shape) <= 2:
            # 后续层保持
            current = current.copy()
            continue
        current = gaussian_filter(current, sigma=1.0)
        if current.ndim == 2:
            new_f = max(1, current.shape[0] // 2)
            new_t = max(1, current.shape[1] // 2)
            current = resample(current, new_f, axis=0)
            current = resample(current, new_t, axis=1)
        else:
            new_len = max(1, current.shape[-1] // 2)
            current = resample(current, new_len, axis=-1)
    return pyramid


def center_surround_maps(
    pyramid: List[np.ndarray],
    center_levels=(2, 3, 4),
    surround_offsets=(3, 4),
) -> List[np.ndarray]:
    maps = []
    for c in center_levels:
        for d in surround_offsets:
            s = c + d
            if c - 1 >= len(pyramid) or s - 1 >= len(pyramid):
                continue
            center = pyramid[c - 1]
            surround = pyramid[s - 1]
            surround_resized = resize_like(surround, center.shape)
            if surround_resized.shape != center.shape:
                print(
                    f"[center_surround_maps] shape mismatch c={c}, s={s}, "
                    f"center={center.shape}, surround_resized={surround_resized.shape}"
                )
                surround_resized = np.resize(surround_resized, center.shape)
            assert (
                surround_resized.shape == center.shape
            ), f"中心/周围尺寸仍不匹配: {center.shape} vs {surround_resized.shape}"
            diff = center - surround_resized
            diff = np.maximum(diff, 0.0)
            if np.any(diff):
                diff /= np.max(np.abs(diff)) + 1e-6
            maps.append(diff.astype(np.float32))
    return maps


def iterative_dog_normalization(maps: List[np.ndarray], iterations: int = 10) -> List[np.ndarray]:
    if not maps:
        return []
    normalized = []
    for fmap in maps:
        M = fmap.copy()
        if M.ndim == 1:
            axis = 0
        else:
            axis = -1

        for _ in range(iterations):
            # 1D DoG kernel长度约 50ms，对应 25 帧（假设 2ms hop）
            length = min(51, M.shape[axis])
            if length < 3:
                break
            grid = np.linspace(-1, 1, length)
            sigma_exc = 0.1
            sigma_inh = 0.3
            dog = np.exp(-grid**2 / (2 * sigma_exc**2)) - np.exp(
                -grid**2 / (2 * sigma_inh**2)
            )
            dog = dog - dog.mean()
            dog = dog.astype(np.float32)
            if M.ndim == 1:
                resp = np.convolve(M, dog, mode="same")
            else:
                resp = convolve2d(M, dog[np.newaxis, :], mode="same", boundary="symm")
            percentile_val = np.percentile(M, 2.0)
            M = M + resp - percentile_val
            M = np.maximum(M, 0.0)

        if np.max(M) > 0:
            M = M / np.max(M)
        normalized.append(M.astype(np.float32))
    return normalized


def combine_feature_maps(feature_maps: List[np.ndarray]) -> np.ndarray:
    if not feature_maps:
        return np.array([])
    target_shape = feature_maps[0].shape
    combined = np.zeros(target_shape, dtype=np.float32)
    for fmap in feature_maps:
        if fmap.shape != target_shape:
            fmap = resize_like(fmap, target_shape)
        combined += fmap
    if np.max(combined) > 0:
        combined = combined / np.max(combined)
    return combined


def kaya_temporal_saliency(y: np.ndarray, sr: int, num_levels: int = 8) -> KayaSaliencyMaps:
    y = _normalize_audio(y)

    env = waveform_envelope(y, sr)  # 1D
    pitch = pitch_feature(y, sr)  # 1D
    spec = auditory_spectrogram(y, sr)  # 2D (freq x time)
    target_len = spec.shape[1]
    if env.shape[-1] != target_len:
        env = resize_like(env, (target_len,))
    if pitch.shape[-1] != target_len:
        pitch = resize_like(pitch, (target_len,))

    rate_map, scale_map = rate_scale_features(spec)  # 2D (Nr/T), (Ns/T)
    if rate_map.shape[-1] != target_len:
        rate_map = resize_like(rate_map, (rate_map.shape[0], target_len))
    if scale_map.shape[-1] != target_len:
        scale_map = resize_like(scale_map, (scale_map.shape[0], target_len))

    # 构建金字塔
    pyramids = {
        "wave": build_pyramid(env[np.newaxis, :], num_levels),
        "pitch": build_pyramid(pitch[np.newaxis, :], num_levels),
        "freq": build_pyramid(spec, num_levels),
        "rate": build_pyramid(rate_map, num_levels),
        "scale": build_pyramid(scale_map, num_levels),
    }

    saliency_components = {}
    component_lengths = {}
    for key, pyr in pyramids.items():
        cs_maps = center_surround_maps(pyr)
        norm_maps = iterative_dog_normalization(cs_maps)
        combined = combine_feature_maps(norm_maps)
        if combined.size == 0:
            saliency_components[key] = np.zeros_like(env)
            continue

        if combined.ndim == 2:
            # 对频率/尺度维度求平均 → 1D
            comp_1d = combined.mean(axis=0)
        else:
            comp_1d = combined.squeeze()

        if np.max(comp_1d) > 0:
            comp_1d = comp_1d / np.max(comp_1d)
        saliency_components[key] = comp_1d.astype(np.float32)
        component_lengths[key] = comp_1d.shape[-1]

    if saliency_components:
        target_len = max(component_lengths.values())
        for key, comp in saliency_components.items():
            if comp.shape[-1] != target_len:
                print(
                    f"[kaya_saliency] 1D长度不一致: {key}={comp.shape[-1]}, 目标={target_len}"
                )
                saliency_components[key] = resize_like(comp, (target_len,))

    return KayaSaliencyMaps(**saliency_components)


def summarize_saliency_curve(curve: np.ndarray) -> Dict[str, float]:
    flat = curve.astype(np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        flat = np.array([0.0])

    stats = {
        "sum": float(np.sum(flat)),
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "energy": float(np.sum(flat**2)),
    }
    threshold = stats["mean"] + stats["std"]
    stats["active_ratio"] = float(np.mean(flat > threshold)) if threshold > 0 else 0.0
    return stats


def compute_saliency_stats_for_audio(
    y: np.ndarray,
    sr: int,
    num_levels: int = 8,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sal_maps = kaya_temporal_saliency(y, sr, num_levels)
    stats = summarize_saliency_curve(
        sal_maps.wave + sal_maps.pitch + sal_maps.freq + sal_maps.rate + sal_maps.scale
    )
    component_stats = {
        key: float(np.sum(val)) for key, val in sal_maps.as_dict().items()
    }
    return stats, component_stats


def build_feature_vector(
    left_stats: Dict[str, float],
    right_stats: Dict[str, float],
    left_components: Dict[str, float],
    right_components: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    global SALIENCY_FEATURE_NAMES
    feature_values = []
    feature_names = []
    eps = 1e-6

    for key in SALIENCY_STAT_KEYS:
        diff = left_stats[key] - right_stats[key]
        ratio = diff / (abs(left_stats[key]) + abs(right_stats[key]) + eps)
        feature_values.extend([diff, ratio])
        feature_names.extend([f"kaya_sal_{key}_diff", f"kaya_sal_{key}_ratio"])

    for comp in FEATURE_KEYS:
        diff = left_components[comp] - right_components[comp]
        ratio = diff / (abs(left_components[comp]) + abs(right_components[comp]) + eps)
        feature_values.extend([diff, ratio])
        feature_names.extend([f"{comp}_sum_diff", f"{comp}_sum_ratio"])

    if not SALIENCY_FEATURE_NAMES:
        SALIENCY_FEATURE_NAMES = feature_names
    return np.asarray(feature_values, dtype=np.float32), feature_names


def compute_pair_saliency_features(
    pair_name: str,
    base_dir: str,
    num_levels: int = 8,
) -> Tuple[np.ndarray, List[str]]:
    cache_key = (pair_name, base_dir)
    if cache_key in SALIENCY_FEATURE_CACHE:
        return SALIENCY_FEATURE_CACHE[cache_key], SALIENCY_FEATURE_NAMES

    left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, base_dir)
    left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
    right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")

    if not os.path.exists(left_path) or not os.path.exists(right_path):
        raise FileNotFoundError(f"音频文件不存在: {left_path} 或 {right_path}")

    y_left, sr = load_mono_5s(left_path)
    y_right, _ = load_mono_5s(right_path)

    left_stats, left_components = compute_saliency_stats_for_audio(
        y_left, sr=sr, num_levels=num_levels
    )
    right_stats, right_components = compute_saliency_stats_for_audio(
        y_right, sr=sr, num_levels=num_levels
    )

    feature_vector, feature_names = build_feature_vector(
        left_stats, right_stats, left_components, right_components
    )
    SALIENCY_FEATURE_CACHE[cache_key] = feature_vector
    return feature_vector, feature_names


def extract_saliency_features(consistent_trials, pair_names, base_dir: str):
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    X_list = []
    y_list = []

    for trial in consistent_trials:
        pair_name = trial.get("pair_name") or trial.get("original_name")
        if pair_name is None:
            continue
        if pair_name not in pair_to_idx:
            continue

        try:
            feature_vector, feature_names = compute_pair_saliency_features(
                pair_name, base_dir
            )
        except Exception as exc:
            print(f"  警告: 计算显著性特征失败 {pair_name}: {exc}")
            continue

        if np.isnan(feature_vector).any():
            continue

        choice = trial.get("choice")
        if choice is None:
            choice = trial.get("response")
        if choice not in (1, 2):
            continue

        X_list.append(feature_vector)
        y_list.append(choice)

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.vstack(X_list)
    y = np.asarray(y_list)
    return X, y, feature_names


def prepare_trials_for_saliency(all_trials):
    """
    将原始试次转换为显著性模型可用的格式（不做左右一致性筛选）
    """
    prepared_trials = []
    for trial in all_trials:
        pair_name = trial.get('pair_name') or trial.get('original_name')
        response = trial.get('choice') or trial.get('response')
        if pair_name is None or response not in (1, 2):
            continue
        prepared_trials.append({
            'pair_name': pair_name,
            'choice': response,
            'reaction_time': trial.get('reaction_time'),
        })
    return prepared_trials


def reset_caches():
    PAIR_FEATURE_CACHE.clear()
    STD_CACHE.clear()
    _cache_stats["pair_feature_hits"] = 0
    _cache_stats["pair_feature_misses"] = 0
    _cache_stats["std_cache_hits"] = 0
    _cache_stats["std_cache_misses"] = 0
    SALIENCY_FEATURE_CACHE.clear()
    SALIENCY_FEATURE_NAMES.clear()


def run_subject_saliency(subject_config, pair_names, base_dir, output_root):
    """
    针对单个被试运行 Kaya 显著性模型
    """
    subject_id = subject_config['subject_id']
    print("\n" + "-" * 60)
    print(f"开始处理被试（Kaya）: {subject_id}")
    print("-" * 60)

    all_trials = load_subject_trials(subject_config)
    if not all_trials:
        print(f"  警告: 未能加载被试 {subject_id} 的任何试次，跳过")
        return None

    use_consistency_filter = subject_config.get('use_consistency_filter', True)
    if use_consistency_filter:
        trials_for_model = check_left_right_consistency(all_trials, pair_names)
    else:
        trials_for_model = prepare_trials_for_saliency(all_trials)
        print(f"  未进行左右一致性筛选，保留 {len(trials_for_model)} 个试次")
    if not trials_for_model:
        print(f"  警告: 被试 {subject_id} 没有可用试次，跳过")
        return None

    X, y_choice, feature_names = extract_saliency_features(trials_for_model, pair_names, base_dir)
    if len(X) == 0:
        print(f"  错误: 被试 {subject_id} 显著性特征为空，跳过")
        return None

    print(f"  特征矩阵: {X.shape[0]} trials × {X.shape[1]} features")
    result = fit_choice_model(X, y_choice, feature_names, model_name=f"Kaya Temporal Saliency ({subject_id})")
    if result is None:
        print(f"  错误: 被试 {subject_id} 模型拟合失败")
        return None

    result['model_name'] = f"Kaya Temporal Saliency - {subject_id}"
    result['model_idx'] = 1
    result['subject_id'] = subject_id

    subject_output_dir = os.path.join(output_root, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    plot_model_comparison([result], subject_output_dir)
    save_results([result], trials_for_model, subject_output_dir)

    summary = {
        'subject_id': subject_id,
        'n_loaded_trials': len(all_trials),
        'n_model_trials': len(trials_for_model),
        'n_features': X.shape[1],
        'accuracy': result['accuracy'],
        'baseline_accuracy': result['baseline_accuracy'],
        'auc': result['auc'],
        'cv_accuracy_mean': result['cv_accuracy_mean'],
        'cv_accuracy_std': result['cv_accuracy_std'],
    }

    print(f"  被试 {subject_id} 完成: accuracy={result['accuracy']:.4f}, AUC={result['auc']:.4f}")
    return summary


def save_summary_table(summaries, output_root):
    """
    将多被试结果保存为CSV
    """
    import csv
    if not summaries:
        return

    summary_path = os.path.join(output_root, "multi_subject_kaya_saliency_summary.csv")
    fieldnames = [
        'subject_id',
        'n_loaded_trials',
        'n_model_trials',
        'n_features',
        'accuracy',
        'baseline_accuracy',
        'auc',
        'cv_accuracy_mean',
        'cv_accuracy_std',
    ]

    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    mean_accuracy = np.mean([row['accuracy'] for row in summaries])
    mean_auc = np.mean([row['auc'] for row in summaries])
    print("\n=== 多被试总体结果 ===")
    print(f"  平均准确率: {mean_accuracy:.4f}")
    print(f"  平均AUC: {mean_auc:.4f}")
    print(f"  已保存汇总表: {summary_path}")


def main():
    print("=" * 70)
    print("个人数据模型：Kaya & Elhilali Temporal Saliency（多被试批处理）")
    print("=" * 70)

    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR
    output_root = os.path.join(project_root, 'kaya_temporal_saliency_results')
    os.makedirs(output_root, exist_ok=True)

    reset_caches()

    summaries = []
    for subject_config in SUBJECT_CONFIGS:
        summary = run_subject_saliency(subject_config, pair_names, base_dir, output_root)
        if summary:
            summaries.append(summary)

    save_summary_table(summaries, output_root)


if __name__ == "__main__":
    main()


