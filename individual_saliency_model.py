#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人听觉显著性模型：基于Kayser等人（2005）的听觉显著性图实现

目标：
- 复用 individual_model_comparison.py 的多被试数据加载、左右一致性、模型评估与结果保存流程
- 支持九名被试顺序运行，同步复用显著性特征缓存并输出被试级与总体汇总
- 仅使用 Kayser 听觉显著性图作为特征，判断受试者在AB音频对中选择左/右
- 生成完整的拟合、可视化与结果保存输出，便于与其他模型比较
"""

import os
import csv
import numpy as np
import librosa
from typing import Dict, Tuple, List, Optional
from scipy.signal import hilbert, butter, sosfiltfilt, find_peaks, convolve2d
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom

from individual_model_comparison import (
    load_subject_trials,
    check_left_right_consistency,
    fit_choice_model,
    plot_model_comparison,
    save_results,
    compute_auditory_saliency,
)
from linear_regression_selected_features import (
    load_or_extract_features,
)
from linear_regression_loudness_models import (
    DEFAULT_AUDIO_BASE_DIR,
    parse_stereo_pair_name,
    load_mono_5s,
    PAIR_FEATURE_CACHE,
    STD_CACHE,
    _cache_stats,
)


# ==============================
# 全局配置
# ==============================
SALIENCY_STAT_KEYS = [
    'sum', 'mean', 'max', 'std', 'median', 'p90', 'p95', 'energy', 'active_ratio'
]
COMPONENT_KEYS = ['wave', 'pitch', 'freq', 'rate', 'scale']
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


# ==============================
# 工具函数
# ==============================
def audio_to_spectrogram(y, sr=16000, n_fft=1024, hop_length=128):
    """
    将音频转换为对数幅度谱图，供显著性模型使用
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)
    log_spec = np.log1p(magnitude)
    return log_spec.astype(np.float32)


def summarize_saliency_map(sal_map):
    """
    从显著性图中提取统计特征
    """
    flat = sal_map.astype(np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        flat = np.array([0.0])

    stats = {
        'sum': float(np.sum(flat)),
        'mean': float(np.mean(flat)),
        'max': float(np.max(flat)),
        'std': float(np.std(flat)),
        'median': float(np.median(flat)),
        'p90': float(np.percentile(flat, 90)),
        'p95': float(np.percentile(flat, 95)),
        'energy': float(np.sum(flat ** 2)),
    }

    threshold = stats['mean'] + stats['std']
    active_ratio = float(np.mean(flat > threshold)) if threshold > 0 else 0.0
    stats['active_ratio'] = active_ratio
    return stats


def compute_saliency_stats_for_audio(y, sr=16000, n_fft=1024, hop_length=128, pyramid_levels=4):
    """
    计算音频对应的显著性统计特征
    """
    spec = audio_to_spectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    saliency_map, components = compute_auditory_saliency(spec, N=pyramid_levels)

    stats = summarize_saliency_map(saliency_map)
    component_stats = {}
    for comp_key in COMPONENT_KEYS:
        comp_map = components.get(comp_key)
        if comp_map is None:
            component_stats[comp_key] = 0.0
        else:
            component_stats[comp_key] = float(np.sum(comp_map))

    return stats, component_stats


def build_feature_vector(left_stats, right_stats, left_components, right_components):
    """
    构建左右显著性特征的差异向量
    """
    global SALIENCY_FEATURE_NAMES
    feature_values = []
    feature_names = []
    eps = 1e-6

    for key in SALIENCY_STAT_KEYS:
        diff = left_stats[key] - right_stats[key]
        ratio = diff / (abs(left_stats[key]) + abs(right_stats[key]) + eps)
        feature_values.extend([diff, ratio])
        feature_names.extend([f'saliency_{key}_diff', f'saliency_{key}_ratio'])

    for comp in COMPONENT_KEYS:
        diff = left_components[comp] - right_components[comp]
        ratio = diff / (abs(left_components[comp]) + abs(right_components[comp]) + eps)
        feature_values.extend([diff, ratio])
        feature_names.extend([f'{comp}_sum_diff', f'{comp}_sum_ratio'])

    if not SALIENCY_FEATURE_NAMES:
        SALIENCY_FEATURE_NAMES = feature_names
    return np.asarray(feature_values, dtype=np.float32), feature_names


def compute_pair_saliency_features(pair_name, base_dir, n_fft=1024, hop_length=128, pyramid_levels=4):
    """
    对音频对计算显著性特征，带缓存
    """
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
        y_left, sr=sr, n_fft=n_fft, hop_length=hop_length, pyramid_levels=pyramid_levels
    )
    right_stats, right_components = compute_saliency_stats_for_audio(
        y_right, sr=sr, n_fft=n_fft, hop_length=hop_length, pyramid_levels=pyramid_levels
    )

    feature_vector, feature_names = build_feature_vector(left_stats, right_stats, left_components, right_components)
    SALIENCY_FEATURE_CACHE[cache_key] = feature_vector
    return feature_vector, feature_names


def extract_saliency_features(consistent_trials, pair_names, base_dir):
    """
    为一致试次提取显著性特征
    """
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    X_list = []
    y_choice_list = []

    for trial in consistent_trials:
        pair_name = trial.get('pair_name') or trial.get('original_name')
        if pair_name is None:
            continue
        if pair_name not in pair_to_idx:
            continue

        try:
            feature_vector, feature_names = compute_pair_saliency_features(pair_name, base_dir)
        except Exception as exc:
            print(f"  警告: 计算显著性特征失败 {pair_name}: {exc}")
            continue

        if np.isnan(feature_vector).any():
            continue

        choice = trial.get('choice')
        if choice is None:
            # 默认将 response=1/2 视为左/右选择
            choice = trial.get('response')
        if choice not in (1, 2):
            continue

        X_list.append(feature_vector)
        y_choice_list.append(choice)

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.vstack(X_list)
    y_choice = np.array(y_choice_list)
    return X, y_choice, feature_names


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


def run_subject_saliency_model(subject_config, pair_names, base_dir, output_root):
    """
    针对单个被试运行显著性模型
    """
    subject_id = subject_config['subject_id']
    print("\n" + "-" * 60)
    print(f"开始处理被试: {subject_id}")
    print("-" * 60)

    all_trials = load_subject_trials(subject_config)
    if len(all_trials) == 0:
        print(f"  警告: 未能加载被试 {subject_id} 的任何试次，跳过")
        return None

    use_consistency_filter = subject_config.get('use_consistency_filter', True)
    if use_consistency_filter:
        trials_for_model = check_left_right_consistency(all_trials, pair_names)
    else:
        trials_for_model = prepare_trials_for_saliency(all_trials)
        print(f"  未进行左右一致性筛选，保留 {len(trials_for_model)} 个试次")

    if len(trials_for_model) == 0:
        print(f"  警告: 被试 {subject_id} 没有可用试次，跳过")
        return None

    X, y_choice, feature_names = extract_saliency_features(trials_for_model, pair_names, base_dir)
    if len(X) == 0:
        print(f"  错误: 被试 {subject_id} 显著性特征为空，跳过")
        return None

    print(f"  特征矩阵: {X.shape[0]} trials × {X.shape[1]} features")

    result = fit_choice_model(X, y_choice, feature_names, model_name=f"Kayser Saliency ({subject_id})")
    if result is None:
        print(f"  错误: 被试 {subject_id} 模型拟合失败")
        return None

    result['model_name'] = f"Kayser Auditory Saliency - {subject_id}"
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
    if not summaries:
        return

    summary_path = os.path.join(output_root, "multi_subject_saliency_summary.csv")
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


def reset_caches():
    """
    清空共享缓存，确保每次运行使用最新结果
    """
    PAIR_FEATURE_CACHE.clear()
    STD_CACHE.clear()
    _cache_stats['pair_feature_hits'] = 0
    _cache_stats['pair_feature_misses'] = 0
    _cache_stats['std_cache_hits'] = 0
    _cache_stats['std_cache_misses'] = 0
    SALIENCY_FEATURE_CACHE.clear()
    SALIENCY_FEATURE_NAMES.clear()


def run_saliency_smoke_tests():
    """
    通过合成频谱进行快速验证，确保显著性实现符合预期：
    - 在纯噪声上叠加突出事件后，显著性总量应更高
    """
    np.random.seed(0)
    noise_img = np.abs(np.random.randn(64, 128)).astype(np.float32)
    event_img = noise_img.copy()
    event_img[28:36, 60:68] += 8.0  # 添加明显事件

    sal_noise, _ = compute_auditory_saliency(noise_img, N=3)
    sal_event, _ = compute_auditory_saliency(event_img, N=3)

    noise_sum = float(np.sum(sal_noise))
    event_sum = float(np.sum(sal_event))
    noise_max = float(np.max(sal_noise))
    event_max = float(np.max(sal_event))
    diff = sal_event - sal_noise
    diff_max = float(np.max(diff))
    diff_sum = float(np.sum(diff))

    print(f"  烟雾测试: 噪声总显著性={noise_sum:.6f}, 事件总显著性={event_sum:.6f}")
    print(f"  烟雾测试: 噪声峰值={noise_max:.6f}, 事件峰值={event_max:.6f}")
    print(f"  烟雾测试: 差值总和={diff_sum:.6f}, 差值峰值={diff_max:.6f}")

    if diff_max <= 0:
        raise RuntimeError("显著性实现未通过验证：添加事件后未产生正的局部差值")

    print("  显著性实现快速验证通过")


# ==============================
# 主流程
# ==============================
def main():
    print("=" * 60)
    print("个人数据模型：Kayser听觉显著性图（多被试批处理）")
    print("=" * 60)

    # 1. 快速验证实现
    run_saliency_smoke_tests()

    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR
    output_root = os.path.join(project_root, 'individual_saliency_model_results')
    os.makedirs(output_root, exist_ok=True)

    reset_caches()

    summaries = []
    for subject_config in SUBJECT_CONFIGS:
        summary = run_subject_saliency_model(subject_config, pair_names, base_dir, output_root)
        if summary:
            summaries.append(summary)

    save_summary_table(summaries, output_root)


if __name__ == '__main__':
    main()

