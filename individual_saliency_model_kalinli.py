#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人听觉显著性模型（Kalinli 2009 版本，多被试批处理）

流程参考 individual_saliency_model.py：
- 循环处理九个被试的全部试次
- 利用 Kalinli 听觉显著性图提取左右音频特征
- 拟合选择模型并输出结果
- 生成每个被试的结果和总体汇总
"""

import os
from typing import Dict, List, Tuple

import numpy as np

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
)
from individual_saliency_model import (
    SUBJECT_CONFIGS,
    prepare_trials_for_saliency,
    save_summary_table,
)
from kalinli_saliency_backend import compute_kalinli_saliency_from_audio, smoke_test as kalinli_smoke_test

# ======================================
# 全局配置
# ======================================

SALIENCY_STAT_KEYS = ['sum', 'mean', 'max', 'std', 'median', 'p90', 'p95', 'energy', 'active_ratio']
COMPONENT_KEYS = ['intensity', 'freq', 'temporal', 'ori1', 'ori2', 'pitch']
SALIENCY_FEATURE_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
SALIENCY_FEATURE_NAMES: List[str] = []

# TARGET_SUBJECT_ID 已移除，现在处理所有被试


# ======================================
# 工具函数
# ======================================

def summarize_saliency_map(sal_map: np.ndarray) -> Dict[str, float]:
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
    stats['active_ratio'] = float(np.mean(flat > threshold)) if threshold > 0 else 0.0
    return stats


def compute_saliency_stats_for_audio(y: np.ndarray, sr: int = 16000, pyramid_levels: int = 4):
    saliency_map, components = compute_kalinli_saliency_from_audio(y, sr=sr, pyramid_levels=pyramid_levels)
    stats = summarize_saliency_map(saliency_map)
    comp_stats = {}
    for key in COMPONENT_KEYS:
        comp_map = components.get(key)
        comp_stats[key] = float(np.sum(comp_map)) if comp_map is not None else 0.0
    return stats, comp_stats


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
        denom = abs(left_stats[key]) + abs(right_stats[key]) + eps
        feature_values.extend([diff, diff / denom])
        feature_names.extend([f'saliency_{key}_diff', f'saliency_{key}_ratio'])

    for key in COMPONENT_KEYS:
        diff = left_components[key] - right_components[key]
        denom = abs(left_components[key]) + abs(right_components[key]) + eps
        feature_values.extend([diff, diff / denom])
        feature_names.extend([f'{key}_sum_diff', f'{key}_sum_ratio'])

    if not SALIENCY_FEATURE_NAMES:
        SALIENCY_FEATURE_NAMES = feature_names
    return np.asarray(feature_values, dtype=np.float32), feature_names


def compute_pair_saliency_features(pair_name: str, base_dir: str, pyramid_levels: int = 4):
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
    left_stats, left_components = compute_saliency_stats_for_audio(y_left, sr=sr, pyramid_levels=pyramid_levels)
    right_stats, right_components = compute_saliency_stats_for_audio(y_right, sr=sr, pyramid_levels=pyramid_levels)

    feature_vector, feature_names = build_feature_vector(left_stats, right_stats, left_components, right_components)
    SALIENCY_FEATURE_CACHE[cache_key] = feature_vector
    return feature_vector, feature_names


def extract_saliency_features(consistent_trials, pair_names, base_dir):
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    X_list = []
    y_choice_list = []

    for trial in consistent_trials:
        pair_name = trial.get('pair_name') or trial.get('original_name')
        if not pair_name or pair_name not in pair_to_idx:
            continue
        try:
            feature_vector, feature_names = compute_pair_saliency_features(pair_name, base_dir)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  警告: 计算显著性特征失败 {pair_name}: {exc}")
            continue
        if np.isnan(feature_vector).any():
            continue
        choice = trial.get('choice') or trial.get('response')
        if choice not in (1, 2):
            continue
        X_list.append(feature_vector)
        y_choice_list.append(choice)

    if not X_list:
        return np.array([]), np.array([]), []
    X = np.vstack(X_list)
    y_choice = np.asarray(y_choice_list)
    return X, y_choice, feature_names


# ======================================
# 主流程
# ======================================

def run_subject_saliency(subject_config, pair_names, base_dir, output_root):
    subject_id = subject_config['subject_id']
    print("\n" + "-" * 60)
    print(f"开始处理被试（Kalinli）: {subject_id}")
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
    result = fit_choice_model(X, y_choice, feature_names, model_name=f"Kalinli Saliency ({subject_id})")
    if result is None:
        print(f"  错误: 被试 {subject_id} 模型拟合失败")
        return None

    result['model_name'] = f"Kalinli Auditory Saliency - {subject_id}"
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


def run_saliency_smoke_tests():
    print("运行 Kalinli 显著性图烟雾测试 ...")
    kalinli_smoke_test()
    print("  Kalinli 显著性实现快速验证通过")


def main():
    print("=" * 60)
    print("个人数据模型：Kalinli 听觉显著性图（多被试批处理）")
    print("=" * 60)

    run_saliency_smoke_tests()

    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR
    output_root = os.path.join(project_root, 'individual_saliency_model_kalinli_results')
    os.makedirs(output_root, exist_ok=True)

    summaries = []
    for subject_config in SUBJECT_CONFIGS:
        summary = run_subject_saliency(subject_config, pair_names, base_dir, output_root)
        if summary:
            summaries.append(summary)

    save_summary_table(summaries, output_root)


if __name__ == '__main__':
    main()


