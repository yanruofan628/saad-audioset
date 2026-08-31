#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于wav2vec惊喜度结果，计算前1秒四类指标（MSE、Cosine、Entropy、Representation Change）
的左右平均值差，并拟合简单线性回归模型预测人类选择概率。

与原版脚本相比，本脚本仅保留：
1. 载入/计算wav2vec惊喜度结果
2. 计算前1秒平均值差（左-右）
3. 拟合单变量OLS模型（包含截距）
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from linear_regression_selected_features import (  # pylint: disable=import-error
    calculate_selection_probability_144,
    load_all_human_decisions,
    merge_balanced_pairs_by_type,
    parse_pair_name,
)
from wav2vec_surprise_regression import (  # pylint: disable=import-error
    check_wav2vec_results,
    compute_wav2vec_results_from_pair_names,
    normalize_audio_filename,
)

# 需要提取的四个方法
TARGET_METHODS = {
    "mse": "mse_first1s_diff",
    "cosine": "cosine_first1s_diff",
    "entropy": "entropy_first1s_diff",
    "representation_change": "representation_change_first1s_diff",
}

# 默认路径配置（与原始脚本一致）
WAV2VEC_OUTPUT_DIR = r"D:\D\research\audioset下载\wav2vec_surprise_heatmaps"
AUDIO_BASE_DIR = r"D:\D\research\audioset下载\clap_select"

TXT_FILES = [
    r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
    r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
    r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
    r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
    r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
    r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt",
]

CSV_FILES = [
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

MAPPING_FILES = [
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
] * 9


def compute_first1s_mean_diff(left_series: np.ndarray, right_series: np.ndarray, frame_duration_ms: float) -> float:
    """计算前1秒平均值差（左-右）。"""
    if left_series.size == 0 or right_series.size == 0:
        return float("nan")

    # 估算前1秒对应的帧数
    if frame_duration_ms <= 0:
        frame_duration_ms = 25.0
    frames_per_second = max(1, int(np.ceil(1000.0 / frame_duration_ms)))

    left_slice = left_series[:frames_per_second]
    right_slice = right_series[:frames_per_second]

    if left_slice.size == 0 or right_slice.size == 0:
        return float("nan")

    left_mean = float(np.nanmean(left_slice))
    right_mean = float(np.nanmean(right_slice))
    return left_mean - right_mean


def extract_first1s_diffs(wav2vec_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """从wav2vec结果中提取前1秒平均值差。"""
    feature_map: Dict[str, Dict[str, float]] = {}

    for result in wav2vec_results:
        pair_name = result.get("audio_file") or result.get("pair_name")
        if not pair_name:
            continue

        methods_results = result.get("methods_results", {})
        if not methods_results:
            continue

        frame_duration_ms = result.get("frame_duration_ms") or result.get("metadata", {}).get("frame_duration_ms") or 25.0

        metrics: Dict[str, float] = {}

        for method, feature_key in TARGET_METHODS.items():
            method_result = methods_results.get(method)
            if not method_result:
                metrics[feature_key] = float("nan")
                continue

            left = np.asarray(method_result.get("left_surprise", []), dtype=float)
            right = np.asarray(method_result.get("right_surprise", []), dtype=float)

            diff_value = compute_first1s_mean_diff(left, right, frame_duration_ms)
            metrics[feature_key] = diff_value

        feature_map[pair_name] = metrics
        normalized = normalize_audio_filename(pair_name)
        if normalized not in feature_map:
            feature_map[normalized] = metrics

    return feature_map


def build_dataset(
    all_pair_names: List[str],
    probabilities: np.ndarray,
    feature_map: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """构建包含四个特征与选择概率的数据表。"""
    rows = []
    for idx, pair_name in enumerate(all_pair_names):
        if idx >= len(probabilities):
            break

        prob = probabilities[idx]
        if np.isnan(prob):
            continue

        metrics = feature_map.get(pair_name) or feature_map.get(normalize_audio_filename(pair_name))
        if not metrics:
            continue

        row = {"pair_name": pair_name, "probability": prob}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def fit_single_feature(df: pd.DataFrame, feature_col: str) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict[str, float]]:
    """对单个特征拟合OLS模型并返回模型与统计指标。"""
    mask = df[feature_col].notna()
    X = df.loc[mask, feature_col].values
    y = df.loc[mask, "probability"].values

    if X.size < 10:
        raise ValueError(f"{feature_col} 有效样本数不足（{X.size} < 10）")

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    stats = {
        "feature": feature_col,
        "n_samples": int(model.nobs),
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "intercept": float(model.params[0]),
        "intercept_pvalue": float(model.pvalues[0]),
        "coefficient": float(model.params[1]),
        "coefficient_pvalue": float(model.pvalues[1]),
    }
    return model, stats


def main():
    print("=== Wav2Vec - 前1秒平均值差回归分析 ===")
    project_root = os.getcwd()
    pair_list_path = os.path.join(project_root, "pair_list.npy")

    if not os.path.exists(pair_list_path):
        raise FileNotFoundError(f"未找到pair_list.npy: {pair_list_path}")

    pair_names = np.load(pair_list_path, allow_pickle=True)
    merged = merge_balanced_pairs_by_type(pair_names)

    all_pair_names: List[str] = []
    for exp_type in ["main", "sub", "nn_main", "nn_sub"]:
        if exp_type in merged:
            all_pair_names.extend(merged[exp_type]["pair_names"])

    print(f"音频对数量: {len(all_pair_names)}")

    print("\n步骤1: 载入人类决策数据")
    all_trials = load_all_human_decisions(TXT_FILES, CSV_FILES, MAPPING_FILES)
    probabilities = calculate_selection_probability_144(all_pair_names, all_trials)

    print("\n步骤2: 载入或计算wav2vec结果")
    wav2vec_results = check_wav2vec_results(WAV2VEC_OUTPUT_DIR)
    if not wav2vec_results:
        if not os.path.exists(AUDIO_BASE_DIR):
            raise FileNotFoundError(f"音频根目录不存在: {AUDIO_BASE_DIR}")
        wav2vec_results = compute_wav2vec_results_from_pair_names(all_pair_names, AUDIO_BASE_DIR, WAV2VEC_OUTPUT_DIR)
        if not wav2vec_results:
            raise RuntimeError("无法获取wav2vec惊喜度结果")

    print("\n步骤3: 提取前1秒平均值差")
    feature_map = extract_first1s_diffs(wav2vec_results)
    dataset = build_dataset(all_pair_names, probabilities, feature_map)

    if dataset.empty:
        raise RuntimeError("没有匹配到任何音频对的特征与概率，请检查wav2vec结果是否覆盖所有pair。")

    output_dir = os.path.join(project_root, "wav2vec_first1s_results")
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = os.path.join(output_dir, "first1s_features.csv")
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    print(f"特征数据已保存: {dataset_path}")

    print("\n步骤4: 拟合OLS模型（每个特征单独拟合）")
    summaries: List[Dict[str, float]] = []
    for method_id, feature_name in TARGET_METHODS.items():
        if feature_name not in dataset.columns:
            print(f"  警告: 特征列缺失 {feature_name}，跳过")
            continue
        try:
            _, stats = fit_single_feature(dataset, feature_name)
            summaries.append(stats)
            print(f"  {feature_name}: R²={stats['rsquared']:.4f}, β={stats['coefficient']:.4f}, p={stats['coefficient_pvalue']:.4e}")
        except ValueError as exc:
            print(f"  警告: {exc}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(output_dir, "first1s_regression_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n模型汇总已保存: {summary_path}")
    else:
        print("\n警告: 没有成功拟合任何模型")

    print("\n=== 分析结束 ===")


if __name__ == "__main__":
    main()

