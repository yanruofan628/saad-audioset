#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 Kayser 听觉显著性的群体吸引力回归模型

思路：
- 复用 linear_regression_loudness_models.py 中的群体行为数据加载与概率计算流程
- 对 144 个左右平衡的音频对，使用 individual_saliency_model.py 中的 Kayser 显著性特征
- 以“左显著性 - 右显著性”的差异向量作为特征，拟合所有被试的整体“选择左侧概率”
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from individual_saliency_model import (
    compute_pair_saliency_features,
    reset_caches,
    run_saliency_smoke_tests,
)
from linear_regression_loudness_models import (
    DEFAULT_AUDIO_BASE_DIR,
    load_or_extract_features,
    merge_balanced_pairs_by_type,
    load_all_human_decisions,
    calculate_selection_probability_144,
    prepare_regression_data,
    build_consistent_folds,
    fit_ols_model,
    extract_model_statistics,
    split_confidence_bounds,
    format_float_list,
    safe_folder_name,
)


# ==============================
# 数据文件配置（与响度模型保持一致）
# ==============================
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
] * 9  # 让 TXT 与 CSV 段落都能循环使用两个映射文件


# ==============================
# Kayser 显著性特征提取
# ==============================
def build_saliency_feature_matrix(
    pair_names: List[str],
    base_dir: str,
    progress_interval: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """
    为音频对列表构建 Kayser 显著性特征矩阵。

    返回:
        feature_matrix: 形状 (n_features, n_pairs)
        feature_names:  特征名称列表
    """
    features = []
    feature_names = None
    expected_dim = None

    for idx, pair_name in enumerate(pair_names, start=1):
        if progress_interval and idx % progress_interval == 0:
            print(f"  显著性特征提取进度 {idx}/{len(pair_names)}")

        try:
            vector, names = compute_pair_saliency_features(pair_name, base_dir)
            expected_dim = len(vector)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  警告: 计算显著性特征失败 {pair_name}: {exc}")
            if expected_dim is None:
                expected_dim = 28  # 9 个统计 + 5 个组件，各含 diff/ratio
            vector = np.full(expected_dim, np.nan, dtype=np.float32)
            names = feature_names or []

        if feature_names is None and names:
            feature_names = list(names)
        features.append(vector.astype(np.float64))

    if not features:
        raise RuntimeError("未能为任何音频对计算显著性特征")

    feature_matrix = np.vstack(features).T  # (n_features, n_pairs)
    if feature_names is None:
        feature_names = [f"saliency_feature_{i+1}" for i in range(feature_matrix.shape[0])]
    return feature_matrix, feature_names


# ==============================
# 主流程
# ==============================
def main():
    print("=" * 70)
    print("Kayser 显著性群体吸引力模型（144 个左右平衡音频对）")
    print("=" * 70)

    # 1. 快速验证与缓存初始化
    run_saliency_smoke_tests()
    reset_caches()

    project_root = os.getcwd()
    base_dir = DEFAULT_AUDIO_BASE_DIR
    output_dir = os.path.join(project_root, "group_saliency_probability_results")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载特征列表与音频对
    print("\n步骤1: 加载音频对索引")
    _, _, pair_names = load_or_extract_features(project_root)
    merged = merge_balanced_pairs_by_type(pair_names)

    all_indices = []
    all_pair_names = []
    for exp_type in ['main', 'sub', 'nn_main', 'nn_sub']:
        if exp_type in merged:
            all_indices.extend(merged[exp_type]['indices'])
            all_pair_names.extend(merged[exp_type]['pair_names'])

    print(f"共获得 {len(all_pair_names)} 个左右平衡音频对")

    # 3. 人类选择概率
    print("\n步骤2: 加载被试选择并计算“选择左侧概率”")
    all_trials = load_all_human_decisions(TXT_FILES, CSV_FILES, MAPPING_FILES)
    probabilities = calculate_selection_probability_144(all_pair_names, all_trials)

    # 4. Kayser 显著性特征
    print("\n步骤3: 计算 Kayser 显著性差异特征")
    feature_matrix, feature_names = build_saliency_feature_matrix(all_pair_names, base_dir)
    print(f"特征矩阵形状: {feature_matrix.shape[1]} 样本 × {feature_matrix.shape[0]} 特征")

    # 5. 准备回归数据
    print("\n步骤4: 构建回归数据并过滤 NaN")
    X, y, valid_mask = prepare_regression_data(feature_matrix, probabilities)
    if len(y) < 10:
        raise RuntimeError("有效样本不足，无法进行回归分析")
    print(f"保留 {len(y)} 个有效样本（共 {len(probabilities)} 个）")

    # 6. 五折交叉验证
    print("\n步骤5: 五折交叉验证评估")
    folds, indices = build_consistent_folds(len(y))
    cv_r2_scores = []
    cv_mse_scores = []
    cv_mae_scores = []
    fold_rows = []

    for fold_idx, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        model = sm.OLS(y_train, sm.add_constant(X_train_df)).fit()
        y_pred = model.predict(sm.add_constant(X_test_df, has_constant='add'))

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mse = float(np.mean((y_test - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_test - y_pred)))

        cv_r2_scores.append(r2)
        cv_mse_scores.append(mse)
        cv_mae_scores.append(mae)
        fold_rows.append({'fold': fold_idx, 'r2': r2, 'mse': mse, 'mae': mae,
                          'n_train': len(y_train), 'n_test': len(y_test)})
        print(f"  折 {fold_idx}/5: R^2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

    avg_r2 = float(np.mean(cv_r2_scores))
    avg_mse = float(np.mean(cv_mse_scores))
    avg_mae = float(np.mean(cv_mae_scores))
    print("\n五折平均性能：")
    print(f"  R^2 = {avg_r2:.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"  MSE = {avg_mse:.4f} ± {np.std(cv_mse_scores):.4f}")
    print(f"  MAE = {avg_mae:.4f} ± {np.std(cv_mae_scores):.4f}")

    # 7. 使用全部样本拟合最终模型
    print("\n步骤6: 全量数据拟合最终 OLS")
    final_model = fit_ols_model(X, y, feature_names)
    stats_dict = extract_model_statistics(final_model, feature_names)
    print(f"最终模型 R^2 = {stats_dict['rsquared']:.4f}, 调整后 R^2 = {stats_dict['rsquared_adj']:.4f}")

    # 8. 导出结果
    print("\n步骤7: 保存结果")
    model_folder = os.path.join(output_dir, safe_folder_name("Kayser_Saliency_Group"))
    os.makedirs(model_folder, exist_ok=True)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(model_folder, "fold_results.csv"), index=False, encoding="utf-8-sig")

    coef_ci_lower, coef_ci_upper = split_confidence_bounds(stats_dict['confidence_intervals'])
    coeff_df = pd.DataFrame({
        'feature': stats_dict['feature_names'],
        'coefficient': stats_dict['coefficients'],
        'pvalue': stats_dict['coefficient_pvalues'],
        'ci_lower': coef_ci_lower,
        'ci_upper': coef_ci_upper,
    })
    coeff_df.to_csv(os.path.join(model_folder, "coefficients.csv"), index=False, encoding="utf-8-sig")

    summary_row = {
        'model_name': 'Kayser_Saliency_Group',
        'n_samples': len(y),
        'n_features': feature_matrix.shape[0],
        'cv_r2_mean': avg_r2,
        'cv_r2_std': float(np.std(cv_r2_scores)),
        'cv_mse_mean': avg_mse,
        'cv_mse_std': float(np.std(cv_mse_scores)),
        'cv_mae_mean': avg_mae,
        'cv_mae_std': float(np.std(cv_mae_scores)),
        'final_rsquared': stats_dict['rsquared'],
        'final_rsquared_adj': stats_dict['rsquared_adj'],
        'aic': stats_dict['aic'],
        'bic': stats_dict['bic'],
        'intercept': stats_dict['intercept'],
        'intercept_pvalue': stats_dict['intercept_pvalue'],
        'intercept_ci_lower': stats_dict['intercept_ci'][0],
        'intercept_ci_upper': stats_dict['intercept_ci'][1],
        'coefficients': format_float_list(stats_dict['coefficients']),
        'coefficient_pvalues': format_float_list(stats_dict['coefficient_pvalues'], scientific=True),
        'coef_ci_lower': format_float_list(coef_ci_lower),
        'coef_ci_upper': format_float_list(coef_ci_upper),
    }
    pd.DataFrame([summary_row]).to_csv(
        os.path.join(model_folder, "summary.csv"), index=False, encoding="utf-8-sig"
    )

    print(f"所有结果已保存到: {model_folder}")
    print("完成 Kayser 显著性群体回归分析。")


if __name__ == "__main__":
    main()


