#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人听觉显著性模型（无一致性筛选版本，多被试批处理）

功能：
- 遍历 individual_saliency_model.py 中配置的所有被试
- 强制对每位被试使用全部试次（不做左右一致性筛选），直接提取显著性特征
- 输出被试级别的可视化/结果，并生成整体验证汇总
"""

import os

from individual_model_comparison import (
    load_subject_trials,
    fit_choice_model,
    plot_model_comparison,
    save_results,
)
from linear_regression_selected_features import load_or_extract_features
from linear_regression_loudness_models import DEFAULT_AUDIO_BASE_DIR
from individual_saliency_model import (
    extract_saliency_features,
    reset_caches,
    run_saliency_smoke_tests,
    SUBJECT_CONFIGS,
    prepare_trials_for_saliency,
    save_summary_table,
)


def run_subject_saliency_no_filter(subject_config, pair_names, base_dir, output_root):
    subject_id = subject_config['subject_id']
    print("\n" + "-" * 60)
    print(f"开始处理被试（不筛选）: {subject_id}")
    print("-" * 60)

    all_trials = load_subject_trials(subject_config)
    if len(all_trials) == 0:
        print(f"  警告: 未能加载被试 {subject_id} 的任何试次，跳过")
        return None

    trials_for_model = prepare_trials_for_saliency(all_trials)
    if len(trials_for_model) == 0:
        print(f"  警告: 被试 {subject_id} 无可用试次，跳过")
        return None

    X, y_choice, feature_names = extract_saliency_features(trials_for_model, pair_names, base_dir)
    if len(X) == 0:
        print(f"  错误: 被试 {subject_id} 显著性特征为空，跳过")
        return None

    print(f"  特征矩阵: {X.shape[0]} trials × {X.shape[1]} features")

    result = fit_choice_model(
        X,
        y_choice,
        feature_names,
        model_name=f"Kayser Auditory Saliency (No Filter) - {subject_id}",
    )
    if result is None:
        print(f"  错误: 被试 {subject_id} 模型拟合失败")
        return None

    result["model_name"] = f"Kayser Auditory Saliency (No Filter) - {subject_id}"
    result["model_idx"] = 1
    result["subject_id"] = subject_id

    subject_output_dir = os.path.join(output_root, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    plot_model_comparison([result], subject_output_dir)
    save_results([result], trials_for_model, subject_output_dir)

    summary = {
        "subject_id": subject_id,
        "n_loaded_trials": len(all_trials),
        "n_model_trials": len(trials_for_model),
        "n_features": X.shape[1],
        "accuracy": result["accuracy"],
        "baseline_accuracy": result["baseline_accuracy"],
        "auc": result["auc"],
        "cv_accuracy_mean": result["cv_accuracy_mean"],
        "cv_accuracy_std": result["cv_accuracy_std"],
    }

    print(f"  被试 {subject_id} 完成: accuracy={result['accuracy']:.4f}, AUC={result['auc']:.4f}")
    return summary


def main():
    print("=" * 60)
    print("个人数据模型：Kayser显著性图（多被试，无一致性筛选）")
    print("=" * 60)

    # 1. 快速验证实现
    run_saliency_smoke_tests()

    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR
    output_root = os.path.join(project_root, "individual_saliency_model_no_filter_results")
    os.makedirs(output_root, exist_ok=True)

    reset_caches()

    summaries = []
    for subject_config in SUBJECT_CONFIGS:
        summary = run_subject_saliency_no_filter(subject_config, pair_names, base_dir, output_root)
        if summary:
            summaries.append(summary)

    save_summary_table(summaries, output_root)


if __name__ == "__main__":
    main()


