#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人数据模型对比（含评分特征版本）

在 individual_model_comparison.py 的基础上，为每位被试的模型特征矩阵添加
其主观评分差值（熟悉度/紧急度/喜爱度，左减右，并做z-score），用于比较
“客观特征 + 评分”模型在 9 位被试上的效果。
"""

import os
import numpy as np
import pandas as pd

from spatiotemporal_glm_model import parse_pair_name
from linear_regression_selected_features import load_or_extract_features
from linear_regression_loudness_models import (
    MODEL_CONFIGS,
    DEFAULT_AUDIO_BASE_DIR,
)

from individual_model_comparison import (
    SUBJECT_CONFIGS,
    load_subject_trials,
    check_left_right_consistency,
    extract_features_for_model,
    reset_feature_caches,
    fit_choice_model,
    plot_model_comparison,
    save_results,
)

RATING_EXCEL_PATH = r"D:\D\research\数据采集\rating.xlsx"
OUTPUT_ROOT_DIRNAME = 'individual_model_comparison_with_ratings_results'
RATING_FEATURE_NAMES = ['Rating_Familiarity', 'Rating_Urgency', 'Rating_Liking']

# 实验数据 subject_id -> rating.xlsx 中的 sheet 名称
SUBJECT_RATING_SHEETS = {
    'aiwenkai': 'aikaiwen',
    'lironghua': 'lironghua',
    'mayunmiao': 'mayunmiao',
    'ShangZiyang': 'shangziyang',
    'wjy': 'wjy',
    'LiuYaorui': 'liuyaorui',
    'liyanchen': 'yanchen',
    'shimin': 'shimin',
    'jiachen': 'jiachen',
}


def load_subject_rating_sheet(excel_path, sheet_name):
    """读取指定sheet的评分数据，返回 {subcategory: {familiarity, urgency, liking}}"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"评分Excel不存在: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    category_col = df.columns[0]
    familiarity_col = urgency_col = liking_col = None

    for col in df.columns:
        col_str = str(col).strip()
        if '熟悉度' in col_str or 'familiarity' in col_str.lower():
            familiarity_col = col
        elif '紧急度' in col_str or 'urgency' in col_str.lower():
            urgency_col = col
        elif '喜爱度' in col_str or 'liking' in col_str.lower():
            liking_col = col

    if familiarity_col is None and len(df.columns) > 1:
        familiarity_col = df.columns[1]
    if urgency_col is None and len(df.columns) > 2:
        urgency_col = df.columns[2]
    if liking_col is None and len(df.columns) > 3:
        liking_col = df.columns[3]

    rating_dict = {}
    for _, row in df.iterrows():
        category = str(row[category_col]).strip()
        if not category or category.lower() == 'nan':
            continue

        try:
            rating_dict[category] = {
                'familiarity': float(row[familiarity_col]),
                'urgency': float(row[urgency_col]),
                'liking': float(row[liking_col]),
            }
        except (ValueError, TypeError):
            continue

    return rating_dict


def append_rating_features(X, y_choice, valid_indices, consistent_trials, rating_dict):
    """
    根据有效试次的pair_name，计算评分差值并拼接到特征矩阵末尾。

    返回: (X_aug, y_aug) 或 (None, None) 如果无法添加评分特征。
    """
    if rating_dict is None or len(rating_dict) == 0:
        return None, None

    keep_rows = []
    rating_diffs = []

    for row_idx, trial_idx in enumerate(valid_indices):
        trial = consistent_trials[trial_idx]
        pair_name = trial['pair_name']
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue

        left_cat, left_id, right_cat, right_id, _ = parse_result
        left_ratings = rating_dict.get(left_cat)
        right_ratings = rating_dict.get(right_cat)

        if left_ratings is None or right_ratings is None:
            continue

        diff = np.array([
            left_ratings['familiarity'] - right_ratings['familiarity'],
            left_ratings['urgency'] - right_ratings['urgency'],
            left_ratings['liking'] - right_ratings['liking'],
        ], dtype=float)

        if np.isnan(diff).any():
            continue

        keep_rows.append(row_idx)
        rating_diffs.append(diff)

    if not rating_diffs:
        return None, None

    rating_arr = np.array(rating_diffs)
    rating_means = rating_arr.mean(axis=0)
    rating_stds = rating_arr.std(axis=0)
    rating_stds[rating_stds == 0] = 1.0
    rating_arr = (rating_arr - rating_means) / rating_stds

    X_kept = X[keep_rows]
    y_kept = y_choice[keep_rows]
    X_aug = np.hstack([X_kept, rating_arr])
    return X_aug, y_kept


def run_models_for_subject_with_ratings(subject_id, consistent_trials, pair_names,
                                        diff_raw, diff_zscore, base_dir,
                                        rating_dict, output_root):
    """针对单个被试，提取特征并追加评分差值，再拟合所有模型。"""
    print(f"\n{'=' * 60}")
    print(f"开始评估被试（含评分）: {subject_id}")
    print(f"{'=' * 60}")

    reset_feature_caches()
    feature_cache = {}
    all_results = []

    for model_idx, model_config in enumerate(MODEL_CONFIGS, 1):
        model_name = model_config['model_name']
        print(f"\n{'-' * 60}")
        print(f"[{subject_id}] 模型 {model_idx}/{len(MODEL_CONFIGS)}: {model_name}")
        print(f"{'-' * 60}")

        try:
            X, y_choice, valid_indices, feature_names = extract_features_for_model(
                consistent_trials, pair_names, model_config, diff_raw, diff_zscore, base_dir, feature_cache
            )

            if len(X) == 0 or len(X) < 10:
                print(f"  警告: 有效试次太少（{len(X)}），跳过该模型")
                all_results.append(None)
                continue

            X_aug, y_aug = append_rating_features(
                X, y_choice, valid_indices, consistent_trials, rating_dict
            )

            if X_aug is None or len(X_aug) < 10:
                print("  警告: 评分数据不足以构建有效样本，跳过该模型")
                all_results.append(None)
                continue

            augmented_feature_names = feature_names + RATING_FEATURE_NAMES
            result = fit_choice_model(X_aug, y_aug, augmented_feature_names, model_name)

            if result is not None:
                result['model_name'] = model_name
                result['model_idx'] = model_idx
                result['subject_id'] = subject_id
                result['n_samples'] = len(result.get('y_true', []))
                all_results.append(result)
            else:
                all_results.append(None)

        except Exception as e:
            print(f"  错误: 处理模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)

    output_dir = os.path.join(output_root, subject_id)
    os.makedirs(output_dir, exist_ok=True)
    plot_model_comparison(all_results, output_dir)
    save_results(all_results, consistent_trials, output_dir, subject_id=subject_id)

    valid_results = [r for r in all_results if r is not None]
    print(f"\n{'=' * 60}")
    print(f"{subject_id} 模型对比总结（含评分特征）")
    print(f"{'=' * 60}")
    if valid_results:
        valid_results.sort(key=lambda x: x.get('cv_accuracy_mean', 0), reverse=True)
        top_k = min(5, len(valid_results))
        print(f"\n成功拟合 {len(valid_results)} 个模型，前{top_k}名（按CV准确率）:")
        for i, result in enumerate(valid_results[:top_k], 1):
            print(f"\n{i}. {result['model_name']}")
            print(f"   准确率: {result['accuracy']:.4f}")
            print(f"   基线准确率: {result['baseline_accuracy']:.4f}")
            print(f"   提升: {(result['accuracy'] - result['baseline_accuracy'])*100:.2f}%")
            print(f"   AUC: {result['auc']:.4f}")
            print(f"   CV准确率: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}")
    else:
        print("没有成功拟合的模型")

    print(f"\n结果已保存到: {output_dir}")
    return valid_results


def main():
    print("=" * 60)
    print("个人数据模型对比（含评分特征）")
    print("=" * 60)

    project_root = os.getcwd()
    output_root = os.path.join(project_root, OUTPUT_ROOT_DIRNAME)
    os.makedirs(output_root, exist_ok=True)

    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR

    overall_summary_rows = []
    processed_subjects = 0

    for subject_config in SUBJECT_CONFIGS:
        subject_id = subject_config['subject_id']
        rating_sheet = SUBJECT_RATING_SHEETS.get(subject_id)

        if rating_sheet is None:
            print(f"\n跳过 {subject_id}: 未配置评分sheet")
            continue

        try:
            rating_dict = load_subject_rating_sheet(RATING_EXCEL_PATH, rating_sheet)
        except Exception as e:
            print(f"\n跳过 {subject_id}: 评分数据加载失败 ({e})")
            continue

        trials = load_subject_trials(subject_config)
        if len(trials) == 0:
            print(f"\n跳过 {subject_id}: 未能加载任何trial数据")
            continue

        consistent_trials = check_left_right_consistency(trials, pair_names)
        if len(consistent_trials) == 0:
            print(f"\n跳过 {subject_id}: 没有满足左右一致性的试次")
            continue

        for trial in consistent_trials:
            trial['subject_id'] = subject_id

        subject_results = run_models_for_subject_with_ratings(
            subject_id,
            consistent_trials,
            pair_names,
            diff_raw,
            diff_zscore,
            base_dir,
            rating_dict,
            output_root
        )

        if subject_results:
            processed_subjects += 1
            for result in subject_results:
                overall_summary_rows.append({
                    'subject_id': subject_id,
                    'model_idx': result.get('model_idx'),
                    'model_name': result.get('model_name', 'Unknown'),
                    'accuracy': result.get('accuracy', np.nan),
                    'baseline_accuracy': result.get('baseline_accuracy', np.nan),
                    'improvement': result.get('accuracy', np.nan) - result.get('baseline_accuracy', np.nan),
                    'auc': result.get('auc', np.nan),
                    'cv_accuracy_mean': result.get('cv_accuracy_mean', np.nan),
                    'cv_accuracy_std': result.get('cv_accuracy_std', np.nan),
                    'n_features': len(result.get('feature_names', [])),
                    'n_samples': result.get('n_samples', len(result.get('y_true', [])))
                })

    if overall_summary_rows:
        summary_df = pd.DataFrame(overall_summary_rows)
        summary_df = summary_df.sort_values(['subject_id', 'cv_accuracy_mean'], ascending=[True, False])
        summary_path = os.path.join(output_root, 'all_subjects_model_comparison_with_ratings_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n所有被试模型对比（含评分）汇总已保存: {summary_path}")
    else:
        print("\n未生成任何被试的模型结果")

    print(f"\n共处理 {processed_subjects}/{len(SUBJECT_CONFIGS)} 个被试")


if __name__ == '__main__':
    main()

