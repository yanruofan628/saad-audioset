#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人选择建模（带位置与评分特征，多被试批处理）

以 individual_choice_with_position.py 为基础，额外引入被试主观评分
（熟悉度/紧急度/喜爱度），并批量处理 9 位被试，比较“含评分特征 vs 不含评分特征”的效果。
"""

import os
import numpy as np
import pandas as pd

from spatiotemporal_glm_model import parse_pair_name
from linear_regression_selected_features import (
    load_or_extract_features,
    extract_significant_features_from_table,
)

# 复用原脚本中的建模与可视化函数
from individual_choice_with_position import (
    load_subject_trials,
    fit_choice_model_without_position,
    fit_choice_model,
    fit_rt_model,
    compare_models_with_without_position,
    plot_results,
    plot_comparison,
    save_results,
)

RATING_FEATURE_NAMES = ['Rating_Familiarity', 'Rating_Urgency', 'Rating_Liking']

DEFAULT_MAPPING_FILES = [
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
]

CSV_BASE = r"D:\D\research\audioset下载\experiment_output2"
TXT_BASE = r"D:\D\research\数据采集"

SUBJECT_CONFIGS = [
    {
        'subject_id': 'aiwenkai',
        'csv_files': [
            fr"{CSV_BASE}\aiwenkai_1.csv",
            fr"{CSV_BASE}\aiwenkai2.csv",
        ],
        'rating_sheet': 'aikaiwen',
    },
    {
        'subject_id': 'lironghua',
        'csv_files': [
            fr"{CSV_BASE}\lironghua.csv",
            fr"{CSV_BASE}\lironghua2.csv",
        ],
        'rating_sheet': 'lironghua',
    },
    {
        'subject_id': 'mayunmiao',
        'csv_files': [
            fr"{CSV_BASE}\mayunmiao_1.csv",
            fr"{CSV_BASE}\mayunmiao2.csv",
        ],
        'rating_sheet': 'mayunmiao',
    },
    {
        'subject_id': 'ShangZiyang',
        'csv_files': [
            fr"{CSV_BASE}\ShangZiyang.csv",
            fr"{CSV_BASE}\ShangZiyang1.csv",
        ],
        'rating_sheet': 'shangziyang',
    },
    {
        'subject_id': 'wjy',
        'csv_files': [
            fr"{CSV_BASE}\wjy1.csv",
            fr"{CSV_BASE}\wjy_2.csv",
        ],
        'rating_sheet': 'wjy',
    },
    {
        'subject_id': 'LiuYaorui',
        'csv_files': [
            fr"{CSV_BASE}\LiuYaorui_1.csv",
            fr"{CSV_BASE}\Liu Yaorui2.csv",
        ],
        'rating_sheet': 'liuyaorui',
    },
    {
        'subject_id': 'liyanchen',
        'txt_files': [
            fr"{TXT_BASE}\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
            fr"{TXT_BASE}\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        ],
        'rating_sheet': 'yanchen',
    },
    {
        'subject_id': 'shimin',
        'txt_files': [
            fr"{TXT_BASE}\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
            fr"{TXT_BASE}\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        ],
        'rating_sheet': 'shimin',
    },
    {
        'subject_id': 'jiachen1017',
        'txt_files': [
            fr"{TXT_BASE}\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
            fr"{TXT_BASE}\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt",
        ],
        'rating_sheet': 'jiachen',
    },
]


def load_subject_rating_sheet(excel_path, sheet_name):
    """
    读取指定被试（sheet）的评分数据。

    返回格式：{subcategory: {'familiarity': x, 'urgency': y, 'liking': z}}
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"评分Excel不存在: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    category_col = df.columns[0]
    familiarity_col = None
    urgency_col = None
    liking_col = None

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


def extract_features_with_ratings(all_trials, pair_names, feature_indices, subject_ratings):
    """
    在原有特征的基础上添加评分差值特征（左-右），并对评分差值做z-score。
    """
    print("\n=== 提取特征（含评分差值） ===")

    project_root = os.getcwd()
    diff_raw, diff_zscore, all_pair_names = load_or_extract_features(project_root)
    pair_to_idx = {name: idx for idx, name in enumerate(all_pair_names)}

    normalized_to_ab_pair = {}
    for pair_name in pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue
        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))
        if key not in normalized_to_ab_pair:
            normalized_to_ab_pair[key] = pair_name

    X_list = []
    y_choice_list = []
    y_rt_list = []
    trial_info_list = []

    rating_diff_list = []
    error_count = 0

    for trial in all_trials:
        pair_name = trial['original_name']
        response = trial['response']
        rt = trial.get('reaction_time', None)

        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            error_count += 1
            continue

        left_cat, left_id, right_cat, right_id, exp_type = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))

        if key in normalized_to_ab_pair:
            ab_pair_name = normalized_to_ab_pair[key]
            is_ab_version = (pair_name == ab_pair_name)
        else:
            ab_pair_name = pair_name
            try:
                left_id_num = int(left_id) if left_id.isdigit() else float('inf')
                right_id_num = int(right_id) if right_id.isdigit() else float('inf')
                is_ab_version = left_id_num < right_id_num
            except Exception:
                is_ab_version = left_id < right_id

        if ab_pair_name not in pair_to_idx:
            error_count += 1
            continue

        idx = pair_to_idx[ab_pair_name]
        feat_diff_ab = diff_zscore[feature_indices, idx]
        feat_diff = feat_diff_ab if is_ab_version else -feat_diff_ab

        # 评分差值（左-右）
        left_ratings = subject_ratings.get(left_cat)
        right_ratings = subject_ratings.get(right_cat)
        if left_ratings is None or right_ratings is None:
            error_count += 1
            continue

        rating_diff = np.array([
            left_ratings['familiarity'] - right_ratings['familiarity'],
            left_ratings['urgency'] - right_ratings['urgency'],
            left_ratings['liking'] - right_ratings['liking'],
        ], dtype=float)

        if np.isnan(rating_diff).any():
            error_count += 1
            continue

        position_code = 0.0 if is_ab_version else 1.0
        features = np.concatenate([feat_diff, rating_diff, np.array([position_code])])

        if np.isnan(features).any() or np.isinf(features).any():
            error_count += 1
            continue

        X_list.append(features)
        y_choice_list.append(response)
        y_rt_list.append(rt if rt is not None else np.nan)
        rating_diff_list.append(rating_diff)

        trial_info_list.append({
            'pair_name': pair_name,
            'ab_pair_name': ab_pair_name,
            'audio_a': audio_a,
            'audio_b': audio_b,
            'is_ab_version': is_ab_version,
            'response': response,
            'exp_type': exp_type,
            'rating_familiarity_diff': rating_diff[0],
            'rating_urgency_diff': rating_diff[1],
            'rating_liking_diff': rating_diff[2],
        })

    if not X_list:
        raise RuntimeError("没有有效试次可用，请检查评分或特征数据。")

    X = np.array(X_list)
    y_choice = np.array(y_choice_list)
    y_rt = np.array(y_rt_list)

    # 对评分差值做z-score（列是 Rating_Familiarity/Urgency/Liking）
    rating_start = len(feature_indices)
    rating_end = rating_start + len(RATING_FEATURE_NAMES)

    rating_array = X[:, rating_start:rating_end]
    rating_means = np.nanmean(rating_array, axis=0)
    rating_stds = np.nanstd(rating_array, axis=0)
    rating_stds[rating_stds == 0] = 1.0
    X[:, rating_start:rating_end] = (rating_array - rating_means) / rating_stds

    print(f"\n特征提取完成: 有效试次 {len(X)}, 跳过 {error_count}")
    print(f"特征总数: {X.shape[1]} = {len(feature_indices)}个客观特征 + "
          f"{len(RATING_FEATURE_NAMES)}个评分特征 + 1个位置编码")

    return X, y_choice, y_rt, trial_info_list, rating_start, rating_end


def compare_models_with_without_ratings(choice_without_ratings, choice_with_ratings,
                                        objective_feature_names, output_dir):
    """
    对比不含评分特征 vs 含评分特征两个模型的表现。
    """
    print("\n" + "=" * 80)
    print("模型对比：无评分特征 vs 含评分特征（均含位置编码）")
    print("=" * 80)

    if choice_without_ratings is None or choice_with_ratings is None:
        print("警告: 模型结果不完整，无法对比")
        return None, None

    metrics = [
        ('准确率', 'accuracy', '{:.4f}'),
        ('基线准确率', 'baseline_accuracy', '{:.4f}'),
        ('AUC', 'auc', '{:.4f}'),
        ('CV准确率', 'cv_accuracy_mean', '{:.4f}'),
        ('伪R²', 'prsquared', '{:.4f}'),
    ]

    comparison_rows = []
    print(f"\n模型性能对比:")
    print(f"{'指标':<30} {'无评分特征':>18} {'含评分特征':>18} {'改进':>15}")
    print("-" * 80)
    for metric_name, key, fmt in metrics:
        val_without = choice_without_ratings.get(key, np.nan)
        val_with = choice_with_ratings.get(key, np.nan)
        if not np.isnan(val_without) and not np.isnan(val_with):
            improvement = val_with - val_without
            improvement_pct = (improvement / val_without * 100) if val_without != 0 else np.nan
            print(f"{metric_name:<30} {fmt.format(val_without):>18} "
                  f"{fmt.format(val_with):>18} {improvement:+.4f} ({improvement_pct:+.2f}%)")
            comparison_rows.append({
                'metric': metric_name,
                'without_ratings': val_without,
                'with_ratings': val_with,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

    # 特征系数对比（仅客观特征，含评分模型控制位置后得到的系数 vs 无评分模型系数）
    print(f"\n特征系数对比（客观特征，含/不含评分）:")
    print(f"{'特征':<40} {'无评分':>12} {'含评分':>12} {'变化':>15}")
    print("-" * 80)
    coef_rows = []
    for name in objective_feature_names:
        coef_without = next((c['coefficient'] for c in choice_without_ratings['coefficients']
                             if c['feature'] == f'Diff_{name}'), None)
        coef_with = next((c['coefficient'] for c in choice_with_ratings['coefficients']
                          if c['feature'] == f'Diff_{name}'), None)
        if coef_without is None or coef_with is None:
            continue
        change = coef_with - coef_without
        change_pct = (change / abs(coef_without) * 100) if coef_without != 0 else np.nan
        print(f"{name:<40} {coef_without:>12.4f} {coef_with:>12.4f} "
              f"{change:+.4f} ({change_pct:+.2f}%)")
        coef_rows.append({
            'feature': name,
            'coef_without_ratings': coef_without,
            'coef_with_ratings': coef_with,
            'change': change,
            'change_pct': change_pct
        })

    os.makedirs(output_dir, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_rows)
    coef_df = pd.DataFrame(coef_rows)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison_ratings.csv'),
                         index=False, encoding='utf-8-sig')
    coef_df.to_csv(os.path.join(output_dir, 'coefficient_comparison_ratings.csv'),
                   index=False, encoding='utf-8-sig')

    print(f"\n对比结果已保存: {output_dir}")
    return comparison_df, coef_df


def ensure_mapping_length(files, mapping_files):
    if files is None:
        return None
    if len(files) == len(mapping_files):
        return mapping_files
    times = int(np.ceil(len(files) / len(mapping_files)))
    expanded = (mapping_files * times)[:len(files)]
    return expanded


def main():
    print("=" * 80)
    print("个人选择建模（含评分特征，多被试批处理）")
    print("=" * 80)

    rating_excel_path = r"D:\D\research\数据采集\rating.xlsx"
    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    feature_indices_0based, _, feature_names = extract_significant_features_from_table()
    combined_feature_names = feature_names + RATING_FEATURE_NAMES

    output_base_dir = os.path.join(project_root, 'individual_choice_with_position_and_ratings_all_subjects')
    os.makedirs(output_base_dir, exist_ok=True)

    summary_rows = []

    for config in SUBJECT_CONFIGS:
        subject_id = config['subject_id']
        print("\n" + "=" * 60)
        print(f"处理被试: {subject_id}")
        print("=" * 60)

        csv_files = config.get('csv_files')
        txt_files = config.get('txt_files')
        mapping_files = config.get('mapping_files', DEFAULT_MAPPING_FILES)

        csv_mapping = ensure_mapping_length(csv_files, mapping_files) if csv_files else None
        txt_mapping = ensure_mapping_length(txt_files, mapping_files) if txt_files else None

        if csv_files and txt_files:
            # 当同时存在CSV/TXT时，分别加载再合并
            trials_csv = load_subject_trials(csv_files=csv_files, mapping_files=csv_mapping)
            trials_txt = load_subject_trials(txt_files=txt_files, mapping_files=txt_mapping)
            all_trials = trials_csv + trials_txt
        else:
            all_trials = load_subject_trials(
                csv_files=csv_files,
                txt_files=txt_files,
                mapping_files=csv_mapping or txt_mapping
            )

        if len(all_trials) == 0:
            print(f"警告: 被试 {subject_id} 无有效试次，跳过")
            continue

        rating_sheet = config.get('rating_sheet', subject_id)
        try:
            subject_ratings = load_subject_rating_sheet(rating_excel_path, rating_sheet)
        except FileNotFoundError as e:
            print(f"错误: {e}")
            continue

        if len(subject_ratings) == 0:
            print(f"警告: 被试 {subject_id} 的评分数据为空，跳过")
            continue

        X, y_choice, y_rt, trial_info_list, rating_start, rating_end = extract_features_with_ratings(
            all_trials, pair_names, feature_indices_0based, subject_ratings
        )

        if len(X) < 10:
            print(f"警告: 被试 {subject_id} 有效试次少于10，跳过")
            continue

        objective_features = X[:, :rating_start]
        position_feature = X[:, -1:].copy()
        X_without_ratings = np.hstack([objective_features, position_feature])

        choice_without_ratings = fit_choice_model(X_without_ratings, y_choice, feature_names)
        choice_with_ratings = fit_choice_model(X, y_choice, combined_feature_names)

        subject_output_dir = os.path.join(output_base_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)

        comparison_df, coef_df = compare_models_with_without_ratings(
            choice_without_ratings,
            choice_with_ratings,
            feature_names,
            subject_output_dir
        )

        rt_results = fit_rt_model(X, y_rt, combined_feature_names)

        plot_results(choice_with_ratings, rt_results, subject_output_dir, combined_feature_names)
        save_results(choice_with_ratings, rt_results, trial_info_list, subject_output_dir, combined_feature_names)
        save_results(choice_without_ratings, None, trial_info_list,
                     os.path.join(subject_output_dir, 'without_ratings'), feature_names)

        summary_rows.append({
            'subject_id': subject_id,
            'with_ratings_accuracy': choice_with_ratings['accuracy'] if choice_with_ratings else np.nan,
            'without_ratings_accuracy': choice_without_ratings['accuracy'] if choice_without_ratings else np.nan,
            'accuracy_improvement': (choice_with_ratings['accuracy'] - choice_without_ratings['accuracy'])
            if choice_with_ratings and choice_without_ratings else np.nan,
            'with_ratings_auc': choice_with_ratings['auc'] if choice_with_ratings else np.nan,
            'without_ratings_auc': choice_without_ratings['auc'] if choice_without_ratings else np.nan,
            'ratings_cv_accuracy': choice_with_ratings['cv_accuracy_mean'] if choice_with_ratings else np.nan,
            'without_ratings_cv_accuracy': choice_without_ratings['cv_accuracy_mean']
            if choice_without_ratings else np.nan
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_base_dir, 'subject_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总结果已保存: {summary_path}")

    print("\n全部被试处理完成。")


if __name__ == "__main__":
    main()

