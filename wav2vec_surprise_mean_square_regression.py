#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于四种Wav2Vec惊喜度指标的平方均值进行线性回归分析

流程：
1. 加载音频对列表并合并实验类型
2. 加载或重新计算Wav2Vec惊喜度结果，并提取特征
3. 加载人类实验数据并计算选择概率
4. 计算四种惊喜度(sum差值)平方后的平均值，构建数据集
5. 使用OLS回归拟合“平方均值惊喜度”对人类选择概率的线性关系

说明：
- 仅保留核心流程；不包含额外的可视化、相关性分析或一致性检验。
- 默认路径与`wav2vec_surprise_regression.py`保持一致，如需修改请调整下方常量。
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from linear_regression_selected_features import (
    merge_balanced_pairs_by_type,
    parse_pair_name,
    load_all_human_decisions,
)

from wav2vec_surprise_regression import (
    check_wav2vec_results,
    compute_wav2vec_results_from_pair_names,
    extract_wav2vec_features,
    normalize_audio_filename,
)

warnings.filterwarnings("ignore")

FOUR_METHODS = ['mse', 'cosine', 'representation_change', 'entropy']

# 默认路径设置
PROJECT_ROOT = os.getcwd()
WAV2VEC_OUTPUT_DIR = r"D:\D\research\audioset下载\wav2vec_surprise_heatmaps"
WAV2VEC_INPUT_DIR = r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120"
REGRESSION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'wav2vec_mean_square_regression')

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


def prepare_pair_names(project_root: str):
    """加载pair_list并返回所有音频对及其实验类型映射。"""
    pair_list_path = os.path.join(project_root, 'pair_list.npy')
    if not os.path.exists(pair_list_path):
        raise FileNotFoundError(f"未找到pair_list.npy: {pair_list_path}")

    pair_names = np.load(pair_list_path, allow_pickle=True)
    merged = merge_balanced_pairs_by_type(pair_names)

    all_pair_names = []
    pair_experiment_type_map = {}

    for exp_type, payload in merged.items():
        for pair_name in payload['pair_names']:
            all_pair_names.append(pair_name)
            pair_experiment_type_map.setdefault(pair_name, exp_type)

    return all_pair_names, pair_experiment_type_map


def load_or_compute_wav2vec_features(pair_names, audio_base_dir, wav2vec_output_dir):
    """加载或重新计算Wav2Vec特征，返回特征字典。"""
    os.makedirs(wav2vec_output_dir, exist_ok=True)

    wav2vec_results = check_wav2vec_results(wav2vec_output_dir)
    if not wav2vec_results:
        print("未找到现有wav2vec结果，开始重新计算...")
        wav2vec_results = compute_wav2vec_results_from_pair_names(
            pair_names,
            audio_base_dir,
            wav2vec_output_dir,
        )

    if not wav2vec_results:
        raise RuntimeError("无法获取Wav2Vec惊喜度结果。")

    return extract_wav2vec_features(wav2vec_results)


def load_human_trials():
    """加载所有被试的实验记录。"""
    return load_all_human_decisions(TXT_FILES, CSV_FILES, MAPPING_FILES)


def compute_selection_probabilities(pair_names, trials):
    """
    对每个pair计算人类选择左音频的概率。
    返回概率数组与计数字典。
    """
    probabilities = []
    selection_counts = {}

    for pair_name in pair_names:
        parsed = parse_pair_name(pair_name)
        if not parsed or not parsed[0]:
            probabilities.append(np.nan)
            selection_counts[pair_name] = (0, 0)
            continue

        left_cat, left_id, right_cat, right_id, _ = parsed
        audio_left = f"{left_cat}_{left_id}"
        audio_right = f"{right_cat}_{right_id}"

        selected_left_count = 0
        total_count = 0

        for trial in trials:
            trial_name = trial.get('original_name', '')
            trial_parsed = parse_pair_name(trial_name)
            if not trial_parsed or not trial_parsed[0]:
                continue

            t_left_cat, t_left_id, t_right_cat, t_right_id, _ = trial_parsed
            trial_left = f"{t_left_cat}_{t_left_id}"
            trial_right = f"{t_right_cat}_{t_right_id}"

            if trial_left == audio_left and trial_right == audio_right:
                total_count += 1
                if trial['response'] == 1:
                    selected_left_count += 1
            elif trial_left == audio_right and trial_right == audio_left:
                total_count += 1
                if trial['response'] == 2:
                    selected_left_count += 1

        prob = selected_left_count / total_count if total_count > 0 else np.nan
        probabilities.append(prob)
        selection_counts[pair_name] = (selected_left_count, total_count)

    return np.array(probabilities), selection_counts


def build_dataset(pair_names, probabilities, features_dict, counts_map, experiment_type_map=None):
    """构建包含平方均值惊喜度的DataFrame。"""
    records = []
    missing_pairs = []

    for pair_name, prob in zip(pair_names, probabilities):
        if np.isnan(prob):
            continue

        features = features_dict.get(pair_name)
        if features is None:
            normalized = normalize_audio_filename(pair_name)
            features = features_dict.get(normalized)

        if features is None:
            missing_pairs.append(pair_name)
            continue

        squared_values = []
        raw_values = []
        for method in FOUR_METHODS:
            key = f"{method}_sum_diff"
            value = features.get(key)
            if value is None or np.isnan(value):
                break
            value = float(value)
            squared_values.append(value ** 2)
            raw_values.append(value)

        if len(squared_values) != len(FOUR_METHODS):
            missing_pairs.append(pair_name)
            continue

        mean_square_surprise = float(np.mean(squared_values))
        mean_signed_diff = float(np.mean(raw_values))
        selected_left, total = counts_map.get(pair_name, (None, None))

        experiment_type = experiment_type_map.get(pair_name) if experiment_type_map else None

        records.append({
            'pair_name': pair_name,
            'experiment_type': experiment_type,
            'probability': float(prob),
            'mean_square_surprise': mean_square_surprise,
            'mean_signed_diff': mean_signed_diff,
            'selected_left_count': selected_left,
            'total_trials': total,
        })

    if missing_pairs:
        print(f"警告: 有 {len(missing_pairs)} 个音频对缺少完整特征，已跳过。")
        print("示例缺失音频对（最多5个）:")
        for sample in missing_pairs[:5]:
            print(f"  - {sample}")

    if not records:
        raise RuntimeError("没有可用于回归的样本。")

    return pd.DataFrame(records)


def calculate_phi_coefficient(a, b, c, d):
    """计算2x2列联表的Phi系数。"""
    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        return np.nan
    return numerator / denominator


def analyze_agreement(dataset: pd.DataFrame):
    """
    使用mean_signed_diff的符号作为模型决策，与人类概率比较，计算一致率与Phi系数。
    Returns (result_dict or None)
    """
    if 'mean_signed_diff' not in dataset.columns:
        print("警告: 数据集中缺少 mean_signed_diff，无法计算一致率")
        return None

    feature_values = dataset['mean_signed_diff'].values
    probabilities = dataset['probability'].values

    valid_mask = ~(np.isnan(feature_values) | np.isnan(probabilities))
    feature_values = feature_values[valid_mask]
    probabilities = probabilities[valid_mask]

    if len(feature_values) == 0:
        print("警告: 无有效样本用于一致率计算")
        return None

    model_choose_left = feature_values > 0
    model_choose_right = feature_values < 0
    human_choose_left = probabilities > 0.5
    human_choose_right = probabilities < 0.5

    valid_decision_mask = (model_choose_left | model_choose_right) & (human_choose_left | human_choose_right)
    model_decisions = model_choose_left[valid_decision_mask]
    human_decisions = human_choose_left[valid_decision_mask]

    total_available = len(feature_values)
    used_samples = len(model_decisions)

    if used_samples == 0:
        print("警告: 所有样本均为平局，无法计算一致率")
        return None

    a = int(np.sum(model_decisions & human_decisions))
    b = int(np.sum(model_decisions & ~human_decisions))
    c = int(np.sum(~model_decisions & human_decisions))
    d = int(np.sum(~model_decisions & ~human_decisions))
    contingency = [[a, b], [c, d]]

    total = a + b + c + d
    agreement_rate = (a + d) / total if total > 0 else np.nan
    phi = calculate_phi_coefficient(a, b, c, d)

    from scipy.stats import chi2_contingency
    chi2, p_value, _, _ = chi2_contingency(contingency, correction=False)

    return {
        'n_total': int(total),
        'n_available': int(total_available),
        'n_used': int(used_samples),
        'n_excluded': int(total_available - used_samples),
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'agreement_rate': float(agreement_rate),
        'phi': float(phi),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'contingency_table': contingency,
    }


def fit_regression(dataset: pd.DataFrame):
    """使用OLS拟合平方均值惊喜度与人类选择概率的线性模型。"""
    X = dataset['mean_square_surprise'].values.reshape(-1, 1)
    y = dataset['probability'].values

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    conf_int = model.conf_int()
    if hasattr(conf_int, "values"):
        conf_int = conf_int.values
    conf_int = np.asarray(conf_int, dtype=float)

    result = {
        'n_samples': int(model.nobs),
        'intercept': float(model.params[0]),
        'coefficient': float(model.params[1]),
        'intercept_pvalue': float(model.pvalues[0]),
        'coefficient_pvalue': float(model.pvalues[1]),
        'intercept_ci': [float(ci) for ci in conf_int[0]],
        'coefficient_ci': [float(ci) for ci in conf_int[1]],
        'rsquared': float(model.rsquared),
        'rsquared_adj': float(model.rsquared_adj),
        'aic': float(model.aic),
        'bic': float(model.bic),
        'fvalue': float(model.fvalue) if model.fvalue is not None else None,
        'f_pvalue': float(model.f_pvalue) if model.f_pvalue is not None else None,
    }

    return model, result


def save_outputs(dataset, regression_result, model_summary_text, output_dir):
    """保存数据集与回归结果。"""
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = os.path.join(output_dir, 'mean_square_dataset.csv')
    dataset.to_csv(dataset_path, index=False, encoding='utf-8-sig')
    print(f"数据集已保存: {dataset_path}")

    result_path = os.path.join(output_dir, 'mean_square_regression_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(regression_result, f, ensure_ascii=False, indent=2)
    print(f"回归结果已保存: {result_path}")

    summary_path = os.path.join(output_dir, 'mean_square_regression_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(model_summary_text)
    print(f"回归摘要已保存: {summary_path}")

    return dataset_path, result_path, summary_path


def save_agreement(result: dict, output_dir: str):
    """保存一致性分析结果。"""
    if result is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mean_square_agreement_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"一致率结果已保存: {output_path}")
    return output_path


def main():
    print("=== Wav2Vec惊喜度平方均值回归 ===")

    if not os.path.exists(WAV2VEC_INPUT_DIR):
        raise FileNotFoundError(f"音频基础目录不存在: {WAV2VEC_INPUT_DIR}")

    print("\n步骤1: 加载音频对列表")
    all_pair_names, pair_experiment_type_map = prepare_pair_names(PROJECT_ROOT)
    print(f"共加载 {len(all_pair_names)} 个音频对")

    target_types = {'main', 'sub'}
    pair_names = [name for name in all_pair_names if pair_experiment_type_map.get(name) in target_types]
    if not pair_names:
        raise RuntimeError("未找到 main/sub 类型的音频对。")
    print(f"保留 main/sub 音频对数量: {len(pair_names)}")

    print("\n步骤2: 获取Wav2Vec惊喜度特征")
    wav2vec_features = load_or_compute_wav2vec_features(
        pair_names,
        WAV2VEC_INPUT_DIR,
        WAV2VEC_OUTPUT_DIR,
    )
    print(f"成功提取 {len(wav2vec_features)} 个音频对的特征")

    print("\n步骤3: 加载人类实验数据并计算选择概率")
    trials = load_human_trials()
    print(f"加载被试记录 {len(trials)} 条")
    probabilities, counts_map = compute_selection_probabilities(pair_names, trials)
    available = np.sum(~np.isnan(probabilities))
    print(f"成功计算概率的音频对数量: {available}")

    print("\n步骤4: 构建数据集")
    dataset = build_dataset(pair_names, probabilities, wav2vec_features, counts_map, pair_experiment_type_map)
    print(f"可用于回归的样本数: {len(dataset)}")

    print("\n步骤5: 拟合线性回归模型")
    model, regression_result = fit_regression(dataset)
    print("OLS回归完成")
    print(f"样本数量: {regression_result['n_samples']}")
    print(f"R²: {regression_result['rsquared']:.4f}")
    print(f"系数: {regression_result['coefficient']:.6f} (p={regression_result['coefficient_pvalue']:.4g})")

    print("\n步骤6: 保存结果")
    dataset_path, regression_path, summary_path = save_outputs(
        dataset,
        regression_result,
        model.summary().as_text(),
        REGRESSION_OUTPUT_DIR,
    )

    print("\n步骤7: 计算一致率与Phi系数")
    agreement_result = analyze_agreement(dataset)
    if agreement_result:
        print(f"一致率: {agreement_result['agreement_rate']:.4f}")
        print(f"Phi系数: {agreement_result['phi']:.4f}")
        print(f"卡方检验: χ²={agreement_result['chi2']:.4f}, p={agreement_result['p_value']:.4g}")
        print(f"显著性(p<0.05): {'是' if agreement_result['significant'] else '否'}")
        agreement_path = save_agreement(agreement_result, REGRESSION_OUTPUT_DIR)
    else:
        agreement_path = None

    print("\n=== 完成 ===")
    print("输出文件：")
    print(f"  数据集: {dataset_path}")
    print(f"  回归结果: {regression_path}")
    print(f"  回归摘要: {summary_path}")
    if agreement_path:
        print(f"  一致率结果: {agreement_path}")


if __name__ == '__main__':
    main()


