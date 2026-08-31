#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个人数据模型对比：使用linear_regression_loudness_models.py中的所有模型特征
拟合个人选择（二分类），对比不同模型的效果

- 检查左右平衡一致性（AB和BA必须选择同一个音频）
- 合并成144个独立对
- 对每个模型配置提取特征
- 拟合个人选择（逻辑回归）
- 对比所有模型的效果

新增：集成Kayser等人2005年的听觉显著性地图算法
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy import signal
from scipy.ndimage import convolve, zoom
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要的函数
from spatiotemporal_glm_model import (
    parse_experiment_data_txt,
    parse_experiment_data_csv,
    parse_audio_mapping,
    parse_pair_name
)
from linear_regression_selected_features import (
    load_or_extract_features,
    build_consistent_folds
)

# 导入linear_regression_loudness_models中的模型配置和特征提取函数
from linear_regression_loudness_models import (
    MODEL_CONFIGS,
    FEATURE_REGISTRY,
    DEFAULT_AUDIO_BASE_DIR,
    build_feature_matrix,
    build_transformed_features_matrix,
    parse_transformation_method,
    SELECTED_FEATURE_INDICES,
    SELECTED_FEATURE_NAMES,
    SELECTED_FEATURE_CONFIGS,
    PAIR_FEATURE_CACHE,
    STD_CACHE,
    _cache_stats
)

CV_SEED = 42
N_FOLDS = 5
MIN_CONSISTENT_TRIALS_WARNING = 50

DATA_COLLECTION_DIR = r"D:\D\research\数据采集"
EXPERIMENT_OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
DEFAULT_MAPPING_FILES = [
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
    r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
]

SUBJECT_CONFIGS = [
    {
        'subject_id': 'liyanchen',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "liyanchen_1_20251010_090338.mff", "benchmark_1_10-1-1.txt"),
            os.path.join(DATA_COLLECTION_DIR, "liyanchen_2_20251010_094603.mff", "benchmark_1_10-1-2.txt")
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'shimin',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "shimin_s1_20251010_115112.mff", "benchmark_1_10-1-1.txt"),
            os.path.join(DATA_COLLECTION_DIR, "shimin_s2_20251010_110715.mff", "benchmark_1_10-1-2.txt")
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'jiachen',
        'txt_files': [
            os.path.join(DATA_COLLECTION_DIR, "jiachen1017_1_20251017_041306.mff", "benchmark_1_10-1-2.txt"),
            os.path.join(DATA_COLLECTION_DIR, "jiachen1017_2_20251017_050037.mff", "benchmark_1_10-1-2.txt")
        ],
        'txt_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'aiwenkai',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "aiwenkai_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "aiwenkai2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'lironghua',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "lironghua.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "lironghua2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'mayunmiao',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "mayunmiao_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "mayunmiao2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'ShangZiyang',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "ShangZiyang.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "ShangZiyang1.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'wjy',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "wjy1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "wjy_2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    },
    {
        'subject_id': 'LiuYaorui',
        'csv_files': [
            os.path.join(EXPERIMENT_OUTPUT_DIR, "LiuYaorui_1.csv"),
            os.path.join(EXPERIMENT_OUTPUT_DIR, "Liu Yaorui2.csv")
        ],
        'csv_mapping_files': list(DEFAULT_MAPPING_FILES)
    }
]


def load_jiachen_trials(txt_files, mapping_files):
    """
    加载被试的两个txt文件并合并（通用函数，可用于任何被试）
    
    Parameters:
    -----------
    txt_files : list
        txt文件路径列表
    mapping_files : list
        mapping文件路径列表
        
    Returns:
    --------
    all_trials : list
        所有试次数据，包含original_name, response, reaction_time
    """
    print("\n=== 加载被试数据 ===")
    all_trials = []
    
    for txt_file, mapping_file in zip(txt_files, mapping_files):
        if not os.path.exists(txt_file):
            print(f"警告: TXT文件不存在: {txt_file}")
            continue
        
        if not os.path.exists(mapping_file):
            print(f"警告: Mapping文件不存在: {mapping_file}")
            continue
        
        print(f"\n处理文件: {txt_file}")
        trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)
        
        print(f"  加载了 {len(trials)} 个试次")
        print(f"  Mapping包含 {len(mapping)} 个映射")
        
        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)
            
            if filename in mapping:
                original_name = mapping[filename]
                trial_data = {
                    'original_name': original_name,
                    'response': trial['response'],
                    'reaction_time': trial.get('reaction_time', None)
                }
                all_trials.append(trial_data)
            else:
                print(f"  警告: 未找到映射 {filename}")
    
    print(f"\n总共加载 {len(all_trials)} 个试次")
    rt_count = sum(1 for t in all_trials if t['reaction_time'] is not None)
    print(f"  有RT数据的试次: {rt_count}/{len(all_trials)}")
    
    return all_trials


def load_subject_trials(subject_config):
    """
    根据配置加载单个被试的trial数据（支持TXT与CSV）
    """
    subject_id = subject_config['subject_id']
    print(f"\n=== 加载被试 {subject_id} 数据 ===")
    trials = []

    txt_files = subject_config.get('txt_files', []) or []
    txt_mapping_files = subject_config.get('txt_mapping_files', subject_config.get('mapping_files', [])) or []

    if txt_files and not txt_mapping_files:
        print(f"  警告: 未提供TXT映射文件，跳过TXT数据")
        txt_files = []

    for txt_file, mapping_file in zip(txt_files, txt_mapping_files):
        if not os.path.exists(txt_file):
            print(f"  TXT文件不存在: {txt_file}")
            continue
        if not os.path.exists(mapping_file):
            print(f"  Mapping文件不存在: {mapping_file}")
            continue

        print(f"  处理TXT: {txt_file}")
        parsed_trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in parsed_trials:
            filename = os.path.basename(trial['wavfile'])
            if filename not in mapping:
                continue
            record = {
                'subject_id': subject_id,
                'original_name': mapping[filename],
                'response': trial['response'],
                'reaction_time': trial.get('reaction_time')
            }
            trials.append(record)

    csv_files = subject_config.get('csv_files', []) or []
    csv_mapping_files = subject_config.get('csv_mapping_files', subject_config.get('mapping_files', [])) or []

    if csv_files and not csv_mapping_files:
        print(f"  警告: 未提供CSV映射文件，跳过CSV数据")
        csv_files = []

    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        if not os.path.exists(csv_file):
            print(f"  CSV文件不存在: {csv_file}")
            continue
        if not os.path.exists(mapping_file):
            print(f"  Mapping文件不存在: {mapping_file}")
            continue

        print(f"  处理CSV: {csv_file}")
        parsed_trials = parse_experiment_data_csv(csv_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in parsed_trials:
            filename = os.path.basename(trial['wavfile'])
            if filename not in mapping:
                continue
            record = {
                'subject_id': subject_id,
                'original_name': mapping[filename],
                'response': trial['response'],
                'reaction_time': None
            }
            trials.append(record)

    print(f"  完成: {len(trials)} 个trial")
    rt_count = sum(1 for t in trials if t.get('reaction_time') is not None)
    if trials:
        print(f"  含RT的trial: {rt_count}/{len(trials)}")

    return trials


def check_left_right_consistency(all_trials, pair_names):
    """
    检查左右平衡一致性，只保留AB和BA都选择同一个音频的试次
    
    Parameters:
    -----------
    all_trials : list
        所有试次数据
    pair_names : list
        所有音频对名称（288个）
        
    Returns:
    --------
    consistent_trials : list
        一致的试次数据，包含pair_name, choice, reaction_time
    """
    print("\n=== 检查左右平衡一致性 ===")
    
    # 创建试次字典：key是original_name，value是trial数据
    trial_dict = {}
    for trial in all_trials:
        pair_name = trial['original_name']
        if pair_name not in trial_dict:
            trial_dict[pair_name] = []
        trial_dict[pair_name].append(trial)
    
    # 创建规范化键字典：用于匹配AB和BA
    normalized_pairs = {}
    for pair_name in pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))
        
        if key not in normalized_pairs:
            normalized_pairs[key] = []
        normalized_pairs[key].append(pair_name)
    
    consistent_trials = []
    inconsistent_count = 0
    
    # 对于每个规范化键（代表一个独立对）
    for key, pair_names_list in normalized_pairs.items():
        if len(pair_names_list) != 2:
            # 应该有AB和BA两个版本
            continue
        
        pair_ab = pair_names_list[0]  # AB版本
        pair_ba = pair_names_list[1]  # BA版本
        
        # 找到对应的试次
        trials_ab = trial_dict.get(pair_ab, [])
        trials_ba = trial_dict.get(pair_ba, [])
        
        if len(trials_ab) == 0 or len(trials_ba) == 0:
            # 缺少某个版本的试次
            inconsistent_count += 1
            continue
        
        # 取第一个试次（假设每个音频对只有一个试次）
        trial_ab = trials_ab[0]
        trial_ba = trials_ba[0]
        
        # 解析AB版本的左右音频
        parse_result_ab = parse_pair_name(pair_ab)
        if parse_result_ab is None:
            continue
        
        left_cat_ab, left_id_ab, right_cat_ab, right_id_ab, _ = parse_result_ab
        audio_a = f"{left_cat_ab}_{left_id_ab}"  # AB版本的左音频
        audio_b = f"{right_cat_ab}_{right_id_ab}"  # AB版本的右音频
        
        # 检查一致性
        # AB版本：response=1表示选左（选A），response=2表示选右（选B）
        # BA版本：response=1表示选左（选B），response=2表示选右（选A）
        
        # 情况1：AB选了A（response=1）且BA选了A（response=2，因为A在BA的右边）
        if trial_ab['response'] == 1 and trial_ba['response'] == 2:
            # 一致：都选了A，保留AB版本，标记为选左（choice=1）
            consistent_trials.append({
                'pair_name': pair_ab,
                'choice': 1,  # 选左（选A）
                'reaction_time': trial_ab.get('reaction_time', None)
            })
        # 情况2：AB选了B（response=2）且BA选了B（response=1，因为B在BA的左边）
        elif trial_ab['response'] == 2 and trial_ba['response'] == 1:
            # 一致：都选了B，保留AB版本，标记为选右（choice=2）
            consistent_trials.append({
                'pair_name': pair_ab,
                'choice': 2,  # 选右（选B）
                'reaction_time': trial_ab.get('reaction_time', None)
            })
        else:
            # 不一致：去掉
            inconsistent_count += 1
    
    print(f"  一致的试次: {len(consistent_trials)}")
    print(f"  不一致的试次: {inconsistent_count}")
    if len(consistent_trials) + inconsistent_count > 0:
        print(f"  保留比例: {len(consistent_trials)/(len(consistent_trials)+inconsistent_count)*100:.1f}%")
    
    return consistent_trials


def extract_features_for_model(consistent_trials, pair_names, model_config, diff_raw, diff_zscore, base_dir, feature_cache):
    """
    为一致的试次提取指定模型的特征
    
    Parameters:
    -----------
    consistent_trials : list
        一致的试次数据
    pair_names : list
        所有音频对名称（288个）
    model_config : dict
        模型配置，包含model_name和feature_ids
    diff_raw : np.ndarray
        原始特征差异矩阵
    diff_zscore : np.ndarray
        z-score归一化后的特征差异矩阵
    base_dir : str
        音频文件基础目录
    feature_cache : dict
        特征缓存
        
    Returns:
    --------
    X : np.ndarray, shape (n_trials, n_features)
        特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    valid_mask : np.ndarray
        有效数据掩码
    feature_names : list
        特征名称列表
    """
    print(f"\n=== 提取模型特征: {model_config['model_name']} ===")
    
    feature_ids = list(model_config['feature_ids'])
    
    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    
    # 检查是否是投票特征
    is_voting_feature = any('vote' in fid for fid in feature_ids)
    
    if is_voting_feature:
        # 投票特征：直接提取特征值
        print(f"  检测到投票特征，直接提取特征值")
        # 获取所有音频对的索引（用于提取特征）
        all_indices = list(range(len(pair_names)))
        feature_matrix = build_feature_matrix(feature_ids, diff_raw, diff_zscore, all_indices, pair_names, base_dir, feature_cache)
        feature_specs = [FEATURE_REGISTRY[fid] for fid in feature_ids]
        feature_names = [spec.display_name for spec in feature_specs]
    else:
        # 非投票特征：使用变换方式应用到9个特征上
        transform_method, alpha = parse_transformation_method(feature_ids[0])
        print(f"  变换方式: {transform_method}" + (f" (alpha={alpha})" if alpha is not None else ""))
        
        if transform_method == 'std_diff_raw':
            # 对于std_diff_raw，从diff_raw中读取对应的特征
            selected_indices_0based = [idx - 1 for idx in SELECTED_FEATURE_INDICES]
            all_indices = list(range(len(pair_names)))
            feature_matrix = diff_raw[selected_indices_0based, :][:, all_indices]
            feature_names = [f"{name} (std_diff_raw)" for name in SELECTED_FEATURE_NAMES]
        else:
            # 使用变换方式应用到9个特征上
            all_indices = list(range(len(pair_names)))
            feature_matrix = build_transformed_features_matrix(
                feature_ids[0], all_indices, pair_names, base_dir, feature_cache, diff_raw=diff_raw
            )
            feature_names = [f"{name} ({transform_method})" for name in SELECTED_FEATURE_NAMES]
    
    print(f"  特征矩阵形状: {feature_matrix.shape}")
    print(f"  特征数量: {len(feature_names)}")
    
    # 提取特征
    X_list = []
    y_choice_list = []
    valid_indices = []
    
    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']
        
        if pair_name not in pair_to_idx:
            continue
        
        idx = pair_to_idx[pair_name]
        
        # 提取特征
        # 特征矩阵是 (n_features, n_pairs)
        if idx >= feature_matrix.shape[1]:
            print(f"  警告: 索引 {idx} 超出特征矩阵范围 {feature_matrix.shape}")
            continue
            
        features = feature_matrix[:, idx]
        
        # 检查是否有NaN
        if np.isnan(features).any():
            continue
        
        X_list.append(features)
        y_choice_list.append(trial['choice'])
        valid_indices.append(i)
    
    if len(X_list) == 0:
        print("  警告: 没有提取到任何有效特征")
        return np.array([]), np.array([]), [], []
    
    X = np.array(X_list)
    y_choice = np.array(y_choice_list)
    
    print(f"  提取了 {len(X)} 个有效试次的特征")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  选左: {np.sum(y_choice == 1)} ({np.mean(y_choice == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(y_choice == 2)} ({np.mean(y_choice == 2)*100:.1f}%)")
    
    return X, y_choice, valid_indices, feature_names


def reset_feature_caches():
    """清空共享缓存，避免被试之间的相互影响"""
    PAIR_FEATURE_CACHE.clear()
    STD_CACHE.clear()
    _cache_stats['pair_feature_hits'] = 0
    _cache_stats['pair_feature_misses'] = 0
    _cache_stats['std_cache_hits'] = 0
    _cache_stats['std_cache_misses'] = 0


def fit_choice_model(X, y_choice, feature_names, model_name):
    """
    拟合个人选择模型（逻辑回归）
    
    Parameters:
    -----------
    X : np.ndarray
        特征矩阵
    y_choice : np.ndarray
        选择（1=选左，2=选右）
    feature_names : list
        特征名称列表
    model_name : str
        模型名称
        
    Returns:
    --------
    results : dict
        拟合结果
    """
    print(f"\n=== 拟合个人选择模型: {model_name} ===")
    
    # 转换为二分类：1=选左 -> 0, 2=选右 -> 1
    y_binary = (y_choice == 2).astype(int)
    
    print(f"  数据统计:")
    print(f"    选左（0）: {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"    选右（1）: {np.sum(y_binary == 1)} ({np.mean(y_binary == 1)*100:.1f}%)")
    
    if len(y_binary) < 10:
        print("  警告: 样本数太少，跳过拟合")
        return None
    
    # 使用statsmodels逻辑回归（有统计信息）
    X_with_const = sm.add_constant(X)
    try:
        logit_model = sm.Logit(y_binary, X_with_const).fit(disp=0, maxiter=1000)
    except Exception as e:
        print(f"  警告: statsmodels拟合失败: {e}")
        print(f"  使用sklearn逻辑回归")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y_binary)
        y_pred_proba = lr.predict_proba(X)[:, 1]
        y_pred = lr.predict(X)
        accuracy = accuracy_score(y_binary, y_pred)
        try:
            auc = roc_auc_score(y_binary, y_pred_proba)
        except:
            auc = np.nan
        
        baseline_accuracy = max(np.mean(y_binary), 1 - np.mean(y_binary))
        
        # 交叉验证
        cv_scores = cross_val_score(lr, X, y_binary, cv=5, scoring='accuracy')
        
        return {
            'model': lr,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_accuracy,
            'auc': auc,
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'y_true': y_binary,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': feature_names,
            'coefficients': None  # sklearn没有详细的系数信息
        }
    
    print(f"\n  模型统计:")
    print(f"    伪R2: {logit_model.prsquared:.4f}")
    print(f"    LL-Null: {logit_model.llnull:.4f}")
    print(f"    LLF: {logit_model.llf:.4f}")
    
    # 预测
    y_pred_proba = logit_model.predict(X_with_const)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 评估指标
    accuracy = accuracy_score(y_binary, y_pred)
    try:
        auc = roc_auc_score(y_binary, y_pred_proba)
    except:
        auc = np.nan
    
    baseline_accuracy = max(np.mean(y_binary), 1 - np.mean(y_binary))
    
    print(f"\n  预测准确率:")
    print(f"    准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    基线（多数类）: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"    提升: {(accuracy - baseline_accuracy)*100:.2f}%")
    print(f"    AUC: {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_binary, y_pred)
    print(f"\n  混淆矩阵:")
    print(f"    TN（真负例，预测左实际左）: {cm[0,0]}")
    print(f"    FP（假正例，预测右实际左）: {cm[0,1]}")
    print(f"    FN（假负例，预测左实际右）: {cm[1,0]}")
    print(f"    TP（真正例，预测右实际右）: {cm[1,1]}")
    
    # 交叉验证
    print(f"\n  5折交叉验证:")
    lr_cv = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(lr_cv, X, y_binary, cv=5, scoring='accuracy')
    print(f"    平均准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 系数信息
    print(f"\n  系数信息:")
    print(f"    截距: {logit_model.params[0]:.4f}, p={logit_model.pvalues[0]:.4f}")
    
    # 保存所有系数
    coefficients = []
    for i, name in enumerate(feature_names):
        coef = logit_model.params[i+1]
        pval = logit_model.pvalues[i+1]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"    {name}: {coef:.4f}, p={pval:.4f} {sig}")
        coefficients.append({
            'feature': name,
            'coefficient': coef,
            'pvalue': pval,
            'significant': sig
        })
    
    return {
        'model': logit_model,
        'accuracy': accuracy,
        'baseline_accuracy': baseline_accuracy,
        'auc': auc,
        'cv_accuracy_mean': np.mean(cv_scores),
        'cv_accuracy_std': np.std(cv_scores),
        'confusion_matrix': cm,
        'y_true': y_binary,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'coefficients': coefficients,
        'feature_names': feature_names
    }


def run_models_for_subject(subject_id, consistent_trials, pair_names, diff_raw, diff_zscore, base_dir, output_root):
    """针对单个被试运行全部模型并保存结果"""
    print(f"\n{'=' * 60}")
    print(f"开始评估被试: {subject_id}")
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

            result = fit_choice_model(X, y_choice, feature_names, model_name)

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
    print(f"{subject_id} 模型对比总结")
    print(f"{'=' * 60}")
    if valid_results:
        valid_results.sort(key=lambda x: x.get('cv_accuracy_mean', 0), reverse=True)
        print(f"\n成功拟合 {len(valid_results)} 个模型")
        top_k = min(5, len(valid_results))
        print(f"\n前{top_k}名模型（按CV准确率）:")
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

    print("\n=== 缓存统计 ===")
    print(f"时间序列缓存:")
    print(f"  命中: {_cache_stats['pair_feature_hits']} 次")
    print(f"  未命中: {_cache_stats['pair_feature_misses']} 次")
    total_pair = _cache_stats['pair_feature_hits'] + _cache_stats['pair_feature_misses']
    if total_pair > 0:
        hit_rate = _cache_stats['pair_feature_hits'] / total_pair * 100
        print(f"  命中率: {hit_rate:.1f}%")
    print(f"std缓存:")
    print(f"  命中: {_cache_stats['std_cache_hits']} 次")
    print(f"  未命中: {_cache_stats['std_cache_misses']} 次")
    total_std = _cache_stats['std_cache_hits'] + _cache_stats['std_cache_misses']
    if total_std > 0:
        hit_rate = _cache_stats['std_cache_hits'] / total_std * 100
        print(f"  命中率: {hit_rate:.1f}%")

    return valid_results


def plot_model_comparison(all_results, output_dir):
    """绘制模型对比图表"""
    print("\n=== 生成模型对比图表 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not all_results:
        print("  没有结果可绘制")
        return
    
    # 提取所有模型的性能指标
    model_names = []
    accuracies = []
    aucs = []
    cv_accuracies = []
    
    for result in all_results:
        if result is None:
            continue
        model_names.append(result.get('model_name', 'Unknown'))
        accuracies.append(result.get('accuracy', np.nan))
        aucs.append(result.get('auc', np.nan))
        cv_accuracies.append(result.get('cv_accuracy_mean', np.nan))
    
    if len(model_names) == 0:
        print("  没有有效结果")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 准确率对比
    ax = axes[0]
    sorted_idx = np.argsort(accuracies)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_acc = [accuracies[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_acc)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('准确率', fontsize=12)
    ax.set_title('模型准确率对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # 2. AUC对比
    ax = axes[1]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_auc = [aucs[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_auc)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('模型AUC对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # 3. 交叉验证准确率对比
    ax = axes[2]
    sorted_idx = np.argsort(cv_accuracies)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_cv_acc = [cv_accuracies[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_cv_acc)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('CV准确率', fontsize=12)
    ax.set_title('模型交叉验证准确率对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  对比图表已保存到: {output_dir}/model_comparison.png")


def save_results(all_results, consistent_trials, output_dir, subject_id=None):
    """保存所有模型的结果到CSV文件"""
    print("\n=== 保存结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存试次数据
    trials_df = pd.DataFrame(consistent_trials)
    trials_df.to_csv(os.path.join(output_dir, 'consistent_trials.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 保存模型对比汇总
    summary_data = []
    for result in all_results:
        if result is None:
            continue
        summary_data.append({
            'subject_id': subject_id,
            'model_name': result.get('model_name', 'Unknown'),
            'accuracy': result.get('accuracy', np.nan),
            'baseline_accuracy': result.get('baseline_accuracy', np.nan),
            'improvement': result.get('accuracy', np.nan) - result.get('baseline_accuracy', np.nan),
            'auc': result.get('auc', np.nan),
            'cv_accuracy_mean': result.get('cv_accuracy_mean', np.nan),
            'cv_accuracy_std': result.get('cv_accuracy_std', np.nan),
            'n_features': len(result.get('feature_names', [])),
            'n_samples': len(result.get('y_true', []))
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('cv_accuracy_mean', ascending=False)
        summary_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False, encoding='utf-8-sig')
        print(f"  模型对比汇总已保存: model_comparison_summary.csv")
    
    # 3. 为每个模型保存详细结果
    for result in all_results:
        if result is None:
            continue
        model_name = result.get('model_name', 'Unknown')
        safe_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        model_dir = os.path.join(output_dir, safe_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存预测结果
        pred_df = pd.DataFrame({
            'actual_choice': result.get('y_true', []),
            'predicted_choice': result.get('y_pred', []),
            'predicted_proba': result.get('y_pred_proba', [])
        })
        pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')
        
        # 保存系数
        if result.get('coefficients') is not None:
            coef_df = pd.DataFrame(result['coefficients'])
            coef_df.to_csv(os.path.join(model_dir, 'coefficients.csv'), index=False, encoding='utf-8-sig')
    
    print(f"  所有结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("个人数据模型对比：使用linear_regression_loudness_models中的所有模型")
    print("=" * 60)

    project_root = os.getcwd()
    output_root = os.path.join(project_root, 'individual_model_comparison_results')
    os.makedirs(output_root, exist_ok=True)

    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)
    base_dir = DEFAULT_AUDIO_BASE_DIR

    overall_summary_rows = []
    processed_subjects = 0

    for subject_config in SUBJECT_CONFIGS:
        subject_id = subject_config['subject_id']
        trials = load_subject_trials(subject_config)

        if len(trials) == 0:
            print(f"\n跳过 {subject_id}: 未能加载任何trial数据")
            continue

        consistent_trials = check_left_right_consistency(trials, pair_names)
        if len(consistent_trials) == 0:
            print(f"\n跳过 {subject_id}: 没有满足左右一致性的试次")
            continue

        if len(consistent_trials) < MIN_CONSISTENT_TRIALS_WARNING:
            print(f"\n警告: {subject_id} 的一致试次仅 {len(consistent_trials)} (<{MIN_CONSISTENT_TRIALS_WARNING})，结果可靠性可能较低（仍继续）")

        for trial in consistent_trials:
            trial['subject_id'] = subject_id

        subject_results = run_models_for_subject(
            subject_id,
            consistent_trials,
            pair_names,
            diff_raw,
            diff_zscore,
            base_dir,
            output_root
        )

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
        summary_path = os.path.join(output_root, 'all_subjects_model_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n所有被试模型对比汇总已保存: {summary_path}")
    else:
        print("\n未生成任何被试的模型结果")

    print(f"\n共处理 {processed_subjects}/{len(SUBJECT_CONFIGS)} 个被试")


# =============================================================================
# Kayser等人2005年听觉显著性地图算法的Python实现
# =============================================================================

def replicate_img(img):
    """
    复制图像边界以避免卷积时的边缘效应

    Parameters:
    -----------
    img : np.ndarray
        输入图像/频谱图

    Returns:
    --------
    out : np.ndarray
        扩展后的图像
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    S = img.shape

    out = np.zeros((S[0]*3, S[1]*3, S[2]))

    for s in range(S[2]):
        # 顶部
        out[0:S[0], 0:S[1], s] = np.fliplr(np.flipud(img[:, :, s]))
        out[0:S[0], S[1]:2*S[1], s] = np.fliplr(np.flipud(img[:, :, s]))
        out[0:S[0], 2*S[1]:3*S[1], s] = np.flipud(img[:, :, s])

        # 底部
        out[2*S[0]:3*S[0], 0:S[1], s] = np.flipud(img[:, :, s])
        out[2*S[0]:3*S[0], S[1]:2*S[1], s] = np.fliplr(np.flipud(img[:, :, s]))
        out[2*S[0]:3*S[0], 2*S[1]:3*S[1], s] = np.fliplr(np.flipud(img[:, :, s]))

        # 左右
        out[S[0]:2*S[0], 0:S[1], s] = np.fliplr(img[:, :, s])
        out[S[0]:2*S[0], 2*S[1]:3*S[1], s] = np.fliplr(img[:, :, s])

        # 中心
        out[S[0]:2*S[0], S[1]:2*S[1], s] = img[:, :, s]

    out = out[int(S[0]/2):int(S[0]/2)+S[0], int(S[1]/2):int(S[1]/2)+S[1], :]
    return out


def cropimg(img):
    """
    裁剪图像边界，撤销replicate_img的效果

    Parameters:
    -----------
    img : np.ndarray
        输入图像

    Returns:
    --------
    out : np.ndarray
        裁剪后的图像
    """
    if img.ndim != 2:
        raise ValueError("cropimg expects a 2D array")
    s = np.ceil(np.array(img.shape)/4).astype(int)
    out = img[s[0]+1:img.shape[0]-s[0]+1, s[1]+1:img.shape[1]-s[1]+1]
    return out


def resize_to_mapsize(img, map_size):
    """
    调整图像尺寸到指定大小
    """
    if img.shape == map_size:
        return img
    zoom_factors = (map_size[0] / img.shape[0], map_size[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)


def auditory_RF(sideband=0, postinh=0):
    """
    生成听觉感受野滤波器（Gabor-like）

    Parameters:
    -----------
    sideband : int
        是否包含侧带 (0: 无, 1: 有)
    postinh : int
        是否包含后抑制 (0: 无, 1: 有)

    Returns:
    --------
    out : np.ndarray
        滤波器核
    """
    # 参数设置
    LAT = np.array([24, 50])  # 延迟
    BW = np.array([0.035, 0.04])  # 带宽
    Freq = 1  # 频率
    DUR = 5  # 持续时间

    if sideband:
        BW[0] = 0.08
    else:
        BW[1] = 0.035

    # 频率轴（四分之一音）
    Fax = np.linspace(0.3, 1.7-0.000001, 32) / 32 * 1.7

    # 时间轴（1ms步长）
    Tax = np.arange(1, 81)

    T, F = np.meshgrid(Tax, Fax)

    LAT_scaled = LAT
    BW_scaled = BW * 4

    # Gabor包络
    env = np.exp(-((T - LAT_scaled[0])**2) / (2 * DUR**2) +
                 -((F - 1)**2) / (2 * BW_scaled[0]**2))

    # 振荡
    osc = np.cos(2 * np.pi * F * Freq)

    out = osc * env

    if postinh:
        DUR2 = DUR * 1.3
        env_post = np.exp(-((T - LAT_scaled[1])**2) / (2 * DUR2**2) +
                          -((F - 1)**2) / (2 * BW_scaled[1]**2))
        out = out - osc * env_post / 2

    # 归一化
    out = out / np.abs(np.sum(out))

    return out


def localextrema(img):
    """
    寻找图像的局部极值

    Parameters:
    -----------
    img : np.ndarray
        输入图像

    Returns:
    --------
    maxima : np.ndarray
        局部最大值位置
    minima : np.ndarray
        局部最小值位置
    """
    # 计算一阶导数
    d1 = np.diff(img, axis=0)
    d2 = np.diff(img, axis=1)

    # 检测零交叉
    dum1 = d1[:-1, :]
    dum2 = d1[1:, :]

    # 向下交叉
    cross11 = (dum1 > 0) * (dum2 <= 0)

    # 向上交叉
    cross21 = (dum1 < 0) * (dum2 >= 0)

    dum1 = d2[:, :-1]
    dum2 = d2[:, 1:]

    # 向下交叉
    cross12 = (dum1 > 0) * (dum2 <= 0)

    # 向上交叉
    cross22 = (dum1 < 0) * (dum2 >= 0)

    s = np.array(img.shape) - 2
    cross11 = cross11[:int(s[0]), :int(s[1])]
    cross12 = cross12[:int(s[0]), :int(s[1])]
    cross21 = cross21[:int(s[0]), :int(s[1])]
    cross22 = cross22[:int(s[0]), :int(s[1])]

    cross11 = np.column_stack([np.zeros((cross11.shape[0], 1)), cross11])
    cross12 = np.column_stack([np.zeros((cross12.shape[0], 1)), cross12])
    cross11 = np.row_stack([np.zeros((1, cross11.shape[1])), cross11])
    cross12 = np.row_stack([np.zeros((1, cross12.shape[1])), cross12])

    cross21 = np.column_stack([np.zeros((cross21.shape[0], 1)), cross21])
    cross22 = np.column_stack([np.zeros((cross22.shape[0], 1)), cross22])
    cross21 = np.row_stack([np.zeros((1, cross21.shape[1])), cross21])
    cross22 = np.row_stack([np.zeros((1, cross22.shape[1])), cross22])

    # 局部最大值：两个方向都向下交叉
    maxima = cross11 * cross12
    minima = cross21 * cross22

    return maxima, minima


def normalizemap(img, win=150):
    """
    使用滑动窗口对特征地图进行规范化

    Parameters:
    -----------
    img : np.ndarray
        输入特征地图
    win : int
        窗口长度

    Returns:
    --------
    out : np.ndarray
        规范化后的地图
    """
    warnings.filterwarnings('ignore')

    ntp = img.shape[1]
    winborder = np.arange(1, ntp+1, win)

    if ntp % win != 0:
        winborder = np.append(winborder, ntp)

    out = np.zeros_like(img)

    # 应用边界遮罩
    scale = 11
    h = np.hanning(scale*2)
    mask1 = np.ones_like(img)
    mask2 = np.ones_like(img)

    mask1[:scale, :] = h[:scale, np.newaxis] * np.ones((1, img.shape[1]))
    mask1[-scale:, :] = h[scale:, np.newaxis] * np.ones((1, img.shape[1]))
    mask2[:, :scale] = np.ones((img.shape[0], 1)) * h[:scale]
    mask2[:, -scale:] = np.ones((img.shape[0], 1)) * h[scale:]

    img = img * mask1 * mask2

    # 归一化到[0,1]
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img / np.max(img)

    # 寻找局部极值
    LocMa, LocMi = localextrema(img)
    LocMa_mask = np.zeros_like(img, dtype=bool)
    rows = min(LocMa.shape[0], img.shape[0])
    cols = min(LocMa.shape[1], img.shape[1])
    LocMa_mask[:rows, :cols] = LocMa[:rows, :cols] > 0

    for W in range(len(winborder)-1):
        I_win = np.arange(winborder[W], winborder[W+1]+1) - 1  # 转换为0索引
        I_all = np.arange(winborder[W]-win, winborder[W+1]+win/3+1) - 1
        I_all = I_all[(I_all >= 0) & (I_all < ntp)].astype(int)

        if len(I_all) == 0:
                continue
            
        data = img[:, I_all]
        max_data = np.max(data)

        if max_data == 0:
            continue

        data = data / max_data
        globmax = np.max(data)

        LocMa_slice = LocMa_mask[:, I_all]
        LocMaX = data[LocMa_slice]
        LocMaX = LocMaX[LocMaX != 1]  # 排除全局最大值

        if len(LocMaX) == 0:
            out_n = img[:, I_win] / max_data
        else:
            out_n = img[:, I_win] / max_data
            out_n = out_n * ((1 - np.mean(LocMaX))**2)

        out[:, I_win] = out_n

    warnings.filterwarnings('default')
    return out


def center_surround_pyramid(LEV_in, PYR, offsets, N, ONOFF):
    """
    在不同空间尺度上进行中心-周围交互

    Parameters:
    -----------
    LEV_in : int
        输入金字塔层级
    PYR : dict
        金字塔数据结构
    offsets : list
        尺度差异列表
    N : int
        金字塔层数
    ONOFF : list
        开关列表 [1,0] 或 [1,1]

    Returns:
    --------
    PYR : dict
        更新后的金字塔
    """
    LEV = LEV_in + 1

    # 初始化中心-周围层的容器
    for f in range(len(PYR['data'])):
        if not PYR['data'][f][LEV]:
            PYR['data'][f][LEV] = [[], []]
        if not PYR['helper'][f][LEV]:
            PYR['helper'][f][LEV] = [[], []]

    for n1 in range(N):
        for offset in offsets:
            n2 = n1 + offset
            if n2 >= N:
                continue

            for f in range(len(PYR['data'])):
                map_center = PYR['data'][f][LEV_in][n1]
                map_surround = PYR['data'][f][LEV_in][n2]

                if ONOFF[0]:
                    dummy = map_center - map_surround
                    cs_map = np.where(dummy > 0, dummy, 0)
                    PYR['data'][f][LEV][0].append(cs_map)
                    PYR['helper'][f][LEV][0].append(offset)

                if ONOFF[1]:
                    dummy = map_surround - map_center
                    cs_map = np.where(dummy > 0, dummy, 0)
                    PYR['data'][f][LEV][1].append(cs_map)
                    PYR['helper'][f][LEV][1].append(offset)

    return PYR


def saliency_map(img, N, ONOFF=[1, 0]):
    """
    计算输入频谱图的显著性地图

    Parameters:
    -----------
    img : np.ndarray
        输入频谱图
    N : int
        金字塔层数
    ONOFF : list
        开关列表 [1,0] 或 [1,1]

    Returns:
    --------
    SALIENCY : dict
        显著性地图，包含 eo, esi, epi
    """
    # 参数设置
    WINNORM = 150
    scale_interact = [1, 2]  # 尺度交互范围

    # 准备图像
    img2 = replicate_img(img)
    if img2.ndim == 3 and img2.shape[2] == 1:
        img2 = img2[:, :, 0]

    # 多尺度重采样
    IMG = []
    for n in range(N):
        scale_factor = 1 / (2 ** n)
        if scale_factor == 1:
            resampled = img2
        else:
            resampled = zoom(img2, (scale_factor, scale_factor), order=1)
        IMG.append(resampled)

    # 地图大小
    MapSize = img.shape

    # 初始化金字塔
    PYR = {
        'data': [[[] for _ in range(3)] for _ in range(3)],  # 3个特征，3个层级
        'helper': [[[] for _ in range(3)] for _ in range(3)]
    }

    # 第一级：特征提取
    LEV = 1
    for n in range(N):
        # EO) 强度特征
        Filter = auditory_RF(0, 0)
        feature_map = convolve(IMG[n], Filter, mode='nearest')
        feature_map = resize_to_mapsize(cropimg(feature_map), MapSize)
        PYR['data'][0][LEV].append(feature_map)

        # ESI) 频率对比特征
        Filter = auditory_RF(1, 0)
        feature_map = convolve(IMG[n], Filter, mode='nearest')
        feature_map = resize_to_mapsize(cropimg(feature_map), MapSize)
        PYR['data'][1][LEV].append(feature_map)

        # EPI) 后抑制特征
        Filter = auditory_RF(0, 1)
        feature_map = convolve(IMG[n], Filter, mode='nearest')
        feature_map = resize_to_mapsize(cropimg(feature_map), MapSize)
        PYR['data'][2][LEV].append(feature_map)

    # 中心-周围交互
    PYR = center_surround_pyramid(LEV, PYR, scale_interact, N, ONOFF)

    # 跨尺度融合
    LEV = 2

    def init_zero_like(feature_idx):
        level_data = PYR['data'][feature_idx][LEV]
        if not level_data:
            return np.zeros(MapSize)
        for maps in level_data:
            if maps:
                return np.zeros_like(maps[0])
        return np.zeros(MapSize)

    SALIENCY = {
        'eo': init_zero_like(0),
        'esi': init_zero_like(1),
        'epi': init_zero_like(2)
    }

    counts = {'eo': 0, 'esi': 0, 'epi': 0}

    for O in np.where(ONOFF)[0]:
        # EO
        for fmap in PYR['data'][0][LEV][O]:
            SALIENCY['eo'] += normalizemap(fmap, WINNORM)
            counts['eo'] += 1

        # ESI
        for fmap in PYR['data'][1][LEV][O]:
            SALIENCY['esi'] += normalizemap(fmap, WINNORM)
            counts['esi'] += 1

        # EPI
        for fmap in PYR['data'][2][LEV][O]:
            SALIENCY['epi'] += normalizemap(fmap, WINNORM)
            counts['epi'] += 1

    # 归一化
    for key in counts:
        if counts[key] > 0:
            SALIENCY[key] /= counts[key]

    return SALIENCY


def compute_auditory_saliency(img, N=4):
    """
    计算听觉显著性地图的主函数

    Parameters:
    -----------
    img : np.ndarray
        输入频谱图 (frequency x time)
    N : int
        金字塔层数，默认4

    Returns:
    --------
    saliency : np.ndarray
        显著性地图 (eo + esi + epi)
    saliency_components : dict
        各个组件的显著性地图
    """
    print(f"  计算显著性地图 (N={N})...")

    # 计算显著性组件
    saliency_components = saliency_map(img, N)

    # 合并所有组件
    saliency = (saliency_components['eo'] +
                saliency_components['esi'] +
                saliency_components['epi'])

    print(f"  显著性地图形状: {saliency.shape}")

    return saliency, saliency_components


if __name__ == '__main__':
    main()

