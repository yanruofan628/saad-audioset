#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAM分析：使用5秒std时间序列，GAM学习时间权重
- 从音频文件中提取选定基础特征的时间序列
- 计算5秒的std时间序列（连续时间序列，不分时间段）
- 使用GAM模型拟合左侧选择概率，时间作为平滑项学习时间权重
"""
import os
import re
import json
import collections
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, normaltest
from scipy import stats
from scipy.ndimage import uniform_filter1d
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import librosa

try:
    from pygam import LinearGAM, s, f, l
    try:
        from pygam import te
        TE_AVAILABLE = True
    except ImportError:
        TE_AVAILABLE = False
    PYGM_AVAILABLE = True
except ImportError:
    PYGM_AVAILABLE = False
    TE_AVAILABLE = False
    print("警告: pygam库未安装，请运行: pip install pygam")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============== 一致的5折划分配置（与Transformer脚本保持一致） ==============
CV_SEED = 42
N_FOLDS = 5

def build_consistent_folds(n_samples: int, seed: int = CV_SEED, k: int = N_FOLDS):
    """
    与 Transformer 脚本一致的折划分方式：
    - 先用固定随机种子打乱索引
    - 再用 np.array_split 切成 k 份
    返回 folds(list[np.ndarray]) 和 indices(np.ndarray, 打乱后的全索引)
    """
    indices = np.random.RandomState(seed).permutation(n_samples)
    folds = np.array_split(indices, k)
    return folds, indices


# ============== 复用原始代码的函数 ==============

def load_or_extract_features(project_root):
    """检测并加载已有特征数据，若不存在则提示"""
    raw_path = os.path.join(project_root, 'pair_feature_diff_raw.npy')
    zscore_path = os.path.join(project_root, 'pair_feature_diff_zscore.npy')
    list_path = os.path.join(project_root, 'pair_list.npy')

    if os.path.exists(raw_path) and os.path.exists(zscore_path) and os.path.exists(list_path):
        print("检测到已有特征文件，正在加载...")
        diff_raw = np.load(raw_path)
        diff_zscore = np.load(zscore_path)
        pair_names = np.load(list_path, allow_pickle=True)
        print(f"成功加载: {diff_raw.shape[1]} 个音频对, {diff_raw.shape[0]} 个特征")
        return diff_raw, diff_zscore, pair_names
    else:
        raise FileNotFoundError("未找到特征文件，请先运行 pair_feature_diff_extraction.py")


def _normalize_pair_key(name):
    """将音频对名称规范化为统一字符串便于匹配"""
    if name is None:
        return None
    if isinstance(name, bytes):
        try:
            name = name.decode('utf-8')
        except Exception:
            name = name.decode('utf-8', errors='ignore')
    name = str(name).strip().strip('"').strip()
    return name


def load_wav2vec_feature_zscore(project_root, pair_names, feature_key='cosine_sum_diff',
                               relative_dir='wav2vec_regression_results_select'):
    """加载Wav2Vec特征并返回与pair_names对齐的z-score数组"""
    candidate_dirs = []
    if relative_dir:
        candidate_dirs.append(relative_dir)
    candidate_dirs.extend([
        'wav2vec_regression_results',
        'wav2vec_regression_results_select'
    ])

    feature_path = None
    tried_paths = []
    for rel in candidate_dirs:
        if not rel:
            continue
        path = os.path.join(project_root, rel, 'wav2vec_features.json')
        tried_paths.append(path)
        if os.path.exists(path):
            feature_path = path
            break

    if feature_path is None:
        print("警告: 未找到Wav2Vec特征文件，已尝试路径:")
        for p in tried_paths:
            print(f"  - {p}")
        return None

    print(f"已检测到Wav2Vec特征文件: {feature_path}")

    try:
        with open(feature_path, 'r', encoding='utf-8') as f:
            feature_data = json.load(f)
    except Exception as exc:
        print(f"警告: 读取Wav2Vec特征文件失败: {exc}")
        return None

    feature_map = {}
    for key, value in feature_data.items():
        norm_key = _normalize_pair_key(key)
        if isinstance(value, dict) and feature_key in value:
            feature_map[norm_key] = float(value[feature_key])

    values = []
    missing_pairs = []
    for name in pair_names:
        norm_name = _normalize_pair_key(name)
        val = feature_map.get(norm_name)
        if val is None:
            missing_pairs.append(norm_name)
            values.append(np.nan)
        else:
            values.append(val)

    values = np.array(values, dtype=float)

    if missing_pairs:
        print(f"警告: Wav2Vec特征 {feature_key} 缺失 {len(missing_pairs)} 个音频对")
        print(f"  示例缺失: {missing_pairs[:5]}")

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 2:
        print(f"警告: Wav2Vec特征 {feature_key} 有效样本不足，返回原始值")
        return {
            'values': values,
            'zscore': values,
            'mean': np.nan,
            'std': np.nan
        }

    mean_val = values[valid_mask].mean()
    std_val = values[valid_mask].std()

    if std_val == 0:
        print(f"警告: Wav2Vec特征 {feature_key} 标准差为0，z-score将置为0")
        zscores = np.zeros_like(values)
        zscores[~valid_mask] = np.nan
    else:
        zscores = np.full_like(values, np.nan)
        zscores[valid_mask] = (values[valid_mask] - mean_val) / std_val

    print(f"已加载Wav2Vec特征 {feature_key}: 有效样本 {valid_mask.sum()}/{len(values)}, 平均值 {mean_val:.4f}, 标准差 {std_val:.4f}")

    return {
        'values': values,
        'zscore': zscores,
        'mean': mean_val,
        'std': std_val
    }


def parse_pair_name(filename):
    """解析音频对名称，提取左右音频标识和实验类型"""
    # 去掉扩展名
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    # 切分左右
    if '+' not in filename:
        return None, None, None, None, None

    left_part, right_part = filename.split('+', 1)

    # 提取实验类型
    experiment_type = None
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
    for p in prefixes:
        if left_part.startswith(p) or right_part.startswith(p):
            experiment_type = p.rstrip('_')
            break

    # 去掉前缀
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break

    # 提取类别和ID
    def extract_cat_id(part):
        # 找到最后一个下划线之前的第二个下划线
        last_us = part.rfind('_')
        if last_us == -1:
            return None, None
        second_last = part.rfind('_', 0, last_us)
        if second_last == -1:
            return None, None
        cat = part[:second_last]
        audio_id = part[second_last+1:]
        return cat, audio_id

    left_cat, left_id = extract_cat_id(left_part)
    right_cat, right_id = extract_cat_id(right_part)

    if left_cat is None or right_cat is None:
        return None, None, None, None, None

    return left_cat, left_id, right_cat, right_id, experiment_type


def merge_balanced_pairs_by_type(pair_names):
    """按实验类型分组合并左右平衡设计的音频对（288→144）"""
    print("\n=== 按实验类型合并左右平衡设计的音频对 ===")

    # 按实验类型分组
    type_groups = {'nn_main': [], 'nn_sub': [], 'main': [], 'sub': []}

    for idx, pair_name in enumerate(pair_names):
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[4] is None:
            print(f"警告: 无法解析实验类型 {pair_name}")
            continue

        left_cat, left_id, right_cat, right_id, experiment_type = parse_result
        type_groups[experiment_type].append((idx, pair_name))

    # 对每个类型进行合并
    merged_results = {}

    for exp_type, pairs in type_groups.items():
        if len(pairs) == 0:
            continue

        print(f"\n处理 {exp_type} 类型: {len(pairs)} 个音频对")

        unique_pairs = {}
        merged_indices_list = []
        merged_pair_names_list = []

        for idx, pair_name in pairs:
            left_cat, left_id, right_cat, right_id, _ = parse_pair_name(pair_name)

            # 创建规范化键（排序确保唯一性）
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            key = tuple(sorted([audio_a, audio_b]))

            if key not in unique_pairs:
                # 第一次出现，记录
                unique_pairs[key] = idx
                merged_indices_list.append(idx)
                merged_pair_names_list.append(pair_name)
            else:
                # 第二次出现（平衡配对），跳过
                pass

        print(f"  合并后唯一对数: {len(unique_pairs)}")

        merged_results[exp_type] = {
            'indices': merged_indices_list,
            'pair_names': merged_pair_names_list,
            'unique_pairs': unique_pairs
        }

    return merged_results


def parse_experiment_data_txt(file_path):
    """解析TXT格式的实验数据文件"""
    trials = []
    encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法读取文件: {file_path}")

    for i, line in enumerate(lines):
        if line.strip().startswith('Level: 3') and i+1 < len(lines) and '*** LogFrame Start ***' in lines[i+1]:
            trial_data = {}
            for j in range(i+1, min(i+20, len(lines))):
                current_line = lines[j].strip()
                if 'wavfile:' in current_line:
                    trial_data['wavfile'] = current_line.split('wavfile:')[1].strip()
                elif 'ImageDisplay1.RESP:' in current_line:
                    trial_data['response'] = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                elif 'ImageDisplay1.RT:' in current_line:
                    # RT单位是毫秒，转换为秒
                    rt_ms = float(current_line.split('ImageDisplay1.RT:')[1].strip())
                    trial_data['reaction_time'] = rt_ms / 1000.0
                elif '*** LogFrame End ***' in current_line:
                    break
            if 'wavfile' in trial_data and 'response' in trial_data:
                trials.append(trial_data)

    return trials


def parse_experiment_data_csv(file_path):
    """解析CSV格式的实验数据文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取CSV失败 {file_path}: {e}")
        return []

    trials = []
    for _, row in df.iterrows():
        audio_file = row['音频文件']
        choice = row['选择']

        if choice.lower() == 'left':
            response = 1
        elif choice.lower() == 'right':
            response = 2
        else:
            continue

        trials.append({
            'wavfile': audio_file,
            'response': response
        })

    return trials


def parse_audio_mapping(file_path):
    """解析音频映射文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        for _, row in df.iterrows():
            mapping[row['new_name']] = row['original_name']
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        return {}


def load_all_human_decisions(txt_files, csv_files, mapping_files):
    """加载所有人类决策数据，并标注被试ID"""
    print("\n=== 加载人类决策数据 ===")

    all_trials = []

    # 处理TXT文件
    for txt_file, mapping_file in zip(txt_files, mapping_files[:len(txt_files)]):
        if not os.path.exists(txt_file):
            print(f"TXT文件不存在: {txt_file}")
            continue

        # 从文件名提取被试ID
        subject_id = os.path.basename(os.path.dirname(txt_file)).split('_')[0]

        trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)

            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'subject_id': subject_id,
                    'original_name': original_name,
                    'response': trial['response']
                })

    # 处理CSV文件
    csv_mapping_files = mapping_files[len(txt_files):]
    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        if not os.path.exists(csv_file):
            print(f"CSV文件不存在: {csv_file}")
            continue

        # 从文件名提取被试ID
        base_name = os.path.basename(csv_file).replace('.csv', '')
        base_name = base_name.replace(' ', '')
        subject_id = re.sub(r'[\d_]+$', '', base_name).strip('_').strip()

        if not subject_id or subject_id.isdigit():
            match = re.match(r'^([a-zA-Z\u4e00-\u9fa5]+(?:\d+)?[a-zA-Z\u4e00-\u9fa5]*)', base_name)
            if match:
                subject_id = match.group(1)
            else:
                subject_id = base_name

        name_mapping = {
            'aiwenkai': 'aiwenkai',
            'lironghua': 'lironghua',
            'mayunmiao': 'mayunmiao',
            'ShangZiyang': 'ShangZiyang',
            'shangziyang': 'ShangZiyang',
            'wjy': 'wjy',
            'LiuYaorui': 'LiuYaorui',
            'liuyaorui': 'LiuYaorui',
        }

        subject_lower = subject_id.lower()
        for key, normalized in name_mapping.items():
            if subject_lower == key or subject_lower.startswith(key):
                subject_id = normalized
                break

        trials = parse_experiment_data_csv(csv_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)

            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'subject_id': subject_id,
                    'original_name': original_name,
                    'response': trial['response']
                })

    print(f"总共加载 {len(all_trials)} 个trial")
    unique_subjects = set(t['subject_id'] for t in all_trials)
    print(f"被试数量: {len(unique_subjects)}")

    return all_trials


def calculate_selection_probability_144(merged_pair_names, all_trials):
    """计算144对的左侧音频选择概率"""
    print("\n=== 计算144对的选择概率 ===")

    probabilities = []

    for pair_name in merged_pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            probabilities.append(np.nan)
            continue

        left_cat, left_id, right_cat, right_id, _ = parse_result

        # 左侧音频标识
        audio_left = f"{left_cat}_{left_id}"
        audio_right = f"{right_cat}_{right_id}"

        # 统计：左侧音频被选择的次数和出现次数
        selected_count = 0
        total_count = 0

        for trial in all_trials:
            trial_name = trial['original_name']
            trial_parse_result = parse_pair_name(trial_name)
            if trial_parse_result is None or trial_parse_result[0] is None:
                continue

            trial_left_cat, trial_left_id, trial_right_cat, trial_right_id, _ = trial_parse_result

            trial_audio_left = f"{trial_left_cat}_{trial_left_id}"
            trial_audio_right = f"{trial_right_cat}_{trial_right_id}"

            # 检查这个trial是否为AB组合对
            if (trial_audio_left == audio_left and trial_audio_right == audio_right) or \
               (trial_audio_left == audio_right and trial_audio_right == audio_left):
                total_count += 1
                # 检查是否选择了audio_left
                if (trial_audio_left == audio_left and trial['response'] == 1) or \
                   (trial_audio_right == audio_left and trial['response'] == 2):
                    selected_count += 1

        prob = selected_count / total_count if total_count > 0 else 0.5
        probabilities.append(prob)

    probabilities = np.array(probabilities)
    print(f"计算完成: {len(probabilities)} 个概率值")
    print(f"概率范围: [{np.nanmin(probabilities):.3f}, {np.nanmax(probabilities):.3f}]")
    print(f"概率均值: {np.nanmean(probabilities):.3f}")

    return probabilities


# ============== GLM版本的分析函数 ==============

def fit_glm_model(X, y, feature_names, family='gaussian'):
    """
    使用statsmodels拟合GLM模型
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)
        feature_names: 特征名称列表
        family: GLM分布族 ('gaussian', 'binomial', 'poisson', 'gamma')
    返回:
        model: 拟合的GLM模型
    """
    # 转换为DataFrame并设置列名
    X_df = pd.DataFrame(X, columns=feature_names)

    # 添加截距项
    X_with_const = sm.add_constant(X_df)

    # 根据family选择分布
    if family == 'gaussian':
        family_obj = sm.families.Gaussian()
    elif family == 'binomial':
        family_obj = sm.families.Binomial()
    elif family == 'poisson':
        family_obj = sm.families.Poisson()
    elif family == 'gamma':
        family_obj = sm.families.Gamma()
    else:
        raise ValueError(f"不支持的family: {family}")

    # 拟合GLM模型
    model = sm.GLM(y, X_with_const, family=family_obj).fit()

    return model


def extract_selected_features(feature_diff, selected_indices):
    """
    提取选定的特征
    参数:
        feature_diff: 完整的特征差矩阵 (n_features, n_pairs)
        selected_indices: 要保留的特征索引（从0开始），例如 [1, 3, 4, 5, 8, 9]
    返回:
        selected_features: 选定的特征矩阵 (n_selected, n_pairs)
    """
    return feature_diff[selected_indices, :]


def prepare_regression_data(feature_matrix, probabilities):
    """
    准备回归数据，去除NaN值
    参数:
        feature_matrix: 特征矩阵 (n_features, n_pairs)
        probabilities: 概率向量 (n_pairs,)
    返回:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)
        valid_mask: 有效数据的掩码
    """
    # 转置特征矩阵：从 (n_features, n_pairs) 转为 (n_pairs, n_features)
    X = feature_matrix.T

    # 检查每一行是否有NaN
    feature_mask = ~np.isnan(X).any(axis=1)
    prob_mask = ~np.isnan(probabilities)
    valid_mask = feature_mask & prob_mask

    X_clean = X[valid_mask, :]
    y_clean = probabilities[valid_mask]

    print(f"原始数据: {len(probabilities)} 个样本")
    print(f"有效数据: {np.sum(valid_mask)} 个样本")
    print(f"缺失数据: {len(probabilities) - np.sum(valid_mask)} 个样本")

    return X_clean, y_clean, valid_mask


def cross_validate_model(X, y, n_folds=N_FOLDS, seed: int = CV_SEED, save_dir: str = None):
    """
    使用与Transformer一致的折划分方式进行K折评估
    参数:
        X: 特征矩阵
        y: 目标变量
        n_folds: 折数
        seed: 随机种子（与Transformer保持一致）
        save_dir: 可选，若提供则保存每折的预测与真实值
    返回:
        cv_results: 交叉验证结果字典
    """
    print(f"\n=== {n_folds}折交叉验证（一致划分，seed={seed}）===")
    folds, indices = build_consistent_folds(len(y), seed=seed, k=n_folds)

    cv_r2_scores = []
    cv_mse_scores = []
    cv_mae_scores = []

    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练GLM模型
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        model = sm.GLM(y_train, X_train_const, family=sm.families.Gaussian()).fit()

        # 预测
        y_pred = model.predict(X_test_const)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fold_dir = os.path.join(save_dir, f'fold_{fold+1}')
            os.makedirs(fold_dir, exist_ok=True)
            pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred
            }).to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')

        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        cv_r2_scores.append(r2)
        cv_mse_scores.append(mse)
        cv_mae_scores.append(mae)

        print(f"Fold {fold+1}: R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

    cv_results = {
        'r2_mean': np.mean(cv_r2_scores),
        'r2_std': np.std(cv_r2_scores),
        'mse_mean': np.mean(cv_mse_scores),
        'mse_std': np.std(cv_mse_scores),
        'mae_mean': np.mean(cv_mae_scores),
        'mae_std': np.std(cv_mae_scores),
        'r2_scores': cv_r2_scores,
        'mse_scores': cv_mse_scores,
        'mae_scores': cv_mae_scores
    }

    print(f"\n交叉验证汇总:")
    print(f"  R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print(f"  MSE: {cv_results['mse_mean']:.4f} ± {cv_results['mse_std']:.4f}")
    print(f"  MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")

    return cv_results


def analyze_feature_correlations(X, feature_names, output_dir):
    """
    分析特征之间的相关性，检查多重共线性
    参数:
        X: 特征矩阵 (n_samples, n_features)
        feature_names: 特征名称列表
        output_dir: 输出目录
    返回:
        correlation_matrix: 相关系数矩阵
        vif_values: VIF值字典
    """
    print("\n=== 特征相关性分析 ===")
    os.makedirs(output_dir, exist_ok=True)

    # 计算相关系数矩阵
    df_features = pd.DataFrame(X, columns=feature_names)
    correlation_matrix = df_features.corr()

    print("\n特征相关系数矩阵:")
    print(correlation_matrix.round(4))

    # 保存相关系数矩阵
    correlation_matrix.to_csv(os.path.join(output_dir, 'feature_correlation_matrix.csv'),
                            encoding='utf-8-sig')

    # 找出高度相关的特征对 (|r| > 0.7)
    print("\n高度相关的特征对 (|r| > 0.7):")
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_value
                })
                print(f"  {feature_names[i]} <-> {feature_names[j]}: r = {corr_value:.4f}")

    if len(high_corr_pairs) == 0:
        print("  无高度相关的特征对")
    else:
        # 保存高度相关的特征对
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df.to_csv(os.path.join(output_dir, 'high_correlation_pairs.csv'),
                           index=False, encoding='utf-8-sig')

    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 只显示下三角
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n相关性热力图已保存: {os.path.join(output_dir, 'feature_correlation_heatmap.png')}")

    # 计算VIF（方差膨胀因子）
    print("\n=== VIF (方差膨胀因子) 分析 ===")
    print("VIF > 10 表示存在严重的多重共线性问题")
    print("VIF > 5 表示可能存在多重共线性问题")

    # 计算VIF需要添加常数项
    # 确保X是numpy数组
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.array(X)

    X_with_const = sm.add_constant(X_array)
    vif_values = {}

    for i in range(1, X_with_const.shape[1]):  # 跳过常数项
        feature_name = feature_names[i-1]
        # variance_inflation_factor 需要numpy数组
        vif = variance_inflation_factor(X_with_const, i)
        vif_values[feature_name] = vif
        status = "严重" if vif > 10 else "可能" if vif > 5 else "正常"
        print(f"  {feature_name}: VIF = {vif:.4f} ({status})")

    # 保存VIF值
    vif_df = pd.DataFrame({
        'feature': list(vif_values.keys()),
        'vif': list(vif_values.values())
    })
    vif_df = vif_df.sort_values('vif', ascending=False)
    vif_df.to_csv(os.path.join(output_dir, 'vif_values.csv'),
                 index=False, encoding='utf-8-sig')

    # 绘制VIF条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_df['vif']]
    bars = ax.barh(vif_df['feature'], vif_df['vif'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='VIF = 5')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.7, label='VIF = 10')
    ax.set_xlabel('VIF (Variance Inflation Factor)', fontsize=12)
    ax.set_title('Variance Inflation Factor (VIF) for Selected Features',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, vif) in enumerate(zip(bars, vif_df['vif'])):
        ax.text(vif, i, f' {vif:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vif_values.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VIF条形图已保存: {os.path.join(output_dir, 'vif_values.png')}")

    return correlation_matrix, vif_values


def extract_model_statistics(model, feature_names):
    """
    提取GLM模型统计信息
    参数:
        model: statsmodels GLM模型
        feature_names: 特征名称列表
    返回:
        stats_dict: 统计信息字典
    """
    # 获取系数（包括截距）
    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()

    # 将conf_int转换为numpy数组（如果是DataFrame）
    if isinstance(conf_int, pd.DataFrame):
        conf_int_values = conf_int.values
    else:
        conf_int_values = np.array(conf_int)

    # 获取特征名称（从模型的参数索引中提取，排除'const'）
    if isinstance(params, pd.Series):
        param_names = params.index.tolist()
        params_values = params.values
    else:
        param_names = [f'x{i}' for i in range(len(params))]  # 如果没有名称，使用默认名称
        params_values = np.array(params)

    if isinstance(pvalues, pd.Series):
        pvalues_values = pvalues.values
    else:
        pvalues_values = np.array(pvalues)

    # 找到截距的位置（通常是'const'或第一个）
    intercept_idx = 0
    if 'const' in param_names:
        intercept_idx = param_names.index('const')

    # 提取特征系数（排除截距），保留特征名称
    feature_names_from_model = [name for i, name in enumerate(param_names) if i != intercept_idx]
    feature_indices = [i for i in range(len(param_names)) if i != intercept_idx]
    feature_params = params_values[feature_indices]
    feature_pvalues = pvalues_values[feature_indices]
    feature_conf_int = conf_int_values[feature_indices]

    # 使用模型中的特征名称（如果可用），否则使用传入的feature_names
    if len(feature_names_from_model) == len(feature_names) and all(n in feature_names for n in feature_names_from_model):
        actual_feature_names = feature_names_from_model
    else:
        # 如果模型中的名称与传入的不匹配，使用传入的feature_names（按顺序）
        actual_feature_names = feature_names if len(feature_names) == len(feature_indices) else feature_names_from_model

    stats_dict = {
        'rsquared': model.pseudo_rsquared() if hasattr(model, 'pseudo_rsquared') else np.nan,  # GLM使用pseudo R²
        'rsquared_adj': np.nan,  # GLM通常不计算adjusted R²
        'aic': model.aic,
        'bic': model.bic,
        'fvalue': np.nan,  # GLM没有F统计量
        'f_pvalue': np.nan,
        'nobs': model.nobs,
        'df_model': model.df_model,
        'df_resid': model.df_resid,
        'intercept': params_values[intercept_idx],
        'intercept_pvalue': pvalues_values[intercept_idx],
        'intercept_conf_int': conf_int_values[intercept_idx],  # 截距的置信区间
        'params': feature_params,
        'pvalues': feature_pvalues,
        'conf_int': feature_conf_int,
        'feature_names': actual_feature_names  # 使用从模型中提取的实际特征名称
    }

    return stats_dict


def plot_predicted_vs_actual(y_true, y_pred, output_path):
    """绘制预测值vs实际值散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # 完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # 计算R²
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    ax.set_xlabel('Actual Probability', fontsize=12)
    ax.set_ylabel('Predicted Probability', fontsize=12)
    ax.set_title(f'Predicted vs Actual\nR² = {r2:.4f}, r = {corr:.4f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测值vs实际值图已保存: {output_path}")


def plot_residuals_vs_predicted(y_pred, residuals, output_path):
    """绘制残差vs预测值图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 散点图
    ax.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # 零线
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

    # 计算残差的统计量
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'Residuals vs Predicted\nMean={residual_mean:.4f}, Std={residual_std:.4f}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"残差vs预测值图已保存: {output_path}")


def plot_residual_qq(residuals, output_path):
    """绘制残差Q-Q图"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Q-Q图
    stats.probplot(residuals, dist="norm", plot=ax)

    # 正态性检验
    stat, p_value = normaltest(residuals)

    ax.set_title(f'Residual Q-Q Plot\nNormality test: p={p_value:.4f}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"残差Q-Q图已保存: {output_path}")


def plot_coefficient_forest(stats_dict, output_path):
    """绘制系数森林图（用于论文展示）"""
    feature_names = stats_dict['feature_names']
    params = stats_dict['params']
    conf_int = stats_dict['conf_int']
    pvalues = stats_dict['pvalues']

    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': params,
        'ci_lower': conf_int[:, 0],
        'ci_upper': conf_int[:, 1],
        'pvalue': pvalues
    })

    # 按系数大小排序
    df = df.sort_values('coefficient')

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(df))

    # 绘制置信区间
    for i, row in enumerate(df.iterrows()):
        idx, data = row
        color = 'red' if data['pvalue'] < 0.05 else 'gray'
        ax.plot([data['ci_lower'], data['ci_upper']], [i, i],
                color=color, linewidth=2, alpha=0.7)
        ax.plot(data['coefficient'], i, 'o', color=color, markersize=8)

        # 添加显著性标记
        sig_mark = '***' if data['pvalue'] < 0.001 else '**' if data['pvalue'] < 0.01 else '*' if data['pvalue'] < 0.05 else ''
        ax.text(data['coefficient'], i, sig_mark, ha='left', va='center', fontsize=10)

    # 零线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['feature'], fontsize=11)
    ax.set_xlabel('Coefficient (95% CI)', fontsize=12)
    ax.set_title('GLM Coefficient Forest Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"系数森林图已保存: {output_path}")


def save_model_results(stats_dict, cv_results, output_dir):
    """保存GLM模型结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存系数和统计信息
    results_df = pd.DataFrame({
        'feature': stats_dict['feature_names'],
        'coefficient': stats_dict['params'],
        'pvalue': stats_dict['pvalues'],
        'ci_lower': stats_dict['conf_int'][:, 0],
        'ci_upper': stats_dict['conf_int'][:, 1]
    })

    results_df.to_csv(os.path.join(output_dir, 'coefficients.csv'),
                      index=False, encoding='utf-8-sig')

    # 保存模型整体统计信息
    model_summary = pd.DataFrame({
        'metric': ['Pseudo R²', 'Adjusted R²', 'AIC', 'BIC', 'F-value', 'F p-value',
                   'N observations', 'DF model', 'DF residual',
                   'Intercept', 'Intercept p-value',
                   'CV R² (mean)', 'CV R² (std)', 'CV MSE (mean)', 'CV MSE (std)',
                   'CV MAE (mean)', 'CV MAE (std)'],
        'value': [
            stats_dict['rsquared'], stats_dict['rsquared_adj'],
            stats_dict['aic'], stats_dict['bic'],
            stats_dict['fvalue'], stats_dict['f_pvalue'],
            stats_dict['nobs'], stats_dict['df_model'], stats_dict['df_resid'],
            stats_dict['intercept'], stats_dict['intercept_pvalue'],
            cv_results['r2_mean'], cv_results['r2_std'],
            cv_results['mse_mean'], cv_results['mse_std'],
            cv_results['mae_mean'], cv_results['mae_std']
        ]
    })

    model_summary.to_csv(os.path.join(output_dir, 'model_summary.csv'),
                        index=False, encoding='utf-8-sig')

    print(f"\nGLM模型结果已保存到: {output_dir}")


def run_analysis_for_dataset_glm(
    dataset_feature_diff,
    dataset_pair_names,
    all_trials,
    selected_feature_names,
    fitting_feature_names,
    fitting_indices,
    exp_type_name,
    output_base_dir
):
    """
    GLM版本的分析函数：完全复制run_analysis_for_dataset，但用GLM替代OLS
    """
    print(f"\n\n{'='*80}")
    print(f"GLM分析数据集: {exp_type_name}")
    print(f"{'='*80}")

    # 创建该数据集的输出目录
    type_output_dir = os.path.join(output_base_dir, exp_type_name)
    os.makedirs(type_output_dir, exist_ok=True)

    # 1. 计算选择概率
    print(f"\n步骤1: 计算选择概率 ({exp_type_name})")
    probabilities = calculate_selection_probability_144(dataset_pair_names, all_trials)

    # 2. 准备回归数据
    print(f"\n步骤2: 准备回归数据 ({exp_type_name})")
    X, y, valid_mask = prepare_regression_data(dataset_feature_diff, probabilities)

    if len(y) < 10:
        print(f"警告: {exp_type_name} 的有效样本数太少 ({len(y)} < 10)，跳过分析")
        return None

    # 3. 特征相关性分析和多重共线性检查
    print(f"\n步骤3: 特征相关性分析和多重共线性检查 ({exp_type_name})")
    correlation_matrix, vif_values = analyze_feature_correlations(X, selected_feature_names, type_output_dir)

    # 4. 准备用于拟合的特征
    print(f"\n步骤4: 准备用于拟合的特征 ({exp_type_name})")
    X_fitting = X[:, fitting_indices]
    print(f"拟合特征矩阵形状: {X_fitting.shape} (原为 {X.shape})")

    # 5. 拟合GLM模型
    print(f"\n步骤5: 拟合GLM模型 ({exp_type_name})")
    model = fit_glm_model(X_fitting, y, fitting_feature_names, family='gaussian')

    # 打印模型摘要
    print(f"\n{'='*60}")
    print(f"GLM模型摘要 ({exp_type_name})")
    print(f"{'='*60}")
    print(model.summary())

    # 6. 提取模型统计信息
    print(f"\n步骤6: 提取模型统计信息 ({exp_type_name})")
    stats_dict = extract_model_statistics(model, fitting_feature_names)

    print(f"\nGLM模型拟合统计 ({exp_type_name}):")
    print(f"  Pseudo R² = {stats_dict['rsquared']:.4f}")
    print(f"  AIC = {stats_dict['aic']:.4f}")
    print(f"  BIC = {stats_dict['bic']:.4f}")

    print(f"\n系数信息 ({exp_type_name}):")
    print(f"  截距 = {stats_dict['intercept']:.4f}, p = {stats_dict['intercept_pvalue']:.4f}")
    for i, name in enumerate(stats_dict['feature_names']):
        print(f"  {name}: β = {stats_dict['params'][i]:.4f}, "
              f"p = {stats_dict['pvalues'][i]:.4f}, "
              f"95% CI = [{stats_dict['conf_int'][i, 0]:.4f}, {stats_dict['conf_int'][i, 1]:.4f}]")

    # 7. 交叉验证
    print(f"\n步骤7: 交叉验证 ({exp_type_name})")
    cv_results = cross_validate_model(
        X_fitting,
        y,
        n_folds=N_FOLDS,
        seed=CV_SEED,
        save_dir=os.path.join(type_output_dir, 'cv_folds')
    )

    # 8. 生成预测值和残差
    print(f"\n步骤8: 计算预测值和残差 ({exp_type_name})")
    X_df = pd.DataFrame(X_fitting, columns=fitting_feature_names)
    X_with_const = sm.add_constant(X_df)
    y_pred = model.predict(X_with_const)
    residuals = y - y_pred

    # 9. 绘制可视化图表
    print(f"\n步骤9: 绘制可视化图表 ({exp_type_name})")
    plot_predicted_vs_actual(y, y_pred, os.path.join(type_output_dir, 'predicted_vs_actual.png'))
    plot_residuals_vs_predicted(y_pred, residuals, os.path.join(type_output_dir, 'residuals_vs_predicted.png'))
    plot_residual_qq(residuals, os.path.join(type_output_dir, 'residual_qq.png'))
    plot_coefficient_forest(stats_dict, os.path.join(type_output_dir, 'coefficient_forest.png'))

    # 10. 保存结果
    print(f"\n步骤10: 保存结果 ({exp_type_name})")
    save_model_results(stats_dict, cv_results, type_output_dir)

    # 保存预测值和残差
    predictions_df = pd.DataFrame({
        'actual': y,
        'predicted': y_pred,
        'residuals': residuals
    })
    predictions_df.to_csv(os.path.join(type_output_dir, 'predictions.csv'),
                         index=False, encoding='utf-8-sig')

    return {
        'exp_type_name': exp_type_name,
        'stats_dict': stats_dict,
        'cv_results': cv_results,
        'n_samples': len(y)
    }


def extract_significant_features_from_table():
    """
    从用户提供的相关性结果表格中提取所有显著特征（significant=True）
    返回去重后的特征索引和名称
    """
    # 用户提供的表格数据（significant=True 的特征）
    # feature_idx 从1开始计数，对应前18个声学特征（或21个包括评分特征）
    significant_features_data = [
        # nn_main
        (3, 'F0 (mean)'),
        (7, 'Spectral Modulation Centroid (mean)'),
        (9, 'Spectral Irregularity (mean)'),
        (10, 'Spectral Irregularity (std)'),
        (12, 'Loudness (std)'),
        (17, 'Temporal Modulation (mean)'),
        # nn_sub
        (6, 'Harmonicity (std)'),
        (10, 'Spectral Irregularity (std)'),  # 重复
        (17, 'Temporal Modulation (mean)'),  # 重复
        # main
        (2, 'Spectral Centroid (std)'),
        (8, 'Spectral Modulation Centroid (std)'),
        (9, 'Spectral Irregularity (mean)'),  # 重复
        (10, 'Spectral Irregularity (std)'),  # 重复
        # main_sub
        (2, 'Spectral Centroid (std)'),  # 重复
        (8, 'Spectral Modulation Centroid (std)'),  # 重复
        (9, 'Spectral Irregularity (mean)'),  # 重复
        (10, 'Spectral Irregularity (std)'),  # 重复
        (17, 'Temporal Modulation (mean)'),  # 重复
        # nn_main_nn_sub
        (7, 'Spectral Modulation Centroid (mean)'),  # 重复
        (9, 'Spectral Irregularity (mean)'),  # 重复
        (10, 'Spectral Irregularity (std)'),  # 重复
        (12, 'Loudness (std)'),  # 重复
        (17, 'Temporal Modulation (mean)'),  # 重复
        # all
        (6, 'Harmonicity (std)'),  # 重复
        (7, 'Spectral Modulation Centroid (mean)'),  # 重复
        (9, 'Spectral Irregularity (mean)'),  # 重复
        (10, 'Spectral Irregularity (std)'),  # 重复
        (17, 'Temporal Modulation (mean)'),  # 重复
    ]

    # 去重：根据 feature_idx 和 feature_name
    unique_features = {}
    for feature_idx, feature_name in significant_features_data:
        if feature_idx not in unique_features:
            unique_features[feature_idx] = feature_name

    # 转换为排序后的列表
    sorted_indices = sorted(unique_features.keys())
    feature_indices_1based = sorted_indices  # 从1开始
    feature_names = [unique_features[idx] for idx in sorted_indices]

    # 转换为从0开始的索引（用于数组索引）
    feature_indices_0based = [idx - 1 for idx in feature_indices_1based]

    return feature_indices_0based, feature_indices_1based, feature_names




# ============== GLM模型函数 ==============
def fit_glm_model(X, y, feature_names, family='gaussian'):
    """
    使用statsmodels拟合GLM模型
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)
        feature_names: 特征名称列表
        family: GLM分布族 ('gaussian', 'binomial', 'poisson', 'gamma')
    返回:
        model: 拟合的GLM模型
    """
    # 转换为DataFrame并设置列名
    X_df = pd.DataFrame(X, columns=feature_names)

    # 添加截距项
    X_with_const = sm.add_constant(X_df)

    # 根据family选择分布
    if family == 'gaussian':
        family_obj = sm.families.Gaussian()
    elif family == 'binomial':
        family_obj = sm.families.Binomial()
    elif family == 'poisson':
        family_obj = sm.families.Poisson()
    elif family == 'gamma':
        family_obj = sm.families.Gamma()
    else:
        raise ValueError(f"不支持的family: {family}")

    # 拟合GLM模型
    model = sm.GLM(y, X_with_const, family=family_obj).fit()

    return model


# ============== 5秒std时间序列特征提取函数 ==============
def load_mono_5s(audio_path, target_sr=16000, duration_s=5.0):
    """加载5秒单声道音频，确保数据格式正确"""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # 确保是实数类型（不是复数）
    if np.iscomplexobj(y):
        y = np.real(y)
    
    # 确保是一维数组
    if y.ndim > 1:
        y = y.flatten()
    
    # 处理 NaN 和 Inf 值
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 确保数据类型是 float32 或 float64（不是 complex）
    y = y.astype(np.float32)
    
    # 如果音频太短或为空，返回零数组
    target_len = int(duration_s * target_sr)
    if len(y) == 0:
        y = np.zeros(target_len, dtype=np.float32)
    elif len(y) >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    
    # 最终检查：确保没有 NaN、Inf 或复数
    assert not np.iscomplexobj(y), f"音频数据仍包含复数: {audio_path}"
    assert np.all(np.isfinite(y)), f"音频数据包含 NaN 或 Inf: {audio_path}"
    
    return y, sr


def compute_rolling_std(series, window_size=5):
    """
    计算滑动窗口标准差时间序列
    
    参数:
        series: 输入时间序列 (T,)
        window_size: 滑动窗口大小（帧数）
    
    返回:
        滑动std时间序列 (T,)
    """
    series = np.asarray(series, dtype=float)
    if series.ndim > 1:
        series = series.flatten()
    elif series.ndim == 0:
        series = np.array([series])
    series = np.nan_to_num(series, nan=0.0)
    
    if len(series) == 0:
        return np.array([0.0], dtype=np.float32)
    
    if len(series) < window_size:
        global_std = np.std(series) if len(series) > 1 else 0.0
        return np.full(len(series), global_std, dtype=np.float32)
    
    window_size = max(1, min(window_size, len(series)))
    
    try:
        series_contiguous = np.ascontiguousarray(series.astype(np.float64))
        rolling_mean = uniform_filter1d(series_contiguous, size=int(window_size), mode='nearest')
        series_sq = series_contiguous ** 2
        rolling_mean_sq = uniform_filter1d(series_sq, size=int(window_size), mode='nearest')
        rolling_var = rolling_mean_sq - rolling_mean**2
        rolling_std = np.sqrt(np.maximum(rolling_var, 0))
        
        if len(rolling_std) != len(series):
            global_std = float(np.std(series)) if len(series) > 1 else 0.0
            return np.full(len(series), global_std, dtype=np.float32)
        
        return rolling_std.astype(np.float32)
    except Exception as e:
        global_std = float(np.std(series)) if len(series) > 1 else 0.0
        return np.full(len(series), global_std, dtype=np.float32)


def parse_stereo_pair_name_for_audio(pair_name, base_dir):
    """
    解析pair_name并找到对应的左右音频文件路径
    使用与 linear_regression_loudness_models.py 相同的解析逻辑
    
    参数:
        pair_name: 音频对名称
        base_dir: 基础目录（按类别组织的音频文件）
    
    返回:
        (left_path, right_path) 或 None
    """
    try:
        # 使用与 linear_regression_loudness_models.py 相同的解析函数
        # 但需要适配返回格式
        from pair_feature_diff_extraction import parse_stereo_pair_name
        
        left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, base_dir)
        
        left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
        right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")
        
        if os.path.exists(left_path) and os.path.exists(right_path):
            return (left_path, right_path)
        else:
            # 文件不存在，尝试备用方法
            if not os.path.exists(left_path):
                raise ValueError(f"左侧音频文件不存在: {left_path}")
            if not os.path.exists(right_path):
                raise ValueError(f"右侧音频文件不存在: {right_path}")
    except (ValueError, ImportError) as e:
        # 如果解析失败，尝试备用方法
        if pair_name.lower().endswith('.wav'):
            filename = pair_name[:-4]
        else:
            filename = pair_name
        
        if '+' not in filename:
            raise ValueError(f"无法找到 '+' 分隔符: {pair_name}")
        
        left_part, right_part = filename.split('+', 1)
        
        prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
        for p in prefixes:
            if left_part.startswith(p):
                left_part = left_part[len(p):]
                break
        for p in prefixes:
            if right_part.startswith(p):
                right_part = right_part[len(p):]
                break
        
        def split_category_and_id_fs(part):
            # 规则：第一个下划线前面是类别名，第一个下划线后面是文件ID
            # 格式通常是: "Helicopter_W3u2hj1x7gY_2" -> 类别: "Helicopter", 文件ID: "W3u2hj1x7gY_2"
            first_underscore_pos = part.find('_')
            if first_underscore_pos == -1:
                return None, None, None
            
            # 第一个下划线前面是类别名，后面是文件ID
            category = part[:first_underscore_pos]
            file_id = part[first_underscore_pos + 1:]
            candidate = os.path.join(base_dir, category, f"{file_id}.wav")
            
            if os.path.exists(candidate):
                return category, file_id, candidate
            else:
                return None, None, None
        
        left_cat, left_id, left_path = split_category_and_id_fs(left_part)
        right_cat, right_id, right_path = split_category_and_id_fs(right_part)
        
        if left_path and right_path:
            return (left_path, right_path)
        else:
            raise ValueError(f"备用解析方法失败 - left_part: {left_part}, right_part: {right_part}, base_dir: {base_dir}")


def extract_std_time_series_for_pair(
    pair_name: str,
    base_dir: str,
    selected_feature_types: List[str],
    hop_length: int = 256,
    n_time_windows: int = 10,
    window_duration: float = 0.5,
    sr: int = 16000
) -> Tuple[np.ndarray, List[str]]:
    """
    为单个音频对提取std时间序列特征
    
    参数:
        pair_name: 音频对名称
        base_dir: 音频文件基础目录
        selected_feature_types: 选定的特征类型列表（如 ['f0', 'loudness']）
        hop_length: 帧移
        n_time_windows: 时间段数量（例如10个时间段）
        window_duration: 每个时间段的时长（秒，例如0.5秒）
        sr: 采样率
    
    返回:
        feature_matrix: (n_features, n_time_windows) 特征矩阵，每个时间段一个统计量
        feature_names: 特征名称列表
    """
    from pair_feature_diff_extraction import extract_time_series_features
    
    # 解析音频路径
    try:
        audio_paths = parse_stereo_pair_name_for_audio(pair_name, base_dir)
        if audio_paths is None:
            raise ValueError(f"无法解析音频路径: {pair_name}")
        left_path, right_path = audio_paths
    except Exception as e:
        raise ValueError(f"解析音频路径失败 {pair_name}: {e}")
    
    # 加载音频
    try:
        yL, srL = load_mono_5s(left_path, target_sr=sr)
        yR, srR = load_mono_5s(right_path, target_sr=sr)
    except Exception as e:
        raise ValueError(f"加载音频失败 {pair_name}: {e}")
    
    # 提取特征时间序列
    try:
        # 确保音频数据格式正确（再次检查，以防万一）
        yL = np.asarray(yL, dtype=np.float32)
        yR = np.asarray(yR, dtype=np.float32)
        if np.iscomplexobj(yL):
            yL = np.real(yL)
        if np.iscomplexobj(yR):
            yR = np.real(yR)
        yL = np.nan_to_num(yL, nan=0.0, posinf=0.0, neginf=0.0)
        yR = np.nan_to_num(yR, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 如果音频太短（少于 hop_length 个样本），跳过 F0 提取
        # 对于其他特征，使用 try-except 包装
        featL_dict = extract_time_series_features(yL, sr=sr, hop_length=hop_length)
        featR_dict = extract_time_series_features(yR, sr=sr, hop_length=hop_length)
    except Exception as e:
        # 如果是 F0 提取失败，尝试跳过 F0 特征
        if 'f0' in selected_feature_types or 'pyin' in str(e).lower():
            print(f"  警告: F0提取失败 {pair_name}，尝试跳过F0特征: {e}")
            # 如果只需要 F0，则失败
            if selected_feature_types == ['f0']:
                raise ValueError(f"提取特征时间序列失败（仅F0特征）{pair_name}: {e}")
            # 否则，尝试提取其他特征
            try:
                # 创建一个不包含 F0 的特征类型列表
                other_feature_types = [ft for ft in selected_feature_types if ft != 'f0']
                if len(other_feature_types) == 0:
                    raise ValueError(f"所有特征类型都失败: {pair_name}")
                # 重新提取（不使用 F0）
                featL_dict = extract_time_series_features(yL, sr=sr, hop_length=hop_length)
                featR_dict = extract_time_series_features(yR, sr=sr, hop_length=hop_length)
                # 从字典中移除 F0
                if 'f0' in featL_dict:
                    del featL_dict['f0']
                if 'f0' in featR_dict:
                    del featR_dict['f0']
            except Exception as e2:
                raise ValueError(f"提取特征时间序列失败 {pair_name}: {e2}")
        else:
            # 如果成功提取了特征（跳过了F0），继续执行
            pass
    
    # 特征名称映射
    feature_name_map = {
        'spectral_centroid': 'Spectral Centroid',
        'f0': 'F0',
        'harmonicity': 'Harmonicity',
        'spec_mod_centroid': 'Spectral Modulation Centroid',
        'spectral_irregularity': 'Spectral Irregularity',
        'loudness': 'Loudness',
        'spectral_flatness': 'Spectral Flatness',
        'spectral_bandwidth': 'Spectral Bandwidth',
        'temporal_mod': 'Temporal Modulation'
    }
    
    # 计算std时间序列并取差值
    feature_matrix_list = []
    feature_names_list = []
    
    for feat_type in selected_feature_types:
        if feat_type not in featL_dict or feat_type not in featR_dict:
            print(f"警告: 特征 {feat_type} 不存在，跳过")
            continue
        
        featL_ts = featL_dict[feat_type]
        featR_ts = featR_dict[feat_type]
        
        # 对于 f0 特征，检查是否全是 NaN（提取失败的情况）
        if feat_type == 'f0':
            # 检查是否全是 NaN
            if np.all(np.isnan(featL_ts)) or np.all(np.isnan(featR_ts)):
                print(f"警告: 特征 {feat_type} 提取失败（全为NaN），跳过")
                continue
        
        # 确保长度一致
        T = min(len(featL_ts), len(featR_ts))
        featL_ts = featL_ts[:T]
        featR_ts = featR_ts[:T]
        
        # 计算每个时间段的帧数
        # 总时长 = T * hop_length / sr 秒
        total_duration = T * hop_length / sr  # 秒
        window_size_frames = int(window_duration * sr / hop_length)  # 每个时间段的帧数
        
        # 将时间序列分成时间段，每个时间段计算std
        stdL_windows = []
        stdR_windows = []
        
        for w in range(n_time_windows):
            # 计算当前时间段的起始和结束帧索引
            start_frame = int(w * T / n_time_windows)
            end_frame = int((w + 1) * T / n_time_windows)
            
            # 提取当前时间段的数据
            segmentL = featL_ts[start_frame:end_frame]
            segmentR = featR_ts[start_frame:end_frame]
            
            # 去除NaN值后计算std
            validL = segmentL[~np.isnan(segmentL)]
            validR = segmentR[~np.isnan(segmentR)]
            
            if len(validL) > 0:
                stdL_windows.append(np.std(validL))
            else:
                stdL_windows.append(0.0)
            
            if len(validR) > 0:
                stdR_windows.append(np.std(validR))
            else:
                stdR_windows.append(0.0)
        
        # 转换为numpy数组
        stdL_windows = np.array(stdL_windows, dtype=np.float32)
        stdR_windows = np.array(stdR_windows, dtype=np.float32)
        
        # 计算差值
        std_diff = stdL_windows - stdR_windows
        
        # 添加到特征矩阵
        feature_matrix_list.append(std_diff)
        feat_name = feature_name_map.get(feat_type, feat_type)
        feature_names_list.append(f"{feat_name} (std diff)")
    
    if len(feature_matrix_list) == 0:
        raise ValueError(f"未能提取任何特征: {pair_name}")
    
    # 堆叠特征矩阵: (n_features, T)
    feature_matrix = np.stack(feature_matrix_list, axis=0)
    
    return feature_matrix, feature_names_list


# ============== 时间窗std时间序列特征辅助函数 ==============
def load_timewindow_features(project_root: str):
    """加载时间窗特征矩阵，用于提取std时间序列"""
    zscore_path = os.path.join(project_root, 'pair_feature_diff_timewindows_zscore.npy')
    list_path = os.path.join(project_root, 'pair_list.npy')
    metadata_path = os.path.join(project_root, 'timewindow_metadata.json')

    for path in [zscore_path, list_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到 {path}，请先运行 pair_feature_diff_extraction.py 生成时间窗特征。"
            )

    diff_timewindow_zscore = np.load(zscore_path)
    pair_names = np.load(list_path, allow_pickle=True)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    n_features = metadata['n_features']
    n_time_windows = metadata['n_time_windows']
    window_duration = metadata['window_duration']

    print(f"成功加载时间窗特征: {diff_timewindow_zscore.shape[1]} 个音频对, {diff_timewindow_zscore.shape[0]} 个特征维度")
    print(f"特征结构: {n_features}个基础特征 × {n_time_windows}个时间窗 × 2统计量(mean/std) = {diff_timewindow_zscore.shape[0]}维")
    print(f"每个时间窗时长: {window_duration}秒")

    return diff_timewindow_zscore, pair_names, metadata


def extract_std_time_series_feature_matrix(
    diff_timewindow_zscore: np.ndarray,
    pair_names: np.ndarray,
    dataset_pair_names: List[str],
    selected_feature_indices: List[int],
    metadata: Dict[str, Any],
) -> (np.ndarray, List[str]):
    """
    提取选定特征的std时间序列（每个特征在各个时间窗的std值）

    参数:
        diff_timewindow_zscore: (180, 288) 时间窗特征矩阵
            结构: 9个特征 × 10个时间窗 × 2统计量(mean/std) = 180维
            顺序: [f0_mean_t0, f0_std_t0, f0_mean_t1, f0_std_t1, ..., f0_mean_t9, f0_std_t9,
                  f1_mean_t0, f1_std_t0, ...]
        pair_names: 288个文件的名称列表
        dataset_pair_names: 当前数据集的音频对名称列表（144个）
        selected_feature_indices: 选定的基础特征索引（0-8，对应9个基础特征）
        metadata: 时间窗元数据
    
    返回:
        matrix: (n_selected_features * n_time_windows, n_pairs) 特征矩阵
        feature_names: 特征名称列表，格式为 "Feature Name (std) [t0-t1s]"
    """
    n_features = metadata['n_features']
    n_time_windows = metadata['n_time_windows']
    window_duration = metadata['window_duration']
    feature_names_base = metadata['feature_names']

    # 找到dataset_pair_names在pair_names中的索引
    dataset_indices = []
    for pair_name in dataset_pair_names:
        for i, full_pair_name in enumerate(pair_names):
            if _normalize_pair_key(full_pair_name) == _normalize_pair_key(pair_name):
                dataset_indices.append(i)
                break

    print(f"数据集包含 {len(dataset_indices)} 个音频对 (映射到原始288个文件的索引)")

    rows = []
    feature_names = []

    # 对于每个选定的基础特征
    for feat_idx in selected_feature_indices:
        if feat_idx >= n_features:
            print(f"警告: 特征索引 {feat_idx} 超出范围 (最大 {n_features-1})，跳过")
            continue
        
        # 对于每个时间窗，提取std值
        for t_idx in range(n_time_windows):
            # 计算在时间窗特征矩阵中的索引
            # 每个特征占 2 * n_time_windows 个位置（mean和std交替）
            # std的索引 = feat_idx * 2 * n_time_windows + t_idx * 2 + 1
            timewindow_feat_idx = feat_idx * 2 * n_time_windows + t_idx * 2 + 1
            
            # 提取该特征在该时间窗的std值
            std_values = diff_timewindow_zscore[timewindow_feat_idx, dataset_indices]
            rows.append(std_values)
            
            # 生成特征名称
            feat_name = feature_names_base[feat_idx]
            time_start = t_idx * window_duration
            time_end = (t_idx + 1) * window_duration
            feature_names.append(f"{feat_name} (std) [{time_start:.1f}-{time_end:.1f}s]")

    matrix = np.array(rows, dtype=float)
    print(f"提取的std时间序列特征矩阵形状: {matrix.shape} (特征×音频对)")
    print(f"  包含 {len(selected_feature_indices)} 个基础特征 × {n_time_windows} 个时间窗 = {len(feature_names)} 个特征")
    
    return matrix, feature_names


# ============== GAM模型拟合函数 ==============
def _safe_get_statistic(stat_dict, key, default=np.nan):
    """
    安全地从GAM统计字典中提取数值
    
    参数:
        stat_dict: GAM模型的statistics_字典
        key: 统计量的键名
        default: 如果无法提取时的默认值
    
    返回:
        数值（float）
    """
    if key not in stat_dict:
        return default
    
    value = stat_dict[key]
    
    # 如果是字典或OrderedDict，尝试提取第一个值或'explained_deviance'
    if isinstance(value, (dict, collections.OrderedDict)):
        if 'explained_deviance' in value:
            return float(value['explained_deviance'])
        elif len(value) > 0:
            # 取第一个值
            return float(list(value.values())[0])
        else:
            return default
    
    # 如果是数值类型，直接返回
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fit_gam_model(X_time_series, y, feature_names, n_splines=5, lam=0.1, 
                  use_smooth_features=True, use_interaction=False, aggregation_method='weighted_mean'):
    """
    使用pygam拟合GAM模型，学习时间权重
    
    参数:
        X_time_series: (n_samples, n_features, T) 时间序列特征矩阵
        y: (n_samples,) 目标变量
        feature_names: 特征名称列表
        n_splines: 平滑样条的数量
        lam: 正则化参数（lambda），越大越平滑，防止过拟合
        use_smooth_features: 如果True，特征使用平滑项s()；如果False，使用线性项l()
        use_interaction: 如果True，添加特征-时间交互项
        aggregation_method: 聚合方法
            - 'weighted_mean': 使用GAM学习时间权重，然后加权平均（推荐）
            - 'flatten': 展平时间序列（旧方法，可能效果差）
            - 'statistics': 提取时间序列统计量（mean, std, max, min等）
    
    返回:
        model: 拟合的GAM模型
        X_gam: 输入特征矩阵
        y_gam: 目标变量
        time_weights: 学习到的时间权重（如果使用weighted_mean）
    """
    if not PYGM_AVAILABLE:
        raise ImportError("pygam库未安装，请运行: pip install pygam")
    
    n_samples, n_features, T = X_time_series.shape
    
    print(f"  聚合方法: {aggregation_method}")
    print(f"  正则化参数 lambda={lam}, 样条数量={n_splines}")
    print(f"  特征项类型: {'平滑项s()' if use_smooth_features else '线性项l()'}")
    
    if aggregation_method == 'weighted_mean':
        # 方法1：使用GAM学习时间权重，然后对特征进行时间加权聚合
        # 对每个特征，学习其在各个时间点的权重
        # 然后使用加权平均聚合特征
        
        # 准备数据：将时间序列展平
        X_flat = []
        y_flat = []
        time_flat = []
        
        for i in range(n_samples):
            for t in range(T):
                # 特征值
                feat_values = X_time_series[i, :, t]
                X_flat.append(feat_values)
                # 目标值（每个时间点使用相同的y值）
                y_flat.append(y[i])
                # 时间索引（归一化到0-1）
                time_flat.append(t / T)
        
        X_flat = np.array(X_flat)  # (n_samples * T, n_features)
        y_flat = np.array(y_flat)  # (n_samples * T,)
        time_flat = np.array(time_flat)  # (n_samples * T,)
        
        print(f"  GAM输入数据形状: X_gam={X_flat.shape[0]} 个观测, {X_flat.shape[1]} 个特征")
        
        # 构建GAM公式：特征 + 时间，学习时间如何影响预测
        if use_smooth_features:
            terms = s(0, n_splines=n_splines, lam=lam)
            for i in range(1, n_features):
                terms = terms + s(i, n_splines=n_splines, lam=lam)
        else:
            terms = l(0)
            for i in range(1, n_features):
                terms = terms + l(i)
        
        # 时间平滑项
        terms = terms + s(n_features, n_splines=n_splines, lam=lam)
        
        # 如果启用交互项
        if use_interaction and TE_AVAILABLE:
            for i in range(n_features):
                terms = terms + te(i, n_features, n_splines=n_splines, lam=lam)
            print(f"  已添加 {n_features} 个特征-时间交互项")
        
        # 合并特征和时间
        X_gam = np.column_stack([X_flat, time_flat])  # (n_samples * T, n_features + 1)
        
        # 拟合GAM模型
        gam = LinearGAM(terms)
        gam.fit(X_gam, y_flat)
        
        # 提取时间权重函数
        XX_time = np.linspace(0, 1, 100).reshape(-1, 1)
        # 创建一个虚拟的特征矩阵（全0），只保留时间维度
        XX_dummy = np.zeros((100, n_features))
        XX_for_time = np.column_stack([XX_dummy, XX_time])
        time_effect = gam.partial_dependence(term=n_features, X=XX_for_time)
        
        return gam, X_gam, y_flat, time_effect
        
    elif aggregation_method == 'statistics':
        # 方法2：提取时间序列统计量，然后使用标准GAM
        # 对每个特征，提取：mean, std, max, min, trend等统计量
        X_agg = []
        
        for i in range(n_samples):
            sample_features = []
            for feat_idx in range(n_features):
                feat_time_series = X_time_series[i, feat_idx, :]
                # 提取统计量
                sample_features.extend([
                    np.mean(feat_time_series),      # 均值
                    np.std(feat_time_series),       # 标准差
                    np.max(feat_time_series),       # 最大值
                    np.min(feat_time_series),       # 最小值
                    np.median(feat_time_series),   # 中位数
                ])
            X_agg.append(sample_features)
        
        X_agg = np.array(X_agg)  # (n_samples, n_features * 5)
        n_agg_features = X_agg.shape[1]
        
        print(f"  聚合后特征数量: {n_agg_features} (每个原始特征提取5个统计量)")
        
        # 构建GAM公式
        if use_smooth_features:
            terms = s(0, n_splines=n_splines, lam=lam)
            for i in range(1, n_agg_features):
                terms = terms + s(i, n_splines=n_splines, lam=lam)
        else:
            terms = l(0)
            for i in range(1, n_agg_features):
                terms = terms + l(i)
        
        X_gam = X_agg
        
        # 拟合GAM模型
        gam = LinearGAM(terms)
        gam.fit(X_gam, y)
        
        return gam, X_gam, y, None
        
    else:  # 'flatten' - 旧方法
        # 方法3：展平时间序列（旧方法，保留用于对比）
        X_flat = []
        y_flat = []
        time_flat = []
        
        for i in range(n_samples):
            for t in range(T):
                feat_values = X_time_series[i, :, t]
                X_flat.append(feat_values)
                y_flat.append(y[i])
                time_flat.append(t / T)
        
        X_flat = np.array(X_flat)
        y_flat = np.array(y_flat)
        time_flat = np.array(time_flat)
        
        print(f"  GAM输入数据形状: X_gam={X_flat.shape[0]} 个观测, {X_flat.shape[1]} 个特征")
        
        if use_smooth_features:
            terms = s(0, n_splines=n_splines, lam=lam)
            for i in range(1, n_features):
                terms = terms + s(i, n_splines=n_splines, lam=lam)
        else:
            terms = l(0)
            for i in range(1, n_features):
                terms = terms + l(i)
        
        terms = terms + s(n_features, n_splines=n_splines, lam=lam)
        
        if use_interaction and TE_AVAILABLE:
            for i in range(n_features):
                terms = terms + te(i, n_features, n_splines=n_splines, lam=lam)
        
        X_gam = np.column_stack([X_flat, time_flat])
        
        gam = LinearGAM(terms)
        gam.fit(X_gam, y_flat)
        
        return gam, X_gam, y_flat, None


def cross_validate_gam(X_time_series, y, n_folds=N_FOLDS, seed: int = CV_SEED, 
                       n_splines=5, lam=0.1, use_smooth_features=True, use_interaction=False,
                       aggregation_method='weighted_mean'):
    """
    使用GAM进行交叉验证
    
    参数:
        X_time_series: (n_samples, n_features, T) 时间序列特征矩阵
        y: (n_samples,) 目标变量
        n_folds: 交叉验证折数
        seed: 随机种子
        n_splines: 平滑样条数量
        lam: 正则化参数（lambda）
        use_smooth_features: 如果True，特征使用平滑项s()；如果False，使用线性项l()
        use_interaction: 如果True，添加特征-时间交互项
        aggregation_method: 聚合方法（'weighted_mean', 'statistics', 'flatten'）
    """
    if not PYGM_AVAILABLE:
        raise ImportError("pygam库未安装，请运行: pip install pygam")
    
    print(f"\n=== {n_folds}折交叉验证（GAM，seed={seed}, lambda={lam}, n_splines={n_splines}）===")
    print(f"  聚合方法: {aggregation_method}")
    print(f"  特征项类型: {'平滑项s()' if use_smooth_features else '线性项l()'}, 交互项: {'启用' if use_interaction else '禁用'}")
    folds, indices = build_consistent_folds(len(y), seed=seed, k=n_folds)
    
    cv_r2_scores = []
    cv_mse_scores = []
    cv_mae_scores = []
    
    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        X_train = X_time_series[train_idx]
        X_test = X_time_series[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # 使用fit_gam_model拟合模型
        gam_model, X_train_gam, y_train_gam, _ = fit_gam_model(
            X_train, y_train, None,  # feature_names在交叉验证中不需要
            n_splines=n_splines,
            lam=lam,
            use_smooth_features=use_smooth_features,
            use_interaction=use_interaction,
            aggregation_method=aggregation_method
        )
        
        # 预测测试集
        if aggregation_method == 'statistics':
            # 对于statistics方法，需要先聚合测试集
            X_test_agg = []
            for i in range(X_test.shape[0]):
                sample_features = []
                for feat_idx in range(X_test.shape[1]):
                    feat_time_series = X_test[i, feat_idx, :]
                    sample_features.extend([
                        np.mean(feat_time_series),
                        np.std(feat_time_series),
                        np.max(feat_time_series),
                        np.min(feat_time_series),
                        np.median(feat_time_series),
                    ])
                X_test_agg.append(sample_features)
            X_test_gam = np.array(X_test_agg)
            y_pred = gam_model.predict(X_test_gam)
        else:
            # 对于weighted_mean和flatten方法，需要展平测试集
            n_test, _, T_test = X_test.shape
            X_test_flat = []
            time_test_flat = []
            for i in range(n_test):
                for t in range(T_test):
                    X_test_flat.append(X_test[i, :, t])
                    time_test_flat.append(t / T_test)
            
            X_test_gam = np.column_stack([np.array(X_test_flat), np.array(time_test_flat)])
            y_pred_flat = gam_model.predict(X_test_gam)
            
            # 对每个样本，取所有时间点的平均预测值
            y_pred = []
            idx = 0
            for i in range(n_test):
                preds = y_pred_flat[idx:idx+T_test]
                y_pred.append(np.mean(preds))
                idx += T_test
            y_pred = np.array(y_pred)
        
        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_r2_scores.append(r2)
        cv_mse_scores.append(mse)
        cv_mae_scores.append(mae)
        
        print(f"Fold {fold+1}: R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
    
    cv_results = {
        'r2_mean': np.mean(cv_r2_scores),
        'r2_std': np.std(cv_r2_scores),
        'mse_mean': np.mean(cv_mse_scores),
        'mse_std': np.std(cv_mse_scores),
        'mae_mean': np.mean(cv_mae_scores),
        'mae_std': np.std(cv_mae_scores),
        'r2_scores': cv_r2_scores,
        'mse_scores': cv_mse_scores,
        'mae_scores': cv_mae_scores
    }
    
    print(f"\n交叉验证汇总:")
    print(f"  R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print(f"  MSE: {cv_results['mse_mean']:.4f} ± {cv_results['mse_std']:.4f}")
    print(f"  MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
    
    return cv_results


# ============== 数据集分析主函数（GAM版本） ==============
def run_analysis_for_dataset_gam(
    dataset_name: str,
    dataset_pair_names: List[str],
    base_dir: str,
    selected_feature_types: List[str],
    all_trials: List[Dict[str, Any]],
    output_base_dir: str,
    hop_length: int = 256,
    n_time_windows: int = 10,
    window_duration: float = 0.5,
    n_splines: int = 5,
    lam: float = 0.1,
    use_smooth_features: bool = True,
    use_interaction: bool = False,
    aggregation_method: str = 'statistics',
) -> Dict[str, Any]:
    """使用GAM模型分析数据集"""
    print(f"\n\n{'='*80}")
    print(f"GAM分析数据集: {dataset_name}")
    print(f"{'='*80}")

    type_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(type_output_dir, exist_ok=True)

    # 1. 计算选择概率（左侧）
    probabilities = calculate_selection_probability_144(dataset_pair_names, all_trials)

    # 2. 提取std时间序列特征（带缓存）
    print(f"\n提取std时间序列特征...")
    print(f"  音频对数量: {len(dataset_pair_names)}")
    print(f"  选定的特征类型: {selected_feature_types}")
    print(f"  音频文件基础目录: {base_dir}")
    
    # 设置缓存目录
    cache_dir = os.path.join(output_base_dir, 'feature_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存键（基于特征类型和参数）
    import hashlib
    cache_key_params = {
        'feature_types': sorted(selected_feature_types),
        'hop_length': hop_length,
        'n_time_windows': n_time_windows,
        'window_duration': window_duration,
        'sr': 16000
    }
    cache_key_str = json.dumps(cache_key_params, sort_keys=True)
    cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:8]
    
    feature_matrices = []
    valid_indices = []
    failed_count = 0
    cached_count = 0
    extracted_count = 0
    feat_names = None  # 用于存储特征名称（所有样本应该相同）
    
    for idx, pair_name in enumerate(dataset_pair_names):
        if (idx + 1) % 20 == 0:
            print(f"  处理进度: {idx + 1}/{len(dataset_pair_names)} (成功: {len(feature_matrices)}, 缓存: {cached_count}, 提取: {extracted_count}, 失败: {failed_count})")
        
        # 生成缓存文件名（基于pair_name和参数）
        safe_pair_name = pair_name.replace('/', '_').replace('\\', '_').replace('+', '_plus_')
        cache_filename = f"{safe_pair_name}_{cache_key_hash}.npz"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # 尝试从缓存加载
        if os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path, allow_pickle=True)
                feat_matrix = cached_data['feature_matrix']
                cached_feat_names = cached_data['feature_names']
                # 处理numpy数组中的字符串数组
                if isinstance(cached_feat_names, np.ndarray):
                    cached_feat_names = cached_feat_names.tolist()
                if feat_names is None:
                    feat_names = cached_feat_names
                feature_matrices.append(feat_matrix)
                valid_indices.append(idx)
                cached_count += 1
                continue
            except Exception as e:
                print(f"  警告: 加载缓存失败 {pair_name}: {e}，重新提取")
        
        # 如果缓存不存在或加载失败，则提取特征
        try:
            feat_matrix, current_feat_names = extract_std_time_series_for_pair(
                pair_name, base_dir, selected_feature_types,
                hop_length=hop_length, 
                n_time_windows=n_time_windows,
                window_duration=window_duration
            )
            feature_matrices.append(feat_matrix)
            valid_indices.append(idx)
            extracted_count += 1
            
            # 保存特征名称（第一次提取时）
            if feat_names is None:
                feat_names = current_feat_names
            
            # 保存到缓存
            try:
                np.savez_compressed(cache_path, 
                                  feature_matrix=feat_matrix,
                                  feature_names=np.array(current_feat_names, dtype=object))
            except Exception as e:
                print(f"  警告: 保存缓存失败 {pair_name}: {e}")
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只显示前5个失败的详细信息
                import traceback
                print(f"  警告: 无法提取特征 {pair_name}: {e}")
                print(f"    错误详情: {traceback.format_exc()}")
            elif failed_count == 6:
                print(f"  ... 还有更多失败案例，不再详细显示")
            continue
        # except Exception as e:
        #     failed_count += 1
        #     if failed_count <= 5:  # 只显示前5个失败的详细信息
        #         import traceback
        #         print(f"  警告: 无法提取特征 {pair_name}: {e}")
        #         print(f"    错误详情: {traceback.format_exc()}")
        #     elif failed_count == 6:
        #         print(f"  ... 还有更多失败案例，不再详细显示")
        #     continue
        #
    print(f"\n特征提取完成:")
    print(f"  成功: {len(feature_matrices)}/{len(dataset_pair_names)}")
    print(f"  从缓存加载: {cached_count}")
    print(f"  新提取: {extracted_count}")
    print(f"  失败: {failed_count}/{len(dataset_pair_names)}")
    
    if len(feature_matrices) == 0:
        print(f"警告: {dataset_name} 没有有效特征，跳过分析")
        return None

    # 堆叠特征矩阵: (n_samples, n_features, T)
    X_time_series = np.stack(feature_matrices, axis=0)
    y = probabilities[valid_indices]
    
    print(f"时间序列特征矩阵形状: {X_time_series.shape} (样本×特征×时间)")
    print(f"目标变量形状: {y.shape}")

    if len(y) < 10:
        print(f"警告: {dataset_name} 的有效样本过少 ({len(y)} < 10)，跳过分析。")
        return None

    # 3. 拟合GAM模型
    # 注意：GAM可以处理不同尺度的特征，不需要标准化
    print(f"\n拟合GAM模型...")
    gam_model, X_gam, y_gam, time_effect = fit_gam_model(
        X_time_series, y, feat_names, 
        n_splines=n_splines, 
        lam=lam,
        use_smooth_features=use_smooth_features,
        use_interaction=use_interaction,
        aggregation_method=aggregation_method
    )
    
    print(f"\nGAM模型摘要 ({dataset_name}):")
    pseudo_r2 = _safe_get_statistic(gam_model.statistics_, 'pseudo_r2')
    aic = _safe_get_statistic(gam_model.statistics_, 'AIC')
    gcv = _safe_get_statistic(gam_model.statistics_, 'GCV')
    print(f"  R² = {pseudo_r2:.4f}")
    print(f"  AIC = {aic:.4f}")
    print(f"  GCV = {gcv:.4f}")

    # 4. 交叉验证
    cv_results = cross_validate_gam(
        X_time_series, y,
        n_folds=N_FOLDS,
        seed=CV_SEED,
        n_splines=n_splines,
        lam=lam,
        use_smooth_features=use_smooth_features,
        use_interaction=use_interaction,
        aggregation_method=aggregation_method
    )

    # 5. 预测与残差
    if aggregation_method == 'statistics':
        # 对于statistics方法，X_gam已经是聚合后的特征，直接预测
        y_pred = gam_model.predict(X_gam)
    else:
        # 对于weighted_mean和flatten方法，需要处理展平的数据
        y_pred_flat = gam_model.predict(X_gam)
        # 对每个样本，取所有时间点的平均预测值
        n_samples = len(y)
        T = X_time_series.shape[2]
        y_pred = []
        idx = 0
        for i in range(n_samples):
            preds = y_pred_flat[idx:idx+T]
            y_pred.append(np.mean(preds))
            idx += T
        y_pred = np.array(y_pred)
    
    residuals = y - y_pred

    # 6. 可视化
    plot_predicted_vs_actual(y, y_pred, os.path.join(type_output_dir, 'predicted_vs_actual.png'))
    plot_residuals_vs_predicted(
        y_pred,
        residuals,
        os.path.join(type_output_dir, 'residuals_vs_predicted.png'),
    )
    plot_residual_qq(residuals, os.path.join(type_output_dir, 'residual_qq.png'))
    
    # 绘制时间平滑函数
    try:
        n_features = X_time_series.shape[1]
        fig, ax = plt.subplots(figsize=(10, 6))
        XX = gam_model.generate_X_grid(term=n_features, n=100)
        pdep, confi = gam_model.partial_dependence(term=n_features, X=XX, width=0.95)
        ax.plot(XX[:, n_features], pdep)
        ax.fill_between(XX[:, n_features], confi[:, 0], confi[:, 1], alpha=0.3)
        ax.set_xlabel('Time (normalized)', fontsize=12)
        ax.set_ylabel('Partial Dependence', fontsize=12)
        ax.set_title('GAM Time Smoothing Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(type_output_dir, 'time_smooth_function.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"时间平滑函数图已保存: {os.path.join(type_output_dir, 'time_smooth_function.png')}")
    except Exception as e:
        print(f"警告: 无法绘制时间平滑函数: {e}")

    # 7. 保存结果
    predictions_df = pd.DataFrame({
        'pair_name': np.array(dataset_pair_names)[valid_indices],
        'actual_probability': y,
        'predicted_probability': y_pred,
        'residual': residuals,
    })
    predictions_df.to_csv(
        os.path.join(type_output_dir, 'predictions.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    
    # 保存模型统计信息
    stats_summary = pd.DataFrame({
        'metric': ['Pseudo R²', 'AIC', 'GCV', 'CV R² (mean)', 'CV R² (std)', 
                   'CV MSE (mean)', 'CV MSE (std)', 'CV MAE (mean)', 'CV MAE (std)'],
        'value': [
            pseudo_r2,
            aic,
            gcv,
            cv_results['r2_mean'], cv_results['r2_std'],
            cv_results['mse_mean'], cv_results['mse_std'],
            cv_results['mae_mean'], cv_results['mae_std']
        ]
    })
    stats_summary.to_csv(
        os.path.join(type_output_dir, 'model_summary.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    return {
        'exp_type_name': dataset_name,
        'gam_model': gam_model,
        'cv_results': cv_results,
        'n_samples': len(y),
    }


# ============== 主流程 ==============
def main():
    if not PYGM_AVAILABLE:
        print("错误: pygam库未安装，请运行: pip install pygam")
        return
    
    print("=== GAM分析：使用5秒std时间序列，GAM学习时间权重 ===")
    project_root = os.getcwd()

    output_dir = os.path.join(project_root, 'gam_std_timeseries')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载音频对列表
    list_path = os.path.join(project_root, 'pair_list.npy')
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"未找到音频对列表文件: {list_path}")
    pair_names = np.load(list_path, allow_pickle=True)
    print(f"加载了 {len(pair_names)} 个音频对")

    # 2. 选定特征（与线性脚本相同）
    selected_indices_0based, selected_indices_1based, selected_feature_names = extract_significant_features_from_table()
    print("\n选定特征（按照显著性表格）：")
    for i, (idx, name) in enumerate(zip(selected_indices_1based, selected_feature_names), 1):
        print(f"  {i}. {name} (feature_idx={idx})")
    
    # 将18个特征索引（mean/std）映射到特征类型名称
    # 映射规则：基础特征索引 = (feature_idx - 1) // 2
    feature_idx_to_type = {
        1: 'spectral_centroid',  # Spectral Centroid (mean)
        2: 'spectral_centroid',  # Spectral Centroid (std)
        3: 'f0',                 # F0 (mean)
        4: 'f0',                 # F0 (std)
        5: 'harmonicity',        # Harmonicity (mean)
        6: 'harmonicity',        # Harmonicity (std)
        7: 'spec_mod_centroid',  # Spectral Modulation Centroid (mean)
        8: 'spec_mod_centroid',  # Spectral Modulation Centroid (std)
        9: 'spectral_irregularity',  # Spectral Irregularity (mean)
        10: 'spectral_irregularity', # Spectral Irregularity (std)
        11: 'loudness',          # Loudness (mean)
        12: 'loudness',          # Loudness (std)
        13: 'spectral_flatness', # Spectral Flatness (mean)
        14: 'spectral_flatness', # Spectral Flatness (std)
        15: 'spectral_bandwidth', # Spectral Bandwidth (mean)
        16: 'spectral_bandwidth', # Spectral Bandwidth (std)
        17: 'temporal_mod',      # Temporal Modulation (mean)
        18: 'temporal_mod',     # Temporal Modulation (std)
    }
    
    selected_feature_types = set()
    for feat_idx_1based in selected_indices_1based:
        if feat_idx_1based in feature_idx_to_type:
            selected_feature_types.add(feature_idx_to_type[feat_idx_1based])
    
    # 排除 f0 和 harmonicity 特征（短时间提取意义不大）
    # F0 需要较长的音频才能稳定提取，5秒对于很多非周期性声音来说太短
    # Harmonicity (HPSS) 也需要足够的频率分辨率，短时间音频可能不够稳定
    excluded_features = []
    if 'f0' in selected_feature_types:
        selected_feature_types.remove('f0')
        excluded_features.append('f0')
    if 'harmonicity' in selected_feature_types:
        selected_feature_types.remove('harmonicity')
        excluded_features.append('harmonicity')
    
    if excluded_features:
        print(f"\n注意: 已排除以下特征（短时间提取意义不大）: {', '.join(excluded_features)}")
    
    selected_feature_types = sorted(list(selected_feature_types))
    
    print(f"\n映射到特征类型（用于提取std时间序列）：")
    for i, feat_type in enumerate(selected_feature_types, 1):
        print(f"  {i}. {feat_type}")
    
    # 3. 设置音频文件基础目录（需要根据实际情况修改）
    # 实际文件在: D:\D\research\audioset下载\clap_select
    # 当前工作目录是: D:\D\research\audioset下载\download_audioset-master\download_audioset-master
    # 需要向上两级目录
    base_dir = os.path.join(project_root, '..', '..', 'clap_select')
    base_dir = os.path.normpath(base_dir)  # 规范化路径，处理 .. 符号
    
    # 如果上面的路径不存在，尝试其他可能的路径
    if not os.path.exists(base_dir):
        possible_dirs = [
            os.path.join(project_root, '..', '..', 'clap_select'),  # 向上两级
            os.path.join(project_root, '..', 'clap_select'),  # 向上一级
            os.path.join(project_root, 'clap_select'),  # 当前目录
            os.path.join(project_root, '音频配对合成_新版', '分组音频', 'folder1'),
            os.path.join(project_root, '音频配对合成_新版', '分组音频', 'folder2'),
            r'D:\D\research\audioset下载\clap_select',  # 绝对路径
        ]
        for pd in possible_dirs:
            pd_normalized = os.path.normpath(pd)
            if os.path.exists(pd_normalized):
                base_dir = pd_normalized
                break
        else:
            print(f"警告: 未找到音频文件目录，请检查base_dir设置")
            print(f"  当前base_dir: {base_dir}")
            print(f"  尝试的路径:")
            for pd in possible_dirs:
                print(f"    - {os.path.normpath(pd)}")
    
    print(f"\n音频文件基础目录: {base_dir}")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"音频文件目录不存在: {base_dir}")

    # 3. 合并音频对
    merged_results = merge_balanced_pairs_by_type(pair_names)

    # 4. 加载人类决策
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt",
    ]
    csv_files = [
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
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
    ] * 9

    all_trials = load_all_human_decisions(txt_files, csv_files, mapping_files)

    # 5. 定义要分析的数据集
    datasets_to_analyze = [
        {
            'name': 'all',
            'types': ['nn_main', 'nn_sub', 'main', 'sub'],
            'description': '所有类型合并'
        }
    ]

    # 6. 对每个数据集进行分析（使用GAM）
    all_results = []

    for dataset_config in datasets_to_analyze:
        # 合并指定类型的数据
        dataset_pair_names = []

        for exp_type in dataset_config['types']:
            if exp_type in merged_results:
                dataset_pair_names.extend(merged_results[exp_type]['pair_names'])

        if len(dataset_pair_names) == 0:
            print(f"\n警告: {dataset_config['name']} 没有数据，跳过")
            continue

        print(f"\n{dataset_config['description']}: {len(dataset_pair_names)} 个音频对")

        # 执行GAM分析
        result = run_analysis_for_dataset_gam(
            dataset_config['name'],
            dataset_pair_names,
            base_dir,
            selected_feature_types,
            all_trials,
            output_dir,
            hop_length=256,
            n_time_windows=10,
            window_duration=0.5,
            n_splines=8,  # 增加样条数量，提高模型灵活性
            lam=0.1,  # 降低正则化，允许更灵活的非线性关系
            use_smooth_features=True,  # 使用平滑项，允许非线性特征关系
            use_interaction=False,  # 暂时不启用交互项，避免过拟合
            aggregation_method='statistics',  # 使用统计量聚合方法（推荐，效果更好）
        )

        if result is not None:
            all_results.append(result)

    # 7. 汇总所有结果
    if len(all_results) > 0:
        print(f"\n\n{'='*80}")
        print("所有数据集GAM结果汇总")
        print(f"{'='*80}")

        summary_data = []
        for result in all_results:
            gam_model = result['gam_model']
            summary_data.append({
                'dataset': result['exp_type_name'],
                'n_samples': result['n_samples'],
                'Pseudo R²': _safe_get_statistic(gam_model.statistics_, 'pseudo_r2'),
                'AIC': _safe_get_statistic(gam_model.statistics_, 'AIC'),
                'GCV': _safe_get_statistic(gam_model.statistics_, 'GCV'),
                'CV R² (mean)': result['cv_results']['r2_mean'],
                'CV R² (std)': result['cv_results']['r2_std'],
                'CV MSE (mean)': result['cv_results']['mse_mean'],
                'CV MSE (std)': result['cv_results']['mse_std'],
                'CV MAE (mean)': result['cv_results']['mae_mean'],
                'CV MAE (std)': result['cv_results']['mae_std']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'all_datasets_gam_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总结果已保存: {summary_path}")
        print("\nGAM汇总表格:")
        print(summary_df.to_string(index=False))

    print("\n=== GAM分析完成 ===")


if __name__ == '__main__':
    main()


