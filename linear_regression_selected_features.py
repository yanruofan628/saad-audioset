#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用选定特征建立线性回归模型预测选择概率
支持在主函数中选择特征，或从显著特征列表中自动提取
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, normaltest
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

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

    print(
        f"已加载Wav2Vec特征 {feature_key}: 有效样本 {valid_mask.sum()}/{len(values)}, 平均值 {mean_val:.4f}, 标准差 {std_val:.4f}")

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
        audio_id = part[second_last + 1:]
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
        if line.strip().startswith('Level: 3') and i + 1 < len(lines) and '*** LogFrame Start ***' in lines[i + 1]:
            trial_data = {}
            for j in range(i + 1, min(i + 20, len(lines))):
                current_line = lines[j].strip()
                if 'wavfile:' in current_line:
                    trial_data['wavfile'] = current_line.split('wavfile:')[1].strip()
                elif 'ImageDisplay1.RESP:' in current_line:
                    trial_data['response'] = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
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


# ============== 新的回归分析函数 ==============

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


def fit_ols_model(X, y, feature_names):
    """
    使用statsmodels拟合OLS模型
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)
        feature_names: 特征名称列表
    返回:
        model: 拟合的OLS模型
    """
    # 转换为DataFrame并设置列名，确保输出使用特征名称
    X_df = pd.DataFrame(X, columns=feature_names)

    # 添加截距项（const列会自动添加）
    X_with_const = sm.add_constant(X_df)

    # 拟合模型
    model = sm.OLS(y, X_with_const).fit()

    return model


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

        # 训练模型
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        model = sm.OLS(y_train, X_train_const).fit()

        # 预测
        y_pred = model.predict(X_test_const)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fold_dir = os.path.join(save_dir, f'fold_{fold + 1}')
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

        print(f"Fold {fold + 1}: R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

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
        for j in range(i + 1, len(feature_names)):
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
        feature_name = feature_names[i - 1]
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
    提取模型统计信息
    参数:
        model: statsmodels OLS模型
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
    if len(feature_names_from_model) == len(feature_names) and all(
            n in feature_names for n in feature_names_from_model):
        actual_feature_names = feature_names_from_model
    else:
        # 如果模型中的名称与传入的不匹配，使用传入的feature_names（按顺序）
        actual_feature_names = feature_names if len(feature_names) == len(feature_indices) else feature_names_from_model

    stats_dict = {
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic,
        'fvalue': model.fvalue,
        'f_pvalue': model.f_pvalue,
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
    """绘制系数棒棒糖图（仅细茎线+圆点，不绘制CI）"""
    # 统一字体与尺寸（高:宽=8.41:15.19 cm）
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    cm_to_inch = 1 / 2.54
    fig_w = 15.19 * cm_to_inch
    fig_h = 8.41 * cm_to_inch
    font_size = 18
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

    # 按系数大小排序，保证从下到上更易读
    df = df.sort_values('coefficient').reset_index(drop=True)

    # 创建图形
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y_positions = np.arange(len(df))

    # 颜色配置：显著项高亮，非显著项弱化
    sig_color = '#9555A3'
    ns_color = '#B8B8B8'
    stem_color = '#D8D8D8'

    # 绘制棒棒糖（0 -> 系数）+ 点估计（不绘制CI）
    for i, (_, data) in enumerate(df.iterrows()):
        is_sig = data['pvalue'] < 0.05
        point_color = sig_color if is_sig else ns_color

        # lollipop stem: 从0连到系数
        ax.plot([0, data['coefficient']], [i, i],
                color=stem_color, linewidth=1.8, alpha=0.95, zorder=1)

        # 点估计
        ax.scatter(data['coefficient'], i, s=72, color=point_color,
                   edgecolor='white', linewidth=0.8, zorder=3)

        # 添加显著性标记
        sig_mark = '***' if data['pvalue'] < 0.001 else '**' if data['pvalue'] < 0.01 else '*' if data['pvalue'] < 0.05 else ''
        if sig_mark:
            ax.text(data['coefficient'], i, f' {sig_mark}', ha='left', va='center',
                    fontsize=font_size, color=point_color)

    # 零线
    ax.axvline(x=0, color='#666666', linestyle='--', linewidth=1.2, alpha=0.9, zorder=0)

    ax.set_yticks(y_positions)
    ytick_font_size = 8.5
    ax.set_yticklabels(df['feature'], fontsize=ytick_font_size)
    ax.set_xlabel('Coefficient', fontsize=font_size)
    ax.set_title('')
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=ytick_font_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(ytick_font_size)
        tick.set_fontname('Arial')
    ax.grid(True, alpha=0.22, axis='x', linestyle='--')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    svg_path = os.path.splitext(output_path)[0] + '.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"系数棒棒糖图已保存: {output_path}")
    print(f"系数棒棒糖图SVG已保存: {svg_path}")


def save_model_results(stats_dict, cv_results, output_dir):
    """保存模型结果到CSV文件"""
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
        'metric': ['R²', 'Adjusted R²', 'AIC', 'BIC', 'F-value', 'F p-value',
                   'N observations', 'DF model', 'DF residual',
                   'Intercept', 'Intercept p-value',
                   'CV R² (mean)', 'CV R² (std)',
                   'CV MSE (mean)', 'CV MSE (std)',
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

    print(f"\n模型结果已保存到: {output_dir}")


def run_analysis_for_dataset(feature_diff, pair_names, all_trials, selected_feature_names,
                             fitting_feature_names, fitting_indices, exp_type_name, output_base_dir):
    """
    对指定数据集执行完整的回归分析

    参数:
        feature_diff: 特征差矩阵 (n_features, n_pairs)
        pair_names: 音频对名称列表
        all_trials: 所有trial数据
        selected_feature_names: 所有选定特征名称（用于相关性分析）
        fitting_feature_names: 用于拟合的特征名称
        fitting_indices: 用于拟合的特征索引（在selected_feature_names中）
        exp_type_name: 实验类型名称（用于输出目录和文件名）
        output_base_dir: 输出基础目录
    """
    print(f"\n\n{'=' * 80}")
    print(f"分析数据集: {exp_type_name}")
    print(f"{'=' * 80}")

    # 创建该数据集的输出目录
    type_output_dir = os.path.join(output_base_dir, exp_type_name)
    os.makedirs(type_output_dir, exist_ok=True)

    # 1. 计算选择概率
    print(f"\n步骤1: 计算选择概率 ({exp_type_name})")
    probabilities = calculate_selection_probability_144(pair_names, all_trials)

    # 2. 准备回归数据
    print(f"\n步骤2: 准备回归数据 ({exp_type_name})")
    X, y, valid_mask = prepare_regression_data(feature_diff, probabilities)

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

    # 5. 拟合OLS模型
    print(f"\n步骤5: 拟合OLS模型 ({exp_type_name})")
    model = fit_ols_model(X_fitting, y, fitting_feature_names)

    # 打印模型摘要
    print(f"\n{'=' * 60}")
    print(f"模型摘要 ({exp_type_name})")
    print(f"{'=' * 60}")
    print(model.summary())

    # 6. 提取模型统计信息
    print(f"\n步骤6: 提取模型统计信息 ({exp_type_name})")
    stats_dict = extract_model_statistics(model, fitting_feature_names)

    print(f"\n模型拟合统计 ({exp_type_name}):")
    print(f"  R² = {stats_dict['rsquared']:.4f}")
    print(f"  调整R² = {stats_dict['rsquared_adj']:.4f}")
    print(f"  AIC = {stats_dict['aic']:.4f}")
    print(f"  BIC = {stats_dict['bic']:.4f}")
    print(f"  F值 = {stats_dict['fvalue']:.4f}, p = {stats_dict['f_pvalue']:.4f}")

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


def main():
    print("=== 线性回归分析：使用选定声学特征预测选择概率 ===")
    project_root = os.getcwd()

    # ============== 特征选择配置 ==============
    # 选项1: 使用自动提取的显著特征
    USE_AUTO_SIGNIFICANT_FEATURES = False

    # 选项2: 手动指定特征（当 USE_AUTO_SIGNIFICANT_FEATURES = False 时使用）
    # feature_idx 从1开始计数（对应前18个声学特征）
    # 十一个特征：Spectral Centroid (std), F0 (mean/std), Harmonicity (mean/std),
    #            Spectral Modulation Centroid (mean/std), Spectral Irregularity (mean/std), 
    #            Loudness (std), Temporal Modulation (mean)
    MANUAL_SELECTED_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 17]  # 十一个特征（包含F0和谐波度）

    # 选项3: 是否追加 Wav2Vec2 特征（默认不使用）
    USE_WAV2VEC_FEATURE = False
    # ===========================================

    # 设置输出目录（统一到 linear_5fold）
    output_dir = os.path.join(project_root, 'linear_5fold')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载特征数据
    print("\n步骤1: 加载特征数据")
    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)

    # 完整的特征名称列表（前18个声学特征，从1开始计数）
    feature_names_full = [
        'Spectral Centroid (mean)',  # 1 (索引0)
        'Spectral Centroid (std)',  # 2 (索引1)
        'F0 (mean)',  # 3 (索引2)
        'F0 (std)',  # 4 (索引3)
        'Harmonicity (mean)',  # 5 (索引4)
        'Harmonicity (std)',  # 6 (索引5)
        'Spectral Modulation Centroid (mean)',  # 7 (索引6)
        'Spectral Modulation Centroid (std)',  # 8 (索引7)
        'Spectral Irregularity (mean)',  # 9 (索引8)
        'Spectral Irregularity (std)',  # 10 (索引9)
        'Loudness (mean)',  # 11 (索引10)
        'Loudness (std)',  # 12 (索引11)
        'Spectral Flatness (mean)',  # 13 (索引12)
        'Spectral Flatness (std)',  # 14 (索引13)
        'Spectral Bandwidth (mean)',  # 15 (索引14)
        'Spectral Bandwidth (std)',  # 16 (索引15)
        'Temporal Modulation (mean)',  # 17 (索引16)
        'Temporal Modulation (std)',  # 18 (索引17)
    ]

    # 使用前18个声学特征（或根据需要调整）
    n_acoustic_features = 18
    acoustic_features = diff_zscore[:n_acoustic_features, :]
    print(f"使用前{n_acoustic_features}个声学特征: {acoustic_features.shape}")

    # 2. 选择特征
    if USE_AUTO_SIGNIFICANT_FEATURES:
        print("\n步骤2: 自动提取显著特征（significant=True）")
        selected_indices_0based, selected_indices_1based, selected_feature_names = extract_significant_features_from_table()
        print(f"\n从相关性结果表格中提取的显著特征（已去重）:")
        for i, (idx_1based, name) in enumerate(zip(selected_indices_1based, selected_feature_names)):
            print(f"  {i + 1}. {name} (feature_idx={idx_1based}, 数组索引={selected_indices_0based[i]})")
    else:
        print("\n步骤2: 使用手动指定的特征")
        # 将手动指定的索引（从1开始）转换为从0开始的索引
        selected_indices_0based = [idx - 1 for idx in MANUAL_SELECTED_INDICES]
        selected_indices_1based = MANUAL_SELECTED_INDICES
        selected_feature_names = [feature_names_full[idx - 1] for idx in MANUAL_SELECTED_INDICES]
        print(f"\n手动选定的特征:")
        for i, (idx_1based, name) in enumerate(zip(selected_indices_1based, selected_feature_names)):
            print(f"  {i + 1}. {name} (feature_idx={idx_1based}, 数组索引={selected_indices_0based[i]})")

    # 确保特征名称为可修改列表
    selected_feature_names = list(selected_feature_names)

    # 提取选定的声学特征
    selected_features = extract_selected_features(acoustic_features, selected_indices_0based)
    print(f"\n选定声学特征矩阵形状: {selected_features.shape}")

    # 额外加载Wav2Vec2 cosine_sum_diff特征（可选）
    if USE_WAV2VEC_FEATURE:
        wav2vec_feature_info = load_wav2vec_feature_zscore(project_root, pair_names, feature_key='cosine_sum_diff')
        wav2vec_feature_name = 'Wav2Vec2 Cosine Sum Diff (z-score)'
        if wav2vec_feature_info is not None:
            wav2vec_zscore = wav2vec_feature_info.get('zscore')
            if wav2vec_zscore is not None:
                selected_features = np.vstack([selected_features, wav2vec_zscore.reshape(1, -1)])
                selected_feature_names.append(wav2vec_feature_name)
                print(f"已追加Wav2Vec特征: {wav2vec_feature_name}")
                print(f"新的特征矩阵形状: {selected_features.shape}")
            else:
                print("警告: 未能获取Wav2Vec特征的z-score，跳过追加")
        else:
            print("警告: 未加载到任何Wav2Vec特征，保持原有声学特征")
    else:
        print("已禁用Wav2Vec特征追加（USE_WAV2VEC_FEATURE=False）")

    # 用于拟合的特征（使用所有可用特征）
    fitting_indices = list(range(len(selected_feature_names)))
    fitting_feature_names = selected_feature_names.copy()
    print(f"\n用于拟合的特征（共{len(fitting_feature_names)}个）:")
    for i, name in enumerate(fitting_feature_names):
        print(f"  {i + 1}. {name}")

    # 3. 合并所有实验类型的数据
    print("\n步骤3: 合并所有实验类型的数据")
    merged_results = merge_balanced_pairs_by_type(pair_names)

    # 4. 加载人类决策数据
    print("\n步骤4: 加载人类决策数据")
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
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
        r"D:\D\research\audioset下载\experiment_output2\Liu Yaorui2.csv"
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

    # 6. 对每个数据集进行分析
    all_results = []

    for dataset_config in datasets_to_analyze:
        # 合并指定类型的数据
        dataset_indices = []
        dataset_pair_names = []

        for exp_type in dataset_config['types']:
            if exp_type in merged_results:
                dataset_indices.extend(merged_results[exp_type]['indices'])
                dataset_pair_names.extend(merged_results[exp_type]['pair_names'])

        if len(dataset_indices) == 0:
            print(f"\n警告: {dataset_config['name']} 没有数据，跳过")
            continue

        # 提取该数据集的特征差
        dataset_feature_diff = selected_features[:, dataset_indices]
        print(f"\n{dataset_config['description']}: {len(dataset_pair_names)} 个音频对")

        # 执行分析
        result = run_analysis_for_dataset(
            dataset_feature_diff,
            dataset_pair_names,
            all_trials,
            selected_feature_names,
            fitting_feature_names,
            fitting_indices,
            dataset_config['name'],
            output_dir
        )

        if result is not None:
            all_results.append(result)

    # 7. 汇总所有结果
    if len(all_results) > 0:
        print(f"\n\n{'=' * 80}")
        print("所有数据集结果汇总")
        print(f"{'=' * 80}")

        summary_data = []
        for result in all_results:
            summary_data.append({
                'dataset': result['exp_type_name'],
                'n_samples': result['n_samples'],
                'R²': result['stats_dict']['rsquared'],
                'Adjusted R²': result['stats_dict']['rsquared_adj'],
                'AIC': result['stats_dict']['aic'],
                'BIC': result['stats_dict']['bic'],
                'F-value': result['stats_dict']['fvalue'],
                'F p-value': result['stats_dict']['f_pvalue'],
                'CV R² (mean)': result['cv_results']['r2_mean'],
                'CV R² (std)': result['cv_results']['r2_std']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'all_datasets_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总结果已保存: {summary_path}")
        print("\n汇总表格:")
        print(summary_df.to_string(index=False))

    print("\n=== 所有分析完成 ===")


if __name__ == '__main__':
    main()

