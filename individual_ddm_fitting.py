#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDM个人数据拟合：使用Drift Diffusion Model拟合个人选择和反应时
- 使用全部288个试次（带位置编码）
- 特征选择：去掉F0和谐波度，时间调制用mean，其他用std
- 拟合DDM参数：漂移率v=f(特征)，阈值a，起始点z，非决策时间t0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, norm
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要的函数
from spatiotemporal_glm_model import (
    parse_experiment_data_txt,
    parse_audio_mapping,
    parse_pair_name
)
from linear_regression_selected_features import (
    load_or_extract_features,
    build_consistent_folds
)

CV_SEED = 42
N_FOLDS = 5


def load_individual_trials(txt_files, mapping_files):
    """加载个人数据的两个txt文件并合并"""
    print("\n=== 加载个人数据 ===")
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
                    'response': trial['response'],  # 1=选左, 2=选右
                    'reaction_time': trial.get('reaction_time', None)
                }
                all_trials.append(trial_data)
            else:
                print(f"  警告: 未找到映射 {filename}")
    
    print(f"\n总共加载 {len(all_trials)} 个试次")
    rt_count = sum(1 for t in all_trials if t['reaction_time'] is not None)
    print(f"  有RT数据的试次: {rt_count}/{len(all_trials)}")
    
    return all_trials


def select_ddm_features():
    """
    选择用于DDM的特征
    去掉F0和谐波度，时间调制用mean，其他用std
    
    返回:
    -----
    feature_indices_0based : list
        特征索引（从0开始，对应18个特征）
    feature_names : list
        特征名称列表
    """
    # 18个特征的完整列表（从linear_regression_selected_features.py）
    feature_names_full = [
        'Spectral Centroid (mean)',      # 0
        'Spectral Centroid (std)',       # 1 ✓
        'F0 (mean)',                      # 2 ✗ 去掉
        'F0 (std)',                       # 3
        'Harmonicity (mean)',             # 4
        'Harmonicity (std)',              # 5 ✗ 去掉
        'Spectral Modulation Centroid (mean)',  # 6
        'Spectral Modulation Centroid (std)',   # 7 ✓
        'Spectral Irregularity (mean)',  # 8
        'Spectral Irregularity (std)',   # 9 ✓
        'Loudness (mean)',               # 10
        'Loudness (std)',                # 11 ✓
        'Spectral Flatness (mean)',      # 12
        'Spectral Flatness (std)',       # 13
        'Spectral Bandwidth (mean)',      # 14
        'Spectral Bandwidth (std)',       # 15
        'Temporal Modulation (mean)',    # 16 ✓
        'Temporal Modulation (std)',    # 17
    ]
    
    # 选择的特征索引（从0开始）
    selected_indices_0based = [
        1,   # Spectral Centroid (std)
        7,   # Spectral Modulation Centroid (std)
        9,   # Spectral Irregularity (std)
        11,  # Loudness (std)
        16,  # Temporal Modulation (mean)
    ]
    
    selected_feature_names = [feature_names_full[i] for i in selected_indices_0based]
    
    print(f"\n选择的DDM特征（共{len(selected_feature_names)}个）:")
    for i, name in enumerate(selected_feature_names):
        print(f"  {i+1}. {name}")
    
    return selected_indices_0based, selected_feature_names


def check_consistent_choices(all_trials, pair_names):
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
    print("\n=== 检查左右平衡一致性（只保留一致的选择） ===")
    
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


def extract_features_for_ddm(consistent_trials, diff_zscore, pair_names, feature_indices_0based):
    """
    为DDM提取特征（只使用一致的选择，不带位置编码）
    
    使用已有的特征数据（pair_feature_diff_zscore.npy）
    只使用AB版本的特征差值（left - right）
    
    Parameters:
    -----------
    consistent_trials : list
        一致的试次数据（已经过check_consistent_choices处理）
    diff_zscore : np.ndarray, shape (18, n_pairs)
        已有的特征差值数据（z-score归一化）
    pair_names : list
        原始pair_list（288个），用于查找索引
    feature_indices_0based : list
        特征索引（从0开始，对应18个特征中的哪些）
        
    Returns:
    --------
    X : np.ndarray, shape (n_trials, n_features)
        特征矩阵（left - right的特征差值）
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    y_rt : np.ndarray, shape (n_trials,)
        反应时（秒）
    trial_info : list
        试次信息（用于调试）
    """
    print("\n=== 提取DDM特征（一致选择，无位置编码） ===")
    
    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    
    X_list = []
    y_choice_list = []
    y_rt_list = []
    trial_info_list = []
    
    valid_count = 0
    error_count = 0
    
    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']
        choice = trial['choice']  # 1=选左，2=选右
        rt = trial.get('reaction_time', None)
        
        try:
            # 查找特征索引（pair_name就是AB版本）
            if pair_name not in pair_to_idx:
                error_count += 1
                continue
            
            idx = pair_to_idx[pair_name]
            
            # 提取特征差值（使用选定的特征）
            # diff_zscore存储的是 left - right（AB版本）
            feat_diff = diff_zscore[feature_indices_0based, idx]  # (n_selected_features,)
            
            # 检查NaN
            if np.isnan(feat_diff).any():
                error_count += 1
                continue
            
            X_list.append(feat_diff)
            y_choice_list.append(choice)
            y_rt_list.append(rt if rt is not None else np.nan)
            trial_info_list.append({
                'pair_name': pair_name,
                'choice': choice
            })
            
            valid_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(consistent_trials)}, 有效: {valid_count}, 错误: {error_count}")
        
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  警告: 提取特征失败 {pair_name}: {e}")
            continue
    
    X = np.array(X_list)
    y_choice = np.array(y_choice_list)
    y_rt = np.array(y_rt_list)
    
    print(f"\n特征提取完成:")
    print(f"  有效试次: {len(X)}/{len(consistent_trials)}")
    print(f"  错误试次: {error_count}")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  选左: {np.sum(y_choice == 1)} ({np.mean(y_choice == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(y_choice == 2)} ({np.mean(y_choice == 2)*100:.1f}%)")
    print(f"  有RT数据: {np.sum(~np.isnan(y_rt))} ({np.sum(~np.isnan(y_rt))/len(y_rt)*100:.1f}%)")
    
    return X, y_choice, y_rt, trial_info_list


def diagnose_data_quality(X, y_choice, y_rt, feature_names):
    """
    诊断数据质量，帮助理解为什么DDM拟合效果差
    """
    print("\n=== 数据质量诊断 ===")
    
    # 1. 数据量检查
    print(f"\n1. 数据量:")
    print(f"   试次数: {len(X)}")
    if len(X) < 50:
        print(f"   ⚠️ 警告: 数据量较少（<50），DDM拟合可能不稳定")
    elif len(X) < 100:
        print(f"   ⚠️ 注意: 数据量适中（50-100），建议至少100个试次")
    else:
        print(f"   ✓ 数据量充足")
    
    # 2. 选择分布检查
    print(f"\n2. 选择分布:")
    prob_left = np.mean(y_choice == 1)
    prob_right = np.mean(y_choice == 2)
    print(f"   选左: {np.sum(y_choice == 1)} ({prob_left*100:.1f}%)")
    print(f"   选右: {np.sum(y_choice == 2)} ({prob_right*100:.1f}%)")
    if abs(prob_left - 0.5) > 0.3:
        print(f"   ⚠️ 警告: 选择分布严重不平衡，可能影响拟合")
    
    # 3. RT分布检查
    valid_rt = y_rt[~np.isnan(y_rt)]
    print(f"\n3. RT分布:")
    print(f"   均值: {np.mean(valid_rt):.3f}秒")
    print(f"   标准差: {np.std(valid_rt):.3f}秒")
    print(f"   范围: [{np.min(valid_rt):.3f}, {np.max(valid_rt):.3f}]秒")
    if np.std(valid_rt) < 0.1:
        print(f"   ⚠️ 警告: RT变异性很小，可能难以拟合")
    
    # 4. 特征与选择的关系
    print(f"\n4. 特征与选择的关系（逻辑回归）:")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_choice_binary = (y_choice == 2).astype(int)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_std_norm = (X - X_mean) / X_std
    
    lr_model = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    lr_model.fit(X_std_norm, y_choice_binary)
    
    lr_pred = lr_model.predict(X_std_norm)
    lr_proba = lr_model.predict_proba(X_std_norm)[:, 1]
    lr_acc = accuracy_score(y_choice_binary, lr_pred)
    try:
        lr_auc = roc_auc_score(y_choice_binary, lr_proba)
    except:
        lr_auc = np.nan
    
    print(f"   逻辑回归准确率: {lr_acc:.4f}")
    print(f"   逻辑回归AUC: {lr_auc:.4f}")
    
    if lr_acc < 0.55:
        print(f"   ⚠️ 警告: 逻辑回归准确率很低，说明特征与选择的关系很弱")
        print(f"   可能原因:")
        print(f"     - 特征本身预测能力弱")
        print(f"     - 数据量太小，统计功效不足")
        print(f"     - 选择可能主要由其他因素决定（如随机性）")
    
    # 5. 特征系数显著性（简化检查）
    print(f"\n5. 特征系数（逻辑回归）:")
    print(f"   截距: {lr_model.intercept_[0]:.4f}")
    for i, name in enumerate(feature_names):
        coef = lr_model.coef_[0][i]
        print(f"   {name}: {coef:.4f}")
        if abs(coef) < 0.1:
            print(f"      ⚠️ 系数很小，可能不重要")
    
    # 6. RT与选择的关系
    print(f"\n6. RT与选择的关系:")
    rt_left = valid_rt[y_choice[~np.isnan(y_rt)] == 1]
    rt_right = valid_rt[y_choice[~np.isnan(y_rt)] == 2]
    if len(rt_left) > 0 and len(rt_right) > 0:
        print(f"   选左的平均RT: {np.mean(rt_left):.3f}秒")
        print(f"   选右的平均RT: {np.mean(rt_right):.3f}秒")
        rt_diff = abs(np.mean(rt_left) - np.mean(rt_right))
        if rt_diff < 0.05:
            print(f"   ⚠️ 警告: 两种选择的RT差异很小（{rt_diff:.3f}秒），可能难以区分")
    
    # 7. 总结和建议
    print(f"\n7. 诊断总结:")
    issues = []
    if len(X) < 50:
        issues.append("数据量太少")
    if lr_acc < 0.55:
        issues.append("特征预测能力弱")
    if np.std(valid_rt) < 0.1:
        issues.append("RT变异性小")
    
    if len(issues) > 0:
        print(f"   发现的问题:")
        for issue in issues:
            print(f"     - {issue}")
        print(f"\n   建议:")
        if len(X) < 50:
            print(f"     - 考虑使用全部288个试次（带位置编码）")
            print(f"     - 或者收集更多数据")
        if lr_acc < 0.55:
            print(f"     - 检查特征选择是否合理")
            print(f"     - 考虑使用其他特征组合")
            print(f"     - 或者特征与选择的关系确实很弱（这是正常的）")
        if np.std(valid_rt) < 0.1:
            print(f"     - RT变异性小可能说明决策过程相对稳定")
    else:
        print(f"   ✓ 数据质量良好")


def ddm_likelihood_single(v, a, z, t0, choice, rt, max_rt=5.0):
    """
    计算单个试次的DDM似然值（使用Wiener过程的首次通过时间分布）
    
    参数:
    -----
    v : float
        漂移率
    a : float
        阈值（边界）
    z : float
        起始点（0到a之间）
    t0 : float
        非决策时间
    choice : int
        选择（1=选左/下边界，2=选右/上边界）
    rt : float
        反应时（秒）
    max_rt : float
        最大反应时（用于截断）
        
    返回:
    -----
    log_likelihood : float
        对数似然值
    """
    # 确保rt在合理范围内
    if rt < t0 or rt > max_rt:
        return -np.inf
    
    # 决策时间（去除非决策时间）
    t = rt - t0
    
    if t <= 0:
        return -np.inf
    
    # 确定选择的边界
    # choice=1表示选左（下边界=0），choice=2表示选右（上边界=a）
    # 起始点：z（0到a之间）
    
    # 使用Wiener过程的首次通过时间分布的解析公式
    # 对于上边界（选右）：
    if choice == 2:
        # 到达上边界a的概率密度
        # 使用Wiener过程的首次通过时间分布（Inverse Gaussian分布）
        # f(t) = (a-z) / sqrt(2*pi*t^3) * exp(-((a-z) - v*t)^2 / (2*t))
        
        # 避免数值问题
        if abs(v) < 1e-10:  # 无漂移情况
            # 无漂移的首次通过时间分布
            if t < 1e-10:
                return -np.inf
            prob = (a - z) / np.sqrt(2 * np.pi * t**3) * \
                   np.exp(-(a - z)**2 / (2 * t))
        else:
            # 有漂移情况
            if t < 1e-10:
                return -np.inf
            # 使用更稳定的公式
            diff = a - z
            prob = diff / np.sqrt(2 * np.pi * t**3) * \
                   np.exp(-(diff - v * t)**2 / (2 * t))
    else:  # choice == 1，选左（下边界=0）
        # 到达下边界0的概率密度
        if abs(v) < 1e-10:  # 无漂移情况
            if t < 1e-10:
                return -np.inf
            prob = z / np.sqrt(2 * np.pi * t**3) * \
                   np.exp(-z**2 / (2 * t))
        else:
            # 有漂移情况
            if t < 1e-10:
                return -np.inf
            prob = z / np.sqrt(2 * np.pi * t**3) * \
                   np.exp(-(z - v * t)**2 / (2 * t))
    
    # 避免数值下溢
    if prob <= 0 or not np.isfinite(prob):
        return -np.inf
    
    log_prob = np.log(prob)
    
    # 确保结果是有限的
    if not np.isfinite(log_prob):
        return -np.inf
    
    return log_prob


def ddm_likelihood(params, X, y_choice, y_rt, valid_mask):
    """
    计算所有试次的DDM对数似然值
    
    参数:
    -----
    params : np.ndarray
        参数向量：[beta_0, beta_1, ..., beta_n, a, z, t0]
        beta_0到beta_n是漂移率的系数
    X : np.ndarray, shape (n_trials, n_features)
        特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    y_rt : np.ndarray, shape (n_trials,)
        反应时（秒）
    valid_mask : np.ndarray
        有效RT数据的掩码
        
    返回:
    -----
    neg_log_likelihood : float
        负对数似然（用于最小化）
    """
    n_features = X.shape[1]
    
    # 解析参数
    beta = params[:n_features+1]  # 包括截距
    a = params[n_features+1]      # 阈值
    z_ratio = params[n_features+2]  # 起始点比例（0到1之间）
    t0 = params[n_features+3]     # 非决策时间
    
    # 确保参数在合理范围内
    if a <= 0 or t0 < 0 or z_ratio < 0 or z_ratio > 1:
        return 1e10  # 返回很大的值
    
    z = z_ratio * a  # 起始点
    
    # 计算每个试次的漂移率
    X_with_const = np.column_stack([np.ones(len(X)), X])  # 添加截距列
    v = np.dot(X_with_const, beta)  # (n_trials,)
    
    # 计算对数似然
    log_likelihood = 0.0
    
    for i in range(len(X)):
        if valid_mask[i]:
            choice = y_choice[i]
            rt = y_rt[i]
            v_i = v[i]
            
            ll = ddm_likelihood_single(v_i, a, z, t0, choice, rt)
            log_likelihood += ll
    
    # 返回负对数似然（用于最小化）
    return -log_likelihood


def fit_ddm_mle(X, y_choice, y_rt, feature_names):
    """
    使用最大似然估计拟合DDM参数
    
    参数:
    -----
    X : np.ndarray, shape (n_trials, n_features)
        特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    y_rt : np.ndarray, shape (n_trials,)
        反应时（秒）
    feature_names : list
        特征名称列表
        
    返回:
    -----
    results : dict
        拟合结果
    """
    print("\n=== 拟合DDM模型（MLE） ===")
    
    # 去除NaN值
    valid_mask = ~np.isnan(y_rt)
    X_valid = X[valid_mask]
    y_choice_valid = y_choice[valid_mask]
    y_rt_valid = y_rt[valid_mask]
    
    print(f"  有效RT数据: {len(y_rt_valid)}/{len(y_rt)}")
    print(f"  RT范围: [{np.min(y_rt_valid):.3f}, {np.max(y_rt_valid):.3f}] 秒")
    print(f"  RT均值: {np.mean(y_rt_valid):.3f} 秒")
    
    if len(y_rt_valid) < 10:
        print("  警告: 有效RT数据太少，跳过拟合")
        return None
    
    n_features = X.shape[1]
    n_params = n_features + 1 + 3  # beta (n_features+1) + a + z + t0
    
    # ========== 改进的初始化策略 ==========
    print(f"\n  步骤1: 使用逻辑回归初始化β系数...")
    
    # 1. 使用逻辑回归初始化β系数
    # 将选择转换为0/1（1=选左，2=选右 -> 0=选左，1=选右）
    y_choice_binary = (y_choice_valid == 2).astype(int)
    
    # 标准化特征（逻辑回归通常需要标准化）
    X_mean = np.mean(X_valid, axis=0)
    X_std = np.std(X_valid, axis=0) + 1e-8
    X_std_norm = (X_valid - X_mean) / X_std
    
    # 拟合逻辑回归
    lr_model = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    lr_model.fit(X_std_norm, y_choice_binary)
    
    # 将逻辑回归系数转换为DDM的β系数
    # 逻辑回归的系数需要缩放，因为DDM的漂移率尺度不同
    # 粗略估计：逻辑回归的系数 * 缩放因子
    # 缩放因子可以根据RT和选择的经验关系调整
    # ========== 可调参数说明 ==========
    # scale_factor: 控制漂移率的尺度
    #   - 太小（<1.0）：漂移率太小，可能无法区分选择
    #   - 太大（>5.0）：漂移率太大，可能导致数值不稳定
    #   - 推荐范围：1.0-3.0
    scale_factor = 2.0  # 可调参数：控制漂移率的尺度
    
    initial_beta = np.zeros(n_features + 1)
    initial_beta[0] = lr_model.intercept_[0] * scale_factor  # 截距
    initial_beta[1:] = lr_model.coef_[0] * scale_factor  # 特征系数
    
    print(f"    逻辑回归系数（缩放前）:")
    print(f"      截距: {lr_model.intercept_[0]:.4f}")
    for i, name in enumerate(feature_names):
        print(f"      {name}: {lr_model.coef_[0][i]:.4f}")
    print(f"    缩放因子: {scale_factor}")
    
    # 2. 初始化DDM参数
    # 阈值a：根据RT分布估计
    # 使用经验公式：a ≈ mean(RT - t0) * std(v) / 2
    # 简化：使用RT的标准差作为参考
    rt_std = np.std(y_rt_valid)
    initial_a = max(0.5, min(3.0, rt_std * 2.0))  # 在合理范围内
    
    # 起始点z：根据选择比例估计
    # 如果选右的比例高，z应该偏向右边（> a/2）
    prob_right = np.mean(y_choice_binary)
    initial_z_ratio = prob_right  # 直接用选择比例作为起始点比例
    
    # 非决策时间t0：使用RT的最小值或经验值
    # 通常t0是RT的10-30%
    initial_t0 = max(0.1, min(np.min(y_rt_valid) * 0.5, np.mean(y_rt_valid) * 0.2))
    
    initial_params = np.concatenate([
        initial_beta,
        [initial_a, initial_z_ratio, initial_t0]
    ])
    
    print(f"\n  参数初始化:")
    print(f"    截距 β₀: {initial_beta[0]:.4f}")
    for i, name in enumerate(feature_names):
        print(f"    {name} β{i+1}: {initial_beta[i+1]:.4f}")
    print(f"    阈值 a: {initial_a:.4f}")
    print(f"    起始点比例 z/a: {initial_z_ratio:.4f}")
    print(f"    非决策时间 t0: {initial_t0:.4f} 秒")
    
    # 计算初始似然值
    initial_ll = ddm_likelihood(initial_params, X_valid, y_choice_valid, y_rt_valid, 
                                np.ones(len(X_valid), dtype=bool))
    print(f"    初始负对数似然: {initial_ll:.4f}")
    
    # ========== 参数边界 ==========
    bounds = []
    # beta: 基于逻辑回归结果设置更合理的范围
    beta_range = max(5.0, np.abs(initial_beta).max() * 3)
    for i in range(n_features + 1):
        bounds.append((-beta_range, beta_range))
    # a: 必须大于0，合理范围
    bounds.append((0.1, 5.0))
    # z_ratio: 0到1之间
    bounds.append((0.0, 1.0))
    # t0: 必须大于0，小于最小RT
    bounds.append((0.0, np.min(y_rt_valid) * 0.9))
    
    # ========== 多次尝试优化 ==========
    print(f"\n  步骤2: 开始优化（多次尝试）...")
    
    best_result = None
    best_ll = np.inf
    n_tries = 3  # 尝试3次不同的初始化
    
    for try_idx in range(n_tries):
        if try_idx > 0:
            # 添加小的随机扰动
            noise_scale = 0.1
            perturbed_params = initial_params.copy()
            perturbed_params[:n_features+1] += np.random.normal(0, noise_scale, n_features+1)
            perturbed_params[n_features+1] = max(0.1, initial_a + np.random.normal(0, 0.1))
            perturbed_params[n_features+2] = np.clip(initial_z_ratio + np.random.normal(0, 0.05), 0, 1)
            perturbed_params[n_features+3] = np.clip(initial_t0 + np.random.normal(0, 0.02), 0, np.min(y_rt_valid) * 0.9)
            current_init = perturbed_params
        else:
            current_init = initial_params
        
        print(f"\n    尝试 {try_idx + 1}/{n_tries}...")
        
        # 优化（改进的设置）
        result = minimize(
            ddm_likelihood,
            current_init,
            args=(X_valid, y_choice_valid, y_rt_valid, np.ones(len(X_valid), dtype=bool)),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 2000,  # 增加最大迭代次数
                'disp': False,  # 减少输出
                'ftol': 1e-6,  # 函数值容差
                'gtol': 1e-5,  # 梯度容差
            }
        )
        
        if result.fun < best_ll:
            best_ll = result.fun
            best_result = result
            print(f"      ✓ 找到更好的解，负对数似然: {result.fun:.4f}")
        else:
            print(f"      × 当前解较差，负对数似然: {result.fun:.4f}")
    
    if best_result is None:
        print("  错误: 所有优化尝试都失败")
        return None
    
    result = best_result
    print(f"\n  最佳负对数似然: {result.fun:.4f}")
    
    if not result.success:
        print(f"  警告: 优化未完全收敛: {result.message}")
    
    # 提取拟合的参数
    beta_fitted = result.x[:n_features+1]
    a_fitted = result.x[n_features+1]
    z_ratio_fitted = result.x[n_features+2]
    z_fitted = z_ratio_fitted * a_fitted
    t0_fitted = result.x[n_features+3]
    
    print(f"\n  拟合结果:")
    print(f"    截距 β₀: {beta_fitted[0]:.4f}")
    for i, name in enumerate(feature_names):
        print(f"    {name} β{i+1}: {beta_fitted[i+1]:.4f}")
    print(f"    阈值 a: {a_fitted:.4f}")
    print(f"    起始点 z: {z_fitted:.4f} (比例: {z_ratio_fitted:.4f})")
    print(f"    非决策时间 t0: {t0_fitted:.4f} 秒")
    print(f"    负对数似然: {result.fun:.4f}")
    
    # 计算漂移率
    X_with_const = np.column_stack([np.ones(len(X_valid)), X_valid])
    v_fitted = np.dot(X_with_const, beta_fitted)
    
    print(f"\n  漂移率统计:")
    print(f"    均值: {np.mean(v_fitted):.4f}")
    print(f"    标准差: {np.std(v_fitted):.4f}")
    print(f"    范围: [{np.min(v_fitted):.4f}, {np.max(v_fitted):.4f}]")
    
    return {
        'beta': beta_fitted,
        'a': a_fitted,
        'z': z_fitted,
        'z_ratio': z_ratio_fitted,
        't0': t0_fitted,
        'v': v_fitted,
        'neg_log_likelihood': result.fun,
        'success': result.success,
        'message': result.message,
        'feature_names': feature_names
    }


def simulate_ddm(v, a, z, t0, n_simulations=1000, dt=0.001, max_rt=5.0):
    """
    模拟DDM过程，生成选择和RT
    
    参数:
    -----
    v : float
        漂移率
    a : float
        阈值
    z : float
        起始点
    t0 : float
        非决策时间
    n_simulations : int
        模拟次数
    dt : float
        时间步长
    max_rt : float
        最大反应时
        
    返回:
    -----
    choices : np.ndarray
        选择（1=选左，2=选右）
    rts : np.ndarray
        反应时（秒）
    """
    choices = []
    rts = []
    
    for _ in range(n_simulations):
        # 起始点
        position = z
        
        # 模拟Wiener过程
        time = 0.0
        while time < max_rt - t0:
            # Wiener过程的增量
            drift = v * dt
            diffusion = np.random.normal(0, np.sqrt(dt))
            position += drift + diffusion
            
            # 检查是否达到边界
            if position >= a:
                # 达到上边界（选右）
                choices.append(2)
                rts.append(time + t0)
                break
            elif position <= 0:
                # 达到下边界（选左）
                choices.append(1)
                rts.append(time + t0)
                break
            
            time += dt
        
        # 如果超时，根据最终位置决定
        if len(choices) < len(rts) + 1:
            if position > a / 2:
                choices.append(2)
            else:
                choices.append(1)
            rts.append(max_rt)
    
    return np.array(choices), np.array(rts)


def predict_ddm(ddm_results, X, y_choice, y_rt):
    """
    使用拟合的DDM模型进行预测
    
    参数:
    -----
    ddm_results : dict
        DDM拟合结果
    X : np.ndarray
        特征矩阵
    y_choice : np.ndarray
        实际选择
    y_rt : np.ndarray
        实际反应时
        
    返回:
    -----
    predictions : dict
        预测结果
    """
    print("\n=== DDM预测 ===")
    
    beta = ddm_results['beta']
    a = ddm_results['a']
    z = ddm_results['z']
    t0 = ddm_results['t0']
    
    # 计算漂移率
    X_with_const = np.column_stack([np.ones(len(X)), X])
    v = np.dot(X_with_const, beta)
    
    # 对于每个试次，模拟DDM过程
    # 为了速度，使用解析方法而不是模拟
    # 使用DDM的解析解计算选择概率和RT
    
    # 选择概率（使用DDM的解析公式，考虑起始点z）
    # 标准DDM公式：P(选上边界) = (exp(2*v*z) - 1) / (exp(2*v*a) - 1)
    # 对于v接近0的情况，使用极限形式
    prob_right = np.zeros_like(v)
    for i, v_i in enumerate(v):
        if abs(v_i) < 1e-6:
            # 无漂移情况：P(选右) = (a - z) / a
            prob_right[i] = (a - z) / a
        else:
            # 有漂移情况
            exp_2va = np.exp(2 * v_i * a)
            exp_2vz = np.exp(2 * v_i * z)
            prob_right[i] = (exp_2vz - 1) / (exp_2va - 1)
            # 确保概率在[0,1]范围内
            prob_right[i] = np.clip(prob_right[i], 0, 1)
    
    prob_left = 1 - prob_right
    
    # 预测选择（基于概率）
    y_pred_choice = np.where(prob_right > 0.5, 2, 1)
    
    # 预测RT（使用DDM的期望RT公式）
    # E[RT] = t0 + a / v * tanh(v*a) （简化公式）
    # 对于v接近0的情况需要特殊处理
    decision_time = np.zeros_like(v)
    for i, v_i in enumerate(v):
        if abs(v_i) < 1e-6:
            # 无漂移情况
            decision_time[i] = a**2 / 2  # 简化近似
        else:
            # 有漂移情况
            decision_time[i] = a / v_i * np.tanh(v_i * a)
    
    y_pred_rt = t0 + decision_time
    
    # 评估选择预测
    valid_mask = ~np.isnan(y_rt)
    y_choice_valid = y_choice[valid_mask]
    y_pred_choice_valid = y_pred_choice[valid_mask]
    prob_right_valid = prob_right[valid_mask]
    
    accuracy = accuracy_score(y_choice_valid, y_pred_choice_valid)
    try:
        auc = roc_auc_score((y_choice_valid == 2).astype(int), prob_right_valid)
    except:
        auc = np.nan
    
    # 评估RT预测
    y_rt_valid = y_rt[valid_mask]
    y_pred_rt_valid = y_pred_rt[valid_mask]
    
    r2 = r2_score(y_rt_valid, y_pred_rt_valid)
    mae = mean_absolute_error(y_rt_valid, y_pred_rt_valid)
    rmse = np.sqrt(mean_squared_error(y_rt_valid, y_pred_rt_valid))
    corr = pearsonr(y_rt_valid, y_pred_rt_valid)[0]
    
    print(f"\n  选择预测:")
    print(f"    准确率: {accuracy:.4f}")
    print(f"    AUC: {auc:.4f}")
    
    print(f"\n  RT预测:")
    print(f"    R²: {r2:.4f}")
    print(f"    MAE: {mae:.4f} 秒")
    print(f"    RMSE: {rmse:.4f} 秒")
    print(f"    相关系数: {corr:.4f}")
    
    return {
        'y_pred_choice': y_pred_choice,
        'y_pred_rt': y_pred_rt,
        'prob_right': prob_right,
        'accuracy': accuracy,
        'auc': auc,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': corr,
        'y_true_choice': y_choice_valid,
        'y_true_rt': y_rt_valid,
        'valid_mask': valid_mask
    }


def plot_ddm_results(ddm_results, predictions, output_dir):
    """绘制DDM结果"""
    print("\n=== 生成可视化图表 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 选择预测：预测概率vs实际
    fig, ax = plt.subplots(figsize=(8, 6))
    y_true = predictions['y_true_choice']
    prob_right = predictions['prob_right'][predictions['valid_mask']]
    
    for label in [1, 2]:
        mask = y_true == label
        ax.hist(prob_right[mask], bins=20, alpha=0.6, 
               label=f'实际{"左" if label==1 else "右"}', density=True)
    
    ax.set_xlabel('预测概率（选右）', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'DDM选择预测概率分布\n准确率={predictions["accuracy"]:.4f}, AUC={predictions["auc"]:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ddm_choice_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RT预测：预测vs实际
    fig, ax = plt.subplots(figsize=(8, 8))
    y_true_rt = predictions['y_true_rt']
    y_pred_rt = predictions['y_pred_rt'][predictions['valid_mask']]
    
    ax.scatter(y_true_rt, y_pred_rt, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_true_rt.min(), y_pred_rt.min())
    max_val = max(y_true_rt.max(), y_pred_rt.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    
    ax.set_xlabel('实际RT（秒）', fontsize=12)
    ax.set_ylabel('预测RT（秒）', fontsize=12)
    ax.set_title(f'DDM RT预测vs实际\nR2={predictions["r2"]:.4f}, MAE={predictions["mae"]:.4f}秒', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ddm_rt_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 漂移率分布
    fig, ax = plt.subplots(figsize=(8, 6))
    v = ddm_results['v']
    ax.hist(v, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='v=0')
    ax.set_xlabel('漂移率 v', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title(f'漂移率分布\n均值={np.mean(v):.4f}, 标准差={np.std(v):.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ddm_drift_rate_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存到: {output_dir}")


def save_ddm_results(ddm_results, predictions, trial_info, output_dir, feature_names):
    """保存DDM结果"""
    print("\n=== 保存结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存参数估计
    params_df = pd.DataFrame({
        'parameter': ['intercept'] + feature_names + ['threshold_a', 'starting_point_z', 'non_decision_time_t0'],
        'value': list(ddm_results['beta']) + [ddm_results['a'], ddm_results['z'], ddm_results['t0']],
        'description': ['漂移率截距'] + [f'漂移率系数({name})' for name in feature_names] + 
                      ['决策阈值', '起始点', '非决策时间']
    })
    params_df.to_csv(os.path.join(output_dir, 'ddm_parameters.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 保存预测结果
    pred_df = pd.DataFrame({
        'actual_choice': predictions['y_true_choice'],
        'predicted_choice': predictions['y_pred_choice'][predictions['valid_mask']],
        'prob_right': predictions['prob_right'][predictions['valid_mask']],
        'actual_rt': predictions['y_true_rt'],
        'predicted_rt': predictions['y_pred_rt'][predictions['valid_mask']],
        'drift_rate': ddm_results['v'][predictions['valid_mask']]
    })
    pred_df.to_csv(os.path.join(output_dir, 'ddm_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # 3. 保存性能指标
    perf_df = pd.DataFrame({
        'metric': ['Choice Accuracy', 'Choice AUC', 'RT R²', 'RT MAE', 'RT RMSE', 'RT Correlation', 
                  'Neg Log Likelihood', 'Threshold a', 'Starting Point z', 'Non-decision Time t0'],
        'value': [
            predictions['accuracy'],
            predictions['auc'],
            predictions['r2'],
            predictions['mae'],
            predictions['rmse'],
            predictions['correlation'],
            ddm_results['neg_log_likelihood'],
            ddm_results['a'],
            ddm_results['z'],
            ddm_results['t0']
        ]
    })
    perf_df.to_csv(os.path.join(output_dir, 'ddm_performance.csv'), index=False, encoding='utf-8-sig')
    
    print(f"  结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("="*60)
    print("DDM个人数据拟合")
    print("="*60)
    
    # 配置路径（可以修改为yanchen或其他被试）
    txt_files = [
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
    ]
    
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    ]
    
    output_dir = os.path.join(os.getcwd(), 'individual_ddm_results')
    
    # 1. 加载个人数据
    all_trials = load_individual_trials(txt_files, mapping_files)
    
    if len(all_trials) == 0:
        print("错误: 未能加载任何试次数据")
        return
    
    # 2. 加载特征数据
    project_root = os.getcwd()
    _, diff_zscore, pair_names = load_or_extract_features(project_root)
    
    # 3. 检查一致的选择（只保留AB和BA都选择同一个音频的试次）
    consistent_trials = check_consistent_choices(all_trials, pair_names)
    
    if len(consistent_trials) < 10:
        print("错误: 一致的试次太少")
        return
    
    # 4. 选择DDM特征
    feature_indices_0based, feature_names = select_ddm_features()
    
    # 5. 提取特征（只使用一致的选择，不带位置编码）
    X, y_choice, y_rt, trial_info = extract_features_for_ddm(
        consistent_trials, diff_zscore, pair_names, feature_indices_0based
    )
    
    if len(X) < 10:
        print("错误: 有效试次太少")
        return
    
    # 6. 数据诊断
    diagnose_data_quality(X, y_choice, y_rt, feature_names)
    
    # 7. 拟合DDM模型
    ddm_results = fit_ddm_mle(X, y_choice, y_rt, feature_names)
    
    if ddm_results is None:
        print("错误: DDM拟合失败")
        return
    
    # 8. 预测
    predictions = predict_ddm(ddm_results, X, y_choice, y_rt)
    
    # 9. 生成可视化
    plot_ddm_results(ddm_results, predictions, output_dir)
    
    # 10. 保存结果
    save_ddm_results(ddm_results, predictions, trial_info, output_dir, feature_names)
    
    # 11. 总结和建议
    print("\n" + "="*60)
    print("DDM拟合总结")
    print("="*60)
    print(f"\n使用 {len(consistent_trials)} 个一致试次，{len(X)} 个有效试次，{len(feature_names)} 个特征")
    print(f"\n选择预测:")
    print(f"  准确率: {predictions['accuracy']:.4f}")
    print(f"  AUC: {predictions['auc']:.4f}")
    print(f"\nRT预测:")
    print(f"  R²: {predictions['r2']:.4f}")
    print(f"  MAE: {predictions['mae']:.4f} 秒")
    print(f"  RMSE: {predictions['rmse']:.4f} 秒")
    print(f"\nDDM参数:")
    print(f"  阈值 a: {ddm_results['a']:.4f}")
    print(f"  起始点 z: {ddm_results['z']:.4f}")
    print(f"  非决策时间 t0: {ddm_results['t0']:.4f} 秒")
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()

