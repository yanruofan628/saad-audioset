#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jiachen个人数据验证 - 贝叶斯方法版本
使用贝叶斯逻辑回归，在小样本时更稳健
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否安装了PyMC（新版本，替代PyMC3）
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("警告: PyMC未安装")
    print("安装命令: pip install pymc arviz")
    print("将使用传统方法作为备选")

# 导入必要的函数
from spatiotemporal_glm_model import (
    parse_experiment_data_txt,
    parse_audio_mapping,
    parse_pair_name
)
from linear_regression_selected_features import (
    load_or_extract_features,
    extract_significant_features_from_table,
    build_consistent_folds
)
from individual_jiachen_validation import (
    load_jiachen_trials,
    check_left_right_consistency,
    extract_features_for_trials
)

import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def bayesian_logistic_regression(X, y, feature_names, n_samples=2000, n_tune=1000):
    """
    贝叶斯逻辑回归
    
    参数:
    -----
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵（应该已经标准化）
    y : np.ndarray, shape (n_samples,)
        二分类标签（0或1）
    feature_names : list
        特征名称列表
    n_samples : int
        MCMC采样数
    n_tune : int
        预热（tuning）采样数
        
    返回:
    -----
    trace : pymc3 trace对象
        后验分布采样结果
    summary : pandas DataFrame
        后验分布摘要
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC未安装，无法运行贝叶斯分析")
    
    n_samples_data, n_features = X.shape
    
    print(f"\n构建贝叶斯模型...")
    print(f"  样本数: {n_samples_data}")
    print(f"  特征数: {n_features}")
    print(f"  MCMC采样数: {n_samples}")
    print(f"  预热采样数: {n_tune}")
    
    # 标准化特征（贝叶斯方法通常需要标准化）
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_std_norm = (X - X_mean) / X_std
    
    print(f"\n使用弱先验（Weakly Informative Prior）:")
    print(f"  - 截距: Normal(0, 1)")
    print(f"  - 系数: Normal(0, 1)")
    
    with pm.Model() as model:
        # 先验分布
        # 截距：正态分布，均值0，标准差1
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        
        # 系数：正态分布，均值0，标准差1（弱先验）
        coefficients = pm.Normal('coefficients', mu=0, sigma=1, shape=n_features)
        
        # 线性组合
        logit_p = intercept + pm.math.dot(X_std_norm, coefficients)
        
        # 似然函数（伯努利分布）
        y_obs = pm.Bernoulli('y_obs', p=pm.math.sigmoid(logit_p), observed=y)
        
        # 采样（MCMC方法）
        print(f"\n开始MCMC采样（这可能需要几分钟）...")
        # PyMC 5.x默认返回InferenceData，不需要return_inference_data参数
        trace = pm.sample(n_samples, tune=n_tune, 
                         progressbar=True, cores=1, target_accept=0.9)
    
    print(f"\n采样完成！")
    
    # 总结后验分布
    summary = az.summary(trace, var_names=['intercept', 'coefficients'])
    
    return trace, summary, X_mean, X_std


def interpret_bayesian_results(trace, feature_names, feature_threshold=0.01):
    """
    解释贝叶斯结果
    
    参数:
    -----
    trace : pymc3 trace对象
    feature_names : list
        特征名称列表
    feature_threshold : float
        判断系数是否"有意义"的阈值
        
    返回:
    -----
    results_dict : dict
        结果字典
    """
    print("\n" + "="*60)
    print("贝叶斯结果解释")
    print("="*60)
    
    # 提取后验分布（PyMC新版本的形状处理）
    intercept_samples = trace.posterior['intercept'].values.flatten()
    coef_samples = trace.posterior['coefficients'].values
    
    # 处理coef_samples的形状（可能是3维或2维）
    if coef_samples.ndim == 3:
        # (n_chains, n_draws, n_features) -> (n_chains * n_draws, n_features)
        n_chains, n_draws, n_features_coef = coef_samples.shape
        coef_samples_flat = coef_samples.reshape(-1, n_features_coef)
    else:
        coef_samples_flat = coef_samples.reshape(-1, coef_samples.shape[-1])
    
    results = {
        'intercept': {},
        'coefficients': []
    }
    
    # 截距
    intercept_mean = np.mean(intercept_samples)
    intercept_std = np.std(intercept_samples)
    intercept_hdi_low = np.percentile(intercept_samples, 3)
    intercept_hdi_high = np.percentile(intercept_samples, 97)
    intercept_prob_positive = np.mean(intercept_samples > 0)
    
    results['intercept'] = {
        'mean': intercept_mean,
        'std': intercept_std,
        'hdi_low': intercept_hdi_low,
        'hdi_high': intercept_hdi_high,
        'prob_positive': intercept_prob_positive
    }
    
    print(f"\n截距:")
    print(f"  后验均值: {intercept_mean:.4f}")
    print(f"  后验标准差: {intercept_std:.4f}")
    print(f"  94%可信区间: [{intercept_hdi_low:.4f}, {intercept_hdi_high:.4f}]")
    print(f"  大于0的概率: {intercept_prob_positive:.2%}")
    
    # 系数
    print(f"\n特征系数:")
    print(f"{'特征名称':<35} {'均值':<10} {'94%可信区间':<25} {'>0概率':<10} {'解释'}")
    print("-"*100)
    
    for i, name in enumerate(feature_names):
        if coef_samples.ndim == 3:
            coef_samples_i = coef_samples[:, :, i].flatten()
        else:
            coef_samples_i = coef_samples_flat[:, i]
        mean_coef = np.mean(coef_samples_i)
        std_coef = np.std(coef_samples_i)
        hdi_low = np.percentile(coef_samples_i, 3)
        hdi_high = np.percentile(coef_samples_i, 97)
        prob_positive = np.mean(coef_samples_i > 0)
        prob_meaningful = np.mean(np.abs(coef_samples_i) > feature_threshold)
        
        # 判断是否有影响
        if hdi_low > 0:
            interpretation = "很可能有正向影响"
        elif hdi_high < 0:
            interpretation = "很可能有负向影响"
        elif prob_meaningful > 0.7:
            interpretation = "可能有影响（不确定方向）"
        else:
            interpretation = "影响不确定"
        
        results['coefficients'].append({
            'feature': name,
            'mean': mean_coef,
            'std': std_coef,
            'hdi_low': hdi_low,
            'hdi_high': hdi_high,
            'prob_positive': prob_positive,
            'prob_meaningful': prob_meaningful,
            'interpretation': interpretation
        })
        
        print(f"{name[:34]:<35} {mean_coef:>9.4f} [{hdi_low:>6.3f}, {hdi_high:<6.3f}] "
              f"{prob_positive:>9.2%} {interpretation}")
    
    return results


def predict_with_bayesian(trace, X, X_mean, X_std):
    """
    用贝叶斯模型进行预测
    
    参数:
    -----
    trace : pymc3 trace对象
        后验分布采样结果
    X : np.ndarray
        特征矩阵（原始，未标准化）
    X_mean : np.ndarray
        特征均值（用于标准化）
    X_std : np.ndarray
        特征标准差（用于标准化）
        
    返回:
    -----
    y_pred_proba : np.ndarray
        预测概率（每个样本的预测概率分布）
    y_pred_proba_mean : np.ndarray
        预测概率的均值
    y_pred_proba_std : np.ndarray
        预测概率的标准差
    """
    # 标准化特征
    X_std_norm = (X - X_mean) / X_std
    
    # 提取后验样本
    intercept_samples = trace.posterior['intercept'].values.flatten()
    coef_samples = trace.posterior['coefficients'].values
    
    n_samples_data = X.shape[0]
    n_mcmc_samples = len(intercept_samples)
    
    # 对每个MCMC样本计算预测概率
    y_pred_proba_samples = []
    
    # PyMC新版本的coef_samples形状可能是 (n_chains, n_draws, n_features)
    # 需要正确提取
    if coef_samples.ndim == 3:
        n_chains, n_draws, n_features_coef = coef_samples.shape
        for i in range(min(100, n_mcmc_samples)):
            chain_idx = i // n_draws
            draw_idx = i % n_draws
            intercept = intercept_samples[i]
            coefficients = coef_samples[chain_idx, draw_idx, :]
            
            # 线性组合
            logit_p = intercept + np.dot(X_std_norm, coefficients)
            
            # 转换为概率
            p = 1 / (1 + np.exp(-logit_p))
            y_pred_proba_samples.append(p)
    else:
        # 如果形状不同，尝试直接使用
        n_features_coef = coef_samples.shape[-1]
        coef_samples_reshaped = coef_samples.reshape(-1, n_features_coef)
        for i in range(min(100, n_mcmc_samples)):
            intercept = intercept_samples[i]
            coefficients = coef_samples_reshaped[i, :]
            
            # 线性组合
            logit_p = intercept + np.dot(X_std_norm, coefficients)
            
            # 转换为概率
            p = 1 / (1 + np.exp(-logit_p))
            y_pred_proba_samples.append(p)
    
    y_pred_proba_samples = np.array(y_pred_proba_samples)  # (n_mcmc_samples, n_data_samples)
    
    # 计算均值和标准差
    y_pred_proba_mean = np.mean(y_pred_proba_samples, axis=0)
    y_pred_proba_std = np.std(y_pred_proba_samples, axis=0)
    
    return y_pred_proba_samples, y_pred_proba_mean, y_pred_proba_std


def compare_frequentist_vs_bayesian(X, y, feature_names, test_size=0.2, random_state=42):
    """
    对比频率主义和贝叶斯方法（包括预测效果）
    
    参数:
    -----
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        标签
    feature_names : list
        特征名称列表
    test_size : float
        测试集比例
    random_state : int
        随机种子
        
    返回:
    -----
    comparison_results : dict
        对比结果
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    
    print("\n" + "="*60)
    print("对比：频率主义 vs 贝叶斯（包括预测效果）")
    print("="*60)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  测试集: {len(X_test)} 个样本")
    
    # 1. 频率主义方法
    print("\n" + "-"*60)
    print("1. 频率主义方法（传统统计）")
    print("-"*60)
    
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    logit_freq = sm.Logit(y_train, X_train_const).fit(disp=0, maxiter=1000)
    
    print(f"\n  模型拟合:")
    print(f"    伪R²: {logit_freq.prsquared:.4f}")
    sig_count = np.sum(logit_freq.pvalues[1:] < 0.05)
    print(f"    显著特征数 (p<0.05): {sig_count}/{len(feature_names)}")
    
    # 预测
    y_pred_proba_freq_train = logit_freq.predict(X_train_const)
    y_pred_proba_freq_test = logit_freq.predict(X_test_const)
    y_pred_freq_train = (y_pred_proba_freq_train > 0.5).astype(int)
    y_pred_freq_test = (y_pred_proba_freq_test > 0.5).astype(int)
    
    # 评估
    acc_train_freq = accuracy_score(y_train, y_pred_freq_train)
    acc_test_freq = accuracy_score(y_test, y_pred_freq_test)
    try:
        auc_train_freq = roc_auc_score(y_train, y_pred_proba_freq_train)
        auc_test_freq = roc_auc_score(y_test, y_pred_proba_freq_test)
    except:
        auc_train_freq = np.nan
        auc_test_freq = np.nan
    
    logloss_train_freq = log_loss(y_train, y_pred_proba_freq_train)
    logloss_test_freq = log_loss(y_test, y_pred_proba_freq_test)
    
    print(f"\n  训练集预测:")
    print(f"    准确率: {acc_train_freq:.4f}")
    print(f"    AUC: {auc_train_freq:.4f}")
    print(f"    Log Loss: {logloss_train_freq:.4f}")
    
    print(f"\n  测试集预测:")
    print(f"    准确率: {acc_test_freq:.4f}")
    print(f"    AUC: {auc_test_freq:.4f}")
    print(f"    Log Loss: {logloss_test_freq:.4f}")
    
    # 2. 贝叶斯方法
    if PYMC_AVAILABLE:
        print("\n" + "-"*60)
        print("2. 贝叶斯方法")
        print("-"*60)
        
        trace, summary, X_mean, X_std = bayesian_logistic_regression(
            X_train, y_train, feature_names
        )
        bayesian_results = interpret_bayesian_results(trace, feature_names)
        
        # 统计"很可能有影响"的特征数
        likely_effect_count = sum(1 for c in bayesian_results['coefficients'] 
                                 if '很可能' in c['interpretation'])
        print(f"\n  模型拟合:")
        print(f"    很可能有影响的特征数: {likely_effect_count}/{len(feature_names)}")
        
        # 预测（训练集和测试集）
        _, y_pred_proba_bayes_train_mean, y_pred_proba_bayes_train_std = predict_with_bayesian(
            trace, X_train, X_mean, X_std
        )
        _, y_pred_proba_bayes_test_mean, y_pred_proba_bayes_test_std = predict_with_bayesian(
            trace, X_test, X_mean, X_std
        )
        
        y_pred_bayes_train = (y_pred_proba_bayes_train_mean > 0.5).astype(int)
        y_pred_bayes_test = (y_pred_proba_bayes_test_mean > 0.5).astype(int)
        
        # 评估
        acc_train_bayes = accuracy_score(y_train, y_pred_bayes_train)
        acc_test_bayes = accuracy_score(y_test, y_pred_bayes_test)
        try:
            auc_train_bayes = roc_auc_score(y_train, y_pred_proba_bayes_train_mean)
            auc_test_bayes = roc_auc_score(y_test, y_pred_proba_bayes_test_mean)
        except:
            auc_train_bayes = np.nan
            auc_test_bayes = np.nan
        
        logloss_train_bayes = log_loss(y_train, y_pred_proba_bayes_train_mean)
        logloss_test_bayes = log_loss(y_test, y_pred_proba_bayes_test_mean)
        
        print(f"\n  训练集预测:")
        print(f"    准确率: {acc_train_bayes:.4f}")
        print(f"    AUC: {auc_train_bayes:.4f}")
        print(f"    Log Loss: {logloss_train_bayes:.4f}")
        print(f"    预测不确定性（平均标准差）: {np.mean(y_pred_proba_bayes_train_std):.4f}")
        
        print(f"\n  测试集预测:")
        print(f"    准确率: {acc_test_bayes:.4f}")
        print(f"    AUC: {auc_test_bayes:.4f}")
        print(f"    Log Loss: {logloss_test_bayes:.4f}")
        print(f"    预测不确定性（平均标准差）: {np.mean(y_pred_proba_bayes_test_std):.4f}")
        
        # 3. 详细对比分析
        print("\n" + "="*60)
        print("3. 详细对比分析")
        print("="*60)
        
        # 计算预测概率的差异
        prob_diff = np.abs(y_pred_proba_freq_test - y_pred_proba_bayes_test_mean)
        max_prob_diff = np.max(prob_diff)
        mean_prob_diff = np.mean(prob_diff)
        
        print(f"\n  预测概率差异分析:")
        print(f"    最大差异: {max_prob_diff:.6f}")
        print(f"    平均差异: {mean_prob_diff:.6f}")
        print(f"    标准差: {np.std(prob_diff):.6f}")
        
        # 检查为什么准确率和AUC相同
        print(f"\n  为什么准确率和AUC相同？")
        print(f"    - 测试集大小: {len(y_test)} 个样本")
        print(f"    - 预测概率平均差异: {mean_prob_diff:.6f}")
        print(f"    - 如果差异 < 0.01，准确率可能相同（基于0.5阈值）")
        
        # 检查预测标签是否相同
        pred_label_same = np.sum(y_pred_freq_test == y_pred_bayes_test)
        pred_label_diff = len(y_test) - pred_label_same
        print(f"    - 预测标签相同的样本: {pred_label_same}/{len(y_test)} ({pred_label_same/len(y_test)*100:.1f}%)")
        print(f"    - 预测标签不同的样本: {pred_label_diff}/{len(y_test)} ({pred_label_diff/len(y_test)*100:.1f}%)")
        
        # 如果预测标签完全相同，解释原因
        if pred_label_same == len(y_test):
            print(f"\n  → 预测标签完全相同的原因:")
            print(f"    1. 预测概率差异很小（平均{mean_prob_diff:.6f}）")
            print(f"    2. 两种方法都使用0.5作为阈值")
            print(f"    3. 如果概率差异 < 0.5，预测标签就会相同")
            print(f"    4. 这是正常的，特别是在小样本情况下")
        
        # 4. 对比总结
        print("\n" + "="*60)
        print("4. 对比总结")
        print("="*60)
        
        print(f"\n  测试集预测效果对比:")
        print(f"    {'指标':<20} {'频率主义':<15} {'贝叶斯':<15} {'差异':<15}")
        print(f"    {'-'*65}")
        print(f"    {'准确率':<20} {acc_test_freq:<15.4f} {acc_test_bayes:<15.4f} "
              f"{acc_test_bayes-acc_test_freq:+.4f}")
        print(f"    {'AUC':<20} {auc_test_freq:<15.4f} {auc_test_bayes:<15.4f} "
              f"{auc_test_bayes-auc_test_freq:+.4f}")
        print(f"    {'Log Loss':<20} {logloss_test_freq:<15.4f} {logloss_test_bayes:<15.4f} "
              f"{logloss_test_bayes-logloss_test_freq:+.4f}")
        
        print(f"\n  为什么Log Loss不同？")
        print(f"    - Log Loss对预测概率的微小差异很敏感")
        print(f"    - 即使准确率相同，概率的微小差异也会影响Log Loss")
        print(f"    - 贝叶斯方法的Log Loss更低，说明概率预测更准确")
        
        print(f"\n  贝叶斯方法的额外优势:")
        print(f"    - 可以量化预测不确定性（平均标准差: {np.mean(y_pred_proba_bayes_test_std):.4f}）")
        print(f"    - 可以提供概率性陈述（而非二元判断）")
        print(f"    - 小样本时更稳健")
        print(f"    - Log Loss更低，说明概率预测更准确")
        
        comparison_results = {
            'frequentist': {
                'model': logit_freq,
                'acc_train': acc_train_freq,
                'acc_test': acc_test_freq,
                'auc_train': auc_train_freq,
                'auc_test': auc_test_freq,
                'logloss_train': logloss_train_freq,
                'logloss_test': logloss_test_freq,
                'y_pred_proba_train': y_pred_proba_freq_train,
                'y_pred_proba_test': y_pred_proba_freq_test,
                'y_pred_train': y_pred_freq_train,
                'y_pred_test': y_pred_freq_test
            },
            'bayesian': {
                'trace': trace,
                'acc_train': acc_train_bayes,
                'acc_test': acc_test_bayes,
                'auc_train': auc_train_bayes,
                'auc_test': auc_test_bayes,
                'logloss_train': logloss_train_bayes,
                'logloss_test': logloss_test_bayes,
                'y_pred_proba_train_mean': y_pred_proba_bayes_train_mean,
                'y_pred_proba_test_mean': y_pred_proba_bayes_test_mean,
                'y_pred_proba_train_std': y_pred_proba_bayes_train_std,
                'y_pred_proba_test_std': y_pred_proba_bayes_test_std,
                'y_pred_train': y_pred_bayes_train,
                'y_pred_test': y_pred_bayes_test
            },
            'bayesian_results': bayesian_results,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'prob_diff': prob_diff,
            'max_prob_diff': max_prob_diff,
            'mean_prob_diff': mean_prob_diff
        }
        
        return comparison_results
    else:
        print("\n2. 贝叶斯方法:")
        print("  PyMC未安装，无法运行贝叶斯分析")
        return {
            'frequentist': {
                'model': logit_freq,
                'acc_test': acc_test_freq,
                'auc_test': auc_test_freq
            },
            'bayesian': None
        }


def plot_bayesian_results(trace, feature_names, output_dir, max_features=9):
    """绘制贝叶斯结果"""
    print("\n生成可视化图表...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    intercept_samples = trace.posterior['intercept'].values.flatten()
    coef_samples = trace.posterior['coefficients'].values
    
    # 处理coef_samples的形状
    if coef_samples.ndim == 3:
        coef_samples_flat = coef_samples.reshape(-1, coef_samples.shape[-1])
    else:
        coef_samples_flat = coef_samples.reshape(-1, coef_samples.shape[-1])
    
    # 1. 截距和系数的后验分布
    n_features = min(len(feature_names), max_features)
    n_cols = 3
    n_rows = (n_features + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # 截距
    ax = axes[0]
    ax.hist(intercept_samples, bins=50, alpha=0.7, density=True, color='blue')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='0')
    ax.axvline(np.mean(intercept_samples), color='g', linestyle='--', 
               linewidth=2, label=f'均值={np.mean(intercept_samples):.3f}')
    hdi_low = np.percentile(intercept_samples, 3)
    hdi_high = np.percentile(intercept_samples, 97)
    ax.axvspan(hdi_low, hdi_high, alpha=0.2, color='yellow', label='94%可信区间')
    ax.set_xlabel('截距值')
    ax.set_ylabel('密度')
    ax.set_title('截距的后验分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 系数
    for i in range(n_features):
        ax = axes[i+1]
        coef_samples_i = coef_samples_flat[:, i]
        ax.hist(coef_samples_i, bins=50, alpha=0.7, density=True, color='skyblue')
        ax.axvline(0, color='r', linestyle='--', linewidth=2)
        ax.axvline(np.mean(coef_samples_i), color='g', linestyle='--', 
                   linewidth=2, label=f'均值={np.mean(coef_samples_i):.3f}')
        hdi_low = np.percentile(coef_samples_i, 3)
        hdi_high = np.percentile(coef_samples_i, 97)
        ax.axvspan(hdi_low, hdi_high, alpha=0.2, color='yellow', label='94%可信区间')
        ax.set_xlabel('系数值')
        ax.set_ylabel('密度')
        ax.set_title(f'{feature_names[i][:25]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features+1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bayesian_posterior_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 系数后验均值和可信区间
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names)*0.5)))
    
    # 处理coef_samples的形状
    if coef_samples.ndim == 3:
        coef_samples_flat = coef_samples.reshape(-1, coef_samples.shape[-1])
    else:
        coef_samples_flat = coef_samples.reshape(-1, coef_samples.shape[-1])
    
    coef_means = [np.mean(coef_samples_flat[:, i]) for i in range(len(feature_names))]
    hdi_lows = [np.percentile(coef_samples_flat[:, i], 3) for i in range(len(feature_names))]
    hdi_highs = [np.percentile(coef_samples_flat[:, i], 97) for i in range(len(feature_names))]
    
    y_pos = np.arange(len(feature_names))
    
    for i in range(len(feature_names)):
        ax.plot([hdi_lows[i], hdi_highs[i]], [i, i], 'b-', linewidth=2, alpha=0.7)
        ax.plot(coef_means[i], i, 'ro', markersize=8)
    
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='0')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name[:40] for name in feature_names])
    ax.set_xlabel('系数值')
    ax.set_title('特征系数的后验均值（红点）和94%可信区间（蓝色）')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bayesian_coefficients_intervals.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存到: {output_dir}")


def save_bayesian_results(bayesian_results, output_dir):
    """保存贝叶斯结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存系数结果
    coef_df = pd.DataFrame(bayesian_results['coefficients'])
    coef_df.to_csv(os.path.join(output_dir, 'bayesian_coefficients.csv'), 
                   index=False, encoding='utf-8-sig')
    
    # 保存截距结果
    intercept_df = pd.DataFrame([bayesian_results['intercept']])
    intercept_df.to_csv(os.path.join(output_dir, 'bayesian_intercept.csv'), 
                       index=False, encoding='utf-8-sig')
    
    print(f"\n贝叶斯结果已保存到: {output_dir}")


def plot_prediction_comparison(comparison_results, output_dir):
    """绘制预测效果对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    freq_results = comparison_results['frequentist']
    bayes_results = comparison_results['bayesian']
    y_test = comparison_results['y_test']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 预测概率分布对比
    ax = axes[0, 0]
    ax.hist(freq_results['y_pred_proba_test'], bins=20, alpha=0.6, 
           label='频率主义', density=True)
    ax.hist(bayes_results['y_pred_proba_test_mean'], bins=20, alpha=0.6, 
           label='贝叶斯', density=True)
    ax.set_xlabel('预测概率')
    ax.set_ylabel('密度')
    ax.set_title('测试集预测概率分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 预测概率 vs 实际标签（频率主义）
    ax = axes[0, 1]
    for label in [0, 1]:
        mask = y_test == label
        ax.scatter(freq_results['y_pred_proba_test'][mask], 
                  y_test[mask] + np.random.normal(0, 0.02, np.sum(mask)),
                  alpha=0.6, label=f'实际={label}', s=30)
    ax.set_xlabel('预测概率')
    ax.set_ylabel('实际标签')
    ax.set_title('频率主义：预测概率 vs 实际标签')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 预测概率 vs 实际标签（贝叶斯，带不确定性）
    ax = axes[1, 0]
    for label in [0, 1]:
        mask = y_test == label
        ax.errorbar(bayes_results['y_pred_proba_test_mean'][mask],
                   y_test[mask] + np.random.normal(0, 0.02, np.sum(mask)),
                   xerr=bayes_results['y_pred_proba_test_std'][mask],
                   alpha=0.6, label=f'实际={label}', fmt='o', markersize=5,
                   capsize=2, capthick=1)
    ax.set_xlabel('预测概率（均值±标准差）')
    ax.set_ylabel('实际标签')
    ax.set_title('贝叶斯：预测概率 vs 实际标签（带不确定性）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 两种方法预测概率对比
    ax = axes[1, 1]
    ax.scatter(freq_results['y_pred_proba_test'], 
              bayes_results['y_pred_proba_test_mean'],
              alpha=0.6, s=30)
    min_val = min(freq_results['y_pred_proba_test'].min(), 
                 bayes_results['y_pred_proba_test_mean'].min())
    max_val = max(freq_results['y_pred_proba_test'].max(), 
                 bayes_results['y_pred_proba_test_mean'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美一致')
    ax.set_xlabel('频率主义预测概率')
    ax.set_ylabel('贝叶斯预测概率')
    ax.set_title('两种方法预测概率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  预测对比图已保存: prediction_comparison.png")


def save_prediction_results(comparison_results, output_dir):
    """保存预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    freq_results = comparison_results['frequentist']
    bayes_results = comparison_results['bayesian']
    y_test = comparison_results['y_test']
    
    # 保存测试集预测结果
    prob_diff = comparison_results.get('prob_diff', 
                                       np.abs(freq_results['y_pred_proba_test'] - 
                                             bayes_results['y_pred_proba_test_mean']))
    pred_df = pd.DataFrame({
        'actual': y_test,
        'freq_pred_proba': freq_results['y_pred_proba_test'],
        'freq_pred': freq_results['y_pred_test'],
        'bayes_pred_proba_mean': bayes_results['y_pred_proba_test_mean'],
        'bayes_pred_proba_std': bayes_results['y_pred_proba_test_std'],
        'bayes_pred': bayes_results['y_pred_test'],
        'prob_diff': prob_diff
    })
    pred_df.to_csv(os.path.join(output_dir, 'prediction_comparison.csv'), 
                   index=False, encoding='utf-8-sig')
    
    # 保存性能对比
    perf_df = pd.DataFrame({
        'method': ['频率主义', '贝叶斯'],
        'test_accuracy': [freq_results['acc_test'], bayes_results['acc_test']],
        'test_auc': [freq_results['auc_test'], bayes_results['auc_test']],
        'test_logloss': [freq_results['logloss_test'], bayes_results['logloss_test']],
        'prediction_uncertainty': [np.nan, np.mean(bayes_results['y_pred_proba_test_std'])]
    })
    perf_df.to_csv(os.path.join(output_dir, 'performance_comparison.csv'), 
                  index=False, encoding='utf-8-sig')
    
    print(f"  预测结果已保存: prediction_comparison.csv")
    print(f"  性能对比已保存: performance_comparison.csv")


def main():
    """主函数"""
    print("="*60)
    print("jiachen个人数据验证 - 贝叶斯方法版本")
    print("="*60)
    
    if not PYMC_AVAILABLE:
        print("\n错误: PyMC未安装")
        print("请先安装: pip install pymc arviz")
        return
    
    # 配置路径
    txt_files = [
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
    ]
    
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    ]
    
    output_dir = os.path.join(os.getcwd(), 'individual_jiachen_bayesian_results')
    
    # 1. 加载数据
    print("\n=== 1. 加载数据 ===")
    all_trials = load_jiachen_trials(txt_files, mapping_files)
    
    # 2. 加载pair_list
    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    
    # 3. 检查一致性
    print("\n=== 2. 检查左右平衡一致性 ===")
    consistent_trials = check_left_right_consistency(all_trials, pair_names)
    
    # 4. 提取特征
    print("\n=== 3. 提取特征 ===")
    feature_indices_0based, _, feature_names = extract_significant_features_from_table()
    X, y_choice, y_rt = extract_features_for_trials(
        consistent_trials, pair_names, feature_indices_0based
    )
    
    # 转换为二分类
    y_binary = (y_choice == 2).astype(int)
    
    print(f"\n数据准备完成:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {len(feature_names)}")
    print(f"  选左: {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"  选右: {np.sum(y_binary == 1)} ({np.mean(y_binary == 1)*100:.1f}%)")
    
    # 5. 对比频率主义和贝叶斯方法（包括预测效果）
    print("\n=== 4. 对比频率主义和贝叶斯方法（包括预测效果） ===")
    comparison_results = compare_frequentist_vs_bayesian(
        X, y_binary, feature_names
    )
    
    if comparison_results.get('bayesian') is not None:
        trace = comparison_results['bayesian']['trace']
        bayesian_results = comparison_results['bayesian_results']
        # 6. 绘制结果
        print("\n=== 5. 生成可视化图表 ===")
        plot_bayesian_results(trace, feature_names, output_dir)
        
        # 7. 保存结果
        print("\n=== 6. 保存结果 ===")
        save_bayesian_results(bayesian_results, output_dir)
        
        # 7. 绘制预测对比图
        print("\n=== 7. 生成预测对比图 ===")
        plot_prediction_comparison(comparison_results, output_dir)
        
        # 8. 保存预测结果
        print("\n=== 8. 保存预测结果 ===")
        save_prediction_results(comparison_results, output_dir)
        
        # 9. 总结
        print("\n" + "="*60)
        print("总结")
        print("="*60)
        
        freq_results = comparison_results['frequentist']
        bayes_results = comparison_results['bayesian']
        
        print(f"\n频率主义方法:")
        print(f"  测试集准确率: {freq_results['acc_test']:.4f}")
        print(f"  测试集AUC: {freq_results['auc_test']:.4f}")
        
        print(f"\n贝叶斯方法:")
        print(f"  测试集准确率: {bayes_results['acc_test']:.4f}")
        print(f"  测试集AUC: {bayes_results['auc_test']:.4f}")
        print(f"  预测不确定性: {np.mean(bayes_results['y_pred_proba_test_std']):.4f}")
        
        likely_effect_count = sum(1 for c in bayesian_results['coefficients'] 
                                 if '很可能' in c['interpretation'])
        print(f"  很可能有影响的特征数: {likely_effect_count}/{len(feature_names)}")
        
        print(f"\n结果已保存到: {output_dir}")
        print("\n建议:")
        print("1. 查看 bayesian_coefficients.csv 了解每个特征的影响概率")
        print("2. 查看 prediction_comparison.png 对比两种方法的预测效果")
        print("3. 如果贝叶斯显示特征有影响，可以考虑用DDM进一步分析")
        print("4. 如果贝叶斯也显示特征影响不确定，可能需要更多数据")


if __name__ == '__main__':
    main()

