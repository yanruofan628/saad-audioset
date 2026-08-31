#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分层贝叶斯注意力模型
Hierarchical Bayesian Attention Model
====================================

使用分层贝叶斯框架建模多被试者的被动听觉注意：
- 群体层：所有被试共享的注意力模式
- 个体层：每个被试的注意力权重（围绕群体分布）
- 试次层：每个试次的预测

模型结构：
1. 时间注意力（temporal attention）：不同时间阶段的权重
2. 特征注意力（feature attention）：不同特征的权重
3. 简化的网络结构：避免过度复杂化
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查PyMC
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("警告: PyMC未安装，无法运行分层贝叶斯分析")
    print("安装命令: pip install pymc arviz")

# 导入现有模块
from individual_choice_with_position import (
    SUBJECT_CONFIGS,
    load_subject_trials,
    check_left_right_consistency,
    build_stage_dataset_for_subject,
    extract_features_for_attention_model,
)
from feature_stage_attention_model import (
    StageFeatureDataset,
    get_feature_configs_from_linear_regression,
    load_or_build_cached_dataset,
    SR,
    HOP,
    DEFAULT_STAGE_SECONDS,
)

RNG_SEED = 42
np.random.seed(RNG_SEED)


# ============================================================================
# 数据准备函数
# ============================================================================
def prepare_hierarchical_data(subject_configs: List[Dict], project_root: str, base_dir: str):
    """
    为所有被试准备分层数据
    
    Returns:
    --------
    hierarchical_data : dict
        {
            'n_subjects': int,
            'n_stages': int,
            'n_features': int,
            'subject_data': [
                {
                    'subject_id': str,
                    'X_stage': np.ndarray,  # (n_trials, n_stages, n_features)
                    'y_choice': np.ndarray,  # (n_trials,)
                    'valid_indices': list
                },
                ...
            ]
        }
    """
    print("\n" + "=" * 80)
    print("准备分层贝叶斯数据")
    print("=" * 80)
    
    feature_configs = get_feature_configs_from_linear_regression()
    
    hierarchical_data = {
        'n_subjects': 0,
        'n_stages': None,
        'n_features': None,
        'subject_data': [],
        'feature_configs': feature_configs,
    }
    
    subject_ids = []
    
    for subject_config in subject_configs:
        subject_id = subject_config['subject_id']
        print(f"\n处理被试: {subject_id}")
        
        # 加载被试数据
        trials = load_subject_trials(subject_config)
        if len(trials) == 0:
            print(f"  警告: 被试 {subject_id} 无数据，跳过")
            continue
        
        # 检查一致性
        from linear_regression_selected_features import load_or_extract_features
        _, _, pair_names = load_or_extract_features(project_root)
        consistent_trials = check_left_right_consistency(trials, pair_names)
        
        if len(consistent_trials) < 10:
            print(f"  警告: 被试 {subject_id} 一致试次太少 ({len(consistent_trials)})，跳过")
            continue
        
        # 构建阶段数据集
        try:
            stage_dataset = build_stage_dataset_for_subject(
                consistent_trials, pair_names, project_root, base_dir
            )
        except Exception as e:
            print(f"  错误: 构建数据集失败: {e}")
            continue
        
        # 提取特征
        X_global, X_stage, y_choice, valid_indices = extract_features_for_attention_model(
            consistent_trials, pair_names, stage_dataset
        )
        
        if len(X_stage) == 0:
            print(f"  警告: 被试 {subject_id} 无有效特征，跳过")
            continue
        
        # 转换为二分类
        y_binary = (y_choice == 2).astype(int)  # 1=选右 -> 1, 1=选左 -> 0
        
        # 存储数据
        hierarchical_data['subject_data'].append({
            'subject_id': subject_id,
            'X_stage': X_stage.astype(np.float32),
            'y_choice': y_binary.astype(int),
            'valid_indices': valid_indices,
            'n_trials': len(X_stage),
        })
        
        subject_ids.append(subject_id)
        
        # 设置维度（所有被试应该相同）
        if hierarchical_data['n_stages'] is None:
            hierarchical_data['n_stages'] = X_stage.shape[1]
            hierarchical_data['n_features'] = X_stage.shape[2]
        
        print(f"  ✓ 成功加载: {len(X_stage)} 个试次")
        print(f"    阶段数: {hierarchical_data['n_stages']}")
        print(f"    特征数: {hierarchical_data['n_features']}")
    
    hierarchical_data['n_subjects'] = len(hierarchical_data['subject_data'])
    hierarchical_data['subject_ids'] = subject_ids
    
    print(f"\n数据准备完成:")
    print(f"  被试数: {hierarchical_data['n_subjects']}")
    print(f"  阶段数: {hierarchical_data['n_stages']}")
    print(f"  特征数: {hierarchical_data['n_features']}")
    total_trials = sum(sd['n_trials'] for sd in hierarchical_data['subject_data'])
    print(f"  总试次数: {total_trials}")
    
    return hierarchical_data


# ============================================================================
# 分层贝叶斯模型（简化版：线性映射 + 时间注意力）
# ============================================================================
def build_hierarchical_attention_model(data: Dict, use_simplified: bool = True):
    """
    构建分层贝叶斯注意力模型
    
    Parameters:
    -----------
    data : dict
        分层数据字典
    use_simplified : bool
        True: 简化版（线性映射，固定时间权重）
        False: 完整版（自适应时间注意力）
    
    Returns:
    --------
    model : pymc.Model
        构建的PyMC模型
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC未安装，无法构建分层贝叶斯模型")
    
    n_subjects = data['n_subjects']
    n_stages = data['n_stages']
    n_features = data['n_features']
    
    print("\n" + "=" * 80)
    print("构建分层贝叶斯注意力模型")
    print("=" * 80)
    print(f"  被试数: {n_subjects}")
    print(f"  阶段数: {n_stages}")
    print(f"  特征数: {n_features}")
    print(f"  模型类型: {'简化版（线性+固定时间权重）' if use_simplified else '完整版（自适应时间注意力）'}")
    
    # 准备数据（转换为PyMC格式）
    X_list = []
    y_list = []
    subject_idx_list = []
    
    for s, subject_data in enumerate(data['subject_data']):
        X_stage = subject_data['X_stage']  # (n_trials, n_stages, n_features)
        y_choice = subject_data['y_choice']  # (n_trials,)
        
        for i in range(len(X_stage)):
            X_list.append(X_stage[i])
            y_list.append(y_choice[i])
            subject_idx_list.append(s)
    
    X_all = np.array(X_list)  # (total_trials, n_stages, n_features)
    y_all = np.array(y_list)  # (total_trials,)
    subject_idx = np.array(subject_idx_list)  # (total_trials,)
    
    print(f"\n  总试次数: {len(X_all)}")
    print(f"  选择分布: 选左={np.sum(y_all==0)} ({np.mean(y_all==0)*100:.1f}%), "
          f"选右={np.sum(y_all==1)} ({np.mean(y_all==1)*100:.1f}%)")
    
    # 标准化特征（每个特征维度）
    X_mean = X_all.mean(axis=(0, 1), keepdims=True)  # (1, 1, n_features)
    X_std = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_normalized = (X_all - X_mean) / X_std
    
    # 转换为PyMC共享变量
    X_shared = pm.Data('X_shared', X_normalized, mutable=True)
    y_shared = pm.Data('y_shared', y_all, mutable=True)
    subject_idx_shared = pm.Data('subject_idx_shared', subject_idx, mutable=True)
    
    with pm.Model() as hierarchical_model:
        # ========== 群体层（超先验） ==========
        print("\n定义群体层先验...")
        
        # 时间注意力权重（群体平均）
        # 使用Dirichlet先验（确保权重和为1）
        alpha_time_group = pm.HalfNormal('alpha_time_group', sigma=1.0, shape=(n_stages,))
        stage_weights_group = pm.Dirichlet('stage_weights_group', a=alpha_time_group)
        
        # 特征权重（群体平均）
        feature_weights_group = pm.Normal('feature_weights_group', 
                                         mu=0, 
                                         sigma=1.0, 
                                         shape=(n_features,))
        
        # 截距（群体平均）
        intercept_group = pm.Normal('intercept_group', mu=0, sigma=1.0)
        
        # 个体差异的方差
        sigma_time = pm.HalfNormal('sigma_time', sigma=0.3)
        sigma_feature = pm.HalfNormal('sigma_feature', sigma=0.5)
        sigma_intercept = pm.HalfNormal('sigma_intercept', sigma=0.3)
        
        # ========== 个体层 ==========
        print("定义个体层...")
        
        stage_weights_subj = []
        feature_weights_subj = []
        intercept_subj = []
        
        for s in range(n_subjects):
            # 个体时间注意力权重（围绕群体分布）
            # 使用Dirichlet分布的近似：log-normal + softmax
            stage_logits_subj = pm.Normal(
                f'stage_logits_subj_{s}',
                mu=pm.math.log(stage_weights_group + 1e-8),
                sigma=sigma_time,
                shape=(n_stages,)
            )
            stage_weights_subj.append(
                pm.Deterministic(
                    f'stage_weights_subj_{s}',
                    pt.softmax(stage_logits_subj)
                )
            )
            
            # 个体特征权重（围绕群体分布）
            feature_weights_subj.append(
                pm.Normal(
                    f'feature_weights_subj_{s}',
                    mu=feature_weights_group,
                    sigma=sigma_feature,
                    shape=(n_features,)
                )
            )
            
            # 个体截距
            intercept_subj.append(
                pm.Normal(
                    f'intercept_subj_{s}',
                    mu=intercept_group,
                    sigma=sigma_intercept
                )
            )
        
        # ========== 试次层 ==========
        print("定义试次层（构建似然函数）...")
        
        # 使用向量化操作（更高效）
        # 思路：将被试索引转换为权重矩阵，然后批量计算
        
        # 构建每个试次对应的被试权重矩阵
        # 更简单的方法：使用索引选择对应的权重
        
        # 由于PyMC的索引操作限制，我们使用更直接的方法
        # 将数据按被试分组处理会更高效，但这里使用循环（对于小样本可接受）
        
        logits_list = []
        
        # 分批处理以避免内存问题
        batch_size = 1000
        n_total = len(X_all)
        
        for batch_start in range(0, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            batch_idx = np.arange(batch_start, batch_end)
            
            batch_logits = []
            
            for i in batch_idx:
                s = subject_idx[i]
                X_trial = X_shared[i]  # (n_stages, n_features)
                
                # 时间加权求和
                weighted_stages = stage_weights_subj[s][:, None] * X_trial
                context = pt.sum(weighted_stages, axis=0)  # (n_features,)
                
                # 特征加权
                weighted_features = context * feature_weights_subj[s]
                feature_sum = pt.sum(weighted_features)
                
                # 最终logit
                logit = intercept_subj[s] + feature_sum
                batch_logits.append(logit)
            
            logits_list.append(pt.stack(batch_logits))
        
        logits_all = pt.concatenate(logits_list)  # (n_trials,)
        
        # 似然函数
        y_obs = pm.Bernoulli('y_obs', 
                            p=pm.math.sigmoid(logits_all),
                            observed=y_shared)
        
        print("\n模型构建完成！")
    
    return hierarchical_model


# ============================================================================
# MCMC采样和诊断
# ============================================================================
def fit_hierarchical_model(model: pm.Model, 
                          n_samples: int = 2000,
                          n_tune: int = 1000,
                          target_accept: float = 0.9,
                          cores: int = 1):
    """
    拟合分层贝叶斯模型
    
    Parameters:
    -----------
    model : pm.Model
        PyMC模型
    n_samples : int
        采样数
    n_tune : int
        预热采样数
    target_accept : float
        目标接受率
    cores : int
        CPU核心数
    
    Returns:
    --------
    trace : InferenceData
        MCMC采样结果
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC未安装")
    
    print("\n" + "=" * 80)
    print("开始MCMC采样")
    print("=" * 80)
    print(f"  采样数: {n_samples}")
    print(f"  预热数: {n_tune}")
    print(f"  目标接受率: {target_accept}")
    print(f"  CPU核心: {cores}")
    print("\n注意：这可能需要较长时间（数小时），请耐心等待...")
    
    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            target_accept=target_accept,
            cores=cores,
            random_seed=RNG_SEED,
            return_inferencedata=True,
            progressbar=True,
        )
    
    print("\n采样完成！")
    
    # 收敛诊断
    print("\n" + "=" * 80)
    print("收敛诊断")
    print("=" * 80)
    
    summary = az.summary(trace)
    
    # 检查R-hat
    rhat_high = summary[summary['r_hat'] > 1.01]
    if len(rhat_high) > 0:
        print(f"  警告: {len(rhat_high)} 个参数的R-hat > 1.01（可能未收敛）")
        print(f"  最高R-hat: {rhat_high['r_hat'].max():.4f}")
    else:
        print(f"  ✓ 所有参数的R-hat ≤ 1.01（收敛良好）")
    
    # 检查ESS
    ess_low = summary[summary['ess_bulk'] < 400]
    if len(ess_low) > 0:
        print(f"  警告: {len(ess_low)} 个参数的ESS < 400（可能采样不足）")
    else:
        print(f"  ✓ 所有参数的ESS ≥ 400（采样充足）")
    
    return trace


# ============================================================================
# 结果提取和可视化
# ============================================================================
def extract_hierarchical_results(trace: az.InferenceData, 
                                data: Dict,
                                output_dir: str):
    """
    提取和保存分层模型结果
    
    Parameters:
    -----------
    trace : InferenceData
        MCMC采样结果
    data : dict
        原始数据字典
    output_dir : str
        输出目录
    """
    print("\n" + "=" * 80)
    print("提取分层模型结果")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取后验摘要
    summary = az.summary(trace)
    summary.to_csv(os.path.join(output_dir, 'posterior_summary.csv'), 
                  encoding='utf-8-sig')
    
    # 提取群体参数
    print("\n提取群体参数...")
    group_params = {}
    
    # 时间注意力权重（群体）
    if 'stage_weights_group' in trace.posterior:
        stage_weights_group_samples = trace.posterior['stage_weights_group'].values
        # 形状: (n_chains, n_draws, n_stages)
        group_params['stage_weights_group'] = {
            'mean': np.mean(stage_weights_group_samples, axis=(0, 1)),
            'std': np.std(stage_weights_group_samples, axis=(0, 1)),
            'hdi_low': np.percentile(stage_weights_group_samples, 2.5, axis=(0, 1)),
            'hdi_high': np.percentile(stage_weights_group_samples, 97.5, axis=(0, 1)),
            'samples': stage_weights_group_samples,
        }
        print(f"  时间注意力权重（群体）:")
        for i, (mean, std) in enumerate(zip(
            group_params['stage_weights_group']['mean'],
            group_params['stage_weights_group']['std']
        )):
            print(f"    Stage {i+1}: {mean:.3f} ± {std:.3f}")
    
    # 特征权重（群体）
    if 'feature_weights_group' in trace.posterior:
        feature_weights_group_samples = trace.posterior['feature_weights_group'].values
        group_params['feature_weights_group'] = {
            'mean': np.mean(feature_weights_group_samples, axis=(0, 1)),
            'std': np.std(feature_weights_group_samples, axis=(0, 1)),
            'hdi_low': np.percentile(feature_weights_group_samples, 2.5, axis=(0, 1)),
            'hdi_high': np.percentile(feature_weights_group_samples, 97.5, axis=(0, 1)),
        }
        print(f"  特征权重（群体）:")
        for i, (mean, std) in enumerate(zip(
            group_params['feature_weights_group']['mean'][:5],  # 只显示前5个
            group_params['feature_weights_group']['std'][:5]
        )):
            print(f"    Feature {i+1}: {mean:.3f} ± {std:.3f}")
    
    # 提取个体参数
    print("\n提取个体参数...")
    subject_params = []
    
    for s in range(data['n_subjects']):
        subject_id = data['subject_data'][s]['subject_id']
        subj_params = {'subject_id': subject_id}
        
        # 个体时间注意力权重
        stage_key = f'stage_weights_subj_{s}'
        if stage_key in trace.posterior:
            stage_samples = trace.posterior[stage_key].values
            subj_params['stage_weights'] = {
                'mean': np.mean(stage_samples, axis=(0, 1)),
                'std': np.std(stage_samples, axis=(0, 1)),
            }
        
        # 个体特征权重
        feat_key = f'feature_weights_subj_{s}'
        if feat_key in trace.posterior:
            feat_samples = trace.posterior[feat_key].values
            subj_params['feature_weights'] = {
                'mean': np.mean(feat_samples, axis=(0, 1)),
                'std': np.std(feat_samples, axis=(0, 1)),
            }
        
        # 个体截距
        intercept_key = f'intercept_subj_{s}'
        if intercept_key in trace.posterior:
            intercept_samples = trace.posterior[intercept_key].values
            subj_params['intercept'] = {
                'mean': float(np.mean(intercept_samples)),
                'std': float(np.std(intercept_samples)),
            }
        
        subject_params.append(subj_params)
    
    # 保存结果
    print("\n保存结果...")
    
    # 群体参数
    if 'stage_weights_group' in group_params:
        stage_df = pd.DataFrame({
            'stage': [f'Stage {i+1}' for i in range(len(group_params['stage_weights_group']['mean']))],
            'mean': group_params['stage_weights_group']['mean'],
            'std': group_params['stage_weights_group']['std'],
            'hdi_low': group_params['stage_weights_group']['hdi_low'],
            'hdi_high': group_params['stage_weights_group']['hdi_high'],
        })
        stage_df.to_csv(os.path.join(output_dir, 'group_stage_weights.csv'), 
                       index=False, encoding='utf-8-sig')
    
    # 个体参数汇总
    subject_summary = []
    for subj_params in subject_params:
        row = {'subject_id': subj_params['subject_id']}
        if 'stage_weights' in subj_params:
            for i, mean in enumerate(subj_params['stage_weights']['mean']):
                row[f'stage_weight_{i+1}'] = mean
        if 'intercept' in subj_params:
            row['intercept'] = subj_params['intercept']['mean']
        subject_summary.append(row)
    
    if subject_summary:
        subject_df = pd.DataFrame(subject_summary)
        subject_df.to_csv(os.path.join(output_dir, 'subject_parameters.csv'), 
                         index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到: {output_dir}")
    
    return group_params, subject_params


# ============================================================================
# 交叉验证评估函数（与individual_model_comparison.py保持一致）
# ============================================================================
def cross_validate_hierarchical_model(data: Dict,
                                     project_root: str,
                                     base_dir: str,
                                     n_folds: int = 5,
                                     cv_seed: int = 42,
                                     n_samples: int = 2000,
                                     n_tune: int = 1000,
                                     output_dir: str = None):
    """
    对分层贝叶斯模型进行5折交叉验证评估
    
    参数:
    -----
    data : dict
        分层数据字典
    n_folds : int
        折数（默认5）
    cv_seed : int
        交叉验证随机种子
    n_samples : int
        MCMC采样数
    n_tune : int
        预热采样数
    output_dir : str
        输出目录
    
    Returns:
    --------
    cv_results : dict
        交叉验证结果
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC未安装")
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    print("\n" + "=" * 80)
    print(f"{n_folds}折交叉验证评估分层贝叶斯模型")
    print("=" * 80)
    
    # 准备所有试次的数据（用于划分）
    X_all_list = []
    y_all_list = []
    subject_idx_all = []
    
    for s, subject_data in enumerate(data['subject_data']):
        X_stage = subject_data['X_stage']
        y_choice = subject_data['y_choice']
        
        for i in range(len(X_stage)):
            X_all_list.append(X_stage[i])
            y_all_list.append(y_choice[i])
            subject_idx_all.append(s)
    
    X_all = np.array(X_all_list)  # (total_trials, n_stages, n_features)
    y_all = np.array(y_all_list)  # (total_trials,)
    subject_idx_all = np.array(subject_idx_all)  # (total_trials,)
    
    n_total = len(X_all)
    print(f"  总试次数: {n_total}")
    
    # 构建5折（使用与individual_model_comparison.py一致的划分方式）
    from linear_regression_selected_features import build_consistent_folds
    folds, indices = build_consistent_folds(n_total, seed=cv_seed, k=n_folds)
    
    # 存储每折的结果
    fold_accuracies = []
    fold_aucs = []
    fold_results = []
    
    for fold_idx, test_idx in enumerate(folds, 1):
        print(f"\n{'=' * 80}")
        print(f"折 {fold_idx}/{n_folds}")
        print(f"{'=' * 80}")
        
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        
        print(f"  训练集: {len(train_idx)} 个试次")
        print(f"  测试集: {len(test_idx)} 个试次")
        
        # 分割数据
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        subject_idx_train = subject_idx_all[train_idx]
        subject_idx_test = subject_idx_all[test_idx]
        
        # 重建训练数据的被试结构（用于分层模型）
        train_data = _rebuild_subject_data_structure(
            X_train, y_train, subject_idx_train, data
        )
        
        # 构建模型（只使用训练数据）
        try:
            model = build_hierarchical_attention_model(train_data, use_simplified=True)
        except Exception as e:
            print(f"  错误: 构建模型失败: {e}")
            continue
        
        # MCMC采样（使用较少的采样数以加快速度）
        print(f"  开始MCMC采样（折 {fold_idx}）...")
        try:
            with model:
                trace = pm.sample(
                    draws=n_samples,
                    tune=n_tune,
                    target_accept=0.9,
                    cores=1,
                    random_seed=RNG_SEED + fold_idx,
                    return_inferencedata=True,
                    progressbar=True,
                )
        except Exception as e:
            print(f"  错误: MCMC采样失败: {e}")
            continue
        
        # 在测试集上预测
        print(f"  在测试集上预测...")
        test_predictions = _predict_on_test_set(
            trace, model, X_test, y_test, subject_idx_test, train_data
        )
        
        if test_predictions is None:
            continue
        
        # 计算指标
        y_pred = test_predictions['y_pred']
        y_pred_proba = test_predictions['y_pred_proba']
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = np.nan
        
        baseline_accuracy = max(np.mean(y_test), 1 - np.mean(y_test))
        
        fold_accuracies.append(accuracy)
        fold_aucs.append(auc)
        fold_results.append({
            'fold': fold_idx,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_accuracy,
            'improvement': accuracy - baseline_accuracy,
            'auc': auc,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })
        
        print(f"  折 {fold_idx} 结果:")
        print(f"    准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"    基线准确率: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
        print(f"    提升: {(accuracy - baseline_accuracy)*100:.2f}%")
        print(f"    AUC: {auc:.4f}")
    
    # 计算平均结果
    cv_results = {
        'fold_results': fold_results,
        'cv_accuracy_mean': np.mean(fold_accuracies),
        'cv_accuracy_std': np.std(fold_accuracies),
        'cv_auc_mean': np.nanmean(fold_aucs),
        'cv_auc_std': np.nanstd(fold_aucs),
        'n_folds': len(fold_results),
    }
    
    print(f"\n{'=' * 80}")
    print(f"交叉验证总结")
    print(f"{'=' * 80}")
    print(f"  平均准确率: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
    print(f"  平均AUC: {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv_df = pd.DataFrame(fold_results)
        cv_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), 
                    index=False, encoding='utf-8-sig')
        
        summary_df = pd.DataFrame([{
            'cv_accuracy_mean': cv_results['cv_accuracy_mean'],
            'cv_accuracy_std': cv_results['cv_accuracy_std'],
            'cv_auc_mean': cv_results['cv_auc_mean'],
            'cv_auc_std': cv_results['cv_auc_std'],
            'n_folds': cv_results['n_folds'],
        }])
        summary_df.to_csv(os.path.join(output_dir, 'cross_validation_summary.csv'), 
                         index=False, encoding='utf-8-sig')
    
    return cv_results


def _rebuild_subject_data_structure(X, y, subject_idx, original_data):
    """从试次数据重建被试数据结构"""
    train_data = {
        'n_subjects': original_data['n_subjects'],
        'n_stages': original_data['n_stages'],
        'n_features': original_data['n_features'],
        'subject_data': [],
        'subject_ids': original_data['subject_ids'],
    }
    
    # 按被试分组
    for s in range(original_data['n_subjects']):
        mask = (subject_idx == s)
        if np.sum(mask) == 0:
            continue
        
        X_subj = X[mask]
        y_subj = y[mask]
        
        train_data['subject_data'].append({
            'subject_id': original_data['subject_ids'][s],
            'X_stage': X_subj,
            'y_choice': y_subj,
            'n_trials': len(X_subj),
        })
    
    train_data['n_subjects'] = len(train_data['subject_data'])
    
    return train_data


def _predict_on_test_set(trace, model, X_test, y_test, subject_idx_test, train_data):
    """在测试集上使用训练好的模型进行预测"""
    if not PYMC_AVAILABLE:
        return None
    
    try:
        # 标准化（使用训练数据的统计量）
        # 这里简化处理，使用测试数据本身的统计量
        X_mean = X_test.mean(axis=(0, 1), keepdims=True)
        X_std = X_test.std(axis=(0, 1), keepdims=True) + 1e-8
        X_test_norm = (X_test - X_mean) / X_std
        
        # 提取后验参数的均值
        # 对于测试集，我们需要使用训练集的被试参数
        # 简化：使用群体参数进行预测
        
        if 'stage_weights_group' not in trace.posterior:
            return None
        
        # 提取群体参数均值
        stage_weights_group = trace.posterior['stage_weights_group'].values
        stage_weights_mean = np.mean(stage_weights_group, axis=(0, 1))  # (n_stages,)
        
        feature_weights_group = trace.posterior['feature_weights_group'].values
        feature_weights_mean = np.mean(feature_weights_group, axis=(0, 1))  # (n_features,)
        
        intercept_group = trace.posterior['intercept_group'].values
        intercept_mean = float(np.mean(intercept_group))
        
        # 对每个测试试次计算预测
        logits = []
        for i in range(len(X_test)):
            X_trial = X_test_norm[i]  # (n_stages, n_features)
            
            # 时间加权求和
            weighted_stages = stage_weights_mean[:, None] * X_trial
            context = np.sum(weighted_stages, axis=0)  # (n_features,)
            
            # 特征加权
            weighted_features = context * feature_weights_mean
            feature_sum = np.sum(weighted_features)
            
            # 最终logit
            logit = intercept_mean + feature_sum
            logits.append(logit)
        
        logits = np.array(logits)
        y_pred_proba = 1 / (1 + np.exp(-logits))  # sigmoid
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }
    except Exception as e:
        print(f"  预测失败: {e}")
        return None


# ============================================================================
# 模型评估函数
# ============================================================================
def evaluate_model_performance(trace: az.InferenceData,
                               data: Dict,
                               model: pm.Model,
                               output_dir: str):
    """
    评估模型性能：预测准确率、模型比较等
    
    Returns:
    --------
    eval_results : dict
        评估结果字典
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC未安装")
    
    print("\n" + "=" * 80)
    print("评估模型性能")
    print("=" * 80)
    
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    
    os.makedirs(output_dir, exist_ok=True)
    eval_results = {}
    
    # 1. 后验预测检查（Posterior Predictive Check）
    print("\n1. 后验预测检查...")
    
    with model:
        # 生成后验预测
        ppc = pm.sample_posterior_predictive(trace, progressbar=True, random_seed=RNG_SEED)
    
    # 提取预测
    y_pred_samples = ppc.posterior_predictive['y_obs'].values
    # 形状: (n_chains, n_draws, n_trials)
    
    # 计算预测概率（所有样本的平均）
    y_pred_proba = np.mean(y_pred_samples, axis=(0, 1))  # (n_trials,)
    
    # 提取真实值
    y_true_all = []
    for subject_data in data['subject_data']:
        y_true_all.extend(subject_data['y_choice'])
    y_true = np.array(y_true_all)
    
    # 预测（基于概率>0.5）
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan
    
    baseline_accuracy = max(np.mean(y_true), 1 - np.mean(y_true))
    
    eval_results['prediction'] = {
        'accuracy': accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': accuracy - baseline_accuracy,
        'auc': auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }
    
    print(f"  预测准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  基线准确率: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"  提升: {(accuracy - baseline_accuracy)*100:.2f}%")
    print(f"  AUC: {auc:.4f}")
    
    # 2. 模型比较（WAIC, LOO）
    print("\n2. 模型比较指标...")
    
    try:
        loo = az.loo(trace)
        waic = az.waic(trace)
        
        eval_results['model_comparison'] = {
            'loo': float(loo.loo),
            'loo_se': float(loo.loo_se),
            'p_loo': float(loo.p_loo),
            'waic': float(waic.waic),
            'waic_se': float(waic.waic_se),
        }
        
        print(f"  LOO: {loo.loo:.2f} ± {loo.loo_se:.2f}")
        print(f"  WAIC: {waic.waic:.2f} ± {waic.waic_se:.2f}")
        print(f"  p_loo (有效参数数): {loo.p_loo:.2f}")
    except Exception as e:
        print(f"  警告: 模型比较计算失败: {e}")
        eval_results['model_comparison'] = None
    
    # 3. 个体差异量化
    print("\n3. 量化个体差异...")
    
    if 'sigma_time' in trace.posterior and 'sigma_feature' in trace.posterior:
        sigma_time_samples = trace.posterior['sigma_time'].values.flatten()
        sigma_feature_samples = trace.posterior['sigma_feature'].values.flatten()
        
        eval_results['individual_differences'] = {
            'sigma_time': {
                'mean': float(np.mean(sigma_time_samples)),
                'std': float(np.std(sigma_time_samples)),
                'hdi_low': float(np.percentile(sigma_time_samples, 2.5)),
                'hdi_high': float(np.percentile(sigma_time_samples, 97.5)),
            },
            'sigma_feature': {
                'mean': float(np.mean(sigma_feature_samples)),
                'std': float(np.std(sigma_feature_samples)),
                'hdi_low': float(np.percentile(sigma_feature_samples, 2.5)),
                'hdi_high': float(np.percentile(sigma_feature_samples, 97.5)),
            },
        }
        
        print(f"  时间注意力个体差异 (σ_time): "
              f"{eval_results['individual_differences']['sigma_time']['mean']:.3f} "
              f"[{eval_results['individual_differences']['sigma_time']['hdi_low']:.3f}, "
              f"{eval_results['individual_differences']['sigma_time']['hdi_high']:.3f}]")
        print(f"  特征权重个体差异 (σ_feature): "
              f"{eval_results['individual_differences']['sigma_feature']['mean']:.3f} "
              f"[{eval_results['individual_differences']['sigma_feature']['hdi_low']:.3f}, "
              f"{eval_results['individual_differences']['sigma_feature']['hdi_high']:.3f}]")
    
    # 4. 被试级别的预测准确率
    print("\n4. 被试级别的预测准确率...")
    
    subject_accuracies = []
    trial_idx = 0
    
    for s, subject_data in enumerate(data['subject_data']):
        n_trials = subject_data['n_trials']
        y_true_subj = subject_data['y_choice']
        
        y_pred_subj = y_pred[trial_idx:trial_idx + n_trials]
        y_pred_proba_subj = y_pred_proba[trial_idx:trial_idx + n_trials]
        
        acc_subj = accuracy_score(y_true_subj, y_pred_subj)
        try:
            auc_subj = roc_auc_score(y_true_subj, y_pred_proba_subj)
        except:
            auc_subj = np.nan
        
        subject_accuracies.append({
            'subject_id': subject_data['subject_id'],
            'accuracy': acc_subj,
            'auc': auc_subj,
            'n_trials': n_trials,
        })
        
        print(f"  {subject_data['subject_id']}: "
              f"准确率={acc_subj:.3f}, AUC={auc_subj:.3f}, "
              f"试次数={n_trials}")
        
        trial_idx += n_trials
    
    eval_results['subject_performance'] = subject_accuracies
    
    # 保存评估结果
    eval_df = pd.DataFrame({
        'metric': ['Overall Accuracy', 'Baseline Accuracy', 'Improvement', 'AUC'],
        'value': [accuracy, baseline_accuracy, accuracy - baseline_accuracy, auc]
    })
    eval_df.to_csv(os.path.join(output_dir, 'model_performance.csv'), 
                  index=False, encoding='utf-8-sig')
    
    subject_perf_df = pd.DataFrame(subject_accuracies)
    subject_perf_df.to_csv(os.path.join(output_dir, 'subject_performance.csv'), 
                          index=False, encoding='utf-8-sig')
    
    print(f"\n评估结果已保存到: {output_dir}")
    
    return eval_results


def visualize_model_evaluation(eval_results: Dict,
                               group_params: Dict,
                               subject_params: List[Dict],
                               output_dir: str):
    """
    可视化模型评估结果
    """
    print("\n" + "=" * 80)
    print("生成评估可视化图表")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 预测准确率图
    if 'prediction' in eval_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 预测 vs 真实
        ax = axes[0]
        y_true = eval_results['prediction']['y_true']
        y_pred_proba = eval_results['prediction']['y_pred_proba']
        
        # 按真实标签分组
        y_pred_left = y_pred_proba[y_true == 0]
        y_pred_right = y_pred_proba[y_true == 1]
        
        ax.hist(y_pred_left, bins=30, alpha=0.6, label='实际选左', density=True)
        ax.hist(y_pred_right, bins=30, alpha=0.6, label='实际选右', density=True)
        ax.axvline(0.5, color='r', linestyle='--', linewidth=2, label='决策阈值')
        ax.set_xlabel('预测概率（选右）', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_title(f'预测概率分布\n准确率: {eval_results["prediction"]["accuracy"]:.3f}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ROC曲线（如果可能）
        ax = axes[1]
        if not np.isnan(eval_results['prediction']['auc']):
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'ROC (AUC = {eval_results["prediction"]["auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
            ax.set_xlabel('假阳性率', fontsize=12)
            ax.set_ylabel('真阳性率', fontsize=12)
            ax.set_title('ROC曲线', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 被试准确率对比
    if 'subject_performance' in eval_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        subject_perf = eval_results['subject_performance']
        subject_ids = [s['subject_id'] for s in subject_perf]
        accuracies = [s['accuracy'] for s in subject_perf]
        
        bars = ax.barh(subject_ids, accuracies, alpha=0.7)
        ax.axvline(eval_results['prediction']['accuracy'], 
                  color='r', linestyle='--', linewidth=2, 
                  label=f'总体准确率 ({eval_results["prediction"]["accuracy"]:.3f})')
        ax.set_xlabel('预测准确率', fontsize=12)
        ax.set_ylabel('被试', fontsize=12)
        ax.set_title('被试级别的预测准确率', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subject_accuracy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 个体差异可视化
    if 'individual_differences' in eval_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 时间注意力个体差异
        ax = axes[0]
        sigma_time = eval_results['individual_differences']['sigma_time']
        ax.barh(['σ_time'], [sigma_time['mean']], 
               xerr=[[sigma_time['mean'] - sigma_time['hdi_low']], 
                     [sigma_time['hdi_high'] - sigma_time['mean']]],
               alpha=0.7, capsize=5)
        ax.set_xlabel('个体差异标准差', fontsize=12)
        ax.set_title('时间注意力个体差异', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 特征权重个体差异
        ax = axes[1]
        sigma_feature = eval_results['individual_differences']['sigma_feature']
        ax.barh(['σ_feature'], [sigma_feature['mean']],
               xerr=[[sigma_feature['mean'] - sigma_feature['hdi_low']],
                     [sigma_feature['hdi_high'] - sigma_feature['mean']]],
               alpha=0.7, capsize=5)
        ax.set_xlabel('个体差异标准差', fontsize=12)
        ax.set_title('特征权重个体差异', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_differences.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 群体 vs 个体参数对比
    if group_params and subject_params:
        if 'stage_weights_group' in group_params and 'stage_weights' in subject_params[0]:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            n_stages = len(group_params['stage_weights_group']['mean'])
            stages = [f'Stage {i+1}' for i in range(n_stages)]
            
            # 群体均值
            group_means = group_params['stage_weights_group']['mean']
            ax.plot(stages, group_means, 'o-', linewidth=3, markersize=10,
                   label='群体平均', color='red', zorder=5)
            
            # 个体均值（带误差）
            x_pos = np.arange(len(stages))
            width = 0.15
            
            for s, subj_params in enumerate(subject_params[:5]):  # 只显示前5个被试
                if 'stage_weights' in subj_params:
                    means = subj_params['stage_weights']['mean']
                    stds = subj_params['stage_weights']['std']
                    
                    offset = (s - 2) * width
                    ax.errorbar(x_pos + offset, means, yerr=stds, 
                              fmt='o', alpha=0.6, capsize=3,
                              label=subj_params.get('subject_id', f'Subj {s+1}'))
            
            ax.set_xlabel('时间阶段', fontsize=12)
            ax.set_ylabel('注意力权重', fontsize=12)
            ax.set_title('群体 vs 个体时间注意力权重对比', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'group_vs_individual_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\n评估可视化图表已保存到: {output_dir}")


def visualize_hierarchical_results(trace: az.InferenceData,
                                   data: Dict,
                                   group_params: Dict,
                                   subject_params: List[Dict],
                                   output_dir: str):
    """
    可视化分层模型结果
    """
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 群体时间注意力权重
    if 'stage_weights_group' in group_params:
        fig, ax = plt.subplots(figsize=(10, 6))
        means = group_params['stage_weights_group']['mean']
        stds = group_params['stage_weights_group']['std']
        stages = [f'Stage {i+1}' for i in range(len(means))]
        
        ax.bar(stages, means, yerr=stds, alpha=0.7, capsize=5)
        ax.set_xlabel('时间阶段', fontsize=12)
        ax.set_ylabel('注意力权重', fontsize=12)
        ax.set_title('群体时间注意力权重（后验均值±标准差）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'group_stage_attention.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 个体时间注意力权重对比
    if subject_params and 'stage_weights' in subject_params[0]:
        n_subjects = len(subject_params)
        n_stages = len(subject_params[0]['stage_weights']['mean'])
        
        fig, axes = plt.subplots(n_subjects, 1, figsize=(10, 2*n_subjects), 
                                sharex=True, sharey=True)
        if n_subjects == 1:
            axes = [axes]
        
        for s, (subj_params, ax) in enumerate(zip(subject_params, axes)):
            means = subj_params['stage_weights']['mean']
            stds = subj_params['stage_weights']['std']
            stages = [f'S{i+1}' for i in range(len(means))]
            
            ax.bar(stages, means, yerr=stds, alpha=0.7, capsize=3)
            ax.set_ylabel(subj_params['subject_id'], fontsize=10, rotation=0, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            if s == n_subjects - 1:
                ax.set_xlabel('时间阶段', fontsize=12)
        
        axes[0].set_title('个体时间注意力权重对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subject_stage_attention_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 后验分布图（示例参数）
    if 'stage_weights_group' in trace.posterior:
        try:
            az.plot_posterior(trace, var_names=['stage_weights_group'],
                            figsize=(12, 8))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'posterior_distributions.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except:
            print("  警告: 后验分布图生成失败（可能参数太多）")
    
    print(f"\n可视化图表已保存到: {output_dir}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数：运行完整的分层贝叶斯分析"""
    if not PYMC_AVAILABLE:
        print("错误: PyMC未安装，无法运行分层贝叶斯分析")
        print("请安装: pip install pymc arviz")
        return
    
    print("=" * 80)
    print("分层贝叶斯注意力模型分析")
    print("=" * 80)
    
    project_root = os.getcwd()
    base_dir = r"D:\D\research\audioset下载\clap_select"
    output_root = os.path.join(project_root, 'hierarchical_bayesian_attention_results')
    os.makedirs(output_root, exist_ok=True)
    
    # 步骤1: 准备数据
    data = prepare_hierarchical_data(SUBJECT_CONFIGS, project_root, base_dir)
    
    if data['n_subjects'] == 0:
        print("错误: 没有成功加载任何被试数据")
        return
    
    # 步骤2: 构建模型
    model = build_hierarchical_attention_model(data, use_simplified=True)
    
    # 步骤3: MCMC采样
    trace = fit_hierarchical_model(model, 
                                   n_samples=2000,
                                   n_tune=1000,
                                   cores=1)
    
    # 步骤4: 提取结果
    group_params, subject_params = extract_hierarchical_results(
        trace, data, output_root
    )
    
    # 步骤5: 5折交叉验证评估（与individual_model_comparison.py一致）
    # 注意：5折交叉验证需要训练5个模型，可能需要数小时
    # 可以通过设置 RUN_CROSS_VALIDATION = False 来跳过
    RUN_CROSS_VALIDATION = True
    
    cv_results = None
    if RUN_CROSS_VALIDATION:
        print("\n" + "=" * 80)
        print("开始5折交叉验证评估")
        print("=" * 80)
        print("注意：5折交叉验证需要训练5个模型，可能需要很长时间...")
        print("     每折将使用较少的MCMC采样数以加快速度")
        
        try:
            cv_results = cross_validate_hierarchical_model(
                data,
                project_root,
                base_dir,
                n_folds=5,
                cv_seed=42,  # 与individual_model_comparison.py一致
                n_samples=1000,  # CV时使用较少的采样数以加快速度
                n_tune=500,
                output_dir=output_root
            )
        except Exception as e:
            print(f"  警告: 交叉验证失败: {e}")
            print(f"  将继续使用完整数据评估...")
            cv_results = None
    else:
        print("\n跳过5折交叉验证（RUN_CROSS_VALIDATION = False）")
    
    # 步骤6: 完整模型评估（可选，用于对比）
    print("\n" + "=" * 80)
    print("完整模型评估（所有数据）")
    print("=" * 80)
    eval_results = evaluate_model_performance(
        trace, data, model, output_root
    )
    
    # 步骤7: 可视化
    visualize_hierarchical_results(
        trace, data, group_params, subject_params, output_root
    )
    
    # 步骤7: 可视化评估结果
    visualize_model_evaluation(
        eval_results, group_params, subject_params, output_root
    )
    
    # 保存trace（用于后续分析）
    trace_file = os.path.join(output_root, 'trace.nc')
    trace.to_netcdf(trace_file)
    print(f"\nMCMC采样结果已保存到: {trace_file}")
    print("可以使用 arviz 重新加载进行分析: trace = az.from_netcdf('trace.nc')")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("模型效果总结")
    print("=" * 80)
    
    # 优先显示交叉验证结果（与individual_model_comparison.py一致）
    if cv_results is not None:
        print(f"\n交叉验证结果（5折）:")
        print(f"  CV准确率: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
        print(f"  CV AUC: {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
    
    print(f"\n完整数据预测性能:")
    print(f"  总体准确率: {eval_results['prediction']['accuracy']:.4f}")
    print(f"  AUC: {eval_results['prediction']['auc']:.4f}")
    print(f"  相比基线提升: {eval_results['prediction']['improvement']*100:.2f}%")
    
    if eval_results['model_comparison']:
        print(f"\n模型拟合度:")
        print(f"  WAIC: {eval_results['model_comparison']['waic']:.2f} ± {eval_results['model_comparison']['waic_se']:.2f}")
        print(f"  LOO: {eval_results['model_comparison']['loo']:.2f} ± {eval_results['model_comparison']['loo_se']:.2f}")
    
    if 'individual_differences' in eval_results:
        print(f"\n个体差异:")
        print(f"  时间注意力差异: σ = {eval_results['individual_differences']['sigma_time']['mean']:.3f}")
        print(f"  特征权重差异: σ = {eval_results['individual_differences']['sigma_feature']['mean']:.3f}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    
    # 保存包含CV结果的最终总结
    if cv_results is not None:
        final_summary = pd.DataFrame([{
            'cv_accuracy_mean': cv_results['cv_accuracy_mean'],
            'cv_accuracy_std': cv_results['cv_accuracy_std'],
            'cv_auc_mean': cv_results['cv_auc_mean'],
            'cv_auc_std': cv_results['cv_auc_std'],
            'full_data_accuracy': eval_results['prediction']['accuracy'],
            'full_data_auc': eval_results['prediction']['auc'],
            'baseline_accuracy': eval_results['prediction']['baseline_accuracy'],
            'improvement': eval_results['prediction']['improvement'],
        }])
        final_summary.to_csv(os.path.join(output_root, 'final_summary_with_cv.csv'), 
                            index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()

