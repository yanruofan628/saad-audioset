#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jiachen个人数据验证：验证特征是否能预测个人选择和反应时
- 检查左右平衡一致性（AB和BA必须选择同一个音频）
- 合并成144个独立对
- 拟合个人选择（二分类）
- 拟合反应时（线性回归）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

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
    extract_significant_features_from_table,
    build_consistent_folds
)

CV_SEED = 42
N_FOLDS = 5


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
    print(f"  保留比例: {len(consistent_trials)/(len(consistent_trials)+inconsistent_count)*100:.1f}%")
    
    return consistent_trials


def extract_features_for_trials(consistent_trials, pair_names, feature_indices_0based):
    """
    为一致的试次提取特征
    
    Parameters:
    -----------
    consistent_trials : list
        一致的试次数据
    pair_names : list
        所有音频对名称（288个）
    feature_indices_0based : list
        特征索引（从0开始）
        
    Returns:
    --------
    X : np.ndarray, shape (n_trials, n_features)
        特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    y_rt : np.ndarray, shape (n_trials,)
        反应时（秒）
    valid_mask : np.ndarray
        有效数据掩码
    """
    print("\n=== 提取特征 ===")
    
    # 加载特征数据
    project_root = os.getcwd()
    diff_raw, diff_zscore, all_pair_names = load_or_extract_features(project_root)
    
    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(all_pair_names)}
    
    # 提取特征
    X_list = []
    y_choice_list = []
    y_rt_list = []
    valid_indices = []
    
    for i, trial in enumerate(consistent_trials):
        pair_name = trial['pair_name']
        
        if pair_name not in pair_to_idx:
            continue
        
        idx = pair_to_idx[pair_name]
        
        # 提取特征（使用z-score归一化后的值）
        features = diff_zscore[feature_indices_0based, idx]
        
        # 检查是否有NaN
        if np.isnan(features).any():
            continue
        
        X_list.append(features)
        y_choice_list.append(trial['choice'])
        
        # RT可能为None
        rt = trial.get('reaction_time')
        if rt is None:
            rt = np.nan
        y_rt_list.append(rt)
        valid_indices.append(i)
    
    X = np.array(X_list)
    y_choice = np.array(y_choice_list)
    y_rt = np.array(y_rt_list)
    
    print(f"  提取了 {len(X)} 个有效试次的特征")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  选左: {np.sum(y_choice == 1)} ({np.mean(y_choice == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(y_choice == 2)} ({np.mean(y_choice == 2)*100:.1f}%)")
    print(f"  有RT数据: {np.sum(~np.isnan(y_rt))} ({np.sum(~np.isnan(y_rt))/len(y_rt)*100:.1f}%)")
    
    return X, y_choice, y_rt


def fit_choice_model(X, y_choice, feature_names):
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
        
    Returns:
    --------
    results : dict
        拟合结果
    """
    print("\n=== 拟合个人选择模型（逻辑回归） ===")
    
    # 转换为二分类：1=选左 -> 0, 2=选右 -> 1
    y_binary = (y_choice == 2).astype(int)
    
    print(f"  数据统计:")
    print(f"    选左（0）: {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"    选右（1）: {np.sum(y_binary == 1)} ({np.mean(y_binary == 1)*100:.1f}%)")
    
    # 使用statsmodels逻辑回归（有统计信息）
    X_with_const = sm.add_constant(X)
    logit_model = sm.Logit(y_binary, X_with_const).fit(disp=0)
    
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
    
    # 系数信息（所有特征的权重）
    print(f"\n  系数信息（所有特征的权重）:")
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
        'coefficients': coefficients,  # 所有特征的权重
        'feature_names': feature_names
    }


def fit_rt_model(X, y_rt, feature_names):
    """
    拟合反应时模型（线性回归）
    
    Parameters:
    -----------
    X : np.ndarray
        特征矩阵
    y_rt : np.ndarray
        反应时（秒）
    feature_names : list
        特征名称列表
        
    Returns:
    --------
    results : dict
        拟合结果
    """
    print("\n=== 拟合反应时模型（线性回归） ===")
    
    # 去除NaN值
    valid_mask = ~np.isnan(y_rt)
    X_valid = X[valid_mask]
    y_rt_valid = y_rt[valid_mask]
    
    print(f"  有效RT数据: {len(y_rt_valid)}/{len(y_rt)}")
    print(f"  RT范围: [{np.min(y_rt_valid):.3f}, {np.max(y_rt_valid):.3f}] 秒")
    print(f"  RT均值: {np.mean(y_rt_valid):.3f} 秒")
    print(f"  RT标准差: {np.std(y_rt_valid):.3f} 秒")
    
    if len(y_rt_valid) < 10:
        print("  警告: 有效RT数据太少，跳过拟合")
        return None
    
    # 使用statsmodels线性回归
    X_with_const = sm.add_constant(X_valid)
    ols_model = sm.OLS(y_rt_valid, X_with_const).fit()
    
    print(f"\n  模型统计:")
    print(f"    R2: {ols_model.rsquared:.4f}")
    print(f"    调整R2: {ols_model.rsquared_adj:.4f}")
    print(f"    AIC: {ols_model.aic:.4f}")
    print(f"    BIC: {ols_model.bic:.4f}")
    
    # 预测
    y_pred = ols_model.predict(X_with_const)
    
    # 评估指标
    r2 = r2_score(y_rt_valid, y_pred)
    mae = mean_absolute_error(y_rt_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_rt_valid, y_pred))
    corr = pearsonr(y_rt_valid, y_pred)[0]
    
    print(f"\n  预测指标:")
    print(f"    R2: {r2:.4f}")
    print(f"    MAE: {mae:.4f} 秒")
    print(f"    RMSE: {rmse:.4f} 秒")
    print(f"    相关系数: {corr:.4f}")
    
    # 交叉验证
    print(f"\n  5折交叉验证:")
    folds, indices = build_consistent_folds(len(y_rt_valid), seed=CV_SEED, k=N_FOLDS)
    
    cv_r2_scores = []
    cv_mae_scores = []
    
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_rt_valid[train_idx], y_rt_valid[test_idx]
        
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        model = sm.OLS(y_train, X_train_const).fit()
        y_pred_cv = model.predict(X_test_const)
        
        cv_r2_scores.append(r2_score(y_test, y_pred_cv))
        cv_mae_scores.append(mean_absolute_error(y_test, y_pred_cv))
    
    print(f"    CV R2: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"    CV MAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")
    
    # 系数信息（所有特征的权重）
    print(f"\n  系数信息（所有特征的权重）:")
    print(f"    截距: {ols_model.params[0]:.4f}, p={ols_model.pvalues[0]:.4f}")
    
    # 保存所有系数
    coefficients = []
    for i, name in enumerate(feature_names):
        coef = ols_model.params[i+1]
        pval = ols_model.pvalues[i+1]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"    {name}: {coef:.4f}, p={pval:.4f} {sig}")
        coefficients.append({
            'feature': name,
            'coefficient': coef,
            'pvalue': pval,
            'significant': sig
        })
    
    return {
        'model': ols_model,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': corr,
        'cv_r2_mean': np.mean(cv_r2_scores),
        'cv_r2_std': np.std(cv_r2_scores),
        'cv_mae_mean': np.mean(cv_mae_scores),
        'cv_mae_std': np.std(cv_mae_scores),
        'y_true': y_rt_valid,
        'y_pred': y_pred,
        'valid_mask': valid_mask,
        'coefficients': coefficients,  # 所有特征的权重
        'feature_names': feature_names
    }


def plot_results(choice_results, rt_results, output_dir, feature_names):
    """绘制结果图表"""
    print("\n=== 生成可视化图表 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 选择预测：预测概率vs实际
    if choice_results is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        y_true = choice_results['y_true']
        y_pred_proba = choice_results['y_pred_proba']
        
        # 按实际类别分组绘制
        for label in [0, 1]:
            mask = y_true == label
            ax.hist(y_pred_proba[mask], bins=20, alpha=0.6, 
                   label=f'实际{"左" if label==0 else "右"}', density=True)
        
        ax.set_xlabel('预测概率（选右）', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_title(f'选择预测概率分布\n准确率={choice_results["accuracy"]:.4f}, AUC={choice_results["auc"]:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'choice_prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. RT预测：预测vs实际
    if rt_results is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        y_true = rt_results['y_true']
        y_pred = rt_results['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 完美预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        
        ax.set_xlabel('实际RT（秒）', fontsize=12)
        ax.set_ylabel('预测RT（秒）', fontsize=12)
        ax.set_title(f'RT预测vs实际\nR2={rt_results["r2"]:.4f}, MAE={rt_results["mae"]:.4f}秒', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rt_prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. RT残差图
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('预测RT（秒）', fontsize=12)
        ax.set_ylabel('残差（秒）', fontsize=12)
        ax.set_title(f'RT残差图\n均值={np.mean(residuals):.4f}, 标准差={np.std(residuals):.4f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rt_residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  图表已保存到: {output_dir}")


def save_results(choice_results, rt_results, consistent_trials, output_dir, feature_names):
    """保存结果到CSV文件"""
    print("\n=== 保存结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存试次数据
    trials_df = pd.DataFrame(consistent_trials)
    trials_df.to_csv(os.path.join(output_dir, 'consistent_trials.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 保存选择预测结果
    if choice_results is not None:
        choice_df = pd.DataFrame({
            'actual_choice': choice_results['y_true'],
            'predicted_choice': choice_results['y_pred'],
            'predicted_proba': choice_results['y_pred_proba']
        })
        choice_df.to_csv(os.path.join(output_dir, 'choice_predictions.csv'), index=False, encoding='utf-8-sig')
        
        # 保存模型摘要
        summary_choice = pd.DataFrame({
            'metric': ['Accuracy', 'Baseline Accuracy', 'AUC', 'CV Accuracy Mean', 'CV Accuracy Std'],
            'value': [
                choice_results['accuracy'],
                choice_results['baseline_accuracy'],
                choice_results['auc'],
                choice_results['cv_accuracy_mean'],
                choice_results['cv_accuracy_std']
            ]
        })
        summary_choice.to_csv(os.path.join(output_dir, 'choice_model_summary.csv'), index=False, encoding='utf-8-sig')
        
        # 保存所有特征的系数（权重）
        if 'coefficients' in choice_results:
            coef_df = pd.DataFrame(choice_results['coefficients'])
            coef_df.to_csv(os.path.join(output_dir, 'choice_coefficients_all.csv'), index=False, encoding='utf-8-sig')
            print(f"  所有特征权重已保存: choice_coefficients_all.csv")
    
    # 3. 保存RT预测结果
    if rt_results is not None:
        rt_df = pd.DataFrame({
            'actual_rt': rt_results['y_true'],
            'predicted_rt': rt_results['y_pred']
        })
        rt_df.to_csv(os.path.join(output_dir, 'rt_predictions.csv'), index=False, encoding='utf-8-sig')
        
        # 保存模型摘要
        summary_rt = pd.DataFrame({
            'metric': ['R2', 'MAE', 'RMSE', 'Correlation', 'CV R2 Mean', 'CV R2 Std', 'CV MAE Mean', 'CV MAE Std'],
            'value': [
                rt_results['r2'],
                rt_results['mae'],
                rt_results['rmse'],
                rt_results['correlation'],
                rt_results['cv_r2_mean'],
                rt_results['cv_r2_std'],
                rt_results['cv_mae_mean'],
                rt_results['cv_mae_std']
            ]
        })
        summary_rt.to_csv(os.path.join(output_dir, 'rt_model_summary.csv'), index=False, encoding='utf-8-sig')
        
        # 保存所有特征的系数（权重）
        if 'coefficients' in rt_results:
            coef_df = pd.DataFrame(rt_results['coefficients'])
            coef_df.to_csv(os.path.join(output_dir, 'rt_coefficients_all.csv'), index=False, encoding='utf-8-sig')
            print(f"  所有特征权重已保存: rt_coefficients_all.csv")
    
    print(f"  结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("="*60)
    print("yanchen个人数据验证")
    print("="*60)
    
    # 配置路径
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt"
    ]
    
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    ]
    
    output_dir = os.path.join(os.getcwd(), 'individual_yanchen_results')
    
    # 1. 加载yanchen数据
    all_trials = load_jiachen_trials(txt_files, mapping_files)  # 函数名保持不变，但实际加载的是yanchen的数据
    
    if len(all_trials) == 0:
        print("错误: 未能加载任何试次数据")
        return
    
    # 2. 加载pair_names（需要知道所有288个音频对）
    project_root = os.getcwd()
    _, _, pair_names = load_or_extract_features(project_root)
    
    # 3. 检查左右平衡一致性
    consistent_trials = check_left_right_consistency(all_trials, pair_names)
    
    if len(consistent_trials) < 50:
        print(f"警告: 一致的试次太少（{len(consistent_trials)}），建议至少50个")
        return
    
    # 4. 提取特征
    feature_indices_0based, feature_indices_1based, feature_names = extract_significant_features_from_table()
    print(f"\n使用的特征（共{len(feature_names)}个）:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    X, y_choice, y_rt = extract_features_for_trials(consistent_trials, pair_names, feature_indices_0based)
    
    if len(X) < 10:
        print("错误: 有效试次太少")
        return
    
    # 5. 拟合个人选择模型
    choice_results = fit_choice_model(X, y_choice, feature_names)
    
    # 6. 拟合反应时模型
    rt_results = fit_rt_model(X, y_rt, feature_names)
    
    # 7. 生成可视化图表
    plot_results(choice_results, rt_results, output_dir, feature_names)
    
    # 8. 保存结果
    save_results(choice_results, rt_results, consistent_trials, output_dir, feature_names)
    
    # 9. 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    if choice_results is not None:
        print(f"\n选择预测:")
        print(f"  准确率: {choice_results['accuracy']:.4f}")
        print(f"  基线准确率: {choice_results['baseline_accuracy']:.4f}")
        print(f"  提升: {(choice_results['accuracy'] - choice_results['baseline_accuracy'])*100:.2f}%")
        print(f"  AUC: {choice_results['auc']:.4f}")
        print(f"  CV准确率: {choice_results['cv_accuracy_mean']:.4f} ± {choice_results['cv_accuracy_std']:.4f}")
    
    if rt_results is not None:
        print(f"\n反应时预测:")
        print(f"  R2: {rt_results['r2']:.4f}")
        print(f"  MAE: {rt_results['mae']:.4f} 秒")
        print(f"  RMSE: {rt_results['rmse']:.4f} 秒")
        print(f"  CV R2: {rt_results['cv_r2_mean']:.4f} ± {rt_results['cv_r2_std']:.4f}")
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()

