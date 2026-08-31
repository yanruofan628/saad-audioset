#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jiachen个人数据验证（全部288个试次，带位置编码）
- 使用全部288个试次（不删除不一致的）
- 编码位置信息（左耳和右耳）
- 提取左耳和右耳的特征（分别提取）
- 拟合个人选择和反应时
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
from sklearn.model_selection import cross_val_score

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
    """加载jiachen的两个txt文件并合并"""
    print("\n=== 加载jiachen数据 ===")
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


def extract_features_with_position(all_trials, diff_zscore, pair_names, feature_indices_0based):
    """
    为每个试次提取特征，包含位置信息
    
    使用已有的特征数据（pair_feature_diff_zscore.npy）
    对于AB版本：直接用特征差值（left - right）
    对于BA版本：左右交换，用 right - left（即取负）
    
    特征包括：
    1. 特征差值（n个）
    2. 位置编码（1个：0=AB版本，1=BA版本）
    
    总共：n + 1个特征
    
    Parameters:
    -----------
    all_trials : list
        所有试次数据
    diff_zscore : np.ndarray, shape (18, n_pairs)
        已有的特征差值数据（z-score归一化）
    pair_names : list
        原始pair_list（288个），用于判断AB/BA版本和查找索引
    feature_indices_0based : list
        特征索引（从0开始，对应18个特征中的哪些）
        
    Returns:
    --------
    X : np.ndarray, shape (n_trials, n_features)
        特征矩阵
    y_choice : np.ndarray, shape (n_trials,)
        选择（1=选左，2=选右）
    y_rt : np.ndarray, shape (n_trials,)
        反应时（秒）
    trial_info : list
        试次信息（用于调试）
    """
    print("\n=== 提取特征（带位置编码，使用已有特征） ===")
    
    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(pair_names)}
    
    # 创建规范化键到AB版本pair_name的映射（用于判断AB/BA版本）
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
            normalized_to_ab_pair[key] = pair_name  # 记录第一次出现的（AB版本）
    
    X_list = []
    y_choice_list = []
    y_rt_list = []
    trial_info_list = []
    
    valid_count = 0
    error_count = 0
    
    for i, trial in enumerate(all_trials):
        pair_name = trial['original_name']
        response = trial['response']
        rt = trial.get('reaction_time', None)
        
        try:
            # 解析pair_name
            parse_result = parse_pair_name(pair_name)
            if parse_result is None:
                error_count += 1
                continue
            
            left_cat, left_id, right_cat, right_id, _ = parse_result
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            key = tuple(sorted([audio_a, audio_b]))
            
            # 判断是AB版本还是BA版本
            if key in normalized_to_ab_pair:
                ab_pair_name = normalized_to_ab_pair[key]
                is_ab_version = (pair_name == ab_pair_name)
            else:
                # 如果不在映射中，使用简化判断
                try:
                    left_id_num = int(left_id) if left_id.isdigit() else float('inf')
                    right_id_num = int(right_id) if right_id.isdigit() else float('inf')
                    is_ab_version = left_id_num < right_id_num
                except:
                    is_ab_version = left_id < right_id
            
            # 找到对应的AB版本pair_name（用于查找特征）
            if key in normalized_to_ab_pair:
                ab_pair_name = normalized_to_ab_pair[key]
            else:
                # 如果不在映射中，假设当前pair_name就是AB版本
                ab_pair_name = pair_name
            
            # 查找特征索引
            if ab_pair_name not in pair_to_idx:
                error_count += 1
                continue
            
            idx = pair_to_idx[ab_pair_name]
            
            # 提取特征差值（使用选定的特征）
            # diff_zscore存储的是 left - right
            feat_diff_ab = diff_zscore[feature_indices_0based, idx]  # (n_selected_features,)
            
            # 如果是BA版本，左右交换：用 right - left（即取负）
            # 如果是AB版本，直接用 left - right
            if is_ab_version:
                feat_diff = feat_diff_ab  # left - right
            else:
                feat_diff = -feat_diff_ab  # right - left（左右交换）
            
            # 位置编码：0=AB版本（A在左，B在右），1=BA版本（B在左，A在右）
            position_code = 0.0 if is_ab_version else 1.0
            
            # 组合特征：[特征差值, 位置编码]
            features = np.concatenate([
                feat_diff,           # (n_feat,)
                np.array([position_code])  # (1,)
            ])
            
            # 检查NaN
            if np.isnan(features).any():
                error_count += 1
                continue
            
            X_list.append(features)
            y_choice_list.append(response)
            y_rt_list.append(rt if rt is not None else np.nan)
            trial_info_list.append({
                'pair_name': pair_name,
                'ab_pair_name': ab_pair_name,
                'audio_a': audio_a,
                'audio_b': audio_b,
                'is_ab_version': is_ab_version,
                'response': response
            })
            
            valid_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(all_trials)}, 有效: {valid_count}, 错误: {error_count}")
        
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # 只打印前5个错误
                print(f"  警告: 提取特征失败 {pair_name}: {e}")
            continue
    
    X = np.array(X_list)
    y_choice = np.array(y_choice_list)
    y_rt = np.array(y_rt_list)
    
    print(f"\n特征提取完成:")
    print(f"  有效试次: {len(X)}/{len(all_trials)}")
    print(f"  错误试次: {error_count}")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  AB版本: {np.sum([t['is_ab_version'] for t in trial_info_list])}")
    print(f"  BA版本: {np.sum([not t['is_ab_version'] for t in trial_info_list])}")
    print(f"  选左: {np.sum(y_choice == 1)} ({np.mean(y_choice == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(y_choice == 2)} ({np.mean(y_choice == 2)*100:.1f}%)")
    print(f"  有RT数据: {np.sum(~np.isnan(y_rt))} ({np.sum(~np.isnan(y_rt))/len(y_rt)*100:.1f}%)")
    
    return X, y_choice, y_rt, trial_info_list


def fit_choice_model(X, y_choice, feature_names):
    """拟合个人选择模型（逻辑回归）"""
    print("\n=== 拟合个人选择模型（逻辑回归） ===")
    
    # 转换为二分类：1=选左 -> 0, 2=选右 -> 1
    y_binary = (y_choice == 2).astype(int)
    
    print(f"  数据统计:")
    print(f"    选左（0）: {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"    选右（1）: {np.sum(y_binary == 1)} ({np.mean(y_binary == 1)*100:.1f}%)")
    
    # 使用statsmodels逻辑回归
    X_with_const = sm.add_constant(X)
    logit_model = sm.Logit(y_binary, X_with_const).fit(disp=0, maxiter=1000)
    
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
    feature_names_full = [f'Diff_{name}' for name in feature_names] + ['Position_Code']
    
    print(f"    截距: {logit_model.params[0]:.4f}, p={logit_model.pvalues[0]:.4f}")
    
    # 保存所有系数
    coefficients = []
    for i, name in enumerate(feature_names_full):
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
        'feature_names': feature_names_full
    }


def fit_rt_model(X, y_rt, feature_names):
    """拟合反应时模型（线性回归）"""
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
        'valid_mask': valid_mask
    }


def plot_results(choice_results, rt_results, output_dir):
    """绘制结果图表"""
    print("\n=== 生成可视化图表 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 选择预测：预测概率vs实际
    if choice_results is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        y_true = choice_results['y_true']
        y_pred_proba = choice_results['y_pred_proba']
        
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
    
    print(f"  图表已保存到: {output_dir}")


def save_results(choice_results, rt_results, trial_info, output_dir):
    """保存结果到CSV文件"""
    print("\n=== 保存结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存试次信息
    trials_df = pd.DataFrame(trial_info)
    trials_df.to_csv(os.path.join(output_dir, 'all_trials_info.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 保存选择预测结果
    if choice_results is not None:
        choice_df = pd.DataFrame({
            'actual_choice': choice_results['y_true'],
            'predicted_choice': choice_results['y_pred'],
            'predicted_proba': choice_results['y_pred_proba']
        })
        choice_df.to_csv(os.path.join(output_dir, 'choice_predictions.csv'), index=False, encoding='utf-8-sig')
        
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
    
    print(f"  结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("="*60)
    print("jiachen个人数据验证（全部288个试次，带位置编码）")
    print("="*60)
    
    # 配置路径
    txt_files = [
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
    ]
    
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    ]
    
    output_dir = os.path.join(os.getcwd(), 'individual_jiachen_full_288_results')
    
    # 1. 加载jiachen数据
    all_trials = load_jiachen_trials(txt_files, mapping_files)
    
    if len(all_trials) == 0:
        print("错误: 未能加载任何试次数据")
        return
    
    # 2. 加载已有特征数据
    project_root = os.getcwd()
    _, diff_zscore, pair_names = load_or_extract_features(project_root)
    
    # 3. 提取特征（带位置编码，使用已有特征）
    feature_indices_0based, feature_indices_1based, feature_names = extract_significant_features_from_table()
    print(f"\n使用的特征（共{len(feature_names)}个）:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    X, y_choice, y_rt, trial_info = extract_features_with_position(
        all_trials, diff_zscore, pair_names, feature_indices_0based
    )
    
    if len(X) < 10:
        print("错误: 有效试次太少")
        return
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"  特征组成: 特征差值({len(feature_names)}) + 位置编码(1) = {len(feature_names) + 1}个特征")
    
    # 4. 拟合个人选择模型
    choice_results = fit_choice_model(X, y_choice, feature_names)
    
    # 5. 拟合反应时模型
    rt_results = fit_rt_model(X, y_rt, feature_names)
    
    # 6. 生成可视化图表
    plot_results(choice_results, rt_results, output_dir)
    
    # 7. 保存结果
    save_results(choice_results, rt_results, trial_info, output_dir)
    
    # 8. 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    print(f"\n使用全部 {len(X)} 个试次（带位置编码）")
    
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

