#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比高度一致和不一致音频对的特征差异
- 读取一致性分析结果
- 分为高度一致组和不一致组
- 计算每组9个特征的统计量（均值或std）
- 对比并画柱状图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要的函数
from linear_regression_selected_features import (
    load_or_extract_features,
    extract_significant_features_from_table,
    parse_pair_name
)


def load_consistency_results(consistency_file):
    """
    加载一致性分析结果
    
    Parameters:
    -----------
    consistency_file : str
        一致性分析结果CSV文件路径
        
    Returns:
    --------
    consistency_df : DataFrame
        包含一致性分析结果的DataFrame
    """
    if not os.path.exists(consistency_file):
        print(f"错误: 一致性分析结果文件不存在: {consistency_file}")
        return None
    
    df = pd.read_csv(consistency_file, encoding='utf-8-sig')
    print(f"\n加载一致性分析结果: {len(df)} 个音频对")
    return df


def group_pairs_by_consistency(consistency_df):
    """
    根据一致性分组音频对
    
    Parameters:
    -----------
    consistency_df : DataFrame
        一致性分析结果
        
    Returns:
    --------
    highly_consistent_pairs : list
        高度一致的音频对名称列表
    inconsistent_pairs : list
        不一致的音频对名称列表
    """
    # 检查列名（可能是'consistency_rate'或'group_consistency'）
    consistency_col = None
    for col in ['consistency_rate', 'group_consistency', 'consistency']:
        if col in consistency_df.columns:
            consistency_col = col
            break
    
    if consistency_col is None:
        print("错误: 找不到一致性列")
        print(f"可用列: {consistency_df.columns.tolist()}")
        return [], []
    
    # 高度一致：群体一致性 > 80%
    highly_consistent = consistency_df[
        (consistency_df[consistency_col] > 0.8) & 
        (consistency_df[consistency_col] <= 1.0)
    ]
    
    # 不一致/随机：群体一致性在 35%-65% 之间
    inconsistent = consistency_df[
        (consistency_df[consistency_col] >= 0.35) & 
        (consistency_df[consistency_col] <= 0.65)
    ]
    
    # 提取音频对名称（需要从pair_key或audio_a/audio_b重建）
    highly_consistent_pairs = []
    inconsistent_pairs = []
    
    # 获取所有pair_names用于匹配
    project_root = os.getcwd()
    _, _, all_pair_names = load_or_extract_features(project_root)
    
    # 创建normalized_key到pair_names的映射
    normalized_to_pairs = {}
    for pair_name in all_pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            continue
        left_cat, left_id, right_cat, right_id, exp_type = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))
        if key not in normalized_to_pairs:
            normalized_to_pairs[key] = []
        normalized_to_pairs[key].append(pair_name)
    
    # 匹配高度一致的音频对
    for _, row in highly_consistent.iterrows():
        audio_a = row['audio_a']
        audio_b = row['audio_b']
        key = tuple(sorted([audio_a, audio_b]))
        if key in normalized_to_pairs:
            # 取第一个匹配的pair_name（AB版本）
            highly_consistent_pairs.extend(normalized_to_pairs[key][:1])  # 只取一个版本
    
    # 匹配不一致的音频对
    for _, row in inconsistent.iterrows():
        audio_a = row['audio_a']
        audio_b = row['audio_b']
        key = tuple(sorted([audio_a, audio_b]))
        if key in normalized_to_pairs:
            inconsistent_pairs.extend(normalized_to_pairs[key][:1])  # 只取一个版本
    
    print(f"\n分组结果:")
    print(f"  高度一致组: {len(highly_consistent_pairs)} 个音频对")
    print(f"  不一致组: {len(inconsistent_pairs)} 个音频对")
    
    return highly_consistent_pairs, inconsistent_pairs


def extract_features_for_pairs(pair_names, feature_indices_0based):
    """
    为音频对提取特征（左-右的差值）
    
    Parameters:
    -----------
    pair_names : list
        音频对名称列表
    feature_indices_0based : list
        特征索引（从0开始）
        
    Returns:
    --------
    features : np.ndarray, shape (n_pairs, n_features)
        特征矩阵（每个特征是左-右的差值，z-score归一化）
    """
    project_root = os.getcwd()
    diff_raw, diff_zscore, all_pair_names = load_or_extract_features(project_root)
    
    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(all_pair_names)}
    
    features_list = []
    valid_pairs = []
    
    for pair_name in pair_names:
        if pair_name not in pair_to_idx:
            continue
        
        idx = pair_to_idx[pair_name]
        
        # 提取特征（使用z-score归一化后的值）
        feat_values = diff_zscore[feature_indices_0based, idx]
        
        # 检查是否有NaN
        if np.isnan(feat_values).any():
            continue
        
        features_list.append(feat_values)
        valid_pairs.append(pair_name)
    
    if len(features_list) == 0:
        print("警告: 没有找到有效的特征数据")
        return None, []
    
    features = np.array(features_list)
    print(f"\n特征提取完成:")
    print(f"  有效音频对: {len(valid_pairs)}/{len(pair_names)}")
    print(f"  特征矩阵形状: {features.shape}")
    
    return features, valid_pairs


def compute_group_statistics(features, feature_names, stat_type='mean'):
    """
    计算组的统计量
    
    Parameters:
    -----------
    features : np.ndarray, shape (n_pairs, n_features)
        特征矩阵
    feature_names : list
        特征名称列表
    stat_type : str
        统计量类型：'mean' 或 'std'
        
    Returns:
    --------
    stats_dict : dict
        每个特征的统计量
    """
    if stat_type == 'mean':
        stats_values = np.mean(features, axis=0)
    elif stat_type == 'std':
        stats_values = np.std(features, axis=0)
    else:
        raise ValueError(f"未知的统计量类型: {stat_type}")
    
    stats_dict = dict(zip(feature_names, stats_values))
    return stats_dict


def plot_comparison(highly_consistent_stats, inconsistent_stats, feature_names, 
                   stat_type='mean', output_dir='.'):
    """
    绘制对比柱状图
    
    Parameters:
    -----------
    highly_consistent_stats : dict
        高度一致组的统计量
    inconsistent_stats : dict
        不一致组的统计量
    feature_names : list
        特征名称列表
    stat_type : str
        统计量类型
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    consistent_values = [highly_consistent_stats[name] for name in feature_names]
    inconsistent_values = [inconsistent_stats[name] for name in feature_names]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, consistent_values, width, 
                   label='Highly Consistent', color='#2ca02c', alpha=0.7)
    bars2 = ax.bar(x + width/2, inconsistent_values, width, 
                   label='Inconsistent', color='#ff7f0e', alpha=0.7)
    
    ax.set_xlabel('Features', fontsize=16, fontweight='bold')
    ylabel = 'Mean' if stat_type == 'mean' else 'Std'
    ax.set_ylabel(f'Feature Difference {ylabel} (z-score, Left-Right)', fontsize=16, fontweight='bold')
    ax.set_title(f'Highly Consistent vs Inconsistent: Feature Difference {ylabel} (Left-Right)', 
                fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'feature_comparison_{stat_type}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存: {output_path}")
    
    # 进行统计检验
    print(f"\n统计检验（{stat_type}）:")
    print(f"{'特征':<40} {'高度一致组':>15} {'不一致组':>15} {'差异':>15} {'p值':>12}")
    print("-" * 97)
    
    comparison_results = []
    for name in feature_names:
        # 需要原始特征值进行统计检验，这里先用组统计量
        consistent_val = highly_consistent_stats[name]
        inconsistent_val = inconsistent_stats[name]
        diff = consistent_val - inconsistent_val
        
        # 由于只有组统计量，无法进行t检验，只显示差异
        comparison_results.append({
            'feature': name,
            'highly_consistent': consistent_val,
            'inconsistent': inconsistent_val,
            'difference': diff,
            'pvalue': np.nan  # 需要原始数据才能计算
        })
        
        print(f"{name:<40} {consistent_val:>15.4f} {inconsistent_val:>15.4f} {diff:>15.4f}")
    
    return comparison_results


def plot_comparison_with_raw_data(highly_consistent_features, inconsistent_features, 
                                 feature_names, stat_type='mean', output_dir='.'):
    """
    使用原始特征数据绘制对比图并进行统计检验
    
    Parameters:
    -----------
    highly_consistent_features : np.ndarray
        高度一致组的特征矩阵
    inconsistent_features : np.ndarray
        不一致组的特征矩阵
    feature_names : list
        特征名称列表
    stat_type : str
        统计量类型
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算统计量
    if stat_type == 'mean':
        consistent_stats = np.mean(highly_consistent_features, axis=0)
        inconsistent_stats = np.mean(inconsistent_features, axis=0)
    else:
        consistent_stats = np.std(highly_consistent_features, axis=0)
        inconsistent_stats = np.std(inconsistent_features, axis=0)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, consistent_stats, width, 
                   label='Highly Consistent', color='#2ca02c', alpha=0.7)
    bars2 = ax.bar(x + width/2, inconsistent_stats, width, 
                   label='Inconsistent', color='#ff7f0e', alpha=0.7)
    
    ax.set_xlabel('Features', fontsize=16, fontweight='bold')
    ylabel = 'Mean' if stat_type == 'mean' else 'Std'
    ax.set_ylabel(f'Feature Difference {ylabel} (z-score, Left-Right)', fontsize=16, fontweight='bold')
    ax.set_title(f'Highly Consistent vs Inconsistent: Feature Difference {ylabel} (Left-Right)', 
                fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加数值标签和显著性标记
    comparison_results = []
    for i, name in enumerate(feature_names):
        # 统计检验（t检验）
        consistent_vals = highly_consistent_features[:, i]
        inconsistent_vals = inconsistent_features[:, i]
        
        try:
            t_stat, p_value = stats.ttest_ind(consistent_vals, inconsistent_vals)
        except:
            t_stat, p_value = np.nan, np.nan
        
        consistent_val = consistent_stats[i]
        inconsistent_val = inconsistent_stats[i]
        diff = consistent_val - inconsistent_val
        
        comparison_results.append({
            'feature': name,
            'highly_consistent_mean': consistent_val,
            'inconsistent_mean': inconsistent_val,
            'difference': diff,
            't_statistic': t_stat,
            'pvalue': p_value
        })
        
        # 添加数值标签
        for bars, val in [(bars1, consistent_val), (bars2, inconsistent_val)]:
            height = bars[i].get_height()
            ax.text(bars[i].get_x() + bars[i].get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=12, fontweight='bold')
        
        # 添加显著性标记
        if not np.isnan(p_value):
            max_val = max(consistent_val, inconsistent_val)
            sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            if sig_marker:
                ax.text(i, max_val * 1.1, sig_marker, ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'feature_comparison_{stat_type}_with_test.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存: {output_path}")
    
    # 打印统计检验结果
    print(f"\n统计检验结果（{stat_type}，t检验）:")
    print(f"{'特征':<40} {'高度一致组':>15} {'不一致组':>15} {'差异':>15} {'p值':>12} {'显著性':>8}")
    print("-" * 105)
    
    for result in comparison_results:
        pval = result['pvalue']
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        pval_str = f'{pval:.4e}' if not np.isnan(pval) else 'N/A'
        print(f"{result['feature']:<40} {result['highly_consistent_mean']:>15.4f} "
              f"{result['inconsistent_mean']:>15.4f} {result['difference']:>15.4f} "
              f"{pval_str:>12} {sig:>8}")
    
    # 保存结果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(output_dir, f'feature_comparison_{stat_type}.csv'), 
                        index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存: feature_comparison_{stat_type}.csv")
    
    return comparison_results


def main():
    """主函数"""
    print("="*80)
    print("高度一致组 vs 不一致组：特征对比分析")
    print("="*80)
    
    # 配置路径
    consistency_file = os.path.join(os.getcwd(), 
                                    'selection_consistency_analysis', 
                                    'group_consistency.csv')
    
    output_dir = os.path.join(os.getcwd(), 'feature_comparison_by_consistency')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载一致性分析结果
    consistency_df = load_consistency_results(consistency_file)
    if consistency_df is None:
        return
    
    # 2. 分组音频对
    highly_consistent_pairs, inconsistent_pairs = group_pairs_by_consistency(consistency_df)
    
    if len(highly_consistent_pairs) == 0 or len(inconsistent_pairs) == 0:
        print("错误: 分组后某一组为空，无法进行对比")
        return
    
    # 3. 提取特征
    feature_indices_0based, feature_indices_1based, feature_names = extract_significant_features_from_table()
    print(f"\n使用的特征（共{len(feature_names)}个）:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    highly_consistent_features, valid_consistent_pairs = extract_features_for_pairs(
        highly_consistent_pairs, feature_indices_0based)
    inconsistent_features, valid_inconsistent_pairs = extract_features_for_pairs(
        inconsistent_pairs, feature_indices_0based)
    
    if highly_consistent_features is None or inconsistent_features is None:
        print("错误: 特征提取失败")
        return
    
    # 4. 计算统计量并对比
    print("\n" + "="*80)
    print("特征均值对比")
    print("="*80)
    comparison_mean = plot_comparison_with_raw_data(
        highly_consistent_features, inconsistent_features, 
        feature_names, stat_type='mean', output_dir=output_dir)
    
    print("\n" + "="*80)
    print("特征标准差对比")
    print("="*80)
    comparison_std = plot_comparison_with_raw_data(
        highly_consistent_features, inconsistent_features, 
        feature_names, stat_type='std', output_dir=output_dir)
    
    # 5. 保存分组信息
    groups_df = pd.DataFrame({
        'group': ['高度一致组'] * len(valid_consistent_pairs) + 
                 ['不一致组'] * len(valid_inconsistent_pairs),
        'pair_name': valid_consistent_pairs + valid_inconsistent_pairs
    })
    groups_df.to_csv(os.path.join(output_dir, 'grouped_pairs.csv'), 
                    index=False, encoding='utf-8-sig')
    print(f"\n分组信息已保存: grouped_pairs.csv")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()

