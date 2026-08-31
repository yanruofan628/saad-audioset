#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯逻辑回归示例
对比传统频率主义方法和贝叶斯方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 贝叶斯库
try:
    import pymc3 as pm
    import arviz as az
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    print("警告: PyMC3未安装，无法运行贝叶斯分析")
    print("安装命令: pip install pymc3 arviz")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def explain_bayesian_vs_frequentist():
    """解释贝叶斯方法和频率主义方法的区别"""
    print("="*60)
    print("贝叶斯方法 vs 频率主义方法")
    print("="*60)
    
    print("\n1. 频率主义方法（传统统计，如p值）:")
    print("   - 参数是固定的（未知但存在真实值）")
    print("   - 通过重复抽样来推断参数")
    print("   - 结果：p值、置信区间")
    print("   - 问题：小样本时检验力低，难以检测到效应")
    
    print("\n2. 贝叶斯方法:")
    print("   - 参数是随机的（有先验分布）")
    print("   - 结合先验知识和数据来更新对参数的信念")
    print("   - 结果：后验分布、可信区间、贝叶斯因子")
    print("   - 优势：小样本时更稳健，可以提供概率性陈述")
    
    print("\n3. 核心公式（贝叶斯定理）:")
    print("   P(参数|数据) = P(数据|参数) × P(参数) / P(数据)")
    print("   后验分布 = 似然 × 先验分布 / 归一化常数")
    print("   后验 ∝ 似然 × 先验")
    
    print("\n4. 关键概念:")
    print("   - 先验分布（Prior）: 在观察数据前对参数的信念")
    print("   - 似然（Likelihood）: 数据在给定参数下的概率")
    print("   - 后验分布（Posterior）: 结合先验和数据后的参数分布")
    print("   - 可信区间（Credible Interval）: 参数有X%概率落在此区间")
    
    print("\n5. 为什么贝叶斯在小样本时更好？")
    print("   - 先验信息可以补充小样本的不足")
    print("   - 不依赖渐近理论（大样本理论）")
    print("   - 可以提供参数的概率分布，而不仅仅是点估计")
    print("   - 可以量化不确定性，而不是简单的显著/不显著")


def bayesian_logistic_regression_example(X, y, feature_names):
    """
    贝叶斯逻辑回归示例
    
    参数:
    -----
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵
    y : np.ndarray, shape (n_samples,)
        二分类标签（0或1）
    feature_names : list
        特征名称列表
    """
    if not PYMC3_AVAILABLE:
        print("\n无法运行贝叶斯分析，PyMC3未安装")
        return None
    
    print("\n" + "="*60)
    print("贝叶斯逻辑回归分析")
    print("="*60)
    
    n_samples, n_features = X.shape
    
    print(f"\n数据信息:")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {n_features}")
    print(f"  类别分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # 标准化特征（贝叶斯方法通常需要标准化）
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    print("\n构建贝叶斯模型...")
    print("使用弱先验（Weakly Informative Prior）:")
    print("  - 截距: Normal(0, 1)")
    print("  - 系数: Normal(0, 1)")
    
    with pm.Model() as model:
        # 先验分布
        # 截距：正态分布，均值0，标准差1
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        
        # 系数：正态分布，均值0，标准差1（弱先验）
        coefficients = pm.Normal('coefficients', mu=0, sigma=1, shape=n_features)
        
        # 线性组合
        logit_p = intercept + pm.math.dot(X_std, coefficients)
        
        # 似然函数（伯努利分布）
        y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)
        
        # 采样（MCMC方法）
        print("\n开始MCMC采样（这可能需要一些时间）...")
        trace = pm.sample(2000, tune=1000, return_inference_data=True, 
                         progressbar=True, cores=1)
    
    print("\n采样完成！")
    
    # 总结后验分布
    print("\n" + "="*60)
    print("后验分布摘要")
    print("="*60)
    
    summary = az.summary(trace, var_names=['intercept', 'coefficients'])
    print(summary)
    
    # 解释结果
    print("\n" + "="*60)
    print("结果解释")
    print("="*60)
    
    print("\n1. 后验均值（mean）: 参数的最可能值")
    print("2. 标准差（sd）: 参数的不确定性")
    print("3. 可信区间（hdi_3%和hdi_97%）: 参数有94%概率落在此区间")
    print("4. 如果可信区间不包含0，说明该特征有影响")
    
    # 提取系数
    intercept_samples = trace.posterior['intercept'].values.flatten()
    coef_samples = trace.posterior['coefficients'].values
    
    print("\n" + "="*60)
    print("特征系数分析")
    print("="*60)
    
    for i, name in enumerate(feature_names):
        coef_samples_i = coef_samples[:, :, i].flatten()
        mean_coef = np.mean(coef_samples_i)
        std_coef = np.std(coef_samples_i)
        hdi_low = np.percentile(coef_samples_i, 3)
        hdi_high = np.percentile(coef_samples_i, 97)
        
        # 计算不包含0的概率（类似p值）
        prob_not_zero = np.mean(np.abs(coef_samples_i) > 0.01)
        
        print(f"\n特征 {i+1}: {name}")
        print(f"  后验均值: {mean_coef:.4f}")
        print(f"  后验标准差: {std_coef:.4f}")
        print(f"  94%可信区间: [{hdi_low:.4f}, {hdi_high:.4f}]")
        print(f"  不包含0的概率: {prob_not_zero:.2%}")
        
        if hdi_low > 0 or hdi_high < 0:
            print(f"  → 该特征有显著影响（可信区间不包含0）")
        else:
            print(f"  → 该特征的影响不确定（可信区间包含0）")
    
    # 可视化
    print("\n生成可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 截距的后验分布
    ax = axes[0, 0]
    ax.hist(intercept_samples, bins=50, alpha=0.7, density=True)
    ax.axvline(0, color='r', linestyle='--', label='0')
    ax.axvline(np.mean(intercept_samples), color='g', linestyle='--', label='均值')
    ax.set_xlabel('截距值')
    ax.set_ylabel('密度')
    ax.set_title('截距的后验分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 前3个系数的后验分布
    ax = axes[0, 1]
    for i in range(min(3, n_features)):
        coef_samples_i = coef_samples[:, :, i].flatten()
        ax.hist(coef_samples_i, bins=30, alpha=0.5, 
               label=f'{feature_names[i][:20]}', density=True)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('系数值')
    ax.set_ylabel('密度')
    ax.set_title('前3个特征系数的后验分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 系数后验均值的条形图
    ax = axes[1, 0]
    coef_means = [np.mean(coef_samples[:, :, i].flatten()) for i in range(n_features)]
    coef_stds = [np.std(coef_samples[:, :, i].flatten()) for i in range(n_features)]
    x_pos = np.arange(n_features)
    ax.barh(x_pos, coef_means, xerr=coef_stds, alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_yticks(x_pos)
    ax.set_yticklabels([name[:30] for name in feature_names])
    ax.set_xlabel('系数值')
    ax.set_title('特征系数的后验均值 ± 标准差')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. 可信区间图
    ax = axes[1, 1]
    hdi_lows = [np.percentile(coef_samples[:, :, i].flatten(), 3) for i in range(n_features)]
    hdi_highs = [np.percentile(coef_samples[:, :, i].flatten(), 97) for i in range(n_features)]
    coef_means = [np.mean(coef_samples[:, :, i].flatten()) for i in range(n_features)]
    
    for i in range(n_features):
        ax.plot([hdi_lows[i], hdi_highs[i]], [i, i], 'b-', linewidth=2, alpha=0.7)
        ax.plot(coef_means[i], i, 'ro', markersize=8)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([name[:30] for name in feature_names])
    ax.set_xlabel('系数值')
    ax.set_title('94%可信区间（蓝色）和均值（红点）')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = 'bayesian_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存: {output_path}")
    plt.close()
    
    return trace, summary


def compare_frequentist_vs_bayesian(X, y, feature_names):
    """对比频率主义和贝叶斯方法的结果"""
    print("\n" + "="*60)
    print("对比：频率主义 vs 贝叶斯")
    print("="*60)
    
    # 频率主义方法
    print("\n1. 频率主义方法（statsmodels）:")
    X_const = sm.add_constant(X)
    logit_freq = sm.Logit(y, X_const).fit(disp=0, maxiter=1000)
    
    print(f"  伪R²: {logit_freq.prsquared:.4f}")
    print(f"  显著特征数 (p<0.05): {np.sum(logit_freq.pvalues[1:] < 0.05)}/{len(feature_names)}")
    
    print("\n  系数和p值:")
    for i, name in enumerate(feature_names):
        coef = logit_freq.params[i+1]
        pval = logit_freq.pvalues[i+1]
        sig = '*' if pval < 0.05 else ''
        print(f"    {name[:30]}: {coef:7.4f}, p={pval:.4f} {sig}")
    
    # 贝叶斯方法
    if PYMC3_AVAILABLE:
        print("\n2. 贝叶斯方法（PyMC3）:")
        trace, summary = bayesian_logistic_regression_example(X, y, feature_names)
        
        print("\n  对比总结:")
        print("  - 频率主义: 给出点估计和p值（显著/不显著）")
        print("  - 贝叶斯: 给出后验分布和可信区间（概率性陈述）")
        print("  - 贝叶斯在小样本时更稳健，可以提供更多信息")
    else:
        print("\n2. 贝叶斯方法:")
        print("  PyMC3未安装，无法运行贝叶斯分析")


def main():
    """主函数"""
    print("="*60)
    print("贝叶斯方法简介和示例")
    print("="*60)
    
    # 解释贝叶斯方法
    explain_bayesian_vs_frequentist()
    
    # 如果有实际数据，可以运行示例
    print("\n" + "="*60)
    print("注意：要运行完整的贝叶斯分析，需要:")
    print("1. 安装PyMC3: pip install pymc3 arviz")
    print("2. 准备数据（X, y, feature_names）")
    print("3. 调用 bayesian_logistic_regression_example() 函数")
    print("="*60)
    
    print("\n贝叶斯方法的优势总结:")
    print("1. 小样本时更稳健（不依赖大样本理论）")
    print("2. 可以结合先验知识（如果有的话）")
    print("3. 提供概率性陈述（而非简单的显著/不显著）")
    print("4. 可以量化不确定性（后验分布）")
    print("5. 更直观的解释（参数有X%概率在某个区间）")
    
    print("\n贝叶斯方法的劣势:")
    print("1. 计算更复杂（需要MCMC采样）")
    print("2. 先验选择可能主观")
    print("3. 需要更多计算资源")


if __name__ == '__main__':
    main()

