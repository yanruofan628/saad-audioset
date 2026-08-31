#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析attention模型过拟合问题
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def analyze_attention_overfitting():
    """分析attention模型的过拟合情况"""

    project_root = os.getcwd()
    features = ['loudness']

    results = []

    for feature in features:
        for model_type in ['single_attention', 'dual_stream']:
            for fold_idx in range(1, 6):
                fold_dir = os.path.join(project_root, 'attention_5fold', feature, model_type, f'fold_{fold_idx}')

                # 读取metrics
                metrics_path = os.path.join(fold_dir, 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)

                    results.append({
                        'feature': feature,
                        'model_type': model_type,
                        'fold': fold_idx,
                        'test_mse': metrics['mse'],
                        'test_mae': metrics['mae'],
                        'test_r2': metrics['r2']
                    })

    df = pd.DataFrame(results)

    # 计算平均性能
    print("=== Attention模型性能汇总 ===")
    summary = df.groupby(['model_type']).agg({
        'test_mse': ['mean', 'std'],
        'test_mae': ['mean', 'std'],
        'test_r2': ['mean', 'std']
    }).round(4)
    print(summary)

    # 与线性回归对比
    print("\n=== 与线性回归对比 ===")

    # 读取线性回归结果
    linear_summary_path = 'linear_regression_results_loudness_all/loudness_models_summary.csv'
    if os.path.exists(linear_summary_path):
        linear_df = pd.read_csv(linear_summary_path)

        # 找出最好的线性模型
        best_linear = linear_df.loc[linear_df['rsquared'].idxmax()]
        print(f"最佳线性模型 ({best_linear['model_name']}): R²={best_linear['rsquared']:.4f}")
        print(f"  样本数: {best_linear['n_samples']}, 特征数: {best_linear['n_features']}")

        # attention模型平均
        attention_avg = df.groupby('model_type')[['test_mse', 'test_mae', 'test_r2']].mean()

        print("\nAttention模型平均:")
        print(f"Single Attention - MSE: {attention_avg.loc['single_attention', 'test_mse']:.4f}, MAE: {attention_avg.loc['single_attention', 'test_mae']:.4f}, R²: {attention_avg.loc['single_attention', 'test_r2']:.4f}")
        print(f"Dual Stream      - MSE: {attention_avg.loc['dual_stream', 'test_mse']:.4f}, MAE: {attention_avg.loc['dual_stream', 'test_mae']:.4f}, R²: {attention_avg.loc['dual_stream', 'test_r2']:.4f}")

    return df

def analyze_model_complexity():
    """分析模型复杂度与性能的关系"""

    print("\n=== 模型复杂度分析 ===")

    # 检查attention模型的参数量
    import torch
    from attention_loudness_model import SimpleAttentionModel, DualStreamAttention

    # 模拟模型参数
    device = torch.device('cpu')
    T = 250  # 假设时间步长

    single_model = SimpleAttentionModel(time_input_dim=2, d_model=64, attn_hidden=64, dropout=0.2)
    dual_model = DualStreamAttention(time_steps=T, dim=64, dropout=0.2)

    single_params = sum(p.numel() for p in single_model.parameters())
    dual_params = sum(p.numel() for p in dual_model.parameters())

    print(f"Single Attention模型参数量: {single_params:,}")
    print(f"Dual Stream模型参数量: {dual_params:,}")

    # 检查数据集大小
    print("\n数据集信息:")
    dataset_size = 144  # 从线性回归结果看出
    train_size = int(0.8 * dataset_size)  # 80%训练
    val_size = int(0.1 * dataset_size)    # 10%验证
    test_size = dataset_size - train_size - val_size

    print(f"总样本数: {dataset_size}")
    print(f"训练集: {train_size}")
    print(f"验证集: {val_size}")
    print(f"测试集: {test_size}")
    print(f"参数/样本比: Single={single_params/train_size:.1f}, Dual={dual_params/train_size:.1f}")
    print("\n过拟合风险评估:")
    print("- 高复杂度模型 + 小数据集 = 高过拟合风险")
    print("- 需要更强的正则化或更简单的模型")

if __name__ == '__main__':
    df = analyze_attention_overfitting()
    analyze_model_complexity()
