#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 ConvPoolOnlyModel 中不同特征通道的权重是否相同
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from attention_loudness_model import ConvPoolOnlyModel

def analyze_conv_weights(model_path, n_features=5, d_model=48):
    """
    分析卷积层权重，检查不同特征通道的权重是否相似
    
    参数:
        model_path: 模型文件路径
        n_features: 特征数量
        d_model: 模型维度
    """
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvPoolOnlyModel(n_features=n_features, d_model=d_model, dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"模型路径: {model_path}")
    print(f"特征数量: {n_features}, 模型维度: {d_model}")
    print("=" * 60)
    
    # 获取卷积层权重
    conv_left_weight = model.conv_left.weight.data.cpu().numpy()  # (d_model, n_features, 3)
    conv_right_weight = model.conv_right.weight.data.cpu().numpy()  # (d_model, n_features, 3)
    
    print(f"\n卷积层权重形状:")
    print(f"  conv_left: {conv_left_weight.shape}")
    print(f"  conv_right: {conv_right_weight.shape}")
    
    # 分析每个输出通道对不同输入特征通道的权重
    def analyze_feature_weights(weight, layer_name):
        """
        分析不同特征通道的权重相似度
        weight: (d_model, n_features, 3)
        """
        print(f"\n{layer_name} 权重分析:")
        print("-" * 60)
        
        # 对每个输出通道，计算不同输入特征通道权重的相似度
        feature_similarities = []
        
        for out_ch in range(d_model):
            # 获取该输出通道对所有输入特征通道的权重 (n_features, 3)
            feature_weights = weight[out_ch, :, :]  # (n_features, 3)
            
            # 计算不同特征通道之间的余弦相似度
            similarities = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    w_i = feature_weights[i, :].flatten()
                    w_j = feature_weights[j, :].flatten()
                    # 使用1减去余弦距离作为相似度
                    sim = 1 - cosine(w_i, w_j)
                    similarities.append(sim)
            
            if len(similarities) > 0:
                avg_sim = np.mean(similarities)
                feature_similarities.append(avg_sim)
        
        # 统计结果
        if len(feature_similarities) > 0:
            print(f"  不同特征通道权重的平均相似度:")
            print(f"    均值: {np.mean(feature_similarities):.4f}")
            print(f"    标准差: {np.std(feature_similarities):.4f}")
            print(f"    最小值: {np.min(feature_similarities):.4f}")
            print(f"    最大值: {np.max(feature_similarities):.4f}")
            
            # 如果相似度很高（>0.9），说明权重可能过于相似
            high_sim_count = np.sum(np.array(feature_similarities) > 0.9)
            print(f"    相似度>0.9的输出通道数: {high_sim_count}/{d_model} ({100*high_sim_count/d_model:.1f}%)")
        
        # 计算每个特征通道的平均权重大小
        print(f"\n  各特征通道的平均权重大小:")
        feature_avg_weights = np.mean(np.abs(weight), axis=(0, 2))  # (n_features,)
        for feat_idx in range(n_features):
            print(f"    特征{feat_idx}: {feature_avg_weights[feat_idx]:.6f}")
        
        # 计算特征通道权重的标准差（衡量不同特征通道权重的差异）
        print(f"\n  各特征通道权重的标准差（衡量差异）:")
        feature_std_weights = np.std(weight, axis=(0, 2))  # (n_features,)
        for feat_idx in range(n_features):
            print(f"    特征{feat_idx}: {feature_std_weights[feat_idx]:.6f}")
        
        # 可视化：不同特征通道的权重分布
        plt.figure(figsize=(12, 6))
        for feat_idx in range(n_features):
            feat_weights_flat = weight[:, feat_idx, :].flatten()
            plt.hist(feat_weights_flat, bins=50, alpha=0.5, label=f'特征{feat_idx}')
        plt.xlabel('权重值')
        plt.ylabel('频数')
        plt.title(f'{layer_name} - 不同特征通道的权重分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = model_path.replace('.pth', f'_{layer_name}_weight_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  权重分布图已保存: {output_path}")
        plt.close()
        
        return feature_similarities, feature_avg_weights, feature_std_weights
    
    # 分析左右卷积层
    left_sims, left_avg, left_std = analyze_feature_weights(conv_left_weight, "conv_left")
    right_sims, right_avg, right_std = analyze_feature_weights(conv_right_weight, "conv_right")
    
    # 比较左右卷积层的权重
    print(f"\n左右卷积层权重比较:")
    print("-" * 60)
    weight_diff = np.abs(conv_left_weight - conv_right_weight)
    print(f"  平均权重差异: {np.mean(weight_diff):.6f}")
    print(f"  最大权重差异: {np.max(weight_diff):.6f}")
    
    # 结论
    print(f"\n{'='*60}")
    print("结论:")
    avg_sim_left = np.mean(left_sims) if len(left_sims) > 0 else 0
    avg_sim_right = np.mean(right_sims) if len(right_sims) > 0 else 0
    
    if avg_sim_left > 0.9 or avg_sim_right > 0.9:
        print("⚠️  警告: 不同特征通道的权重相似度很高（>0.9）")
        print("   这可能意味着模型没有充分学习到不同特征的重要性差异")
        print("   建议:")
        print("   1. 检查特征是否标准化过度，导致分布过于相似")
        print("   2. 增加模型容量（d_model）")
        print("   3. 减少正则化强度（dropout, weight_decay）")
        print("   4. 考虑使用特征特定的投影层")
    else:
        print("✓ 不同特征通道的权重存在明显差异，模型能够区分不同特征")
    
    return {
        'left_similarities': left_sims,
        'right_similarities': right_sims,
        'left_avg_weights': left_avg,
        'right_avg_weights': right_avg,
        'left_std_weights': left_std,
        'right_std_weights': right_std,
    }


if __name__ == '__main__':
    import sys
    
    # 示例：检查一个训练好的模型
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        d_model = int(sys.argv[3]) if len(sys.argv) > 3 else 48
        
        if os.path.exists(model_path):
            analyze_conv_weights(model_path, n_features, d_model)
        else:
            print(f"错误: 模型文件不存在: {model_path}")
    else:
        print("用法: python check_convpool_weights.py <model_path> [n_features] [d_model]")
        print("\n示例:")
        print("  python check_convpool_weights.py attention_5fold/multi_feature/conv_pool_only/fold_1/attention_model.pth 5 48")

