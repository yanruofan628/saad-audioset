#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型诊断和调参辅助工具
帮助诊断模型问题，提供调参建议
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def diagnose_binary_classification(y_true, y_pred_proba, output_path=None):
    """
    诊断二分类模型
    
    参数:
        y_true: 真实标签 (0/1)
        y_pred_proba: 预测概率
        output_path: 可选，保存诊断图
    """
    print("\n=== 二分类模型诊断 ===")
    
    # 基本统计
    print(f"\n1. 数据分布:")
    print(f"   类别0 (右)数量: {np.sum(y_true == 0)} ({np.mean(y_true == 0)*100:.1f}%)")
    print(f"   类别1 (左)数量: {np.sum(y_true == 1)} ({np.mean(y_true == 1)*100:.1f}%)")
    print(f"   类别不平衡比例: {np.sum(y_true == 0) / np.sum(y_true == 1):.2f}:1")
    
    # 预测概率分布
    print(f"\n2. 预测概率分布:")
    print(f"   最小值: {y_pred_proba.min():.4f}")
    print(f"   最大值: {y_pred_proba.max():.4f}")
    print(f"   均值: {y_pred_proba.mean():.4f}")
    print(f"   中位数: {np.median(y_pred_proba):.4f}")
    print(f"   标准差: {y_pred_proba.std():.4f}")
    
    # 检查预测是否过于极端
    extreme_low = np.sum(y_pred_proba < 0.1)
    extreme_high = np.sum(y_pred_proba > 0.9)
    print(f"   极端预测 (<0.1): {extreme_low} ({extreme_low/len(y_pred_proba)*100:.1f}%)")
    print(f"   极端预测 (>0.9): {extreme_high} ({extreme_high/len(y_pred_proba)*100:.1f}%)")
    
    if extreme_low + extreme_high > len(y_pred_proba) * 0.8:
        print("   ⚠️  警告: 预测过于极端，模型可能过拟合或需要调整阈值")
    
    # 指标
    y_pred = (y_pred_proba > 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
        acc = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred_proba)
        
        print(f"\n3. 模型指标:")
        print(f"   AUC: {auc:.4f}")
        print(f"   准确率: {acc:.4f}")
        print(f"   Log Loss: {ll:.4f}")
        
        # AUC解释
        if auc < 0.6:
            print("   ⚠️  AUC < 0.6: 模型性能很差，可能比随机猜测还差")
        elif auc < 0.7:
            print("   ⚠️  AUC < 0.7: 模型性能较差，需要改进")
        elif auc < 0.8:
            print("   ✓ AUC 0.7-0.8: 模型性能一般，有改进空间")
        elif auc < 0.9:
            print("   ✓✓ AUC 0.8-0.9: 模型性能良好")
        else:
            print("   ✓✓✓ AUC > 0.9: 模型性能优秀")
    except Exception as e:
        print(f"   计算指标失败: {e}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n4. 混淆矩阵:")
    print(f"   真负例 (TN): {cm[0,0]}")
    print(f"   假正例 (FP): {cm[0,1]}")
    print(f"   假负例 (FN): {cm[1,0]}")
    print(f"   真正例 (TP): {cm[1,1]}")
    
    # 精确率和召回率
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[0,0]) if (cm[1,1] + cm[0,0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n5. 分类指标:")
    print(f"   精确率 (Precision): {precision:.4f}")
    print(f"   召回率 (Recall): {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    
    # 可视化
    if output_path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 预测概率分布
        ax = axes[0, 0]
        ax.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='实际: 右', color='coral')
        ax.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='实际: 左', color='steelblue')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='阈值=0.5')
        ax.set_xlabel('预测概率 (选择左)')
        ax.set_ylabel('频数')
        ax.set_title('预测概率分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 混淆矩阵
        ax = axes[0, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['预测: 右', '预测: 左'],
                   yticklabels=['实际: 右', '实际: 左'])
        ax.set_title('混淆矩阵')
        
        # 3. ROC曲线（简化版：概率分布对比）
        ax = axes[1, 0]
        ax.boxplot([y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]],
                   labels=['实际: 右', '实际: 左'])
        ax.axhline(0.5, color='red', linestyle='--', linewidth=2)
        ax.set_ylabel('预测概率')
        ax.set_title('预测概率箱线图')
        ax.grid(True, alpha=0.3)
        
        # 4. 指标总结
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
模型诊断报告

AUC: {auc:.4f}
准确率: {acc:.4f}
Log Loss: {ll:.4f}
精确率: {precision:.4f}
召回率: {recall:.4f}
F1分数: {f1:.4f}

类别分布:
  右: {np.sum(y_true == 0)} ({np.mean(y_true == 0)*100:.1f}%)
  左: {np.sum(y_true == 1)} ({np.mean(y_true == 1)*100:.1f}%)
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n诊断图已保存: {output_path}")
    
    return {
        'auc': auc if 'auc' in locals() else None,
        'accuracy': acc if 'acc' in locals() else None,
        'log_loss': ll if 'll' in locals() else None,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def suggest_hyperparameters(current_auc=None, current_loss=None):
    """
    根据当前模型性能提供调参建议
    """
    print("\n=== 调参建议 ===")
    
    if current_auc is not None:
        if current_auc < 0.6:
            print("\n🔴 模型性能很差 (AUC < 0.6):")
            print("   可能原因:")
            print("   1. 特征选择不当 - 尝试使用更多/不同的特征")
            print("   2. 数据质量问题 - 检查数据预处理和缺失值")
            print("   3. 模型过于简单 - 考虑增加模型复杂度")
            print("   4. 类别不平衡 - 尝试类别权重或采样")
            print("\n   建议:")
            print("   - 检查特征与目标的相关性")
            print("   - 尝试更简单的基线模型（如逻辑回归）")
            print("   - 增加数据量或使用数据增强")
        
        elif current_auc < 0.7:
            print("\n🟡 模型性能较差 (AUC 0.6-0.7):")
            print("   建议:")
            print("   1. 调整正则化强度:")
            print("      - 减小L2正则化 (增大C值): C=1.0, 10.0")
            print("      - 或增加L2正则化 (减小C值): C=0.1, 0.01")
            print("   2. 特征工程:")
            print("      - 尝试特征交互项")
            print("      - 特征选择（移除不相关特征）")
            print("   3. 模型复杂度:")
            print("      - 增加样条基函数数量: n_splines=15, 20")
            print("      - 或减少样条基函数数量: n_splines=5, 8")
        
        elif current_auc < 0.8:
            print("\n🟢 模型性能一般 (AUC 0.7-0.8):")
            print("   建议:")
            print("   1. 精细调参:")
            print("      - 调整平滑度参数: lam=0.3, 0.5, 0.8, 1.0")
            print("      - 调整样条数量: n_splines=8, 10, 12, 15")
            print("   2. 集成方法:")
            print("      - 尝试多个模型的集成")
            print("   3. 交叉验证:")
            print("      - 使用更严格的交叉验证评估")
        
        else:
            print("\n✅ 模型性能良好 (AUC > 0.8):")
            print("   可以尝试:")
            print("   1. 进一步优化:")
            print("      - 网格搜索超参数")
            print("      - 特征选择优化")
            print("   2. 模型解释:")
            print("      - 分析特征重要性")
            print("      - 可视化时间权重曲线")
    
    print("\n📝 通用调参策略:")
    print("   1. 学习率 (深度学习):")
    print("      - 太大: 1e-2, 5e-3 → 尝试: 1e-3, 5e-4")
    print("      - 太小: 1e-5, 1e-6 → 尝试: 1e-4, 1e-3")
    print("   2. 正则化:")
    print("      - 过拟合: 增加 weight_decay (1e-4 → 1e-3, 5e-4)")
    print("      - 欠拟合: 减少 weight_decay (1e-4 → 1e-5, 0)")
    print("   3. Dropout:")
    print("      - 过拟合: 增加 dropout (0.1 → 0.2, 0.3)")
    print("      - 欠拟合: 减少 dropout (0.2 → 0.1, 0.05)")
    print("   4. 批次大小:")
    print("      - 不稳定: 增大 batch_size (16 → 32, 64)")
    print("      - 内存不足: 减小 batch_size (32 → 16, 8)")
    print("   5. 训练轮数:")
    print("      - 早停: 使用验证集监控，patience=10-20")
    print("      - 充分训练: 至少50-100轮")


def compare_models(results_dict, output_path=None):
    """
    比较多个模型的结果
    
    参数:
        results_dict: {模型名称: {y_true, y_pred_proba, ...}}
    """
    print("\n=== 模型比较 ===")
    
    comparison_data = []
    for model_name, results in results_dict.items():
        try:
            auc = roc_auc_score(results['y_true'], results['y_pred_proba'])
            acc = accuracy_score(results['y_true'], (results['y_pred_proba'] > 0.5).astype(int))
            ll = log_loss(results['y_true'], results['y_pred_proba'])
            
            comparison_data.append({
                'model': model_name,
                'AUC': auc,
                'Accuracy': acc,
                'Log Loss': ll
            })
        except:
            pass
    
    df_compare = pd.DataFrame(comparison_data)
    df_compare = df_compare.sort_values('AUC', ascending=False)
    
    print("\n模型性能排名:")
    print(df_compare.to_string(index=False))
    
    if output_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['AUC', 'Accuracy', 'Log Loss']
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            df_compare_sorted = df_compare.sort_values(metric, ascending=(metric == 'Log Loss'))
            ax.barh(df_compare_sorted['model'], df_compare_sorted[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} 比较')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n比较图已保存: {output_path}")
    
    return df_compare


if __name__ == '__main__':
    print("模型诊断和调参辅助工具")
    print("=" * 50)
    print("\n使用示例:")
    print("""
    from model_diagnosis_helper import diagnose_binary_classification, suggest_hyperparameters
    
    # 诊断模型
    results = diagnose_binary_classification(y_true, y_pred_proba, 'diagnosis.png')
    
    # 获取调参建议
    suggest_hyperparameters(current_auc=results['auc'])
    """)

