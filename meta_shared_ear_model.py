#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
群体-个体差异的元学习SharedEarComparisonModel
========================

主线：群体共享的耳朵比较机制 + 个体特有的适配策略
使用元学习框架捕捉群体一致性和个体差异性

核心思想：
- 群体层面：学习普遍的耳朵比较模式
- 个体层面：个性化适配参数
- 元学习：快速适应个体差异

运行方式：
python meta_shared_ear_model.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class MetaSharedEarConfig:
    """元学习SharedEar配置"""
    # 群体共享参数
    shared_hidden_dim: int = 64
    dropout: float = 0.2

    # 个体适配参数
    adaptation_lr: float = 0.01  # 内循环学习率
    meta_lr: float = 1e-3       # 外循环学习率
    adaptation_steps: int = 5   # 内循环适配步数

    # 训练参数
    meta_epochs: int = 50
    support_samples: int = 20   # 支撑集大小
    query_samples: int = 20     # 查询集大小
    batch_size: int = 4         # 元学习批大小

    # 评估参数
    cv_folds: int = 5
    test_adaptation_steps: int = 10

    # 设备设置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class MetaSharedEarComparisonModel(nn.Module):
    """
    元学习版本的SharedEarComparisonModel

    包含：
    - 群体共享的特征处理和比较机制
    - 个体特有的适配参数
    - 支持元学习的快速适配
    """

    def __init__(self, ear_feature_dim: int, config: MetaSharedEarConfig):
        super().__init__()
        self.ear_feature_dim = ear_feature_dim
        self.config = config

        # 群体共享的特征门控
        self.shared_feature_gate = nn.Sequential(
            nn.Linear(ear_feature_dim, config.shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.shared_hidden_dim, ear_feature_dim),
            nn.Sigmoid()
        )

        # 群体共享的评分器
        self.shared_scorer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(ear_feature_dim, config.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.shared_hidden_dim, 1),
        )

        # 个体适配参数（群体共享的适配模板）
        self.adaptation_params = nn.ParameterDict({
            'feature_bias': nn.Parameter(torch.randn(ear_feature_dim) * 0.1),
            'attention_bias': nn.Parameter(torch.randn(ear_feature_dim) * 0.1),
            'decision_threshold': nn.Parameter(torch.randn(1) * 0.1)
        })

        # 参数初始化
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.01)

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                adaptation_params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            left_feat: (batch, ear_dim)
            right_feat: (batch, ear_dim)
            adaptation_params: 个体适配参数（可选）
        """

        # 群体共享的特征处理
        left_gated = left_feat * self.shared_feature_gate(left_feat)
        right_gated = right_feat * self.shared_feature_gate(right_feat)

        # 应用个体适配（如果提供）
        if adaptation_params is not None:
            # 确保适配参数与输入batch size匹配
            batch_size = left_gated.size(0)
            feature_dim = left_gated.size(1)

            # 获取适配参数
            feature_bias = adaptation_params['feature_bias']
            attention_bias = adaptation_params['attention_bias']

            # 适配参数应该是 (feature_dim,)，需要扩展为 (batch_size, feature_dim)
            if feature_bias.dim() == 1 and feature_bias.size(0) == feature_dim:
                feature_bias = feature_bias.unsqueeze(0).expand(batch_size, -1)
            elif feature_bias.size(0) != batch_size:
                # 如果维度不匹配，使用零向量
                feature_bias = torch.zeros_like(left_gated)

            if attention_bias.dim() == 1 and attention_bias.size(0) == feature_dim:
                attention_bias = attention_bias.unsqueeze(0).expand(batch_size, -1)
            elif attention_bias.size(0) != batch_size:
                # 如果维度不匹配，使用零向量
                attention_bias = torch.zeros_like(left_gated)

            # 调试信息
            print(f"DEBUG: left_gated.shape={left_gated.shape}, feature_bias.shape={feature_bias.shape}")
            print(f"DEBUG: right_gated.shape={right_gated.shape}, attention_bias.shape={attention_bias.shape}")

            left_gated = left_gated + feature_bias
            right_gated = right_gated + attention_bias

        # 评分和比较
        left_score = self.shared_scorer(left_gated)
        right_score = self.shared_scorer(right_gated)

        # 决策
        score_diff = left_score - right_score
        choice_prob = torch.sigmoid(score_diff).squeeze(-1)

        return choice_prob

    def get_base_adaptation_params(self) -> Dict[str, torch.Tensor]:
        """获取基础适配参数（群体模板）"""
        # 确保参数在模型所在的设备上
        device = next(self.parameters()).device
        return {
            'feature_bias': self.adaptation_params['feature_bias'].to(device),
            'attention_bias': self.adaptation_params['attention_bias'].to(device),
            'decision_threshold': self.adaptation_params['decision_threshold'].to(device)
        }


class MetaSharedEarLearner:
    """
    元学习器：学习如何快速适配个体
    """

    def __init__(self, model: MetaSharedEarComparisonModel, config: MetaSharedEarConfig):
        self.model = model
        self.config = config
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_lr)
        self.criterion = nn.BCELoss()

    def adapt_to_subject(self, support_left: torch.Tensor, support_right: torch.Tensor,
                        support_targets: torch.Tensor, base_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        内循环：基于个体数据快速适配参数
        """
        # 复制基础参数用于适配，确保在相同设备上
        device = support_left.device
        adapted_params = {k: v.clone().detach().to(device).requires_grad_(True)
                         for k, v in base_params.items()}

        # 内循环优化器
        inner_optimizer = torch.optim.SGD(adapted_params.values(), lr=self.config.adaptation_lr)

        # 内循环适配
        for _ in range(self.config.adaptation_steps):
            inner_optimizer.zero_grad()

            # 前向传播 - 这里传入的adapted_params应该与输入数据batch size匹配
            preds = self.model(support_left, support_right, adapted_params)

            # 计算损失
            loss = self.criterion(preds, support_targets)

            # 反向传播
            loss.backward()
            inner_optimizer.step()

        return adapted_params

    def meta_learning_step(self, support_batch: List[Tuple], query_batch: List[Tuple]) -> float:
        """
        外循环：基于多个被试的适配表现更新元学习器
        """
        total_meta_loss = 0.0

        for support_data, query_data in zip(support_batch, query_batch):
            # 解包数据
            support_left, support_right, support_targets = support_data
            query_left, query_right, query_targets = query_data

            # 获取基础适配参数（为每个被试独立）
            base_params = self.model.get_base_adaptation_params()

            # 内循环适配 - 为这个被试的数据
            adapted_params = self.adapt_to_subject(
                support_left, support_right, support_targets, base_params
            )

            # 外循环评估：用适配参数在查询集上表现
            query_preds = self.model(query_left, query_right, adapted_params)
            query_loss = self.criterion(query_preds, query_targets)
            total_meta_loss += query_loss

        # 平均元损失
        meta_loss = total_meta_loss / len(support_batch)

        # 更新元学习器
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


def create_subject_datasets(base_dataset, n_subjects: int = 9, samples_per_subject: int = 50, test_subjects: int = 2):
    """
    创建被试数据集：训练被试 + 测试被试

    Args:
        base_dataset: 基础数据集
        n_subjects: 总被试数
        samples_per_subject: 每个被试的样本数
        test_subjects: 测试被试数

    Returns:
        train_datasets: 训练被试数据集（用于元学习）
        test_datasets: 测试被试数据集（用于最终评估）
    """
    np.random.seed(42)

    # 计算训练和测试被试数
    n_train_subjects = n_subjects - test_subjects

    all_datasets = {}
    used_indices = set()

    # 先创建所有被试的数据集
    for subj_id in range(n_subjects):
        # 随机选择该被试的样本（不重复）
        available_indices = list(set(range(len(base_dataset))) - used_indices)
        if len(available_indices) < samples_per_subject:
            available_indices = list(range(len(base_dataset)))

        subject_indices = np.random.choice(
            available_indices,
            size=min(samples_per_subject, len(available_indices)),
            replace=False
        )

        all_datasets[f"subject_{subj_id}"] = Subset(base_dataset, subject_indices)
        used_indices.update(subject_indices)

    # 分割为训练和测试被试
    train_datasets = {k: v for k, v in all_datasets.items() if int(k.split('_')[1]) < n_train_subjects}
    test_datasets = {k: v for k, v in all_datasets.items() if int(k.split('_')[1]) >= n_train_subjects}

    print(f"训练被试: {len(train_datasets)} 个")
    print(f"测试被试: {len(test_datasets)} 个")

    return train_datasets, test_datasets


def train_meta_shared_ear_learner(meta_learner: MetaSharedEarLearner,
                                subject_datasets: Dict,
                                config: MetaSharedEarConfig) -> Dict:
    """训练元学习器"""

    print("=== 开始元学习训练 ===")

    training_history = {'meta_loss': []}
    best_loss = float('inf')
    patience = 0

    for epoch in range(config.meta_epochs):
        # 随机选择一批被试
        subject_ids = list(subject_datasets.keys())
        batch_subject_ids = np.random.choice(
            subject_ids,
            size=min(config.batch_size, len(subject_ids)),
            replace=False
        )

        # 准备批次数据
        support_batch = []
        query_batch = []

        for subj_id in batch_subject_ids:
            dataset = subject_datasets[subj_id]
            n_samples = len(dataset)

            # 分割支撑集和查询集
            indices = np.random.permutation(n_samples)
            support_size = min(config.support_samples, n_samples // 2)
            query_size = min(config.query_samples, n_samples - support_size)

            support_indices = indices[:support_size]
            query_indices = indices[support_size:support_size + query_size]

            # 加载数据
            support_loader = DataLoader(Subset(dataset, support_indices),
                                      batch_size=len(support_indices), shuffle=False)
            query_loader = DataLoader(Subset(dataset, query_indices),
                                    batch_size=len(query_indices), shuffle=False)

            support_left, support_right, support_targets = next(iter(support_loader))
            query_left, query_right, query_targets = next(iter(query_loader))

            # 移动到指定设备
            support_left = support_left.to(config.device)
            support_right = support_right.to(config.device)
            query_left = query_left.to(config.device)
            query_right = query_right.to(config.device)

            # 转换为二分类目标
            support_targets = (support_targets == 2).float().to(config.device)
            query_targets = (query_targets == 2).float().to(config.device)

            support_batch.append((support_left, support_right, support_targets))
            query_batch.append((query_left, query_right, query_targets))

        # 元学习步
        meta_loss = meta_learner.meta_learning_step(support_batch, query_batch)
        training_history['meta_loss'].append(meta_loss)

        if epoch % 10 == 0:
            print(".4f")

        # 早停
        if meta_loss < best_loss:
            best_loss = meta_loss
            patience = 0
        else:
            patience += 1

        if patience >= 10:
            print(f"早停于第{epoch}轮")
            break

    print("=== 元学习训练完成 ===")
    return training_history


def evaluate_subject(meta_learner: MetaSharedEarLearner,
                    subject_dataset: Dataset,
                    subject_id: str,
                    config: MetaSharedEarConfig) -> Dict:
    """评估单个被试"""

    print(f"\n评估被试: {subject_id}")

    # 加载数据
    data_loader = DataLoader(subject_dataset, batch_size=len(subject_dataset), shuffle=False)
    left_global, right_global, targets = next(iter(data_loader))

    # 移动到指定设备
    left_global = left_global.to(config.device)
    right_global = right_global.to(config.device)
    targets = targets.to(config.device)

    targets_binary = (targets == 2).cpu().numpy().astype(int)

    n_samples = len(targets_binary)

    # 交叉验证
    cv_accuracies = []
    cv_aucs = []

    indices = np.random.permutation(n_samples)
    fold_size = n_samples // config.cv_folds

    for fold in range(config.cv_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < config.cv_folds - 1 else n_samples

        test_indices = indices[test_start:test_end]
        train_indices = np.setdiff1d(indices, test_indices)

        # 准备数据
        train_left = left_global[train_indices]
        train_right = right_global[train_indices]
        train_targets = (targets[train_indices] == 2).float()

        test_left = left_global[test_indices]
        test_right = right_global[test_indices]
        test_targets = targets_binary[test_indices]

        # 获取基础参数并适配
        base_params = meta_learner.model.get_base_adaptation_params()
        adapted_params = meta_learner.adapt_to_subject(
            train_left, train_right, train_targets, base_params
        )

        # 评估
        with torch.no_grad():
            test_preds = meta_learner.model(test_left, test_right, adapted_params)
            test_preds_binary = (test_preds > 0.5).cpu().numpy().astype(int)

        acc = accuracy_score(test_targets, test_preds_binary)
        try:
            auc = roc_auc_score(test_targets, test_preds.cpu().numpy())
        except:
            auc = np.nan

        cv_accuracies.append(acc)
        cv_aucs.append(auc)

    # 全量评估
    full_targets = (targets == 2).float()
    base_params = meta_learner.model.get_base_adaptation_params()
    adapted_params = meta_learner.adapt_to_subject(
        left_global, right_global, full_targets, base_params
    )

    with torch.no_grad():
        final_preds = meta_learner.model(left_global, right_global, adapted_params)

    final_preds_binary = (final_preds > 0.5).cpu().numpy().astype(int)
    final_accuracy = accuracy_score(targets_binary, final_preds_binary)
    baseline_accuracy = max(np.mean(targets_binary), 1 - np.mean(targets_binary))

    try:
        final_auc = roc_auc_score(targets_binary, final_preds.cpu().numpy())
    except:
        final_auc = np.nan

    cm = confusion_matrix(targets_binary, final_preds_binary)

    results = {
        "subject_id": subject_id,
        "model_name": "MetaSharedEarComparisonModel",
        "accuracy": final_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "auc": final_auc,
        "cv_accuracy_mean": np.mean(cv_accuracies),
        "cv_accuracy_std": np.std(cv_accuracies),
        "cv_auc_mean": np.nanmean(cv_aucs),
        "cv_auc_std": np.nanstd(cv_aucs),
        "confusion_matrix": cm,
        "y_true": targets_binary,
        "y_pred": final_preds_binary,
        "y_pred_proba": final_preds.numpy(),
        "n_samples": n_samples,
        "cv_accuracies": cv_accuracies,
        "cv_aucs": cv_aucs
    }

    print(".4f")
    print(".4f")

    return results


def run_meta_shared_ear_analysis():
    """运行群体-个体差异的元学习SharedEar分析"""

    print("=" * 60)
    print("群体-个体差异的元学习SharedEarComparisonModel")
    print("=" * 60)

    # 配置
    config = MetaSharedEarConfig()

    # 创建模拟数据集（实际应用中应该用真实数据）
    try:
        from feature_stage_attention_model import build_datasets, DEFAULT_STAGE_SECONDS
    except ImportError:
        print("错误：无法导入feature_stage_attention_model，请确保文件存在")
        return None, None

    # 加载基础数据集
    base_dataset = build_datasets(DEFAULT_STAGE_SECONDS)

    # 创建被试数据集：训练被试 + 测试被试
    train_datasets, test_datasets = create_subject_datasets(base_dataset)

    print(f"训练被试数据集:")
    for subj_id, dataset in train_datasets.items():
        print(f"  {subj_id}: {len(dataset)} 个样本")

    print(f"测试被试数据集:")
    for subj_id, dataset in test_datasets.items():
        print(f"  {subj_id}: {len(dataset)} 个样本")

    # 初始化模型 - 获取ear特征维度
    # ear特征维度可以通过stage_dataset获取
    ear_dim = base_dataset.left_feat_dim

    model = MetaSharedEarComparisonModel(ear_dim, config).to(config.device)
    meta_learner = MetaSharedEarLearner(model, config)

    # 训练元学习器（只用训练被试）
    training_history = train_meta_shared_ear_learner(meta_learner, train_datasets, config)

    # 评估测试被试（真正的未见数据）
    test_results = []
    output_dir = "meta_shared_ear_results"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 测试被试评估（未见数据）===")
    for subject_id, dataset in test_datasets.items():
        result = evaluate_subject(meta_learner, dataset, subject_id, config)
        test_results.append(result)

    # 可选：也评估训练被试（用于对比）
    print("\n=== 训练被试评估（用于对比）===")
    train_results = []
    for subject_id, dataset in train_datasets.items():
        result = evaluate_subject(meta_learner, dataset, subject_id, config)
        train_results.append(result)

    all_results = train_results + test_results

    # 保存结果
    summary_data = []
    for result in all_results:
        is_test_subject = result['subject_id'] in test_datasets
        summary_data.append({
            'subject_id': result['subject_id'],
            'subject_type': 'test' if is_test_subject else 'train',
            'model_name': result['model_name'],
            'accuracy': result['accuracy'],
            'baseline_accuracy': result['baseline_accuracy'],
            'improvement': result['accuracy'] - result['baseline_accuracy'],
            'auc': result['auc'],
            'cv_accuracy_mean': result['cv_accuracy_mean'],
            'cv_accuracy_std': result['cv_accuracy_std'],
            'n_samples': result['n_samples']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['subject_type', 'cv_accuracy_mean'], ascending=[True, False])
    summary_path = os.path.join(output_dir, 'meta_shared_ear_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # 计算训练和测试的平均性能
    train_summary = summary_df[summary_df['subject_type'] == 'train']
    test_summary = summary_df[summary_df['subject_type'] == 'test']

    print("\n=== 群体-个体差异分析结果 ===")

    if len(train_summary) > 0:
        print("训练被试性能:")
        print(".4f")
        print(".4f")

    if len(test_summary) > 0:
        print("测试被试性能（未见数据）:")
        print(".4f")
        print(".4f")

    print(f"结果已保存到: {output_dir}")

    return {
        'all_results': all_results,
        'train_results': train_results,
        'test_results': test_results,
        'training_history': training_history
    }


if __name__ == "__main__":
    results_dict = run_meta_shared_ear_analysis()
    print("\n运行完成！")
    print("训练被试数量:", len(results_dict['train_results']))
    print("测试被试数量:", len(results_dict['test_results']))
