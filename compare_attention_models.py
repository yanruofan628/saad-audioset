#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较单耳（single_attention）与双耳（dual_stream）注意力模型的效果。
默认读取 attention_loudness_model.py 训练后保存在 attention_all/{feature}/{model_type}/ 的模型文件，
并在同一数据集上计算 MSE、MAE、R²。

脚本会优先载入 attention_all/{feature}/dataset_cache.npz，避免重复提取特征。
若缓存不存在，仅首次运行时会提取特征并自动缓存。
"""
import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from attention_loudness_model import (
    FeatureTimeSeriesDataset,
    SimpleAttentionModel,
    DualStreamAttention,
    load_or_extract_features,
    merge_balanced_pairs_by_type,
    get_trials,
    calculate_selection_probability_144,
    SR,
    HOP,
)


class CachedArrayDataset(Dataset):
    """用于加载缓存的 (N,2,T) 与 (N,) 数组，避免重复特征提取。"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(feature_type: str, project_root: str, base_dir: str, force_rebuild: bool = False):
    """复用 attention_loudness_model.py 中的数据构建流程，带缓存。"""
    cache_dir = os.path.join(project_root, 'attention_all', feature_type)
    cache_path = os.path.join(cache_dir, 'dataset_cache.npz')

    if not force_rebuild and os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"  载入缓存数据集: {cache_path}")
        return CachedArrayDataset(data['X'], data['y'])

    os.makedirs(cache_dir, exist_ok=True)
    print("  未找到缓存，开始提取特征（首次运行耗时较长）...")
    diff_raw, diff_z, pair_names = load_or_extract_features(project_root)
    merged = merge_balanced_pairs_by_type(pair_names)

    all_pair_names = []
    for t in ['nn_main', 'nn_sub', 'main', 'sub']:
        if t in merged:
            all_pair_names.extend(merged[t]['pair_names'])

    trials = get_trials()
    probs = calculate_selection_probability_144(all_pair_names, trials).astype(np.float32)
    fdataset = FeatureTimeSeriesDataset(
        all_pair_names,
        base_dir,
        probs,
        feature_type,
        sr=SR,
        hop_length=HOP,
    )
    np.savez_compressed(cache_path, X=fdataset.X, y=fdataset.y)
    print(f"  已缓存数据集: {cache_path}")
    return CachedArrayDataset(fdataset.X, fdataset.y)


def load_model_for_feature(model_type: str, feature_type: str, dataset, project_root: str, device: torch.device):
    """根据模型类型实例化并加载参数。"""
    model_dir = os.path.join(project_root, 'attention_all', feature_type, model_type)
    ckpt = os.path.join(model_dir, 'attention_model.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"找不到模型文件: {ckpt}")

    time_steps = dataset[0][0].shape[1]
    if model_type == 'dual_stream':
        model = DualStreamAttention(time_steps=time_steps, dim=64).to(device)
    elif model_type == 'single_attention':
        model = SimpleAttentionModel(time_input_dim=2, d_model=64, attn_hidden=64).to(device)
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, model_dir


def evaluate_model(model, dataset, device) -> Dict[str, float]:
    """计算 MSE、MAE 与 R²。"""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_preds = []
    all_targets = []
    mse_sum = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if isinstance(model, DualStreamAttention):
                preds, _, _ = model(xb)
            else:
                preds, _ = model(xb)
            mse_sum += F.mse_loss(preds, yb, reduction='sum').item()
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    mse = mse_sum / len(dataset)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        r2 = float('nan')
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {'mse': float(mse), 'mae': mae, 'r2': r2}


def main():
    parser = argparse.ArgumentParser(description="比较单耳与双耳注意力模型性能")
    parser.add_argument('--feature', default='f0', help="特征类型，需与训练时一致")
    parser.add_argument('--project_root', default=os.getcwd(), help="项目根目录（含 attention_all 文件夹）")
    parser.add_argument('--base_dir', default=r"D:\D\research\audioset下载\clap_select", help="音频基础目录")
    parser.add_argument('--rebuild', action='store_true', help="忽略缓存，强制重新提取特征")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=== 构建数据集 ===")
    dataset = load_dataset(args.feature, args.project_root, args.base_dir, force_rebuild=args.rebuild)
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，无法评估。")
    print(f"  样本数: {len(dataset)}, 单样本形状: {dataset[0][0].shape}")

    results = {}
    for model_type in ['single_attention', 'dual_stream']:
        print(f"\n=== 评估模型: {model_type} ===")
        try:
            model, model_dir = load_model_for_feature(model_type, args.feature, dataset, args.project_root, device)
        except FileNotFoundError as e:
            print(f"  跳过：{e}")
            continue
        metrics = evaluate_model(model, dataset, device)
        results[model_type] = metrics
        print(f"  模型目录: {model_dir}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")

    if len(results) == 2:
        better = min(results.items(), key=lambda kv: kv[1]['mse'])[0]
        print(f"\n>>> MSE 更低的模型：{better}")
        better_r2 = max(results.items(), key=lambda kv: (float('-inf') if np.isnan(kv[1]['r2']) else kv[1]['r2']))[0]
        print(f">>> R² 更高的模型：{better_r2}")
    elif len(results) == 0:
        print("未找到任何模型，请先运行 attention_loudness_model.py 训练模型。")


if __name__ == '__main__':
    main()


