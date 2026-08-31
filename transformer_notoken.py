#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer 注意力模型比较脚本：
- 构建与 attention_loudness_model 相同的 (2,T) 数据集
- 训练因果 Transformer 单耳/双耳模型
- 输出 MSE / MAE / R^2 指标用于对比
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from attention_loudness_model import (
    FeatureTimeSeriesDataset,
    load_or_extract_features,
    merge_balanced_pairs_by_type,
    get_trials,
    calculate_selection_probability_144,
    SR,
    HOP,
)


# ================= Positional Encoding =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, d_model)
        T = x.size(0)
        return x + self.pe[:T]


def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# ================= Transformer Models =================
class CausalTransformerSingle(nn.Module):
    """(2,T) -> 因果 Transformer -> 概率"""

    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, T)
        x = x.transpose(1, 2)  # (batch, T, 2)
        x = self.input_proj(x)  # (batch, T, d_model)
        x = x.transpose(0, 1)   # (T, batch, d_model)
        x = self.pos_encoding(x)
        mask = generate_causal_mask(x.size(0), x.device)
        encoded = self.encoder(x, mask=mask)  # (T, batch, d_model)
        last = encoded[-1]  # (batch, d_model)
        logits = self.head(last).squeeze(-1)
        return torch.sigmoid(logits)


class CausalTransformerDual(nn.Module):
    """左右耳分别编码，拼接后输出概率"""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.left_proj = nn.Linear(input_dim, d_model)
        self.right_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def encode_branch(self, seq: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        # seq: (batch, T, 1)
        seq = proj(seq)           # (batch, T, d_model)
        seq = seq.transpose(0, 1) # (T, batch, d_model)
        seq = self.pos_encoding(seq)
        mask = generate_causal_mask(seq.size(0), seq.device)
        encoded = self.encoder(seq, mask=mask)
        return encoded[-1]  # (batch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, T)
        left = x[:, 0, :].unsqueeze(-1)   # (batch, T, 1)
        right = x[:, 1, :].unsqueeze(-1)  # (batch, T, 1)
        left_ctx = self.encode_branch(left, self.left_proj)
        right_ctx = self.encode_branch(right, self.right_proj)
        combined = torch.cat([left_ctx, right_ctx], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return torch.sigmoid(logits)


# ================= Dataset Helper =================
class CachedArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class IndexedViewDataset(Dataset):
    """对 CachedArrayDataset 的索引视图，避免重复构建/加载特征"""
    def __init__(self, base: CachedArrayDataset, indices: np.ndarray):
        assert isinstance(base, CachedArrayDataset)
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.base.X[idx], self.base.y[idx]


def load_dataset(feature_type: str, project_root: str, base_dir: str, force_rebuild: bool = False) -> Dataset:
    cache_dir = os.path.join(project_root, 'transformer_cache', feature_type)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'dataset_cache.npz')

    if not force_rebuild and os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"  载入缓存数据集: {cache_path}")
        return CachedArrayDataset(data['X'], data['y'])

    print("  未找到缓存，开始构建数据集...")
    diff_raw, diff_z, pair_names = load_or_extract_features(project_root)
    merged = merge_balanced_pairs_by_type(pair_names)

    all_pair_names = []
    for t in ['nn_main', 'nn_sub', 'main', 'sub']:
        if t in merged:
            all_pair_names.extend(merged[t]['pair_names'])

    trials = get_trials()
    probs = calculate_selection_probability_144(all_pair_names, trials).astype(np.float32)

    dataset = FeatureTimeSeriesDataset(
        all_pair_names,
        base_dir,
        probs,
        feature_type,
        sr=SR,
        hop_length=HOP,
    )
    np.savez_compressed(cache_path, X=dataset.X, y=dataset.y)
    print(f"  数据集缓存已保存: {cache_path}")
    return CachedArrayDataset(dataset.X, dataset.y)


# ================= Training / Evaluation =================
@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_ratio: float = 0.1
    patience: int = 20
    seed: int = 42


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    if val_ratio <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def train_transformer(model: nn.Module, train_ds: Dataset, cfg: TrainConfig, device: torch.device, val_ds: Dataset = None) -> None:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        train_mse = total / max(count, 1)
        if val_loader is None:
            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch}/{cfg.epochs} - train MSE={train_mse:.6f}")
            continue
        # 验证
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_count = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_total += loss_fn(preds, yb).item() * xb.size(0)
                val_count += xb.size(0)
            val_mse = val_total / max(val_count, 1)
        improved = val_mse + 1e-12 < best_val
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{cfg.epochs} - train MSE={train_mse:.6f}  val MSE={val_mse:.6f}")
        if epochs_no_improve >= cfg.patience:
            print(f"    早停触发：在验证集上连续 {cfg.patience} 个epoch无提升（最佳 val MSE={best_val:.6f}）")
            break
    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate_model(model: nn.Module, dataset: Dataset, device: torch.device) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds_list = []
    targets_list = []
    mse_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            mse_sum += F.mse_loss(preds, yb, reduction='sum').item()
            preds_list.append(preds.cpu())
            targets_list.append(yb.cpu())
    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()
    mse = mse_sum / len(dataset)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float('nan') if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {'mse': float(mse), 'mae': mae, 'r2': r2}


def save_metrics(path: str, metrics: Dict[str, float]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_model(model_type: str, d_model: int, nhead: int, num_layers: int, dropout: float) -> nn.Module:
    if model_type == 'transformer_single':
        return CausalTransformerSingle(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_type == 'transformer_dual':
        return CausalTransformerDual(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"未知模型类型: {model_type}")


# ================= Main =================
def main():
    parser = argparse.ArgumentParser(description="Transformer 单耳/双耳模型训练与对比")
    parser.add_argument('--features', nargs='+', default=['temporal_mod'], help="特征类型列表")
    parser.add_argument('--model_types', nargs='+', default=['transformer_single', 'transformer_dual'])
    parser.add_argument('--project_root', default=os.getcwd())
    parser.add_argument('--base_dir', default=r"D:\D\research\audioset下载\clap_select")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--val_ratio', type=float, default=0.1, help=">0 则拆分验证集")
    parser.add_argument('--patience', type=int, default=20, help="早停耐心轮数")
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--rebuild_dataset', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
    )

    print("=== Transformer 模型比较 ===")
    for feature in args.features:
        print(f"\n=== 特征: {feature} ===")
        # 读取/构建全量缓存数据集，然后进行5折，仅运行首折
        full_ds = load_dataset(feature, args.project_root, args.base_dir, force_rebuild=args.rebuild_dataset)
        print(f"  全量样本数: {len(full_ds)}, 单样本形状: {full_ds[0][0].shape}")
        num_pairs = len(full_ds)
        indices = np.random.RandomState(cfg.seed).permutation(num_pairs)
        folds = np.array_split(indices, 5)

        for model_type in args.model_types:
            print(f"\n  -> 5折评估模型(仅首折): {model_type}")

            # 仅首折
            fold_idx = 0
            test_idx = folds[fold_idx]
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            train_full = IndexedViewDataset(full_ds, train_idx)
            test_ds = IndexedViewDataset(full_ds, test_idx)

            # 从训练折内再切分验证
            inner_val_ratio = cfg.val_ratio if cfg.val_ratio > 0 else 0.1
            train_ds, val_ds = split_dataset(train_full, inner_val_ratio, cfg.seed + fold_idx)

            print(f"    折 {fold_idx+1}/5 - 训练样本: {len(train_ds)} 验证样本: {len(val_ds) if val_ds is not None else 0} 测试样本: {len(test_ds)}")

            model = get_model(model_type, args.d_model, args.nhead, args.num_layers, args.dropout).to(device)
            train_transformer(model, train_ds, cfg, device, val_ds=val_ds)

            # 保存模型与指标到CV目录
            result_dir = os.path.join(args.project_root, 'transformer_results_cv5', feature, model_type, f'fold_{fold_idx+1}')
            ensure_dir(result_dir)
            model_path = os.path.join(result_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)

            metrics = evaluate_model(model, test_ds, device)
            metrics_path = os.path.join(result_dir, 'metrics.json')
            save_metrics(metrics_path, metrics)

            print(f"    保存模型: {model_path}")
            print(f"    MSE: {metrics['mse']:.6f}")
            print(f"    MAE: {metrics['mae']:.6f}")
            print(f"    R^2: {metrics['r2']:.6f}")


if __name__ == '__main__':
    main()


