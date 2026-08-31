#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析“基于模型打分分组”的四条注意力曲线：
- attn_mean_scoreLeft_leftEar.npy
- attn_mean_scoreLeft_rightEar.npy
- attn_mean_scoreRight_leftEar.npy
- attn_mean_scoreRight_rightEar.npy

输出：
- attention_score_groups_metrics.csv        （原始曲线指标）
- attention_score_groups_metrics_k20.csv    （k=20帧平滑后的指标）
- attn_score_groups_curves.png              （四条曲线对比）
- attn_score_groups_smoothing_*.png         （每条的多尺度平滑叠加图）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy import stats

RESULT_DIR = os.path.join(os.getcwd(), 'attention_loudness_results')
SR = 16000
HOP = 256

FILES = [
    ('scoreLeft_leftEar',  'attn_mean_scoreLeft_leftEar.npy'),
    ('scoreLeft_rightEar', 'attn_mean_scoreLeft_rightEar.npy'),
    ('scoreRight_leftEar', 'attn_mean_scoreRight_leftEar.npy'),
    ('scoreRight_rightEar','attn_mean_scoreRight_rightEar.npy'),
]


def load_vec(path):
    return np.load(path) if os.path.exists(path) else None


def normalize(v):
    v = np.clip(np.asarray(v, dtype=float), 0, None)
    s = v.sum()
    return (v / s) if s > 0 else np.zeros_like(v)


def entropy_norm(v):
    p = normalize(v) + 1e-12
    H = -(p * np.log(p)).sum()
    return H / np.log(len(p))


def gini(v):
    p = normalize(v)
    x = np.sort(p)
    n = len(x)
    if n == 0:
        return 0.0
    cum = np.cumsum(x)
    return max(0.0, min(1.0, 1.0 - 2.0 * np.sum(cum) / (n * np.sum(x)) + 1.0 / n))


def peak_ratio(v):
    v = np.asarray(v, dtype=float)
    m = v.mean() if len(v) else 1.0
    return (v.max() / m) if m > 0 else 0.0


def top5(v):
    p = normalize(v)
    k = max(1, int(round(len(p) * 0.05)))
    return np.sum(np.sort(p)[-k:])


def slope(v):
    if len(v) < 2:
        return 0.0
    t = np.arange(len(v))
    s, *_ = stats.linregress(t, v)
    return s


def front_back(v, seconds=1.0):
    p = normalize(v)
    frames = int(round(seconds * SR / HOP))
    if frames <= 0 or frames > len(v):
        return 0.0, 0.0
    return p[:frames].sum(), p[-frames:].sum()


def metrics_row(name, v):
    f1, b1 = front_back(v, 1.0)
    return {
        'series': name,
        'T': len(v),
        'entropy_norm': entropy_norm(v),
        'gini': gini(v),
        'peak_ratio': peak_ratio(v),
        'top5pct_share': top5(v),
        'slope_per_frame': slope(v),
        'front1s_share': f1,
        'back1s_share': b1,
        'back_minus_front': b1 - f1,
    }


def smoothing(v, k):
    return ndimage.gaussian_filter1d(v, sigma=k/3.0)


def main():
    series, labels = [], []
    for name, fname in FILES:
        vec = load_vec(os.path.join(RESULT_DIR, fname))
        if vec is not None:
            series.append(np.asarray(vec, dtype=float))
            labels.append(name)
    if not series:
        raise RuntimeError('未找到分组注意力文件，请先运行 analyze_trial_level_attention.py')

    # 原始指标
    rows = [metrics_row(lbl, v) for lbl, v in zip(labels, series)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, 'attention_score_groups_metrics.csv'), index=False, encoding='utf-8-sig')
    print('原始指标已保存: attention_score_groups_metrics.csv')

    # 曲线对比
    T = len(series[0])
    time_axis = np.arange(T) * (HOP / SR)
    plt.figure(figsize=(12, 3.2))
    for v, lbl in zip(series, labels):
        plt.plot(time_axis, v, linewidth=2, label=lbl)
    plt.xlabel('Time (s)')
    plt.ylabel('Attention')
    plt.title('Attention Curves by Model Score Groups')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'attn_score_groups_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # k=20平滑 + 指标
    rows_k20 = []
    for v, lbl in zip(series, labels):
        v20 = smoothing(v, 20)
        rows_k20.append(metrics_row(lbl + '_k20', v20))
        # 单条叠加图
        plt.figure(figsize=(12, 3))
        for k in (5, 10, 20):
            plt.plot(smoothing(v, k), label=f'k={k}')
        plt.plot(v, alpha=0.3, color='gray', label='raw')
        plt.legend(); plt.xlabel('Time (16ms frames)'); plt.ylabel('Attention')
        plt.title(f'Multi-scale Smoothing ({lbl})')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f'attn_score_groups_smoothing_{lbl}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    pd.DataFrame(rows_k20).to_csv(
        os.path.join(RESULT_DIR, 'attention_score_groups_metrics_k20.csv'),
        index=False, encoding='utf-8-sig')
    print('k=20平滑指标已保存: attention_score_groups_metrics_k20.csv')


if __name__ == '__main__':
    main()
