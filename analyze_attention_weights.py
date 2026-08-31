#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析注意力权重的量化指标，并生成对比图。
读取目录 attention_loudness_results 下的：
- attention_weights_left.npy  (T,)
- attention_weights_right.npy (T,)
- attention_weights.npy       (T,)  可选：单一注意力版本存在时读取

输出：
- attention_metrics.csv   （三条原始曲线的指标）
- attention_metrics_smooth_k20.csv （三条在k=20帧平滑后的指标）
- attention_curve_with_bands.png  （曲线对比+分区均值）
- attention_smoothing_*.png （多尺度平滑叠加图，left/right/single）
- attention_weights_*_smooth_k{K}.npy （平滑后的权重）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.ndimage as ndimage

RESULT_DIR = os.path.join(os.getcwd(), 'attention_loudness_results')
SR = 16000
HOP = 256  # 16ms/帧


def load_vec(path):
    if os.path.exists(path):
        v = np.load(path)
        v = np.asarray(v, dtype=float)
        return v
    return None


def normalize(v):
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0, None)
    s = v.sum()
    return (v / s) if s > 0 else np.zeros_like(v)


def normalized_entropy(p):
    p = normalize(p) + 1e-12
    H = -(p * np.log(p)).sum()
    Hmax = np.log(len(p)) if len(p) > 0 else 1.0
    return H / Hmax


def gini_coefficient(x):
    x = normalize(x)
    if len(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x)
    gini = 1.0 - 2.0 * np.sum(cumx) / (n * np.sum(sorted_x)) + 1.0 / n
    return max(0.0, min(1.0, gini))


def peak_ratio(x):
    x = np.asarray(x, dtype=float)
    m = np.mean(x) if len(x) else 1.0
    return (np.max(x) / m) if m > 0 else 0.0


def topk_share(x, k_ratio=0.05):
    x = normalize(x)
    n = len(x)
    k = max(1, int(round(n * k_ratio)))
    return np.sum(np.sort(x)[-k:])


def linear_trend_slope(x):
    # a[t] vs t 的线性回归斜率（单位：每帧）
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x))
    slope, intercept, r, p, stderr = stats.linregress(t, x)
    return slope


def front_back_share(x, seconds=1.0):
    # 前seconds 与 后seconds 的权重占比
    x = normalize(x)
    frames = int(round(seconds * SR / HOP))
    if frames <= 0 or frames > len(x):
        return 0.0, 0.0
    front = x[:frames].sum()
    back = x[-frames:].sum()
    return front, back


def compute_metrics(name, v):
    metrics = {
        'series': name,
        'T': len(v),
        'entropy_norm': normalized_entropy(v),
        'gini': gini_coefficient(v),
        'peak_ratio': peak_ratio(v),
        'top5pct_share': topk_share(v, 0.05),
        'slope_per_frame': linear_trend_slope(v),
    }
    f1, b1 = front_back_share(v, 1.0)
    metrics['front1s_share'] = f1
    metrics['back1s_share'] = b1
    metrics['back_minus_front'] = b1 - f1
    return metrics


def plot_curves(series, labels, out_path):
    T = len(series[0])
    time_axis = np.arange(T) * (HOP / SR)
    plt.figure(figsize=(12, 4))
    for v, lbl in zip(series, labels):
        plt.plot(time_axis, v, linewidth=2, label=lbl)
    # 分区：前1s / 中段 / 后1s
    f = int(round(1.0 * SR / HOP))
    if T > 2 * f:
        plt.axvspan(0, f * (HOP / SR), color='gray', alpha=0.05)
        plt.axvspan((T - f) * (HOP / SR), T * (HOP / SR), color='gray', alpha=0.05)
    plt.xlabel('Time (s)')
    plt.ylabel('Attention')
    plt.title('Attention Curves (Left / Right / Single)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def smooth_and_plot(name, v, kernel_sizes=(5, 10, 20)):
    # 平滑并输出图与npy，返回 {k: smoothed}
    results = {}
    plt.figure(figsize=(12, 3))
    for k in kernel_sizes:
        sigma = k / 3.0
        smoothed = ndimage.gaussian_filter1d(v, sigma=sigma)
        results[k] = smoothed
        np.save(os.path.join(RESULT_DIR, f'attention_weights_{name}_smooth_k{k}.npy'), smoothed)
        plt.plot(smoothed, label=f'Smooth Window={k} frames')
    plt.plot(v, alpha=0.3, color='gray', label='Raw (16ms)')
    plt.legend()
    plt.xlabel('Time (16ms frames)')
    plt.ylabel('Smoothed Attention Weight')
    plt.title(f'Multi-scale Smoothing ({name})')
    out_png = os.path.join(RESULT_DIR, f'attention_smoothing_{name}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    return results


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    left = load_vec(os.path.join(RESULT_DIR, 'attention_weights_left.npy'))
    right = load_vec(os.path.join(RESULT_DIR, 'attention_weights_right.npy'))
    single = load_vec(os.path.join(RESULT_DIR, 'attention_weights.npy'))  # 可能不存在

    series = []
    labels = []
    if left is not None:
        series.append(left)
        labels.append('left')
    if right is not None:
        series.append(right)
        labels.append('right')
    if single is not None:
        series.append(single)
        labels.append('single')

    if not series:
        raise RuntimeError('未找到任何注意力权重文件。')

    # 原始指标
    rows = [compute_metrics(lbl, v) for lbl, v in zip(labels, series)]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, 'attention_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'指标已保存: {csv_path}')
    print(df.to_string(index=False))

    # 曲线对比与分区均值（前1s/中段/后1s）
    plot_curves(series, labels, os.path.join(RESULT_DIR, 'attention_curve_with_bands.png'))

    # 多尺度平滑与保存，并对k=20计算指标
    smooth_rows = []
    if left is not None:
        res = smooth_and_plot('left', left)
        if 20 in res:
            smooth_rows.append(compute_metrics('left_k20', res[20]))
    if right is not None:
        res = smooth_and_plot('right', right)
        if 20 in res:
            smooth_rows.append(compute_metrics('right_k20', res[20]))
    if single is not None:
        res = smooth_and_plot('single', single)
        if 20 in res:
            smooth_rows.append(compute_metrics('single_k20', res[20]))

    if smooth_rows:
        df2 = pd.DataFrame(smooth_rows)
        csv_path2 = os.path.join(RESULT_DIR, 'attention_metrics_smooth_k20.csv')
        df2.to_csv(csv_path2, index=False, encoding='utf-8-sig')
        print(f'平滑(k=20)指标已保存: {csv_path2}')
        print(df2.to_string(index=False))


if __name__ == '__main__':
    main()
