#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对“贡献曲线”做平滑与指标统计（k=20帧），并画出平滑后曲线：
读取 attention_loudness_results 下：
- contrib_scoreLeft_left_norm.npy / contrib_scoreLeft_right_norm.npy
- contrib_scoreRight_left_norm.npy / contrib_scoreRight_right_norm.npy
（可选）原始：
- contrib_scoreLeft_left_raw.npy / ...

输出：
- contrib_metrics_k20_norm.csv   （四条归一化贡献，k=20平滑后的指标）
- contrib_metrics_k20_raw.csv    （若存在raw文件则计算）
- contrib_curves_k20_norm.png    （四条归一化贡献的k=20平滑曲线对比）
- contrib_curves_k20_raw.png     （若存在raw则保存）
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

FILES_NORM = [
    ('scoreLeft_left_norm',  'contrib_scoreLeft_left_norm.npy'),
    ('scoreLeft_right_norm', 'contrib_scoreLeft_right_norm.npy'),
    ('scoreRight_left_norm', 'contrib_scoreRight_left_norm.npy'),
    ('scoreRight_right_norm','contrib_scoreRight_right_norm.npy'),
]

FILES_RAW = [
    ('scoreLeft_left_raw',  'contrib_scoreLeft_left_raw.npy'),
    ('scoreLeft_right_raw', 'contrib_scoreLeft_right_raw.npy'),
    ('scoreRight_left_raw', 'contrib_scoreRight_left_raw.npy'),
    ('scoreRight_right_raw','contrib_scoreRight_right_raw.npy'),
]

def load_series(file_list):
    series, labels = [], []
    for name, fname in file_list:
        path = os.path.join(RESULT_DIR, fname)
        if os.path.exists(path):
            v = np.load(path).astype(float)
            series.append(v)
            labels.append(name)
    return series, labels


def normalize_sum1(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s != 0 else v


def entropy_norm(v):
    p = normalize_sum1(np.abs(v)) + 1e-12  # 贡献允许有负号，指标用绝对值分布
    H = -(p * np.log(p)).sum()
    return H / np.log(len(p))


def gini(v):
    p = np.sort(normalize_sum1(np.abs(v)))
    n = len(p)
    if n == 0:
        return 0.0
    cum = np.cumsum(p)
    return max(0.0, min(1.0, 1.0 - 2.0 * np.sum(cum) / (n * np.sum(p)) + 1.0 / n))


def peak_ratio(v):
    a = np.abs(v)
    m = a.mean() if len(a) else 1.0
    return (a.max() / m) if m > 0 else 0.0


def top5(v):
    a = np.sort(np.abs(v))
    k = max(1, int(round(len(a) * 0.05)))
    return a[-k:].sum() / a.sum() if a.sum() > 0 else 0.0


def slope(v):
    if len(v) < 2:
        return 0.0
    t = np.arange(len(v))
    s, *_ = stats.linregress(t, v)
    return s


def front_back(v, seconds=1.0):
    a = np.abs(v)
    s = a.sum(); a = a / s if s > 0 else a
    frames = int(round(seconds * SR / HOP))
    if frames <= 0 or frames > len(a):
        return 0.0, 0.0
    return a[:frames].sum(), a[-frames:].sum()


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
        'front1s_share(|v|)': f1,
        'back1s_share(|v|)': b1,
        'back_minus_front(|v|)': b1 - f1,
    }


def k20(v):
    return ndimage.gaussian_filter1d(v, sigma=20/3.0)


def plot_curves(series, labels, out_png, title):
    T = len(series[0])
    x = np.arange(T) * (HOP / SR)
    plt.figure(figsize=(12, 3.2))
    for v, lbl in zip(series, labels):
        plt.plot(x, v, linewidth=2, label=lbl)
    plt.xlabel('Time (s)'); plt.ylabel('Contribution (k=20 smoothed)')
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 归一化贡献（首推，可比性更好）
    series, labels = load_series(FILES_NORM)
    if not series:
        raise RuntimeError('未找到 *_norm.npy，请先运行 analyze_trial_contrib_attention.py')
    series_k20 = [k20(v) for v in series]
    rows = [metrics_row(lbl + '_k20', v) for lbl, v in zip(labels, series_k20)]
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'contrib_metrics_k20_norm.csv'), index=False, encoding='utf-8-sig')
    plot_curves(series_k20, labels, os.path.join(RESULT_DIR, 'contrib_curves_k20_norm.png'), 'Contribution Curves (Normalized, k=20)')

    # 若有原始贡献，也做同样输出
    series_raw, labels_raw = load_series(FILES_RAW)
    if series_raw:
        series_raw_k20 = [k20(v) for v in series_raw]
        rows_raw = [metrics_row(lbl + '_k20', v) for lbl, v in zip(labels_raw, series_raw_k20)]
        pd.DataFrame(rows_raw).to_csv(os.path.join(RESULT_DIR, 'contrib_metrics_k20_raw.csv'), index=False, encoding='utf-8-sig')
        plot_curves(series_raw_k20, labels_raw, os.path.join(RESULT_DIR, 'contrib_curves_k20_raw.png'), 'Contribution Curves (Raw, k=20)')

    print('已生成：contrib_metrics_k20_norm.csv（及raw版本若存在）与 k=20 平滑曲线图')


if __name__ == '__main__':
    main()
