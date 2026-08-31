#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""根据 within_subject_rating_correlations.csv 绘制显著性热力图。"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_significance_figure(csv_path, output_path=None, p_threshold=0.06):
    """读取相关系数结果并绘制热力图，标注显著性。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Preserve blank cells if r/p missing
    dimension_order = ['Familiarity', 'Urgency', 'Liking']
    df['dimension'] = pd.Categorical(df['dimension'], categories=dimension_order, ordered=True)
    df = df.sort_values(['subject_id', 'dimension'])

    pivot_r = df.pivot(index='subject_id', columns='dimension', values='r').reindex(columns=dimension_order)
    pivot_p = df.pivot(index='subject_id', columns='dimension', values='p').reindex(columns=dimension_order)

    # 准备显示文本
    annot = pivot_r.copy().astype(object)

    for subj in pivot_r.index:
        for dim in pivot_r.columns:
            r_val = pivot_r.loc[subj, dim]
            p_val = pivot_p.loc[subj, dim]

            if pd.isna(r_val) or pd.isna(p_val):
                annot.loc[subj, dim] = ''
                continue

            star = '*' if p_val < p_threshold else ''
            annot.loc[subj, dim] = f"{r_val:.2f}{star}\n(p={p_val:.2f})"

    # 绘图
    plt.figure(figsize=(10, max(4, len(pivot_r.index) * 0.5)))
    ax = sns.heatmap(
        pivot_r,
        annot=annot,
        fmt='',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '相关系数 r'},
        vmin=-0.5,
        vmax=0.5
    )

    # Highlight significant cells
    n_rows, n_cols = pivot_r.shape
    for i in range(n_rows):
        for j in range(n_cols):
            r_val = pivot_r.iloc[i, j]
            p_val = pivot_p.iloc[i, j]
            if pd.notna(r_val) and pd.notna(p_val) and p_val < p_threshold:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='#FFD700', linewidth=2.5))

    ax.set_title('Within-Subject Rating vs. Choice Correlations (p < 0.06 marked)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rating Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Participant ID', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Significance heatmap saved: {output_path}")

    plt.show()


def main():
    csv_path = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\correlation_results_by_type\within_subject_rating_correlations.csv"
    output_path = r"D:\D\research\audioset下载\展示的图片\within_subject_significance_heatmap.png"

    create_significance_figure(csv_path, output_path=output_path, p_threshold=0.06)


if __name__ == "__main__":
    main()


