#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot R-squared with significance markers for six model variants across all features."""

import os
import pandas as pd
import matplotlib.pyplot as plt


VARIANT_LABELS = {
    'mean_diff_raw': 'Accumulator Model',
    'max_diff': 'Peak Model',
    'first2s_diff': 'Primacy Model',
    'last2s_diff': 'Recency Model',
    'std_diff_raw': 'Fluctuation Model',
    'weighted_diff': 'Decayed Memory Model',
}

FAMILY_LABELS = {
    'f0': 'F0',
    'harmonicity': 'Harmonicity',
    'spectral_irregularity': 'Spectral Irregularity',
    'spectral_modulation_centroid': 'Spectral Modulation Centroid',
    'temporal_modulation': 'Temporal Modulation',
    'loudness': 'Loudness',
}


def main():
    project_root = os.getcwd()
    summary_path = os.path.join(
        project_root,
        'linear_regression_results_loudness_all',
        'loudness_models_summary.csv',
    )

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'未找到结果文件: {summary_path}')

    df = pd.read_csv(summary_path)

    rows = []
    for _, row in df.iterrows():
        feature_ids = str(row['feature_ids']).split('|')
        if len(feature_ids) != 1:
            continue
        fid = feature_ids[0]

        base_key = None
        for key in FAMILY_LABELS:
            if fid.startswith(f'{key}_'):
                base_key = key
                break
        if base_key is None:
            continue

        variant_part = fid[len(base_key) + 1:]
        if variant_part not in VARIANT_LABELS:
            # handle cases like mean_diff_raw etc.
            segments = variant_part.split('_')
            for i in range(len(segments)):
                candidate = '_'.join(segments[i:])
                if candidate in VARIANT_LABELS:
                    variant_part = candidate
                    break
        if variant_part not in VARIANT_LABELS:
            continue

        rows.append({
            'family': FAMILY_LABELS[base_key],
            'variant': VARIANT_LABELS[variant_part],
            'rsquared': row['rsquared'],
            'pvalue': row['f_pvalue'],
        })

    if not rows:
        raise RuntimeError('No matching models found in summary file')

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values(['family', 'variant'])

    families = plot_df['family'].unique()
    variants = list(VARIANT_LABELS.values())

    x_positions = []
    labels = []
    heights = []
    pvalues = []
    for family in families:
        subset = plot_df[plot_df['family'] == family]
        for variant in variants:
            match = subset[subset['variant'] == variant]
            if match.empty:
                continue
            x_positions.append(f'{family}\n{variant}')
            labels.append(variant)
            heights.append(match['rsquared'].values[0])
            pvalues.append(match['pvalue'].values[0])

    fig, ax = plt.subplots(figsize=(max(12, len(x_positions) * 0.6), 6))
    bars = ax.bar(range(len(x_positions)), heights, color='#4C78A8')

    for bar, pval in zip(bars, pvalues):
        height = bar.get_height()
        label = '*' if pval < 0.05 else 'n.s.'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            label,
            ha='center',
            va='bottom',
            fontsize=12,
            color='#222222',
            fontweight='bold' if pval < 0.05 else 'normal',
        )

    ax.set_xticks(range(len(x_positions)))
    ax.set_xticklabels(x_positions, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.set_ylabel('R-squared', fontsize=14, fontweight='bold')
    ax.set_title('Model Fits Across Acoustic Features', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(heights) * 1.2)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(project_root, 'linear_regression_results_loudness_all', 'all_feature_models_r2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'图像已保存: {output_path}')


if __name__ == '__main__':
    main()


