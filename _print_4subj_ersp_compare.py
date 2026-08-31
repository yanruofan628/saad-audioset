#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import pandas as pd

base = Path(r'A:/standard_data_interp_ica_99/nme_fusion_results/individual')
paths = {
    'baseline': base / 'gated_ablation' / 'residual_gated' / 'mirror' / 'all_subjects_summary.csv',
    'baseline_rerun': base / 'gated_eeg_internal' / 'residual_gated' / 'mirror' / 'all_subjects_summary.csv',
    'external_ersp': base / 'gated_ablation' / 'residual_gated_ersp' / 'mirror' / 'all_subjects_summary.csv',
    'early_ersp': base / 'gated_eeg_internal' / 'residual_gated_eeg_ersp_early' / 'mirror' / 'all_subjects_summary.csv',
}

subjects = ['yanxingzhuo', 'jinxiaoyue', 'chenxianwei', 'yeziyuan']
series = {}
for name, p in paths.items():
    if not p.exists():
        print(f'missing: {name}')
        continue
    df = pd.read_csv(p)
    bcol = [c for c in df.columns if 'balanced' in c][0]
    s = df.set_index('subject')[bcol]
    series[name] = s
    print(f'loaded {name}: n={len(df)}')

rows = []
for subj in subjects:
    row = {'subject': subj}
    for name, s in series.items():
        if subj in s.index:
            row[name] = float(s.loc[subj])
    rows.append(row)

out = pd.DataFrame(rows)
print('\n四人 BACC 对比:')
print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()
for name in series:
    vals = out[name].dropna() if name in out.columns else []
    if len(vals):
        print(f'{name} 平均 BACC = {vals.mean():.4f}')
