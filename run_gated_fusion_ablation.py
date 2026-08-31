#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""融合方式消融：SGCM / no-gate / cross-attn（mirror 单人 CV，五人）。

已完成的被试（存在 *_residual_cv_summary.txt）会自动跳过。
"""
import os
from eeg_audio_residual import main_individual, OUTPUT_DIR

ABLATION_MODELS = [
    'residual_gated',              # SGCM
    'residual_gated_nogate',       # concat MLP
    'residual_gated_cross_attn',   # Q=audio, K/V=EEG
]

SUBJECTS = [
    'yanxingzhuo',
    'jinxiaoyue',
    'chenxianwei',
    'yeziyuan',
    'hanglei',
]


def _done(output_base_dir, subject):
    return os.path.isfile(
        os.path.join(output_base_dir, 'mirror', f'{subject}_residual_cv_summary.txt')
    )


def main():
    ablation_root = f'{OUTPUT_DIR}/individual/gated_fusion_ablation'
    for model_name in ABLATION_MODELS:
        output_base_dir = f'{ablation_root}/{model_name}'
        remaining = [s for s in SUBJECTS if not _done(output_base_dir, s)]
        print('\n' + '=' * 72)
        print(f'融合消融: {model_name}')
        print(f'跳过已完成: {[s for s in SUBJECTS if s not in remaining]}')
        print(f'待跑: {remaining}')
        print('=' * 72)
        if not remaining:
            print(f'{model_name} 全部完成，跳过')
            continue
        main_individual(
            subjects_to_run=remaining,
            filter_consistent=False,
            models_to_train=[model_name],
            cv_scheme='mirror',
            output_base_dir=output_base_dir,
        )


if __name__ == '__main__':
    main()
