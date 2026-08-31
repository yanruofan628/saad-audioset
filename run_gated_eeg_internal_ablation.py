#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""内置 EEG 增强消融：通道注意力 / 多分支空间 / ERSP 早期融合（mirror 单人 CV）。"""
from eeg_audio_residual import main_individual, OUTPUT_DIR

ABLATION_MODELS = [
    'residual_gated',
    'residual_gated_eeg_attn',
    'residual_gated_eeg_multispatial',
    'residual_gated_eeg_ersp_early',
]

SUBJECTS = ['yanxingzhuo', 'jinxiaoyue', 'chenxianwei', 'yeziyuan']


def main():
    ablation_root = f'{OUTPUT_DIR}/individual/gated_eeg_internal'
    for model_name in ABLATION_MODELS:
        print('\n' + '=' * 72)
        print(f'开始内置 EEG 消融: {model_name}')
        print('=' * 72)
        main_individual(
            subjects_to_run=SUBJECTS,
            filter_consistent=False,
            models_to_train=[model_name],
            cv_scheme='mirror',
            output_base_dir=f'{ablation_root}/{model_name}',
        )


if __name__ == '__main__':
    main()
