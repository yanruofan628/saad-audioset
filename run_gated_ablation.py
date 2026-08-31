#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""依次跑 residual_gated 基线 + spatial / ersp / plv 消融（mirror 单人 CV）。"""
from eeg_audio_residual import main_individual, OUTPUT_DIR

ABLATION_MODELS = [
    'residual_gated',
    'residual_gated_spatial',
    'residual_gated_ersp',
    'residual_gated_plv',
]

SUBJECTS = ['yanxingzhuo', 'jinxiaoyue', 'chenxianwei', 'yeziyuan']


def main():
    ablation_root = f'{OUTPUT_DIR}/individual/gated_ablation'
    for model_name in ABLATION_MODELS:
        print('\n' + '=' * 72)
        print(f'开始消融: {model_name}')
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
