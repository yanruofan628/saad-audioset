#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补跑 AM+FM residual_gated 基线（与 gated_ablation 同目录、同四人，便于对比）。"""
from eeg_audio_residual import main_individual, OUTPUT_DIR

SUBJECTS = ['yanxingzhuo', 'jinxiaoyue', 'chenxianwei', 'yeziyuan']
BASELINE_DIR = f'{OUTPUT_DIR}/individual/gated_ablation/residual_gated'


def main():
    print('=' * 72)
    print('AM+FM residual_gated 基线（mirror, 4 被试）')
    print('=' * 72)
    main_individual(
        subjects_to_run=SUBJECTS,
        filter_consistent=False,
        models_to_train=['residual_gated'],
        cv_scheme='mirror',
        output_base_dir=BASELINE_DIR,
    )


if __name__ == '__main__':
    main()
