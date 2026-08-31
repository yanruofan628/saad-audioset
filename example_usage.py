#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用示例：展示如何调用修改后的eeg_audio_fusion_individual.py

注意：现在直接指定被试名称进行处理
"""

from eeg_audio_fusion_individual import main

if __name__ == "__main__":
    # 选择要处理的被试
    subject_name = 'hanglei'  # 可以改为其他被试名称: 'yanxingzhuo', 'jinxiaoyue', 'chenxianwei', 'yeziyuan'

    print(f"处理被试: {subject_name}")
    main(subject_name=subject_name)