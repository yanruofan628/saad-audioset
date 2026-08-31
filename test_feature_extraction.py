#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试特征提取函数"""
import numpy as np
import librosa
from pair_feature_diff_extraction import extract_time_series_features

# 创建一个5秒的测试音频
sr = 16000
duration = 5.0
y = np.random.randn(int(sr * duration))

print(f"音频长度: {len(y)} 采样点 ({duration}秒)")
print(f"测试 hop_length=8000")

try:
    feat_dict = extract_time_series_features(y, sr=sr, hop_length=8000)
    print(f"成功！特征字典键: {list(feat_dict.keys())}")
    for k, v in feat_dict.items():
        print(f"  {k}: shape={v.shape}, len={len(v)}")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

