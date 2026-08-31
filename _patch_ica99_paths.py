#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Point SUBJECTS_CONFIG processed_data_path to ica_99 (where trials exist)."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / 'eeg_audio_residual.py'
text = path.read_text(encoding='utf-8', errors='replace')
text2 = text.replace(
    "A:/standard_data_interp_no_ica/",
    "A:/standard_data_interp_ica_99/",
)
text2 = re.sub(
    r"OUTPUT_DIR = '[^']+'",
    "OUTPUT_DIR = 'A:/standard_data_interp_ica_99/nme_fusion_results'",
    text2,
    count=1,
)
path.write_text(text2, encoding='utf-8')
print('patched paths to ica_99')

# list runnable subjects
import os
import eeg_audio_residual as m
base = r'A:/standard_data_interp_ica_99'
ready = []
for name, cfg in m.SUBJECTS_CONFIG.items():
    p = cfg['processed_data_path']
    if os.path.isdir(p) and any(f.endswith('_trials.npy') for f in os.listdir(p)):
        ready.append(name)
print('runnable', len(ready), ready)
