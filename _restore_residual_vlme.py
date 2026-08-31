#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Restore vLME eeg_audio_residual and patch SUBJECTS_CONFIG from current backup."""
import os
import re
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
backup = root / 'eeg_audio_residual_backup_20260720.py'
current = root / 'eeg_audio_residual.py'
vlme = root / '_recovered' / 'vLME_eeg_audio_residual.py'

if not backup.exists():
    shutil.copy(current, backup)
    print('created backup from current')

shutil.copy(vlme, current)
print('restored vLME -> eeg_audio_residual.py')

bak = backup.read_text(encoding='utf-8', errors='replace')
cur = current.read_text(encoding='utf-8', errors='replace')

# SUBJECTS_CONFIG from backup (brace match)
s0 = bak.find('SUBJECTS_CONFIG = {')
i = bak.find('{', s0)
depth = 0
end_bak = None
for j, ch in enumerate(bak[i:], i):
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth == 0:
            end_bak = j + 1
            break
subj_block = bak[s0:end_bak]

c0 = cur.find('SUBJECTS_CONFIG = {')
i = cur.find('{', c0)
depth = 0
end_cur = None
for j, ch in enumerate(cur[i:], i):
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth == 0:
            end_cur = j + 1
            break

new = cur[:c0] + subj_block + cur[end_cur:]

m_bak = re.search(r"OUTPUT_DIR = '[^']+'", bak)
m_cur = re.search(r"OUTPUT_DIR = '[^']+'", new)
if m_bak and m_cur:
    new = new.replace(m_cur.group(0), m_bak.group(0), 1)
    print('OUTPUT_DIR ->', m_bak.group(0))

current.write_text(new, encoding='utf-8')
print('patched SUBJECTS_CONFIG')

# unique subject keys and disk check
keys = []
for line in subj_block.splitlines():
    line = line.strip()
    if line.startswith("'") and "': {" in line:
        keys.append(line.split("'")[1])
keys = list(dict.fromkeys(keys))
base = r'A:/standard_data_interp_no_ica'
ok = [k for k in keys if os.path.isdir(os.path.join(base, k))]
missing = [k for k in keys if k not in ok]
print('subjects in config:', len(keys))
print('data ok:', len(ok), ok)
print('missing:', missing)
