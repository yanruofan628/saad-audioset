#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recover lost source files from Cursor agent transcripts (Write tool calls)."""
import json
import os
from pathlib import Path

transcripts = [
    Path(r'C:\Users\Lenovo\.cursor\projects\d-D-research-audioset-download-audioset-master-download-audioset-master\agent-transcripts\b7171af4-d8f8-45cc-a568-d5a9e34f2f5f\b7171af4-d8f8-45cc-a568-d5a9e34f2f5f.jsonl'),
    Path(r'C:\Users\Lenovo\.cursor\projects\d-D-research-audioset-download-audioset-master-download-audioset-master\agent-transcripts\1c9316a2-7988-4d49-a134-821f731ab12a\1c9316a2-7988-4d49-a134-821f731ab12a.jsonl'),
]

keywords = ('am_fm_salience', 'eeg_audio_residual', 'eeg_multiview', 'salience_beta')
writes = {}

for tp in transcripts:
    if not tp.exists():
        print('missing', tp)
        continue
    with tp.open('r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if '"Write"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            content = obj.get('message', {}).get('content', [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get('name') != 'Write':
                    continue
                inp = block.get('input') or {}
                path = inp.get('path') or ''
                contents = inp.get('contents')
                if not path or not contents:
                    continue
                if not any(k in path.replace('\\', '/') for k in keywords):
                    continue
                writes[path] = contents
                print(f'WRITE {Path(path).name} ({len(contents)} chars) line={i} from {tp.parent.name[:8]}')

out_dir = Path(r'd:\D\research\audioset下载\download_audioset-master\download_audioset-master\_recovered')
out_dir.mkdir(exist_ok=True)
for path, contents in writes.items():
    name = Path(path).name
    dest = out_dir / name
    dest.write_text(contents, encoding='utf-8')
    print('saved', dest, len(contents))

print('done, n=', len(writes))
