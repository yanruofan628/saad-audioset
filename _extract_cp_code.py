import json
import re
from pathlib import Path

p = Path(
    r"C:/Users/Lenovo/.cursor/projects/d-D-research-audioset-download-audioset-master-download-audioset-master/agent-transcripts/b7171af4-d8f8-45cc-a568-d5a9e34f2f5f/b7171af4-d8f8-45cc-a568-d5a9e34f2f5f.jsonl"
)
out = Path(
    r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/_extract_cp.txt"
)
needles = [
    "class AMFMContentPriority",
    "am_fm_content_priority_fm",
    "def build_category_heldout_folds",
    "def get_pairing_type_key",
    "def build_global_pairing_type_fold_map",
]
found = []
for i, line in enumerate(p.open(encoding="utf-8", errors="ignore")):
    if not any(n in line for n in needles):
        continue
    if '"role":"assistant"' not in line and '"Write"' not in line and '"StrReplace"' not in line:
        # still parse
        pass
    obj = json.loads(line)
    parts = obj.get("message", {}).get("content", [])
    for c in parts:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "tool_use" and c.get("name") in ("Write", "StrReplace"):
            inp = c.get("input", {})
            text = inp.get("contents") or inp.get("new_string") or ""
            if any(n in text for n in needles):
                found.append(f"===== LINE {i} {c.get('name')} =====\n{text[:12000]}\n")

out.write_text("\n".join(found[:20]), encoding="utf-8")
print("blocks", len(found), "chars", sum(len(x) for x in found))
