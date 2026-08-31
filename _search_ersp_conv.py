import json
from pathlib import Path

p = Path(
    r"C:/Users/Lenovo/.cursor/projects/d-D-research-audioset-download-audioset-master-download-audioset-master/agent-transcripts/1c9316a2-7988-4d49-a134-821f731ab12a/1c9316a2-7988-4d49-a134-821f731ab12a.jsonl"
)
out = Path(
    r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/_ersp_long_search.txt"
)

# Find user/assistant messages about ERSP baseline length change
needles = [
    "基线",
    "baseline",
    "ERSP",
    "ersp",
    "-1",
    "−1",
    "更长",
    "边缘",
    "Morlet",
]
lines_out = []
for i, line in enumerate(p.open(encoding="utf-8")):
    obj = json.loads(line)
    role = obj.get("role")
    parts = obj.get("message", {}).get("content", [])
    text = ""
    for c in parts:
        if isinstance(c, dict) and c.get("type") == "text":
            text += c.get("text", "")
    if not text:
        continue
    # Focus on plotting ERSP validation / all-trial, not model fusion
    if "plot_ersp" in text or ("ERSP" in text and ("基线" in text or "baseline" in text or "边缘" in text)):
        lines_out.append(f"===== LINE {i} role={role} =====\n{text[:6000]}\n")

out.write_text("\n".join(lines_out), encoding="utf-8")
print("n_blocks", len(lines_out), "chars", sum(len(x) for x in lines_out))
