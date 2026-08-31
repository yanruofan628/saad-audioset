import json
from pathlib import Path

roots = [
    Path(r"C:/Users/Lenovo/.cursor/projects/d-D-research-audioset-download-audioset-master-download-audioset-master/agent-transcripts"),
]
keys = [
    "baseline=(-1",
    "baseline=(-0.5",
    "−1.0",
    "-1.0 s",
    "-1 s",
    "基线 −1",
    "基线 -1",
    "更长基线",
    "基线更长",
    "pre_s = 1",
    "pre_s=1",
    "−1000",
    "-1000 ms",
    "1 s baseline",
    "1-s baseline",
    "longer baseline",
    "baseline_window",
    "BASELINE",
]

hits = []
for root in roots:
    for p in root.rglob("*.jsonl"):
        for i, line in enumerate(p.open(encoding="utf-8", errors="ignore")):
            if "ERSP" not in line and "ersp" not in line and "tfr_morlet" not in line:
                continue
            if any(k in line for k in keys):
                hits.append((str(p), i, line[:200]))

print("hits", len(hits))
for h in hits[:30]:
    print(h[0][-80:], "line", h[1])
    print(h[2][:180])
    print("---")
