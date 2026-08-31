import json
from pathlib import Path

p = Path(
    r"C:/Users/Lenovo/.cursor/projects/d-D-research-audioset-download-audioset-master-download-audioset-master/agent-transcripts/b7171af4-d8f8-45cc-a568-d5a9e34f2f5f/b7171af4-d8f8-45cc-a568-d5a9e34f2f5f.jsonl"
)
out = Path(
    r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/_cp_class.txt"
)
chunks = []
for i, line in enumerate(p.open(encoding="utf-8", errors="ignore")):
    if "AMFMSalienceContentPriorityFMClassifier" not in line and "class AMFMSalienceContentPriority" not in line:
        continue
    obj = json.loads(line)
    for c in obj.get("message", {}).get("content", []):
        if not isinstance(c, dict) or c.get("type") != "tool_use":
            continue
        inp = c.get("input", {})
        text = inp.get("contents") or inp.get("new_string") or ""
        if "class AMFMSalienceContentPriority" in text:
            chunks.append(text)
out.write_text("\n\n#####\n\n".join(chunks[:5]), encoding="utf-8")
print("n", len(chunks), "len0", len(chunks[0]) if chunks else 0)
