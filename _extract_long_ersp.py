import json
from pathlib import Path

p = Path(
    r"C:/Users/Lenovo/.cursor/projects/d-D-research-audioset-download-audioset-master-download-audioset-master/agent-transcripts/1c9316a2-7988-4d49-a134-821f731ab12a/1c9316a2-7988-4d49-a134-821f731ab12a.jsonl"
)
out_dir = Path(
    r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master"
)

# Extract assistant messages around ERSP plotting (lines ~170-450)
for want in [170, 174, 420, 430, 431, 432, 434, 440, 441, 442, 448]:
    for i, line in enumerate(p.open(encoding="utf-8")):
        if i != want:
            continue
        obj = json.loads(line)
        role = obj.get("role")
        parts = obj.get("message", {}).get("content", [])
        texts = []
        for c in parts:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "tool_use" and c.get("name") == "Write":
                    inp = c.get("input", {})
                    path = inp.get("path", "")
                    contents = inp.get("contents", "")
                    texts.append(f"[Write {path}]\n{contents[:8000]}")
                elif c.get("type") == "tool_use" and c.get("name") == "StrReplace":
                    inp = c.get("input", {})
                    texts.append(
                        f"[StrReplace {inp.get('path','')}]\nNEW:\n{inp.get('new_string','')[:4000]}"
                    )
        text = "\n".join(texts)
        out = out_dir / f"_ersp_long_{want}.txt"
        out.write_text(text, encoding="utf-8")
        print(want, role, len(text), "preview:", text[:100].replace("\n", " "))
