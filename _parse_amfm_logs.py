from pathlib import Path
import re, statistics as st
from collections import defaultdict

logs = [
    "mirror_amfm_vs_salience_beta.log",
    "mirror_amfm_vs_content_priority_subj1-5.log",
    "mirror_audio_vs_amfm_salience.log",
]
root = Path(r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master")
out_lines = []
for name in logs:
    p = root / name
    raw = p.read_bytes()
    text = raw.decode("utf-16") if (b"\x00" in raw[:200]) else raw.decode("utf-8", errors="ignore")
    # models
    mline = re.search(r"训练模型: (.+)", text)
    out_lines.append(f"=== {name} ===")
    out_lines.append(mline.group(0) if mline else "no model line")
    # fold results: Fold k 结果: a=x, b=y
    fold_re = re.compile(r"Fold \d+ 结果: ([^\n]+)")
    subj_re = re.compile(r"交叉验证: mirror.*?被试 (\w+)")
    cur=None
    folds=defaultdict(list)
    for line in text.splitlines():
        sm=subj_re.search(line)
        if sm: cur=sm.group(1)
        fm=fold_re.search(line)
        if fm and cur:
            folds[cur].append(fm.group(1))
    # parse model=acc pairs
    # collect per-subject means
    subj_means=defaultdict(list)
    for s, fs in folds.items():
        per_model=defaultdict(list)
        for f in fs:
            for part in f.split(", "):
                if "=" in part:
                    k,v=part.split("=")
                    per_model[k.strip()].append(float(v))
        for k,vs in per_model.items():
            subj_means[k].append(sum(vs)/len(vs))
        out_lines.append(f"  {s}: " + ", ".join(f"{k}={sum(vs)/len(vs):.4f}" for k,vs in sorted(per_model.items())))
    out_lines.append("  GROUP:")
    for k,vs in sorted(subj_means.items()):
        if len(vs)>1:
            out_lines.append(f"    {k}: {st.mean(vs):.4f} ± {st.stdev(vs):.4f} (n={len(vs)})")
        else:
            out_lines.append(f"    {k}: {vs[0]:.4f} (n=1)")
    out_lines.append("")

Path(r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/_amfm_ablation_summary.txt").write_text("\n".join(out_lines), encoding="utf-8")
print("\n".join(out_lines))
