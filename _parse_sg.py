from pathlib import Path
import re
from collections import defaultdict

p = Path(r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/scalar_gate_compare_mirror_bs16.log")
raw = p.read_bytes()
# detect utf-16
if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or (b"\x00" in raw[:100]):
    text = raw.decode("utf-16")
else:
    text = raw.decode("utf-8", errors="ignore")

out = Path(r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/_sg_summary.txt")

# subject blocks
subj_pat = re.compile(r"交叉验证: mirror.*?被试 (\w+)")
fold_pat = re.compile(
    r"Fold \d+ 结果: am_fm_content_priority_fm=([0-9.]+), residual_gated=([0-9.]+), residual_scalar_gate=([0-9.]+)"
)
# per-subject mean lines often look like 平均准确率
mean_pat = re.compile(r"(am_fm_content_priority_fm|residual_gated|residual_scalar_gate).*?(准确率|ACC|BACC|平衡).*?([0-9.]+)")

subjects = []
folds = defaultdict(list)
cur = None
for line in text.splitlines():
    m = subj_pat.search(line)
    if m:
        cur = m.group(1)
        subjects.append(cur)
    m = fold_pat.search(line)
    if m and cur:
        folds[cur].append(tuple(map(float, m.groups())))

lines = []
lines.append(f"subjects found: {subjects}")
lines.append(f"n subjects with folds: {len(folds)}")
for s, fs in folds.items():
    if not fs:
        continue
    a = sum(x[0] for x in fs) / len(fs)
    b = sum(x[1] for x in fs) / len(fs)
    c = sum(x[2] for x in fs) / len(fs)
    lines.append(f"{s}: nfold={len(fs)} am_fm={a:.4f} gated={b:.4f} scalar={c:.4f}")

if folds:
    all_a = [sum(x[0] for x in fs)/len(fs) for fs in folds.values() if fs]
    all_b = [sum(x[1] for x in fs)/len(fs) for fs in folds.values() if fs]
    all_c = [sum(x[2] for x in fs)/len(fs) for fs in folds.values() if fs]
    import statistics as st
    lines.append("--- group ---")
    lines.append(f"am_fm_content_priority_fm: {st.mean(all_a):.4f} ± {st.stdev(all_a) if len(all_a)>1 else 0:.4f} (n={len(all_a)})")
    lines.append(f"residual_gated:            {st.mean(all_b):.4f} ± {st.stdev(all_b) if len(all_b)>1 else 0:.4f}")
    lines.append(f"residual_scalar_gate:      {st.mean(all_c):.4f} ± {st.stdev(all_c) if len(all_c)>1 else 0:.4f}")

# also find any printed subject summaries
for i, line in enumerate(text.splitlines()):
    if "群体" in line or "跨被试" in line or "所有被试" in line or "summary" in line.lower():
        lines.append(f"L{i}: {line}")
    if "平均准确率" in line or "平均平衡" in line:
        lines.append(f"L{i}: {line}")

out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8")[:3000])
