from pathlib import Path
import re

base = Path(r"A:/standard_data_interp_ica_99/nme_fusion_results/individual/category_heldout")
ran = sorted(
    p.name.replace("_residual_cv_results.csv", "")
    for p in base.glob("*_residual_cv_results.csv")
)
print("category_heldout ran:", len(ran))
print(ran)

text = Path(
    r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/eeg_audio_residual.py"
).read_text(encoding="utf-8")
subj_keys = re.findall(
    r"'([a-z]+)':\s*\{\s*\n\s*'processed_data_path'",
    text,
)
# unique preserve order
seen = set()
config = []
for s in subj_keys:
    if s not in seen:
        seen.add(s)
        config.append(s)
print("SUBJECTS_CONFIG unique:", len(config))
print(config)

# run script list
t2 = Path(
    r"d:/D/research/audioset下载/download_audioset-master/download_audioset-master/run_residual_gated_baseline_all.py"
).read_text(encoding="utf-8")
m = re.search(r"SUBJECTS\s*=\s*\[(.*?)\]", t2, re.S)
run_list = re.findall(r"'([a-z]+)'", m.group(1)) if m else []
print("baseline_all SUBJECTS:", len(run_list), run_list)

# A drive subject dirs with trials
root = Path(r"A:/standard_data_interp_ica_99")
dirs = sorted(
    p.name
    for p in root.iterdir()
    if p.is_dir()
    and re.match(r"^[a-z]+$", p.name)
)
print("A subject-like dirs:", len(dirs), dirs)

missing_vs_config = [s for s in config if s not in ran]
missing_vs_a = [s for s in dirs if s not in ran]
print("in config but not category_heldout:", missing_vs_config)
print("in A dirs but not category_heldout:", missing_vs_a)

# check mirror summary
mir = Path(r"A:/standard_data_interp_ica_99/nme_fusion_results/individual/mirror/all_subjects_summary.csv")
if mir.exists():
    import pandas as pd
    df = pd.read_csv(mir)
    print("mirror n:", len(df), list(df.subject))
    print("in mirror not category:", sorted(set(df.subject) - set(ran)))
    print("in category not mirror:", sorted(set(ran) - set(df.subject)))

# check which missing have EEG data
for s in missing_vs_config:
    p = root / s
    trials = list(p.glob("*trials*.npy")) if p.exists() else []
    print(f" missing {s}: exists={p.exists()} trials={len(trials)} path_cfg_check")
