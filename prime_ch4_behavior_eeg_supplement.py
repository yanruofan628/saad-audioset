#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter-4 / PRIME 行为—EEG 补充分析（按优先级一气跑完）

1) Pair-level stimulus priority + mirrored AB/BA consistency
2) Trial-level acoustic prior π_stim（LOSO / CV，无当前试次标签泄漏）
3) RT ~ competition strength（chronometric；post-stimulus latency）
4) 偏离 stimulus prior 的 EEG / RT 对比
5) Acoustic–EEG congruence / conflict
6) Individual differences（β_A, β_E, ΔAUC 等）
7) Ratings beyond acoustics（可选，不进 PRIME 输入）

输出目录: prime_ch4_supplement/
"""
from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
EEG_ROOT = Path(r"A:/standard_data_interp_ica_99")
MAPPING_BASE = Path(r"D:\D\research\audioset下载\audio_pairs_2s\分组音频")
MAPPING_FILES = [
    MAPPING_BASE / "folder1" / "file_mapping_folder1.csv",
    MAPPING_BASE / "folder2" / "file_mapping_folder2.csv",
    MAPPING_BASE / "folder3" / "file_mapping_folder3.csv",
]
AUDIO_CACHE = ROOT / "audio_features_cache" / "all_audio_features_1b2a4a1a.npy"
AUDIO_META = ROOT / "audio_features_cache" / "all_audio_features_1b2a4a1a_meta.json"
RATING_CSV = ROOT / "rating_dim_human_vs_clap" / "trials_with_rating_deltas.csv"
OUT = ROOT / "prime_ch4_supplement"
OUT.mkdir(parents=True, exist_ok=True)

SR = 250.0
STIM_ONSET_S = 3.0
# Mid-latency CT window relative to sound onset
WIN_LO_S, WIN_HI_S = 0.30, 0.50
# GSN-HydroCel-128 ≈ 10-20 (EGI convention used in prior ERP plots)
CH_C3, CH_T7, CH_C4, CH_T8 = 35, 44, 103, 107  # 0-based for EEG036/045/104/108
ACOUSTIC_NAMES = [f"a{i:02d}" for i in range(10)]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


# -----------------------------------------------------------------------------
# Helpers: pair identity / parsing
# -----------------------------------------------------------------------------
def parse_pair_name(filename: str):
    if filename.lower().endswith(".wav"):
        filename = filename[:-4]
    if "+" not in filename:
        return None
    left_part, right_part = filename.split("+", 1)
    prefixes = ["nn_main_", "nn_sub_", "main_", "sub_"]
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break

    def extract_cat_id(part):
        last_us = part.rfind("_")
        if last_us < 0:
            return None, None
        second_last = part.rfind("_", 0, last_us)
        if second_last < 0:
            return None, None
        return part[:second_last], part[second_last + 1 :]

    lc, li = extract_cat_id(left_part)
    rc, ri = extract_cat_id(right_part)
    if lc is None or rc is None:
        return None
    return lc, li, rc, ri


def unordered_pair_key(pair_name: str) -> str | None:
    parsed = parse_pair_name(pair_name)
    if parsed is None:
        return None
    lc, li, rc, ri = parsed
    a = f"{lc}|{li}"
    b = f"{rc}|{ri}"
    return "||".join(sorted([a, b]))


def sound_ids(pair_name: str):
    parsed = parse_pair_name(pair_name)
    if parsed is None:
        return None, None
    lc, li, rc, ri = parsed
    return f"{lc}|{li}", f"{rc}|{ri}"


MAIN_CAT = {
    "Telephone bell ringing": "High Ecology",
    "Ambulance (siren)": "High Ecology",
    "Baby cry, infant cry": "High Ecology",
    "Computer keyboard": "Low Ecology",
    "Helicopter": "Low Ecology",
    "Chicken, rooster": "Low Ecology",
    "Male speech, man speaking": "speech",
    "Female speech, woman speaking": "speech",
    "Child speech, kid speaking": "speech",
    "Bass drum": "music",
    "Funny music": "music",
    "Sad music": "music",
}


def parse_benchmark_txt(path: Path):
    """Return list of dicts: wavfile, response(1/2), reaction_time_ms."""
    encodings = ["utf-16-le", "utf-16", "utf-8", "utf-8-sig", "gbk", "latin-1"]
    lines = None
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            # require actual trial frames (utf-16 vs utf-16-le / wrong encoding can yield garbage)
            if "LogFrame Start" not in text and "*** LogFrame Start ***" not in text:
                continue
            lines = text.splitlines()
            break
        except Exception:
            continue
    if lines is None:
        return []
    trials = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Level: 3") and i + 1 < len(lines) and "LogFrame Start" in lines[i + 1]:
            td = {}
            for j in range(i + 1, min(i + 30, len(lines))):
                line = lines[j].strip()
                if line.startswith("wavfile:"):
                    td["wavfile"] = line.split("wavfile:", 1)[1].strip()
                elif "ImageDisplay" in line and ".RESP:" in line:
                    td["response"] = int(line.split("RESP:")[1].strip())
                elif "ImageDisplay" in line and ".RT:" in line:
                    try:
                        td["reaction_time"] = float(line.split("RT:")[1].strip())
                    except Exception:
                        pass
                elif "LogFrame End" in line:
                    break
            if "wavfile" in td and "response" in td:
                trials.append(td)
            i = j
        i += 1
    return trials


def load_mappings():
    maps = []
    for mf in MAPPING_FILES:
        df = pd.read_csv(mf, encoding="utf-8")
        m = {}
        for _, row in df.iterrows():
            m[str(row["new_name"])] = str(row["original_name"])
            # also basename without folder
            m[Path(str(row["new_name"])).name] = str(row["original_name"])
        maps.append(m)
    return maps


def _pick_benchmark(folder: Path) -> Path | None:
    cands = sorted(folder.glob("benchmark_*.txt"))
    return cands[0] if cands else None


def find_benchmarks_for_subject(subject: str, info: dict) -> list[Path]:
    """Locate 3 session benchmark txt files from trial_info mff folders."""
    mff_files = info.get("mff_files") or []
    benches: list[Path] = []

    for mf in mff_files:
        folder = Path("A:/") / Path(mf).name
        bp = _pick_benchmark(folder) if folder.exists() else None
        if bp is None:
            for lg in info.get("log_files") or []:
                p = Path(str(lg).replace("\\\\", "\\"))
                if Path(mf).name in str(p) and p.parent.exists():
                    bp = _pick_benchmark(p.parent)
                    if bp is not None:
                        break
        if bp is None:
            raise FileNotFoundError(f"{subject}: no benchmark in {folder}")
        benches.append(bp)

    if len(benches) != 3:
        patterns = [f"{subject}_*.mff"]
        if subject == "zhanghanglei":
            patterns.append("hanglei_*.mff")
        found = []
        for pat in patterns:
            for d in sorted(Path("A:/").glob(pat)):
                bp = _pick_benchmark(d)
                if bp is not None:
                    found.append(bp)
            if len(found) >= 3:
                break
        if len(found) >= 3:
            benches = found[:3]

    if len(benches) != 3:
        raise FileNotFoundError(f"{subject}: expected 3 benchmarks, got {len(benches)}")
    return benches


def build_trial_table(subjects: list[str] | None = None) -> pd.DataFrame:
    maps = load_mappings()
    audio_meta = json.loads(AUDIO_META.read_text(encoding="utf-8"))
    pair_names = audio_meta["pair_names"]
    pair_to_idx = {p: i for i, p in enumerate(pair_names)}
    X_audio = np.load(AUDIO_CACHE)
    assert X_audio.shape[1] == 20
    delta_A_all = X_audio[:, :10] - X_audio[:, 10:]  # L - R, positive => left louder/higher

    if subjects is None:
        subjects = sorted(
            p.name
            for p in EEG_ROOT.iterdir()
            if p.is_dir() and (p / f"{p.name}_trials.npy").exists()
        )

    rows = []
    for subject in subjects:
        sdir = EEG_ROOT / subject
        info = json.loads((sdir / f"{subject}_trial_info.json").read_text(encoding="utf-8"))
        labels = pd.read_csv(sdir / f"{subject}_labels.csv")
        rts = pd.read_csv(sdir / f"{subject}_rts.csv")
        # Label: 0=left(RESP1), 1=right(RESP2)
        y = labels["Label"].to_numpy().astype(int)
        rt = rts["RT_ms"].to_numpy().astype(float)

        try:
            benches = find_benchmarks_for_subject(subject, info)
        except Exception as e:
            print(f"[skip] {subject}: {e}")
            continue

        pair_list = []
        session_list = []
        for sess_i, bp in enumerate(benches):
            parsed = parse_benchmark_txt(bp)
            mapping = maps[sess_i]
            # hanglei / zhanghanglei missing last trial in sess 0/1 historically
            if subject in ("hanglei", "zhanghanglei") and sess_i in (0, 1) and len(parsed) > 0:
                # EEG has 480; if benchmark longer, drop last
                pass
            for tr in parsed:
                fn = Path(tr["wavfile"]).name
                if fn not in mapping and fn.replace("audio/", "") in mapping:
                    fn = fn.replace("audio/", "")
                # mapping keys are often a001.wav
                key = fn
                if key not in mapping:
                    key = Path(fn).name
                if key not in mapping:
                    continue
                pair_list.append(mapping[key])
                session_list.append(sess_i + 1)

        n = min(len(y), len(pair_list), len(rt))
        if len(pair_list) != len(y):
            print(f"  {subject}: pairs={len(pair_list)} labels={len(y)} -> use {n}")

        for i in range(n):
            pn = pair_list[i]
            if pn not in pair_to_idx:
                continue
            aidx = pair_to_idx[pn]
            dA = delta_A_all[aidx]
            left_id, right_id = sound_ids(pn)
            ukey = unordered_pair_key(pn)
            lc = parse_pair_name(pn)
            left_cat = lc[0] if lc else None
            right_cat = lc[2] if lc else None
            left_main = MAIN_CAT.get(left_cat)
            right_main = MAIN_CAT.get(right_cat)
            rows.append({
                "subject": subject,
                "trial_idx": i,
                "session": session_list[i] if i < len(session_list) else None,
                "trial_in_session": i % 160,
                "pair_name": pn,
                "pair_key": ukey,
                "left_id": left_id,
                "right_id": right_id,
                "left_cat": left_cat,
                "right_cat": right_cat,
                "left_main": left_main,
                "right_main": right_main,
                "y_right": int(y[i]),  # 1 = choose right
                "rt_ms": float(rt[i]),
                **{f"dA_{n}": float(dA[k]) for k, n in enumerate(ACOUSTIC_NAMES)},
                "audio_idx": aidx,
            })
        print(f"  loaded {subject}: {n} trials")

    df = pd.DataFrame(rows)
    # category deltas (right - left one-hot main cats) → compact scalars
    for cat in ["High Ecology", "Low Ecology", "speech", "music"]:
        df[f"dC_{cat}"] = (
            (df["right_main"] == cat).astype(float) - (df["left_main"] == cat).astype(float)
        )
    df["log_rt"] = np.log(np.clip(df["rt_ms"], 50, None))
    return df


# -----------------------------------------------------------------------------
# 1. Pair-level priority + mirrored consistency
# -----------------------------------------------------------------------------
def analysis_pair_priority(df: pd.DataFrame):
    print("\n=== 1. Pair-level stimulus priority & mirrored consistency ===")
    # For each unordered pair: P(choose sound A) independent of ear
    # Define canonical sound order from pair_key
    recs = []
    for pk, g in df.groupby("pair_key"):
        if pk is None or len(g) < 4:
            continue
        # canonical: first id in sorted key
        a_id, b_id = pk.split("||")
        # choose_A: if left==A and y_right==0, or right==A and y_right==1
        choose_A = np.where(
            g["left_id"].values == a_id,
            1 - g["y_right"].values,
            g["y_right"].values,
        )
        pi = float(np.mean(choose_A))
        C = abs(2 * pi - 1)
        # mirrored AB vs BA
        ab = g[g["left_id"] == a_id]
        ba = g[g["left_id"] == b_id]
        p_A_given_AB = float((1 - ab["y_right"]).mean()) if len(ab) else np.nan
        p_A_given_BA = float(ba["y_right"].mean()) if len(ba) else np.nan
        recs.append({
            "pair_key": pk,
            "n": len(g),
            "n_ab": len(ab),
            "n_ba": len(ba),
            "pi_choose_A": pi,
            "C_strength": C,
            "p_A_AB": p_A_given_AB,
            "p_A_BA": p_A_given_BA,
            "mirror_abs_diff": abs(p_A_given_AB - p_A_given_BA) if np.isfinite(p_A_given_AB) and np.isfinite(p_A_given_BA) else np.nan,
        })
    pair_df = pd.DataFrame(recs)
    pair_df.to_csv(OUT / "01_pair_level_priority.csv", index=False)

    # subject right bias
    subj = df.groupby("subject").agg(
        right_rate=("y_right", "mean"),
        n=("y_right", "size"),
        median_rt=("rt_ms", "median"),
    ).reset_index()
    subj.to_csv(OUT / "01_subject_right_bias.csv", index=False)

    # split-half reliability of pair π
    rng = np.random.default_rng(42)
    half_corrs = []
    for _ in range(50):
        pis1, pis2 = [], []
        for pk, g in df.groupby("pair_key"):
            if pk is None or len(g) < 8:
                continue
            a_id = pk.split("||")[0]
            choose_A = np.where(g["left_id"].values == a_id, 1 - g["y_right"].values, g["y_right"].values)
            idx = np.arange(len(choose_A))
            rng.shuffle(idx)
            h = len(idx) // 2
            pis1.append(choose_A[idx[:h]].mean())
            pis2.append(choose_A[idx[h:]].mean())
        if len(pis1) > 10:
            half_corrs.append(np.corrcoef(pis1, pis2)[0, 1])
    split_half = float(np.nanmean(half_corrs)) if half_corrs else np.nan

    # mirror consistency: correlate p_A_AB vs p_A_BA
    m = pair_df.dropna(subset=["p_A_AB", "p_A_BA"])
    mirror_r = stats.pearsonr(m["p_A_AB"], m["p_A_BA"]) if len(m) > 5 else (np.nan, np.nan)

    summary = {
        "n_pairs": int(len(pair_df)),
        "mean_C": float(pair_df["C_strength"].mean()),
        "frac_strong_C_gt_0.3": float((pair_df["C_strength"] > 0.3).mean()),
        "frac_ambiguous_C_lt_0.1": float((pair_df["C_strength"] < 0.1).mean()),
        "mean_mirror_abs_diff": float(pair_df["mirror_abs_diff"].mean()),
        "mirror_pA_AB_vs_BA_r": float(mirror_r[0]),
        "mirror_pA_AB_vs_BA_p": float(mirror_r[1]),
        "split_half_r_mean": split_half,
        "group_right_rate_mean": float(subj["right_rate"].mean()),
        "group_right_rate_std": float(subj["right_rate"].std()),
    }
    (OUT / "01_pair_priority_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].hist(pair_df["C_strength"], bins=20, color="#3d5a80", edgecolor="white")
    axes[0].set_xlabel(r"Preference strength $C_j=|2\pi_j-1|$")
    axes[0].set_ylabel("N pairs")
    axes[0].set_title("Pair-level priority strength")

    axes[1].scatter(m["p_A_AB"], m["p_A_BA"], s=18, alpha=0.55, c="#ee6c4d")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel(r"$P$(choose A | A$_L$ B$_R$)")
    axes[1].set_ylabel(r"$P$(choose A | B$_L$ A$_R$)")
    axes[1].set_title(f"Mirrored consistency  r={mirror_r[0]:.2f}")

    axes[2].bar(np.arange(len(subj)), subj["right_rate"].values, color="#98c1d9")
    axes[2].axhline(0.5, color="k", ls="--", lw=1)
    axes[2].set_ylabel("P(choose right)")
    axes[2].set_xlabel("Subject")
    axes[2].set_title("Right-ear response bias")
    fig.tight_layout()
    fig.savefig(OUT / "01_pair_priority_mirror.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return pair_df, summary


# -----------------------------------------------------------------------------
# 2. Trial-level mixed / LOSO acoustic prior π_stim
# -----------------------------------------------------------------------------
def _design_matrices(df: pd.DataFrame):
    a_cols = [f"dA_{n}" for n in ACOUSTIC_NAMES]
    c_cols = [c for c in df.columns if c.startswith("dC_")]
    Xa = df[a_cols].to_numpy(dtype=float)
    Xc = df[c_cols].to_numpy(dtype=float)
    X = np.concatenate([Xa, Xc], axis=1)
    y = df["y_right"].to_numpy(dtype=int)
    return X, y, a_cols + c_cols


def fit_pi_stim_loso(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-subject-out logistic: π_stim = P(y=1 | audio/category only)."""
    print("\n=== 2. LOSO acoustic prior π_stim ===")
    X, y, feat_names = _design_matrices(df)
    subjects = df["subject"].to_numpy()
    pi = np.full(len(df), np.nan)
    coefs = []

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups=subjects):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        pi[test_idx] = clf.predict_proba(Xte)[:, 1]
        coefs.append(clf.coef_.ravel())

    df = df.copy()
    df["pi_stim"] = pi
    df["priority_strength"] = np.abs(pi - 0.5)
    df["logit_pi"] = np.log(np.clip(pi, 1e-4, 1 - 1e-4) / np.clip(1 - pi, 1e-4, 1 - 1e-4))

    # pair-level empirical π (leave out current subject), vectorized per pair_name
    pi_pair = np.full(len(df), np.nan)
    # precompute subject-excluded means for each (pair_name)
    for pn, g in df.groupby("pair_name"):
        for subject in g["subject"].unique():
            others = g[g["subject"] != subject]["y_right"]
            if len(others) >= 2:
                val = float(others.mean())
                idx = g.index[g["subject"] == subject]
                pi_pair[idx] = val
    df["pi_pair_loso"] = pi_pair

    mean_coef = np.nanmean(np.stack(coefs, 0), 0)
    coef_tbl = pd.DataFrame({"feature": feat_names, "mean_coef_loso": mean_coef})
    coef_tbl.to_csv(OUT / "02_loso_acoustic_coefs.csv", index=False)

    # report predictive AUC of π_stim
    auc = roc_auc_score(y, pi)
    bacc = balanced_accuracy_score(y, (pi >= 0.5).astype(int))
    summary = {"pi_stim_auc": float(auc), "pi_stim_bacc": float(bacc), "n": int(len(df))}
    (OUT / "02_pi_stim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)

    # Cluster-robust logistic (subject clusters) as practical mixed-model proxy;
    # plus subject-FE logistic for acoustic features.
    try:
        import statsmodels.api as sm

        top = (
            coef_tbl.reindex(coef_tbl["mean_coef_loso"].abs().sort_values(ascending=False).index)["feature"]
            .head(8)
            .tolist()
        )
        use_cols = [c for c in top if c in df.columns]
        sub = df.dropna(subset=use_cols + ["y_right", "subject"]).copy()
        X = sm.add_constant(sub[use_cols].astype(float))
        y = sub["y_right"].astype(float)
        # subject fixed effects (drop first)
        subj_dummies = pd.get_dummies(sub["subject"], drop_first=True, dtype=float)
        X_fe = pd.concat([X.reset_index(drop=True), subj_dummies.reset_index(drop=True)], axis=1)
        model = sm.Logit(y.to_numpy(), X_fe.to_numpy())
        # cluster by subject
        groups = pd.Categorical(sub["subject"]).codes
        fit = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": groups})
        with open(OUT / "02_mixed_logit_subjectFE_clusterSE.txt", "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
            f.write("\n\nfeatures (first columns): const + " + ", ".join(use_cols))
            f.write("\n+ subject fixed effects; cluster-robust SE by subject")
        # save fixed-effect acoustic/category coefs only
        n_base = 1 + len(use_cols)
        coef_out = pd.DataFrame({
            "feature": ["const"] + use_cols,
            "coef": fit.params[:n_base],
            "se_cluster": fit.bse[:n_base],
            "z": fit.tvalues[:n_base],
            "p": fit.pvalues[:n_base],
        })
        coef_out.to_csv(OUT / "02_mixed_logit_coefs.csv", index=False)
        print("  wrote subject-FE cluster-robust logistic")
    except Exception as e:
        print(f"  mixed model skipped: {e}")

    return df


# -----------------------------------------------------------------------------
# 3. RT ~ competition
# -----------------------------------------------------------------------------
def analysis_rt_competition(df: pd.DataFrame):
    print("\n=== 3. RT vs competition strength ===")
    d = df.dropna(subset=["pi_stim", "log_rt", "rt_ms"]).copy()
    # bins for chronometric
    d["prio_bin"] = pd.qcut(d["priority_strength"], 8, duplicates="drop")
    chron = d.groupby("prio_bin", observed=True).agg(
        strength=("priority_strength", "mean"),
        median_rt=("rt_ms", "median"),
        mean_log_rt=("log_rt", "mean"),
        n=("rt_ms", "size"),
    ).reset_index()

    # subject-level mixed-ish: OLS with subject FE
    from sklearn.linear_model import LinearRegression

    # demean within subject
    d["log_rt_dm"] = d["log_rt"] - d.groupby("subject")["log_rt"].transform("mean")
    d["prio_dm"] = d["priority_strength"] - d.groupby("subject")["priority_strength"].transform("mean")
    d["session_c"] = d["session"] - d["session"].mean()
    d["trial_c"] = d["trial_idx"] - d.groupby("subject")["trial_idx"].transform("mean")

    X = np.column_stack([d["prio_dm"], d["session_c"], d["trial_c"]])
    y = d["log_rt_dm"].to_numpy()
    lr = LinearRegression().fit(X, y)
    # Spearman at trial level (pooled) + within-subject mean Spearman
    sp = stats.spearmanr(d["priority_strength"], d["log_rt"])
    within_rs = []
    for _, g in d.groupby("subject"):
        if len(g) < 30:
            continue
        r = stats.spearmanr(g["priority_strength"], g["log_rt"])
        if np.isfinite(r.correlation):
            within_rs.append(r.correlation)
    # also vs pair-level empirical ambiguity 1-C using pi_pair
    if "pi_pair_loso" in d.columns:
        d["pair_strength"] = np.abs(d["pi_pair_loso"] - 0.5)
        sp_pair = stats.spearmanr(d["pair_strength"].dropna(), d.loc[d["pair_strength"].notna(), "log_rt"])
    else:
        sp_pair = (np.nan, np.nan)

    summary = {
        "gamma_priority_strength_within_subj": float(lr.coef_[0]),
        "gamma_session": float(lr.coef_[1]),
        "gamma_trial_index": float(lr.coef_[2]),
        "spearman_strength_logRT_r": float(sp.correlation),
        "spearman_strength_logRT_p": float(sp.pvalue),
        "within_subj_spearman_mean": float(np.mean(within_rs)) if within_rs else None,
        "within_subj_spearman_ttest_vs0_p": float(stats.ttest_1samp(within_rs, 0).pvalue) if len(within_rs) > 3 else None,
        "spearman_pairStrength_logRT_r": float(sp_pair.correlation) if hasattr(sp_pair, "correlation") else float(sp_pair[0]),
        "spearman_pairStrength_logRT_p": float(sp_pair.pvalue) if hasattr(sp_pair, "pvalue") else float(sp_pair[1]),
        "note": "RT is post-stimulus response latency after 1s delay, not online decision time",
    }
    (OUT / "03_rt_competition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    chron.to_csv(OUT / "03_chronometric_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.errorbar(chron["strength"], chron["median_rt"], fmt="o-", color="#3d5a80", lw=2, ms=7)
    ax.set_xlabel(r"Stimulus priority strength $|\pi^{stim}-0.5|$")
    ax.set_ylabel("Median RT (ms)")
    ax.set_title("Chronometric curve\n(post-stimulus response latency)")
    fig.tight_layout()
    fig.savefig(OUT / "03_chronometric_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(summary)
    return summary


# -----------------------------------------------------------------------------
# EEG lateralization ΔE
# -----------------------------------------------------------------------------
def compute_delta_E_for_subject(subject: str) -> np.ndarray:
    """Return (n_trials,) ΔE = mean(C3,T7) - mean(C4,T8) in 300–500 ms post-onset."""
    path = EEG_ROOT / subject / f"{subject}_trials.npy"
    # shape (n, 128, 1500) full 6 s
    data = np.load(path, mmap_mode="r")
    n = data.shape[0]
    t0 = int(round((STIM_ONSET_S + WIN_LO_S) * SR))
    t1 = int(round((STIM_ONSET_S + WIN_HI_S) * SR))
    # baseline 0–3 s mean per channel (light correction)
    # For speed: only needed channels
    chs = [CH_C3, CH_T7, CH_C4, CH_T8]
    # load slice into memory
    seg = np.asarray(data[:, chs, :], dtype=np.float32)  # (n,4,T)
    base = seg[:, :, : int(round(3.0 * SR))].mean(axis=2, keepdims=True)
    seg = seg - base
    win = seg[:, :, t0:t1].mean(axis=2)  # (n,4)
    left_ct = win[:, 0:2].mean(axis=1)
    right_ct = win[:, 2:4].mean(axis=1)
    # trials.npy stored in Volts → report µV
    return (left_ct - right_ct) * 1e6  # positive: left CT more positive


def attach_delta_E(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Computing ΔE (CT lateralization 300–500 ms) ===")
    dE = np.full(len(df), np.nan)
    # positional index aligned with df.reset_index
    df = df.reset_index(drop=True)
    for subject, g in df.groupby("subject"):
        try:
            arr = compute_delta_E_for_subject(subject)
        except Exception as e:
            print(f"  {subject} EEG failed: {e}")
            continue
        for pos, trial_idx in zip(g.index.to_numpy(), g["trial_idx"].to_numpy()):
            if int(trial_idx) < len(arr):
                dE[int(pos)] = arr[int(trial_idx)]
        print(f"  {subject}: ΔE ok")
    df["delta_E"] = dE
    return df


# -----------------------------------------------------------------------------
# 4. Deviation from prior
# -----------------------------------------------------------------------------
def analysis_deviation(df: pd.DataFrame):
    print("\n=== 4. Deviation from stimulus prior ===")
    d = df.dropna(subset=["pi_stim", "y_right"]).copy()
    # expected = choose side favored by π_stim
    d["expected_right"] = (d["pi_stim"] >= 0.5).astype(int)
    d["is_deviation"] = (d["y_right"] != d["expected_right"]).astype(int)
    # residual
    d["r_resid"] = d["y_right"] - d["pi_stim"]

    # Choice-aligned ΔE: positive => EEG favors the chosen side
    # ΔE>0 favors left; y_right=0 is left → aligned = ΔE * (1 - 2*y_right)
    d["delta_E_aligned"] = d["delta_E"] * (1 - 2 * d["y_right"])
    d["abs_delta_E"] = d["delta_E"].abs()

    rows = []
    for name, col in [
        ("delta_E_uV", "delta_E"),
        ("delta_E_choice_aligned", "delta_E_aligned"),
        ("abs_delta_E", "abs_delta_E"),
        ("rt_ms", "rt_ms"),
        ("log_rt", "log_rt"),
    ]:
        if col not in d.columns:
            continue
        a = d.loc[d["is_deviation"] == 0, col].dropna()
        b = d.loc[d["is_deviation"] == 1, col].dropna()
        if len(a) > 10 and len(b) > 10:
            t = stats.ttest_ind(a, b, equal_var=False)
            rows.append({
                "measure": name,
                "mean_expected": float(a.mean()),
                "mean_deviation": float(b.mean()),
                "t": float(t.statistic),
                "p": float(t.pvalue),
                "n_expected": int(len(a)),
                "n_deviation": int(len(b)),
            })
    pd.DataFrame(rows).to_csv(OUT / "04_deviation_vs_expected.csv", index=False)

    # Does ΔE predict residual choice beyond π_stim?
    dd = d.dropna(subset=["delta_E", "pi_stim"]).copy()
    # logistic: y ~ logit_pi + delta_E
    X = np.column_stack([dd["logit_pi"], dd["delta_E"]])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000).fit(Xs, dd["y_right"])
    # LOSO ΔAUC: audio-only vs audio+EEG
    subjects = dd["subject"].to_numpy()
    y = dd["y_right"].to_numpy()
    pi = dd["pi_stim"].to_numpy()
    dE = dd["delta_E"].to_numpy()
    auc_audio, auc_both = [], []
    logo = LeaveOneGroupOut()
    X_both = np.column_stack([pi, dE])
    for tr, te in logo.split(X_both, y, groups=subjects):
        auc_audio.append(roc_auc_score(y[te], pi[te]))
        clf2 = LogisticRegression(max_iter=2000).fit(
            StandardScaler().fit_transform(X_both[tr]), y[tr]
        )
        Xte = StandardScaler().fit(X_both[tr]).transform(X_both[te])
        auc_both.append(roc_auc_score(y[te], clf2.predict_proba(Xte)[:, 1]))
    summary = {
        "coef_logit_pi": float(clf.coef_[0, 0]),
        "coef_delta_E": float(clf.coef_[0, 1]),
        "mean_auc_audio": float(np.mean(auc_audio)),
        "mean_auc_audio_plus_E": float(np.mean(auc_both)),
        "delta_auc": float(np.mean(auc_both) - np.mean(auc_audio)),
        "frac_deviation_trials": float(d["is_deviation"].mean()),
    }
    (OUT / "04_deviation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plot ΔE by expected vs deviation
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    if "delta_E" in d.columns:
        data_plot = [d.loc[d["is_deviation"] == 0, "delta_E"].dropna(),
                     d.loc[d["is_deviation"] == 1, "delta_E"].dropna()]
        axes[0].boxplot(data_plot, labels=["Follow prior", "Deviate"])
        axes[0].set_ylabel(r"$\Delta E$ (C3/T7 − C4/T8), 300–500 ms")
        axes[0].set_title("CT laterality by prior adherence")
    axes[1].boxplot(
        [d.loc[d["is_deviation"] == 0, "rt_ms"].dropna(),
         d.loc[d["is_deviation"] == 1, "rt_ms"].dropna()],
        labels=["Follow prior", "Deviate"],
    )
    axes[1].set_ylabel("RT (ms)")
    axes[1].set_title("Response latency by prior adherence")
    fig.tight_layout()
    fig.savefig(OUT / "04_deviation_eeg_rt.png", bbox_inches="tight")
    plt.close(fig)
    print(summary)
    return d, summary


# -----------------------------------------------------------------------------
# 5. Acoustic–EEG congruence
# -----------------------------------------------------------------------------
def analysis_congruence(df: pd.DataFrame):
    print("\n=== 5. Acoustic–EEG congruence / conflict ===")
    d = df.dropna(subset=["pi_stim", "delta_E", "y_right"]).copy()
    # acoustic favor right if pi>0.5; EEG favor right if delta_E < 0?
    # ΔE = left_CT - right_CT; prior ERP: left choice → higher left CT (C3/T7).
    # So positive ΔE favors left (y_right=0); negative ΔE favors right.
    d["audio_favor_right"] = (d["pi_stim"] >= 0.5).astype(int)
    d["eeg_favor_right"] = (d["delta_E"] < 0).astype(int)
    weak_audio = d["priority_strength"] < d["priority_strength"].median()
    weak_eeg = d["delta_E"].abs() < d["delta_E"].abs().median()

    def quad(row):
        if weak_audio.loc[row.name] and weak_eeg.loc[row.name]:
            return "weak_both"
        if weak_audio.loc[row.name]:
            return "weak_audio"
        if weak_eeg.loc[row.name]:
            return "weak_eeg"
        if row["audio_favor_right"] == row["eeg_favor_right"]:
            return "congruent"
        return "conflict"

    d["quad"] = d.apply(quad, axis=1)
    # simpler 2x2 ignoring weak for main table
    d["cong"] = np.where(
        d["audio_favor_right"] == d["eeg_favor_right"], "congruent", "conflict"
    )

    tab = d.groupby("cong").agg(
        p_right=("y_right", "mean"),
        median_rt=("rt_ms", "median"),
        n=("y_right", "size"),
        mean_abs_resid=("r_resid", lambda x: np.nanmean(np.abs(x)) if "r_resid" in d else np.nan),
    ).reset_index()
    # fix resid
    d["abs_resid"] = (d["y_right"] - d["pi_stim"]).abs()
    tab = d.groupby("cong").agg(
        p_right=("y_right", "mean"),
        median_rt=("rt_ms", "median"),
        mean_abs_resid=("abs_resid", "mean"),
        n=("y_right", "size"),
    ).reset_index()
    tab.to_csv(OUT / "05_congruence_table.csv", index=False)

    # 2x2 heatmap of P(choose right)
    mat = np.zeros((2, 2))
    counts = np.zeros((2, 2))
    for ar in [0, 1]:
        for er in [0, 1]:
            m = (d["audio_favor_right"] == ar) & (d["eeg_favor_right"] == er)
            mat[1 - ar, er] = d.loc[m, "y_right"].mean() if m.sum() else np.nan
            counts[1 - ar, er] = m.sum()
    # rows: audio favors left / right; cols: EEG favors left / right
    # reorder: row0 audio left, row1 audio right
    mat2 = np.array([
        [
            d.loc[(d.audio_favor_right == 0) & (d.eeg_favor_right == 0), "y_right"].mean(),
            d.loc[(d.audio_favor_right == 0) & (d.eeg_favor_right == 1), "y_right"].mean(),
        ],
        [
            d.loc[(d.audio_favor_right == 1) & (d.eeg_favor_right == 0), "y_right"].mean(),
            d.loc[(d.audio_favor_right == 1) & (d.eeg_favor_right == 1), "y_right"].mean(),
        ],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat2, cmap="RdBu_r", vmin=0.3, vmax=0.7)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["EEG→Left", "EEG→Right"])
    ax.set_yticklabels(["Audio→Left", "Audio→Right"])
    for i in range(2):
        for j in range(2):
            n = ((d.audio_favor_right == i) & (d.eeg_favor_right == j)).sum()
            ax.text(j, i, f"{mat2[i, j]:.2f}\nn={n}", ha="center", va="center", color="k")
    ax.set_title("P(choose right) by acoustic–EEG quadrant")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(right)")
    fig.tight_layout()
    fig.savefig(OUT / "05_congruence_quadrants.png", bbox_inches="tight")
    plt.close(fig)

    # RT congruent vs conflict
    t = stats.ttest_ind(
        d.loc[d["cong"] == "congruent", "rt_ms"].dropna(),
        d.loc[d["cong"] == "conflict", "rt_ms"].dropna(),
        equal_var=False,
    )
    summary = {
        "median_rt_congruent": float(d.loc[d.cong == "congruent", "rt_ms"].median()),
        "median_rt_conflict": float(d.loc[d.cong == "conflict", "rt_ms"].median()),
        "rt_ttest_p": float(t.pvalue),
        "p_right_congruent": float(d.loc[d.cong == "congruent", "y_right"].mean()),
        "p_right_conflict": float(d.loc[d.cong == "conflict", "y_right"].mean()),
    }
    (OUT / "05_congruence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)
    return d, summary


# -----------------------------------------------------------------------------
# 6. Individual differences
# -----------------------------------------------------------------------------
def analysis_individual(df: pd.DataFrame):
    print("\n=== 6. Individual differences ===")
    rows = []
    a_cols = [f"dA_{n}" for n in ACOUSTIC_NAMES]
    for subject, g in df.groupby("subject"):
        g = g.dropna(subset=["y_right"] + a_cols)
        if len(g) < 80:
            continue
        y = g["y_right"].to_numpy()
        Xa = StandardScaler().fit_transform(g[a_cols].to_numpy())
        clf_a = LogisticRegression(max_iter=2000).fit(Xa, y)
        # scalar acoustic sensitivity: std of predicted logit from audio
        beta_A_norm = float(np.linalg.norm(clf_a.coef_))
        auc_a = roc_auc_score(y, clf_a.predict_proba(Xa)[:, 1])

        beta_E = np.nan
        auc_e = np.nan
        delta_auc = np.nan
        if g["delta_E"].notna().sum() > 80:
            dE = g["delta_E"].to_numpy()
            mask = np.isfinite(dE)
            clf_e = LogisticRegression(max_iter=2000).fit(
                StandardScaler().fit_transform(dE[mask].reshape(-1, 1)), y[mask]
            )
            beta_E = float(clf_e.coef_[0, 0])
            auc_e = roc_auc_score(y[mask], clf_e.predict_proba(
                StandardScaler().fit_transform(dE[mask].reshape(-1, 1)))[:, 1])
            Xb = np.column_stack([Xa[mask], StandardScaler().fit_transform(dE[mask].reshape(-1, 1))])
            clf_b = LogisticRegression(max_iter=2000).fit(Xb, y[mask])
            auc_b = roc_auc_score(y[mask], clf_b.predict_proba(Xb)[:, 1])
            delta_auc = auc_b - auc_a

        # pair consistency: mean C within subject using mirrored pairs
        cons = []
        for pk, gg in g.groupby("pair_key"):
            if pk is None or len(gg) < 2:
                continue
            a_id = pk.split("||")[0]
            choose_A = np.where(gg["left_id"].values == a_id, 1 - gg["y_right"].values, gg["y_right"].values)
            cons.append(abs(2 * choose_A.mean() - 1))
        rows.append({
            "subject": subject,
            "right_bias": float(y.mean()),
            "beta_A_norm": beta_A_norm,
            "auc_audio": float(auc_a),
            "beta_E": beta_E,
            "auc_eeg": auc_e,
            "delta_auc": delta_auc,
            "mean_pair_C": float(np.mean(cons)) if cons else np.nan,
            "median_rt": float(g["rt_ms"].median()),
        })
    ind = pd.DataFrame(rows)
    ind.to_csv(OUT / "06_individual_differences.csv", index=False)

    summary = {}
    if ind["delta_auc"].notna().sum() > 5:
        r1 = stats.spearmanr(ind["beta_A_norm"], ind["delta_auc"], nan_policy="omit")
        r2 = stats.spearmanr(ind["auc_eeg"], ind["delta_auc"], nan_policy="omit")
        r3 = stats.spearmanr(ind["beta_A_norm"], ind["auc_audio"], nan_policy="omit")
        summary = {
            "spearman_betaA_vs_deltaAUC_r": float(r1.correlation) if r1.correlation is not None else None,
            "spearman_betaA_vs_deltaAUC_p": float(r1.pvalue) if r1.pvalue is not None else None,
            "spearman_aucE_vs_deltaAUC_r": float(r2.correlation) if r2.correlation is not None else None,
            "spearman_aucE_vs_deltaAUC_p": float(r2.pvalue) if r2.pvalue is not None else None,
            "spearman_betaA_vs_aucAudio_r": float(r3.correlation) if r3.correlation is not None else None,
            "n_subjects": int(len(ind)),
        }
    (OUT / "06_individual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    if ind["delta_auc"].notna().any():
        axes[0].scatter(ind["beta_A_norm"], ind["delta_auc"], c="#ee6c4d")
        axes[0].set_xlabel(r"Acoustic dependence $\|\beta_A\|$")
        axes[0].set_ylabel(r"EEG gain $\Delta$AUC")
        axes[0].set_title("Acoustic dependence vs EEG gain")
        axes[1].scatter(ind["auc_eeg"], ind["delta_auc"], c="#3d5a80")
        axes[1].set_xlabel("EEG-only AUC")
        axes[1].set_ylabel(r"$\Delta$AUC")
        axes[1].set_title("EEG effect vs multimodal gain")
    fig.tight_layout()
    fig.savefig(OUT / "06_individual_diffs.png", bbox_inches="tight")
    plt.close(fig)
    print(summary)
    return ind, summary


# -----------------------------------------------------------------------------
# 7. Ratings beyond acoustics
# -----------------------------------------------------------------------------
def analysis_ratings(df: pd.DataFrame):
    print("\n=== 7. Ratings beyond acoustics (optional) ===")
    if not RATING_CSV.exists():
        print("  rating CSV missing, skip")
        return None
    rat = pd.read_csv(RATING_CSV)
    # aggregate rating deltas by pair_name (mean across participants in that file)
    # Note: rating file may use a different participant set; join on original_name
    agg = rat.groupby("original_name").agg(
        d_urgency=("d_urgency", "mean"),
        d_familiarity=("d_familiarity", "mean"),
        d_liking=("d_liking", "mean"),
    ).reset_index().rename(columns={"original_name": "pair_name"})
    m = df.merge(agg, on="pair_name", how="left")
    m = m.dropna(subset=["d_urgency", "y_right"])
    if len(m) < 100:
        print("  too few rating-matched trials, skip")
        return None

    a_cols = [f"dA_{n}" for n in ACOUSTIC_NAMES]
    y = m["y_right"].to_numpy()
    # model A: acoustics only
    Xa = StandardScaler().fit_transform(m[a_cols].to_numpy())
    # model B: ratings only
    Xr = StandardScaler().fit_transform(m[["d_urgency", "d_familiarity", "d_liking"]].to_numpy())
    # model C: both
    Xb = np.concatenate([Xa, Xr], axis=1)

    def loso_auc(X, y, groups):
        aucs = []
        for tr, te in LeaveOneGroupOut().split(X, y, groups=groups):
            clf = LogisticRegression(max_iter=2000).fit(
                StandardScaler().fit_transform(X[tr]), y[tr]
            )
            sc = StandardScaler().fit(X[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        return float(np.mean(aucs))

    groups = m["subject"].to_numpy()
    auc_a = loso_auc(Xa, y, groups)
    auc_r = loso_auc(Xr, y, groups)
    auc_b = loso_auc(Xb, y, groups)

    # univariate
    uni = {}
    for col in ["d_urgency", "d_familiarity", "d_liking"]:
        r = stats.pointbiserialr(y, m[col].to_numpy())
        uni[col] = {"r": float(r.correlation), "p": float(r.pvalue)}

    summary = {
        "auc_acoustics_only": auc_a,
        "auc_ratings_only": auc_r,
        "auc_acoustics_plus_ratings": auc_b,
        "delta_auc_ratings": auc_b - auc_a,
        "univariate": uni,
        "n_trials": int(len(m)),
    }
    (OUT / "07_ratings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)
    return summary


# -----------------------------------------------------------------------------
# Master
# -----------------------------------------------------------------------------
def main():
    print("Building trial table...")
    df = build_trial_table()
    print(f"Total trials: {len(df)}, subjects: {df['subject'].nunique()}")
    df.to_csv(OUT / "00_trial_table_behavior.csv", index=False)

    pair_df, _ = analysis_pair_priority(df)
    df = fit_pi_stim_loso(df)
    analysis_rt_competition(df)

    df = attach_delta_E(df)
    df.to_csv(OUT / "00_trial_table_with_pi_and_E.csv", index=False)

    df_dev, _ = analysis_deviation(df)
    # carry resid
    df["r_resid"] = df["y_right"] - df["pi_stim"]
    analysis_congruence(df)
    analysis_individual(df)
    analysis_ratings(df)

    # final master summary
    files = sorted(OUT.glob("*.json"))
    master = {}
    for f in files:
        master[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    (OUT / "MASTER_SUMMARY.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
