#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leave-One-Pair-Out (LOPO) protocols for Bradley–Terry intrinsic priority.

Protocol 1 — LOPO-General: after removing pair (A,B), require deg_train(A)>=1 and deg_train(B)>=1
Protocol 2 — LOPO-Strict:  require deg_train(A)>=2 and deg_train(B)>=2

Each eligible unordered pair: hold out ALL trials of that pair (all subjects, AB/BA),
refit M0 and MBT on remaining trials only, predict held-out trials.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TRIAL_CSV = ROOT / "prime_ch4_supplement" / "00_trial_table_behavior.csv"
OUT = ROOT / "bt_intrinsic_priority" / "lopo"
OUT.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(ROOT))
from bt_intrinsic_priority_behavior import (  # noqa: E402
    C_DEFAULT,
    build_design,
    extract_q_from_coef,
    fit_logit,
    predict_proba_pos,
)

RNG = np.random.default_rng(0)
N_BOOT = 2000


def metrics_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def build_graph(left, right):
    opp = defaultdict(set)
    trials = defaultdict(int)
    subjects = defaultdict(set)
    return opp, trials, subjects  # filled by caller


def pair_components(df: pd.DataFrame):
    """Map pair_key -> (A, B) using sorted ids for stability in reporting."""
    rows = []
    for pk, g in df.groupby("pair_key"):
        a = str(g["left_id"].iloc[0])
        b = str(g["right_id"].iloc[0])
        A, B = sorted([a, b])
        rows.append((pk, A, B, g.index.to_numpy()))
    return rows


def train_side_stats(dtr: pd.DataFrame, clip: str):
    L = dtr["left_id"].astype(str).values
    R = dtr["right_id"].astype(str).values
    S = dtr["subject"].astype(str).values
    opp = set()
    n_trials = 0
    subs = set()
    for a, b, s in zip(L, R, S):
        if a == clip:
            n_trials += 1
            subs.add(s)
            opp.add(b)
        elif b == clip:
            n_trials += 1
            subs.add(s)
            opp.add(a)
    return int(n_trials), int(len(subs)), int(len(opp)), sorted(opp)


def fit_q_on_train(dtr, items, pairs, subj_list, C=C_DEFAULT):
    X, y, names, meta, *_ = build_design(
        dtr,
        level="clip",
        include_items=True,
        include_pairs=False,
        item_list=items,
        pair_list=pairs,
        pair_ref=pairs[0],
        subj_list=subj_list,
        subj_ref=subj_list[0],
    )
    clf = fit_logit(X, y, C=C)
    qdf = extract_q_from_coef(names, clf.coef_.ravel(), meta["item_ref"])
    return clf, dict(zip(qdf["item"], qdf["q"])), names, meta


def predict_models(dtr, dte, items, pairs, subj_list, C=C_DEFAULT):
    # M0
    Xtr0, ytr0, *_ = build_design(
        dtr, level="clip", include_items=False, item_list=items,
        pair_list=pairs, pair_ref=pairs[0], subj_list=subj_list, subj_ref=subj_list[0],
    )
    Xte0, yte, *_ = build_design(
        dte, level="clip", include_items=False, item_list=items,
        pair_list=pairs, pair_ref=pairs[0], subj_list=subj_list, subj_ref=subj_list[0],
    )
    p0 = predict_proba_pos(fit_logit(Xtr0, ytr0, C=C), Xte0)

    # MBT (+ q)
    clf_bt, qmap, names, meta = fit_q_on_train(dtr, items, pairs, subj_list, C=C)
    Xte_bt, *_ = build_design(
        dte, level="clip", include_items=True, item_list=items,
        pair_list=pairs, pair_ref=pairs[0], subj_list=subj_list, subj_ref=subj_list[0],
    )
    p_bt = predict_proba_pos(clf_bt, Xte_bt)
    return yte, p0, p_bt, qmap


def pair_cluster_bootstrap_auc(pair_ids, y, p0, pbt, n_boot=N_BOOT, seed=0):
    """Resample held-out pairs with replacement; compute AUC deltas."""
    rng = np.random.default_rng(seed)
    # group indices by pair
    by_pair = defaultdict(list)
    for i, pk in enumerate(pair_ids):
        by_pair[pk].append(i)
    uniq = np.array(list(by_pair.keys()))
    n = len(uniq)
    boot_m0, boot_bt, boot_d = [], [], []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=n, replace=True)
        idx = np.concatenate([by_pair[pk] for pk in samp])
        ys, a0, ab = y[idx], p0[idx], pbt[idx]
        if len(np.unique(ys)) < 2:
            continue
        m0 = roc_auc_score(ys, a0)
        mb = roc_auc_score(ys, ab)
        boot_m0.append(m0)
        boot_bt.append(mb)
        boot_d.append(mb - m0)
    def ci(a):
        a = np.asarray(a, dtype=float)
        return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    return {
        "n_boot_used": int(len(boot_d)),
        "auc_M0_ci95": ci(boot_m0),
        "auc_MBT_ci95": ci(boot_bt),
        "delta_auc_ci95": ci(boot_d),
    }


def run_lopo(df: pd.DataFrame):
    items = sorted(set(df["left_id"].astype(str)) | set(df["right_id"].astype(str)))
    pairs = sorted(df["pair_key"].astype(str).unique())
    subj_list = sorted(df["subject"].astype(str).unique())
    pair_list = pair_components(df)

    # full-graph degrees for quick eligibility (deg-1 after removing edge)
    opp_full = defaultdict(set)
    for a, b in zip(df["left_id"].astype(str), df["right_id"].astype(str)):
        opp_full[a].add(b)
        opp_full[b].add(a)

    per_pair_rows = []
    # OOF stores
    oof = {
        "general": {"pair": [], "y": [], "p0": [], "pbt": []},
        "strict": {"pair": [], "y": [], "p0": [], "pbt": []},
    }

    for pk, A, B, idx in tqdm(pair_list, desc="LOPO pairs"):
        # remaining degrees after removing this edge
        dA = len(opp_full[A]) - 1
        dB = len(opp_full[B]) - 1
        eligible_general = dA >= 1 and dB >= 1
        eligible_strict = dA >= 2 and dB >= 2
        if not eligible_general:
            continue

        te_mask = np.zeros(len(df), dtype=bool)
        te_mask[idx] = True
        # also ensure any trial with this unordered pair is held out (safety)
        # pair_key already unique for unordered pair in this dataset
        dtr = df.loc[~te_mask].copy()
        dte = df.loc[te_mask].copy()

        # verify no A-B / B-A in train
        L = dtr["left_id"].astype(str).values
        R = dtr["right_id"].astype(str).values
        assert not np.any(((L == A) & (R == B)) | ((L == B) & (R == A)))

        nA, nsubA, degA_tr, _ = train_side_stats(dtr, A)
        nB, nsubB, degB_tr, _ = train_side_stats(dtr, B)
        # deg_train should match dA/dB from full-graph-minus-edge
        assert degA_tr == dA and degB_tr == dB, (pk, degA_tr, dA, degB_tr, dB)

        yte, p0, pbt, qmap = predict_models(dtr, dte, items, pairs, subj_list)
        qA = float(qmap.get(A, np.nan))
        qB = float(qmap.get(B, np.nan))
        auc0 = metrics_auc(yte, p0)
        aucb = metrics_auc(yte, pbt)

        base = {
            "held_out_pair": pk,
            "clip_A": A,
            "clip_B": B,
            "train_degree_A": int(degA_tr),
            "train_degree_B": int(degB_tr),
            "train_trials_A": int(nA),
            "train_trials_B": int(nB),
            "n_subjects_A": int(nsubA),
            "n_subjects_B": int(nsubB),
            "q_A_train": qA,
            "q_B_train": qB,
            "n_test_trials": int(len(dte)),
            "auc_M0": auc0,
            "auc_MBT": aucb,
            "delta_auc": (aucb - auc0) if np.isfinite(auc0) and np.isfinite(aucb) else np.nan,
        }

        # always record general
        row_g = dict(base)
        row_g["protocol"] = "general"
        per_pair_rows.append(row_g)
        oof["general"]["pair"].extend([pk] * len(yte))
        oof["general"]["y"].extend(yte.tolist())
        oof["general"]["p0"].extend(p0.tolist())
        oof["general"]["pbt"].extend(pbt.tolist())

        if eligible_strict:
            row_s = dict(base)
            row_s["protocol"] = "strict"
            per_pair_rows.append(row_s)
            oof["strict"]["pair"].extend([pk] * len(yte))
            oof["strict"]["y"].extend(yte.tolist())
            oof["strict"]["p0"].extend(p0.tolist())
            oof["strict"]["pbt"].extend(pbt.tolist())

    per_pair = pd.DataFrame(per_pair_rows)
    per_pair.to_csv(OUT / "lopo_per_pair.csv", index=False)

    summary = {
        "n_unordered_pairs_total": 240,
        "n_eligible_general": int((per_pair.protocol == "general").sum()),
        "n_eligible_strict": int((per_pair.protocol == "strict").sum()),
        "protocols": {},
    }

    for proto in ("general", "strict"):
        y = np.asarray(oof[proto]["y"], dtype=int)
        p0 = np.asarray(oof[proto]["p0"], dtype=float)
        pbt = np.asarray(oof[proto]["pbt"], dtype=float)
        pids = np.asarray(oof[proto]["pair"])
        auc0 = metrics_auc(y, p0)
        aucb = metrics_auc(y, pbt)
        boot = pair_cluster_bootstrap_auc(pids, y, p0, pbt, n_boot=N_BOOT, seed=0)
        # also mean of per-pair delta
        sub = per_pair[per_pair.protocol == proto]
        summary["protocols"][proto] = {
            "n_pairs": int(len(sub)),
            "n_test_trials": int(len(y)),
            "auc_M0": auc0,
            "auc_MBT": aucb,
            "delta_auc": aucb - auc0,
            "pair_cluster_bootstrap": boot,
            "mean_per_pair_delta_auc": float(sub["delta_auc"].mean()),
            "median_per_pair_delta_auc": float(sub["delta_auc"].median()),
            "frac_pairs_delta_auc_gt0": float((sub["delta_auc"] > 0).mean()),
        }
        # save oof trial predictions
        pd.DataFrame(
            {
                "held_out_pair": pids,
                "y_right": y,
                "p_M0": p0,
                "p_MBT": pbt,
                "protocol": proto,
            }
        ).to_csv(OUT / f"lopo_oof_trials_{proto}.csv", index=False)

    (OUT / "lopo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown table
    lines = [
        "# LOPO results",
        "",
        "| Protocol | Eligible pairs | Trials | AUC M0 | AUC MBT | ΔAUC | ΔAUC 95% CI (pair bootstrap) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for proto, lab in (("general", "LOPO-General (d≥1)"), ("strict", "LOPO-Strict (d≥2)")):
        s = summary["protocols"][proto]
        lo, hi = s["pair_cluster_bootstrap"]["delta_auc_ci95"]
        lines.append(
            f"| {lab} | {s['n_pairs']} | {s['n_test_trials']} | "
            f"{s['auc_M0']:.3f} | {s['auc_MBT']:.3f} | {s['delta_auc']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] |"
        )
    lines += [
        "",
        "Per-pair table: `lopo_per_pair.csv`",
        "Fields: held_out_pair, clip_A/B, train_degree/trials/subjects, q_A/B_train, n_test_trials, auc_M0/MBT, delta_auc, protocol",
    ]
    (OUT / "lopo_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summary


def main():
    df = pd.read_csv(TRIAL_CSV)
    df["y_right"] = df["y_right"].astype(int)
    for c in ["left_id", "right_id", "pair_key", "subject"]:
        df[c] = df[c].astype(str)
    assert df["pair_key"].nunique() == 240
    run_lopo(df)


if __name__ == "__main__":
    main()
