#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fold-level audit for Bradley–Terry CV (LOSO + held-out pair).

Exports:
  - fold × clip training statistics + q_hat
  - held-out test-pair side statistics
  - identifiable / low-degree subsets
  - L2 shrinkage diagnostics
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent
TRIAL_CSV = ROOT / "prime_ch4_supplement" / "00_trial_table_behavior.csv"
OUT = ROOT / "bt_intrinsic_priority" / "fold_audit"
OUT.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(ROOT))
from bt_intrinsic_priority_behavior import (  # noqa: E402
    C_DEFAULT,
    MAX_ITER,
    RANDOM_STATE,
    build_design,
    extract_q_from_coef,
    fit_logit,
    predict_proba_pos,
)


def train_clip_stats(dtr: pd.DataFrame) -> pd.DataFrame:
    """Per-clip stats on a training fold."""
    L = dtr["left_id"].astype(str).values
    R = dtr["right_id"].astype(str).values
    subj = dtr["subject"].astype(str).values
    opp = defaultdict(set)
    trials = defaultdict(int)
    subjects = defaultdict(set)
    for i, (a, b, s) in enumerate(zip(L, R, subj)):
        trials[a] += 1
        trials[b] += 1
        subjects[a].add(s)
        subjects[b].add(s)
        opp[a].add(b)
        opp[b].add(a)
    items = sorted(set(L) | set(R))
    rows = []
    for it in items:
        rows.append(
            {
                "clip": it,
                "train_trials": int(trials[it]),
                "n_subjects": int(len(subjects[it])),
                "n_opponents": int(len(opp[it])),
                "degree": int(len(opp[it])),
            }
        )
    return pd.DataFrame(rows)


def fit_q_map(dtr: pd.DataFrame, items: list[str], pairs: list[str], subj_list: list[str], C: float) -> dict[str, float]:
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
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y.astype(int))
    qdf = extract_q_from_coef(names, clf.coef_.ravel(), meta["item_ref"])
    return dict(zip(qdf["item"], qdf["q"]))


def metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "logloss": float("nan"), "balanced_acc": float("nan"), "n": int(len(y))}
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "balanced_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
        "n": int(len(y)),
    }


def audit_loso(df: pd.DataFrame, items, pairs, subj_list, C: float = C_DEFAULT):
    logo = LeaveOneGroupOut()
    y = df["y_right"].astype(int).values
    groups = df["subject"].astype(str).values
    clip_rows = []
    fold_meta = []
    for fold_i, (tr, te) in enumerate(logo.split(df, y, groups)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        held_subj = str(dte["subject"].iloc[0])
        stats = train_clip_stats(dtr)
        qmap = fit_q_map(dtr, items, pairs, subj_list, C=C)
        stats = stats.copy()
        stats["fold"] = fold_i
        stats["scheme"] = "loso"
        stats["held_out"] = held_subj
        stats["q_hat"] = stats["clip"].map(qmap)
        # clips not in train get NaN q
        clip_rows.append(stats)
        fold_meta.append(
            {
                "scheme": "loso",
                "fold": fold_i,
                "held_out": held_subj,
                "n_train_trials": int(len(dtr)),
                "n_test_trials": int(len(dte)),
                "n_train_subjects": int(dtr["subject"].nunique()),
                "n_train_pairs": int(dtr["pair_key"].nunique()),
                "n_train_clips": int(len(stats)),
                "joint_fit": "all training subjects + all training pairs (single L2 logit)",
            }
        )
    return pd.concat(clip_rows, ignore_index=True), pd.DataFrame(fold_meta)


def audit_heldout_pair(df: pd.DataFrame, items, pairs, subj_list, C: float = C_DEFAULT, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    y = df["y_right"].astype(int).values
    groups = df["pair_key"].astype(str).values
    clip_rows = []
    pair_rows = []
    fold_meta = []
    preds = {
        "all": {"m0": np.full(len(df), np.nan), "mbt": np.full(len(df), np.nan)},
        "ident_deg_ge1": {"m0": np.full(len(df), np.nan), "mbt": np.full(len(df), np.nan), "mask": np.zeros(len(df), bool)},
        "ident_deg_ge2": {"m0": np.full(len(df), np.nan), "mbt": np.full(len(df), np.nan), "mask": np.zeros(len(df), bool)},
        "ident_deg_eq1_either": {"m0": np.full(len(df), np.nan), "mbt": np.full(len(df), np.nan), "mask": np.zeros(len(df), bool)},
    }

    for fold_i, (tr, te) in enumerate(gkf.split(df, y, groups)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        stats = train_clip_stats(dtr)
        qmap = fit_q_map(dtr, items, pairs, subj_list, C=C)
        st = stats.copy()
        st["fold"] = fold_i
        st["scheme"] = "heldout_pair"
        st["held_out"] = f"pairfold_{fold_i}"
        st["q_hat"] = st["clip"].map(qmap)
        clip_rows.append(st)

        # index train stats
        by_clip = stats.set_index("clip")
        tropp = defaultdict(set)
        for a, b in zip(dtr["left_id"].astype(str), dtr["right_id"].astype(str)):
            tropp[a].add(b)
            tropp[b].add(a)

        # predictions for subset analysis
        for include_items, key in ((False, "m0"), (True, "mbt")):
            Xtr, ytr, *_ = build_design(
                dtr, level="clip", include_items=include_items, item_list=items,
                pair_list=pairs, pair_ref=pairs[0], subj_list=subj_list, subj_ref=subj_list[0],
            )
            Xte, *_ = build_design(
                dte, level="clip", include_items=include_items, item_list=items,
                pair_list=pairs, pair_ref=pairs[0], subj_list=subj_list, subj_ref=subj_list[0],
            )
            p = predict_proba_pos(fit_logit(Xtr, ytr, C=C), Xte)
            preds["all"][key][te] = p
            preds["ident_deg_ge1"][key][te] = p
            preds["ident_deg_ge2"][key][te] = p
            preds["ident_deg_eq1_either"][key][te] = p

        te_idx = dte.index.to_numpy()
        L = dte["left_id"].astype(str).values
        R = dte["right_id"].astype(str).values
        pk = dte["pair_key"].astype(str).values
        yy = dte["y_right"].astype(int).values
        mask_ge1 = np.zeros(len(dte), dtype=bool)
        mask_ge2 = np.zeros(len(dte), dtype=bool)
        mask_eq1 = np.zeros(len(dte), dtype=bool)

        for j, (left, right, pair_key, yj, orig_i) in enumerate(zip(L, R, pk, yy, te_idx)):
            def side(clip):
                if clip not in by_clip.index:
                    return dict(
                        in_train=0, train_trials=0, n_subjects=0, n_opponents=0, degree=0, q_hat=np.nan
                    )
                r = by_clip.loc[clip]
                return dict(
                    in_train=1,
                    train_trials=int(r["train_trials"]),
                    n_subjects=int(r["n_subjects"]),
                    n_opponents=int(r["n_opponents"]),
                    degree=int(r["degree"]),
                    q_hat=float(qmap.get(clip, np.nan)),
                )

            sl, sr = side(left), side(right)
            both = sl["in_train"] and sr["in_train"]
            deg_l, deg_r = sl["degree"], sr["degree"]
            ge1 = both and deg_l >= 1 and deg_r >= 1
            ge2 = both and deg_l >= 2 and deg_r >= 2
            eq1_either = ge1 and (deg_l == 1 or deg_r == 1)
            mask_ge1[j] = ge1
            mask_ge2[j] = ge2
            mask_eq1[j] = eq1_either
            pair_rows.append(
                {
                    "scheme": "heldout_pair",
                    "fold": fold_i,
                    "pair_key": pair_key,
                    "left_id": left,
                    "right_id": right,
                    "y_right": int(yj),
                    "both_in_train": int(both),
                    "ident_deg_ge1": int(ge1),
                    "ident_deg_ge2": int(ge2),
                    "ident_deg_eq1_either": int(eq1_either),
                    "L_train_trials": sl["train_trials"],
                    "L_n_subjects": sl["n_subjects"],
                    "L_n_opponents": sl["n_opponents"],
                    "L_degree": sl["degree"],
                    "L_q_hat": sl["q_hat"],
                    "R_train_trials": sr["train_trials"],
                    "R_n_subjects": sr["n_subjects"],
                    "R_n_opponents": sr["n_opponents"],
                    "R_degree": sr["degree"],
                    "R_q_hat": sr["q_hat"],
                    "min_side_degree": int(min(deg_l, deg_r)) if both else 0,
                    "min_side_trials": int(min(sl["train_trials"], sr["train_trials"])) if both else 0,
                }
            )

        preds["ident_deg_ge1"]["mask"][te_idx] = mask_ge1
        preds["ident_deg_ge2"]["mask"][te_idx] = mask_ge2
        preds["ident_deg_eq1_either"]["mask"][te_idx] = mask_eq1

        fold_meta.append(
            {
                "scheme": "heldout_pair",
                "fold": fold_i,
                "held_out": f"pairfold_{fold_i}",
                "n_train_trials": int(len(dtr)),
                "n_test_trials": int(len(dte)),
                "n_train_subjects": int(dtr["subject"].nunique()),
                "n_train_pairs": int(dtr["pair_key"].nunique()),
                "n_test_pairs": int(dte["pair_key"].nunique()),
                "n_train_clips": int(len(stats)),
                "frac_test_both_in_train": float(np.mean([r["both_in_train"] for r in pair_rows if r["fold"] == fold_i])),
                "frac_test_ident_deg_ge1": float(mask_ge1.mean()),
                "frac_test_ident_deg_ge2": float(mask_ge2.mean()),
                "frac_test_ident_deg_eq1_either": float(mask_eq1.mean()),
                "joint_fit": "all training subjects + all training pairs (single L2 logit)",
            }
        )

    # subset metrics
    y_all = df["y_right"].astype(int).values
    subset_metrics = {}
    for name, blob in preds.items():
        if name == "all":
            mask = np.ones(len(df), dtype=bool)
        else:
            mask = blob["mask"]
        subset_metrics[name] = {
            "frac_trials": float(mask.mean()),
            "n": int(mask.sum()),
            "M0": metrics(y_all[mask], blob["m0"][mask]),
            "MBT": metrics(y_all[mask], blob["mbt"][mask]),
            "delta_auc": float(
                metrics(y_all[mask], blob["mbt"][mask])["auc"]
                - metrics(y_all[mask], blob["m0"][mask])["auc"]
            ),
        }

    return (
        pd.concat(clip_rows, ignore_index=True),
        pd.DataFrame(pair_rows),
        pd.DataFrame(fold_meta),
        subset_metrics,
    )


def l2_shrinkage_audit(df, items, pairs, subj_list):
    """Compare q under strong/default/weak L2 on full data and on one pair-fold."""
    Cs = [0.1, 1.0, 10.0, 100.0]
    # full-data descriptive
    full_rows = []
    stats_full = train_clip_stats(df)
    q_by_C = {}
    for C in Cs:
        q_by_C[C] = fit_q_map(df, items, pairs, subj_list, C=C)
    for _, r in stats_full.iterrows():
        clip = r["clip"]
        row = {
            "scope": "full_data",
            "clip": clip,
            "train_trials": int(r["train_trials"]),
            "n_opponents": int(r["n_opponents"]),
            "degree": int(r["degree"]),
        }
        for C in Cs:
            row[f"q_C{C}"] = q_by_C[C].get(clip, np.nan)
        # shrinkage vs weak reg: |q_C1| / |q_C100|
        q_weak = q_by_C[100.0].get(clip, np.nan)
        q_def = q_by_C[1.0].get(clip, np.nan)
        row["abs_q_C1"] = abs(q_def) if pd.notna(q_def) else np.nan
        row["abs_q_C100"] = abs(q_weak) if pd.notna(q_weak) else np.nan
        row["shrink_ratio_C1_over_C100"] = (
            abs(q_def) / abs(q_weak) if pd.notna(q_def) and pd.notna(q_weak) and abs(q_weak) > 1e-8 else np.nan
        )
        full_rows.append(row)
    full_df = pd.DataFrame(full_rows)

    # one held-out pair fold (fold 0)
    gkf = GroupKFold(n_splits=5)
    y = df["y_right"].astype(int).values
    groups = df["pair_key"].astype(str).values
    tr, te = next(gkf.split(df, y, groups))
    dtr = df.iloc[tr]
    stats_tr = train_clip_stats(dtr)
    qfold = {C: fit_q_map(dtr, items, pairs, subj_list, C=C) for C in Cs}
    fold_rows = []
    for _, r in stats_tr.iterrows():
        clip = r["clip"]
        row = {
            "scope": "heldout_pair_fold0_train",
            "clip": clip,
            "train_trials": int(r["train_trials"]),
            "n_opponents": int(r["n_opponents"]),
            "degree": int(r["degree"]),
        }
        for C in Cs:
            row[f"q_C{C}"] = qfold[C].get(clip, np.nan)
        q_weak = qfold[100.0].get(clip, np.nan)
        q_def = qfold[1.0].get(clip, np.nan)
        row["abs_q_C1"] = abs(q_def) if pd.notna(q_def) else np.nan
        row["abs_q_C100"] = abs(q_weak) if pd.notna(q_weak) else np.nan
        row["shrink_ratio_C1_over_C100"] = (
            abs(q_def) / abs(q_weak) if pd.notna(q_def) and pd.notna(q_weak) and abs(q_weak) > 1e-8 else np.nan
        )
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)

    def summarize(d: pd.DataFrame, label: str):
        out = {"scope": label}
        for deg_lab, mask in (
            ("deg1", d["degree"] == 1),
            ("deg2", d["degree"] == 2),
            ("deg_ge3", d["degree"] >= 3),
            ("trials_le_median", d["train_trials"] <= d["train_trials"].median()),
            ("trials_gt_median", d["train_trials"] > d["train_trials"].median()),
        ):
            sub = d.loc[mask, "shrink_ratio_C1_over_C100"].dropna()
            out[f"{deg_lab}_n"] = int(mask.sum())
            out[f"{deg_lab}_mean_shrink_ratio"] = float(sub.mean()) if len(sub) else float("nan")
            out[f"{deg_lab}_median_shrink_ratio"] = float(sub.median()) if len(sub) else float("nan")
            out[f"{deg_lab}_mean_abs_q_C1"] = float(d.loc[mask, "abs_q_C1"].mean())
            out[f"{deg_lab}_mean_abs_q_C100"] = float(d.loc[mask, "abs_q_C100"].mean())
        return out

    summary = {
        "note": "shrink_ratio = |q(C=1)| / |q(C=100)|; smaller => stronger L2 shrinkage toward 0",
        "full_data": summarize(full_df, "full_data"),
        "heldout_pair_fold0_train": summarize(fold_df, "heldout_pair_fold0_train"),
    }
    return pd.concat([full_df, fold_df], ignore_index=True), summary


def main():
    df = pd.read_csv(TRIAL_CSV)
    df["y_right"] = df["y_right"].astype(int)
    for c in ["left_id", "right_id", "pair_key", "subject"]:
        df[c] = df[c].astype(str)

    items = sorted(set(df["left_id"]) | set(df["right_id"]))
    pairs = sorted(df["pair_key"].unique())
    subj_list = sorted(df["subject"].unique())

    print("=== LOSO fold×clip audit ===")
    loso_clips, loso_meta = audit_loso(df, items, pairs, subj_list)
    loso_clips.to_csv(OUT / "fold_clip_loso.csv", index=False)
    loso_meta.to_csv(OUT / "fold_meta_loso.csv", index=False)

    print("=== Held-out pair fold×clip + test-pair sides ===")
    hp_clips, hp_pairs, hp_meta, subset_metrics = audit_heldout_pair(df, items, pairs, subj_list)
    hp_clips.to_csv(OUT / "fold_clip_heldout_pair.csv", index=False)
    hp_pairs.to_csv(OUT / "heldout_pair_test_sides.csv", index=False)
    hp_meta.to_csv(OUT / "fold_meta_heldout_pair.csv", index=False)

    # compact table user asked: fold | clip | train_trials | n_subjects | n_opponents | degree | q_hat
    compact = pd.concat(
        [
            loso_clips[["fold", "scheme", "held_out", "clip", "train_trials", "n_subjects", "n_opponents", "degree", "q_hat"]],
            hp_clips[["fold", "scheme", "held_out", "clip", "train_trials", "n_subjects", "n_opponents", "degree", "q_hat"]],
        ],
        ignore_index=True,
    )
    compact.to_csv(OUT / "fold_clip_qhat_compact.csv", index=False)

    # focus: unique test pairs with side stats (collapse identical pair_key within fold)
    focus = (
        hp_pairs.groupby(["fold", "pair_key"], as_index=False)
        .agg(
            n_trials=("y_right", "count"),
            both_in_train=("both_in_train", "max"),
            ident_deg_ge1=("ident_deg_ge1", "max"),
            ident_deg_ge2=("ident_deg_ge2", "max"),
            ident_deg_eq1_either=("ident_deg_eq1_either", "max"),
            left_id=("left_id", "first"),
            right_id=("right_id", "first"),
            L_train_trials=("L_train_trials", "first"),
            L_n_opponents=("L_n_opponents", "first"),
            L_degree=("L_degree", "first"),
            L_q_hat=("L_q_hat", "first"),
            R_train_trials=("R_train_trials", "first"),
            R_n_opponents=("R_n_opponents", "first"),
            R_degree=("R_degree", "first"),
            R_q_hat=("R_q_hat", "first"),
            min_side_degree=("min_side_degree", "first"),
            min_side_trials=("min_side_trials", "first"),
        )
    )
    focus.to_csv(OUT / "heldout_pair_focus_by_pair.csv", index=False)

    print("=== L2 shrinkage ===")
    shrink_df, shrink_sum = l2_shrinkage_audit(df, items, pairs, subj_list)
    shrink_df.to_csv(OUT / "l2_shrinkage_by_clip.csv", index=False)

    # answers / summary
    # degree==1 in current identifiable_ge1 definition?
    n_ge1 = int(hp_pairs["ident_deg_ge1"].sum())
    n_eq1 = int(hp_pairs["ident_deg_eq1_either"].sum())
    summary = {
        "joint_fit_confirmation": (
            "YES. In each fold, a single L2 logistic regression is fit on ALL training subjects "
            "and ALL training pairs jointly (design: intercept + subject FE + item-diff q columns). "
            "There is no separate per-subject or per-pair q estimation."
        ),
        "identifiable_definition_current": (
            "ident_deg_ge1 = both clips appear in the training graph (degree>=1 each). "
            "This INCLUDES clips that faced only one opponent in train (degree==1)."
        ),
        "deg1_included_in_ident_ge1": True,
        "heldout_pair_subset_metrics": subset_metrics,
        "heldout_pair_deg1_share_among_ident_ge1_trials": float(n_eq1 / n_ge1) if n_ge1 else float("nan"),
        "l2_shrinkage_summary": shrink_sum,
        "files": {
            "compact": str(OUT / "fold_clip_qhat_compact.csv"),
            "loso_clips": str(OUT / "fold_clip_loso.csv"),
            "heldout_clips": str(OUT / "fold_clip_heldout_pair.csv"),
            "heldout_test_sides": str(OUT / "heldout_pair_test_sides.csv"),
            "heldout_focus_pairs": str(OUT / "heldout_pair_focus_by_pair.csv"),
            "l2_shrinkage": str(OUT / "l2_shrinkage_by_clip.csv"),
        },
    }
    (OUT / "AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # human-readable brief
    lines = [
        "# Fold-level BT audit summary",
        "",
        "## Joint fit?",
        summary["joint_fit_confirmation"],
        "",
        "## Identifiable subset & degree==1",
        summary["identifiable_definition_current"],
        f"- Among ident_deg_ge1 trials, fraction with either side degree==1: {summary['heldout_pair_deg1_share_among_ident_ge1_trials']:.3f}",
        "",
        "## Held-out pair ΔAUC by subset",
    ]
    for k, v in subset_metrics.items():
        lines.append(
            f"- {k}: frac={v['frac_trials']:.3f}, n={v['n']}, "
            f"M0_AUC={v['M0']['auc']:.3f}, MBT_AUC={v['MBT']['auc']:.3f}, ΔAUC={v['delta_auc']:.3f}"
        )
    lines += [
        "",
        "## L2 shrinkage (|q_C1|/|q_C100|; smaller = more shrinkage)",
        f"- full_data deg1 mean ratio: {shrink_sum['full_data'].get('deg1_mean_shrink_ratio')}",
        f"- full_data deg_ge3 mean ratio: {shrink_sum['full_data'].get('deg_ge3_mean_shrink_ratio')}",
        f"- fold0-train deg1 mean ratio: {shrink_sum['heldout_pair_fold0_train'].get('deg1_mean_shrink_ratio')}",
        f"- fold0-train deg_ge3 mean ratio: {shrink_sum['heldout_pair_fold0_train'].get('deg_ge3_mean_shrink_ratio')}",
        "",
        f"Tables in: {OUT}",
    ]
    (OUT / "AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
