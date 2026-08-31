#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bradley–Terry intrinsic-priority behavioral tests (4 steps).

Data: prime_ch4_supplement/00_trial_table_behavior.csv
Main: clip-level q_i; supplement: subcategory-level q.
Outputs: bt_intrinsic_priority/
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
TRIAL_CSV = ROOT / "prime_ch4_supplement" / "00_trial_table_behavior.csv"
OUT = ROOT / "bt_intrinsic_priority"
OUT.mkdir(parents=True, exist_ok=True)

# L2 strength for sklearn Logit (smaller C = stronger penalty)
C_DEFAULT = 1.0
C_GRID = [0.1, 1.0, 10.0]
MAX_ITER = 400
RANDOM_STATE = 0
N_PAIR_FOLDS = 5
N_TRANS_NULL = 100
RNG = np.random.default_rng(RANDOM_STATE)


# -----------------------------------------------------------------------------
# Metrics / helpers
# -----------------------------------------------------------------------------
def _safe_auc(y, p) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_logloss(y, p) -> float:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y, p, labels=[0, 1]))


def _safe_bacc(y, p) -> float:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    return float(balanced_accuracy_score(y, pred))


def metrics_dict(y, p) -> dict:
    return {
        "auc": _safe_auc(y, p),
        "logloss": _safe_logloss(y, p),
        "balanced_acc": _safe_bacc(y, p),
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
    }


def fit_logit(X: np.ndarray, y: np.ndarray, C: float = C_DEFAULT) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y.astype(int))
    return clf


def predict_proba_pos(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


# -----------------------------------------------------------------------------
# Design matrices
# -----------------------------------------------------------------------------
def _subject_dummies(subjects: np.ndarray, all_subjects: list[str], ref: str):
    """Fixed vocabulary subject FE so train/test column counts match."""
    subs = np.asarray(subjects)
    cols = [s for s in all_subjects if s != ref]
    idx = {s: i for i, s in enumerate(cols)}
    X = np.zeros((len(subs), len(cols)), dtype=float)
    for i, s in enumerate(subs):
        j = idx.get(s)
        if j is not None:
            X[i, j] = 1.0
    return X, cols


def _item_diff_matrix(left: np.ndarray, right: np.ndarray, items: list[str], ref: str):
    """Columns for non-ref items: 1{right=i} - 1{left=i}."""
    cols = [it for it in items if it != ref]
    idx = {it: i for i, it in enumerate(cols)}
    X = np.zeros((len(left), len(cols)), dtype=float)
    for t, (L, R) in enumerate(zip(left, right)):
        if R in idx:
            X[t, idx[R]] += 1.0
        if L in idx:
            X[t, idx[L]] -= 1.0
    return X, cols


def _pair_residual_matrix(
    left: np.ndarray,
    right: np.ndarray,
    pair_keys: np.ndarray,
    all_pairs: list[str],
    ref_pair: str,
):
    """
    Antisymmetric unordered-pair residual:
    if right is lex-larger than left, +r_pair else -r_pair.
    """
    keys = np.asarray(pair_keys)
    cols = [k for k in all_pairs if k != ref_pair]
    idx = {k: i for i, k in enumerate(cols)}
    X = np.zeros((len(keys), len(cols)), dtype=float)
    for t, (L, R, pk) in enumerate(zip(left, right, keys)):
        j = idx.get(pk)
        if j is None:
            continue
        sign = 1.0 if R > L else -1.0
        X[t, j] = sign
    return X, cols


def build_design(
    df: pd.DataFrame,
    level: str = "clip",
    include_items: bool = True,
    include_pairs: bool = False,
    item_list: list[str] | None = None,
    pair_list: list[str] | None = None,
    pair_ref: str | None = None,
    subj_list: list[str] | None = None,
    subj_ref: str | None = None,
):
    """
    level='clip' uses left_id/right_id; level='subcat' uses left_cat/right_cat.
    Always includes intercept (side bias) + subject FE with fixed vocabularies.
    """
    if level == "clip":
        left = df["left_id"].astype(str).values
        right = df["right_id"].astype(str).values
    else:
        left = df["left_cat"].astype(str).values
        right = df["right_cat"].astype(str).values

    y = df["y_right"].astype(int).values
    subjects = df["subject"].astype(str).values
    pair_keys = df["pair_key"].astype(str).values

    if subj_list is None:
        subj_list = sorted(pd.unique(subjects))
    if subj_ref is None:
        subj_ref = subj_list[0]

    Xs, sub_cols = _subject_dummies(subjects, subj_list, ref=subj_ref)
    parts = [np.ones((len(df), 1), dtype=float), Xs]
    names = ["intercept"] + [f"subj:{c}" for c in sub_cols]
    meta = {"subj_ref": subj_ref, "item_ref": None, "pair_ref": None, "item_cols": [], "pair_cols": []}

    if include_items:
        if item_list is None:
            item_list = sorted(set(left) | set(right))
        item_ref = item_list[0]
        Xi, item_cols = _item_diff_matrix(left, right, item_list, item_ref)
        parts.append(Xi)
        names += [f"q:{c}" for c in item_cols]
        meta["item_ref"] = item_ref
        meta["item_cols"] = item_cols

    if include_pairs:
        if pair_list is None:
            pair_list = sorted(pd.unique(pair_keys))
        if pair_ref is None:
            pair_ref = pair_list[0]
        Xp, pair_cols = _pair_residual_matrix(left, right, pair_keys, all_pairs=pair_list, ref_pair=pair_ref)
        parts.append(Xp)
        names += [f"r:{c}" for c in pair_cols]
        meta["pair_ref"] = pair_ref
        meta["pair_cols"] = pair_cols

    X = np.hstack(parts) if parts else np.ones((len(df), 1))
    return X, y, names, meta, left, right, pair_keys


def extract_q_from_coef(names: list[str], coef: np.ndarray, item_ref: str) -> pd.DataFrame:
    """Map differential coefficients back to q with q_ref=0, then center."""
    q = {item_ref: 0.0}
    for n, c in zip(names, coef):
        if n.startswith("q:"):
            q[n[2:]] = float(c)
    # center to sum-to-zero
    vals = np.array(list(q.values()))
    mean = vals.mean()
    rows = [{"item": k, "q": v - mean} for k, v in q.items()]
    return pd.DataFrame(rows).sort_values("q", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# CV evaluators
# -----------------------------------------------------------------------------
def cv_loso(df: pd.DataFrame, level: str, include_items: bool, include_pairs: bool = False, C: float = C_DEFAULT):
    logo = LeaveOneGroupOut()
    groups = df["subject"].astype(str).values
    y_all = df["y_right"].astype(int).values
    preds = np.full(len(df), np.nan)

    if level == "clip":
        items = sorted(set(df["left_id"].astype(str)) | set(df["right_id"].astype(str)))
    else:
        items = sorted(set(df["left_cat"].astype(str)) | set(df["right_cat"].astype(str)))
    pairs = sorted(df["pair_key"].astype(str).unique())
    pair_ref = pairs[0]
    subj_list = sorted(df["subject"].astype(str).unique())
    subj_ref = subj_list[0]

    fold_rows = []
    for tr, te in logo.split(df, y_all, groups):
        dtr, dte = df.iloc[tr], df.iloc[te]
        Xtr, ytr, names, meta, *_ = build_design(
            dtr, level=level, include_items=include_items, include_pairs=include_pairs,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        Xte, yte, *_ = build_design(
            dte, level=level, include_items=include_items, include_pairs=include_pairs,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        clf = fit_logit(Xtr, ytr, C=C)
        p = predict_proba_pos(clf, Xte)
        preds[te] = p
        m = metrics_dict(yte, p)
        m["held_subject"] = str(dte["subject"].iloc[0])
        fold_rows.append(m)

    overall = metrics_dict(y_all, preds)
    return overall, pd.DataFrame(fold_rows), preds


def cv_by_pair(df: pd.DataFrame, level: str, include_items: bool, include_pairs: bool = False, C: float = C_DEFAULT, n_splits: int = N_PAIR_FOLDS):
    groups = df["pair_key"].astype(str).values
    y_all = df["y_right"].astype(int).values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.full(len(df), np.nan)

    if level == "clip":
        items = sorted(set(df["left_id"].astype(str)) | set(df["right_id"].astype(str)))
    else:
        items = sorted(set(df["left_cat"].astype(str)) | set(df["right_cat"].astype(str)))
    pairs = sorted(df["pair_key"].astype(str).unique())
    pair_ref = pairs[0]
    subj_list = sorted(df["subject"].astype(str).unique())
    subj_ref = subj_list[0]

    fold_rows = []
    for fold_i, (tr, te) in enumerate(gkf.split(df, y_all, groups)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        Xtr, ytr, *_ = build_design(
            dtr, level=level, include_items=include_items, include_pairs=include_pairs,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        Xte, yte, *_ = build_design(
            dte, level=level, include_items=include_items, include_pairs=include_pairs,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        clf = fit_logit(Xtr, ytr, C=C)
        p = predict_proba_pos(clf, Xte)
        preds[te] = p
        m = metrics_dict(yte, p)
        m["fold"] = fold_i
        fold_rows.append(m)

    overall = metrics_dict(y_all, preds)
    return overall, pd.DataFrame(fold_rows), preds


# -----------------------------------------------------------------------------
# Step 1
# -----------------------------------------------------------------------------
def step1(df: pd.DataFrame) -> dict:
    rows = []
    details = {}
    for level in ("clip", "subcat"):
        for model_name, include_items in (("M0_side", False), ("MBT", True)):
            for scheme, fn in (("loso", cv_loso), ("heldout_pair", cv_by_pair)):
                overall, fold_df, preds = fn(df, level=level, include_items=include_items)
                row = {
                    "level": level,
                    "model": model_name,
                    "scheme": scheme,
                    **{k: overall[k] for k in ("auc", "logloss", "balanced_acc", "n", "base_rate")},
                }
                rows.append(row)
                key = f"{level}_{model_name}_{scheme}"
                details[key] = {"overall": overall, "folds": fold_df.to_dict(orient="records")}
                fold_df.to_csv(OUT / f"01_folds_{key}.csv", index=False)

    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / "01_step1_model_compare.csv", index=False)

    # Delta MBT - M0
    piv = []
    for level in ("clip", "subcat"):
        for scheme in ("loso", "heldout_pair"):
            m0 = cmp[(cmp.level == level) & (cmp.model == "M0_side") & (cmp.scheme == scheme)].iloc[0]
            mb = cmp[(cmp.level == level) & (cmp.model == "MBT") & (cmp.scheme == scheme)].iloc[0]
            piv.append({
                "level": level,
                "scheme": scheme,
                "delta_auc": mb["auc"] - m0["auc"],
                "delta_logloss": mb["logloss"] - m0["logloss"],
                "delta_balanced_acc": mb["balanced_acc"] - m0["balanced_acc"],
                "m0_auc": m0["auc"],
                "mbt_auc": mb["auc"],
                "m0_logloss": m0["logloss"],
                "mbt_logloss": mb["logloss"],
            })
    delta = pd.DataFrame(piv)
    delta.to_csv(OUT / "01_step1_deltas.csv", index=False)

    summary = {"compare": rows, "deltas": piv}
    with open(OUT / "01_step1_model_compare.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# -----------------------------------------------------------------------------
# Step 2: leave-one-opponent-out
# -----------------------------------------------------------------------------
def opponent_degree(df: pd.DataFrame, level: str = "clip") -> pd.DataFrame:
    if level == "clip":
        L, R = "left_id", "right_id"
    else:
        L, R = "left_cat", "right_cat"
    opp = defaultdict(set)
    left = df[L].astype(str).values
    right = df[R].astype(str).values
    for a, b in zip(left, right):
        opp[a].add(b)
        opp[b].add(a)
    rows = [{"item": k, "n_opponents": len(v), "opponents": "|".join(sorted(v))} for k, v in opp.items()]
    return pd.DataFrame(rows).sort_values("n_opponents", ascending=False)


def step2_loo_opponent(df: pd.DataFrame, level: str = "clip", C: float = C_DEFAULT) -> dict:
    if level == "clip":
        Lcol, Rcol = "left_id", "right_id"
    else:
        Lcol, Rcol = "left_cat", "right_cat"

    left = df[Lcol].astype(str).values
    right = df[Rcol].astype(str).values
    y = df["y_right"].astype(int).values
    items = sorted(set(left) | set(right))
    pairs = sorted(df["pair_key"].astype(str).unique())
    pair_ref = pairs[0]
    subj_list = sorted(df["subject"].astype(str).unique())
    subj_ref = subj_list[0]

    # map item -> opponents and trial indices for each unordered {A,O}
    deg = opponent_degree(df, level=level)
    multi = set(deg.loc[deg["n_opponents"] >= 2, "item"])

    # Unique unordered matchups (one fit per held-out pair)
    pair_trials = {}
    for a, b in zip(left, right):
        key = tuple(sorted([a, b]))
        if key not in pair_trials:
            mask = ((left == key[0]) & (right == key[1])) | ((left == key[1]) & (right == key[0]))
            pair_trials[key] = np.where(mask)[0]

    tasks = []
    for (a, b), te_idx in pair_trials.items():
        focals = [x for x in (a, b) if x in multi]
        if not focals:
            continue
        tasks.append((a, b, focals, te_idx))

    by_clip_rows = []
    all_y, all_p_bt, all_p_m0, all_p_emp = [], [], [], []

    print(f"[Step2 {level}] LOOO unique matchups: {len(tasks)}")
    for ti, (a, b, focals, te_idx) in enumerate(tasks):
        te_set = set(te_idx.tolist())
        tr_mask = np.ones(len(df), dtype=bool)
        tr_mask[te_idx] = False
        if tr_mask.sum() < 50 or len(te_idx) < 1:
            continue
        dtr = df.iloc[tr_mask]
        dte = df.iloc[te_idx]

        Xtr, ytr, *_ = build_design(
            dtr, level=level, include_items=True, include_pairs=False,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        Xte, yte, *_ = build_design(
            dte, level=level, include_items=True, include_pairs=False,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        clf_bt = fit_logit(Xtr, ytr, C=C)
        p_bt = predict_proba_pos(clf_bt, Xte)

        Xtr0, ytr0, *_ = build_design(
            dtr, level=level, include_items=False, include_pairs=False,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        Xte0, *_ = build_design(
            dte, level=level, include_items=False, include_pairs=False,
            item_list=items, pair_list=pairs, pair_ref=pair_ref, subj_list=subj_list, subj_ref=subj_ref,
        )
        clf0 = fit_logit(Xtr0, ytr0, C=C)
        p_m0 = predict_proba_pos(clf0, Xte0)

        Ltr = dtr[Lcol].astype(str).values
        Rtr = dtr[Rcol].astype(str).values
        ytr_arr = dtr["y_right"].astype(int).values
        Lte = dte[Lcol].astype(str).values
        Rte = dte[Rcol].astype(str).values

        # Record once in pooled pools (avoid double-counting identical y/p)
        all_y.append(yte)
        all_p_bt.append(p_bt)
        all_p_m0.append(p_m0)

        # Emp baseline pooled later per-focal; store first focal's emp for overall pool using matchup mid
        # Use neutral 0.5 for pooled emp placeholder; per-focal rows below have proper emp
        all_p_emp.append(np.full(len(yte), 0.5))

        for focal in focals:
            opp = b if focal == a else a
            foc_right = Rtr == focal
            foc_left = Ltr == focal
            wins = []
            if foc_right.any():
                wins.append(ytr_arr[foc_right].mean())
            if foc_left.any():
                wins.append(1.0 - ytr_arr[foc_left].mean())
            p_win_focal = float(np.mean(wins)) if wins else 0.5
            p_emp = np.where(Rte == focal, p_win_focal, 1.0 - p_win_focal)

            m_bt = metrics_dict(yte, p_bt)
            m_m0 = metrics_dict(yte, p_m0)
            m_emp = metrics_dict(yte, p_emp)
            by_clip_rows.append({
                "level": level,
                "focal": focal,
                "opponent": opp,
                "n_test": int(len(yte)),
                "auc_bt": m_bt["auc"],
                "auc_m0": m_m0["auc"],
                "auc_emp": m_emp["auc"],
                "logloss_bt": m_bt["logloss"],
                "logloss_m0": m_m0["logloss"],
                "logloss_emp": m_emp["logloss"],
                "bacc_bt": m_bt["balanced_acc"],
                "bacc_m0": m_m0["balanced_acc"],
                "bacc_emp": m_emp["balanced_acc"],
            })

        if (ti + 1) % 25 == 0:
            print(f"  ... {ti+1}/{len(tasks)}", flush=True)

    by_df = pd.DataFrame(by_clip_rows)
    by_df.to_csv(OUT / f"02_step2_loo_opponent_by_pair_{level}.csv", index=False)

    ycat = np.concatenate(all_y) if all_y else np.array([])
    overall = {
        "level": level,
        "n_tasks": len(by_clip_rows),
        "bt": metrics_dict(ycat, np.concatenate(all_p_bt)) if len(ycat) else {},
        "m0": metrics_dict(ycat, np.concatenate(all_p_m0)) if len(ycat) else {},
        "emp": metrics_dict(ycat, np.concatenate(all_p_emp)) if len(ycat) else {},
    }
    if overall["bt"] and overall["m0"]:
        overall["delta_auc_bt_minus_m0"] = overall["bt"]["auc"] - overall["m0"]["auc"]
        overall["delta_logloss_bt_minus_m0"] = overall["bt"]["logloss"] - overall["m0"]["logloss"]
        overall["delta_auc_bt_minus_emp"] = overall["bt"]["auc"] - overall["emp"]["auc"]

    # per-focal aggregate
    if len(by_df):
        foc = by_df.groupby("focal", as_index=False).agg(
            n_opponents_tested=("opponent", "count"),
            mean_auc_bt=("auc_bt", "mean"),
            mean_auc_m0=("auc_m0", "mean"),
            mean_logloss_bt=("logloss_bt", "mean"),
            mean_logloss_m0=("logloss_m0", "mean"),
        )
        foc.to_csv(OUT / f"02_step2_loo_opponent_by_clip_{level}.csv", index=False)

    with open(OUT / f"02_step2_loo_opponent_summary_{level}.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    return overall


# -----------------------------------------------------------------------------
# Step 3: BT vs Context
# -----------------------------------------------------------------------------
def step3(df: pd.DataFrame) -> dict:
    rows = []
    for level in ("clip", "subcat"):
        for C in C_GRID:
            for model_name, inc_items, include_pairs in (
                ("MBT", True, False),
                ("MContext", True, True),
            ):
                overall, fold_df, _ = cv_by_pair(
                    df, level=level, include_items=inc_items, include_pairs=include_pairs, C=C
                )
                rows.append({
                    "level": level,
                    "model": model_name,
                    "C": C,
                    "scheme": "heldout_pair",
                    **{k: overall[k] for k in ("auc", "logloss", "balanced_acc", "n")},
                })
                fold_df.to_csv(OUT / f"03_folds_{level}_{model_name}_C{C}.csv", index=False)

    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / "03_step3_bt_vs_context.csv", index=False)

    deltas = []
    for level in ("clip", "subcat"):
        for C in C_GRID:
            mbt = cmp[(cmp.level == level) & (cmp.model == "MBT") & (cmp.C == C)].iloc[0]
            mcx = cmp[(cmp.level == level) & (cmp.model == "MContext") & (cmp.C == C)].iloc[0]
            deltas.append({
                "level": level,
                "C": C,
                "delta_auc": mcx["auc"] - mbt["auc"],
                "delta_logloss": mcx["logloss"] - mbt["logloss"],
                "delta_balanced_acc": mcx["balanced_acc"] - mbt["balanced_acc"],
                "mbt_auc": mbt["auc"],
                "mctx_auc": mcx["auc"],
            })
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(OUT / "03_step3_deltas.csv", index=False)

    # Train-set residual pairs (interpretive only): fit MBT on full data, look at pair-level residuals
    residual_tables = {}
    for level in ("clip",):
        items = sorted(set(df["left_id"].astype(str)) | set(df["right_id"].astype(str)))
        pairs = sorted(df["pair_key"].astype(str).unique())
        subj_list = sorted(df["subject"].astype(str).unique())
        X, y, names, meta, left, right, pkeys = build_design(
            df, level=level, include_items=True, include_pairs=False,
            item_list=items, pair_list=pairs, subj_list=subj_list,
        )
        clf = fit_logit(X, y, C=C_DEFAULT)
        p = predict_proba_pos(clf, X)
        tmp = df[["pair_key", "y_right"]].copy()
        tmp["p_bt"] = p
        tmp["resid"] = tmp["y_right"] - tmp["p_bt"]
        agg = tmp.groupby("pair_key", as_index=False).agg(
            n=("y_right", "count"),
            mean_y=("y_right", "mean"),
            mean_p=("p_bt", "mean"),
            mean_resid=("resid", "mean"),
        )
        agg["abs_resid"] = agg["mean_resid"].abs()
        agg = agg.sort_values("abs_resid", ascending=False)
        agg.to_csv(OUT / "03_train_pair_residuals_clip.csv", index=False)
        residual_tables["clip_top20"] = agg.head(20).to_dict(orient="records")

    out = {"compare": rows, "deltas": deltas, "train_residuals_top": residual_tables}
    with open(OUT / "03_step3_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# -----------------------------------------------------------------------------
# Fit full-data q
# -----------------------------------------------------------------------------
def fit_full_q(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "clip":
        items = sorted(set(df["left_id"].astype(str)) | set(df["right_id"].astype(str)))
    else:
        items = sorted(set(df["left_cat"].astype(str)) | set(df["right_cat"].astype(str)))
    pairs = sorted(df["pair_key"].astype(str).unique())
    subj_list = sorted(df["subject"].astype(str).unique())
    X, y, names, meta, *_ = build_design(
        df, level=level, include_items=True, include_pairs=False,
        item_list=items, pair_list=pairs, subj_list=subj_list,
    )
    clf = fit_logit(X, y, C=C_DEFAULT)
    coef = clf.coef_.ravel()
    qdf = extract_q_from_coef(names, coef, meta["item_ref"])
    qdf.to_csv(OUT / f"04_q_{level}.csv", index=False)
    return qdf


# -----------------------------------------------------------------------------
# Transitivity auxiliary
# -----------------------------------------------------------------------------
def empirical_win_rate(df: pd.DataFrame, level: str = "clip"):
    """Directed win counts: for unordered pair, aggregate choose-content rates."""
    if level == "clip":
        Lcol, Rcol = "left_id", "right_id"
    else:
        Lcol, Rcol = "left_cat", "right_cat"
    wins = defaultdict(float)
    tot = defaultdict(float)
    for L, R, y in zip(df[Lcol].astype(str), df[Rcol].astype(str), df["y_right"].astype(int)):
        # y=1 => right wins
        if y == 1:
            wins[(R, L)] += 1
        else:
            wins[(L, R)] += 1
        tot[(R, L)] += 1
        tot[(L, R)] += 1
    # P(A beats B) from trials where they meet
    rate = {}
    pairs_seen = set()
    left = df[Lcol].astype(str).values
    right = df[Rcol].astype(str).values
    for a, b in zip(left, right):
        key = tuple(sorted([a, b]))
        if key in pairs_seen or a == b:
            continue
        pairs_seen.add(key)
        a, b = key
        # count A wins when A on right or left
        mask_ar = (right == a) & (left == b)
        mask_al = (left == a) & (right == b)
        y = df["y_right"].astype(int).values
        n = int(mask_ar.sum() + mask_al.sum())
        if n == 0:
            continue
        a_wins = int(y[mask_ar].sum() + (1 - y[mask_al]).sum())
        rate[(a, b)] = a_wins / n
        rate[(b, a)] = 1.0 - rate[(a, b)]
    return rate


def transitivity_aux(df: pd.DataFrame, qdf: pd.DataFrame, level: str = "clip") -> dict:
    qmap = dict(zip(qdf["item"], qdf["q"]))
    rate = empirical_win_rate(df, level=level)

    # triples where all three directed edges have empirical rates
    items = list(qmap.keys())
    # only items that appear in rate graph
    neighbors = defaultdict(set)
    for (a, b) in rate:
        neighbors[a].add(b)

    cycles = 0
    transitive = 0
    consistent_with_q = 0
    total = 0
    # sample triples among items with degree>=2 for speed
    cand = [i for i in items if len(neighbors[i]) >= 2]
    # enumerate limited triples
    checked = 0
    max_triples = 5000
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            for k in range(j + 1, len(cand)):
                a, b, c = cand[i], cand[j], cand[k]
                edges = [(a, b), (b, c), (a, c)]
                if not all(e in rate for e in edges) and not all(e in rate for e in [(a, b), (b, c), (c, a)]):
                    # need all three pairwise comparisons
                    if (a, b) not in rate or (b, c) not in rate or (a, c) not in rate:
                        continue
                if (a, b) not in rate or (b, c) not in rate or (a, c) not in rate:
                    continue
                total += 1
                # empirical tournament orientation by majority
                ab = rate[(a, b)] >= 0.5
                bc = rate[(b, c)] >= 0.5
                ac = rate[(a, c)] >= 0.5
                # cycle if A>B, B>C, C>A or opposite
                cyc = (ab and bc and (not ac)) or ((not ab) and (not bc) and ac)
                if cyc:
                    cycles += 1
                else:
                    transitive += 1
                # q order prediction for A vs C given q
                q_ac = qmap[a] > qmap[c]
                if q_ac == ac:
                    consistent_with_q += 1
                checked += 1
                if checked >= max_triples:
                    break
            if checked >= max_triples:
                break
        if checked >= max_triples:
            break

    # Monte Carlo null: shuffle outcomes within each unordered pair, recompute cycle rate
    null_cycle_rates = []
    pair_groups = list(df.groupby("pair_key").indices.items())
    y0 = df["y_right"].astype(int).values.copy()
    for _ in range(N_TRANS_NULL):
        y_shuf = y0.copy()
        for _, idxs in pair_groups:
            idxs = np.asarray(list(idxs))
            y_shuf[idxs] = RNG.permutation(y_shuf[idxs])
        df_s = df.copy()
        df_s["y_right"] = y_shuf
        rate_s = empirical_win_rate(df_s, level=level)
        cyc_n = 0
        tot_n = 0
        checked = 0
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                for k in range(j + 1, len(cand)):
                    a, b, c = cand[i], cand[j], cand[k]
                    if (a, b) not in rate_s or (b, c) not in rate_s or (a, c) not in rate_s:
                        continue
                    ab = rate_s[(a, b)] >= 0.5
                    bc = rate_s[(b, c)] >= 0.5
                    ac = rate_s[(a, c)] >= 0.5
                    cyc = (ab and bc and (not ac)) or ((not ab) and (not bc) and ac)
                    tot_n += 1
                    cyc_n += int(cyc)
                    checked += 1
                    if checked >= 2000:
                        break
                if checked >= 2000:
                    break
            if checked >= 2000:
                break
        null_cycle_rates.append(cyc_n / tot_n if tot_n else np.nan)

    obs_cycle_rate = cycles / total if total else float("nan")
    null_mean = float(np.nanmean(null_cycle_rates)) if null_cycle_rates else float("nan")
    null_p = float(np.nanmean(np.array(null_cycle_rates) >= obs_cycle_rate)) if null_cycle_rates else float("nan")

    out = {
        "level": level,
        "n_triples": total,
        "n_cycles": cycles,
        "n_transitive": transitive,
        "cycle_rate": obs_cycle_rate,
        "q_edge_consistency": consistent_with_q / total if total else float("nan"),
        "null_cycle_rate_mean": null_mean,
        "null_cycle_rate_p_ge_obs": null_p,
        "note": "Transitivity is auxiliary; random cycles expected with finite trials.",
    }
    with open(OUT / "05_transitivity_aux.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# -----------------------------------------------------------------------------
# Architecture decision
# -----------------------------------------------------------------------------
def write_decision(step1_sum, step2_clip, step3_sum, trans, deg_clip) -> dict:
    # Primary evidence from clip-level held-out pairing + LOSO
    d1 = {(r["level"], r["scheme"]): r for r in step1_sum["deltas"]}
    delta_loso = d1.get(("clip", "loso"), {})
    delta_pair = d1.get(("clip", "heldout_pair"), {})

    bt_supported = bool(
        delta_loso.get("delta_auc", 0) > 0.01
        and delta_pair.get("delta_auc", 0) > 0.01
        and delta_loso.get("delta_logloss", 0) < 0
        and delta_pair.get("delta_logloss", 0) < 0
    )
    bt_supported_soft = bool(
        delta_loso.get("delta_auc", 0) > 0
        and delta_pair.get("delta_auc", 0) > 0
    )

    looo_ok = False
    if step2_clip.get("bt") and step2_clip.get("m0"):
        looo_ok = bool(
            step2_clip.get("delta_auc_bt_minus_m0", 0) > 0
            and step2_clip.get("delta_logloss_bt_minus_m0", 1) < 0
        )

    ctx_deltas = [d for d in step3_sum["deltas"] if d["level"] == "clip"]
    ctx_improve = bool(
        all(d["delta_auc"] > 0.005 for d in ctx_deltas)
        and all(d["delta_logloss"] < 0 for d in ctx_deltas)
    )
    ctx_mild = bool(any(d["delta_auc"] > 0.01 for d in ctx_deltas))

    if bt_supported_soft and not ctx_improve:
        architecture = "BT_only"
        formula = "x_L -> q_L, x_R -> q_R, S = q_R - q_L"
        rationale = (
            "Held-out comparisons show MBT outperforms side+subject bias alone, "
            "while pair-specific residuals do not stably improve held-out prediction."
        )
    elif ctx_improve or (ctx_mild and bt_supported_soft):
        architecture = "BT_plus_context"
        formula = "S = (q_R - q_L) + r(z_L, z_R)"
        rationale = (
            "Pair residuals improve held-out pairing prediction beyond intrinsic priority differences; "
            "an interaction module is justified by behavioral evidence."
        )
    else:
        architecture = "inconclusive_lean_BT"
        formula = "x_L -> q_L, x_R -> q_R, S = q_R - q_L"
        rationale = (
            "Evidence for opponent-invariant priority is mixed or weak; default to the simpler BT form "
            "unless stronger context gains appear."
        )

    multi_frac = float((deg_clip["n_opponents"] >= 2).mean()) if len(deg_clip) else float("nan")

    def _py(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py(v) for v in o]
        return o

    decision = _py({
        "architecture": architecture,
        "formula": formula,
        "rationale": rationale,
        "bt_supported_strict": bt_supported,
        "bt_supported_soft": bt_supported_soft,
        "looo_supports_invariant_q": looo_ok,
        "context_stable_improve": ctx_improve,
        "step1_delta_loso_auc": delta_loso.get("delta_auc"),
        "step1_delta_pair_auc": delta_pair.get("delta_auc"),
        "step2_delta_auc_bt_m0": step2_clip.get("delta_auc_bt_minus_m0"),
        "step3_clip_deltas": ctx_deltas,
        "clip_multi_opponent_fraction": multi_frac,
        "transitivity": trans,
    })

    paper_en = (
        "Behavioral analysis first tested whether spontaneous auditory selection could be explained "
        "by opponent-invariant stimulus priorities using a Bradley–Terry model. We then tested whether "
        "pair-specific contextual effects provided additional predictive information beyond the difference "
        "in intrinsic priorities."
    )

    md = f"""# Architecture decision (behavioral BT tests)

## Decision

- **Architecture**: `{architecture}`
- **Recommended score**: `{formula}`
- **Rationale**: {rationale}

## Evidence snapshot (clip-level)

| Check | Result |
|-------|--------|
| Step1 ΔAUC (LOSO, MBT−M0) | {delta_loso.get('delta_auc')} |
| Step1 ΔAUC (held-out pair, MBT−M0) | {delta_pair.get('delta_auc')} |
| Step2 LOOO ΔAUC (BT−M0) | {step2_clip.get('delta_auc_bt_minus_m0')} |
| Step3 context stable held-out gain | {ctx_improve} |
| Fraction of clips with ≥2 opponents | {multi_frac:.3f} |
| Empirical cycle rate (aux) | {trans.get('cycle_rate')} (null mean {trans.get('null_cycle_rate_mean')}) |

## Implication for neural models

"""
    if architecture == "BT_plus_context":
        md += (
            "Keep separate ear encoders producing `q_L`, `q_R`, but add an interaction residual "
            "`r(z_L, z_R)` (e.g. on existing residual/gated fusion paths). Do not replace BT difference "
            "with a purely pair-specific scorer.\n"
        )
    else:
        md += (
            "Prefer the clean Bradley–Terry / intrinsic-priority architecture: "
            "shared (or mirrored) encoders → scalar priorities → `S = q_R - q_L`. "
            "Do not add a cross-source interaction module unless future data show stable held-out gains.\n"
        )

    md += f"""
## Paper wording

{paper_en}

## Notes

- Priority unit: clip-level primary; subcategory reported as supplement in CSVs.
- Subject effects entered as fixed effects; side bias is the intercept under `y_right`.
- Pair residual `r_LR` is antisymmetric in item identity (not a raw ear-side dummy).
- Transitivity is auxiliary and not used alone to reject BT.
"""
    (OUT / "ARCHITECTURE_DECISION.md").write_text(md, encoding="utf-8")
    with open(OUT / "ARCHITECTURE_DECISION.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    return decision


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Loading", TRIAL_CSV)
    df = pd.read_csv(TRIAL_CSV)
    df["y_right"] = df["y_right"].astype(int)
    df["left_id"] = df["left_id"].astype(str)
    df["right_id"] = df["right_id"].astype(str)
    df["left_cat"] = df["left_cat"].astype(str)
    df["right_cat"] = df["right_cat"].astype(str)
    df["pair_key"] = df["pair_key"].astype(str)
    df["subject"] = df["subject"].astype(str)

    print(f"n_trials={len(df)} n_subj={df.subject.nunique()} n_pair={df.pair_key.nunique()}")

    deg_clip = opponent_degree(df, "clip")
    deg_clip.to_csv(OUT / "00_clip_opponent_degree.csv", index=False)
    deg_sub = opponent_degree(df, "subcat")
    deg_sub.to_csv(OUT / "00_subcat_opponent_degree.csv", index=False)
    print("Opponent degree: clips ≥2:", int((deg_clip.n_opponents >= 2).sum()), "/", len(deg_clip))

    print("=== Step1: M0 vs MBT ===")
    step1_sum = step1(df)

    print("=== Fit full-data q ===")
    q_clip = fit_full_q(df, "clip")
    q_sub = fit_full_q(df, "subcat")

    print("=== Step2: leave-one-opponent-out ===")
    step2_clip = step2_loo_opponent(df, level="clip")
    step2_sub = step2_loo_opponent(df, level="subcat")

    print("=== Step3: MBT vs MContext ===")
    step3_sum = step3(df)

    print("=== Transitivity aux ===")
    trans = transitivity_aux(df, q_clip, level="clip")

    print("=== Architecture decision ===")
    decision = write_decision(step1_sum, step2_clip, step3_sum, trans, deg_clip)

    run_summary = {
        "n_trials": int(len(df)),
        "n_subjects": int(df.subject.nunique()),
        "n_pair_keys": int(df.pair_key.nunique()),
        "n_clips": int(len(deg_clip)),
        "n_clips_multi_opponent": int((deg_clip.n_opponents >= 2).sum()),
        "step1_deltas": step1_sum["deltas"],
        "step2_clip": step2_clip,
        "step2_subcat": {k: step2_sub[k] for k in step2_sub if k != "bt"},
        "step3_deltas": step3_sum["deltas"],
        "transitivity": trans,
        "architecture": decision,
        "q_clip_top10": q_clip.head(10).to_dict(orient="records"),
        "q_subcat": q_sub.to_dict(orient="records"),
    }
    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print("Done. Outputs in", OUT)
    print("Architecture:", decision["architecture"], decision["formula"])


if __name__ == "__main__":
    main()
