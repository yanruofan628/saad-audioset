#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

OUT = Path(__file__).resolve().parent / "bt_intrinsic_priority" / "lopo"
RNG = np.random.default_rng(0)
N_BOOT = 2000


def bacc(y, p):
    return float(balanced_accuracy_score(y, (np.asarray(p) >= 0.5).astype(int)))


def auc(y, p):
    return float(roc_auc_score(y, p))


def boot_pair(pair_ids, y, p0, pbt, n_boot=N_BOOT):
    by = defaultdict(list)
    for i, pk in enumerate(pair_ids):
        by[pk].append(i)
    uniq = np.array(list(by.keys()))
    n = len(uniq)
    d_auc, d_bacc, a0, ab, b0, bb = [], [], [], [], [], []
    for _ in range(n_boot):
        samp = RNG.choice(uniq, size=n, replace=True)
        idx = np.concatenate([by[pk] for pk in samp])
        ys, x0, xb = y[idx], p0[idx], pbt[idx]
        if len(np.unique(ys)) < 2:
            continue
        m0a, mba = auc(ys, x0), auc(ys, xb)
        m0b, mbb = bacc(ys, x0), bacc(ys, xb)
        a0.append(m0a)
        ab.append(mba)
        d_auc.append(mba - m0a)
        b0.append(m0b)
        bb.append(mbb)
        d_bacc.append(mbb - m0b)

    def ci(a):
        a = np.asarray(a, float)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "auc_M0_ci": ci(a0),
        "auc_MBT_ci": ci(ab),
        "delta_auc_ci": ci(d_auc),
        "bacc_M0_ci": ci(b0),
        "bacc_MBT_ci": ci(bb),
        "delta_bacc_ci": ci(d_bacc),
    }


rows = []
for proto in ["general", "strict"]:
    df = pd.read_csv(OUT / f"lopo_oof_trials_{proto}.csv")
    y = df.y_right.to_numpy()
    p0 = df.p_M0.to_numpy()
    pbt = df.p_MBT.to_numpy()
    pk = df.held_out_pair.to_numpy()
    m = {
        "protocol": proto,
        "n_pairs": int(df.held_out_pair.nunique()),
        "n_trials": int(len(df)),
        "auc_M0": auc(y, p0),
        "auc_MBT": auc(y, pbt),
        "delta_auc": auc(y, pbt) - auc(y, p0),
        "bacc_M0": bacc(y, p0),
        "bacc_MBT": bacc(y, pbt),
        "delta_bacc": bacc(y, pbt) - bacc(y, p0),
    }
    m.update(boot_pair(pk, y, p0, pbt))
    rows.append(m)

pd.DataFrame(rows).to_csv(OUT / "lopo_auc_bacc_compare.csv", index=False)

lines = [
    "# LOPO: AUC vs BACC",
    "",
    "| Protocol | Pairs | AUC M0 | AUC MBT | ΔAUC [95% CI] | BACC M0 | BACC MBT | ΔBACC [95% CI] |",
    "|---|---:|---:|---:|---|---:|---:|---|",
]
for m in rows:
    lab = "LOPO-General (d≥1)" if m["protocol"] == "general" else "LOPO-Strict (d≥2)"
    da = m["delta_auc_ci"]
    db = m["delta_bacc_ci"]
    lines.append(
        f"| {lab} | {m['n_pairs']} | {m['auc_M0']:.3f} | {m['auc_MBT']:.3f} | "
        f"{m['delta_auc']:+.3f} [{da[0]:.3f}, {da[1]:.3f}] | "
        f"{m['bacc_M0']:.3f} | {m['bacc_MBT']:.3f} | "
        f"{m['delta_bacc']:+.3f} [{db[0]:.3f}, {db[1]:.3f}] |"
    )

text = "\n".join(lines)
(OUT / "lopo_auc_bacc_compare.md").write_text(text, encoding="utf-8")
print(text)
