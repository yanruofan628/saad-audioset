#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print n23 summary CSVs for held-out ablations."""
from __future__ import annotations

import os
import sys

import pandas as pd

from eeg_audio_residual import OUTPUT_DIR

ROOT = os.path.join(OUTPUT_DIR, "individual")


def _read(tag_dir, pattern):
    d = os.path.join(ROOT, tag_dir)
    if not os.path.isdir(d):
        return None, f"missing dir: {d}"
    hits = [f for f in os.listdir(d) if f.endswith("_summary.csv") and pattern in f]
    if not hits:
        return None, f"no summary in {d}"
    hits.sort(key=lambda x: os.path.getmtime(os.path.join(d, x)), reverse=True)
    path = os.path.join(d, hits[0])
    return pd.read_csv(path), path


def _mean(df, col):
    if df is None or col not in df.columns:
        return None
    return float(df[col].mean())


def _print_df(title, df, path, cols):
    print("=" * 72)
    print(title)
    if df is None:
        print("  ", path)
        return
    print("  file:", path)
    print("  n =", len(df))
    show = [c for c in cols if c in df.columns]
    if show:
        print(df[["subject"] + show].to_string(index=False) if "subject" in df.columns else df[show].to_string(index=False))
        print("-- group means --")
        for c in show:
            print(f"  {c}: {_mean(df, c):.4f}")
    else:
        print(df.to_string(index=False))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    jobs = argv or [
        "twostage:category_heldout_twostage_freeze_nomargin_seeded_shrink:twostage",
        "joint:category_heldout_anat_scalar_detach_dual_nomargin_seeded_shrink:scalar",
        "audio:category_heldout_anat_prio_nomargin_seeded:anat_prio",
        "twostage_old:category_heldout_twostage_freeze_nomargin_seeded:twostage",
        "joint_old:category_heldout_anat_scalar_detach_dual_nomargin_seeded:scalar",
    ]
    for spec in jobs:
        parts = spec.split(":")
        title, tag_dir, pat = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
        df, path = _read(tag_dir, pat)
        cols = [
            "bacc_audio",
            "bacc_frozen_s",
            "bacc_twostage",
            "bacc_twostage_shrink",
            "bacc_fusion",
            "bacc_fusion_shrink",
            "lambda_star_mean",
            "delta_d_minus_audio",
            "delta_shrink_minus_audio",
            "gate_mean",
        ]
        _print_df(title, df, path, cols)


if __name__ == "__main__":
    main()
