#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare Audio vs two-stage S+B at Stage-2 lr in {1e-3, 3e-4, 1e-4}."""
from __future__ import annotations

import os

import pandas as pd

from prime_ablation_ctrl import SUBJECTS_23

ROOT_1E3 = r"E:\saad_reproduce_twostage_sb_pairing_heldout"
ROOT_3E4 = r"E:\saad_twostage_sb_s2lr3e-4_pairing_heldout"
ROOT_1E4 = r"E:\saad_twostage_sb_s2lr1e-4_pairing_heldout"
OUT_DIR = r"E:\saad_twostage_sb_pairing_heldout_lr_compare"


def _load(root: str) -> pd.DataFrame:
    path = os.path.join(root, "n23_twostage_summary.csv")
    df = pd.read_csv(path)
    df["subject"] = df["subject"].astype(str)
    return df.set_index("subject")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = _load(ROOT_1E3)
    b = _load(ROOT_3E4)
    c = _load(ROOT_1E4)
    rows = []
    for subj in SUBJECTS_23:
        audio = float(a.loc[subj, "bacc_audio"]) * 100.0
        d1e3 = float(a.loc[subj, "bacc_twostage"]) * 100.0
        d3e4 = float(b.loc[subj, "bacc_twostage"]) * 100.0
        d1e4 = float(c.loc[subj, "bacc_twostage"]) * 100.0
        rows.append(
            {
                "subject": subj,
                "Audio": audio,
                "lr=1e-3": d1e3,
                "lr=3e-4": d3e4,
                "lr=1e-4": d1e4,
                "1e-3 - Audio": d1e3 - audio,
                "3e-4 - Audio": d3e4 - audio,
                "1e-4 - Audio": d1e4 - audio,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "n23_lr_compare.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    n = len(df)
    lines = [
        "pairing-held-out two-stage S+B  Stage-2 LR compare",
        "fusion = S+B, no λ, no gate; only lr changed",
        f"n = {n}",
        "",
        df[["subject", "Audio", "lr=1e-3", "lr=3e-4", "lr=1e-4"]].to_string(
            index=False, float_format=lambda x: f"{x:7.2f}"
        ),
        "",
        f"mean Audio:   {df['Audio'].mean():.2f}%",
        f"mean lr=1e-3: {df['lr=1e-3'].mean():.2f}%   "
        f"gain {df['1e-3 - Audio'].mean():+.2f} pp   "
        f">Audio {(df['lr=1e-3'] > df['Audio']).sum()}/{n}",
        f"mean lr=3e-4: {df['lr=3e-4'].mean():.2f}%   "
        f"gain {df['3e-4 - Audio'].mean():+.2f} pp   "
        f">Audio {(df['lr=3e-4'] > df['Audio']).sum()}/{n}",
        f"mean lr=1e-4: {df['lr=1e-4'].mean():.2f}%   "
        f"gain {df['1e-4 - Audio'].mean():+.2f} pp   "
        f">Audio {(df['lr=1e-4'] > df['Audio']).sum()}/{n}",
        "",
        f"1e-3 dir: {ROOT_1E3}",
        f"3e-4 dir: {ROOT_3E4}",
        f"1e-4 dir: {ROOT_1E4}",
        f"table:    {csv_path}",
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
