#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from prime_ablation_ctrl import SUBJECTS_23

AUDIO = r"A:\standard_data_interp_ica_99\nme_fusion_results\individual\category_heldout_anat_prio_nomargin_seeded\n23_anat_prio_summary.csv"
EEG = r"A:\standard_data_interp_ica_99\nme_fusion_results\individual\category_heldout_eeg_only_dual_nomargin_seeded\n23_eeg_only_summary.csv"
JOINT = r"A:\standard_data_interp_ica_99\nme_fusion_results\individual\category_heldout_anat_scalar_detach_dual_nomargin_seeded\n23_scalar_detach_dual_summary.csv"
TWO = r"E:\saad_reproduce_twostage_sb_pairing_heldout\n23_twostage_summary.csv"
OUT = r"E:\saad_reproduce_twostage_sb_pairing_heldout\n23_audio_eeg_joint_twostage.csv"


def main():
    a = pd.read_csv(AUDIO).set_index("subject")["bacc_anat_prio"] * 100
    e = pd.read_csv(EEG).set_index("subject")["bacc_eeg_only"] * 100
    j = pd.read_csv(JOINT).set_index("subject")["bacc_fusion"] * 100
    t = pd.read_csv(TWO).set_index("subject")["bacc_twostage"] * 100
    s = pd.read_csv(TWO).set_index("subject")["bacc_frozen_s"] * 100
    rows = []
    for subj in SUBJECTS_23:
        rows.append(
            {
                "subject": subj,
                "Audio only": float(a.loc[subj]),
                "EEG only": float(e.loc[subj]),
                "Joint fusion": float(j.loc[subj]),
                "Two-stage S+B": float(t.loc[subj]),
                "Joint - Audio": float(j.loc[subj] - a.loc[subj]),
                "S+B - Audio": float(t.loc[subj] - a.loc[subj]),
                "S+B - Joint": float(t.loc[subj] - j.loc[subj]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig", float_format="%.4f")
    n = len(df)
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print()
    print(f"mean Audio only:    {df['Audio only'].mean():.2f}%")
    print(f"mean EEG only:      {df['EEG only'].mean():.2f}%")
    print(f"mean Joint fusion:  {df['Joint fusion'].mean():.2f}%")
    print(f"mean Two-stage S+B: {df['Two-stage S+B'].mean():.2f}%")
    print(f"Joint > Audio: {(df['Joint fusion'] > df['Audio only']).sum()}/{n}")
    print(f"S+B > Audio:   {(df['Two-stage S+B'] > df['Audio only']).sum()}/{n}")
    print(f"S+B > Joint:   {(df['Two-stage S+B'] > df['Joint fusion']).sum()}/{n}")
    print(f"max |frozen S - Audio|: {(s - a).abs().max():.6f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
