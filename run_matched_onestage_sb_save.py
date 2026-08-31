#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Matched one-stage S+B: D = sg(S) + B, no gate, no λ, EEG-only B.

Architecture matches FrozenAudioEEGResidual(use_gate=False).
Audio is trained by CE(S); CE(D) does not update S. Ear is not frozen.
Saves the same artifacts as run_twostage_sb_reproduce_save.py.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eeg_audio_residual import (
    SUBJECTS_CONFIG,
    SAMPLING_RATE,
    load_all_mappings,
    load_pair_stereo_paths,
    align_single_subject_data,
    _select_audio_for_trials,
    _build_single_subject_fold_data,
)
from am_fm_salience import (
    build_am_fm_cache,
    align_am_fm_to_pair_names,
    batch_am_fm_from_cache,
    AM_FM_CACHE_TAG,
)
from anatomical_acoustic_priority import MatchedJointStopgradSB
from linear_regression_loudness_models import load_stereo_2s, parse_pair_name
from prime_ablation_ctrl import (
    SUBJECTS_23,
    ABLATION_SEED,
    fold_seed,
    seed_everything,
    epoch_perm,
    seed_batch,
)
from run_eeg_only_category_heldout_23 import (
    build_global_pairing_type_fold_map,
    build_category_heldout_folds,
    _fix01,
)

_env_subjects = os.environ.get("SUBJECTS", "").strip()
SUBJECTS = (
    [s.strip() for s in _env_subjects.split(",") if s.strip()]
    if _env_subjects
    else list(SUBJECTS_23)
)

N_FOLDS = 6
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
EEG_MODE = "multi_scale_dual"
PRIO_W = 0.0
LAMBDA_FUSION = 1.0

OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT", r"E:\saad_matched_onestage_sb_pairing_heldout"
)
EEG_ONLY_SUMMARY = os.environ.get(
    "EEG_ONLY_SUMMARY",
    r"A:\standard_data_interp_ica_99\nme_fusion_results\individual"
    r"\category_heldout_eeg_only_dual_nomargin_seeded\n23_eeg_only_summary.csv",
)
AUDIO_SUMMARY = os.environ.get(
    "AUDIO_SUMMARY",
    r"A:\standard_data_interp_ica_99\nme_fusion_results\individual"
    r"\category_heldout_anat_prio_nomargin_seeded\n23_anat_prio_summary.csv",
)
TWOSTAGE_SUMMARY = os.environ.get(
    "TWOSTAGE_SUMMARY",
    r"E:\saad_reproduce_twostage_sb_pairing_heldout\n23_twostage_summary.csv",
)

CKPT_DIR = os.path.join(OUTPUT_ROOT, "ckpts")
LOGIT_DIR = os.path.join(OUTPUT_ROOT, "logits")
CURVE_DIR = os.path.join(OUTPUT_ROOT, "curves")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")

_max_folds = os.environ.get("MAX_FOLDS", "").strip()
MAX_FOLDS = int(_max_folds) if _max_folds.isdigit() else None

HPARAMS = {
    "split": "pairing-held-out 6-fold",
    "global_seed": int(ABLATION_SEED),
    "fold_seed_formula": "42 + fold * 1009",
    "init_seed": "fold_seed (no +777)",
    "audio_model": "AnatomicalEarEncoder",
    "mel_cache": AM_FM_CACHE_TAG,
    "PRIO_W": PRIO_W,
    "eeg_extractor": EEG_MODE,
    "eeg_head": "32-64-2 ReLU dropout=0.2",
    "B": "EEG only",
    "optimizer": "Adam",
    "lr": LR,
    "weight_decay": WD,
    "batch_size": BATCH_SIZE,
    "max_epochs": N_EPOCHS,
    "patience": PATIENCE,
    "early_stopping": "val_acc of D",
    "loss": "CE(S) + CE(D)",
    "fusion": "D = sg(S) + B",
    "use_gate": False,
    "lambda": None,
    "ear_frozen": False,
}


def _pair_key(pair_name: str) -> str:
    parsed = parse_pair_name(pair_name)
    if not parsed or parsed[0] is None:
        return str(pair_name)
    lc, li, rc, ri, _ = parsed
    a = f"{lc}|{li}"
    b = f"{rc}|{ri}"
    return "||".join(sorted([a, b]))


def _trial_pair_name(trial) -> str:
    if not isinstance(trial, dict):
        return str(trial)
    return str(trial.get("pair_name") or trial.get("original_name") or "")


def _eval_logits(logits, y):
    pred = logits.argmax(1).cpu().numpy()
    prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
    }


def _infer_split(
    model,
    eeg,
    labels,
    pair_indices,
    trials,
    am_fm,
    device,
    subject,
    fold_idx,
    split,
):
    model.eval()
    n = len(labels)
    rows = []
    max_abs_resid = 0.0
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            sl = slice(start, min(start + BATCH_SIZE, n))
            Xe = torch.FloatTensor(eeg[sl]).unsqueeze(1).to(device)
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(
                am_fm, pair_indices[sl], device
            )
            D, S, B, _g = model(
                Xe, mel_l, mel_r, s_l, s_r, train_detach=False, return_parts=True
            )
            S_np = S.detach().cpu().numpy()
            B_np = B.detach().cpu().numpy()
            D_np = D.detach().cpu().numpy()
            p_s = F.softmax(S, dim=1)[:, 1].detach().cpu().numpy()
            p_b = F.softmax(B, dim=1)[:, 1].detach().cpu().numpy()
            p_d = F.softmax(D, dim=1)[:, 1].detach().cpu().numpy()
            y_np = np.asarray(labels[sl], dtype=np.int64)
            batch_trials = trials[sl]
            for j in range(len(y_np)):
                s_m = float(S_np[j, 1] - S_np[j, 0])
                b_m = float(B_np[j, 1] - B_np[j, 0])
                d_m = float(D_np[j, 1] - D_np[j, 0])
                resid = float(d_m - (s_m + b_m))
                max_abs_resid = max(max_abs_resid, abs(resid))
                trial = batch_trials[j]
                pn = _trial_pair_name(trial)
                rows.append(
                    {
                        "subject": subject,
                        "fold": int(fold_idx),
                        "split": split,
                        "pair_key": _pair_key(pn),
                        "pair_name": pn,
                        "label": int(y_np[j]),
                        "audio_logit_0": float(S_np[j, 0]),
                        "audio_logit_1": float(S_np[j, 1]),
                        "eeg_logit_0": float(B_np[j, 0]),
                        "eeg_logit_1": float(B_np[j, 1]),
                        "fusion_logit_0": float(D_np[j, 0]),
                        "fusion_logit_1": float(D_np[j, 1]),
                        "S_margin": s_m,
                        "B_margin": b_m,
                        "D_margin": d_m,
                        "D_minus_S_minus_B": resid,
                        "p_audio": float(p_s[j]),
                        "p_eeg": float(p_b[j]),
                        "p_fusion": float(p_d[j]),
                    }
                )
    return rows, max_abs_resid


def train_fold(
    fold_data,
    train_trials,
    val_trials,
    am_fm,
    device,
    fold_idx: int,
    subject: str,
    out_paths: dict,
):
    train_eeg = fold_data["train_eeg"]
    train_y = _fix01(fold_data["train_labels"])
    val_eeg = fold_data["val_eeg"]
    val_y = _fix01(fold_data["val_labels"])
    tr_pi = fold_data["train_pair_indices"]
    va_pi = fold_data["val_pair_indices"]

    init_seed = fold_seed(fold_idx)
    seed_everything(init_seed)
    model = MatchedJointStopgradSB(
        eeg_channels=train_eeg.shape[1],
        eeg_samples=train_eeg.shape[2],
        sfreq=SAMPLING_RATE,
        eeg_mode=EEG_MODE,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()

    X_val = torch.FloatTensor(val_eeg).unsqueeze(1).to(device)
    y_val_t = torch.LongTensor(val_y).to(device)
    _, _, mel_l_v, mel_r_v, s_l_v, s_r_v = batch_am_fm_from_cache(am_fm, va_pi, device)
    n_train = len(train_y)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_acc, best_state, best_opt, best_epoch, wait = 0.0, None, None, 0, 0
    curve_rows = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = epoch_perm(n_train, fold_idx, epoch)
        run_loss = run_loss_s = run_loss_d = 0.0
        n_seen = 0
        tr_pred = []
        tr_true = []
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            seed_batch(fold_idx, epoch, b)
            Xe = torch.FloatTensor(train_eeg[idx]).unsqueeze(1).to(device)
            _, _, mel_l, mel_r, s_l, s_r = batch_am_fm_from_cache(
                am_fm, tr_pi[idx], device
            )
            yb = torch.LongTensor(train_y[idx]).to(device)
            opt.zero_grad()
            D, S, B, _g = model(
                Xe, mel_l, mel_r, s_l, s_r, train_detach=True, return_parts=True
            )
            loss_s = crit(S, yb)
            loss_d = crit(D, yb)
            loss = loss_s + LAMBDA_FUSION * loss_d
            loss.backward()
            opt.step()
            bs = int(len(idx))
            run_loss += float(loss.item()) * bs
            run_loss_s += float(loss_s.item()) * bs
            run_loss_d += float(loss_d.item()) * bs
            n_seen += bs
            tr_pred.append(D.detach().argmax(1).cpu().numpy())
            tr_true.append(train_y[idx])

        train_loss = run_loss / max(n_seen, 1)
        tr_pred_np = np.concatenate(tr_pred)
        tr_true_np = np.concatenate(tr_true)
        train_acc = float(accuracy_score(tr_true_np, tr_pred_np))
        train_bacc = float(balanced_accuracy_score(tr_true_np, tr_pred_np))

        model.eval()
        with torch.no_grad():
            D, S, B, _g = model(
                X_val, mel_l_v, mel_r_v, s_l_v, s_r_v,
                train_detach=False, return_parts=True,
            )
            pred = D.argmax(1).cpu().numpy()
            val_loss_s = float(crit(S, y_val_t).item())
            val_loss_d = float(crit(D, y_val_t).item())
            val_loss = val_loss_s + LAMBDA_FUSION * val_loss_d
        acc = accuracy_score(val_y, pred)
        val_bacc = float(balanced_accuracy_score(val_y, pred))
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_opt = copy.deepcopy(opt.state_dict())
            wait = 0
        else:
            wait += 1

        curve_rows.append(
            {
                "subject": subject,
                "fold": int(fold_idx),
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_S": run_loss_s / max(n_seen, 1),
                "train_loss_D": run_loss_d / max(n_seen, 1),
                "val_loss_S": val_loss_s,
                "val_loss_D": val_loss_d,
                "train_acc": train_acc,
                "val_acc": float(acc),
                "train_BACC": train_bacc,
                "val_BACC": val_bacc,
                "is_best": bool(is_best),
                "wait": int(wait),
            }
        )
        print(
            f"    epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={acc:.4f}  val_BACC={val_bacc:.4f}"
            f"{'  *best*' if is_best else ''}",
            flush=True,
        )
        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ext_state = {
        k[len("eeg_extractor.") :]: v
        for k, v in (best_state or {}).items()
        if k.startswith("eeg_extractor.")
    }
    head_state = {
        k[len("residual_head.") :]: v
        for k, v in (best_state or {}).items()
        if k.startswith("residual_head.")
    }
    ear_state = {
        k[len("ear.") :]: v
        for k, v in (best_state or {}).items()
        if k.startswith("ear.")
    }
    ckpt = {
        "ear": ear_state,
        "eeg_extractor": ext_state,
        "eeg_head": head_state,
        "full_state": best_state,
        "optimizer": best_opt,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_acc),
        "subject": subject,
        "fold": int(fold_idx),
        "seed": int(init_seed),
        "fold_seed": int(init_seed),
        "hparams": HPARAMS,
    }
    torch.save(ckpt, out_paths["ckpt"])
    pd.DataFrame(curve_rows).to_csv(out_paths["curve"], index=False, encoding="utf-8-sig")

    test_rows, resid_te = _infer_split(
        model, val_eeg, val_y, va_pi, val_trials, am_fm, device, subject, fold_idx, "test"
    )
    val_rows, resid_va = _infer_split(
        model, train_eeg, train_y, tr_pi, train_trials, am_fm, device, subject, fold_idx, "val"
    )
    pd.DataFrame(val_rows + test_rows).to_csv(
        out_paths["logits"], index=False, encoding="utf-8-sig"
    )
    max_resid = max(resid_te, resid_va)
    print(f"  D≈S+B check  max |resid|={max_resid:.3e}", flush=True)

    with torch.no_grad():
        D, S, B, _g = model(
            X_val, mel_l_v, mel_r_v, s_l_v, s_r_v,
            train_detach=False, return_parts=True,
        )
    m_s = _eval_logits(S, val_y)
    m_b = _eval_logits(B, val_y)
    m_d = _eval_logits(D, val_y)
    fold_summary = {
        "subject": subject,
        "fold": int(fold_idx),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_acc),
        "n_train": int(n_train),
        "n_test": int(len(val_y)),
        "Audio_BACC": m_s["bacc"],
        "EEG_BACC": m_b["bacc"],
        "Fusion_BACC": m_d["bacc"],
        "Fusion_minus_Audio": m_d["bacc"] - m_s["bacc"],
        "Audio_acc": m_s["acc"],
        "EEG_acc": m_b["acc"],
        "Fusion_acc": m_d["acc"],
        "D_minus_S_minus_B_maxabs": float(max_resid),
        "stopped_epoch": int(curve_rows[-1]["epoch"]) if curve_rows else 0,
    }
    return fold_summary, {
        "val_balanced_accuracy_audio_head": m_s["bacc"],
        "val_balanced_accuracy_eeg_B": m_b["bacc"],
        "val_balanced_accuracy_fusion": m_d["bacc"],
        "val_accuracy_fusion": m_d["acc"],
        "val_auc_fusion": m_d["auc"],
        "best_epoch": int(best_epoch),
    }


def _fold_paths(subject, fold_idx):
    return {
        "ckpt": os.path.join(CKPT_DIR, f"{subject}_fold{fold_idx}_best.pt"),
        "logits": os.path.join(LOGIT_DIR, f"{subject}_fold{fold_idx}_trials.csv"),
        "curve": os.path.join(CURVE_DIR, f"{subject}_fold{fold_idx}_curves.csv"),
        "fold_json": os.path.join(LOG_DIR, f"{subject}_fold{fold_idx}_summary.json"),
    }


def run_subject(subject_name, am_fm, pair_name_to_idx, type_map, device):
    print(f"\n{'=' * 60}\n被试: {subject_name}\n{'=' * 60}", flush=True)
    consistent_eeg, consistent_labels, consistent_trials, meta, _ = align_single_subject_data(
        subject_name, filter_consistent=False, project_root=project_root
    )
    n_pairs = len(pair_name_to_idx)
    dummy_audio = np.zeros((n_pairs, 2), dtype=np.float32)
    consistent_eeg, consistent_labels, consistent_trials, pair_indices = _select_audio_for_trials(
        consistent_eeg, consistent_labels, consistent_trials, dummy_audio, pair_name_to_idx
    )
    folds, shuffled_indices = build_category_heldout_folds(consistent_trials, type_map)

    fold_rows = []
    fold_summaries = []
    for fold_idx, val_idx in enumerate(folds, 1):
        if MAX_FOLDS is not None and fold_idx > MAX_FOLDS:
            break
        if len(val_idx) == 0:
            print(f"  Fold {fold_idx}: empty val, skip", flush=True)
            continue
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=False)
        paths = _fold_paths(subject_name, fold_idx)
        print(
            f"\n--- Fold {fold_idx}/{N_FOLDS} train={len(train_idx)} val/test={len(val_idx)} ---",
            flush=True,
        )
        fold_data = _build_single_subject_fold_data(
            consistent_eeg, consistent_labels, pair_indices, train_idx, val_idx
        )
        train_trials = [consistent_trials[int(i)] for i in train_idx]
        val_trials = [consistent_trials[int(i)] for i in val_idx]
        fold_sum, m_res = train_fold(
            fold_data, train_trials, val_trials, am_fm, device, fold_idx, subject_name, paths
        )
        with open(paths["fold_json"], "w", encoding="utf-8") as f:
            json.dump(fold_sum, f, indent=2, ensure_ascii=False)
        fold_summaries.append(fold_sum)
        fold_rows.append(
            {
                "subject": subject_name,
                "cv_scheme": "category_heldout",
                "fold_idx": fold_idx,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                **m_res,
            }
        )
        print(
            f"  Fold {fold_idx}: best_epoch={fold_sum['best_epoch']}  "
            f"S={fold_sum['Audio_BACC']:.4f}  B={fold_sum['EEG_BACC']:.4f}  "
            f"D={fold_sum['Fusion_BACC']:.4f}  "
            f"D-S={fold_sum['Fusion_minus_Audio']:+.4f}",
            flush=True,
        )

    df = pd.DataFrame(fold_rows)
    df.to_csv(
        os.path.join(LOG_DIR, f"{subject_name}_onestage_cv_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    b_s = float(df["val_balanced_accuracy_audio_head"].mean()) if len(df) else float("nan")
    b_d = float(df["val_balanced_accuracy_fusion"].mean()) if len(df) else float("nan")
    b_b = float(df["val_balanced_accuracy_eeg_B"].mean()) if len(df) else float("nan")
    print(
        f"单人 matched one-stage - {subject_name}\n"
        f"S (this model) BACC: {b_s:.4f}\n"
        f"B BACC: {b_b:.4f}\n"
        f"D=sg(S)+B BACC: {b_d:.4f}\n"
        f"delta (D-S): {b_d - b_s:+.4f}\n",
        flush=True,
    )
    return {
        "subject": subject_name,
        "bacc_audio_head": b_s,
        "bacc_eeg_B": b_b,
        "bacc_fusion": b_d,
        "delta_d_minus_s": b_d - b_s,
    }, fold_summaries


def _write_final_tables(rows, all_fold_summaries):
    fold_df = pd.DataFrame(all_fold_summaries)
    fold_path = os.path.join(OUTPUT_ROOT, "fold_summaries.csv")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(
        os.path.join(OUTPUT_ROOT, "n23_onestage_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    audio = {}
    eeg_only = {}
    two = {}
    if os.path.isfile(AUDIO_SUMMARY):
        adf = pd.read_csv(AUDIO_SUMMARY)
        audio = dict(zip(adf["subject"].astype(str), adf["bacc_anat_prio"]))
    if os.path.isfile(EEG_ONLY_SUMMARY):
        edf = pd.read_csv(EEG_ONLY_SUMMARY)
        eeg_only = dict(zip(edf["subject"].astype(str), edf["bacc_eeg_only"]))
    if os.path.isfile(TWOSTAGE_SUMMARY):
        tdf_in = pd.read_csv(TWOSTAGE_SUMMARY)
        two = dict(zip(tdf_in["subject"].astype(str), tdf_in["bacc_twostage"]))

    table_rows = []
    for r in rows:
        subj = r["subject"]
        a = float(audio[subj]) if subj in audio else float("nan")
        e = float(eeg_only[subj]) if subj in eeg_only else float("nan")
        fusion = float(r["bacc_fusion"])
        tw = float(two[subj]) if subj in two else float("nan")
        table_rows.append(
            {
                "subject": subj,
                "Audio only": a * 100.0,
                "EEG only": e * 100.0,
                "matched one-stage": fusion * 100.0,
                "two-stage S+B": tw * 100.0,
                "one-stage - Audio": (fusion - a) * 100.0,
                "S+B - Audio": (tw - a) * 100.0,
                "S+B - one-stage": (tw - fusion) * 100.0,
                "one-stage S": float(r["bacc_audio_head"]) * 100.0,
            }
        )
    tdf = pd.DataFrame(table_rows)
    table_path = os.path.join(OUTPUT_ROOT, "n23_subject_table.csv")
    tdf.to_csv(table_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    logit_files = sorted(
        os.path.join(LOGIT_DIR, f)
        for f in os.listdir(LOGIT_DIR)
        if f.endswith("_trials.csv")
    )
    if logit_files:
        pd.concat([pd.read_csv(p) for p in logit_files], ignore_index=True).to_csv(
            os.path.join(LOGIT_DIR, "all_trials_val_test.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    curve_files = sorted(
        os.path.join(CURVE_DIR, f)
        for f in os.listdir(CURVE_DIR)
        if f.endswith("_curves.csv")
    )
    if curve_files:
        pd.concat([pd.read_csv(p) for p in curve_files], ignore_index=True).to_csv(
            os.path.join(CURVE_DIR, "all_curves.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    n = len(tdf)
    mean_audio = float(tdf["Audio only"].mean()) if n else float("nan")
    mean_eeg = float(tdf["EEG only"].mean()) if n else float("nan")
    mean_one = float(tdf["matched one-stage"].mean()) if n else float("nan")
    mean_two = float(tdf["two-stage S+B"].mean()) if n else float("nan")
    mean_s = float(tdf["one-stage S"].mean()) if n else float("nan")
    n_one_up = int((tdf["matched one-stage"] > tdf["Audio only"]).sum()) if n else 0
    n_two_up = int((tdf["two-stage S+B"] > tdf["Audio only"]).sum()) if n else 0
    n_two_gt_one = int((tdf["two-stage S+B"] > tdf["matched one-stage"]).sum()) if n else 0

    show = tdf[
        ["subject", "Audio only", "EEG only", "matched one-stage", "two-stage S+B"]
    ]
    report = [
        "matched one-stage vs two-stage S+B  (D = sg(S)+B vs freeze then S+B)",
        f"n = {n}",
        "",
        show.to_string(index=False, float_format=lambda x: f"{x:7.2f}"),
        "",
        f"mean Audio only:         {mean_audio:.2f}%",
        f"mean EEG only:           {mean_eeg:.2f}%",
        f"mean one-stage S (head): {mean_s:.2f}%",
        f"mean matched one-stage:  {mean_one:.2f}%   vs Audio {mean_one - mean_audio:+.2f} pp   >Audio {n_one_up}/{n}",
        f"mean two-stage S+B:      {mean_two:.2f}%   vs Audio {mean_two - mean_audio:+.2f} pp   >Audio {n_two_up}/{n}",
        f"two-stage > one-stage:   {n_two_gt_one}/{n}",
        "",
        f"fold summaries: {fold_path}",
        f"subject table:  {table_path}",
        f"trial logits:   {os.path.join(LOGIT_DIR, 'all_trials_val_test.csv')}",
        f"curves:         {os.path.join(CURVE_DIR, 'all_curves.csv')}",
    ]
    text = "\n".join(report) + "\n"
    with open(os.path.join(OUTPUT_ROOT, "REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text, flush=True)
    return tdf


def main():
    for d in (OUTPUT_ROOT, CKPT_DIR, LOGIT_DIR, CURVE_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)

    missing = [s for s in SUBJECTS if s not in SUBJECTS_CONFIG]
    if missing:
        raise SystemExit(f"SUBJECTS_CONFIG 缺少: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    print("subjects:", SUBJECTS, flush=True)
    print("OUTPUT_ROOT:", OUTPUT_ROOT, flush=True)
    print("EEG_MODE:", EEG_MODE, flush=True)
    print("LR:", LR, "WD:", WD, flush=True)
    print("fusion: D = sg(S) + B; loss = CE(S)+CE(D); no gate, no λ", flush=True)
    print("hparams:", json.dumps(HPARAMS, ensure_ascii=False), flush=True)

    mappings, _ = load_all_mappings()
    pair_names = sorted({v for m in mappings for v in m.values()})
    stereo_path_map = load_pair_stereo_paths()
    pair_name_to_idx = {pn: i for i, pn in enumerate(pair_names)}

    cache_dir = os.path.join(project_root, "audio_features_cache")
    am_fm_raw = build_am_fm_cache(
        pair_names, stereo_path_map, load_stereo_2s, cache_dir, cache_tag=AM_FM_CACHE_TAG
    )
    am_fm = align_am_fm_to_pair_names(am_fm_raw, pair_names)
    type_map, _ = build_global_pairing_type_fold_map(pair_names)

    rows = []
    all_fold_summaries = []
    for subj in SUBJECTS:
        try:
            row, fold_sums = run_subject(
                subj, am_fm, pair_name_to_idx, type_map, device
            )
            rows.append(row)
            all_fold_summaries.extend(fold_sums)
            _write_final_tables(rows, all_fold_summaries)
        except Exception as e:
            print(f"\n[FAIL] {subj}: {e}", flush=True)
            traceback.print_exc()

    if rows:
        _write_final_tables(rows, all_fold_summaries)
        done_path = os.path.join(OUTPUT_ROOT, "DONE.txt")
        with open(done_path, "w", encoding="utf-8") as f:
            f.write(f"n_subjects={len(rows)}\nn_folds_saved={len(all_fold_summaries)}\n")
        print("DONE", done_path, flush=True)


if __name__ == "__main__":
    main()
