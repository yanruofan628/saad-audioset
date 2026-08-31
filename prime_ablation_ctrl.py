# -*- coding: utf-8 -*-
"""Shared seed / shuffle control for audio-only vs fusion ablation."""
from __future__ import annotations

import random

import numpy as np
import torch

ABLATION_SEED = 42

SUBJECTS_23 = [
    "yanxingzhuo",
    "jinxiaoyue",
    "chenxianwei",
    "yeziyuan",
    "zhangzhiyao",
    "haoxiang",
    "hehaohuai",
    "qiuhaiyun",
    "zhouyu",
    "honghaokai",
    "caolulu",
    "yanyinsong",
    "huangxiaohang",
    "xufan",
    "qiusiqi",
    "machenxiang",
    "lizhuhang",
    "zhanghanglei",
    "jichengzhi",
    "liuzehao",
    "zengdexin",
    "zhangyajie",
    "zhangyufei",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fold_seed(fold_idx: int) -> int:
    return ABLATION_SEED + int(fold_idx) * 1009


def epoch_perm(n: int, fold_idx: int, epoch: int) -> np.ndarray:
    rng = np.random.RandomState(ABLATION_SEED + int(fold_idx) * 100003 + int(epoch))
    return rng.permutation(n)


def seed_batch(fold_idx: int, epoch: int, batch_idx: int) -> None:
    """Reset torch RNG so ear dropout matches across audio-only and fusion."""
    s = ABLATION_SEED + int(fold_idx) * 1_000_000 + int(epoch) * 1_000 + int(batch_idx)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def fit_lambda_shrinkage(
    S: torch.Tensor,
    B: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 0.05,
    lam_min: float = 0.0,
    lam_max: float = 2.0,
    n_grid: int = 201,
):
    """λ* = argmin_λ [CE(y, S+λB) + γ λ²], λ ≥ 0.

    Fit on train (or inner-val), never on the reported test fold.
    """
    import torch.nn.functional as F

    S = S.detach()
    B = B.detach()
    y = y.detach().long().to(S.device)
    best_lam, best_obj = 0.0, float("inf")
    for lam in np.linspace(lam_min, lam_max, n_grid):
        logits = S + float(lam) * B
        ce = float(F.cross_entropy(logits, y))
        obj = ce + float(gamma) * (float(lam) ** 2)
        if obj < best_obj:
            best_obj = obj
            best_lam = float(lam)
    return best_lam, best_obj


def collect_sb(model, eeg, pair_idx, y, am_fm, device, batch_size: int, joint: bool):
    """Collect S, B logits for shrinkage. y is numpy labels already 0/1."""
    from am_fm_salience import batch_am_fm_from_cache

    model.eval()
    Ss, Bs = [], []
    n = len(y)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            Xe = torch.FloatTensor(eeg[sl]).unsqueeze(1).to(device)
            _, _, ml, mr, s_l, s_r = batch_am_fm_from_cache(am_fm, pair_idx[sl], device)
            if joint:
                _d, S, B, *_ = model(
                    Xe, ml, mr, s_l, s_r, train_detach=False, return_parts=True
                )
            else:
                _d, S, B, *_ = model(Xe, ml, mr, s_l, s_r, return_parts=True)
            Ss.append(S)
            Bs.append(B)
    y_t = torch.LongTensor(np.asarray(y)).to(device)
    return torch.cat(Ss, 0), torch.cat(Bs, 0), y_t
