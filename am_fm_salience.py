#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AM + FM binaural salience (Phase 1, no prototypes).

Per ear:
  AM  – Hilbert envelope downsampled to fixed frames
  FM  – log-mel spectrogram

Shared ear encoder → z_L, z_R → classifier → choice logits (pure audio).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert

# Defaults aligned with 2 s dichotic stimuli @ 16 kHz
DEFAULT_SR = 16000
DEFAULT_DURATION_S = 2.0
DEFAULT_HOP = 400          # 25 ms → 80 frames for 2 s
DEFAULT_N_FFT = 512
DEFAULT_N_MELS = 32
DEFAULT_FMIN = 80.0
DEFAULT_FMAX = 8000.0

# v2: cache also stores pre-zscore envelope RMS (s_left/s_right)
AM_FM_CACHE_TAG = "am_fm_v2"
FM_BRANCH_DIM = 32
DEFAULT_CONTENT_PRIO_LOSS_WEIGHT = 0.2


def compute_am_envelope(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP,
    duration_s: float = DEFAULT_DURATION_S,
) -> Tuple[np.ndarray, float]:
    """Hilbert envelope → fixed-length AM frames.

    Returns
    -------
    am_z : (T,) float32
        Per-clip temporal z-score (network input only).
    s_raw : float
        RMS of the frame envelope *before* z-score (explicit energy/prominence).
    """
    y = np.asarray(y, dtype=np.float32)
    env = np.abs(hilbert(y)).astype(np.float32)
    target_len = int(duration_s * sr)
    if len(env) < target_len:
        env = np.pad(env, (0, target_len - len(env)))
    else:
        env = env[:target_len]

    n_frames = max(1, int(np.ceil(target_len / hop_length)))
    frames = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + hop_length, target_len)
        if start >= target_len:
            break
        frames.append(float(env[start:end].mean()))
    out = np.asarray(frames, dtype=np.float32)
    s_raw = float(np.sqrt(np.mean(np.square(out)) + 1e-12))
    if out.std() > 1e-8:
        out = (out - out.mean()) / (out.std() + 1e-8)
    return out, s_raw


def compute_fm_logmel(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    duration_s: float = DEFAULT_DURATION_S,
) -> np.ndarray:
    """Log-mel spectrogram with per-clip z-score."""
    y = np.asarray(y, dtype=np.float32)
    target_len = int(duration_s * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=min(fmax, sr / 2 - 1),
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)
    if log_mel.std() > 1e-8:
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel


def extract_am_fm_pair(
    y_left: np.ndarray,
    y_right: np.ndarray,
    sr: int = DEFAULT_SR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    am_l, s_l = compute_am_envelope(y_left, sr=sr)
    am_r, s_r = compute_am_envelope(y_right, sr=sr)
    mel_l = compute_fm_logmel(y_left, sr=sr)
    mel_r = compute_fm_logmel(y_right, sr=sr)
    return am_l, am_r, mel_l, mel_r, s_l, s_r


class EarSalienceEncoder(nn.Module):
    """AM (1D) + FM (2D) → per-ear salience embedding."""

    def __init__(self, n_mels: int = DEFAULT_N_MELS, embed_dim: int = 32):
        super().__init__()
        self.am_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fuse = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward_parts(
        self, am: torch.Tensor, mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        am:  (B, T_am)
        mel: (B, n_mels, T_fm)
        returns: z, am_h, fm_h
        """
        am_x = am.unsqueeze(1)
        mel_x = mel.unsqueeze(1)
        am_h = self.am_conv(am_x).squeeze(-1)
        fm_h = self.fm_conv(mel_x).view(am_h.size(0), -1)
        z = self.fuse(torch.cat([am_h, fm_h], dim=1))
        return z, am_h, fm_h

    def forward(self, am: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """
        am:  (B, T_am)
        mel: (B, n_mels, T_fm)
        """
        z, _, _ = self.forward_parts(am, mel)
        return z


def am_envelope_rms(am: torch.Tensor) -> torch.Tensor:
    """RMS over time: (B, T) -> (B, 1).

    Do **not** use this on per-clip z-scored AM (RMS ≈ 1). Prefer cached
    ``s_left`` / ``s_right`` from pre-zscore envelopes.
    """
    return am.pow(2).mean(dim=-1, keepdim=True).sqrt()


def batch_am_fm_from_cache(
    am_fm: Dict[str, np.ndarray],
    pair_indices,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index cached AM/FM (+ pre-zscore s) onto device tensors."""
    idx = pair_indices
    am_l = torch.FloatTensor(am_fm['am_left'][idx]).to(device)
    am_r = torch.FloatTensor(am_fm['am_right'][idx]).to(device)
    mel_l = torch.FloatTensor(am_fm['mel_left'][idx]).to(device)
    mel_r = torch.FloatTensor(am_fm['mel_right'][idx]).to(device)
    if 's_left' not in am_fm or 's_right' not in am_fm:
        raise KeyError(
            "AM/FM cache missing s_left/s_right. Rebuild with AM_FM_CACHE_TAG=am_fm_v2 "
            "(pre-zscore envelope RMS)."
        )
    s_l = torch.FloatTensor(am_fm['s_left'][idx]).to(device)
    s_r = torch.FloatTensor(am_fm['s_right'][idx]).to(device)
    if s_l.ndim == 1:
        s_l = s_l.unsqueeze(-1)
    if s_r.ndim == 1:
        s_r = s_r.unsqueeze(-1)
    return am_l, am_r, mel_l, mel_r, s_l, s_r


def content_priority_margin_loss(
    p_l: torch.Tensor,
    p_r: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Encourage chosen-ear priority score to exceed the other by a margin."""
    diff = (p_l - p_r).squeeze(-1)
    sign = torch.where(y == 0, torch.ones_like(diff), -torch.ones_like(diff))
    return F.relu(margin - sign * diff).mean()


class AMFMSalienceClassifier(nn.Module):
    """
    Binaural AM+FM salience → choice logits.
    Shared ear encoder; no prototypes.
    """

    def __init__(self, n_mels: int = DEFAULT_N_MELS, embed_dim: int = 32, n_classes: int = 2):
        super().__init__()
        self.ear_encoder = EarSalienceEncoder(n_mels=n_mels, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(
        self,
        am_left: torch.Tensor,
        am_right: torch.Tensor,
        mel_left: torch.Tensor,
        mel_right: torch.Tensor,
    ) -> torch.Tensor:
        z_l = self.ear_encoder(am_left, mel_left)
        z_r = self.ear_encoder(am_right, mel_right)
        return self.classifier(torch.cat([z_l, z_r], dim=1))


def _pad_am_mel_batches(
    am_list: List[np.ndarray],
    mel_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    max_am = max(a.shape[0] for a in am_list)
    max_mel_t = max(m.shape[1] for m in mel_list)
    n_mels = mel_list[0].shape[0]

    am_pad = np.zeros((len(am_list), max_am), dtype=np.float32)
    mel_pad = np.zeros((len(mel_list), n_mels, max_mel_t), dtype=np.float32)
    for i, (a, m) in enumerate(zip(am_list, mel_list)):
        am_pad[i, : a.shape[0]] = a
        mel_pad[i, :, : m.shape[1]] = m
    return am_pad, mel_pad


def build_am_fm_cache(
    pair_names: List[str],
    stereo_path_map: Dict[str, str],
    load_stereo_fn,
    cache_dir: str,
    cache_tag: str = AM_FM_CACHE_TAG,
) -> Dict[str, np.ndarray]:
    """Extract and cache AM/FM tensors for all unique pairs."""
    os.makedirs(cache_dir, exist_ok=True)
    pair_names_str = ','.join(sorted(pair_names))
    pair_hash = hashlib.md5(pair_names_str.encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f'am_fm_{cache_tag}_{pair_hash}.npz')
    meta_path = os.path.join(cache_dir, f'am_fm_{cache_tag}_{pair_hash}_meta.json')

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if (
            meta.get('pair_names') == pair_names
            and meta.get('cache_tag') == cache_tag
            and meta.get('s_raw_pre_zscore') is True
        ):
            data = np.load(cache_path, allow_pickle=True)
            if 's_left' in data.files and 's_right' in data.files:
                print(f"从缓存加载 AM+FM 特征: {cache_path}")
                return {k: data[k] for k in data.files}
            print(f"缓存缺少 s_left/s_right，重新提取: {cache_path}")

    print(f"提取 {len(pair_names)} 个音频对的 AM+FM 特征...")
    am_l_list, am_r_list, mel_l_list, mel_r_list = [], [], [], []
    s_l_list, s_r_list = [], []
    valid_names = []
    for pn in pair_names:
        sp = stereo_path_map.get(pn)
        if not sp or not os.path.exists(sp):
            print(f"  跳过 {pn}: 立体声文件不存在")
            continue
        y_l, y_r, sr = load_stereo_fn(sp)
        am_l, am_r, mel_l, mel_r, s_l, s_r = extract_am_fm_pair(y_l, y_r, sr=sr)
        am_l_list.append(am_l)
        am_r_list.append(am_r)
        mel_l_list.append(mel_l)
        mel_r_list.append(mel_r)
        s_l_list.append(s_l)
        s_r_list.append(s_r)
        valid_names.append(pn)

    if not valid_names:
        raise RuntimeError("未能为任何音频对提取 AM+FM 特征")

    am_left, mel_left = _pad_am_mel_batches(am_l_list, mel_l_list)
    am_right, mel_right = _pad_am_mel_batches(am_r_list, mel_r_list)

    # align mel_right time dim with mel_left padding
    _, mel_right = _pad_am_mel_batches(am_r_list, mel_r_list)
    if mel_right.shape[2] != mel_left.shape[2]:
        t = max(mel_left.shape[2], mel_right.shape[2])
        ml = np.zeros((mel_left.shape[0], mel_left.shape[1], t), dtype=np.float32)
        mr = np.zeros((mel_right.shape[0], mel_right.shape[1], t), dtype=np.float32)
        ml[:, :, : mel_left.shape[2]] = mel_left
        mr[:, :, : mel_right.shape[2]] = mel_right
        mel_left, mel_right = ml, mr

    s_left = np.asarray(s_l_list, dtype=np.float32).reshape(-1, 1)
    s_right = np.asarray(s_r_list, dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(
        cache_path,
        am_left=am_left,
        am_right=am_right,
        mel_left=mel_left,
        mel_right=mel_right,
        s_left=s_left,
        s_right=s_right,
        pair_names=np.array(valid_names, dtype=object),
    )
    meta = {
        'pair_names': valid_names,
        'n_pairs': len(valid_names),
        'cache_tag': cache_tag,
        's_raw_pre_zscore': True,
        'shapes': {
            'am_left': list(am_left.shape),
            'mel_left': list(mel_left.shape),
            's_left': list(s_left.shape),
        },
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"AM+FM 特征已缓存: {cache_path}")
    print(
        f"  pre-zscore s: mean_L={s_left.mean():.4g} mean_R={s_right.mean():.4g} "
        f"|Δs|mean={np.abs(s_left - s_right).mean():.4g}"
    )

    return {
        'am_left': am_left,
        'am_right': am_right,
        'mel_left': mel_left,
        'mel_right': mel_right,
        's_left': s_left,
        's_right': s_right,
        'pair_names': np.array(valid_names, dtype=object),
    }


def align_am_fm_to_pair_names(
    am_fm_data: Dict[str, np.ndarray],
    pair_names: List[str],
) -> Dict[str, np.ndarray]:
    """Re-index cached AM/FM arrays to match global pair_names order."""
    cached_names = list(am_fm_data.get('pair_names', []))
    if isinstance(cached_names, np.ndarray):
        cached_names = cached_names.tolist()
    if not cached_names:
        raise ValueError("am_fm_data 缺少 pair_names")

    name_to_idx = {n: i for i, n in enumerate(cached_names)}
    indices = []
    for pn in pair_names:
        if pn not in name_to_idx:
            raise KeyError(f"AM+FM 缓存中缺少音频对: {pn}")
        indices.append(name_to_idx[pn])
    idx = np.asarray(indices, dtype=np.int32)
    out = {
        'am_left': am_fm_data['am_left'][idx],
        'am_right': am_fm_data['am_right'][idx],
        'mel_left': am_fm_data['mel_left'][idx],
        'mel_right': am_fm_data['mel_right'][idx],
    }
    if 's_left' in am_fm_data and 's_right' in am_fm_data:
        out['s_left'] = am_fm_data['s_left'][idx]
        out['s_right'] = am_fm_data['s_right'][idx]
    else:
        raise KeyError(
            "AM/FM cache missing s_left/s_right; rebuild with am_fm_v2 "
            "(pre-zscore envelope RMS)."
        )
    return out
