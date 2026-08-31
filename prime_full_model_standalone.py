#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRIME 整模型（可独立复制）

音频前端（离线/在线）：
  wav → Mel(z-score) + 响度标量 s（包络 RMS，z-score 前）

声学优先：
  Mel → MultiScaleEnv + Spectrotemporal → z → q
  S = [-(q_R-q_L), +(q_R-q_L)]

融合（训练 detach）：
  D = S.detach() + g * B
  B = residual([eeg_feat || z_L||z_R.detach()])
  g = σ(gate(|dS|, |dB|, dS·dB))
  EEG = multi_scale_dual（空间 → 多尺度时间 → signed+power）

依赖：torch, numpy, librosa, scipy
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_SR = 16000
DEFAULT_DURATION_S = 2.0
DEFAULT_HOP = 400
DEFAULT_N_FFT = 512
DEFAULT_N_MELS = 32
DEFAULT_FMIN = 50.0
DEFAULT_FMAX = 8000.0
DEFAULT_CONTENT_PRIO_LOSS_WEIGHT = 0.2

ST_BRANCH_DIM = 32
ENV_BRANCH_DIM = 32
ENV_CONV_CHANNELS = 8
S_ENV_DIM = 4
Z_EAR_DIM = ENV_BRANCH_DIM + ST_BRANCH_DIM + S_ENV_DIM + 1 + 1  # 70


# ===========================================================================
# 1) 特征：Mel + 响度标量 s
# ===========================================================================
def compute_am_envelope(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP,
    duration_s: float = DEFAULT_DURATION_S,
) -> Tuple[np.ndarray, float]:
    """Hilbert 包络 → 帧平均序列 out；s = RMS(out)（z-score 前）。"""
    from scipy.signal import hilbert

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
    """Log-Mel + per-clip z-score → (n_mels, T)。"""
    import librosa

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


def extract_pair_features(
    y_left: np.ndarray, y_right: np.ndarray, sr: int = DEFAULT_SR
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """返回 mel_l, mel_r, s_l, s_r（解剖模型只需这四项）。"""
    _, s_l = compute_am_envelope(y_left, sr=sr)
    _, s_r = compute_am_envelope(y_right, sr=sr)
    mel_l = compute_fm_logmel(y_left, sr=sr)
    mel_r = compute_fm_logmel(y_right, sr=sr)
    return mel_l, mel_r, s_l, s_r


# ===========================================================================
# 2) 声学优先：EarEncoder → q → S
# ===========================================================================
def compute_s_env(mel: torch.Tensor) -> torch.Tensor:
    """Mel (B,K,T) → 4 维固定标量。"""
    e_t = mel.mean(dim=1)
    rms = e_t.pow(2).mean(dim=-1, keepdim=True).sqrt()
    tstd = e_t.std(dim=-1, keepdim=True)
    peak = e_t.abs().amax(dim=-1, keepdim=True)
    crest = peak / (rms + 1e-8)
    if e_t.size(-1) > 1:
        dabs = (e_t[:, 1:] - e_t[:, :-1]).abs().mean(dim=-1, keepdim=True)
    else:
        dabs = torch.zeros_like(rms)
    return torch.cat([rms, tstd, crest, dabs], dim=1)


class _DepthwiseScale(nn.Module):
    def __init__(self, n_bands: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.dw = nn.Conv1d(
            n_bands, n_bands, kernel_size=kernel_size, padding=padding, groups=n_bands
        )
        self.pw = nn.Conv1d(n_bands, out_ch, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class MultiScaleEnvEncoder(nn.Module):
    def __init__(self, n_bands: int = DEFAULT_N_MELS, out_dim: int = ENV_BRANCH_DIM):
        super().__init__()
        c = ENV_CONV_CHANNELS
        self.short = _DepthwiseScale(n_bands, c, kernel_size=5, padding=2)
        self.mid = _DepthwiseScale(n_bands, c, kernel_size=15, padding=7)
        self.long = _DepthwiseScale(n_bands, c, kernel_size=31, padding=15)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(3 * c, out_dim), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = torch.cat(
            [
                self.pool(self.short(mel)).squeeze(-1),
                self.pool(self.mid(mel)).squeeze(-1),
                self.pool(self.long(mel)).squeeze(-1),
            ],
            dim=1,
        )
        return self.proj(h)


class SpectrotemporalEncoder(nn.Module):
    def __init__(self, out_dim: int = ST_BRANCH_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.conv(mel.unsqueeze(1)).view(mel.size(0), -1)


class AnatomicalEarEncoder(nn.Module):
    def __init__(self, n_mels: int = DEFAULT_N_MELS):
        super().__init__()
        self.env = MultiScaleEnvEncoder(n_bands=n_mels, out_dim=ENV_BRANCH_DIM)
        self.st = SpectrotemporalEncoder(out_dim=ST_BRANCH_DIM)
        self.content_head = nn.Sequential(
            nn.Linear(ST_BRANCH_DIM, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1)
        )
        self.priority_head = nn.Sequential(
            nn.Linear(Z_EAR_DIM, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1)
        )

    def forward(
        self, mel: torch.Tensor, s_loud: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if s_loud.ndim == 1:
            s_loud = s_loud.unsqueeze(-1)
        h_env = self.env(mel)
        h_st = self.st(mel)
        s_env = compute_s_env(mel)
        p = self.content_head(h_st)
        z = torch.cat([h_env, h_st, s_env, s_loud, p], dim=1)
        q = self.priority_head(z)
        return q, p, z


class AnatomicalAcousticPriority(nn.Module):
    """纯声学：S = [-(q_R-q_L), +(q_R-q_L)]。"""

    def __init__(self, n_mels: int = DEFAULT_N_MELS, n_classes: int = 2):
        super().__init__()
        assert n_classes == 2
        self.ear = AnatomicalEarEncoder(n_mels=n_mels)

    def forward(
        self,
        mel_left: torch.Tensor,
        mel_right: torch.Tensor,
        s_left: torch.Tensor,
        s_right: torch.Tensor,
        return_parts: bool = False,
    ):
        q_l, p_l, z_l = self.ear(mel_left, s_left)
        q_r, p_r, z_r = self.ear(mel_right, s_right)
        d_s = q_r - q_l
        logits = torch.cat([-d_s, d_s], dim=1)
        if return_parts:
            return logits, q_l, q_r, p_l, p_r, d_s, z_l, z_r
        return logits


# ===========================================================================
# 3) EEG：multi_scale_dual
# ===========================================================================
class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 4,
        kernels: tuple = (25, 63, 125),
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    bias=False,
                )
                for k in kernels
            ]
        )
        self.out_channels = branch_channels * len(kernels)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.cat([b(x) for b in self.branches], dim=1))


class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Conv1d(feature_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        w = torch.softmax(self.score(x).squeeze(1), dim=-1)
        return torch.sum(x * w.unsqueeze(1), dim=-1), w


class SignedPowerDualReadout(nn.Module):
    def __init__(self, feature_channels: int, output_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.signed_pool = TemporalAttentionPooling(feature_channels)
        self.proj = nn.Sequential(
            nn.Linear(2 * feature_channels, output_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_ms: torch.Tensor):
        h = h_ms.squeeze(2)
        h_signed, attn = self.signed_pool(F.elu(h))
        h_power = h.pow(2).mean(dim=-1)
        return self.proj(torch.cat([h_signed, h_power], dim=1)), attn


class EEGNetMultiScaleDualExtractor(nn.Module):
    """EEG (B,1,C,T) → 32-d。"""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_spatial: int = 8,
        branch_channels: int = 4,
        kernels: tuple = (25, 63, 125),
        eeg_out_dim: int = 32,
    ):
        super().__init__()
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), bias=False)
        self.multi_scale_temporal = MultiScaleTemporalConv(
            n_spatial, branch_channels, kernels
        )
        self.dual_readout = SignedPowerDualReadout(
            self.multi_scale_temporal.out_channels, eeg_out_dim, 0.3
        )
        self.out_dim = eeg_out_dim

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h = F.dropout(self.conv_ica(x), 0.15, training=self.training)
        h_ms = self.multi_scale_temporal(h)
        h_eeg, attn = self.dual_readout(h_ms)
        return (h_eeg, attn) if return_attn else h_eeg


# ===========================================================================
# 4) 融合：D = S + g·B（训练 detach）
# ===========================================================================
class ResidualScalarGateDetachAnatomical(nn.Module):
    """
    train_detach=True 时：
      ctx / gate 证据 / S → detach，融合不回传更新声学。
    """

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        n_mels: int = DEFAULT_N_MELS,
        n_classes: int = 2,
        hidden_dim: int = 64,
        sfreq: float = 250.0,
    ):
        super().__init__()
        assert n_classes == 2
        self.ear = AnatomicalEarEncoder(n_mels=n_mels)
        self.context_dim = 2 * Z_EAR_DIM
        self.eeg_extractor = EEGNetMultiScaleDualExtractor(eeg_channels, eeg_samples)
        eeg_dim = self.eeg_extractor.out_dim
        self.residual_head = nn.Sequential(
            nn.Linear(eeg_dim + self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.gate_head = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(
        self,
        eeg_data: torch.Tensor,
        mel_left: torch.Tensor,
        mel_right: torch.Tensor,
        s_left: torch.Tensor,
        s_right: torch.Tensor,
        train_detach: bool = False,
        return_parts: bool = False,
    ):
        q_l, p_l, z_l = self.ear(mel_left, s_left)
        q_r, p_r, z_r = self.ear(mel_right, s_right)
        d_prio = q_r - q_l
        S = torch.cat([-d_prio, d_prio], dim=1)
        audio_context = torch.cat([z_l, z_r], dim=1)

        eeg_feat = self.eeg_extractor(eeg_data)
        ctx = audio_context.detach() if train_detach else audio_context
        B = self.residual_head(torch.cat([eeg_feat, ctx], dim=-1))

        d_s = S[:, 1:2] - S[:, 0:1]
        d_b = B[:, 1:2] - B[:, 0:1]
        if train_detach:
            gate_input = torch.cat(
                [d_s.detach().abs(), d_b.detach().abs(), d_s.detach() * d_b.detach()],
                dim=-1,
            )
            g = torch.sigmoid(self.gate_head(gate_input))
            D = S.detach() + g * B
        else:
            gate_input = torch.cat([d_s.abs(), d_b.abs(), d_s * d_b], dim=-1)
            g = torch.sigmoid(self.gate_head(gate_input))
            D = S + g * B

        if return_parts:
            return D, S, B, g, p_l, p_r
        return D


# ===========================================================================
# 5) 损失（可选）
# ===========================================================================
def content_priority_margin_loss(
    p_l: torch.Tensor, p_r: torch.Tensor, y: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    """y: 0=左, 1=右；希望被选侧 p 更大。"""
    # chosen - unchosen
    diff = torch.where(y.view(-1, 1) == 1, p_r - p_l, p_l - p_r)
    return F.relu(margin - diff).mean()


def audio_loss(
    S: torch.Tensor,
    y: torch.Tensor,
    p_l: Optional[torch.Tensor] = None,
    p_r: Optional[torch.Tensor] = None,
    prio_w: float = DEFAULT_CONTENT_PRIO_LOSS_WEIGHT,
) -> torch.Tensor:
    loss = F.cross_entropy(S, y)
    if p_l is not None and p_r is not None and prio_w > 0:
        loss = loss + prio_w * content_priority_margin_loss(p_l, p_r, y)
    return loss


# ===========================================================================
# smoke test
# ===========================================================================
if __name__ == "__main__":
    B, C, T, K, Tm = 4, 64, 500, 32, 80
    mel_l = torch.randn(B, K, Tm)
    mel_r = torch.randn(B, K, Tm)
    s_l = torch.rand(B)
    s_r = torch.rand(B)
    eeg = torch.randn(B, 1, C, T)
    y = torch.randint(0, 2, (B,))

    audio = AnatomicalAcousticPriority()
    S, *_rest = audio(mel_l, mel_r, s_l, s_r, return_parts=True)
    print("S", S.shape)

    fusion = ResidualScalarGateDetachAnatomical(eeg_channels=C, eeg_samples=T)
    D, S2, B2, g, p_l, p_r = fusion(
        eeg, mel_l, mel_r, s_l, s_r, train_detach=True, return_parts=True
    )
    print("D", D.shape, "g", g.shape)
    print("loss", audio_loss(S2, y, p_l, p_r) + F.cross_entropy(D, y))
