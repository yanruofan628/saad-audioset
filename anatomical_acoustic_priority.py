#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anatomically anchored acoustic-priority model (audio-only).

Cochlear-inspired Mel channelization → multi-scale subband envelope +
spectrotemporal branch → per-ear scalar priority q → antisymmetric
dS = q_R − q_L → choice logits [-dS, dS].

Not an anatomical simulation; functionally aligned acoustic priority.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from am_fm_salience import DEFAULT_N_MELS, DEFAULT_CONTENT_PRIO_LOSS_WEIGHT

ST_BRANCH_DIM = 32
ENV_BRANCH_DIM = 32
ENV_CONV_CHANNELS = 8
S_ENV_DIM = 4  # rms, temporal_std, crest, mean_abs_delta
# z = h_ENV || h_ST || s_ENV || s_loud || p  → 32+32+4+1+1
Z_EAR_DIM = ENV_BRANCH_DIM + ST_BRANCH_DIM + S_ENV_DIM + 1 + 1


def compute_s_env(mel: torch.Tensor) -> torch.Tensor:
    """Fixed interpretable scalars from Mel energy (B, K, T) → (B, 4)."""
    # treat mel as band energy over time; mean across bands then time stats
    e_t = mel.mean(dim=1)  # (B, T)
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
    """Within-band temporal filter, then pointwise cross-band mix."""

    def __init__(self, n_bands: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.dw = nn.Conv1d(
            n_bands,
            n_bands,
            kernel_size=kernel_size,
            padding=padding,
            groups=n_bands,  # one temporal kernel per Mel band
            bias=True,
        )
        self.pw = nn.Conv1d(n_bands, out_ch, kernel_size=1)  # cross-band after within-band
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class MultiScaleEnvEncoder(nn.Module):
    """Subband envelope / modulation: depthwise temporal → pointwise cross-band.

    Keeps cochlear-inspired channelization: modulate each band first,
    then integrate across bands (not mixing all bands in the first layer).
    """

    def __init__(self, n_bands: int = DEFAULT_N_MELS, out_dim: int = ENV_BRANCH_DIM):
        super().__init__()
        c = ENV_CONV_CHANNELS
        self.short = _DepthwiseScale(n_bands, c, kernel_size=5, padding=2)
        self.mid = _DepthwiseScale(n_bands, c, kernel_size=15, padding=7)
        self.long = _DepthwiseScale(n_bands, c, kernel_size=31, padding=15)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(3 * c, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, K, T) — K treated as independent cochlear-like channels
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
    """Spectrotemporal / auditory-object branch (same capacity as prior FM CNN)."""

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
        # mel: (B, K, T) → (B, 1, K, T)
        return self.conv(mel.unsqueeze(1)).view(mel.size(0), -1)


class AnatomicalEarEncoder(nn.Module):
    """Per-ear: env + ST + scalars → auxiliary p + priority q."""

    def __init__(self, n_mels: int = DEFAULT_N_MELS):
        super().__init__()
        self.env = MultiScaleEnvEncoder(n_bands=n_mels, out_dim=ENV_BRANCH_DIM)
        self.st = SpectrotemporalEncoder(out_dim=ST_BRANCH_DIM)
        self.content_head = nn.Sequential(
            nn.Linear(ST_BRANCH_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(Z_EAR_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(
        self, mel: torch.Tensor, s_loud: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns q (B,1), p (B,1), z (B, Z_EAR_DIM).
        """
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
    """
    Shared ear encoder; binaural competition via antisymmetric dS = q_R − q_L.
    Audio-only; trainable / evaluable independently.
    """

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
        d_s = q_r - q_l  # (B, 1)
        logits = torch.cat([-d_s, d_s], dim=1)
        if return_parts:
            return logits, q_l, q_r, p_l, p_r, d_s, z_l, z_r
        return logits


class ResidualGatedAnatomicalPriority(nn.Module):
    """AnatomicalAcousticPriority backbone + SGCM-style gated EEG residual."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        n_mels: int = DEFAULT_N_MELS,
        n_classes: int = 2,
        gate_hidden: int = 64,
        hidden_dim: int = 64,
        sfreq: float = 250.0,
    ):
        super().__init__()
        assert n_classes == 2
        from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM

        self.ear = AnatomicalEarEncoder(n_mels=n_mels)
        self.feat_dim = 2 * Z_EAR_DIM  # z_L || z_R for gate context
        self.eeg_extractor = build_eegnet_extractor(
            "default", eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, "out_dim", EEGNET_OUT_DIM)
        self.fusion_stem = nn.Sequential(
            nn.Linear(eeg_dim + self.feat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden * 2),
        )
        self.channel_gate = nn.Linear(gate_hidden, gate_hidden)
        self.residual_head = nn.Sequential(
            nn.Linear(gate_hidden + self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        eeg_data: torch.Tensor,
        mel_left: torch.Tensor,
        mel_right: torch.Tensor,
        s_left: torch.Tensor,
        s_right: torch.Tensor,
        return_parts: bool = False,
    ):
        q_l, p_l, z_l = self.ear(mel_left, s_left)
        q_r, p_r, z_r = self.ear(mel_right, s_right)
        d_s = q_r - q_l
        logits_audio = torch.cat([-d_s, d_s], dim=1)
        feat = torch.cat([z_l, z_r], dim=1)
        eeg_feat = self.eeg_extractor(eeg_data)
        q, k = self.fusion_stem(torch.cat([eeg_feat, feat], dim=1)).chunk(2, dim=1)
        gated = torch.sigmoid(self.channel_gate(q)) * k
        delta = self.residual_head(torch.cat([gated, feat], dim=1))
        logits = logits_audio + delta
        if return_parts:
            return logits, logits_audio, delta, p_l, p_r, q_l, q_r
        return logits


class ResidualScalarGateDetachAnatomical(nn.Module):
    """D = S + g·B with train-time detach so fusion does not update Audio.

    - Audio learns only from loss_audio (and optional content-priority on p).
    - Residual / gate learn from loss_fusion on D_train = S.detach() + g·B.
    - Audio context into residual and gate evidence are detached.
    Forward value equals S + g·B; detach only affects autograd.
    """

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        n_mels: int = DEFAULT_N_MELS,
        n_classes: int = 2,
        hidden_dim: int = 64,
        eeg_mode: str = "default",
        sfreq: float = 250.0,
    ):
        super().__init__()
        assert n_classes == 2
        from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM

        self.ear = AnatomicalEarEncoder(n_mels=n_mels)
        self.context_dim = 2 * Z_EAR_DIM
        self.eeg_extractor = build_eegnet_extractor(
            eeg_mode, eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, "out_dim", EEGNET_OUT_DIM)
        self.residual_head = nn.Sequential(
            nn.Linear(eeg_dim + self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        # gate_input: |d_s|, |d_b|, d_s*d_b → 3 dims
        self.gate_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

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


class FrozenAudioEEGResidual(nn.Module):
    """Stage-2 readout: frozen ear provides S; EEG-only residual B.

    Default: D = S + σ(α) B. If use_gate=False: D = S + B (no learned mix).
    B does not see acoustic z.
    """

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        n_mels: int = DEFAULT_N_MELS,
        hidden_dim: int = 64,
        eeg_mode: str = "multi_scale_dual",
        sfreq: float = 250.0,
        use_gate: bool = True,
    ):
        super().__init__()
        from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM

        self.use_gate = bool(use_gate)
        self.ear = AnatomicalEarEncoder(n_mels=n_mels)
        self.eeg_extractor = build_eegnet_extractor(
            eeg_mode, eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, "out_dim", EEGNET_OUT_DIM)
        # Match EEG-only classifier head: 32 → 64 → 2
        self.residual_head = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )
        if self.use_gate:
            self.gate_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate_logit", None)

    def freeze_ear(self) -> None:
        self.ear.eval()
        for p in self.ear.parameters():
            p.requires_grad = False

    def forward(
        self,
        eeg_data: torch.Tensor,
        mel_left: torch.Tensor,
        mel_right: torch.Tensor,
        s_left: torch.Tensor,
        s_right: torch.Tensor,
        return_parts: bool = False,
    ):
        with torch.no_grad():
            q_l, _, _ = self.ear(mel_left, s_left)
            q_r, _, _ = self.ear(mel_right, s_right)
            d_prio = q_r - q_l
            S = torch.cat([-d_prio, d_prio], dim=1)
        B = self.residual_head(self.eeg_extractor(eeg_data))
        if self.use_gate:
            g = torch.sigmoid(self.gate_logit)
            D = S + g * B
        else:
            g = B.new_ones(1)
            D = S + B
        if return_parts:
            return D, S, B, g
        return D


class MatchedJointStopgradSB(nn.Module):
    """One-stage counterpart of FrozenAudioEEGResidual(use_gate=False).

    Same architecture: EEG-only B, head 32→64→2, no gate, D = S + B.
    Train-time D = sg(S) + B so CE(D) updates only the EEG residual;
    Audio is trained by CE(S). Eval uses D = S + B (identical values).
    Ear is NOT frozen.
    """

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        n_mels: int = DEFAULT_N_MELS,
        hidden_dim: int = 64,
        eeg_mode: str = "multi_scale_dual",
        sfreq: float = 250.0,
    ):
        super().__init__()
        from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM

        self.ear = AnatomicalEarEncoder(n_mels=n_mels)
        self.eeg_extractor = build_eegnet_extractor(
            eeg_mode, eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, "out_dim", EEGNET_OUT_DIM)
        self.residual_head = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

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
        q_l, _, _ = self.ear(mel_left, s_left)
        q_r, _, _ = self.ear(mel_right, s_right)
        d_prio = q_r - q_l
        S = torch.cat([-d_prio, d_prio], dim=1)
        B = self.residual_head(self.eeg_extractor(eeg_data))
        D = (S.detach() if train_detach else S) + B
        g = B.new_ones(1)
        if return_parts:
            return D, S, B, g
        return D


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "AnatomicalAcousticPriority",
    "ResidualGatedAnatomicalPriority",
    "ResidualScalarGateDetachAnatomical",
    "FrozenAudioEEGResidual",
    "MatchedJointStopgradSB",
    "AnatomicalEarEncoder",
    "MultiScaleEnvEncoder",
    "SpectrotemporalEncoder",
    "compute_s_env",
    "count_parameters",
    "DEFAULT_CONTENT_PRIO_LOSS_WEIGHT",
    "Z_EAR_DIM",
]
