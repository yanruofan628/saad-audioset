#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Priority–Person–Gate models (additive; do not replace older baselines).

- residual_scalar_gate: AM/FM + Priority → S; EEGNet → B; DV = S + g·B
- residual_scalar_gate_tap / _ltar: Temporal Attention Pooling (replace AvgPool)
- residual_scalar_gate_segattn: 4-window segment attention (crop-then-encode)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from am_fm_salience import (
    DEFAULT_N_MELS,
    FM_BRANCH_DIM,
    EarSalienceEncoder,
)
from eeg_multiview_branches import build_eegnet_extractor, EEGNET_OUT_DIM


class _PriorityPersonGateBase(nn.Module):
    """Shared acoustic Priority pathway + scalar gate; EEG backend is pluggable."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        n_mels: int = DEFAULT_N_MELS,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_classes: int = 2,
        gate_hidden: int = 64,
        scalar_gate_hidden: int = 16,
        sfreq: int = 250,
        eeg_mode: str = 'default',
    ):
        super().__init__()
        self.audio_emb_dim = embed_dim * 2
        self.acoustic_feat_dim = embed_dim * 2 + 4
        self.gate_hidden = gate_hidden

        self.ear_encoder = EarSalienceEncoder(n_mels=n_mels, embed_dim=embed_dim)
        self.priority_head = nn.Sequential(
            nn.Linear(FM_BRANCH_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        self.eeg_extractor = build_eegnet_extractor(
            eeg_mode, eeg_channels, eeg_samples, sfreq=sfreq
        )
        eeg_dim = getattr(self.eeg_extractor, 'out_dim', EEGNET_OUT_DIM)

        self.acoustic_head = nn.Sequential(
            nn.Linear(self.acoustic_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(eeg_dim + self.audio_emb_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gate_hidden, gate_hidden),
            nn.ReLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(gate_hidden + self.audio_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )
        self.scalar_gate_mlp = nn.Sequential(
            nn.Linear(6, scalar_gate_hidden),
            nn.ReLU(),
            nn.Linear(scalar_gate_hidden, 1),
        )

    def _encode_ear(self, am, mel, s=None):
        z, _, fm_h = self.ear_encoder.forward_parts(am, mel)
        p = self.priority_head(fm_h)
        if s is None:
            raise ValueError(
                "s (pre-zscore envelope RMS) is required; do not compute RMS on "
                "z-scored AM (≈1). Pass s_left/s_right from am_fm_v2 cache."
            )
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        return z, p, s

    def _acoustic_forward(self, am_left, am_right, mel_left, mel_right, s_left, s_right):
        z_l, p_l, s_l = self._encode_ear(am_left, mel_left, s_left)
        z_r, p_r, s_r = self._encode_ear(am_right, mel_right, s_right)
        delta_p = p_l - p_r
        delta_am = s_l - s_r
        feat = torch.cat([z_l, z_r, p_l, p_r, delta_p, delta_am], dim=1)
        logits_audio = self.acoustic_head(feat)
        audio_emb = torch.cat([z_l, z_r], dim=1)
        return logits_audio, audio_emb, p_l, p_r, delta_p, delta_am

    def _eeg_feat(self, eeg_data, return_attn: bool = False):
        if return_attn and hasattr(self.eeg_extractor, 'forward'):
            try:
                return self.eeg_extractor(eeg_data, return_attn=True)
            except TypeError:
                pass
        return self.eeg_extractor(eeg_data), None

    def _scalar_gate_features(self, logits_audio, delta_p, delta_am, delta):
        s_conf = (logits_audio[:, 0] - logits_audio[:, 1]).abs()
        s_dp = delta_p.abs().squeeze(-1)
        s_dam = delta_am.abs().squeeze(-1)
        b_conf = (delta[:, 0] - delta[:, 1]).abs()
        b0 = delta[:, 0].abs()
        b1 = delta[:, 1].abs()
        return torch.stack([s_conf, s_dp, s_dam, b_conf, b0, b1], dim=1)

    def forward(
        self,
        eeg_data,
        am_left,
        am_right,
        mel_left,
        mel_right,
        s_left=None,
        s_right=None,
        return_parts: bool = False,
    ):
        logits_audio, audio_emb, p_l, p_r, delta_p, delta_am = self._acoustic_forward(
            am_left, am_right, mel_left, mel_right, s_left, s_right
        )
        eeg_feat, attn = self._eeg_feat(eeg_data, return_attn=return_parts)
        fused = self.fusion_mlp(torch.cat([eeg_feat, audio_emb], dim=1))
        delta = self.residual_head(torch.cat([fused, audio_emb], dim=1))
        gate_in = self._scalar_gate_features(logits_audio, delta_p, delta_am, delta)
        gate = torch.sigmoid(self.scalar_gate_mlp(gate_in))
        logits = logits_audio + gate * delta
        if return_parts:
            return logits, logits_audio, delta, p_l, p_r, gate, attn
        return logits


class ResidualScalarGate(_PriorityPersonGateBase):
    """DV = S + g·B with standard EEGNet pooling (no temporal attention)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('eeg_mode', 'default')
        super().__init__(*args, **kwargs)


class ResidualScalarGateTAP(_PriorityPersonGateBase):
    """
    Same Priority–Person–Gate; B readout uses Temporal Attention Pooling:
    after temporal conv, (B,F,1,T') → 1×1 score → softmax_t → Σ α_t h_t
    (replaces AdaptiveAvgPool over time).
    """

    def __init__(self, *args, **kwargs):
        kwargs['eeg_mode'] = 'temporal_attn'
        super().__init__(*args, **kwargs)


# Alias: earlier name for the same TAP backend.
ResidualScalarGateLTAR = ResidualScalarGateTAP


class ResidualScalarGateSegAttn(_PriorityPersonGateBase):
    """
    Same Priority–Person–Gate, but B is read out with segment attention:
    shared EEGNet on 0–200 / 200–500 / 500–800 / 800–2000 ms, then α=softmax(w).
    """

    def __init__(self, *args, **kwargs):
        kwargs['eeg_mode'] = 'segment_attn'
        super().__init__(*args, **kwargs)


MODEL_BUILDERS = {
    'residual_scalar_gate': ResidualScalarGate,
    'residual_scalar_gate_tap': ResidualScalarGateTAP,
    'residual_scalar_gate_ltar': ResidualScalarGateLTAR,
    'residual_scalar_gate_segattn': ResidualScalarGateSegAttn,
}
