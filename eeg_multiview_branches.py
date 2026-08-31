#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EEG 多视角支路 + 内置增强版 EEGNet（通道注意力 / 多分支空间 / ERSP 早期融合）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EEG_MULTIVIEW_BRANCH_DIM = 32
EEGNET_OUT_DIM = 160
SPATIAL_BRANCH_FILTERS = (4, 8, 16)


class SpatialChannelAttention(nn.Module):
    """conv_ica 之后的通道注意力：x * sigmoid(MLP(GlobalAvgPool(x)))。"""

    def __init__(self, n_filters: int, reduction: int = 2):
        super().__init__()
        mid = max(1, n_filters // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, mid),
            nn.ReLU(),
            nn.Linear(mid, n_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_filters, 1, time)
        weights = x.mean(dim=(2, 3))
        weights = torch.sigmoid(self.mlp(weights)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class _EEGNetTemporalHead(nn.Module):
    """EEGNet 共享时域头：permute → conv_time → BN → square → pool → 160 维。"""

    def __init__(self, n_spatial: int = 8, pool_spatial: int = 8):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 20, (1, 41), stride=(1, 1), bias=False)
        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5)
        self.poolmean = nn.AdaptiveAvgPool2d((pool_spatial, 1))
        self.out_dim = 20 * pool_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_spatial, 1, time)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.conv_time(x)
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.mul(x, x)
        x = self.poolmean(x)
        return x.view(x.size(0), -1)


class EEGNetFeatureExtractor(nn.Module):
    """标准 EEGNet 特征提取器（与 eeg_audio_residual 原版一致）。"""

    def __init__(self, n_channels: int, n_samples: int, n_spatial: int = 8):
        super().__init__()
        self.out_dim = EEGNET_OUT_DIM
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.temporal_head = _EEGNetTemporalHead(n_spatial=n_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_ica(x)
        return self.temporal_head(x)


class MultiScaleTemporalConv(nn.Module):
    """
    Parallel temporal convolutions after spatial EEG filtering.

    Input:  (B, C_in, 1, T)
    Output: (B, 3 * branch_channels, 1, T)  after BN
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 4,
        kernels: tuple[int, ...] = (25, 63, 125),
    ):
        super().__init__()
        self.kernels = tuple(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, kernel_size // 2),
                    bias=False,
                )
                for kernel_size in self.kernels
            ]
        )
        self.out_channels = branch_channels * len(self.kernels)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.bn(h)


class EEGNetMultiScaleTemporalExtractor(nn.Module):
    """
    空间滤波 → 多尺度并行时间卷积 → BN → square → AdaptiveAvgPool。

    相对 default EEGNet：只换时间卷积为多尺度；保留功率型平方与平均池化。
    默认 kernels (25,63,125) @ 250Hz ≈ 100/252/500 ms receptive fields。
    输出：3 * branch_channels（默认 12）；可选投影到 proj_dim。
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_spatial: int = 8,
        branch_channels: int = 4,
        kernels: tuple[int, ...] = (25, 63, 125),
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.multi_scale_temporal = MultiScaleTemporalConv(
            in_channels=n_spatial,
            branch_channels=branch_channels,
            kernels=kernels,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = self.multi_scale_temporal.out_channels
        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, proj_dim),
                nn.ELU(),
                nn.Dropout(0.3),
            )
            self.out_dim = proj_dim
        else:
            self.proj = None
            self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        h = self.conv_ica(x)  # (B, n_spatial, 1, T)
        h = F.dropout(h, 0.15, training=self.training)
        h = self.multi_scale_temporal(h)  # BN inside
        h = torch.mul(h, h)  # power / square（与原版一致）
        h = self.pool(h).flatten(1)
        if self.proj is not None:
            h = self.proj(h)
        return h


class TemporalAttentionPooling(nn.Module):
    """Lightweight temporal attention pool: (B, C, T) → (B, C), weights (B, T)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.score(x).squeeze(1)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        pooled = torch.sum(x * attention_weights.unsqueeze(1), dim=-1)
        return pooled, attention_weights


class SignedPowerDualReadout(nn.Module):
    """
    Dual readout from multi-scale EEG feature maps.

    Input:  h_ms (B, C, 1, T)
    Output: h_eeg (B, output_dim), attention_weights (B, T)
    """

    def __init__(
        self,
        feature_channels: int,
        output_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.signed_pool = TemporalAttentionPooling(feature_dim=feature_channels)
        self.proj = nn.Sequential(
            nn.Linear(2 * feature_channels, output_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_ms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = h_ms.squeeze(2)  # (B, C, T)
        h_signed_input = F.elu(h)
        h_signed, attention_weights = self.signed_pool(h_signed_input)
        h_power = h.pow(2).mean(dim=-1)
        h_eeg = self.proj(torch.cat([h_signed, h_power], dim=1))
        return h_eeg, attention_weights


class EEGNetMultiScaleDualExtractor(nn.Module):
    """
    空间滤波 → 多尺度时间卷积 → signed (TAP) + power 双读出 → 投影。

    相对 multi_scale_temporal：不再 square→全局池化一条路，而是并行保留极性与强度。
    默认输出 32 维。
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_spatial: int = 8,
        branch_channels: int = 4,
        kernels: tuple[int, ...] = (25, 63, 125),
        eeg_out_dim: int = 32,
    ):
        super().__init__()
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.multi_scale_temporal = MultiScaleTemporalConv(
            in_channels=n_spatial,
            branch_channels=branch_channels,
            kernels=kernels,
        )
        ms_channels = self.multi_scale_temporal.out_channels
        self.dual_readout = SignedPowerDualReadout(
            feature_channels=ms_channels,
            output_dim=eeg_out_dim,
            dropout=0.3,
        )
        self.out_dim = eeg_out_dim

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (B, 1, C, T)
        h = self.conv_ica(x)
        h = F.dropout(h, 0.15, training=self.training)
        h_ms = self.multi_scale_temporal(h)
        h_eeg, temporal_weights = self.dual_readout(h_ms)
        if return_attn:
            return h_eeg, temporal_weights
        return h_eeg


class EEGNetChannelAttnExtractor(nn.Module):
    """conv_ica 后插入通道注意力，再走时域头。"""

    def __init__(self, n_channels: int, n_samples: int, n_spatial: int = 8):
        super().__init__()
        self.out_dim = EEGNET_OUT_DIM
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.channel_attn = SpatialChannelAttention(n_spatial)
        self.temporal_head = _EEGNetTemporalHead(n_spatial=n_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_ica(x)
        x = self.channel_attn(x)
        return self.temporal_head(x)


class EEGNetMultiSpatialExtractor(nn.Module):
    """
    多分支空间头：不同 n_filters 的 depthwise spatial conv，
    在中间 feature map 上 concat → 1x1 融合，再走共享时域头。
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        branch_filters: tuple[int, ...] = SPATIAL_BRANCH_FILTERS,
        fuse_to: int = 8,
    ):
        super().__init__()
        self.out_dim = EEGNET_OUT_DIM
        self.spatial_branches = nn.ModuleList([
            nn.Conv2d(1, n_f, (n_channels, 1), bias=False) for n_f in branch_filters
        ])
        total_filters = sum(branch_filters)
        self.spatial_fusion = nn.Conv2d(total_filters, fuse_to, (1, 1), bias=False)
        self.fusion_bn = nn.BatchNorm2d(fuse_to)
        self.temporal_head = _EEGNetTemporalHead(n_spatial=fuse_to)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_maps = [branch(x) for branch in self.spatial_branches]
        x = torch.cat(branch_maps, dim=1)
        x = F.elu(self.fusion_bn(self.spatial_fusion(x)))
        return self.temporal_head(x)


class EEGNetTemporalAttnExtractor(nn.Module):
    """
    Temporal Attention Pooling (TAP): replace final AdaptiveAvgPool over time.

    After spatial + temporal conv (+ square), feature is reshaped to (B, F, 1, T'),
    scored with a 1×1 conv → α = softmax_t, z = Σ_t α_t h_t (F=160).
    Inserted where the old time-pool sat (resolution still ~T'), not on raw samples.
    """

    def __init__(self, n_channels: int, n_samples: int, n_spatial: int = 8):
        super().__init__()
        self.out_dim = EEGNET_OUT_DIM
        self.n_spatial = n_spatial
        feat_ch = 20 * n_spatial  # matches AdaptiveAvgPool((n_spatial,1)) → 160
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.conv_time = nn.Conv2d(1, 20, (1, 41), stride=(1, 1), bias=False)
        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5)
        # 1×1 score on (B, F, 1, T') → (B, 1, 1, T')
        self.temp_score = nn.Conv2d(feat_ch, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = self.conv_ica(x)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.conv_time(x)
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.mul(x, x)
        # (B, 20, n_spatial, T') → (B, F, 1, T'); replace AvgPool(time→1)
        bsz, c, s, t = x.shape
        h = x.reshape(bsz, c * s, 1, t)
        scores = self.temp_score(h)
        alpha = torch.softmax(scores, dim=-1)
        ctx = (h * alpha).sum(dim=-1).squeeze(-1)  # (B, F)
        if return_attn:
            return ctx, alpha.squeeze(1).squeeze(1)  # (B, T')
        return ctx


# Mid-latency-aligned analysis windows (ms relative to stimulus onset).
DEFAULT_SEG_WINDOWS_MS = (
    (0, 200),
    (200, 500),
    (500, 800),
    (800, 2000),
)


class EEGNetSegmentAttnExtractor(nn.Module):
    """
    Segment Attention readout (MLBR-style): crop EEG into fixed analysis
    windows, encode each with a *shared* EEGNet, then α = softmax(w) over
    the K segments (K=4 by default). Output dim matches EEGNet (160).

    Unlike dense LTAR, attention is only over semantically clear windows
    (early / mid / late / tail), so mid-latency (200–500 ms) stays interpretable.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sfreq: int = 250,
        n_spatial: int = 8,
        windows_ms: tuple[tuple[int, int], ...] = DEFAULT_SEG_WINDOWS_MS,
        min_seg_samples: int = 41,
    ):
        super().__init__()
        self.out_dim = EEGNET_OUT_DIM
        self.sfreq = int(sfreq)
        self.n_samples = int(n_samples)
        self.windows_ms = tuple(windows_ms)
        self.min_seg_samples = int(min_seg_samples)
        self.n_segments = len(self.windows_ms)

        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.temporal_head = _EEGNetTemporalHead(n_spatial=n_spatial)
        # 4 logits only (global); trial-invariant window preference.
        self.seg_logits = nn.Parameter(torch.zeros(self.n_segments))

        # Precompute sample index ranges (clamped to available length).
        ranges: list[tuple[int, int]] = []
        for t0_ms, t1_ms in self.windows_ms:
            i0 = int(round(t0_ms * self.sfreq / 1000.0))
            i1 = int(round(t1_ms * self.sfreq / 1000.0))
            i0 = max(0, min(i0, self.n_samples))
            i1 = max(i0 + 1, min(i1, self.n_samples))
            ranges.append((i0, i1))
        self.register_buffer(
            '_seg_starts',
            torch.tensor([r[0] for r in ranges], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            '_seg_ends',
            torch.tensor([r[1] for r in ranges], dtype=torch.long),
            persistent=False,
        )

    def _crop_segment(self, x: torch.Tensor, i0: int, i1: int) -> torch.Tensor:
        """x: (B, 1, C, T) → crop time; right-pad if shorter than min_seg_samples."""
        seg = x[..., i0:i1]
        need = self.min_seg_samples - seg.shape[-1]
        if need > 0:
            seg = F.pad(seg, (0, need))
        return seg

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (B, 1, C, T)
        emb_list = []
        for k in range(self.n_segments):
            i0 = int(self._seg_starts[k].item())
            i1 = int(self._seg_ends[k].item())
            seg = self._crop_segment(x, i0, i1)
            h = self.temporal_head(self.conv_ica(seg))
            emb_list.append(h)
        embs = torch.stack(emb_list, dim=1)  # (B, K, 160)
        alpha = torch.softmax(self.seg_logits, dim=0)  # (K,)
        ctx = (embs * alpha.view(1, -1, 1)).sum(dim=1)
        if return_attn:
            # Broadcast α to batch for a stable API with LTAR.
            return ctx, alpha.unsqueeze(0).expand(x.size(0), -1)
        return ctx


class EEGNetEarlyERSPExtractor(nn.Module):
    """
    ERSP 早期融合：conv_ica 空间图与 STFT 时频图在中间 feature map 融合，再走时域头。
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sfreq: int = 250,
        n_spatial: int = 8,
        n_fft: int = 64,
        hop_length: int = 16,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out_dim = EEGNET_OUT_DIM
        self.conv_ica = nn.Conv2d(1, n_spatial, (n_channels, 1), stride=(1, 1), bias=False)
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        self.ersp_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, n_spatial, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.ersp_fusion = nn.Conv2d(n_spatial * 2, n_spatial, (1, 1), bias=False)
        self.fusion_bn = nn.BatchNorm2d(n_spatial)
        self.temporal_head = _EEGNetTemporalHead(n_spatial=n_spatial)

    def _ersp_feature_map(self, x: torch.Tensor, target_time: int) -> torch.Tensor:
        eeg = x.squeeze(1).mean(dim=1)
        window = self.window.to(device=eeg.device, dtype=eeg.dtype)
        stft = torch.stft(
            eeg,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        mag = torch.log1p(stft.abs().unsqueeze(1))
        feat = self.ersp_encoder(mag)
        return F.adaptive_avg_pool2d(feat, (1, target_time))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_map = self.conv_ica(x)
        ersp_map = self._ersp_feature_map(x, spatial_map.shape[-1])
        fused = torch.cat([spatial_map, ersp_map], dim=1)
        fused = F.elu(self.fusion_bn(self.ersp_fusion(fused)))
        return self.temporal_head(fused)


def build_eegnet_extractor(
    mode: str,
    n_channels: int,
    n_samples: int,
    sfreq: int = 250,
) -> nn.Module:
    """构建 EEGNet 特征提取器。

    mode: default | multi_scale_temporal | multi_scale_dual | channel_attn |
          multi_spatial | early_ersp | temporal_attn | segment_attn
    """
    builders = {
        'default': lambda: EEGNetFeatureExtractor(n_channels, n_samples),
        'multi_scale_temporal': lambda: EEGNetMultiScaleTemporalExtractor(
            n_channels, n_samples
        ),
        'multi_scale_dual': lambda: EEGNetMultiScaleDualExtractor(
            n_channels, n_samples
        ),
        'channel_attn': lambda: EEGNetChannelAttnExtractor(n_channels, n_samples),
        'multi_spatial': lambda: EEGNetMultiSpatialExtractor(n_channels, n_samples),
        'early_ersp': lambda: EEGNetEarlyERSPExtractor(n_channels, n_samples, sfreq=sfreq),
        'temporal_attn': lambda: EEGNetTemporalAttnExtractor(n_channels, n_samples),
        'segment_attn': lambda: EEGNetSegmentAttnExtractor(
            n_channels, n_samples, sfreq=sfreq
        ),
    }
    if mode not in builders:
        raise ValueError(f"未知 EEG 模式: {mode}，可选 {sorted(builders)}")
    return builders[mode]()


# --- 外挂消融支路（保留，供 residual_gated_spatial / residual_gated_ersp 使用）---


class EEGSpatialFeatureExtractor(nn.Module):
    """可学习空间滤波支路（CSP 风格），外挂 concat 用。"""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_filters: int = 8,
        out_dim: int = EEG_MULTIVIEW_BRANCH_DIM,
    ):
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, n_filters, (n_channels, 1), bias=False)
        self.temporal_conv = nn.Conv2d(n_filters, n_filters, (1, 25), padding=(0, 12), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_filters, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = F.elu(self.bn(self.temporal_conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class EEGERSPFeatureExtractor(nn.Module):
    """ERSP 外挂支路：通道平均 STFT + 小 CNN，concat 用。"""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sfreq: int = 250,
        out_dim: int = EEG_MULTIVIEW_BRANCH_DIM,
        n_fft: int = 64,
        hop_length: int = 16,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 16, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eeg = x.squeeze(1).mean(dim=1)
        window = self.window.to(device=eeg.device, dtype=eeg.dtype)
        stft = torch.stft(
            eeg,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        mag = torch.log1p(stft.abs().unsqueeze(1))
        h = self.encoder(mag).flatten(1)
        return self.fc(h)
