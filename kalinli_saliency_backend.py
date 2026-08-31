#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalinli 等人（2009）听觉显著性图的 Python 复现（简化版）

该模块包含以下步骤：
1. ERB Gammatone 滤波器组 + IHC/侧抑制，生成 cochleagram（scm）
2. 多尺度高斯金字塔提取强度/频率/时间/方向/音高等特征
3. 跨尺度中心-周围抑制 + DoG 归一化
4. 线性组合得到显著性图

参考：Kalinli & Narayanan, "A Saliency-Based Auditory Attention Model...", Interspeech 2009.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import lfilter, resample
from scipy.ndimage import gaussian_filter, zoom

# ============
# ERB 工具
# ============

EAR_Q = 9.26449
MIN_BW = 24.7


def erb_space(num_channels: int, low_freq: float, high_freq: float) -> np.ndarray:
    """生成在 ERB 轴上线性间隔的中心频率"""
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")
    if high_freq <= low_freq:
        raise ValueError("high_freq must be greater than low_freq")

    ear_q = EAR_Q
    min_bw = MIN_BW
    low = low_freq + ear_q * min_bw
    high = high_freq + ear_q * min_bw
    cf = -(ear_q * min_bw) + np.exp(
        np.linspace(num_channels, 1, num_channels) * (-np.log(high) + np.log(low)) / num_channels
    ) * high
    return cf.astype(np.float64)


def make_erb_filters(
    fs: float, num_channels: int, low_freq: float = 100.0, hi_freq: float | None = None, order: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """生成 ERB Gammatone 滤波器参数"""
    if hi_freq is None:
        hi_freq = fs / 2.0
    cf = erb_space(num_channels, low_freq, hi_freq)
    T = 1.0 / fs

    erb = ((cf / EAR_Q) ** order + MIN_BW**order) ** (1.0 / order)
    B = 1.019 * 2 * np.pi * erb

    A0 = np.full_like(cf, T)
    A2 = np.zeros_like(cf)
    B0 = np.ones_like(cf)
    B1 = -2 * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    common = -2 * np.exp(4j * cf * np.pi * T) * T + 2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T
    gain = np.abs(
        common
        * (np.cos(2 * cf * np.pi * T) - np.sqrt(3 - 2 ** 1.5) * np.sin(2 * cf * np.pi * T))
        * (np.cos(2 * cf * np.pi * T) + np.sqrt(3 - 2 ** 1.5) * np.sin(2 * cf * np.pi * T))
        * (np.cos(2 * cf * np.pi * T) - np.sqrt(3 + 2 ** 1.5) * np.sin(2 * cf * np.pi * T))
        * (np.cos(2 * cf * np.pi * T) + np.sqrt(3 + 2 ** 1.5) * np.sin(2 * cf * np.pi * T))
        / (-2 / np.exp(2 * B * T) - 2 * np.exp(4j * cf * np.pi * T) + 2 * (1 + np.exp(4j * cf * np.pi * T)) / np.exp(B * T)) ** 4
    )

    A11 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) + 2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
    A12 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) - 2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
    A13 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) + 2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
    A14 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) - 2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2

    fcoefs = np.stack([A0, A11, A12, A13, A14, A2, B0, B1, B2, gain], axis=1)
    return fcoefs.astype(np.float64), cf


def erb_filter_bank(x: np.ndarray, fcoefs: np.ndarray) -> np.ndarray:
    """将输入信号通过 Gammatone 滤波器组"""
    if x.ndim != 1:
        raise ValueError("Input waveform must be 1-D")
    num_channels = fcoefs.shape[0]
    output = np.zeros((num_channels, x.shape[0]), dtype=np.float64)

    for chan in range(num_channels):
        A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = fcoefs[chan]
        y = lfilter([A0 / gain, A11 / gain, A2 / gain], [B0, B1, B2], x)
        y = lfilter([A0, A12, A2], [B0, B1, B2], y)
        y = lfilter([A0, A13, A2], [B0, B1, B2], y)
        y = lfilter([A0, A14, A2], [B0, B1, B2], y)
        output[chan, :] = y
    return output


# ============
# SCM / 侧抑制
# ============


def scm(
    s: np.ndarray,
    fs: float,
    chan_spec: Iterable[float] = (128, 100, 8000),
    bin_size_ms: float = 5.0,
    do_hpf: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Spectro-temporal cochleagram 模型
    返回：eResp (channels x frames), fx (帧采样率), cf (中心频率), tx (时间)
    """
    if s.ndim != 1:
        raise ValueError("Input waveform must be mono 1-D")

    n_chans, low_freq, hi_freq = chan_spec
    fcoefs, cf = make_erb_filters(fs, int(n_chans), low_freq, hi_freq)
    x = erb_filter_bank(s, fcoefs)

    if do_hpf:
        for i in range(x.shape[0]):
            x[i, :] = lfilter([1], [1, -0.99], x[i, :])

    x = np.maximum(x, 0.0)
    fx = int(round(1000.0 / bin_size_ms))
    num_frames = max(1, int(round(len(s) * fx / fs)))
    s_fac = 2e7
    x = resample(x, num_frames, axis=1) * s_fac
    x_floor = 1e-4
    x = np.maximum(x, 0.0) + x_floor
    dt = bin_size_ms / 1000.0
    tx = np.arange(num_frames, dtype=np.float64) * dt
    return x.astype(np.float32), float(fx), cf, tx


def _lin_kernel(radius: int = 1) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            d = dx * dx + dy * dy
            kernel[dx + radius, dy + radius] = 0.0 if d < 1 else 0.7 / d
    return kernel


@lru_cache(maxsize=8)
def _cached_lin_kernel(radius: int) -> np.ndarray:
    return _lin_kernel(radius)


def lateral_inhibition(x: np.ndarray, radius: int = 1) -> np.ndarray:
    """侧抑制网络（LIN）"""
    from scipy.signal import convolve2d

    kernel = _cached_lin_kernel(radius)
    inhibited = x - convolve2d(x, kernel, mode="same", boundary="symm")
    inhibited[inhibited < 0] = 0.0
    return inhibited.astype(np.float32)


# ============
# 特征提取
# ============


def _gabor_kernel(theta: float, mode: int) -> np.ndarray:
    t = 3.0
    g = 0.56 * t
    r = g
    if mode == 0:
        g = 0.5
        r = 0.3

    kernel = np.zeros((11, 11), dtype=np.float32)
    xs = np.linspace(-8, 8, 11)
    for xi, x in enumerate(xs):
        for yi, y in enumerate(xs):
            x1 = x * np.cos(theta) + y * np.sin(theta)
            y1 = -x * np.sin(theta) + y * np.cos(theta)
            kernel[yi, xi] = np.exp(
                -(0.2**2 * x1**2 / (2 * r**2) + 0.2**2 * y1**2 / (2 * g**2))
            ) * np.cos((2 * np.pi * 0.2 * x1) / t)
    if mode == 1:
        third = int(math.ceil(kernel.shape[1] / 3))
        kernel[:, :third] = 0.0
    return kernel - np.mean(kernel)


def _apply_feature_filter(img: np.ndarray, kernel: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    from scipy.signal import convolve2d

    response = np.abs(convolve2d(img, kernel, mode="same", boundary="symm"))
    if response.shape != target_shape:
        zoom_factors = (target_shape[0] / response.shape[0], target_shape[1] / response.shape[1])
        response = zoom(response, zoom_factors, order=1)
    return response.astype(np.float32)


def _pitch_feature(rescaled: np.ndarray, orig: np.ndarray) -> np.ndarray:
    xw, yw = rescaled.shape
    if orig.shape[1] == 0:
        return np.zeros_like(rescaled)
    block = max(1, int(np.ceil(8.0 / (orig.shape[1] / max(1, yw)))))
    max_lag = block // 2
    feature = np.zeros((xw, max(1, yw // block * (2 * max_lag + 1))), dtype=np.float32)
    for ch in range(xw):
        row = rescaled[ch]
        for m in range(0, yw - block, block):
            segment = row[m : m + block]
            corr = np.correlate(segment, segment, mode="full")
            mid = len(corr) // 2
            feature[ch, m // block * (2 * max_lag + 1) : m // block * (2 * max_lag + 1) + (2 * max_lag + 1)] = corr[
                mid - max_lag : mid + max_lag + 1
            ]
    if feature.shape != rescaled.shape:
        zoom_factors = (rescaled.shape[0] / feature.shape[0], rescaled.shape[1] / max(feature.shape[1], 1))
        feature = zoom(feature, zoom_factors, order=1)
    return np.abs(feature).astype(np.float32)


def downsample_pyramid(img: np.ndarray, levels: int) -> Tuple[np.ndarray, ...]:
    imgs = []
    current = img
    for _ in range(levels):
        imgs.append(current)
        if _ != levels - 1:
            current = zoom(current, 0.5, order=1)
    return tuple(imgs)


def center_surround_maps(feature_pyr: Tuple[np.ndarray, ...], offsets: Tuple[int, ...] = (2, 3)) -> Tuple[np.ndarray, ...]:
    cs_maps = []
    num_levels = len(feature_pyr)
    for c in range(num_levels):
        for off in offsets:
            s = c + off
            if s >= num_levels:
                continue
            diff = feature_pyr[c] - feature_pyr[s]
            diff = np.clip(diff, 0.0, None)
            cs_maps.append(diff.astype(np.float32))
    return tuple(cs_maps)


def dog_normalize(feature: np.ndarray, iterations: int = 20) -> np.ndarray:
    if not np.isfinite(feature).any():
        return np.zeros_like(feature, dtype=np.float32)

    normalized = feature.copy().astype(np.float32)
    max_val = float(np.max(normalized))
    if max_val > 0:
        normalized /= max_val
    normalized[normalized < 0] = 0.0
    for _ in range(iterations):
        excit = gaussian_filter(normalized, sigma=1.0)
        inhib = gaussian_filter(normalized, sigma=3.0)
        normalized = normalized + (excit - inhib) - 0.02
        normalized = np.clip(normalized, 0.0, None)
    return normalized


def kalinli_saliency_map(img: np.ndarray, levels: int = 4) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    根据多尺度特征计算显著性图
    返回： saliency_map, components(dict)
    """
    base_shape = img.shape
    pyramid = downsample_pyramid(img, levels)

    feature_pyrs: Dict[str, Tuple[np.ndarray, ...]] = {
        "intensity": [],
        "freq": [],
        "temporal": [],
        "ori1": [],
        "ori2": [],
        "pitch": [],
    }

    kernels = {
        "intensity": _gabor_kernel(np.pi, 0),
        "freq": _gabor_kernel(np.pi / 2, 2),
        "temporal": _gabor_kernel(np.pi, 1),
        "ori1": _gabor_kernel(np.pi / 4, 2),
        "ori2": _gabor_kernel(3 * np.pi / 4, 2),
    }

    for level_img in pyramid:
        level_features = {}
        for key, kernel in kernels.items():
            resp = _apply_feature_filter(level_img, kernel, base_shape)
            feature_pyrs[key].append(resp)
            level_features[key] = resp
        pitch = _pitch_feature(level_img, img)
        if pitch.shape != base_shape:
            pitch = zoom(pitch, (base_shape[0] / pitch.shape[0], base_shape[1] / max(1, pitch.shape[1])), order=1)
        feature_pyrs["pitch"].append(pitch.astype(np.float32))

    for key in feature_pyrs:
        feature_pyrs[key] = tuple(feature_pyrs[key])

    components: Dict[str, np.ndarray] = {}
    for key in feature_pyrs:
        cs_maps = center_surround_maps(feature_pyrs[key])
        if not cs_maps:
            components[key] = np.zeros(base_shape, dtype=np.float32)
            continue
        accum = np.zeros(base_shape, dtype=np.float32)
        for cs_map in cs_maps:
            accum += dog_normalize(cs_map)
        accum /= len(cs_maps)
        components[key] = accum

    saliency = (
        components["intensity"]
        + components["freq"]
        + components["temporal"]
        + components["ori1"]
        + components["ori2"]
        + components["pitch"]
    ) / 6.0

    return saliency.astype(np.float32), components


def compute_kalinli_saliency_from_audio(
    y: np.ndarray,
    sr: int = 16000,
    chan_spec: Tuple[int, int, int] = (128, 100, 8000),
    bin_ms: float = 5.0,
    pyramid_levels: int = 4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """完整音频 -> 显著性组件"""
    if y.ndim != 1:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    e_resp, _, _, _ = scm(y.astype(np.float32), sr, chan_spec, bin_ms, False)
    inhibited = lateral_inhibition(e_resp, radius=1)
    saliency, components = kalinli_saliency_map(inhibited, levels=pyramid_levels)
    return saliency, components


def smoke_test(iterations: int = 3) -> None:
    """简单的噪声对比测试"""
    _ = iterations
    rng = np.random.default_rng(0)
    base = rng.normal(size=16000).astype(np.float32)
    sal_base, _ = compute_kalinli_saliency_from_audio(base)
    boosted = base.copy()
    boosted[2000:2500] += 6.0
    sal_boosted, _ = compute_kalinli_saliency_from_audio(boosted)
    diff = sal_boosted - sal_base
    if np.max(np.abs(diff)) <= 1e-6:
        raise RuntimeError("Kalinli 显著性烟雾测试失败：添加事件未引入可检测差异")


