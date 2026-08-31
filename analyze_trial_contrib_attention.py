#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于“贡献曲线”的trial级分析（按模型左右得分分组）：
- 对每个trial：
  - 提取(2,T)响度 → 双流注意力模型前向 → 得到 left_proj(t), right_proj(t), 左右注意力 wL(t), wR(t)
  - 计算每帧线性贡献：
      left_contrib(t)  = wL(t) * (wL_vec · left_proj(t))
      right_contrib(t) = wR(t) * (wR_vec · right_proj(t))
  - 计算左右总分：left_score = wL_vec · (Σ wL(t)left_proj(t))，right_score 同理
  - 按 left_score vs right_score 分到 Score-Left / Score-Right 组
- 对每组分别求“左/右贡献曲线”的平均；同时提供“联合归一化（按总绝对贡献）”版本便于比较：
      norm_contrib = contrib / (Σ|left_contrib| + Σ|right_contrib|)

输出目录 attention_loudness_results：
- 原始平均贡献：
  attn_contrib_scoreLeft_left_raw.npy / ..._right_raw.npy
  attn_contrib_scoreRight_left_raw.npy / ..._right_raw.npy
- 归一化平均贡献（按总绝对贡献）：
  attn_contrib_scoreLeft_left_norm.npy / ...
  attn_contrib_scoreRight_left_norm.npy / ...
- 热力图与曲线：
  *_heatmap_*.png, *_curves.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from linear_regression_loudness_models import (
    load_all_human_decisions,
    parse_stereo_pair_name,
    load_mono_5s,
    extract_loudness_time_series,
)
from attention_loudness_model import DualStreamAttention

SR = 16000
HOP = 256
RESULT_DIR = os.path.join(os.getcwd(), 'attention_loudness_results')
BASE_DIR = r"D:\D\research\audioset下载\clap_select"


def get_trials():
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt",
    ]
    csv_files = [
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai2.csv",
        r"D:\D\research\audioset下载\experiment_output2\lironghua.csv",
        r"D:\D\research\audioset下载\experiment_output2\lironghua2.csv",
        r"D:\D\research\audioset下载\experiment_output2\mayunmiao_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\mayunmiao2.csv",
        r"D:\D\research\audioset下载\experiment_output2\ShangZiyang.csv",
        r"D:\D\research\audioset下载\experiment_output2\ShangZiyang1.csv",
        r"D:\D\research\audioset下载\experiment_output2\wjy1.csv",
        r"D:\D\research\audioset下载\experiment_output2\wjy_2.csv",
        r"D:\D\research\audioset下载\experiment_output2\LiuYaorui_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\Liu Yaorui2.csv",
    ]
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
    ] * 9
    return load_all_human_decisions(txt_files, csv_files, mapping_files)


def build_time_series_for_trial(pair_name):
    left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, BASE_DIR)
    left_path = os.path.join(BASE_DIR, left_cat, f"{left_id}.wav")
    right_path = os.path.join(BASE_DIR, right_cat, f"{right_id}.wav")
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        raise FileNotFoundError(f"找不到音频: {left_path} 或 {right_path}")
    yL, sr = load_mono_5s(left_path)
    yR, _ = load_mono_5s(right_path)
    loud_L = extract_loudness_time_series(yL, sr=sr, hop_length=HOP)
    loud_R = extract_loudness_time_series(yR, sr=sr, hop_length=HOP)
    T = min(len(loud_L), len(loud_R))
    x = np.stack([loud_L[:T], loud_R[:T]], axis=0).astype(np.float32)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return x  # (2,T)


def per_trial_contrib(model: DualStreamAttention, x_np: np.ndarray, T_ref: int, device):
    x_np = x_np[:, :T_ref]
    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,2,T)

    # 分离左右耳到 (1,T,1)
    left_ear = xb[:, 0, :].unsqueeze(-1)
    right_ear = xb[:, 1, :].unsqueeze(-1)

    # 投影 (1,T,dim)
    left_proj = torch.tanh(model.left_proj(left_ear))
    right_proj = torch.tanh(model.right_proj(right_ear))

    # 注意力权重 (1,T)
    left_scores = model.left_attention(left_proj).squeeze(-1)
    right_scores = model.right_attention(right_proj).squeeze(-1)
    wL = torch.softmax(left_scores, dim=-1)
    wR = torch.softmax(right_scores, dim=-1)

    # 分类器权重拆分
    W = model.classifier.weight  # (1,2*dim)
    dim = left_proj.shape[-1]
    wL_vec = W[:, :dim].transpose(0,1)   # (dim,1)
    wR_vec = W[:, dim:].transpose(0,1)   # (dim,1)

    # 每帧线性贡献 (1,T)
    # (1,T,dim) · (dim,1) -> (1,T,1) -> squeeze -> (T,)
    left_lin = torch.matmul(left_proj, wL_vec).squeeze(-1)
    right_lin = torch.matmul(right_proj, wR_vec).squeeze(-1)
    left_contrib = (wL * left_lin).squeeze(0).cpu().numpy()
    right_contrib = (wR * right_lin).squeeze(0).cpu().numpy()

    # 总分用于分组
    left_score = left_lin.mul(wL).sum().item()
    right_score = right_lin.mul(wR).sum().item()

    # 归一化（按总绝对贡献）
    denom = np.sum(np.abs(left_contrib)) + np.sum(np.abs(right_contrib))
    if denom <= 0:
        left_norm = np.zeros_like(left_contrib)
        right_norm = np.zeros_like(right_contrib)
    else:
        left_norm = left_contrib / denom
        right_norm = right_contrib / denom

    return left_contrib, right_contrib, left_norm, right_norm, left_score, right_score


def plot_heatmap(vec, title, path):
    plt.figure(figsize=(12, 2.8))
    sns.heatmap(vec[np.newaxis, :], cmap='coolwarm', center=0.0, cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel('Time (frames ≈ 16ms each)')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print('=== Trial-level contribution curves grouped by model score ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULT_DIR, exist_ok=True)

    trials = get_trials()
    if not trials:
        raise RuntimeError('未读取到trial。')

    # 确定统一T
    T_ref = None
    for t in trials:
        try:
            x = build_time_series_for_trial(t['original_name'])
            T_ref = x.shape[1] if T_ref is None else min(T_ref, x.shape[1])
        except Exception:
            continue
    if T_ref is None:
        raise RuntimeError('无法确定T。')

    # 加载模型
    model = DualStreamAttention(time_steps=T_ref, dim=64).to(device)
    state_path = os.path.join(RESULT_DIR, 'attention_model.pth')
    if not os.path.exists(state_path):
        raise FileNotFoundError(f'未找到模型参数: {state_path}')
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    # 累积四个分组（raw与norm）
    SL_L_raw, SL_R_raw, SR_L_raw, SR_R_raw = [], [], [], []
    SL_L_norm, SL_R_norm, SR_L_norm, SR_R_norm = [], [], [], []

    with torch.no_grad():
        for t in trials:
            try:
                x = build_time_series_for_trial(t['original_name'])
                L_raw, R_raw, L_norm, R_norm, sL, sR = per_trial_contrib(model, x, T_ref, device)
                if sL >= sR:
                    SL_L_raw.append(L_raw); SL_R_raw.append(R_raw)
                    SL_L_norm.append(L_norm); SL_R_norm.append(R_norm)
                else:
                    SR_L_raw.append(L_raw); SR_R_raw.append(R_raw)
                    SR_L_norm.append(L_norm); SR_R_norm.append(R_norm)
            except Exception:
                continue

    def mean_list(lst):
        return np.nanmean(np.stack(lst, axis=0), axis=0) if lst else np.full(T_ref, np.nan)

    # 组平均
    sl_l_raw = mean_list(SL_L_raw); sl_r_raw = mean_list(SL_R_raw)
    sr_l_raw = mean_list(SR_L_raw); sr_r_raw = mean_list(SR_R_raw)
    sl_l_norm = mean_list(SL_L_norm); sl_r_norm = mean_list(SL_R_norm)
    sr_l_norm = mean_list(SR_L_norm); sr_r_norm = mean_list(SR_R_norm)

    # 保存npy
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreLeft_left_raw.npy'), sl_l_raw)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreLeft_right_raw.npy'), sl_r_raw)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreRight_left_raw.npy'), sr_l_raw)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreRight_right_raw.npy'), sr_r_raw)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreLeft_left_norm.npy'), sl_l_norm)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreLeft_right_norm.npy'), sl_r_norm)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreRight_left_norm.npy'), sr_l_norm)
    np.save(os.path.join(RESULT_DIR, 'contrib_scoreRight_right_norm.npy'), sr_r_norm)

    # 热力图（以0为中心显示正负）
    plot_heatmap(sl_l_raw,  'Contrib (Raw) Score-Left / Left Ear',  os.path.join(RESULT_DIR, 'contrib_heatmap_scoreLeft_left_raw.png'))
    plot_heatmap(sl_r_raw,  'Contrib (Raw) Score-Left / Right Ear', os.path.join(RESULT_DIR, 'contrib_heatmap_scoreLeft_right_raw.png'))
    plot_heatmap(sr_l_raw,  'Contrib (Raw) Score-Right / Left Ear', os.path.join(RESULT_DIR, 'contrib_heatmap_scoreRight_left_raw.png'))
    plot_heatmap(sr_r_raw,  'Contrib (Raw) Score-Right / Right Ear',os.path.join(RESULT_DIR, 'contrib_heatmap_scoreRight_right_raw.png'))

    plot_heatmap(sl_l_norm, 'Contrib (Norm) Score-Left / Left Ear',  os.path.join(RESULT_DIR, 'contrib_heatmap_scoreLeft_left_norm.png'))
    plot_heatmap(sl_r_norm, 'Contrib (Norm) Score-Left / Right Ear', os.path.join(RESULT_DIR, 'contrib_heatmap_scoreLeft_right_norm.png'))
    plot_heatmap(sr_l_norm, 'Contrib (Norm) Score-Right / Left Ear', os.path.join(RESULT_DIR, 'contrib_heatmap_scoreRight_left_norm.png'))
    plot_heatmap(sr_r_norm, 'Contrib (Norm) Score-Right / Right Ear',os.path.join(RESULT_DIR, 'contrib_heatmap_scoreRight_right_norm.png'))

    # 叠加曲线（raw 与 norm 各一张）
    t_axis = np.arange(T_ref) * (HOP / SR)
    def plot_curves(a,b,c,d, title, fname):
        plt.figure(figsize=(12, 3.2))
        plt.plot(t_axis, a, label='Score-Left / LeftEar',  color='red')
        plt.plot(t_axis, b, label='Score-Left / RightEar', color='orange')
        plt.plot(t_axis, c, label='Score-Right / LeftEar', color='blue')
        plt.plot(t_axis, d, label='Score-Right / RightEar',color='green')
        plt.xlabel('Time (s)'); plt.ylabel('Contribution')
        plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR, fname), dpi=300, bbox_inches='tight'); plt.close()

    plot_curves(sl_l_raw, sl_r_raw, sr_l_raw, sr_r_raw, 'Contribution Curves (Raw)',  'contrib_curves_raw.png')
    plot_curves(sl_l_norm, sl_r_norm, sr_l_norm, sr_r_norm, 'Contribution Curves (Normalized by |total|)', 'contrib_curves_norm.png')

    print('已输出 贡献曲线（raw与norm） 的npy/图到:', RESULT_DIR)


if __name__ == '__main__':
    main()
