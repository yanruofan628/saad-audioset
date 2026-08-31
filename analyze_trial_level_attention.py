#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按trial级进行前向推理（基于模型打分分组）：
- 对每个trial提取(2, T)响度时间序列，使用已训练的双流注意力模型前向，得到左/右注意力与左右上下文
- 使用分类器权重分解logit贡献：logit = wL·left_context + wR·right_context + b
- 以 (wL·left_context) 与 (wR·right_context) 大小比较，决定该trial归入“左侧得分更大”或“右侧得分更大”分组
- 分别对组内的左耳/右耳注意力取平均，输出四条平均注意力曲线与热力图：
  - score_left_bigger：左耳/右耳
  - score_right_bigger：左耳/右耳
输出目录：attention_loudness_results
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

# 复用已有的数据与特征函数
from linear_regression_loudness_models import (
    load_all_human_decisions,
    parse_stereo_pair_name,
    load_mono_5s,
    extract_loudness_time_series,
)
# 复用已实现的双流注意力模型结构
from attention_loudness_model import DualStreamAttention

SR = 16000
HOP = 256  # 16ms/帧
RESULT_DIR = os.path.join(os.getcwd(), 'attention_loudness_results')
BASE_DIR = r"D:\D\research\audioset下载\clap_select"


def get_trials():
    # 与训练阶段一致的trial来源
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

    trials = load_all_human_decisions(txt_files, csv_files, mapping_files)
    # 每条trial: {'subject_id', 'original_name', 'response' (1=left, 2=right)}
    return trials


def build_time_series_for_trial(pair_name):
    # 通过文件系统解析左右wav实际路径
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
    # 标准化到每通道零均值单位方差
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return x  # (2, T)


def plot_and_save(vec, title, out_png):
    plt.figure(figsize=(12, 2.8))
    sns.heatmap(vec[np.newaxis, :], cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel('Time (frames ≈ 16ms each)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def compute_context_and_attn(model: DualStreamAttention, x_np: np.ndarray, T_ref: int, device):
    # x_np: (2,T)
    x_np = x_np[:, :T_ref]
    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,2,T)
    # 手动复现前向，得到上下文
    # 分离左右耳
    left_ear = xb[:, 0, :].unsqueeze(-1)   # (1,T,1)
    right_ear = xb[:, 1, :].unsqueeze(-1)  # (1,T,1)
    # 投影
    left_proj = torch.tanh(model.left_proj(left_ear))     # (1,T,dim)
    right_proj = torch.tanh(model.right_proj(right_ear))  # (1,T,dim)
    # 注意力权重
    left_scores = model.left_attention(left_proj).squeeze(-1)   # (1,T)
    right_scores = model.right_attention(right_proj).squeeze(-1) # (1,T)
    left_weights = torch.softmax(left_scores, dim=-1)    # (1,T)
    right_weights = torch.softmax(right_scores, dim=-1)  # (1,T)
    # 上下文
    left_context = torch.sum(left_proj * left_weights.unsqueeze(-1), dim=1)   # (1,dim)
    right_context = torch.sum(right_proj * right_weights.unsqueeze(-1), dim=1) # (1,dim)
    # 分类器权重分块
    W = model.classifier.weight  # (1, 2*dim)
    b = model.classifier.bias    # (1,)
    dim = left_context.shape[-1]
    wL = W[:, :dim]   # (1,dim)
    wR = W[:, dim:]   # (1,dim)
    # 贡献分解
    left_score = torch.sum(wL * left_context, dim=1)   # (1,)
    right_score = torch.sum(wR * right_context, dim=1) # (1,)
    # 同时返回注意力权重
    return (
        left_context.squeeze(0).cpu().numpy(),
        right_context.squeeze(0).cpu().numpy(),
        left_weights.squeeze(0).cpu().numpy(),
        right_weights.squeeze(0).cpu().numpy(),
        left_score.item(),
        right_score.item(),
    )


def main():
    print("=== Trial-level attention grouping by model score (left vs right) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1) 读取全部trial
    trials = get_trials()
    if len(trials) == 0:
        raise RuntimeError('未读取到trial。')

    # 2) 先扫一遍确定统一T
    T_ref = None
    for t in trials:
        try:
            x = build_time_series_for_trial(t['original_name'])
            T_ref = x.shape[1] if T_ref is None else min(T_ref, x.shape[1])
        except Exception:
            continue
    if T_ref is None:
        raise RuntimeError('无法确定时间长度T。')

    # 3) 构建模型并加载权重
    model = DualStreamAttention(time_steps=T_ref, dim=64).to(device)
    state_path = os.path.join(RESULT_DIR, 'attention_model.pth')
    if not os.path.exists(state_path):
        raise FileNotFoundError(f'未找到模型参数: {state_path}')
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    # 4) 按“模型打分左右更大”分组累积注意力
    Lb_left_list, Lb_right_list = [], []   # 左比分组：左耳/右耳
    Rb_left_list, Rb_right_list = [], []   # 右比分组：左耳/右耳

    with torch.no_grad():
        for t in trials:
            try:
                x = build_time_series_for_trial(t['original_name'])
                (cL, cR, wL, wR, sL, sR) = compute_context_and_attn(model, x, T_ref, device)
                if sL >= sR:
                    Lb_left_list.append(wL)
                    Lb_right_list.append(wR)
                else:
                    Rb_left_list.append(wL)
                    Rb_right_list.append(wR)
            except Exception:
                continue

    def mean_or_nan(lst):
        return np.nanmean(np.stack(lst, axis=0), axis=0) if len(lst) else np.full(T_ref, np.nan)

    Lb_left = mean_or_nan(Lb_left_list)
    Lb_right = mean_or_nan(Lb_right_list)
    Rb_left = mean_or_nan(Rb_left_list)
    Rb_right = mean_or_nan(Rb_right_list)

    # 5) 保存npy
    np.save(os.path.join(RESULT_DIR, 'attn_mean_scoreLeft_leftEar.npy'), Lb_left)
    np.save(os.path.join(RESULT_DIR, 'attn_mean_scoreLeft_rightEar.npy'), Lb_right)
    np.save(os.path.join(RESULT_DIR, 'attn_mean_scoreRight_leftEar.npy'), Rb_left)
    np.save(os.path.join(RESULT_DIR, 'attn_mean_scoreRight_rightEar.npy'), Rb_right)

    # 6) 热力图
    def plot(vec, title, fname):
        plt.figure(figsize=(12, 2.8))
        sns.heatmap(vec[np.newaxis, :], cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.xlabel('Time (frames ≈ 16ms each)')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()

    plot(Lb_left,  'Average Attention (Score-Left / Left Ear)',  'attn_heatmap_scoreLeft_leftEar.png')
    plot(Lb_right, 'Average Attention (Score-Left / Right Ear)', 'attn_heatmap_scoreLeft_rightEar.png')
    plot(Rb_left,  'Average Attention (Score-Right / Left Ear)', 'attn_heatmap_scoreRight_leftEar.png')
    plot(Rb_right, 'Average Attention (Score-Right / Right Ear)','attn_heatmap_scoreRight_rightEar.png')

    # 7) 叠加曲线对比图（四条）
    time_axis = np.arange(T_ref) * (HOP / SR)
    plt.figure(figsize=(12, 3.2))
    plt.plot(time_axis, Lb_left,  label='Score-Left / LeftEar',  color='red')
    plt.plot(time_axis, Lb_right, label='Score-Left / RightEar', color='orange')
    plt.plot(time_axis, Rb_left,  label='Score-Right / LeftEar', color='blue')
    plt.plot(time_axis, Rb_right, label='Score-Right / RightEar',color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Attention')
    plt.title('Average Attention by Higher Model Score (Left vs Right)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'attn_curves_by_model_score.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print('已输出 基于模型左右得分分组 的注意力npy/图到:', RESULT_DIR)


if __name__ == '__main__':
    main()
