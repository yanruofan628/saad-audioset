#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 EEG 残差的多模态分类模型（baseline + cross-attention 变体）

本文件最初只实现“音频 + EEG 残差”相关的模型类。
现在增加一个可直接运行的入口：调用原 `eeg_audio_residual.py` 的数据与交叉验证流程，
但只训练 / 对比三个模型：
    - residual: 纯 EEG 残差 baseline（本文件中的 ResidualEEGFusionClassifier）
    - residual_eeg_cross_single: 单 query cross-attention
    - residual_eeg_cross_dual: 双 query cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from linear_regression_loudness_models import DEFAULT_AUDIO_BASE_DIR, parse_stereo_pair_name
import os
import librosa
import hashlib


class EEGNetFeatureExtractorWithSeq(nn.Module):
    """
    EEGNet 特征提取器的变体：
    - 支持返回 poolmean 后的 160 维特征（与原 EEGNetFeatureExtractor 一致）
    - 也可以返回卷积后但未在时间维上池化的时序特征序列，用于后续 cross-attention
    - 还可以返回跨特征维度平均的功率时间序列，用于与语音包络对齐的 loss
    """

    def __init__(self, n_channels: int, n_samples: int):
        super().__init__()
        # 与 eeg_audio_residual.EEGNetFeatureExtractor 中保持一致的结构
        self.conv_time = nn.Conv2d(1, 20, (1, 41), stride=(1, 1), bias=False)
        self.conv_ica = nn.Conv2d(1, 8, (n_channels, 1), stride=(1, 1), bias=False)
        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5)
        self.poolmean = nn.AdaptiveAvgPool2d((8, 1))  # 输出 (batch, 20, 8, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_seq: bool = False,
        return_power: bool = False,
    ) -> torch.Tensor:
        """
        x: (batch, 1, n_channels, n_timepoints)

        - 如果 return_seq=False 且 return_power=False:
            返回 (batch, 160) 的全局 EEG 特征（与原 EEGNet 相同）
        - 如果 return_seq=True:
            返回 (batch, T_eeg, 160) 的时序特征序列
        - 如果 return_power=True:
            返回 (batch, T_eeg) 的功率时间序列 p(t)
        - 如果二者都为 True:
            返回 (x_seq, p_series)
        """
        # conv_ica: 空间滤波
        x = self.conv_ica(x)  # (B, 8, 1, T_raw)
        x = F.dropout(x, 0.15, training=self.training)

        # 交换维度以便在时间维上做卷积
        x = torch.permute(x, (0, 2, 1, 3))  # (B, 1, 8, T_raw)

        # conv_time: 时间卷积
        x = self.conv_time(x)  # (B, 20, 8, T_conv)
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)

        # ERDS 非线性（功率）
        x = torch.mul(x, x)  # (B, 20, 8, T_conv)

        if return_seq or return_power:
            # 当前形状 (B, 20, 8, T_conv)，将通道和频带展平为特征维，时间作为序列长度
            b, c, f, t = x.shape
            x_seq = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T_conv, 160)
            # 功率时间序列：跨特征维度取均值
            p_series = x_seq.mean(dim=2)  # (B, T_conv)

            if return_seq and return_power:
                return x_seq, p_series
            elif return_seq:
                return x_seq
            else:
                return p_series

        # 否则，保持与原 EEGNet 一致的池化与展平行为
        x = self.poolmean(x)  # (B, 20, 8, 1)
        x = x.view(x.size(0), -1)  # (B, 160)
        return x


class ResidualEEGFusionClassifier(nn.Module):
    """
    仅包含“音频基础决策 + EEG 残差”的 baseline 模型（不含同步性分支）

    结构：
    - audio_head(audio_features) → logits_audio
    - eeg_extractor(eeg_data) → eeg_feat
    - residual_head([eeg_feat, audio_features]) → delta_eeg
    - logits = logits_audio + delta_eeg
    """

    def __init__(self, eeg_channels, eeg_samples, audio_dim, hidden_dim=64, n_classes=2):
        super().__init__()

        # EEG 特征提取器：只用全局 160 维特征
        self.eeg_extractor = EEGNetFeatureExtractorWithSeq(eeg_channels, eeg_samples)

        # 音频分支：与 eeg_audio_residual.ResidualFusionClassifier 中保持一致
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # EEG 残差分支
        total_dim = 160 + audio_dim
        self.residual_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, eeg_data, audio_features, return_parts: bool = False):
        """
        eeg_data: (batch, 1, n_channels, n_timepoints)
        audio_features: (batch, audio_dim)
        如果 return_parts=True，则返回 (logits_final, logits_audio, delta_eeg)
        """
        # EEG 全局特征
        eeg_feat = self.eeg_extractor(eeg_data, return_seq=False)  # (B, 160)

        # 音频基础决策
        logits_audio = self.audio_head(audio_features)  # (B, n_classes)

        # 残差预测
        combined = torch.cat([eeg_feat, audio_features], dim=1)  # (B, 160+audio_dim)
        delta_eeg = self.residual_head(combined)  # (B, n_classes)

        logits_final = logits_audio + delta_eeg

        if return_parts:
            return logits_final, logits_audio, delta_eeg
        return logits_final


class ResidualEEGSharedSpaceClassifier(nn.Module):
    """
    残差仍然是 EEG-only：
        logits = logits_audio + delta_eeg
    audio–EEG 的交互只通过共享子空间 embedding 的对齐 loss 完成：
        u = f_eeg(eeg_feat), v = f_audio(audio_features)
        训练时额外加入 L_align = ||u - v||^2
    """

    def __init__(self, eeg_channels, eeg_samples, audio_dim,
                 hidden_dim=64, embed_dim=32, n_classes=2):
        super().__init__()

        # EEG 特征提取器：与 ResidualEEGFusionClassifier 一致
        self.eeg_extractor = EEGNetFeatureExtractorWithSeq(eeg_channels, eeg_samples)

        # 音频基础分支（baseline）
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # EEG-only 残差头（不 concat audio）
        self.residual_head = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

        # 共享子空间 embedding 头（只用于 loss）
        self.eeg_embed = nn.Sequential(
            nn.Linear(160, embed_dim),
            nn.ReLU(),
        )
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, eeg_data, audio_features,
                return_parts: bool = False,
                return_embeds: bool = False):
        # EEG 特征 (B, 160)
        eeg_feat = self.eeg_extractor(eeg_data, return_seq=False)

        # Audio baseline logits
        logits_audio = self.audio_head(audio_features)  # (B, n_classes)

        # EEG-only 残差
        delta_eeg = self.residual_head(eeg_feat)        # (B, n_classes)
        logits_final = logits_audio + delta_eeg

        outputs = [logits_final]
        if return_parts:
            outputs += [logits_audio, delta_eeg]
        if return_embeds:
            u = self.eeg_embed(eeg_feat)               # (B, embed_dim)
            v = self.audio_embed(audio_features)       # (B, embed_dim)
            outputs += [u, v]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class ResidualEEGEnvAlignClassifier(nn.Module):
    """
    残差结构：
        logits = logits_audio + delta_eeg   (delta 只看 EEG)
    额外 loss：
        L_env: EEG 功率时间序列 p(t) 与 语音包络 e(t) 对齐
    """

    def __init__(self, eeg_channels, eeg_samples, audio_dim,
                 hidden_dim=64, n_classes=2):
        super().__init__()
        self.eeg_extractor = EEGNetFeatureExtractorWithSeq(eeg_channels, eeg_samples)

        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # EEG-only 残差头
        self.residual_head = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, eeg_data, audio_features,
                return_parts: bool = False,
                return_power: bool = False):
        """
        如果 return_power=True，则同时返回功率时间序列 p_series (B,T_eeg)
        """
        if return_power:
            eeg_seq, p_series = self.eeg_extractor(
                eeg_data,
                return_seq=True,
                return_power=True
            )  # eeg_seq: (B,T,160), p_series: (B,T)
            # 简单做时间平均得到全局 EEG 特征（也可以换成 poolmean）
            eeg_feat = eeg_seq.mean(dim=1)  # (B,160)
        else:
            eeg_feat = self.eeg_extractor(eeg_data, return_seq=False)
            p_series = None

        logits_audio = self.audio_head(audio_features)
        delta_eeg = self.residual_head(eeg_feat)
        logits_final = logits_audio + delta_eeg

        outputs = [logits_final]
        if return_parts:
            outputs += [logits_audio, delta_eeg]
        if return_power:
            outputs += [p_series]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class ResidualEEGFusionCrossAttnSingleQuery(nn.Module):
    """
    方案 A：单一全局 audio query → EEG 时间序列 cross-attention 的残差融合模型

    - EEG 分支：使用 EEGNetFeatureExtractorWithSeq 提取时序特征序列 eeg_seq ∈ (B, T_eeg, 160)
    - Audio 分支：audio_features 先过 audio_head 得到 logits_audio，再线性映射为 query 向量 q ∈ (B, 1, 160)
    - Cross-attention：以 q 为 query，eeg_seq 为 key/value，经 MultiheadAttention 得到加权 EEG 表征 eeg_attn ∈ (B, 160)
    - Residual：使用 [eeg_attn, audio_features] 预测 delta_eeg，最终 logits = logits_audio + delta_eeg
    """

    def __init__(
        self,
        eeg_channels,
        eeg_samples,
        audio_dim,
        hidden_dim=64,
        n_classes=2,
        num_heads=4,
    ):
        super().__init__()

        self.eeg_extractor = EEGNetFeatureExtractorWithSeq(eeg_channels, eeg_samples)

        # 音频基础分支
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # 将全局 audio 特征映射为单一 query（与 EEG 特征同维度）
        self.query_proj = nn.Linear(audio_dim, 160)

        # 跨模态 Multi-Head Attention，batch_first=True 方便使用 (B, T, D) 形状
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=160,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # LayerNorm 稳定注意力输出
        self.attn_norm = nn.LayerNorm(160)

        # 残差头：输入为注意力后的 EEG 表征 + 原始音频特征
        total_dim = 160 + audio_dim
        self.residual_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, eeg_data, audio_features, return_parts: bool = False, return_attn: bool = False):
        """
        eeg_data: (B, 1, n_channels, n_timepoints)
        audio_features: (B, audio_dim)

        如果 return_parts=True，则返回 (logits_final, logits_audio, delta_eeg)
        如果 return_attn=True，则额外返回 attention 权重 (B, T_eeg)
        """
        # Layer 1: 音频基础决策
        logits_audio = self.audio_head(audio_features)  # (B, n_classes)

        # EEG 时序特征
        eeg_seq = self.eeg_extractor(eeg_data, return_seq=True)  # (B, T_eeg, 160)

        # 构造单一 query
        q = self.query_proj(audio_features).unsqueeze(1)  # (B, 1, 160)

        # Cross-attention: Q=(B,1,160), K/V=(B,T,160)
        attn_output, attn_weights = self.cross_attn(q, eeg_seq, eeg_seq)  # 输出 (B,1,160), (B,1,T)
        attn_output = attn_output.squeeze(1)  # (B,160)
        attn_output = self.attn_norm(attn_output)

        # EEG 残差
        combined = torch.cat([attn_output, audio_features], dim=1)  # (B,160+audio_dim)
        delta_eeg = self.residual_head(combined)  # (B,n_classes)

        logits_final = logits_audio + delta_eeg

        outputs = (logits_final,)
        if return_parts:
            outputs += (logits_audio, delta_eeg)
        if return_attn:
            # 将 (B,1,T) 压缩为 (B,T)
            time_attn = attn_weights.squeeze(1)
            outputs += (time_attn,)

        if len(outputs) == 1:
            return outputs[0]
        return outputs


class ResidualEEGFusionCrossAttnDualQuery(nn.Module):
    """
    方案 B：左右耳分别作为 query 的 dual-query EEG cross-attention 残差融合模型

    假设上游能够提供左右耳音频特征 left_feat, right_feat ∈ (B, d_ear)。
    如果只提供 audio_features ∈ (B, audio_dim)，可在外部拆分/映射为左右耳特征后再传入。
    """

    def __init__(
        self,
        eeg_channels,
        eeg_samples,
        ear_feature_dim,
        audio_dim,
        hidden_dim=64,
        n_classes=2,
        num_heads=4,
    ):
        super().__init__()

        self.eeg_extractor = EEGNetFeatureExtractorWithSeq(eeg_channels, eeg_samples)

        # 音频基础分支（使用全局 audio_features，而不是左右特征）
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # 左右耳各自映射为一个 query
        self.query_proj_left = nn.Linear(ear_feature_dim, 160)
        self.query_proj_right = nn.Linear(ear_feature_dim, 160)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=160,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(160)

        # 将左右耳的注意力融合后的 EEG 表征拼接或压缩
        dual_eeg_dim = 2 * 160
        self.dual_eeg_fuser = nn.Sequential(
            nn.Linear(dual_eeg_dim, 160),
            nn.ReLU(),
        )

        # 残差头：融合 dual EEG 表征与全局音频特征
        total_dim = 160 + audio_dim
        self.residual_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        eeg_data,
        audio_features,
        left_feat,
        right_feat,
        return_parts: bool = False,
        return_attn: bool = False,
    ):
        """
        eeg_data: (B, 1, n_channels, n_timepoints)
        audio_features: (B, audio_dim) - 全局音频特征
        left_feat, right_feat: (B, ear_feature_dim) - 左右耳特征
        """
        # Layer 1: 音频基础决策
        logits_audio = self.audio_head(audio_features)  # (B,n_classes)

        # EEG 时序特征
        eeg_seq = self.eeg_extractor(eeg_data, return_seq=True)  # (B,T_eeg,160)

        # 左右耳 query
        q_left = self.query_proj_left(left_feat).unsqueeze(1)   # (B,1,160)
        q_right = self.query_proj_right(right_feat).unsqueeze(1)  # (B,1,160)

        # 分别对 EEG 序列做 cross-attn
        out_left, attn_left = self.cross_attn(q_left, eeg_seq, eeg_seq)   # (B,1,160), (B,1,T)
        out_right, attn_right = self.cross_attn(q_right, eeg_seq, eeg_seq)  # (B,1,160), (B,1,T)

        out_left = self.attn_norm(out_left.squeeze(1))    # (B,160)
        out_right = self.attn_norm(out_right.squeeze(1))  # (B,160)

        # 左右耳注意力后的 EEG 表征融合
        dual_eeg = torch.cat([out_left, out_right], dim=1)  # (B,320)
        dual_eeg = self.dual_eeg_fuser(dual_eeg)            # (B,160)

        # EEG 残差
        combined = torch.cat([dual_eeg, audio_features], dim=1)  # (B,160+audio_dim)
        delta_eeg = self.residual_head(combined)  # (B,n_classes)

        logits_final = logits_audio + delta_eeg

        outputs = (logits_final,)
        if return_parts:
            outputs += (logits_audio, delta_eeg)
        if return_attn:
            # 返回左右耳各自的时间注意力 (B,T)
            outputs += (attn_left.squeeze(1), attn_right.squeeze(1))

        if len(outputs) == 1:
            return outputs[0]
        return outputs


def train_fold_model_residual_and_attn(fold_data, all_audio_features, pair_name_to_idx, pair_names, all_sync_features=None):
    """
    复制自 eeg_audio_residual.train_fold_model，但只保留三个模型：
    - residual: ResidualEEGFusionClassifier
    - residual_eeg_cross_single: ResidualEEGFusionCrossAttnSingleQuery
    - residual_eeg_cross_dual: ResidualEEGFusionCrossAttnDualQuery

    其余逻辑（数据拆分、训练、early stopping、指标计算）保持一致，便于公平对比。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据
    train_eeg_all = fold_data['train_eeg']
    train_labels_all = fold_data['train_labels']
    train_pair_indices = fold_data['train_pair_indices']
    train_global_indices = fold_data.get('train_global_indices', None)

    val_eeg_all = fold_data['val_eeg']
    val_labels_all = fold_data['val_labels']
    val_pair_indices = fold_data['val_pair_indices']
    val_global_indices = fold_data.get('val_global_indices', None)
    val_subjects_all = fold_data['val_subjects']  # 用于计算各被试个人准确率

    # 标签检查（与原脚本保持一致）
    for arr, name in [(train_labels_all, 'train_labels'), (val_labels_all, 'val_labels')]:
        unq = np.unique(arr)
        if not (np.all(np.isin(unq, [0, 1])) and len(unq) >= 1):
            raise ValueError(
                f"标签 {name} 含有非 0/1 的值: {unq.tolist()}。"
                "请检查 choice/response 是否为 1 或 2，并已转换为 0/1。"
            )

    # 选择音频特征
    X_audio_train = all_audio_features[train_pair_indices]
    X_audio_val = all_audio_features[val_pair_indices]

    # 同步性特征在本对比中不使用，但保留接口以兼容调用
    X_sync_train = None
    X_sync_val = None
    use_sync = False

    # 模型输入维度
    eeg_channels = train_eeg_all.shape[1]
    eeg_samples = train_eeg_all.shape[2]
    audio_dim = X_audio_train.shape[1]

    # 预计算或加载所有 pair 的语音包络，用于 env 对齐 loss
    # 使用与 all_audio_features 相同的 pair_names 顺序
    project_root = os.getcwd()
    cache_dir = os.path.join(project_root, 'audio_features_cache')
    os.makedirs(cache_dir, exist_ok=True)
    pair_names_str = ','.join(sorted(pair_names))
    pair_names_hash = hashlib.md5(pair_names_str.encode()).hexdigest()[:8]
    env_cache_path = os.path.join(cache_dir, f'all_audio_envelopes_{pair_names_hash}.npy')

    if os.path.exists(env_cache_path):
        all_audio_envelopes = np.load(env_cache_path)
        print(f"成功从缓存加载所有音频包络: {all_audio_envelopes.shape}")
    else:
        print(f"未发现音频包络缓存，开始为 {len(pair_names)} 个音频对计算包络...")
        envelopes = []
        # 使用 DEFAULT_AUDIO_BASE_DIR 和 parse_stereo_pair_name 找到左右耳音频路径
        for pname in pair_names:
            try:
                left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pname, DEFAULT_AUDIO_BASE_DIR)
                left_path = os.path.join(DEFAULT_AUDIO_BASE_DIR, left_cat, f"{left_id}.wav")
                right_path = os.path.join(DEFAULT_AUDIO_BASE_DIR, right_cat, f"{right_id}.wav")
                # 读取左右耳音频，简单平均作为整体信号
                y_l, sr_l = librosa.load(left_path, sr=None)
                y_r, sr_r = librosa.load(right_path, sr=None)
                sr = sr_l
                if sr_r != sr_l:
                    # 若采样率不一致，将右耳重采样
                    y_r = librosa.resample(y_r, orig_sr=sr_r, target_sr=sr_l)
                y = 0.5 * (y_l + y_r)
                # 使用短时能量作为包络近似
                frame_length = 512
                hop_length = 256
                rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]  # (T_env,)
                envelopes.append(rms.astype(np.float32))
            except Exception as e:
                print(f"计算包络失败 {pname}: {e}")
                envelopes.append(np.zeros(1, dtype=np.float32))

        # 对齐长度：取所有包络的最大长度，短的用零填充
        max_len = max(env.shape[0] for env in envelopes)
        env_array = np.zeros((len(envelopes), max_len), dtype=np.float32)
        for i, env in enumerate(envelopes):
            env_array[i, :env.shape[0]] = env

        all_audio_envelopes = env_array
        np.save(env_cache_path, all_audio_envelopes)
        print(f"音频包络已缓存到: {env_cache_path}, 形状: {all_audio_envelopes.shape}")

    # 依据当前 fold 提取训练 / 验证的包络
    E_audio_train = all_audio_envelopes[train_pair_indices]
    E_audio_val = all_audio_envelopes[val_pair_indices]

    # 这里只对比三个模型：
    # - residual: 原始 EEG 残差 baseline
    # - residual_eeg_shared: 共享子空间对齐 loss 的 EEG-only 残差模型
    # - residual_eeg_env: EEG 功率–语音包络对齐 loss 模型
    models = {
        'residual': ResidualEEGFusionClassifier(
            eeg_channels=eeg_channels,
            eeg_samples=eeg_samples,
            audio_dim=audio_dim
        ),
        'residual_eeg_shared': ResidualEEGSharedSpaceClassifier(
            eeg_channels=eeg_channels,
            eeg_samples=eeg_samples,
            audio_dim=audio_dim
        ),
        'residual_eeg_env': ResidualEEGEnvAlignClassifier(
            eeg_channels=eeg_channels,
            eeg_samples=eeg_samples,
            audio_dim=audio_dim
        ),
    }

    for m in models.values():
        m.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizers = {
        name: optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        for name, model in models.items()
    }

    # 训练按 batch；验证集全量前向
    n_train = len(train_labels_all)
    batch_size = 64
    n_batches = (n_train + batch_size - 1) // batch_size

    X_eeg_val = torch.FloatTensor(val_eeg_all).unsqueeze(1).to(device)
    X_audio_val_t = torch.FloatTensor(X_audio_val).to(device)
    y_val = torch.LongTensor(val_labels_all).to(device)

    n_epochs = 100
    patience = 15
    lambda_reg = 0.05
    lambda_align = 0.01
    lambda_env = 0.01

    best_states = {name: None for name in models.keys()}
    best_accs = {name: 0.0 for name in models.keys()}
    patience_counters = {name: 0 for name in models.keys()}
    last_train = {name: {'loss': 0.0, 'ce': 0.0, 'reg': 0.0, 'acc': 0.0} for name in models.keys()}

    for epoch in range(n_epochs):
        for name, model in models.items():
            model.train()
            epoch_loss = epoch_ce = epoch_reg = 0.0
            epoch_correct = epoch_total = 0

            for b in range(n_batches):
                s = b * batch_size
                e = min(s + batch_size, n_train)
                Xe = torch.FloatTensor(train_eeg_all[s:e]).unsqueeze(1).to(device)
                Xa = torch.FloatTensor(X_audio_train[s:e]).to(device)
                yb = torch.LongTensor(train_labels_all[s:e]).to(device)

                optimizers[name].zero_grad()

                if name == 'residual':
                    logits, _, delta = model(Xe, Xa, return_parts=True)
                    ce_loss = criterion(logits, yb)
                    dl2 = torch.norm(delta, p=2, dim=1).mean()
                    reg = dl2.item()
                    loss = ce_loss + lambda_reg * dl2
                    ce = ce_loss.item()
                elif name == 'residual_eeg_shared':
                    # 共享子空间模型：同时返回 embeddings 用于对齐 loss
                    logits, _, delta, u, v = model(
                        Xe, Xa,
                        return_parts=True,
                        return_embeds=True
                    )
                    ce_loss = criterion(logits, yb)
                    dl2 = torch.norm(delta, p=2, dim=1).mean()
                    align_loss = ((u - v) ** 2).mean()
                    reg = dl2.item() + align_loss.item()
                    loss = ce_loss + lambda_reg * dl2 + lambda_align * align_loss
                    ce = ce_loss.item()
                elif name == 'residual_eeg_env':
                    # env 模型：需要对应 batch 的语音包络
                    e_batch = torch.FloatTensor(E_audio_train[s:e]).to(device)  # (B, T_env_raw)
                    # forward: 同时返回功率时间序列
                    logits, _, delta, p_series = model(
                        Xe, Xa,
                        return_parts=True,
                        return_power=True
                    )
                    ce_loss = criterion(logits, yb)
                    dl2 = torch.norm(delta, p=2, dim=1).mean()

                    # 对 p_series 和 e_batch 做长度对齐：截断到较短长度
                    T_eeg = p_series.shape[1]
                    T_env = e_batch.shape[1]
                    T_min = min(T_eeg, T_env)
                    p_aligned = p_series[:, :T_min]
                    e_aligned = e_batch[:, :T_min]

                    # 标准化后计算 MSE 对齐 loss
                    p_norm = (p_aligned - p_aligned.mean(dim=1, keepdim=True)) / (
                        p_aligned.std(dim=1, keepdim=True) + 1e-6
                    )
                    e_norm = (e_aligned - e_aligned.mean(dim=1, keepdim=True)) / (
                        e_aligned.std(dim=1, keepdim=True) + 1e-6
                    )
                    env_loss = ((p_norm - e_norm) ** 2).mean()

                    reg = dl2.item() + env_loss.item()
                    loss = ce_loss + lambda_reg * dl2 + lambda_env * env_loss
                    ce = ce_loss.item()
                else:
                    logits = model(Xe, Xa)
                    loss = criterion(logits, yb)
                    ce, reg = loss.item(), 0.0

                loss.backward()
                optimizers[name].step()

                epoch_loss += loss.item()
                epoch_ce += ce
                epoch_reg += reg
                with torch.no_grad():
                    _, pred = torch.max(logits, 1)
                epoch_correct += (pred == yb).sum().item()
                epoch_total += yb.size(0)

            last_train[name]['loss'] = epoch_loss / n_batches
            last_train[name]['ce'] = epoch_ce / n_batches
            last_train[name]['reg'] = epoch_reg / n_batches
            last_train[name]['acc'] = epoch_correct / epoch_total if epoch_total else 0.0

        if (epoch + 1) % 10 == 0:
            for name, model in models.items():
                model.eval()
                with torch.no_grad():
                    val_out = model(X_eeg_val, X_audio_val_t)
                    val_loss = criterion(val_out, y_val).item()
                    _, val_pred = torch.max(val_out, 1)
                    val_pred_np = val_pred.cpu().numpy()
                val_acc = accuracy_score(val_labels_all, val_pred_np)

                if val_acc > best_accs[name]:
                    best_accs[name] = val_acc
                    best_states[name] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counters[name] = 0
                else:
                    patience_counters[name] += 1

                tl = last_train[name]['loss']
                tc = last_train[name]['ce']
                tr = last_train[name]['reg']
                ta = last_train[name]['acc']
                print(
                    f"    Epoch {epoch + 1}/{n_epochs} [{name:24s}] "
                    f"Train Loss: {tl:.4f} (Cls: {tc:.4f}, Reg: {tr:.4f}), "
                    f"Train Acc: {ta:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            if all(patience_counters[n] >= patience for n in models.keys()):
                break

    actual_epochs = epoch + 1
    print(f"    实际训练了 {actual_epochs} 个 epoch")

    for name, model in models.items():
        if best_states[name] is not None:
            model.load_state_dict(best_states[name])
        model.eval()

    metrics = {}
    with torch.no_grad():
        for name, model in models.items():
            val_out = model(X_eeg_val, X_audio_val_t)
            val_probs_np = F.softmax(val_out, dim=1).cpu().numpy()
            _, val_pred = torch.max(val_out, 1)
            val_pred_np = val_pred.cpu().numpy()

            acc = accuracy_score(val_labels_all, val_pred_np)
            bal_acc = balanced_accuracy_score(val_labels_all, val_pred_np)
            auc = roc_auc_score(val_labels_all, val_probs_np[:, 1])

            metrics[f'val_accuracy_{name}'] = acc
            metrics[f'val_balanced_accuracy_{name}'] = bal_acc
            metrics[f'val_auc_{name}'] = auc

            # 各被试个人准确率（本 fold 验证集内）
            per_subj = {}
            for sid in np.unique(val_subjects_all):
                mask = val_subjects_all == sid
                if mask.sum() == 0:
                    continue
                per_subj[int(sid)] = accuracy_score(
                    np.asarray(val_labels_all)[mask],
                    val_pred_np[mask]
                )
            metrics[f'val_accuracy_per_subject_{name}'] = per_subj

            print(
                f"    {name:24s} - 准确率: {acc:.4f}, 平衡准确率: {bal_acc:.4f}, "
                f"AUC: {auc:.4f}, Epochs: {actual_epochs}"
            )

    return metrics


if __name__ == "__main__":
    """
    作为独立脚本运行时：
    - 复用 `eeg_audio_residual.main` 的数据加载与交叉验证逻辑
    - 将其中的 `train_fold_model` 替换为本文件的 `train_fold_model_residual_and_attn`
    - 仅保存并汇总 residual / residual_eeg_cross_single / residual_eeg_cross_dual 三个模型的结果
    """
    import eeg_audio_residual as ear

    # 替换单 fold 训练函数
    ear.train_fold_model = train_fold_model_residual_and_attn

    # 包装保存函数，只关注三个模型
    original_save_fn = ear.save_cross_validation_results

    def save_cv_results_wrapper(all_fold_results, subject_names, per_subject_mean=None, model_names=None):
        target_models = ['residual', 'residual_eeg_shared', 'residual_eeg_env']
        return original_save_fn(all_fold_results, subject_names, per_subject_mean, model_names=target_models)

    ear.save_cross_validation_results = save_cv_results_wrapper

    # 调用原脚本的主入口，完成数据加载与交叉验证
    ear.main(filter_consistent=True)


