#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用响度特征建立6个线性回归模型预测选择概率

模型定义：
1. 模型1 - 平均响度差：使用现有 pair_feature_diff_raw.npy[5, :]（左右平均响度之差，16ms/帧）
2. 模型2 - 最大响度差：左边最大响度 - 右边最大响度（16ms/帧）
3. 模型3 - 前2秒平均响度差：前2秒左平均 - 前2秒右平均（16ms/帧，只取前2秒）
4. 模型4 - 后2秒平均响度差：后2秒左平均 - 后2秒右平均（16ms/帧，只取后2秒）
5. 模型5 - 响度标准差差：使用现有 pair_feature_diff_raw.npy[9, :]（左右响度标准差之差，16ms/帧）
6. 模型6 - 加权平均响度差：前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4 的左右差（16ms/帧）
"""
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import librosa
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# ============== 一致的5折划分配置（与attention_loudness_model.py保持一致） ==============
CV_SEED = 42
N_FOLDS = 5

def build_consistent_folds(n_samples: int, seed: int = CV_SEED, k: int = N_FOLDS):
    """
    与 attention_loudness_model.py 一致的折划分方式：
    - 先用固定随机种子打乱索引
    - 再用 np.array_split 切成 k 份
    返回 folds(list[np.ndarray]) 和 indices(np.ndarray, 打乱后的全索引)
    """
    indices = np.random.RandomState(seed).permutation(n_samples)
    folds = np.array_split(indices, k)
    return folds, indices


# ============== 复用原始代码的函数 ==============

def load_or_extract_features(project_root):
    """检测并加载已有特征数据，若不存在则提示"""
    raw_path = os.path.join(project_root, 'pair_feature_diff_raw.npy')
    zscore_path = os.path.join(project_root, 'pair_feature_diff_zscore.npy')
    list_path = os.path.join(project_root, 'pair_list.npy')

    if os.path.exists(raw_path) and os.path.exists(zscore_path) and os.path.exists(list_path):
        print("检测到已有特征文件，正在加载...")
        diff_raw = np.load(raw_path)
        diff_zscore = np.load(zscore_path)
        pair_names = np.load(list_path, allow_pickle=True)
        print(f"成功加载: {diff_raw.shape[1]} 个音频对, {diff_raw.shape[0]} 个特征")
        return diff_raw, diff_zscore, pair_names
    else:
        raise FileNotFoundError("未找到特征文件，请先运行 pair_feature_diff_extraction.py")


def parse_pair_name(filename):
    """解析音频对名称，提取左右音频标识和实验类型"""
    # 去掉扩展名
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    # 切分左右
    if '+' not in filename:
        return None, None, None, None, None

    left_part, right_part = filename.split('+', 1)

    # 提取实验类型
    experiment_type = None
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
    for p in prefixes:
        if left_part.startswith(p) or right_part.startswith(p):
            experiment_type = p.rstrip('_')
            break

    # 去掉前缀
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break

    # 提取类别和ID
    # 与 parse_stereo_pair_name 保持一致：使用第一个下划线分割
    def extract_cat_id(part):
        # 找到第一个下划线的位置
        first_underscore_pos = part.find('_')
        if first_underscore_pos == -1:
            return None, None
        # 第一个下划线前面是类别名，后面是文件ID
        cat = part[:first_underscore_pos]
        audio_id = part[first_underscore_pos + 1:]
        return cat, audio_id

    left_cat, left_id = extract_cat_id(left_part)
    right_cat, right_id = extract_cat_id(right_part)

    if left_cat is None or right_cat is None:
        return None, None, None, None, None

    return left_cat, left_id, right_cat, right_id, experiment_type


def parse_stereo_pair_name(filename, base_dir):
    """解析音频对名称并验证文件存在（用于提取新特征）"""
    # 去掉扩展名
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    # 切分左右
    if '+' not in filename:
        raise ValueError(f"无法解析左右两侧: {filename}")

    left_part, right_part = filename.split('+', 1)

    # 去掉前缀
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break

    # 基于文件系统解析类别和ID
    # 规则：第一个下划线前面是类别名，第一个下划线后面是文件ID
    def split_category_and_id_fs(part):
        # 找到第一个下划线的位置
        first_underscore_pos = part.find('_')
        if first_underscore_pos == -1:
            raise ValueError(f"未找到下划线分隔符: {part}")
        
        # 第一个下划线前面是类别名，后面是文件ID
        category = part[:first_underscore_pos]
        file_id = part[first_underscore_pos + 1:]
        
        # 验证文件是否存在
        candidate = os.path.join(base_dir, category, f"{file_id}.wav")
        if os.path.exists(candidate):
            return category, file_id
        else:
            raise ValueError(f"文件不存在: {candidate} (类别: {category}, 文件ID: {file_id})")

    left_cat, left_id = split_category_and_id_fs(left_part)
    right_cat, right_id = split_category_and_id_fs(right_part)
    return left_cat, left_id, right_cat, right_id


def merge_balanced_pairs_by_type(pair_names):
    """按实验类型分组合并左右平衡设计的音频对（288→144）"""
    print("\n=== 按实验类型合并左右平衡设计的音频对 ===")

    # 按实验类型分组
    type_groups = {'nn_main': [], 'nn_sub': [], 'main': [], 'sub': []}

    for idx, pair_name in enumerate(pair_names):
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[4] is None:
            print(f"警告: 无法解析实验类型 {pair_name}")
            continue

        left_cat, left_id, right_cat, right_id, experiment_type = parse_result
        type_groups[experiment_type].append((idx, pair_name))

    # 对每个类型进行合并
    merged_results = {}

    for exp_type, pairs in type_groups.items():
        if len(pairs) == 0:
            continue

        print(f"\n处理 {exp_type} 类型: {len(pairs)} 个音频对")

        unique_pairs = {}
        merged_indices_list = []
        merged_pair_names_list = []

        for idx, pair_name in pairs:
            left_cat, left_id, right_cat, right_id, _ = parse_pair_name(pair_name)

            # 创建规范化键（排序确保唯一性）
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            key = tuple(sorted([audio_a, audio_b]))

            if key not in unique_pairs:
                # 第一次出现，记录
                unique_pairs[key] = idx
                merged_indices_list.append(idx)
                merged_pair_names_list.append(pair_name)
            else:
                # 第二次出现（平衡配对），跳过
                pass

        print(f"  合并后唯一对数: {len(unique_pairs)}")

        merged_results[exp_type] = {
            'indices': merged_indices_list,
            'pair_names': merged_pair_names_list,
            'unique_pairs': unique_pairs
        }

    return merged_results


def parse_experiment_data_txt(file_path):
    """解析TXT格式的实验数据文件"""
    trials = []
    encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法读取文件: {file_path}")

    for i, line in enumerate(lines):
        if line.strip().startswith('Level: 3') and i + 1 < len(lines) and '*** LogFrame Start ***' in lines[i + 1]:
            trial_data = {}
            for j in range(i + 1, min(i + 20, len(lines))):
                current_line = lines[j].strip()
                if 'wavfile:' in current_line:
                    trial_data['wavfile'] = current_line.split('wavfile:')[1].strip()
                elif 'ImageDisplay1.RESP:' in current_line:
                    trial_data['response'] = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                elif '*** LogFrame End ***' in current_line:
                    break
            if 'wavfile' in trial_data and 'response' in trial_data:
                trials.append(trial_data)

    return trials


def parse_experiment_data_csv(file_path):
    """解析CSV格式的实验数据文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取CSV失败 {file_path}: {e}")
        return []

    trials = []
    for _, row in df.iterrows():
        audio_file = row['音频文件']
        choice = row['选择']

        if choice.lower() == 'left':
            response = 1
        elif choice.lower() == 'right':
            response = 2
        else:
            continue

        trials.append({
            'wavfile': audio_file,
            'response': response
        })

    return trials


def parse_audio_mapping(file_path):
    """解析音频映射文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        for _, row in df.iterrows():
            mapping[row['new_name']] = row['original_name']
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        return {}


def load_all_human_decisions(txt_files, csv_files, mapping_files):
    """加载所有人类决策数据，并标注被试ID"""
    print("\n=== 加载人类决策数据 ===")

    all_trials = []

    # 处理TXT文件
    for txt_file, mapping_file in zip(txt_files, mapping_files[:len(txt_files)]):
        if not os.path.exists(txt_file):
            print(f"TXT文件不存在: {txt_file}")
            continue

        # 从文件名提取被试ID
        subject_id = os.path.basename(os.path.dirname(txt_file)).split('_')[0]

        trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)

            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'subject_id': subject_id,
                    'original_name': original_name,
                    'response': trial['response']
                })

    # 处理CSV文件
    csv_mapping_files = mapping_files[len(txt_files):]
    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        if not os.path.exists(csv_file):
            print(f"CSV文件不存在: {csv_file}")
            continue

        # 从文件名提取被试ID
        base_name = os.path.basename(csv_file).replace('.csv', '')
        base_name = base_name.replace(' ', '')
        subject_id = re.sub(r'[\d_]+$', '', base_name).strip('_').strip()

        if not subject_id or subject_id.isdigit():
            match = re.match(r'^([a-zA-Z\u4e00-\u9fa5]+(?:\d+)?[a-zA-Z\u4e00-\u9fa5]*)', base_name)
            if match:
                subject_id = match.group(1)
            else:
                subject_id = base_name

        name_mapping = {
            'aiwenkai': 'aiwenkai',
            'lironghua': 'lironghua',
            'mayunmiao': 'mayunmiao',
            'ShangZiyang': 'ShangZiyang',
            'shangziyang': 'ShangZiyang',
            'wjy': 'wjy',
            'LiuYaorui': 'LiuYaorui',
            'liuyaorui': 'LiuYaorui',
        }

        subject_lower = subject_id.lower()
        for key, normalized in name_mapping.items():
            if subject_lower == key or subject_lower.startswith(key):
                subject_id = normalized
                break

        trials = parse_experiment_data_csv(csv_file)
        mapping = parse_audio_mapping(mapping_file)

        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)

            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'subject_id': subject_id,
                    'original_name': original_name,
                    'response': trial['response']
                })

    print(f"总共加载 {len(all_trials)} 个trial")
    unique_subjects = set(t['subject_id'] for t in all_trials)
    print(f"被试数量: {len(unique_subjects)}")

    return all_trials


def calculate_selection_probability_144(merged_pair_names, all_trials):
    """计算144对的左侧音频选择概率"""
    print("\n=== 计算144对的选择概率 ===")

    probabilities = []

    for pair_name in merged_pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            probabilities.append(np.nan)
            continue

        left_cat, left_id, right_cat, right_id, _ = parse_result

        # 左侧音频标识
        audio_left = f"{left_cat}_{left_id}"
        audio_right = f"{right_cat}_{right_id}"

        # 统计：左侧音频被选择的次数和出现次数
        selected_count = 0
        total_count = 0

        for trial in all_trials:
            trial_name = trial['original_name']
            trial_parse_result = parse_pair_name(trial_name)
            if trial_parse_result is None or trial_parse_result[0] is None:
                continue

            trial_left_cat, trial_left_id, trial_right_cat, trial_right_id, _ = trial_parse_result

            trial_audio_left = f"{trial_left_cat}_{trial_left_id}"
            trial_audio_right = f"{trial_right_cat}_{trial_right_id}"

            # 检查这个trial是否为AB组合对
            if (trial_audio_left == audio_left and trial_audio_right == audio_right) or \
                    (trial_audio_left == audio_right and trial_audio_right == audio_left):
                total_count += 1
                # 检查是否选择了audio_left
                if (trial_audio_left == audio_left and trial['response'] == 1) or \
                        (trial_audio_right == audio_left and trial['response'] == 2):
                    selected_count += 1

        prob = selected_count / total_count if total_count > 0 else 0.5
        probabilities.append(prob)

    probabilities = np.array(probabilities)
    print(f"计算完成: {len(probabilities)} 个概率值")
    print(f"概率范围: [{np.nanmin(probabilities):.3f}, {np.nanmax(probabilities):.3f}]")
    print(f"概率均值: {np.nanmean(probabilities):.3f}")

    return probabilities


# ============== 响度特征提取函数 ==============

def load_mono_5s(audio_path, target_sr=16000, duration_s=5.0):
    """加载5秒单声道音频，确保返回正确类型的连续数组"""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    target_len = int(duration_s * target_sr)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode='constant')
    # 确保是 float32 类型的连续数组
    y = np.ascontiguousarray(y, dtype=np.float32)
    return y, target_sr


def load_stereo_2s(stereo_path, target_sr=16000, duration_s=2.0):
    """加载实验用 2s 立体声对：第 0 通道=左耳，第 1 通道=右耳。"""
    y, sr = librosa.load(stereo_path, sr=target_sr, mono=False)
    if y.ndim == 1:
        y_left = y_right = y
    else:
        y_left, y_right = y[0], y[1]

    target_len = int(duration_s * target_sr)
    if len(y_left) >= target_len:
        y_left = y_left[:target_len]
        y_right = y_right[:target_len]
    else:
        pad = target_len - len(y_left)
        y_left = np.pad(y_left, (0, pad), mode='constant')
        y_right = np.pad(y_right, (0, pad), mode='constant')

    y_left = np.ascontiguousarray(y_left, dtype=np.float32)
    y_right = np.ascontiguousarray(y_right, dtype=np.float32)
    return y_left, y_right, sr


def extract_loudness_time_series(y, sr=16000, hop_length=256, n_fft=2048):
    """提取响度时间序列"""
    # STFT magnitude
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # 显式转换magnitude为float32，避免复数类型兼容性问题
    magnitude = np.abs(stft).astype(np.float32)

    # 28梅尔滤波器
    safe_fmax = min(12500, int(sr / 2) - 1)
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=28, fmin=250, fmax=safe_fmax)
    mel_spec = np.dot(mel_filters, magnitude) + 1e-10

    # 响度（对数压缩的梅尔能量和）
    loudness = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        loudness.append(np.sum(20 * np.log10(frame)))
    loudness = np.asarray(loudness)

    return loudness


def extract_loudness_max_diff(pair_name, base_dir):
    """
    模型2：提取最大响度差（左边最大响度 - 右边最大响度）
    使用16ms/帧（hop_length=256）
    使用 get_pair_feature_series 缓存避免重复提取
    """
    try:
        # 使用缓存的特征提取
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        loudness_L = features_L.get('loudness')
        loudness_R = features_R.get('loudness')
        
        if loudness_L is None or loudness_R is None:
            return np.nan

        # 计算最大响度差
        max_loudness_L = np.nanmax(loudness_L)
        max_loudness_R = np.nanmax(loudness_R)

        return max_loudness_L - max_loudness_R
    except Exception as e:
        print(f"提取最大响度差失败 {pair_name}: {e}")
        return np.nan


def extract_loudness_first2s_diff(pair_name, base_dir):
    """
    模型3：提取前2秒平均响度差（前2秒左平均 - 前2秒右平均）
    使用16ms/帧（hop_length=256）
    使用 get_pair_feature_series 缓存避免重复提取
    """
    try:
        # 使用缓存的特征提取
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        loudness_L = features_L.get('loudness')
        loudness_R = features_R.get('loudness')
        
        if loudness_L is None or loudness_R is None:
            return np.nan

        # 前2秒对应的帧数：2秒 / 0.016秒/帧 = 125帧
        frames_2s = int(2.0 * sr / 256)  # 2.0 * 16000 / 256 = 125

        # 只取前2秒的帧
        loudness_L_first2s = loudness_L[:frames_2s]
        loudness_R_first2s = loudness_R[:frames_2s]

        # 计算平均响度差
        mean_loudness_L = np.nanmean(loudness_L_first2s)
        mean_loudness_R = np.nanmean(loudness_R_first2s)

        return mean_loudness_L - mean_loudness_R
    except Exception as e:
        print(f"提取前2秒平均响度差失败 {pair_name}: {e}")
        return np.nan


def extract_loudness_last2s_diff(pair_name, base_dir):
    """
    模型4：提取后2秒平均响度差（后2秒左平均 - 后2秒右平均）
    使用16ms/帧（hop_length=256）
    使用 get_pair_feature_series 缓存避免重复提取
    """
    try:
        # 使用缓存的特征提取
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        loudness_L = features_L.get('loudness')
        loudness_R = features_R.get('loudness')
        
        if loudness_L is None or loudness_R is None:
            return np.nan

        # 只取后2秒的帧（从第3秒开始，即索引从3秒处开始）
        start_frame = int(3.0 * sr / 256)  # 3.0 * 16000 / 256 = 187.5，约188帧
        loudness_L_last2s = loudness_L[start_frame:]
        loudness_R_last2s = loudness_R[start_frame:]

        # 计算平均响度差
        mean_loudness_L = np.nanmean(loudness_L_last2s)
        mean_loudness_R = np.nanmean(loudness_R_last2s)

        return mean_loudness_L - mean_loudness_R
    except Exception as e:
        print(f"提取后2秒平均响度差失败 {pair_name}: {e}")
        return np.nan


def extract_loudness_weighted_diff(pair_name, base_dir):
    """
    模型6：提取加权平均响度差
    权重：前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4
    使用16ms/帧（hop_length=256）
    使用 get_pair_feature_series 缓存避免重复提取
    """
    try:
        # 使用缓存的特征提取
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        loudness_L = features_L.get('loudness')
        loudness_R = features_R.get('loudness')
        
        if loudness_L is None or loudness_R is None:
            return np.nan

        # 前1秒（0-1秒）
        frames_1s = int(1.0 * sr / 256)
        loudness_L_first1s = loudness_L[:frames_1s]
        loudness_R_first1s = loudness_R[:frames_1s]
        mean_L_first1s = np.nanmean(loudness_L_first1s)
        mean_R_first1s = np.nanmean(loudness_R_first1s)

        # 中间3秒（1-4秒）
        start_frame_mid = int(1.0 * sr / 256)
        end_frame_mid = int(4.0 * sr / 256)
        loudness_L_mid3s = loudness_L[start_frame_mid:end_frame_mid]
        loudness_R_mid3s = loudness_R[start_frame_mid:end_frame_mid]
        mean_L_mid3s = np.nanmean(loudness_L_mid3s)
        mean_R_mid3s = np.nanmean(loudness_R_mid3s)

        # 最后1秒（4-5秒）
        start_frame_last = int(4.0 * sr / 256)
        loudness_L_last1s = loudness_L[start_frame_last:]
        loudness_R_last1s = loudness_R[start_frame_last:]
        mean_L_last1s = np.nanmean(loudness_L_last1s)
        mean_R_last1s = np.nanmean(loudness_R_last1s)

        # 计算加权平均响度
        weighted_L = mean_L_first1s * 0.6 + mean_L_mid3s * 0.2 + mean_L_last1s * 0.4
        weighted_R = mean_R_first1s * 0.6 + mean_R_mid3s * 0.2 + mean_R_last1s * 0.4

        # 返回左右差
        return weighted_L - weighted_R
    except Exception as e:
        print(f"提取加权平均响度差失败 {pair_name}: {e}")
        return np.nan


def extract_loudness_weighted_std_diff(pair_name, base_dir):
    """
    提取加权标准差响度差
    权重：前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4
    使用16ms/帧（hop_length=256）
    使用 get_pair_feature_series 缓存避免重复提取
    """
    return extract_feature_weighted_std_diff(pair_name, base_dir, 'loudness')


# ============== 通用特征时间序列与缓存 ==============


def extract_full_feature_time_series(y, sr=16000, hop_length=256, n_fft=2048):
    """提取多种声学特征的时间序列（与 pair_feature_diff_extraction 保持一致）"""
    # 确保输入是正确类型的连续数组
    y = np.ascontiguousarray(y, dtype=np.float32)
    
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # 显式转换magnitude为float32，避免复数类型兼容性问题
    magnitude = np.abs(stft).astype(np.float32)

    safe_fmax = min(12500, int(sr / 2) - 1)
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=28, fmin=250, fmax=safe_fmax)
    mel_spec = np.dot(mel_filters, magnitude) + 1e-10

    # 与 pair_feature_diff_extraction.py 保持一致：使用 y 参数而不是 S
    # 这样可以避免复数类型兼容性问题
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).flatten()

    spec_flat = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        gmean = np.exp(np.mean(np.log(frame)))
        amean = np.mean(frame)
        spec_flat.append(gmean / amean if amean > 0 else 0.0)
    spec_flat = np.asarray(spec_flat)

    spec_irreg = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        spec_irreg.append(np.sum(np.abs(np.diff(frame))))
    spec_irreg = np.asarray(spec_irreg)

    # F0 和 Harmonicity 特征已完全移除（短时间提取意义不大，与 pair_feature_diff_extraction.py 保持一致）
    # 不再提取这两个特征，避免 librosa.pyin 和 librosa.effects.hpss 内部的 STFT 调用导致复数类型问题

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)
    temporal_mod = np.mean(np.abs(delta_mfcc), axis=0)

    coeff_idx = np.arange(mfcc.shape[0], dtype=float)
    spec_mod_centroid = []
    abs_mfcc = np.abs(mfcc)
    for t in range(mfcc.shape[1]):
        w = abs_mfcc[:, t]
        s = np.sum(w)
        spec_mod_centroid.append(float(np.sum(coeff_idx * w) / s) if s > 0 else 0.0)
    spec_mod_centroid = np.asarray(spec_mod_centroid)

    loudness = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        loudness.append(np.sum(20 * np.log10(frame)))
    loudness = np.asarray(loudness)

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spec_flat,
        'spectral_irregularity': spec_irreg,
        # 'f0' 和 'harmonicity' 已移除（短时间提取意义不大）
        'temporal_mod': temporal_mod,
        'spec_mod_centroid': spec_mod_centroid,
        'loudness': loudness,
    }


PAIR_FEATURE_CACHE: Dict[Tuple[str, str, int], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]] = {}

# 缓存中间计算结果：存储 (pair_name, base_dir, feature_key, start_sec, end_sec) -> (left_std, right_std)
STD_CACHE: Dict[Tuple[str, str, str, float, float], Tuple[float, float]] = {}

# 缓存统计
_cache_stats = {
    'pair_feature_hits': 0,
    'pair_feature_misses': 0,
    'std_cache_hits': 0,
    'std_cache_misses': 0
}


def get_pair_feature_series(pair_name, base_dir, hop_length=256):
    """
    获取音频对的特征时间序列（带缓存）
    同一个pair_name只会提取一次，后续直接从缓存读取
    """
    key = (pair_name, base_dir, hop_length)
    if key in PAIR_FEATURE_CACHE:
        _cache_stats['pair_feature_hits'] += 1
        return PAIR_FEATURE_CACHE[key]

    _cache_stats['pair_feature_misses'] += 1
    left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, base_dir)
    left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
    right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")

    if not os.path.exists(left_path) or not os.path.exists(right_path):
        raise FileNotFoundError(f"音频文件不存在: {left_path} 或 {right_path}")

    yL, sr = load_mono_5s(left_path)
    yR, _ = load_mono_5s(right_path)

    try:
        features_L = extract_full_feature_time_series(yL, sr=sr, hop_length=hop_length)
        features_R = extract_full_feature_time_series(yR, sr=sr, hop_length=hop_length)
    except Exception as e:
        # 如果提取失败，清理可能的缓存条目并重新抛出异常
        if key in PAIR_FEATURE_CACHE:
            del PAIR_FEATURE_CACHE[key]
        raise RuntimeError(f"提取特征失败 {pair_name}: {e}") from e

    PAIR_FEATURE_CACHE[key] = (features_L, features_R, sr)
    return PAIR_FEATURE_CACHE[key]


def nanmax_safe(values):
    if values is None or len(values) == 0:
        return np.nan
    mask = ~np.isnan(values)
    if not np.any(mask):
        return np.nan
    return float(np.nanmax(values[mask]))


def pooled_nanmax(series, window_size):
    if series is None or len(series) == 0:
        return np.nan
    window_size = max(1, int(window_size))
    if window_size == 1:
        return nanmax_safe(series)

    max_val = np.nan
    length = len(series)
    for start in range(0, length, window_size):
        chunk = series[start:start + window_size]
        val = nanmax_safe(chunk)
        if np.isnan(val):
            continue
        if np.isnan(max_val) or val > max_val:
            max_val = val
    return max_val


def segment_mean(series, sr, hop_length, start_sec, end_sec):
    if series is None or len(series) == 0:
        return np.nan
    start_frame = int(start_sec * sr / hop_length)
    end_frame = int(end_sec * sr / hop_length)
    if end_frame <= start_frame:
        return np.nan
    segment = series[start_frame:end_frame]
    if segment.size == 0:
        return np.nan
    mask = ~np.isnan(segment)
    if not np.any(mask):
        return np.nan
    return float(np.mean(segment[mask]))


def segment_std(series, sr, hop_length, start_sec, end_sec):
    """计算区段的标准差"""
    if series is None or len(series) == 0:
        return np.nan
    start_frame = int(start_sec * sr / hop_length)
    end_frame = int(end_sec * sr / hop_length)
    if end_frame <= start_frame:
        return np.nan
    segment = series[start_frame:end_frame]
    if segment.size == 0:
        return np.nan
    mask = ~np.isnan(segment)
    if not np.any(mask) or np.sum(mask) < 2:
        return np.nan
    return float(np.std(segment[mask]))


def get_cached_stat_values(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec):
    """
    获取或计算缓存的 left_stat 和 right_stat 值（支持mean或std）
    避免重复计算，供不同变换函数共享使用
    
    参数:
        stat_type: 'mean' 或 'std'
    """
    cache_key = (pair_name, base_dir, feature_key, stat_type, start_sec, end_sec)
    if cache_key in STD_CACHE:
        _cache_stats['std_cache_hits'] += 1
        return STD_CACHE[cache_key]
    
    _cache_stats['std_cache_misses'] += 1
    # 计算并缓存
    try:
        # 从缓存获取时间序列（如果已缓存则直接使用，否则会提取并缓存）
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        if left_series is None or right_series is None:
            return np.nan, np.nan
        
        # 计算该时间段的stat（mean或std）
        if stat_type == 'mean':
            left_stat = segment_mean(left_series, sr, 256, start_sec, end_sec)
            right_stat = segment_mean(right_series, sr, 256, start_sec, end_sec)
        elif stat_type == 'std':
            left_stat = segment_std(left_series, sr, 256, start_sec, end_sec)
            right_stat = segment_std(right_series, sr, 256, start_sec, end_sec)
        else:
            raise ValueError(f"未知的统计类型: {stat_type}")
        
        # 缓存计算结果
        STD_CACHE[cache_key] = (left_stat, right_stat)
        return left_stat, right_stat
    except Exception as exc:
        print(f"获取{stat_type}值失败 {pair_name} ({feature_key}): {exc}")
        return np.nan, np.nan


def get_cached_std_values(pair_name, base_dir, feature_key, start_sec, end_sec):
    """
    获取或计算缓存的 left_std 和 right_std 值
    避免重复计算，供绝对差异和相对差异函数共享使用
    
    缓存机制：
    1. 首先检查 STD_CACHE，如果已缓存则直接返回（避免重复计算std）
    2. 如果未缓存，从 PAIR_FEATURE_CACHE 获取时间序列（避免重复提取音频）
    3. 计算std并缓存结果
    """
    cache_key = (pair_name, base_dir, feature_key, start_sec, end_sec)
    if cache_key in STD_CACHE:
        _cache_stats['std_cache_hits'] += 1
        return STD_CACHE[cache_key]
    
    _cache_stats['std_cache_misses'] += 1
    # 计算并缓存
    try:
        # 从缓存获取时间序列（如果已缓存则直接使用，否则会提取并缓存）
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        if left_series is None or right_series is None:
            return np.nan, np.nan
        
        # 计算该时间段的std
        left_std = segment_std(left_series, sr, 256, start_sec, end_sec)
        right_std = segment_std(right_series, sr, 256, start_sec, end_sec)
        
        # 缓存计算结果
        STD_CACHE[cache_key] = (left_std, right_std)
        return left_std, right_std
    except Exception as exc:
        print(f"获取std值失败 {pair_name} ({feature_key}): {exc}")
        return np.nan, np.nan


def extract_feature_max_diff(pair_name, base_dir, feature_key):
    """
    提取特征最大差（左边最大值 - 右边最大值）
    使用16ms/帧（hop_length=256），直接取整个序列的最大值
    """
    try:
        base_hop = 256
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, base_hop)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        # 16ms/帧，直接取整个序列的最大值（不需要窗口聚合）
        left_max = nanmax_safe(left_series)
        right_max = nanmax_safe(right_series)
        if np.isnan(left_max) or np.isnan(right_max):
            return np.nan
        return left_max - right_max
    except Exception as exc:
        print(f"提取特征最大差失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_segment_diff(pair_name, base_dir, feature_key, start_sec, end_sec):
    try:
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        if left_series is None or right_series is None:
            return np.nan
        left_mean = segment_mean(left_series, sr, 256, start_sec, end_sec)
        right_mean = segment_mean(right_series, sr, 256, start_sec, end_sec)
        if np.isnan(left_mean) or np.isnan(right_mean):
            return np.nan
        return left_mean - right_mean
    except Exception as exc:
        print(f"提取特征区段差失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_segment_std_diff(pair_name, base_dir, feature_key, start_sec, end_sec):
    """提取特征区段标准差差（使用缓存避免重复计算）"""
    try:
        left_std, right_std = get_cached_std_values(pair_name, base_dir, feature_key, start_sec, end_sec)
        if np.isnan(left_std) or np.isnan(right_std):
            return np.nan
        return left_std - right_std
    except Exception as exc:
        print(f"提取特征区段标准差差失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_first1s_std_diff(pair_name, base_dir, feature_key):
    """提取前一秒的标准差差"""
    return extract_feature_segment_std_diff(pair_name, base_dir, feature_key, 0.0, 1.0)


def extract_feature_last1s_std_diff(pair_name, base_dir, feature_key):
    """提取最后一秒的标准差差"""
    return extract_feature_segment_std_diff(pair_name, base_dir, feature_key, 4.0, 5.0)


def extract_feature_first2s_std_diff(pair_name, base_dir, feature_key):
    """提取前2秒的标准差差"""
    return extract_feature_segment_std_diff(pair_name, base_dir, feature_key, 0.0, 2.0)


def extract_feature_last2s_std_diff(pair_name, base_dir, feature_key):
    """提取后2秒的标准差差"""
    return extract_feature_segment_std_diff(pair_name, base_dir, feature_key, 3.0, 5.0)


def extract_feature_all5s_std_diff(pair_name, base_dir, feature_key):
    """提取全部5秒的标准差差"""
    return extract_feature_segment_std_diff(pair_name, base_dir, feature_key, 0.0, 5.0)


def extract_feature_first2s_diff(pair_name, base_dir, feature_key):
    return extract_feature_segment_diff(pair_name, base_dir, feature_key, 0.0, 2.0)


def extract_feature_last2s_diff(pair_name, base_dir, feature_key):
    return extract_feature_segment_diff(pair_name, base_dir, feature_key, 3.0, 5.0)


def extract_feature_weighted_diff(pair_name, base_dir, feature_key):
    try:
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        if left_series is None or right_series is None:
            return np.nan

        segments = [
            (0.0, 1.0, 0.6),
            (1.0, 4.0, 0.2),
            (4.0, 5.0, 0.4),
        ]

        def weighted_sum(series):
            values = []
            for start_sec, end_sec, weight in segments:
                mean_val = segment_mean(series, sr, 256, start_sec, end_sec)
                if np.isnan(mean_val):
                    return np.nan
                values.append(mean_val * weight)
            return float(np.sum(values))

        left_weighted = weighted_sum(left_series)
        right_weighted = weighted_sum(right_series)
        if np.isnan(left_weighted) or np.isnan(right_weighted):
            return np.nan
        return left_weighted - right_weighted
    except Exception as exc:
        print(f"提取特征加权差失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_weighted_std_diff(pair_name, base_dir, feature_key):
    """
    提取加权标准差差
    权重：前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4
    使用16ms/帧（hop_length=256）
    """
    try:
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        if left_series is None or right_series is None:
            # 添加调试信息
            if not hasattr(extract_feature_weighted_std_diff, '_debug_missing'):
                print(f"  警告: {pair_name} 的特征 {feature_key} 不存在")
                print(f"    可用特征键: {list(features_L.keys()) if features_L else 'None'}")
                extract_feature_weighted_std_diff._debug_missing = True
            return np.nan

        segments = [
            (0.0, 1.0, 0.6),
            (1.0, 4.0, 0.2),
            (4.0, 5.0, 0.4),
        ]

        def weighted_sum(series):
            values = []
            for start_sec, end_sec, weight in segments:
                std_val = segment_std(series, sr, 256, start_sec, end_sec)
                if np.isnan(std_val):
                    # 如果某个segment是NaN，继续处理其他segment，而不是直接返回NaN
                    # 这样可以提高鲁棒性
                    continue
                values.append(std_val * weight)
            # 如果所有segment都是NaN，返回NaN
            if len(values) == 0:
                return np.nan
            return float(np.sum(values))

        left_weighted = weighted_sum(left_series)
        right_weighted = weighted_sum(right_series)
        if np.isnan(left_weighted) or np.isnan(right_weighted):
            return np.nan
        return left_weighted - right_weighted
    except Exception as exc:
        # 只在第一次失败时打印，避免输出过多
        if not hasattr(extract_feature_weighted_std_diff, '_debug_error'):
            print(f"提取特征加权标准差差失败 {pair_name} ({feature_key}): {exc}")
            import traceback
            traceback.print_exc()
            extract_feature_weighted_std_diff._debug_error = True
        return np.nan


def extract_feature_relative_diff(pair_name, base_dir, feature_key, start_sec, end_sec):
    """
    提取特征相对差异：(left_std - right_std) / (left_std + right_std)
    范围在 [-1, 1]，-1表示完全右优势，1表示完全左优势，0表示平衡
    使用缓存避免重复计算（与绝对差异共享缓存）
    """
    try:
        left_std, right_std = get_cached_std_values(pair_name, base_dir, feature_key, start_sec, end_sec)
        if np.isnan(left_std) or np.isnan(right_std):
            return np.nan
        denominator = left_std + right_std
        if denominator == 0 or np.isnan(denominator):
            return np.nan
        return (left_std - right_std) / denominator
    except Exception as exc:
        print(f"提取特征相对差异失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_nonlinear_competition(pair_name, base_dir, feature_key, start_sec, end_sec, alpha=0.5):
    """
    提取非线性竞争强度：sign(left_std - right_std) × |left_std - right_std|^α
    alpha: 竞争强度指数，默认0.5（平方根），1.0为线性，<1.0为次线性（饱和），>1.0为超线性
    使用缓存避免重复计算
    """
    try:
        left_std, right_std = get_cached_std_values(pair_name, base_dir, feature_key, start_sec, end_sec)
        if np.isnan(left_std) or np.isnan(right_std):
            return np.nan
        diff = left_std - right_std
        abs_diff = np.abs(diff)
        if abs_diff == 0:
            return 0.0
        sign = np.sign(diff)
        return sign * (abs_diff ** alpha)
    except Exception as exc:
        print(f"提取非线性竞争强度失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_ratio_competition(pair_name, base_dir, feature_key, start_sec, end_sec):
    """
    提取比值竞争：left_std / right_std
    范围在 (0, +∞)，1表示平衡，>1表示左优势，<1表示右优势
    使用对数变换：log(left_std / right_std) 使其对称
    使用缓存避免重复计算
    """
    try:
        left_std, right_std = get_cached_std_values(pair_name, base_dir, feature_key, start_sec, end_sec)
        if np.isnan(left_std) or np.isnan(right_std) or right_std == 0:
            return np.nan
        ratio = left_std / right_std
        # 使用对数变换使其对称：log(ratio)，0表示平衡，正数表示左优势，负数表示右优势
        return np.log(ratio)
    except Exception as exc:
        print(f"提取比值竞争失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_competition_index(pair_name, base_dir, feature_key, start_sec, end_sec):
    """
    提取竞争指数：综合多个维度的竞争强度
    定义：|left_std - right_std| / max(left_std, right_std)
    范围在 [0, 1]，0表示完全平衡，1表示完全不平衡
    带方向：sign(left_std - right_std) × competition_index
    使用缓存避免重复计算
    """
    try:
        left_std, right_std = get_cached_std_values(pair_name, base_dir, feature_key, start_sec, end_sec)
        if np.isnan(left_std) or np.isnan(right_std):
            return np.nan
        max_std = max(left_std, right_std)
        if max_std == 0:
            return 0.0
        diff = left_std - right_std
        competition_index = np.abs(diff) / max_std
        sign = np.sign(diff)
        return sign * competition_index
    except Exception as exc:
        print(f"提取竞争指数失败 {pair_name} ({feature_key}): {exc}")
        return np.nan


def extract_feature_relative_diff_with_stat(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec):
    """
    提取特征相对差异（支持mean或std）：(left_stat - right_stat) / (left_stat + right_stat)
    使用get_cached_stat_values支持mean和std
    """
    try:
        left_stat, right_stat = get_cached_stat_values(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec)
        if np.isnan(left_stat) or np.isnan(right_stat):
            return np.nan
        denominator = left_stat + right_stat
        if denominator == 0 or np.isnan(denominator):
            return np.nan
        return (left_stat - right_stat) / denominator
    except Exception as exc:
        print(f"提取特征相对差异失败 {pair_name} ({feature_key}, {stat_type}): {exc}")
        return np.nan


def extract_feature_nonlinear_competition_with_stat(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec, alpha=0.5):
    """
    提取非线性竞争强度（支持mean或std）：sign(left_stat - right_stat) × |left_stat - right_stat|^α
    使用get_cached_stat_values支持mean和std
    """
    try:
        left_stat, right_stat = get_cached_stat_values(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec)
        if np.isnan(left_stat) or np.isnan(right_stat):
            return np.nan
        diff = left_stat - right_stat
        abs_diff = np.abs(diff)
        if abs_diff == 0:
            return 0.0
        sign = np.sign(diff)
        return sign * (abs_diff ** alpha)
    except Exception as exc:
        print(f"提取非线性竞争强度失败 {pair_name} ({feature_key}, {stat_type}): {exc}")
        return np.nan


def extract_feature_ratio_competition_with_stat(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec):
    """
    提取比值竞争（支持mean或std）：log(left_stat / right_stat)
    使用get_cached_stat_values支持mean和std
    """
    try:
        left_stat, right_stat = get_cached_stat_values(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec)
        if np.isnan(left_stat) or np.isnan(right_stat) or right_stat == 0:
            return np.nan
        ratio = left_stat / right_stat
        return np.log(ratio)
    except Exception as exc:
        print(f"提取比值竞争失败 {pair_name} ({feature_key}, {stat_type}): {exc}")
        return np.nan


def extract_feature_competition_index_with_stat(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec):
    """
    提取竞争指数（支持mean或std）：sign(diff) × |diff| / max(left_stat, right_stat)
    使用get_cached_stat_values支持mean和std
    """
    try:
        left_stat, right_stat = get_cached_stat_values(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec)
        if np.isnan(left_stat) or np.isnan(right_stat):
            return np.nan
        max_stat = max(left_stat, right_stat)
        if max_stat == 0:
            return 0.0
        diff = left_stat - right_stat
        competition_index = np.abs(diff) / max_stat
        sign = np.sign(diff)
        return sign * competition_index
    except Exception as exc:
        print(f"提取竞争指数失败 {pair_name} ({feature_key}, {stat_type}): {exc}")
        return np.nan


def extract_feature_segment_diff_with_stat(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec):
    """
    提取特征区段差（支持mean或std）：left_stat - right_stat
    使用get_cached_stat_values支持mean和std
    """
    try:
        left_stat, right_stat = get_cached_stat_values(pair_name, base_dir, feature_key, stat_type, start_sec, end_sec)
        if np.isnan(left_stat) or np.isnan(right_stat):
            return np.nan
        return left_stat - right_stat
    except Exception as exc:
        print(f"提取特征区段差失败 {pair_name} ({feature_key}, {stat_type}): {exc}")
        return np.nan


# 便捷函数：针对不同时间段的相对差异
def extract_feature_first2s_relative_diff(pair_name, base_dir, feature_key):
    """提取前2秒的相对差异"""
    return extract_feature_relative_diff(pair_name, base_dir, feature_key, 0.0, 2.0)


def extract_feature_last2s_relative_diff(pair_name, base_dir, feature_key):
    """提取后2秒的相对差异"""
    return extract_feature_relative_diff(pair_name, base_dir, feature_key, 3.0, 5.0)


def extract_feature_first2s_nonlinear_competition(pair_name, base_dir, feature_key, alpha=0.5):
    """提取前2秒的非线性竞争强度"""
    return extract_feature_nonlinear_competition(pair_name, base_dir, feature_key, 0.0, 2.0, alpha)


def extract_feature_last2s_nonlinear_competition(pair_name, base_dir, feature_key, alpha=0.5):
    """提取后2秒的非线性竞争强度"""
    return extract_feature_nonlinear_competition(pair_name, base_dir, feature_key, 3.0, 5.0, alpha)


def extract_feature_first2s_ratio_competition(pair_name, base_dir, feature_key):
    """提取前2秒的比值竞争"""
    return extract_feature_ratio_competition(pair_name, base_dir, feature_key, 0.0, 2.0)


def extract_feature_last2s_ratio_competition(pair_name, base_dir, feature_key):
    """提取后2秒的比值竞争"""
    return extract_feature_ratio_competition(pair_name, base_dir, feature_key, 3.0, 5.0)


def extract_feature_first2s_competition_index(pair_name, base_dir, feature_key):
    """提取前2秒的竞争指数"""
    return extract_feature_competition_index(pair_name, base_dir, feature_key, 0.0, 2.0)


def extract_feature_last2s_competition_index(pair_name, base_dir, feature_key):
    """提取后2秒的竞争指数"""
    return extract_feature_competition_index(pair_name, base_dir, feature_key, 3.0, 5.0)


# 全部秒（0-5秒）的便捷函数
def extract_feature_all5s_relative_diff(pair_name, base_dir, feature_key):
    """提取全部5秒的相对差异"""
    return extract_feature_relative_diff(pair_name, base_dir, feature_key, 0.0, 5.0)


def extract_feature_all5s_nonlinear_competition(pair_name, base_dir, feature_key, alpha=0.5):
    """提取全部5秒的非线性竞争强度"""
    return extract_feature_nonlinear_competition(pair_name, base_dir, feature_key, 0.0, 5.0, alpha)


def extract_feature_all5s_ratio_competition(pair_name, base_dir, feature_key):
    """提取全部5秒的比值竞争"""
    return extract_feature_ratio_competition(pair_name, base_dir, feature_key, 0.0, 5.0)


def extract_feature_all5s_competition_index(pair_name, base_dir, feature_key):
    """提取全部5秒的竞争指数"""
    return extract_feature_competition_index(pair_name, base_dir, feature_key, 0.0, 5.0)


# ============== 多特征投票机制 ==============

def extract_feature_vote(pair_name, base_dir, feature_key, start_sec, end_sec, threshold=0.0):
    """
    提取特征投票：如果 left_std > right_std + threshold，返回 +1（投给左边）
                 如果 right_std > left_std + threshold，返回 -1（投给右边）
                 否则返回 0（弃权）
    
    参数:
        threshold: 投票阈值，只有当差异超过阈值时才投票
    """
    try:
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        left_series = features_L.get(feature_key)
        right_series = features_R.get(feature_key)
        if left_series is None or right_series is None:
            return 0.0
        left_std = segment_std(left_series, sr, 256, start_sec, end_sec)
        right_std = segment_std(right_series, sr, 256, start_sec, end_sec)
        if np.isnan(left_std) or np.isnan(right_std):
            return 0.0
        
        diff = left_std - right_std
        if diff > threshold:
            return 1.0  # 投给左边
        elif diff < -threshold:
            return -1.0  # 投给右边
        else:
            return 0.0  # 弃权（差异太小）
    except Exception as exc:
        print(f"提取特征投票失败 {pair_name} ({feature_key}): {exc}")
        return 0.0


def extract_multi_feature_vote_score(pair_name, base_dir, feature_keys, start_sec, end_sec, 
                                     weights=None, threshold=0.0, vote_type='simple'):
    """
    提取多特征投票得分：多个特征分别投票，然后统计得分
    
    参数:
        feature_keys: 特征键列表，如 ['loudness', 'spectral_centroid', ...]
        weights: 每个特征的权重列表，如果为None则权重相等
        threshold: 投票阈值
        vote_type: 投票类型
            - 'simple': 简单投票（+1/-1/0）
            - 'weighted': 加权投票（权重 × 投票）
            - 'soft': 软投票（使用差异的符号，但保留强度）
    
    返回:
        投票得分：正数表示左优势，负数表示右优势，绝对值表示优势强度
    """
    try:
        # 确保feature_keys是列表
        if not isinstance(feature_keys, (list, tuple)):
            raise ValueError(f"feature_keys必须是列表或元组，当前类型: {type(feature_keys)}, 值: {feature_keys}")
        feature_keys = list(feature_keys)  # 转换为列表
        
        if len(feature_keys) == 0:
            raise ValueError(f"feature_keys不能为空，pair_name: {pair_name}")
        
        
        if weights is None:
            weights = [1.0] * len(feature_keys)
        elif len(weights) != len(feature_keys):
            weights = [1.0] * len(feature_keys)
        
        # 一次性获取所有特征序列（避免重复获取）
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        
        # 调试：检查特征字典的键
        if not hasattr(extract_multi_feature_vote_score, '_debug_keys_printed'):
            print(f"  调试: features_L的键: {list(features_L.keys())}")
            print(f"  调试: 需要的feature_keys: {feature_keys}")
            extract_multi_feature_vote_score._debug_keys_printed = True
        
        total_score = 0.0
        total_weight = 0.0
        found_features = 0
        missing_features = []
        nan_features = []
        
        for feature_key, weight in zip(feature_keys, weights):
            left_series = features_L.get(feature_key)
            right_series = features_R.get(feature_key)
            
            if left_series is None or right_series is None:
                missing_features.append(feature_key)
                continue
            
            left_std = segment_std(left_series, sr, 256, start_sec, end_sec)
            right_std = segment_std(right_series, sr, 256, start_sec, end_sec)
            
            if np.isnan(left_std) or np.isnan(right_std):
                nan_features.append(feature_key)
                continue
            
            found_features += 1
            diff = left_std - right_std
            
            if vote_type == 'simple':
                # 简单投票：+1/-1/0
                if diff > threshold:
                    vote = 1.0
                elif diff < -threshold:
                    vote = -1.0
                else:
                    vote = 0.0
                total_score += weight * vote
                if vote != 0:
                    total_weight += weight
            elif vote_type == 'weighted':
                # 加权投票：权重 × 投票
                if diff > threshold:
                    vote = 1.0
                elif diff < -threshold:
                    vote = -1.0
                else:
                    vote = 0.0
                total_score += weight * vote
                if vote != 0:
                    total_weight += weight
            elif vote_type == 'soft':
                # 软投票：保留差异强度，但归一化
                if abs(diff) > threshold:
                    # 归一化到 [-1, 1]
                    max_std = max(left_std, right_std)
                    if max_std > 0:
                        normalized_diff = diff / max_std
                        total_score += weight * normalized_diff
                        total_weight += weight
            else:
                raise ValueError(f"未知的投票类型: {vote_type}")
        
        # 如果所有特征都找不到，返回NaN而不是0.0
        if found_features == 0:
            # 只在第一个样本时打印详细调试信息
            if not hasattr(extract_multi_feature_vote_score, '_debug_missing_printed'):
                print(f"  警告: {pair_name} 所有特征都找不到或都是NaN")
                if missing_features:
                    print(f"    缺失的特征: {missing_features}")
                if nan_features:
                    print(f"    NaN的特征: {nan_features}")
                print(f"    可用的特征键: {list(features_L.keys())}")
                extract_multi_feature_vote_score._debug_missing_printed = True
            return np.nan
        
        if total_weight == 0:
            # 如果所有特征的投票都是0（diff在threshold范围内），返回0.0
            return 0.0
        
        # 归一化得分
        return total_score / total_weight if vote_type in ['simple', 'weighted'] else total_score
    except Exception as exc:
        print(f"提取多特征投票得分失败 {pair_name}: {exc}")
        import traceback
        traceback.print_exc()
        return np.nan


def extract_feature_vote_count(pair_name, base_dir, feature_keys, start_sec, end_sec, threshold=0.0):
    """
    提取特征投票计数：统计投给左边和右边的特征数量
    
    返回:
        (left_votes, right_votes, total_votes): 左边票数、右边票数、总票数
    """
    try:
        # 一次性获取所有特征序列（避免重复获取）
        features_L, features_R, sr = get_pair_feature_series(pair_name, base_dir, hop_length=256)
        
        left_votes = 0
        right_votes = 0
        total_votes = 0
        
        for feature_key in feature_keys:
            left_series = features_L.get(feature_key)
            right_series = features_R.get(feature_key)
            
            if left_series is None or right_series is None:
                continue
            
            left_std = segment_std(left_series, sr, 256, start_sec, end_sec)
            right_std = segment_std(right_series, sr, 256, start_sec, end_sec)
            
            if np.isnan(left_std) or np.isnan(right_std):
                continue
            
            diff = left_std - right_std
            if diff > threshold:
                left_votes += 1
                total_votes += 1
            elif diff < -threshold:
                right_votes += 1
                total_votes += 1
        
        return left_votes, right_votes, total_votes
    except Exception as exc:
        print(f"提取特征投票计数失败 {pair_name}: {exc}")
        return 0, 0, 0


def extract_feature_vote_margin(pair_name, base_dir, feature_keys, start_sec, end_sec, threshold=0.0):
    """
    提取特征投票优势：left_votes - right_votes（票数差）
    
    返回:
        投票优势：正数表示左优势，负数表示右优势
    """
    try:
        left_votes, right_votes, _ = extract_feature_vote_count(
            pair_name, base_dir, feature_keys, start_sec, end_sec, threshold
        )
        return left_votes - right_votes
    except Exception as exc:
        print(f"提取特征投票优势失败 {pair_name}: {exc}")
        return 0.0


def extract_feature_vote_ratio(pair_name, base_dir, feature_keys, start_sec, end_sec, threshold=0.0):
    """
    提取特征投票比例：(left_votes - right_votes) / total_votes
    
    返回:
        投票比例：范围 [-1, 1]，正数表示左优势，负数表示右优势
    """
    try:
        left_votes, right_votes, total_votes = extract_feature_vote_count(
            pair_name, base_dir, feature_keys, start_sec, end_sec, threshold
        )
        if total_votes == 0:
            return 0.0
        return (left_votes - right_votes) / total_votes
    except Exception as exc:
        print(f"提取特征投票比例失败 {pair_name}: {exc}")
        return 0.0


# 便捷函数：针对不同时间段和特征组合
def extract_feature_first2s_vote_margin(pair_name, base_dir, feature_keys, threshold=0.0):
    """提取前2秒的特征投票优势"""
    return extract_feature_vote_margin(pair_name, base_dir, feature_keys, 0.0, 2.0, threshold)


def extract_feature_last2s_vote_margin(pair_name, base_dir, feature_keys, threshold=0.0):
    """提取后2秒的特征投票优势"""
    return extract_feature_vote_margin(pair_name, base_dir, feature_keys, 3.0, 5.0, threshold)


def extract_feature_first2s_vote_ratio(pair_name, base_dir, feature_keys, threshold=0.0):
    """提取前2秒的特征投票比例"""
    return extract_feature_vote_ratio(pair_name, base_dir, feature_keys, 0.0, 2.0, threshold)


def extract_feature_last2s_vote_ratio(pair_name, base_dir, feature_keys, threshold=0.0):
    """提取后2秒的特征投票比例"""
    return extract_feature_vote_ratio(pair_name, base_dir, feature_keys, 3.0, 5.0, threshold)


def extract_feature_first2s_vote_score(pair_name, base_dir, feature_keys, weights=None, threshold=0.0, vote_type='simple'):
    """提取前2秒的多特征投票得分"""
    return extract_multi_feature_vote_score(pair_name, base_dir, feature_keys, 0.0, 2.0, weights, threshold, vote_type)


def extract_feature_last2s_vote_score(pair_name, base_dir, feature_keys, weights=None, threshold=0.0, vote_type='simple'):
    """提取后2秒的多特征投票得分"""
    return extract_multi_feature_vote_score(pair_name, base_dir, feature_keys, 3.0, 5.0, weights, threshold, vote_type)


# 全部秒（0-5秒）的投票函数
def extract_feature_all5s_vote_margin(pair_name, base_dir, feature_keys, threshold=0.0):
    """提取全部5秒的特征投票优势"""
    return extract_feature_vote_margin(pair_name, base_dir, feature_keys, 0.0, 5.0, threshold)


def extract_feature_all5s_vote_ratio(pair_name, base_dir, feature_keys, threshold=0.0):
    """提取全部5秒的特征投票比例"""
    return extract_feature_vote_ratio(pair_name, base_dir, feature_keys, 0.0, 5.0, threshold)


def extract_feature_all5s_vote_score(pair_name, base_dir, feature_keys, weights=None, threshold=0.0, vote_type='simple'):
    """提取全部5秒的多特征投票得分"""
    return extract_multi_feature_vote_score(pair_name, base_dir, feature_keys, 0.0, 5.0, weights, threshold, vote_type)


def extract_feature_all5s_vote_probability(pair_name, base_dir, feature_keys, weights=None, 
                                           threshold=0.0, vote_type='simple', prob_method='sigmoid'):
    """提取全部5秒的特征投票概率"""
    return extract_feature_vote_probability(
        pair_name, base_dir, feature_keys, 0.0, 5.0, 
        weights, threshold, vote_type, prob_method
    )


def vote_score_to_probability(vote_score, method='sigmoid'):
    """
    将投票得分转换为概率
    
    参数:
        vote_score: 投票得分（通常在 [-1, 1] 或类似范围）
        method: 转换方法
            - 'sigmoid': 使用sigmoid函数 (1 / (1 + exp(-k*x)))，k=3使转换更陡峭
            - 'linear': 线性映射 (x + 1) / 2，将 [-1, 1] 映射到 [0, 1]
            - 'tanh_linear': 先tanh再线性映射
    
    返回:
        概率值（0到1之间）
    """
    if np.isnan(vote_score):
        return 0.5  # 默认概率
    
    if method == 'sigmoid':
        # sigmoid: 1 / (1 + exp(-k*x))
        # k=3 使转换更陡峭，vote_score=0时概率=0.5
        k = 3.0
        return 1.0 / (1.0 + np.exp(-k * vote_score))
    elif method == 'linear':
        # 线性映射：将 [-1, 1] 映射到 [0, 1]
        # 如果vote_score超出范围，先clip
        vote_score_clipped = np.clip(vote_score, -1.0, 1.0)
        return (vote_score_clipped + 1.0) / 2.0
    elif method == 'tanh_linear':
        # 先tanh归一化，再线性映射
        vote_score_tanh = np.tanh(vote_score)
        return (vote_score_tanh + 1.0) / 2.0
    else:
        raise ValueError(f"未知的转换方法: {method}")


def extract_feature_vote_probability(pair_name, base_dir, feature_keys, start_sec, end_sec, 
                                     weights=None, threshold=0.0, vote_type='simple', prob_method='sigmoid'):
    """
    提取特征投票概率：多个特征投票后转换为选择左边的概率
    
    参数:
        feature_keys: 特征键列表
        weights: 权重列表
        threshold: 投票阈值
        vote_type: 投票类型（'simple', 'weighted', 'soft'）
        prob_method: 概率转换方法（'sigmoid', 'linear', 'tanh_linear'）
    
    返回:
        选择左边的概率（0到1之间）
    """
    try:
        vote_score = extract_multi_feature_vote_score(
            pair_name, base_dir, feature_keys, start_sec, end_sec, 
            weights, threshold, vote_type
        )
        # 如果vote_score是NaN，直接返回NaN
        if np.isnan(vote_score):
            return np.nan
        return vote_score_to_probability(vote_score, prob_method)
    except Exception as exc:
        print(f"提取特征投票概率失败 {pair_name}: {exc}")
        import traceback
        traceback.print_exc()
        return np.nan  # 返回NaN而不是0.5


def extract_feature_first2s_vote_probability(pair_name, base_dir, feature_keys, weights=None, 
                                             threshold=0.0, vote_type='simple', prob_method='sigmoid'):
    """提取前2秒的特征投票概率"""
    return extract_feature_vote_probability(
        pair_name, base_dir, feature_keys, 0.0, 2.0, 
        weights, threshold, vote_type, prob_method
    )


def extract_feature_last2s_vote_probability(pair_name, base_dir, feature_keys, weights=None, 
                                           threshold=0.0, vote_type='simple', prob_method='sigmoid'):
    """提取后2秒的特征投票概率"""
    return extract_feature_vote_probability(
        pair_name, base_dir, feature_keys, 3.0, 5.0, 
        weights, threshold, vote_type, prob_method
    )


# ============== 特征注册表 ==============


@dataclass
class FeatureSpec:
    feature_id: str
    display_name: str
    source: str  # 'diff_raw', 'diff_zscore', 'custom'
    description: str = ''
    index: Optional[int] = None
    extractor: Optional[Callable[[str, str], float]] = None
    progress_interval: Optional[int] = None
    tags: Sequence[str] = field(default_factory=tuple)


FEATURE_REGISTRY: Dict[str, FeatureSpec] = {}


def register_feature(spec: FeatureSpec):
    if spec.feature_id in FEATURE_REGISTRY:
        raise ValueError(f"重复注册特征: {spec.feature_id}")
    FEATURE_REGISTRY[spec.feature_id] = spec


def _register_precomputed_acoustic_features():
    # 前18个声学特征的索引映射（从0开始）：
    # 0: Spectral Centroid (mean)
    # 1: Spectral Centroid (std)
    # 2: F0 (mean)
    # 3: F0 (std)
    # 4: Harmonicity (mean)
    # 5: Harmonicity (std)
    # 6: Spectral Modulation Centroid (mean)
    # 7: Spectral Modulation Centroid (std)
    # 8: Spectral Irregularity (mean)
    # 9: Spectral Irregularity (std)
    # 10: Loudness (mean)
    # 11: Loudness (std)
    # ...
    
    acoustic_features = [
        ("f0_mean", "F0 (mean)", 2),
        ("f0_std", "F0 (std)", 3),
        ("harmonicity_mean", "Harmonicity (mean)", 4),
        ("harmonicity_std", "Harmonicity (std)", 5),
        ("loudness_mean", "Loudness (mean)", 10),
        ("loudness_std", "Loudness (std)", 11),
    ]

    for base_id, display_name, idx in acoustic_features:
        register_feature(FeatureSpec(
            feature_id=f"{base_id}_z",
            display_name=f"{display_name} (z-score diff)",
            source='diff_zscore',
            index=idx,
            description='来自 pair_feature_diff_zscore.npy 的标准化差值',
            tags=('acoustic', 'zscore')
        ))


def _register_raw_loudness_features():
    register_feature(FeatureSpec(
        feature_id='loudness_mean_diff_raw',
        display_name='Loudness (mean) diff (raw)',
        source='diff_raw',
        index=5,
        description='来自 pair_feature_diff_raw.npy 的平均响度差',
        tags=('loudness', 'raw')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_std_diff_raw',
        display_name='Loudness (std) diff (raw)',
        source='diff_raw',
        index=9,
        description='来自 pair_feature_diff_raw.npy 的响度标准差之差',
        tags=('loudness', 'raw')
    ))


def _register_custom_features():
    register_feature(FeatureSpec(
        feature_id='loudness_max_diff',
        display_name='Loudness max diff (16ms)',
        source='custom',
        extractor=extract_loudness_max_diff,
        progress_interval=20,
        description='左右声道最大响度差，使用16ms/帧',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_first2s_diff',
        display_name='Loudness mean diff (first 2s)',
        source='custom',
        extractor=extract_loudness_first2s_diff,
        progress_interval=20,
        description='前2秒的平均响度差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_diff',
        display_name='Loudness mean diff (last 2s)',
        source='custom',
        extractor=extract_loudness_last2s_diff,
        progress_interval=20,
        description='后2秒的平均响度差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_weighted_diff',
        display_name='Loudness weighted diff (1-3-1s)',
        source='custom',
        extractor=extract_loudness_weighted_diff,
        progress_interval=20,
        description='前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4 的加权响度差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_weighted_std_diff',
        display_name='Loudness weighted std diff (1-3-1s)',
        source='custom',
        extractor=extract_loudness_weighted_std_diff,
        progress_interval=20,
        description='前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4 的加权响度标准差差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_first1s_std_diff',
        display_name='Loudness std diff (first 1s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first1s_std_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='前一秒的响度标准差差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last1s_std_diff',
        display_name='Loudness std diff (last 1s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last1s_std_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='最后一秒的响度标准差差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_first2s_std_diff',
        display_name='Loudness std diff (first 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first2s_std_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='前2秒的响度标准差差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_std_diff',
        display_name='Loudness std diff (last 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last2s_std_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='后2秒的响度标准差差',
        tags=('loudness', 'custom')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_all5s_std_diff',
        display_name='Loudness std diff (all 5s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_all5s_std_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='全部5秒的响度标准差差',
        tags=('loudness', 'custom')
    ))

    # 相对差异特征
    register_feature(FeatureSpec(
        feature_id='loudness_first2s_relative_diff',
        display_name='Loudness relative diff (first 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first2s_relative_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='前2秒的相对差异：(left_std - right_std) / (left_std + right_std)',
        tags=('loudness', 'relative', 'competition')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_relative_diff',
        display_name='Loudness relative diff (last 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last2s_relative_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='后2秒的相对差异：(left_std - right_std) / (left_std + right_std)',
        tags=('loudness', 'relative', 'competition')
    ))

    # 非线性竞争特征（alpha=0.5，平方根）
    register_feature(FeatureSpec(
        feature_id='loudness_first2s_nonlinear_competition_alpha05',
        display_name='Loudness nonlinear competition (first 2s, α=0.5)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first2s_nonlinear_competition(pair_name, base_dir, 'loudness', alpha=0.5),
        progress_interval=20,
        description='前2秒的非线性竞争强度：sign(diff) × |diff|^0.5',
        tags=('loudness', 'nonlinear', 'competition')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_nonlinear_competition_alpha05',
        display_name='Loudness nonlinear competition (last 2s, α=0.5)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last2s_nonlinear_competition(pair_name, base_dir, 'loudness', alpha=0.5),
        progress_interval=20,
        description='后2秒的非线性竞争强度：sign(diff) × |diff|^0.5',
        tags=('loudness', 'nonlinear', 'competition')
    ))

    # 非线性竞争特征（alpha=1.0，线性）
    register_feature(FeatureSpec(
        feature_id='loudness_first2s_nonlinear_competition_alpha10',
        display_name='Loudness nonlinear competition (first 2s, α=1.0)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first2s_nonlinear_competition(pair_name, base_dir, 'loudness', alpha=1.0),
        progress_interval=20,
        description='前2秒的非线性竞争强度：sign(diff) × |diff|^1.0（线性）',
        tags=('loudness', 'nonlinear', 'competition')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_nonlinear_competition_alpha10',
        display_name='Loudness nonlinear competition (last 2s, α=1.0)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last2s_nonlinear_competition(pair_name, base_dir, 'loudness', alpha=1.0),
        progress_interval=20,
        description='后2秒的非线性竞争强度：sign(diff) × |diff|^1.0（线性）',
        tags=('loudness', 'nonlinear', 'competition')
    ))

    # 比值竞争特征
    register_feature(FeatureSpec(
        feature_id='loudness_first2s_ratio_competition',
        display_name='Loudness ratio competition (first 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first2s_ratio_competition(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='前2秒的比值竞争：log(left_std / right_std)',
        tags=('loudness', 'ratio', 'competition')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_ratio_competition',
        display_name='Loudness ratio competition (last 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last2s_ratio_competition(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='后2秒的比值竞争：log(left_std / right_std)',
        tags=('loudness', 'ratio', 'competition')
    ))

    # 竞争指数特征
    register_feature(FeatureSpec(
        feature_id='loudness_first2s_competition_index',
        display_name='Loudness competition index (first 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_first2s_competition_index(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='前2秒的竞争指数：sign(diff) × |diff| / max(left_std, right_std)',
        tags=('loudness', 'competition_index')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_last2s_competition_index',
        display_name='Loudness competition index (last 2s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_last2s_competition_index(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='后2秒的竞争指数：sign(diff) × |diff| / max(left_std, right_std)',
        tags=('loudness', 'competition_index')
    ))

    # 全部秒（0-5秒）的竞争特征
    register_feature(FeatureSpec(
        feature_id='loudness_all5s_relative_diff',
        display_name='Loudness relative diff (all 5s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_all5s_relative_diff(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='全部5秒的相对差异：(left_std - right_std) / (left_std + right_std)',
        tags=('loudness', 'relative', 'competition', 'all5s')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_all5s_nonlinear_competition_alpha05',
        display_name='Loudness nonlinear competition (all 5s, α=0.5)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_all5s_nonlinear_competition(pair_name, base_dir, 'loudness', alpha=0.5),
        progress_interval=20,
        description='全部5秒的非线性竞争强度：sign(diff) × |diff|^0.5',
        tags=('loudness', 'nonlinear', 'competition', 'all5s')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_all5s_ratio_competition',
        display_name='Loudness ratio competition (all 5s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_all5s_ratio_competition(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='全部5秒的比值竞争：log(left_std / right_std)',
        tags=('loudness', 'ratio', 'competition', 'all5s')
    ))

    register_feature(FeatureSpec(
        feature_id='loudness_all5s_competition_index',
        display_name='Loudness competition index (all 5s)',
        source='custom',
        extractor=lambda pair_name, base_dir: extract_feature_all5s_competition_index(pair_name, base_dir, 'loudness'),
        progress_interval=20,
        description='全部5秒的竞争指数：sign(diff) × |diff| / max(left_std, right_std)',
        tags=('loudness', 'competition_index', 'all5s')
    ))


def _register_voting_features():
    """
    注册多特征投票特征
    定义常用的特征组合，然后为每个组合注册投票特征
    """
    # 定义常用的特征组合（使用实际的特征键名称）
    feature_combinations = {
        'all_5_features': {
            'keys': ['spectral_centroid', 'spec_mod_centroid', 'spectral_irregularity', 
                     'loudness', 'temporal_mod'],
            'display_name': 'All 5 Features',
            'description': '5个主要特征：Spectral Centroid, Spectral Modulation Centroid, Spectral Irregularity, Loudness, Temporal Modulation'
        },
        'spectral_features': {
            'keys': ['spectral_centroid', 'spec_mod_centroid', 'spectral_irregularity'],
            'display_name': 'Spectral Features',
            'description': '谱特征：Spectral Centroid, Spectral Modulation Centroid, Spectral Irregularity'
        },
        'loudness_only': {
            'keys': ['loudness'],
            'display_name': 'Loudness Only',
            'description': '仅响度特征'
        },
        'selected_4_features': {
            'keys': ['spectral_centroid', 'spec_mod_centroid', 'spectral_irregularity', 'loudness'],
            'display_name': 'Selected 4 Features',
            'description': '4个选定特征（不含Temporal Modulation）'
        }
    }
    
    # 创建包装函数来避免Lambda闭包问题
    def make_vote_score_extractor(feature_keys_list, vote_type='simple'):
        """创建投票得分提取器"""
        keys_copy = list(feature_keys_list)  # 创建副本
        def extractor(pair_name, base_dir):
            return extract_feature_all5s_vote_score(
                pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type=vote_type
            )
        return extractor
    
    def make_vote_probability_extractor(feature_keys_list, prob_method='sigmoid'):
        """创建投票概率提取器"""
        keys_copy = list(feature_keys_list)  # 创建副本
        def extractor(pair_name, base_dir):
            return extract_feature_all5s_vote_probability(
                pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type='simple', prob_method=prob_method
            )
        return extractor
    
    def make_vote_ratio_extractor(feature_keys_list):
        """创建投票比例提取器"""
        keys_copy = list(feature_keys_list)  # 创建副本
        def extractor(pair_name, base_dir):
            return extract_feature_all5s_vote_ratio(
                pair_name, base_dir, keys_copy, threshold=0.0
            )
        return extractor
    
    def make_vote_margin_extractor(feature_keys_list, time_range='first2s'):
        """创建投票优势提取器"""
        keys_copy = list(feature_keys_list)  # 创建副本
        if time_range == 'first2s':
            def extractor(pair_name, base_dir):
                return extract_feature_first2s_vote_margin(pair_name, base_dir, keys_copy, threshold=0.0)
        elif time_range == 'last2s':
            def extractor(pair_name, base_dir):
                return extract_feature_last2s_vote_margin(pair_name, base_dir, keys_copy, threshold=0.0)
        else:
            def extractor(pair_name, base_dir):
                return extract_feature_all5s_vote_margin(pair_name, base_dir, keys_copy, threshold=0.0)
        return extractor
    
    def make_vote_ratio_extractor_time(feature_keys_list, time_range='first2s'):
        """创建投票比例提取器（指定时间段）"""
        keys_copy = list(feature_keys_list)  # 创建副本
        if time_range == 'first2s':
            def extractor(pair_name, base_dir):
                return extract_feature_first2s_vote_ratio(pair_name, base_dir, keys_copy, threshold=0.0)
        elif time_range == 'last2s':
            def extractor(pair_name, base_dir):
                return extract_feature_last2s_vote_ratio(pair_name, base_dir, keys_copy, threshold=0.0)
        else:
            def extractor(pair_name, base_dir):
                return extract_feature_all5s_vote_ratio(pair_name, base_dir, keys_copy, threshold=0.0)
        return extractor
    
    def make_vote_score_extractor_time(feature_keys_list, time_range='first2s', vote_type='simple'):
        """创建投票得分提取器（指定时间段）"""
        keys_copy = list(feature_keys_list)  # 创建副本
        if time_range == 'first2s':
            def extractor(pair_name, base_dir):
                return extract_feature_first2s_vote_score(pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type=vote_type)
        elif time_range == 'last2s':
            def extractor(pair_name, base_dir):
                return extract_feature_last2s_vote_score(pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type=vote_type)
        else:
            def extractor(pair_name, base_dir):
                return extract_feature_all5s_vote_score(pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type=vote_type)
        return extractor
    
    def make_vote_probability_extractor_time(feature_keys_list, time_range='first2s', prob_method='sigmoid'):
        """创建投票概率提取器（指定时间段）"""
        keys_copy = list(feature_keys_list)  # 创建副本
        if time_range == 'first2s':
            def extractor(pair_name, base_dir):
                return extract_feature_first2s_vote_probability(pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type='simple', prob_method=prob_method)
        elif time_range == 'last2s':
            def extractor(pair_name, base_dir):
                return extract_feature_last2s_vote_probability(pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type='simple', prob_method=prob_method)
        else:
            def extractor(pair_name, base_dir):
                return extract_feature_all5s_vote_probability(pair_name, base_dir, keys_copy, weights=None, threshold=0.0, vote_type='simple', prob_method=prob_method)
        return extractor
    
    # 为每个特征组合注册投票特征
    for combo_id, combo_info in feature_combinations.items():
        feature_keys = combo_info['keys']
        display_base = combo_info['display_name']
        
        # 投票优势（票数差）
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_margin_first2s',
            display_name=f'{display_base} Vote Margin (first 2s)',
            source='custom',
            extractor=make_vote_margin_extractor(feature_keys, time_range='first2s'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 前2秒投票优势（left_votes - right_votes）',
            tags=('voting', 'vote_margin', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_margin_last2s',
            display_name=f'{display_base} Vote Margin (last 2s)',
            source='custom',
            extractor=make_vote_margin_extractor(feature_keys, time_range='last2s'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 后2秒投票优势（left_votes - right_votes）',
            tags=('voting', 'vote_margin', combo_id)
        ))
        
        # 投票比例
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_ratio_first2s',
            display_name=f'{display_base} Vote Ratio (first 2s)',
            source='custom',
            extractor=make_vote_ratio_extractor_time(feature_keys, time_range='first2s'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 前2秒投票比例（(left_votes - right_votes) / total_votes）',
            tags=('voting', 'vote_ratio', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_ratio_last2s',
            display_name=f'{display_base} Vote Ratio (last 2s)',
            source='custom',
            extractor=make_vote_ratio_extractor_time(feature_keys, time_range='last2s'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 后2秒投票比例（(left_votes - right_votes) / total_votes）',
            tags=('voting', 'vote_ratio', combo_id)
        ))
        
        # 投票得分（简单投票）
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_score_simple_first2s',
            display_name=f'{display_base} Vote Score Simple (first 2s)',
            source='custom',
            extractor=make_vote_score_extractor_time(feature_keys, time_range='first2s', vote_type='simple'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 前2秒简单投票得分（等权重）',
            tags=('voting', 'vote_score', 'simple', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_score_simple_last2s',
            display_name=f'{display_base} Vote Score Simple (last 2s)',
            source='custom',
            extractor=make_vote_score_extractor_time(feature_keys, time_range='last2s', vote_type='simple'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 后2秒简单投票得分（等权重）',
            tags=('voting', 'vote_score', 'simple', combo_id)
        ))
        
        # 投票得分（软投票）
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_score_soft_first2s',
            display_name=f'{display_base} Vote Score Soft (first 2s)',
            source='custom',
            extractor=make_vote_score_extractor_time(feature_keys, time_range='first2s', vote_type='soft'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 前2秒软投票得分（保留差异强度）',
            tags=('voting', 'vote_score', 'soft', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_score_soft_last2s',
            display_name=f'{display_base} Vote Score Soft (last 2s)',
            source='custom',
            extractor=make_vote_score_extractor_time(feature_keys, time_range='last2s', vote_type='soft'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 后2秒软投票得分（保留差异强度）',
            tags=('voting', 'vote_score', 'soft', combo_id)
        ))
        
        # 投票概率特征（sigmoid转换）
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_probability_sigmoid_first2s',
            display_name=f'{display_base} Vote Probability Sigmoid (first 2s)',
            source='custom',
            extractor=make_vote_probability_extractor_time(feature_keys, time_range='first2s', prob_method='sigmoid'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 前2秒投票概率（sigmoid转换，可直接用于概率拟合）',
            tags=('voting', 'vote_probability', 'sigmoid', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_probability_sigmoid_last2s',
            display_name=f'{display_base} Vote Probability Sigmoid (last 2s)',
            source='custom',
            extractor=make_vote_probability_extractor_time(feature_keys, time_range='last2s', prob_method='sigmoid'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 后2秒投票概率（sigmoid转换，可直接用于概率拟合）',
            tags=('voting', 'vote_probability', 'sigmoid', combo_id)
        ))
        
        # 投票概率特征（线性转换）
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_probability_linear_first2s',
            display_name=f'{display_base} Vote Probability Linear (first 2s)',
            source='custom',
            extractor=make_vote_probability_extractor_time(feature_keys, time_range='first2s', prob_method='linear'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 前2秒投票概率（线性转换，可直接用于概率拟合）',
            tags=('voting', 'vote_probability', 'linear', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_probability_linear_last2s',
            display_name=f'{display_base} Vote Probability Linear (last 2s)',
            source='custom',
            extractor=make_vote_probability_extractor_time(feature_keys, time_range='last2s', prob_method='linear'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 后2秒投票概率（线性转换，可直接用于概率拟合）',
            tags=('voting', 'vote_probability', 'linear', combo_id)
        ))
        
        # 全部秒（0-5秒）的投票特征
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_probability_sigmoid_all5s',
            display_name=f'{display_base} Vote Probability Sigmoid (all 5s)',
            source='custom',
            extractor=make_vote_probability_extractor(feature_keys, prob_method='sigmoid'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 全部5秒投票概率（sigmoid转换，可直接用于概率拟合）',
            tags=('voting', 'vote_probability', 'sigmoid', 'all5s', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_probability_linear_all5s',
            display_name=f'{display_base} Vote Probability Linear (all 5s)',
            source='custom',
            extractor=make_vote_probability_extractor(feature_keys, prob_method='linear'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 全部5秒投票概率（线性转换，可直接用于概率拟合）',
            tags=('voting', 'vote_probability', 'linear', 'all5s', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_score_simple_all5s',
            display_name=f'{display_base} Vote Score Simple (all 5s)',
            source='custom',
            extractor=make_vote_score_extractor(feature_keys, vote_type='simple'),
            progress_interval=20,
            description=f'{combo_info["description"]} - 全部5秒简单投票得分（等权重）',
            tags=('voting', 'vote_score', 'simple', 'all5s', combo_id)
        ))
        
        register_feature(FeatureSpec(
            feature_id=f'{combo_id}_vote_ratio_all5s',
            display_name=f'{display_base} Vote Ratio (all 5s)',
            source='custom',
            extractor=make_vote_ratio_extractor(feature_keys),
            progress_interval=20,
            description=f'{combo_info["description"]} - 全部5秒投票比例（(left_votes - right_votes) / total_votes）',
            tags=('voting', 'vote_ratio', 'all5s', combo_id)
        ))


def _register_feature_families_with_generic_metrics():
    families = [
        ('f0', 'F0', 2, 3, 'f0'),
        ('harmonicity', 'Harmonicity', 4, 5, 'harmonicity'),
        ('spectral_modulation_centroid', 'Spectral Modulation Centroid', 6, 7, 'spec_mod_centroid'),
        ('spectral_irregularity', 'Spectral Irregularity', 8, 9, 'spectral_irregularity'),
        ('temporal_modulation', 'Temporal Modulation', 16, 17, 'temporal_mod'),
    ]

    for base_id, display_name, mean_idx, std_idx, ts_key in families:
        register_feature(FeatureSpec(
            feature_id=f'{base_id}_mean_diff_raw',
            display_name=f'{display_name} (mean) diff (raw)',
            source='diff_raw',
            index=mean_idx,
            description=f'来自 pair_feature_diff_raw.npy 的 {display_name} 平均差',
            tags=(base_id, 'raw')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_std_diff_raw',
            display_name=f'{display_name} (std) diff (raw)',
            source='diff_raw',
            index=std_idx,
            description=f'来自 pair_feature_diff_raw.npy 的 {display_name} 标准差之差',
            tags=(base_id, 'raw')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_max_diff',
            display_name=f'{display_name} max diff (16ms)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_max_diff(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 最大值差 (16ms/帧)',
            tags=(base_id, 'custom')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_first2s_diff',
            display_name=f'{display_name} mean diff (first 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_first2s_diff(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 前2秒平均差',
            tags=(base_id, 'custom')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_last2s_diff',
            display_name=f'{display_name} mean diff (last 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_last2s_diff(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 后2秒平均差',
            tags=(base_id, 'custom')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_weighted_diff',
            display_name=f'{display_name} weighted diff (1-3-1s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_weighted_diff(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 加权平均差 (前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4)',
            tags=(base_id, 'custom')
        ))

        # 相对差异特征
        register_feature(FeatureSpec(
            feature_id=f'{base_id}_first2s_relative_diff',
            display_name=f'{display_name} relative diff (first 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_first2s_relative_diff(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 前2秒相对差异：(left_std - right_std) / (left_std + right_std)',
            tags=(base_id, 'relative', 'competition')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_last2s_relative_diff',
            display_name=f'{display_name} relative diff (last 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_last2s_relative_diff(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 后2秒相对差异：(left_std - right_std) / (left_std + right_std)',
            tags=(base_id, 'relative', 'competition')
        ))

        # 非线性竞争特征（alpha=0.5）
        register_feature(FeatureSpec(
            feature_id=f'{base_id}_first2s_nonlinear_competition_alpha05',
            display_name=f'{display_name} nonlinear competition (first 2s, α=0.5)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_first2s_nonlinear_competition(pair_name, base_dir, key, alpha=0.5),
            progress_interval=20,
            description=f'{display_name} 前2秒非线性竞争强度：sign(diff) × |diff|^0.5',
            tags=(base_id, 'nonlinear', 'competition')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_last2s_nonlinear_competition_alpha05',
            display_name=f'{display_name} nonlinear competition (last 2s, α=0.5)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_last2s_nonlinear_competition(pair_name, base_dir, key, alpha=0.5),
            progress_interval=20,
            description=f'{display_name} 后2秒非线性竞争强度：sign(diff) × |diff|^0.5',
            tags=(base_id, 'nonlinear', 'competition')
        ))

        # 非线性竞争特征（alpha=1.0）
        register_feature(FeatureSpec(
            feature_id=f'{base_id}_first2s_nonlinear_competition_alpha10',
            display_name=f'{display_name} nonlinear competition (first 2s, α=1.0)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_first2s_nonlinear_competition(pair_name, base_dir, key, alpha=1.0),
            progress_interval=20,
            description=f'{display_name} 前2秒非线性竞争强度：sign(diff) × |diff|^1.0（线性）',
            tags=(base_id, 'nonlinear', 'competition')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_last2s_nonlinear_competition_alpha10',
            display_name=f'{display_name} nonlinear competition (last 2s, α=1.0)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_last2s_nonlinear_competition(pair_name, base_dir, key, alpha=1.0),
            progress_interval=20,
            description=f'{display_name} 后2秒非线性竞争强度：sign(diff) × |diff|^1.0（线性）',
            tags=(base_id, 'nonlinear', 'competition')
        ))

        # 比值竞争特征
        register_feature(FeatureSpec(
            feature_id=f'{base_id}_first2s_ratio_competition',
            display_name=f'{display_name} ratio competition (first 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_first2s_ratio_competition(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 前2秒比值竞争：log(left_std / right_std)',
            tags=(base_id, 'ratio', 'competition')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_last2s_ratio_competition',
            display_name=f'{display_name} ratio competition (last 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_last2s_ratio_competition(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 后2秒比值竞争：log(left_std / right_std)',
            tags=(base_id, 'ratio', 'competition')
        ))

        # 竞争指数特征
        register_feature(FeatureSpec(
            feature_id=f'{base_id}_first2s_competition_index',
            display_name=f'{display_name} competition index (first 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_first2s_competition_index(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 前2秒竞争指数：sign(diff) × |diff| / max(left_std, right_std)',
            tags=(base_id, 'competition_index')
        ))

        register_feature(FeatureSpec(
            feature_id=f'{base_id}_last2s_competition_index',
            display_name=f'{display_name} competition index (last 2s)',
            source='custom',
            extractor=lambda pair_name, base_dir, key=ts_key: extract_feature_last2s_competition_index(pair_name, base_dir, key),
            progress_interval=20,
            description=f'{display_name} 后2秒竞争指数：sign(diff) × |diff| / max(left_std, right_std)',
            tags=(base_id, 'competition_index')
        ))


_register_precomputed_acoustic_features()
_register_raw_loudness_features()
_register_custom_features()
_register_feature_families_with_generic_metrics()
_register_voting_features()


DEFAULT_AUDIO_BASE_DIR = r"D:\D\research\audioset下载\clap_select"

# 可选特征列表（feature_id -> display_name）
# - spectral_centroid_mean_z -> Spectral Centroid (mean) (z-score diff)
# - spectral_centroid_std_z -> Spectral Centroid (std) (z-score diff)
# - f0_mean_z -> F0 (mean) (z-score diff)
# - f0_std_z -> F0 (std) (z-score diff)
# - harmonicity_mean_z -> Harmonicity (mean) (z-score diff)
# - harmonicity_std_z -> Harmonicity (std) (z-score diff)
# - spectral_modulation_centroid_mean_z -> Spectral Modulation Centroid (mean) (z-score diff)
# - spectral_modulation_centroid_std_z -> Spectral Modulation Centroid (std) (z-score diff)
# - spectral_irregularity_mean_z -> Spectral Irregularity (mean) (z-score diff)
# - spectral_irregularity_std_z -> Spectral Irregularity (std) (z-score diff)
# - loudness_mean_z -> Loudness (mean) (z-score diff)
# - loudness_std_z -> Loudness (std) (z-score diff)
# - spectral_flatness_mean_z -> Spectral Flatness (mean) (z-score diff)
# - spectral_flatness_std_z -> Spectral Flatness (std) (z-score diff)
# - spectral_bandwidth_mean_z -> Spectral Bandwidth (mean) (z-score diff)
# - spectral_bandwidth_std_z -> Spectral Bandwidth (std) (z-score diff)
# - temporal_modulation_mean_z -> Temporal Modulation (mean) (z-score diff)
# - temporal_modulation_std_z -> Temporal Modulation (std) (z-score diff)
# - loudness_mean_diff_raw -> Loudness (mean) diff (raw)
# - loudness_std_diff_raw -> Loudness (std) diff (raw)
# - loudness_max_diff -> Loudness max diff (16ms)
# - loudness_first2s_diff -> Loudness mean diff (first 2s)
# - loudness_last2s_diff -> Loudness mean diff (last 2s)
# - loudness_weighted_diff -> Loudness weighted diff (1-3-1s)

# ============== 使用选定的9个声学特征 ==============
# 从 linear_regression_selected_features.py 中使用的7个特征 + F0和谐波度的mean
# feature_idx 从1开始计数（对应前18个声学特征）
SELECTED_FEATURE_INDICES = [2, 3, 4, 7, 8, 9, 10, 12, 17]  # 九个特征
SELECTED_FEATURE_NAMES = [
    'Spectral Centroid (std)',  # 2 (索引1)
    'F0 (mean)',  # 3 (索引2)
    'Harmonicity (mean)',  # 4 (索引3)
    'Spectral Modulation Centroid (mean)',  # 7 (索引6)
    'Spectral Modulation Centroid (std)',  # 8 (索引7)
    'Spectral Irregularity (mean)',  # 9 (索引8)
    'Spectral Irregularity (std)',  # 10 (索引9)
    'Loudness (std)',  # 12 (索引11)
    'Temporal Modulation (mean)',  # 17 (索引16)
]

# 选定特征对应的feature_key和统计类型（用于提取时间序列特征）
# 格式：(feature_key, stat_type) 其中stat_type为'mean'或'std'
# 重要：stat_type必须与SELECTED_FEATURE_NAMES中的标注一致
# - 如果特征名包含 (std)，则stat_type='std'
# - 如果特征名包含 (mean)，则stat_type='mean'
SELECTED_FEATURE_CONFIGS = [
    ('spectral_centroid', 'std'),      # Spectral Centroid (std) - 使用std
    ('f0', 'mean'),                    # F0 (mean) - 使用mean
    ('harmonicity', 'mean'),           # Harmonicity (mean) - 使用mean
    ('spec_mod_centroid', 'mean'),     # Spectral Modulation Centroid (mean) - 使用mean
    ('spec_mod_centroid', 'std'),      # Spectral Modulation Centroid (std) - 使用std
    ('spectral_irregularity', 'mean'), # Spectral Irregularity (mean) - 使用mean
    ('spectral_irregularity', 'std'),  # Spectral Irregularity (std) - 使用std
    ('loudness', 'std'),               # Loudness (std) - 使用std
    ('temporal_mod', 'mean'),          # Temporal Modulation (mean) - 使用mean
]

MODEL_CONFIGS: List[Dict[str, Sequence[str]]] = []

MODEL_FAMILIES: List[Tuple[str, str]] = [
    ('Loudness', 'loudness'),  # 只保留响度特征
]

MODEL_VARIANTS: List[Tuple[str, Callable[[str], str]]] = [
    ('模型1_整体标准差差', lambda base: f'{base}_std_diff_raw'),
    ('模型2_加权标准差差', lambda base: f'{base}_weighted_std_diff'),
    ('模型3_全部5秒标准差差', lambda base: f'{base}_all5s_std_diff'),
    # ('模型4_前2秒标准差差', lambda base: f'{base}_first2s_std_diff'),  # 注释掉前2秒
    # ('模型5_后2秒标准差差', lambda base: f'{base}_last2s_std_diff'),  # 注释掉后2秒
]

for family_name, family_id in MODEL_FAMILIES:
    for variant_name, feature_fn in MODEL_VARIANTS:
        MODEL_CONFIGS.append({
            'model_name': f'{family_name}_{variant_name}',
            'feature_ids': [feature_fn(family_id)]
        })

# 添加新的竞争特征模型配置
MODEL_CONFIGS.extend([
    # 相对差异特征（注释掉前2秒和后2秒）
    # {
    #     'model_name': 'Loudness_相对差异_前2秒',
    #     'feature_ids': ['loudness_first2s_relative_diff']
    # },
    # {
    #     'model_name': 'Loudness_相对差异_后2秒',
    #     'feature_ids': ['loudness_last2s_relative_diff']
    # },
    
    # 非线性竞争特征（alpha=0.5）（注释掉前2秒和后2秒）
    # {
    #     'model_name': 'Loudness_非线性竞争_前2秒_alpha05',
    #     'feature_ids': ['loudness_first2s_nonlinear_competition_alpha05']
    # },
    # {
    #     'model_name': 'Loudness_非线性竞争_后2秒_alpha05',
    #     'feature_ids': ['loudness_last2s_nonlinear_competition_alpha05']
    # },
    
    # 比值竞争特征（注释掉前2秒和后2秒）
    # {
    #     'model_name': 'Loudness_比值竞争_前2秒',
    #     'feature_ids': ['loudness_first2s_ratio_competition']
    # },
    # {
    #     'model_name': 'Loudness_比值竞争_后2秒',
    #     'feature_ids': ['loudness_last2s_ratio_competition']
    # },
    
    # 竞争指数特征（注释掉前2秒和后2秒）
    # {
    #     'model_name': 'Loudness_竞争指数_前2秒',
    #     'feature_ids': ['loudness_first2s_competition_index']
    # },
    # {
    #     'model_name': 'Loudness_竞争指数_后2秒',
    #     'feature_ids': ['loudness_last2s_competition_index']
    # },
    
    # 投票概率特征（sigmoid）（注释掉前2秒和后2秒）
    # {
    #     'model_name': '投票概率_5特征_sigmoid_前2秒',
    #     'feature_ids': ['all_5_features_vote_probability_sigmoid_first2s']
    # },
    # {
    #     'model_name': '投票概率_5特征_sigmoid_后2秒',
    #     'feature_ids': ['all_5_features_vote_probability_sigmoid_last2s']
    # },
    # {
    #     'model_name': '投票概率_4特征_sigmoid_前2秒',
    #     'feature_ids': ['selected_4_features_vote_probability_sigmoid_first2s']
    # },
    # {
    #     'model_name': '投票概率_4特征_sigmoid_后2秒',
    #     'feature_ids': ['selected_4_features_vote_probability_sigmoid_last2s']
    # },
    
    # 投票概率特征（线性）（注释掉前2秒和后2秒）
    # {
    #     'model_name': '投票概率_5特征_linear_前2秒',
    #     'feature_ids': ['all_5_features_vote_probability_linear_first2s']
    # },
    # {
    #     'model_name': '投票概率_5特征_linear_后2秒',
    #     'feature_ids': ['all_5_features_vote_probability_linear_last2s']
    # },
    # {
    #     'model_name': '投票概率_4特征_linear_前2秒',
    #     'feature_ids': ['selected_4_features_vote_probability_linear_first2s']
    # },
    # {
    #     'model_name': '投票概率_4特征_linear_后2秒',
    #     'feature_ids': ['selected_4_features_vote_probability_linear_last2s']
    # },
    
    # 投票得分特征（用于对比）（注释掉前2秒和后2秒）
    # {
    #     'model_name': '投票得分_5特征_简单_前2秒',
    #     'feature_ids': ['all_5_features_vote_score_simple_first2s']
    # },
    # {
    #     'model_name': '投票得分_5特征_简单_后2秒',
    #     'feature_ids': ['all_5_features_vote_score_simple_last2s']
    # },
    # {
    #     'model_name': '投票比例_5特征_前2秒',
    #     'feature_ids': ['all_5_features_vote_ratio_first2s']
    # },
    # {
    #     'model_name': '投票比例_5特征_后2秒',
    #     'feature_ids': ['all_5_features_vote_ratio_last2s']
    # },
    
    # 全部秒（0-5秒）的竞争特征模型
    {
        'model_name': 'Loudness_相对差异_全部5秒',
        'feature_ids': ['loudness_all5s_relative_diff']
    },
    {
        'model_name': 'Loudness_非线性竞争_全部5秒_alpha05',
        'feature_ids': ['loudness_all5s_nonlinear_competition_alpha05']
    },
    {
        'model_name': 'Loudness_比值竞争_全部5秒',
        'feature_ids': ['loudness_all5s_ratio_competition']
    },
    {
        'model_name': 'Loudness_竞争指数_全部5秒',
        'feature_ids': ['loudness_all5s_competition_index']
    },
    
    # 全部秒（0-5秒）的投票概率模型（sigmoid）
    {
        'model_name': '投票概率_5特征_sigmoid_全部5秒',
        'feature_ids': ['all_5_features_vote_probability_sigmoid_all5s']
    },
    {
        'model_name': '投票概率_4特征_sigmoid_全部5秒',
        'feature_ids': ['selected_4_features_vote_probability_sigmoid_all5s']
    },
    
    # 全部秒（0-5秒）的投票概率模型（线性）
    {
        'model_name': '投票概率_5特征_linear_全部5秒',
        'feature_ids': ['all_5_features_vote_probability_linear_all5s']
    },
    {
        'model_name': '投票概率_4特征_linear_全部5秒',
        'feature_ids': ['selected_4_features_vote_probability_linear_all5s']
    },
    
    # 全部秒（0-5秒）的投票得分/比例模型（用于对比）
    {
        'model_name': '投票得分_5特征_简单_全部5秒',
        'feature_ids': ['all_5_features_vote_score_simple_all5s']
    },
    {
        'model_name': '投票比例_5特征_全部5秒',
        'feature_ids': ['all_5_features_vote_ratio_all5s']
    },
])


ENABLE_AUTO_SINGLE_FEATURE_MODELS = False


if ENABLE_AUTO_SINGLE_FEATURE_MODELS:
    existing_feature_sets = {tuple(config['feature_ids']) for config in MODEL_CONFIGS}
    for feature_id, spec in FEATURE_REGISTRY.items():
        if spec.source == 'diff_zscore':
            key = (feature_id,)
            if key in existing_feature_sets:
                continue
            MODEL_CONFIGS.append({
                'model_name': f"单特征_{spec.display_name}",
                'feature_ids': [feature_id]
            })
            existing_feature_sets.add(key)


# ============== 回归分析函数 ==============

def prepare_regression_data(feature_matrix, probabilities):
    """
    准备回归数据，去除包含NaN的样本
    参数:
        feature_matrix: 特征矩阵 (n_features, n_pairs)
        probabilities: 概率向量 (n_pairs,)
    返回:
        X_clean: 有效特征矩阵 (n_samples, n_features)
        y_clean: 有效目标变量 (n_samples,)
        valid_mask: 有效数据掩码 (n_pairs,)
    """
    X = feature_matrix.T
    feature_mask = ~np.isnan(X).any(axis=1)
    prob_mask = ~np.isnan(probabilities)
    valid_mask = feature_mask & prob_mask

    print(f"原始数据: {len(probabilities)} 个样本")
    print(f"有效数据: {np.sum(valid_mask)} 个样本")
    print(f"缺失数据: {len(probabilities) - np.sum(valid_mask)} 个样本")
    
    # 添加调试信息：如果所有样本都被过滤，显示原因
    if np.sum(valid_mask) == 0 and len(probabilities) > 0:
        nan_by_feature = np.isnan(X).sum(axis=0)
        nan_by_sample = np.isnan(X).sum(axis=1)
        prob_nan_count = np.isnan(probabilities).sum()
        print(f"  调试信息:")
        print(f"    每个特征的NaN数量: {nan_by_feature}")
        print(f"    每个样本的NaN数量: {nan_by_sample[:5]}... (前5个样本)")
        print(f"    概率值中的NaN数量: {prob_nan_count}")
        if np.sum(nan_by_feature) > 0:
            print(f"    有NaN的特征索引: {np.where(nan_by_feature > 0)[0]}")

    return X[valid_mask, :], probabilities[valid_mask], valid_mask


def fit_ols_model(X, y, feature_names):
    """
    使用statsmodels拟合OLS模型
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)
        feature_names: 特征名称列表
    返回:
        model: 拟合的OLS模型
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    X_with_const = sm.add_constant(X_df)
    model = sm.OLS(y, X_with_const).fit()
    return model


def extract_model_statistics(model, feature_names):
    """提取模型统计信息"""
    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()

    if isinstance(params, pd.Series):
        param_index = params.index.tolist()
    else:
        param_index = [f'x{i}' for i in range(len(params))]

    if 'const' in param_index:
        intercept_value = float(params['const'])
        intercept_pvalue = float(pvalues['const'])
        intercept_ci = conf_int.loc['const'].tolist()
    else:
        intercept_value = float(params.iloc[0])
        intercept_pvalue = float(pvalues.iloc[0])
        intercept_ci = conf_int.iloc[0].tolist()

    feature_params = []
    feature_pvalues = []
    feature_conf_int = []
    actual_feature_names = []

    if isinstance(conf_int, pd.DataFrame):
        conf_int_index = conf_int.index.tolist()
    else:
        conf_int_index = [f'x{i}' for i in range(conf_int.shape[0])]

    for name in param_index:
        if name == 'const':
            continue
        feature_params.append(float(params[name]))
        feature_pvalues.append(float(pvalues[name]))
        if name in conf_int_index:
            ci_bounds = conf_int.loc[name].tolist()
        else:
            idx = param_index.index(name)
            ci_bounds = conf_int.iloc[idx].tolist()
        feature_conf_int.append([float(ci_bounds[0]), float(ci_bounds[1])])
        actual_feature_names.append(name)

    if len(actual_feature_names) != len(feature_names):
        actual_feature_names = list(feature_names)

    stats_dict = {
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic,
        'fvalue': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'nobs': int(model.nobs),
        'df_model': float(model.df_model),
        'df_resid': float(model.df_resid),
        'intercept': intercept_value,
        'intercept_pvalue': intercept_pvalue,
        'intercept_ci': [float(intercept_ci[0]), float(intercept_ci[1])],
        'feature_names': actual_feature_names,
        'coefficients': feature_params,
        'coefficient_pvalues': feature_pvalues,
        'confidence_intervals': feature_conf_int,
    }

    return stats_dict


def compute_feature_values(feature_id, diff_raw, diff_zscore, all_indices, all_pair_names, base_dir, cache):
    """
    计算特征值，带缓存机制
    对于custom特征，会利用全局的 PAIR_FEATURE_CACHE 和 STD_CACHE 避免重复提取
    """
    if feature_id in cache:
        return cache[feature_id]

    if feature_id not in FEATURE_REGISTRY:
        raise KeyError(f"未注册的特征: {feature_id}")

    spec = FEATURE_REGISTRY[feature_id]

    if spec.source == 'diff_raw':
        if spec.index is None:
            raise ValueError(f"特征 {feature_id} 缺少索引")
        values = np.asarray(diff_raw[spec.index, all_indices], dtype=float)
    elif spec.source == 'diff_zscore':
        if spec.index is None:
            raise ValueError(f"特征 {feature_id} 缺少索引")
        values = np.asarray(diff_zscore[spec.index, all_indices], dtype=float)
    elif spec.source == 'custom':
        if spec.extractor is None:
            raise ValueError(f"特征 {feature_id} 未提供提取函数")
        
        # 记录缓存统计的初始值
        initial_pair_hits = _cache_stats['pair_feature_hits']
        initial_std_hits = _cache_stats['std_cache_hits']
        
        values = np.full(len(all_pair_names), np.nan, dtype=float)
        for i, pair_name in enumerate(all_pair_names):
            if spec.progress_interval and spec.progress_interval > 0 and (i + 1) % spec.progress_interval == 0:
                print(f"  {spec.display_name} 提取进度: {i + 1}/{len(all_pair_names)}")
            try:
                values[i] = spec.extractor(pair_name, base_dir)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  特征 {spec.feature_id} 提取失败 {pair_name}: {exc}")
                values[i] = np.nan
        
        # 显示缓存利用情况
        pair_cache_used = _cache_stats['pair_feature_hits'] - initial_pair_hits
        std_cache_used = _cache_stats['std_cache_hits'] - initial_std_hits
        if pair_cache_used > 0 or std_cache_used > 0:
            print(f"    缓存利用: 时间序列缓存命中 {pair_cache_used} 次, std缓存命中 {std_cache_used} 次")
    else:
        raise ValueError(f"未知特征来源: {spec.source}")

    values = np.asarray(values, dtype=float)
    valid_count = int(np.sum(~np.isnan(values)))
    print(f"  特征 {spec.display_name} 提取完成: {valid_count}/{len(values)} 个有效值")

    cache[feature_id] = values
    return values


def build_feature_matrix(feature_ids, diff_raw, diff_zscore, all_indices, all_pair_names, base_dir, cache):
    feature_values = []
    for feature_id in feature_ids:
        values = compute_feature_values(feature_id, diff_raw, diff_zscore, all_indices, all_pair_names, base_dir, cache)
        feature_values.append(values)
    return np.vstack(feature_values)


def parse_transformation_method(feature_id):
    """
    从feature_id中解析出变换方式
    例如：
    - 'loudness_std_diff_raw' -> ('std_diff_raw', None)
    - 'loudness_all5s_relative_diff' -> ('all5s_relative_diff', None)
    - 'loudness_all5s_nonlinear_competition_alpha05' -> ('all5s_nonlinear_competition', 0.5)
    
    返回: (transform_method, alpha) 其中alpha可能为None
    """
    # 去掉前缀（如'loudness_'）
    parts = feature_id.split('_', 1)
    if len(parts) < 2:
        return None, None
    
    transform_part = parts[1]
    
    # 解析alpha值（如果有）
    alpha = None
    if 'alpha05' in transform_part:
        alpha = 0.5
        transform_part = transform_part.replace('_alpha05', '')
    elif 'alpha10' in transform_part:
        alpha = 1.0
        transform_part = transform_part.replace('_alpha10', '')
    
    return transform_part, alpha


def apply_transformation_to_selected_features(pair_name, base_dir, transform_method, alpha, all_pair_names):
    """
    将变换方式应用到选定的9个特征上
    
    参数:
        transform_method: 变换方式（如'all5s_relative_diff', 'std_diff_raw'等）
        alpha: 非线性竞争的alpha值（如果适用）
    
    返回:
        9个特征的变换值列表
    """
    values = []
    
    for feature_key, stat_type in SELECTED_FEATURE_CONFIGS:
        try:
            if transform_method == 'std_diff_raw':
                # 对于std_diff_raw，需要从diff_raw中读取
                # 但这里我们需要根据feature_key和stat_type找到对应的索引
                # 暂时返回NaN，因为std_diff_raw是从预计算的特征矩阵中读取的
                values.append(np.nan)
            elif transform_method == 'all5s_relative_diff':
                val = extract_feature_relative_diff_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0)
                values.append(val)
            elif transform_method == 'all5s_nonlinear_competition':
                if alpha is None:
                    alpha = 0.5
                val = extract_feature_nonlinear_competition_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0, alpha)
                values.append(val)
            elif transform_method == 'all5s_ratio_competition':
                val = extract_feature_ratio_competition_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0)
                values.append(val)
            elif transform_method == 'all5s_competition_index':
                val = extract_feature_competition_index_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0)
                values.append(val)
            else:
                # 未知的变换方式，返回NaN
                values.append(np.nan)
        except Exception as exc:
            print(f"应用变换{transform_method}到{feature_key}({stat_type})失败: {exc}")
            values.append(np.nan)
    
    return values


def build_transformed_features_matrix(feature_id, all_indices, all_pair_names, base_dir, cache, diff_raw=None):
    """
    使用模型的变换方式，应用到选定的9个特征上，构建特征矩阵
    
    参数:
        feature_id: 模型的特征ID（如'loudness_all5s_relative_diff'）
        all_indices: 音频对的索引
        all_pair_names: 音频对名称列表
        base_dir: 音频文件基础目录
        cache: 特征缓存
        diff_raw: 预计算的特征差异矩阵（可选，用于F0和Harmonicity）
    
    返回:
        特征矩阵 (9, n_pairs)
    """
    cache_key = f"transformed_{feature_id}"
    if cache_key in cache:
        return cache[cache_key]
    
    # 解析变换方式
    transform_method, alpha = parse_transformation_method(feature_id)
    
    if transform_method is None:
        # 如果无法解析，返回NaN矩阵
        return np.full((9, len(all_pair_names)), np.nan)
    
    # 对于std_diff_raw，直接从diff_raw中读取
    if transform_method == 'std_diff_raw':
        # 需要从diff_raw中读取对应的特征
        # 这里我们需要根据SELECTED_FEATURE_INDICES来读取
        # 但diff_raw的索引对应关系需要确认
        # 暂时返回NaN，需要进一步处理
        values_matrix = np.full((9, len(all_pair_names)), np.nan)
    else:
        # 对每个音频对应用变换
        values_matrix = []
        # F0和Harmonicity的特征键索引（在SELECTED_FEATURE_CONFIGS中的位置）
        f0_idx = 1  # F0 (mean) 在SELECTED_FEATURE_CONFIGS中的索引
        harmonicity_idx = 2  # Harmonicity (mean) 在SELECTED_FEATURE_CONFIGS中的索引
        
        # diff_raw中的索引映射：0=Spectral Centroid(mean), 1=Spectral Centroid(std), 
        # 2=F0(mean), 3=F0(std), 4=Harmonicity(mean), 5=Harmonicity(std), ...
        f0_raw_idx = 2  # F0 (mean) 在diff_raw中的索引
        harmonicity_raw_idx = 4  # Harmonicity (mean) 在diff_raw中的索引
        
        for pair_idx, pair_name in enumerate(all_pair_names):
            values = []
            for feat_idx, (feature_key, stat_type) in enumerate(SELECTED_FEATURE_CONFIGS):
                # 对于F0和Harmonicity，如果时间序列不可用，则从diff_raw中读取
                if (feature_key == 'f0' or feature_key == 'harmonicity') and diff_raw is not None:
                    try:
                        # 从diff_raw中读取对应的值
                        if feature_key == 'f0' and stat_type == 'mean':
                            raw_idx = f0_raw_idx
                        elif feature_key == 'harmonicity' and stat_type == 'mean':
                            raw_idx = harmonicity_raw_idx
                        else:
                            # 对于std类型，暂时返回NaN（如果需要可以添加）
                            values.append(np.nan)
                            continue
                        
                        # 获取pair_name在all_indices中的位置
                        if pair_idx < len(all_indices):
                            pair_raw_idx = all_indices[pair_idx]
                            val = diff_raw[raw_idx, pair_raw_idx]
                            # 对于F0和Harmonicity，由于时间序列不可用，无法计算需要左右声道原始值的竞争变换
                            # 因此对于这些变换方式，使用diff_raw中的原始差值作为替代
                            # 注意：这不是真正的竞争变换，但至少保留了特征信息
                            if transform_method in ['all5s_relative_diff', 'all5s_nonlinear_competition', 
                                                    'all5s_ratio_competition', 'all5s_competition_index',
                                                    'all5s_std_diff', 'weighted_std_diff']:
                                # 对于需要时间序列的变换，F0和Harmonicity无法计算，返回NaN
                                # 或者使用原始差值作为替代（对于std_diff类型，可以使用std类型的原始差值）
                                if transform_method in ['all5s_std_diff', 'weighted_std_diff']:
                                    # 对于标准差差变换，f0和harmonicity无法计算真正的加权标准差差
                                    # 对于weighted_std_diff，根据stat_type决定：
                                    #   - stat_type == 'std': 使用std差值作为替代
                                    #   - stat_type == 'mean': 使用mean差值作为替代（计算加权平均差）
                                    # 对于all5s_std_diff，如果stat_type是std，使用std差值；如果是mean，返回NaN
                                    if transform_method == 'weighted_std_diff':
                                        # 对于weighted_std_diff，根据stat_type使用对应的差值
                                        # val已经是根据stat_type从diff_raw读取的mean或std差值
                                        values.append(float(val))
                                    elif transform_method == 'all5s_std_diff':
                                        # 对于all5s_std_diff，如果stat_type是std，尝试从diff_raw读取std差值
                                        if stat_type == 'std':
                                            # 尝试找到对应的std索引
                                            if feature_key == 'f0':
                                                std_raw_idx = 3  # F0 (std) 在diff_raw中的索引
                                            elif feature_key == 'harmonicity':
                                                std_raw_idx = 5  # Harmonicity (std) 在diff_raw中的索引
                                            else:
                                                values.append(np.nan)
                                                continue
                                            try:
                                                std_val = diff_raw[std_raw_idx, pair_raw_idx]
                                                values.append(float(std_val))
                                            except:
                                                values.append(np.nan)
                                        else:
                                            # mean类型无法计算std_diff，返回NaN
                                            values.append(np.nan)
                                else:
                                    # 对于竞争变换，使用原始差值（虽然不够准确，但比NaN好）
                                    values.append(float(val))
                            else:
                                # 未知变换方式，返回原始差值
                                values.append(float(val))
                        else:
                            values.append(np.nan)
                    except Exception as exc:
                        print(f"从diff_raw读取{feature_key}({stat_type})失败 {pair_name}: {exc}")
                        values.append(np.nan)
                else:
                    # 对于其他特征，使用正常的变换方式
                    try:
                        if transform_method == 'all5s_relative_diff':
                            val = extract_feature_relative_diff_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0)
                            values.append(val)
                        elif transform_method == 'all5s_nonlinear_competition':
                            if alpha is None:
                                alpha = 0.5
                            val = extract_feature_nonlinear_competition_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0, alpha)
                            values.append(val)
                        elif transform_method == 'all5s_ratio_competition':
                            val = extract_feature_ratio_competition_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0)
                            values.append(val)
                        elif transform_method == 'all5s_competition_index':
                            val = extract_feature_competition_index_with_stat(pair_name, base_dir, feature_key, stat_type, 0.0, 5.0)
                            values.append(val)
                        elif transform_method == 'all5s_std_diff':
                            # 全部5秒的标准差差
                            # 对于f0和harmonicity，它们没有时间序列，无法计算标准差差
                            if feature_key in ['f0', 'harmonicity']:
                                values.append(np.nan)
                            else:
                                val = extract_feature_segment_std_diff(pair_name, base_dir, feature_key, 0.0, 5.0)
                                values.append(val)
                        elif transform_method == 'weighted_std_diff':
                            # 加权统计量差：前1秒*0.6 + 中间3秒*0.2 + 最后1秒*0.4
                            # 根据stat_type决定使用std还是mean：
                            # - stat_type == 'std': 计算加权标准差差
                            # - stat_type == 'mean': 计算加权平均差
                            if feature_key in ['f0', 'harmonicity']:
                                # 对于f0和harmonicity，它们没有时间序列，从diff_raw读取对应的值
                                if diff_raw is not None and pair_idx < len(all_indices):
                                    try:
                                        pair_raw_idx = all_indices[pair_idx]
                                        if feature_key == 'f0':
                                            if stat_type == 'mean':
                                                raw_idx = 2  # F0 (mean) 在diff_raw中的索引
                                            else:  # std
                                                raw_idx = 3  # F0 (std) 在diff_raw中的索引
                                        elif feature_key == 'harmonicity':
                                            if stat_type == 'mean':
                                                raw_idx = 4  # Harmonicity (mean) 在diff_raw中的索引
                                            else:  # std
                                                raw_idx = 5  # Harmonicity (std) 在diff_raw中的索引
                                        else:
                                            values.append(np.nan)
                                            continue
                                        raw_val = diff_raw[raw_idx, pair_raw_idx]
                                        values.append(float(raw_val))  # 使用diff_raw中的值作为替代
                                    except Exception as e:
                                        # 添加调试信息
                                        if not hasattr(build_transformed_features_matrix, '_debug_f0_harm'):
                                            print(f"  警告: 从diff_raw读取{feature_key}({stat_type})失败: {e}")
                                            build_transformed_features_matrix._debug_f0_harm = True
                                        values.append(np.nan)
                                else:
                                    values.append(np.nan)
                            else:
                                # 对于有时间序列的特征，根据stat_type选择使用std还是mean
                                if stat_type == 'std':
                                    # 计算加权标准差差
                                    val = extract_feature_weighted_std_diff(pair_name, base_dir, feature_key)
                                elif stat_type == 'mean':
                                    # 计算加权平均差
                                    val = extract_feature_weighted_diff(pair_name, base_dir, feature_key)
                                else:
                                    val = np.nan
                                values.append(val)
                                # 添加调试信息（只在第一个样本时）
                                if pair_idx == 0 and feat_idx == 0 and not hasattr(build_transformed_features_matrix, '_debug_weighted'):
                                    print(f"  调试: {pair_name} 特征 {feature_key} ({stat_type}) 加权统计量差 = {val}")
                                    build_transformed_features_matrix._debug_weighted = True
                        else:
                            values.append(np.nan)
                    except Exception as exc:
                        print(f"应用变换{transform_method}到{feature_key}({stat_type})失败: {exc}")
                        values.append(np.nan)
            values_matrix.append(values)
        values_matrix = np.array(values_matrix).T  # 转置为 (9, n_pairs)
    
    cache[cache_key] = values_matrix
    return values_matrix


def _is_nan(value):
    try:
        return np.isnan(value)
    except Exception:  # pylint: disable=broad-except
        return False


def format_float_list(values, precision=6, scientific=False):
    formatted = []
    for value in values:
        if _is_nan(value):
            formatted.append('nan')
        else:
            if scientific:
                formatted.append(f"{value:.3e}")
            else:
                formatted.append(f"{value:.{precision}f}")
    return '|'.join(formatted)


def split_confidence_bounds(confidence_intervals):
    lower = []
    upper = []
    for ci in confidence_intervals:
        if ci is None or len(ci) != 2 or _is_nan(ci[0]) or _is_nan(ci[1]):
            lower.append(float('nan'))
            upper.append(float('nan'))
        else:
            lower.append(ci[0])
            upper.append(ci[1])
    return lower, upper


def safe_folder_name(name: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fa5]+', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    if not sanitized:
        sanitized = 'model'
    return sanitized[:80]


def main():
    print("=== 使用选定9个声学特征的回归模型分析：五折交叉验证 ===")
    print("选定的9个特征:")
    for i, (idx, name) in enumerate(zip(SELECTED_FEATURE_INDICES, SELECTED_FEATURE_NAMES)):
        print(f"  {i + 1}. {name} (feature_idx={idx})")
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, 'linear_regression_5fold_selected_features')
    os.makedirs(output_dir, exist_ok=True)

    base_dir = DEFAULT_AUDIO_BASE_DIR
    print(f"\n音频根目录: {base_dir}")

    print("\n步骤1: 加载特征数据")
    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)

    print("\n步骤2: 合并main和sub类型的数据")
    merged_results = merge_balanced_pairs_by_type(pair_names)

    all_indices = []
    all_pair_names = []
    for exp_type in ['main', 'sub', 'nn_main', 'nn_sub']:
        if exp_type in merged_results:
            all_indices.extend(merged_results[exp_type]['indices'])
            all_pair_names.extend(merged_results[exp_type]['pair_names'])

    print(f"\n总共合并: {len(all_pair_names)} 个音频对 (main, sub, nn_main, nn_sub)")

    print("\n步骤3: 加载人类决策数据")
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
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
        r"D:\D\research\audioset下载\experiment_output2\Liu Yaorui2.csv"
    ]

    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
    ] * 9

    all_trials = load_all_human_decisions(txt_files, csv_files, mapping_files)

    print("\n步骤4: 计算选择概率")
    probabilities = calculate_selection_probability_144(all_pair_names, all_trials)

    print("\n步骤5: 准备选定的9个声学特征")
    # 将特征索引从1开始转换为从0开始
    selected_indices_0based = [idx - 1 for idx in SELECTED_FEATURE_INDICES]
    # 从diff_zscore中提取选定的9个特征（使用前18个声学特征）
    n_acoustic_features = 18
    acoustic_features = diff_zscore[:n_acoustic_features, :]
    selected_features_matrix = acoustic_features[selected_indices_0based, :]
    print(f"选定的9个特征矩阵形状: {selected_features_matrix.shape}")
    print("选定的特征:")
    for i, (idx, name) in enumerate(zip(SELECTED_FEATURE_INDICES, SELECTED_FEATURE_NAMES)):
        print(f"  {i + 1}. {name} (feature_idx={idx}, 数组索引={selected_indices_0based[i]})")

    print("\n步骤6: 五折交叉验证评估")
    print(f"计划评估 {len(MODEL_CONFIGS)} 个模型")
    print("注意: 每个模型将使用选定的9个声学特征进行多特征回归")
    
    # 清空缓存，确保使用最新的提取函数（避免旧缓存中的数据类型问题）
    PAIR_FEATURE_CACHE.clear()
    STD_CACHE.clear()
    
    # 重置缓存统计
    _cache_stats['pair_feature_hits'] = 0
    _cache_stats['pair_feature_misses'] = 0
    _cache_stats['std_cache_hits'] = 0
    _cache_stats['std_cache_misses'] = 0

    feature_cache: Dict[str, np.ndarray] = {}
    all_results: List[Dict[str, object]] = []

    if len(all_pair_names) == 0:
        print("警告: 未找到任何音频对，结束流程")
        return

    # 准备五折数据
    X_full = None
    y_full = probabilities
    valid_mask = None

    for model_idx, config in enumerate(MODEL_CONFIGS, 1):
        model_name = config['model_name']
        feature_ids = list(config['feature_ids'])
        print(f"\n--- 模型 {model_idx}/{len(MODEL_CONFIGS)}: {model_name} ---")
        print(f"原始特征ID: {', '.join(feature_ids)}")

        # 检查是否是投票特征（投票特征是完整特征，不需要应用到9个特征上）
        is_voting_feature = any('vote' in fid for fid in feature_ids)
        
        if is_voting_feature:
            # 投票特征：直接提取特征值，不需要应用到9个特征上
            print(f"检测到投票特征，直接提取特征值")
            feature_matrix = build_feature_matrix(feature_ids, diff_raw, diff_zscore, all_indices, all_pair_names, base_dir, feature_cache)
            feature_specs = [FEATURE_REGISTRY[fid] for fid in feature_ids]
            feature_names = [spec.display_name for spec in feature_specs]
            print(f"实际使用: {len(feature_ids)}个投票特征")
        else:
            # 非投票特征：使用变换方式应用到9个特征上
            transform_method, alpha = parse_transformation_method(feature_ids[0])
            print(f"变换方式: {transform_method}" + (f" (alpha={alpha})" if alpha is not None else ""))

            if transform_method == 'std_diff_raw':
                # 对于std_diff_raw，从diff_raw中读取对应的特征
                # SELECTED_FEATURE_INDICES对应diff_raw中的索引（需要确认索引对应关系）
                # diff_raw的索引：0=Spectral Centroid(mean), 1=Spectral Centroid(std), ...
                # SELECTED_FEATURE_INDICES: [2, 3, 4, 7, 8, 9, 10, 12, 17] 从1开始计数
                # 转换为从0开始的索引：[1, 2, 3, 6, 7, 8, 9, 11, 16]
                selected_indices_0based = [idx - 1 for idx in SELECTED_FEATURE_INDICES]
                feature_matrix = diff_raw[selected_indices_0based, :][:, all_indices]
                feature_names = [f"{name} (std_diff_raw)" for name in SELECTED_FEATURE_NAMES]
            else:
                # 使用变换方式应用到9个特征上
                # 传入diff_raw以便F0和Harmonicity可以从预计算数据中读取
                feature_matrix = build_transformed_features_matrix(feature_ids[0], all_indices, all_pair_names, base_dir, feature_cache, diff_raw=diff_raw)
                feature_names = [f"{name} ({transform_method})" for name in SELECTED_FEATURE_NAMES]
            
            print(f"实际使用: 9个特征，每个特征使用{transform_method}变换方式")
        
        X, y, valid_mask_current = prepare_regression_data(feature_matrix, probabilities)

        if len(y) < 10:
            print(f"  警告: 有效样本数过少 ({len(y)} < 10)，跳过该模型")
            continue

        # 五折交叉验证
        folds, indices = build_consistent_folds(len(y), seed=CV_SEED, k=N_FOLDS)

        fold_results = []
        cv_r2_scores = []
        cv_mse_scores = []
        cv_mae_scores = []

        for fold_idx, test_idx in enumerate(folds, 1):
            print(f"  折 {fold_idx}/5: ", end="")

            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 训练模型
            model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

            # 预测
            y_pred = model.predict(sm.add_constant(X_test))

            # 计算指标
            r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))

            cv_r2_scores.append(r2)
            cv_mse_scores.append(mse)
            cv_mae_scores.append(mae)

            fold_results.append({
                'fold': fold_idx,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })

            print(f"R^2 = {r2:.4f}")
        # 计算平均性能
        avg_r2 = np.mean(cv_r2_scores)
        avg_mse = np.mean(cv_mse_scores)
        avg_mae = np.mean(cv_mae_scores)

        print("\n  五折平均性能:")
        print(f"  R^2 = {avg_r2:.4f} ± {np.std(cv_r2_scores):.4f}")
        print(f"  MSE = {avg_mse:.4f} ± {np.std(cv_mse_scores):.4f}")
        print(f"  MAE = {avg_mae:.4f} ± {np.std(cv_mae_scores):.4f}")
        # 使用全数据训练最终模型（用于参数估计）
        final_model = fit_ols_model(X, y, feature_names)
        stats_dict = extract_model_statistics(final_model, feature_names)

        print("\n  最终模型参数 (全数据训练):")
        print(f"  R^2 = {stats_dict['rsquared']:.4f}")
        print(f"  截距 = {stats_dict['intercept']:.4f}, p = {stats_dict['intercept_pvalue']:.4e}")

        for name, coef, pval, ci in zip(
                stats_dict['feature_names'],
                stats_dict['coefficients'],
                stats_dict['coefficient_pvalues'],
                stats_dict['confidence_intervals']):
            ci_lower, ci_upper = ci
            print(f"  {name}: β = {coef:.4f}, p = {pval:.4e}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

        # 创建模型文件夹
        model_folder = os.path.join(output_dir, safe_folder_name(model_name))
        os.makedirs(model_folder, exist_ok=True)

        # 保存折间结果
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(os.path.join(model_folder, 'fold_results.csv'), index=False, encoding='utf-8-sig')

        # 保存最终模型统计
        coef_ci_lower, coef_ci_upper = split_confidence_bounds(stats_dict['confidence_intervals'])

        coeff_df = pd.DataFrame({
            'feature': stats_dict['feature_names'],
            'coefficient': stats_dict['coefficients'],
            'pvalue': stats_dict['coefficient_pvalues'],
            'ci_lower': coef_ci_lower,
            'ci_upper': coef_ci_upper
        })
        coeff_df.to_csv(os.path.join(model_folder, 'coefficients.csv'), index=False, encoding='utf-8-sig')

        # 保存模型摘要
        model_summary = {
            'model_idx': model_idx,
            'model_name': model_name,
            'feature_ids': '|'.join(feature_ids),
            'feature_names': '|'.join(feature_names),
            'n_features': len(feature_ids),
            'n_samples': len(y),
            'cv_r2_mean': avg_r2,
            'cv_r2_std': np.std(cv_r2_scores),
            'cv_mse_mean': avg_mse,
            'cv_mse_std': np.std(cv_mse_scores),
            'cv_mae_mean': avg_mae,
            'cv_mae_std': np.std(cv_mae_scores),
            'final_rsquared': stats_dict['rsquared'],
            'final_rsquared_adj': stats_dict['rsquared_adj'],
            'aic': stats_dict['aic'],
            'bic': stats_dict['bic'],
            'intercept': stats_dict['intercept'],
            'intercept_pvalue': stats_dict['intercept_pvalue'],
            'intercept_ci_lower': stats_dict['intercept_ci'][0],
            'intercept_ci_upper': stats_dict['intercept_ci'][1]
        }

        pd.DataFrame([model_summary]).to_csv(
            os.path.join(model_folder, 'summary.csv'), index=False, encoding='utf-8-sig'
        )

        all_results.append({
            'model_idx': model_idx,
            'model_name': model_name,
            'feature_ids': '|'.join(feature_ids),
            'feature_names': '|'.join(feature_names),
            'n_features': len(feature_ids),
            'n_samples': len(y),
            'cv_r2_mean': avg_r2,
            'cv_r2_std': np.std(cv_r2_scores),
            'cv_mse_mean': avg_mse,
            'cv_mse_std': np.std(cv_mse_scores),
            'cv_mae_mean': avg_mae,
            'cv_mae_std': np.std(cv_mae_scores),
            'final_rsquared': stats_dict['rsquared'],
            'final_rsquared_adj': stats_dict['rsquared_adj'],
            'aic': stats_dict['aic'],
            'bic': stats_dict['bic'],
            'intercept': stats_dict['intercept'],
            'intercept_pvalue': stats_dict['intercept_pvalue'],
            'intercept_ci_lower': stats_dict['intercept_ci'][0],
            'intercept_ci_upper': stats_dict['intercept_ci'][1],
            'coefficients': format_float_list(stats_dict['coefficients']),
            'coefficient_pvalues': format_float_list(stats_dict['coefficient_pvalues'], scientific=True),
            'coef_ci_lower': format_float_list(coef_ci_lower),
            'coef_ci_upper': format_float_list(coef_ci_upper)
        })

    print("\n步骤7: 保存汇总结果")

    if not all_results:
        print("未生成任何模型结果，请检查配置或数据")
        return

    results_df = pd.DataFrame(all_results)
    columns_order = [
        'model_idx', 'model_name', 'feature_ids', 'feature_names', 'n_features', 'n_samples',
        'cv_r2_mean', 'cv_r2_std', 'cv_mse_mean', 'cv_mse_std', 'cv_mae_mean', 'cv_mae_std',
        'final_rsquared', 'final_rsquared_adj', 'aic', 'bic',
        'intercept', 'intercept_pvalue', 'intercept_ci_lower', 'intercept_ci_upper',
        'coefficients', 'coefficient_pvalues', 'coef_ci_lower', 'coef_ci_upper'
    ]
    results_df = results_df[columns_order]

    output_path = os.path.join(output_dir, 'selected_features_models_5fold_summary.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {output_path}")

    print("\n五折交叉验证汇总表格:")
    print(results_df[['model_name', 'cv_r2_mean', 'cv_r2_std', 'final_rsquared']].to_string(index=False))
    
    # 显示缓存统计
    print("\n=== 缓存统计 ===")
    print(f"时间序列缓存:")
    print(f"  命中: {_cache_stats['pair_feature_hits']} 次")
    print(f"  未命中: {_cache_stats['pair_feature_misses']} 次")
    if _cache_stats['pair_feature_hits'] + _cache_stats['pair_feature_misses'] > 0:
        hit_rate = _cache_stats['pair_feature_hits'] / (_cache_stats['pair_feature_hits'] + _cache_stats['pair_feature_misses']) * 100
        print(f"  命中率: {hit_rate:.1f}%")
    print(f"std缓存:")
    print(f"  命中: {_cache_stats['std_cache_hits']} 次")
    print(f"  未命中: {_cache_stats['std_cache_misses']} 次")
    if _cache_stats['std_cache_hits'] + _cache_stats['std_cache_misses'] > 0:
        hit_rate = _cache_stats['std_cache_hits'] / (_cache_stats['std_cache_hits'] + _cache_stats['std_cache_misses']) * 100
        print(f"  命中率: {hit_rate:.1f}%")
    print(f"已缓存的时间序列数量: {len(PAIR_FEATURE_CACHE)}")
    print(f"已缓存的std值数量: {len(STD_CACHE)}")

    print("\n=== 所有分析完成 ===")


if __name__ == '__main__':
    main()

