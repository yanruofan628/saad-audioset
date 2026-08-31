#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindMix微调脚本用于hanglei数据
基于MindMix_clip_finetune.py的模型架构，使用eeg_audio_fusion_classify_hanglei.py的数据加载逻辑
使用全部数据（不筛选）进行微调训练
"""
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 禁用特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", message=".*?torch.distributed.*?")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange
from timm.models import create_model
import argparse
from collections import OrderedDict
import torch.distributed as dist
from transformers import Wav2Vec2Model
from tqdm import tqdm
import json
# 不使用sklearn的KFold，使用与individual_choice_with_position.py一致的build_consistent_folds
from linear_regression_selected_features import build_consistent_folds
from scipy import signal
import librosa
from pathlib import Path
import sys

# 禁用transformers的警告
import transformers
transformers.logging.set_verbosity_error()

# 设置torch的警告级别
torch.set_warn_always(False)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据加载函数
from eegnet_classify_trials_5s import (
    find_data_files, load_meta, reshape_trials, 
    apply_baseline_correction, select_time_segment, select_last_seconds,
    exponential_moving_standardize, DataMeta
)
from individual_choice_with_position import (
    check_left_right_consistency, build_global_stage_dataset,
    extract_features_for_attention_model, parse_pair_name
)
from linear_regression_loudness_models import (
    parse_stereo_pair_name, load_mono_5s, DEFAULT_AUDIO_BASE_DIR,
    parse_experiment_data_txt, parse_experiment_data_csv
)

# 导入模型相关类
# 注意：MindMix_clip_finetune.py 会导入 utils，而 utils 会导入 h5py
# 如果 h5py 导入失败，创建一个假的 h5py 模块来避免错误
import sys
from types import ModuleType

# 在导入 MindMix_clip_finetune 之前，先处理 h5py 导入问题
if 'h5py' not in sys.modules:
    try:
        import h5py
    except (ImportError, OSError):
        # 如果 h5py 导入失败（DLL 问题），创建一个假的 h5py 模块
        # 因为 utils.py 导入了 h5py 但实际上没有使用它
        fake_h5py = ModuleType('h5py')
        sys.modules['h5py'] = fake_h5py

from MindMix_clip_finetune import (
    get_args, get_models, EEGEncoder, CLIPModel, ClipLoss,
    train_model, evaluate_model, set_random_seed
)

# 数据配置
DATA_NAME = "hanglei"
SAMPLING_RATE = 250
TARGET_SAMPLING_RATE = 200  # 降采样目标采样率
PROCESSED_DATA_PATH = 'A:/standard_data_noica/hanglei'

# 音频路径配置
AUDIO_BASE_DIR = DEFAULT_AUDIO_BASE_DIR
MAPPING_BASE_DIR = r"D:\D\research\audioset下载\audio_pairs_2s\分组音频"
MAPPING_FILES = [
    os.path.join(MAPPING_BASE_DIR, "folder1", "file_mapping_folder1.csv"),
    os.path.join(MAPPING_BASE_DIR, "folder2", "file_mapping_folder2.csv"),
    os.path.join(MAPPING_BASE_DIR, "folder3", "file_mapping_folder3.csv"),
]

# Benchmark文件路径
BENCHMARK_FILES = [
    r"A:\hanglei_1_20251224_060149.mff\benchmark_1_10-1-1.txt",
    r"A:\hanglei_2_20251224_063021.mff\benchmark_1_10-1-2.txt",
    r"A:\hanglei_3_20251224_065517.mff\benchmark_1_10-1-3.txt",
]

# 输出目录
OUTPUT_DIR = os.path.join(PROCESSED_DATA_PATH, "mindmix_finetune_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交叉验证配置
CV_SEED = 42
N_FOLDS = 5

# Wav2Vec2要求的采样率
AUDIO_TARGET_SR = 16000


# ============== 分布式训练辅助函数（从utils.py复制，避免h5py依赖） ==============
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))


def init_distributed_mode(args):
    """初始化分布式训练模式（避免导入h5py）"""
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def create_ds_config(args):
    """创建DeepSpeed配置文件（避免导入h5py）"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def downsample_eeg_to_200hz(eeg_data, original_fs=250, target_fs=200):
    """
    将EEG数据从250Hz降采样到200Hz
    
    输入: (n_trials, n_channels, 500) - 250Hz采样率，2秒
    输出: (n_trials, n_channels, 400) - 200Hz采样率，2秒
    
    降采样公式：target_length = original_length * (target_fs / original_fs)
    400 = 500 * (200 / 250)
    """
    n_trials, n_channels, n_timepoints = eeg_data.shape
    target_length = int(n_timepoints * (target_fs / original_fs))  # 400
    
    print(f"降采样EEG数据: {n_timepoints}个时间点 -> {target_length}个时间点")
    print(f"采样率: {original_fs}Hz -> {target_fs}Hz")
    
    resampled = np.zeros((n_trials, n_channels, target_length), dtype=np.float32)
    
    for i in range(n_trials):
        for j in range(n_channels):
            resampled[i, j] = signal.resample(eeg_data[i, j], target_length)
    
    return resampled


def load_and_resample_audio(audio_path, target_sr=16000):
    """
    加载音频并重采样到16kHz（Wav2Vec2Model要求）
    
    参数:
        audio_path: 音频文件路径
        target_sr: 目标采样率（默认16kHz）
    
    返回:
        y: 音频波形（1D数组）
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y.astype(np.float32)
    except Exception as e:
        print(f"加载音频失败 {audio_path}: {e}")
        return None


def load_audio_pair(pair_name, base_dir):
    """
    加载音频对的左右两个音频文件
    
    参数:
        pair_name: 音频对名称（如 "main_cat1_id1+main_cat2_id2"）
        base_dir: 音频文件基础目录
    
    返回:
        (left_audio, right_audio): 左右音频波形（1D数组，16kHz采样率）
    """
    try:
        # 解析音频对名称
        left_cat, left_id, right_cat, right_id = parse_stereo_pair_name(pair_name, base_dir)
        
        # 构建路径
        left_path = os.path.join(base_dir, left_cat, f"{left_id}.wav")
        right_path = os.path.join(base_dir, right_cat, f"{right_id}.wav")
        
        # 加载并重采样
        left_audio = load_and_resample_audio(left_path, target_sr=AUDIO_TARGET_SR)
        right_audio = load_and_resample_audio(right_path, target_sr=AUDIO_TARGET_SR)
        
        if left_audio is None or right_audio is None:
            return None, None
        
        return left_audio, right_audio
    except Exception as e:
        print(f"加载音频对失败 {pair_name}: {e}")
        return None, None


def parse_audio_mapping_custom(file_path):
    """
    解析音频映射文件（自定义版本，适配实际的CSV格式）
    返回映射字典：{映射后的名字: 原始名字}
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        
        # 检查是否是标准格式（有new_name和original_name列）
        if 'new_name' in df.columns and 'original_name' in df.columns:
            for _, row in df.iterrows():
                new_name = str(row['new_name']).strip()
                original_name = str(row['original_name']).strip()
                
                if pd.isna(row['new_name']) or pd.isna(row['original_name']) or not new_name or not original_name:
                    continue
                
                mapping[new_name] = original_name
                if new_name.lower().endswith('.wav'):
                    mapping[new_name[:-4]] = original_name
        elif len(df.columns) >= 2:
            # 自定义格式：第一列是原始名称，第二列是映射后的文件名
            col1 = df.columns[0]
            col2 = df.columns[1]
            
            for idx, row in df.iterrows():
                original_name = str(row[col1]).strip()
                new_name = str(row[col2]).strip()
                
                if pd.isna(row[col1]) or pd.isna(row[col2]) or not original_name or not new_name:
                    continue
                
                mapping[new_name] = original_name
                if new_name.lower().endswith('.wav'):
                    mapping[new_name[:-4]] = original_name
        
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_all_mappings():
    """加载所有三个folder的映射文件"""
    all_mappings = {}
    folder_ranges = [
        (0, 159, 0),
        (159, 318, 1),
        (318, 478, 2),
    ]
    
    mappings = []
    for mapping_file in MAPPING_FILES:
        if os.path.exists(mapping_file):
            mapping = parse_audio_mapping_custom(mapping_file)
            mappings.append(mapping)
            print(f"加载映射文件: {mapping_file}, 映射数量: {len(mapping) // 2 if any(k.endswith('.wav') for k in mapping) else len(mapping)}")
        else:
            print(f"警告: 映射文件不存在: {mapping_file}")
            mappings.append({})
    
    return mappings, folder_ranges


def load_eeg_data():
    """加载hanglei的EEG数据"""
    print("\n=== 加载EEG数据 ===")
    data_dir = Path(PROCESSED_DATA_PATH)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {PROCESSED_DATA_PATH}")
    
    trials_path, labels_path, info_path = find_data_files(data_dir)
    
    # 加载元信息
    meta = load_meta(info_path, SAMPLING_RATE, None)
    print(f"采样率: {meta.sampling_rate_hz} Hz, 通道数: {meta.num_channels}")
    
    # 加载数据
    trials = np.load(trials_path)
    labels_df = pd.read_csv(labels_path)
    
    # 获取标签
    if 'Label' in labels_df.columns:
        labels = labels_df['Label'].to_numpy()
    else:
        if labels_df.shape[1] >= 2:
            labels = labels_df.iloc[:, 1].to_numpy()
        else:
            labels = labels_df.iloc[:, 0].to_numpy()
    
    # 形状处理
    trials_3d = reshape_trials(trials, meta.num_channels)
    print(f"原始trials形状: {trials_3d.shape}")
    
    # 获取trial时长
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    trial_duration = info.get("trial_duration")
    if trial_duration is None:
        trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz
    print(f"Trial时长: {trial_duration:.2f}秒")
    
    # 截取完整trial数据
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    
    # 基线矫正：使用前3秒作为基线
    print("进行基线矫正...")
    data_baseline_corrected = apply_baseline_correction(
        data_full,
        fs=meta.sampling_rate_hz,
        baseline_start=0.0,
        baseline_end=3.0
    )
    
    # 提取3-5秒用于特征提取（刺激期）
    if trial_duration < 5.0:
        raise ValueError(f"数据时长{trial_duration}秒不足以提取3-5秒的刺激期")
    
    stimulus_start_seconds = 3.0
    stimulus_duration = 2.0
    stimulus_start_idx = int(round(stimulus_start_seconds * meta.sampling_rate_hz))
    stimulus_end_idx = int(round((stimulus_start_seconds + stimulus_duration) * meta.sampling_rate_hz))
    data = data_baseline_corrected[:, :, stimulus_start_idx:stimulus_end_idx]
    print(f"提取3-5秒（刺激期）后形状: {data.shape}")
    
    # 确保数据和标签数量匹配
    if data.shape[0] != labels.shape[0]:
        min_n = min(data.shape[0], labels.shape[0])
        print(f"警告: trial数量不匹配，使用前{min_n}个")
        data = data[:min_n]
        labels = labels[:min_n]
    
    return data, labels, meta


def load_hanglei_trials(benchmark_files=None, mappings=None, folder_ranges=None):
    """
    加载hanglei的trial信息（三个session分别对应三个folder）
    
    参数:
        benchmark_files: benchmark文件路径列表（3个文件）
        mappings: 三个folder的映射文件列表
        folder_ranges: folder范围列表
    
    返回:
        all_trials: 包含original_name和response的trial列表
    """
    print("\n=== 加载hanglei的trial信息 ===")
    all_trials = []
    
    if benchmark_files is None or len(benchmark_files) != 3:
        raise ValueError("需要提供3个benchmark文件路径")
    
    # 依次加载三个session的benchmark文件
    for session_idx, benchmark_file in enumerate(benchmark_files):
        if not os.path.exists(benchmark_file):
            print(f"警告: benchmark文件不存在: {benchmark_file}")
            continue
        
        print(f"\n加载session {session_idx + 1}: {benchmark_file}")
        parsed_trials = parse_experiment_data_txt(benchmark_file)
        print(f"  解析了 {len(parsed_trials)} 个trials")
        
        # 获取对应的映射文件
        mapping = mappings[session_idx] if session_idx < len(mappings) else mappings[0]
        
        for trial in parsed_trials:
            wavfile = trial.get('wavfile')
            if not wavfile:
                continue
            
            filename = os.path.basename(wavfile)
            
            # 应用映射
            if filename in mapping:
                original_name = mapping[filename]
            elif filename.replace('.wav', '') in mapping:
                original_name = mapping[filename.replace('.wav', '')]
            else:
                # 如果找不到映射，跳过这个trial
                continue
            
            all_trials.append({
                'original_name': original_name,
                'response': trial.get('response'),
                'reaction_time': trial.get('reaction_time'),
                'session': session_idx + 1
            })
    
    print(f"\n总共加载了 {len(all_trials)} 个trials")
    print(f"  Session 1: {sum(1 for t in all_trials if t.get('session') == 1)} 个trials")
    print(f"  Session 2: {sum(1 for t in all_trials if t.get('session') == 2)} 个trials")
    print(f"  Session 3: {sum(1 for t in all_trials if t.get('session') == 3)} 个trials")
    
    return all_trials


class DownstreamDataset(Dataset):
    """下游任务数据集类"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        eeg = sample['eeg']
        target_audio = sample['target_audio']
        negative_audio = sample['negetive_audio']  # 注意：原始代码中拼写为negetive
        label = sample['attended_label']
        
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        target_audio_tensor = torch.tensor(target_audio, dtype=torch.float32)
        negative_audio_tensor = torch.tensor(negative_audio, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return eeg_tensor, target_audio_tensor, negative_audio_tensor, label_tensor


def build_dataset(eeg_data, all_trials, meta):
    """
    构建数据集DataFrame
    
    参数:
        eeg_data: EEG数据 (n_trials, n_channels, 400) - 已降采样
        all_trials: trial信息列表
        meta: EEG元信息
    
    返回:
        df: 包含eeg、target_audio、negative_audio、attended_label的DataFrame
    """
    print("\n=== 构建数据集 ===")
    
    # 创建pair_name到EEG索引的映射（通过trial顺序匹配）
    # 假设eeg_data和all_trials的顺序是对应的
    pair_name_to_eeg_idx = {}
    for eeg_idx in range(min(len(eeg_data), len(all_trials))):
        trial = all_trials[eeg_idx]
        pair_name = trial.get('original_name')
        if pair_name and pair_name not in pair_name_to_eeg_idx:
            pair_name_to_eeg_idx[pair_name] = eeg_idx
    
    dataset_rows = []
    valid_count = 0
    skip_count = 0
    
    # 遍历所有trials
    for trial_idx, trial in enumerate(all_trials):
        pair_name = trial['original_name']
        response = trial.get('response')
        
        # 跳过没有response的trial
        if response is None or response not in [1, 2]:
            skip_count += 1
            continue
        
        # 查找对应的EEG数据索引
        # 优先使用trial索引，如果超出范围则使用pair_name映射
        if trial_idx < len(eeg_data):
            matching_eeg_idx = trial_idx
        elif pair_name in pair_name_to_eeg_idx:
            matching_eeg_idx = pair_name_to_eeg_idx[pair_name]
        else:
            skip_count += 1
            continue
        
        # 如果找不到匹配的EEG数据，跳过
        if matching_eeg_idx >= len(eeg_data):
            skip_count += 1
            continue
        
        # 加载音频对
        left_audio, right_audio = load_audio_pair(pair_name, AUDIO_BASE_DIR)
        if left_audio is None or right_audio is None:
            skip_count += 1
            continue
        
        # 根据response确定target和negative
        # response=1表示选左，response=2表示选右
        if response == 1:
            target_audio = left_audio
            negative_audio = right_audio
            attended_label = 1.0
        else:  # response == 2
            target_audio = right_audio
            negative_audio = left_audio
            attended_label = 1.0
        
        # 获取EEG数据 (n_channels, 400)
        eeg_sample = eeg_data[matching_eeg_idx]  # (n_channels, 400)
        
        dataset_rows.append({
            'eeg': eeg_sample,
            'target_audio': target_audio,
            'negetive_audio': negative_audio,  # 注意：原始代码中拼写为negetive
            'attended_label': attended_label
        })
        valid_count += 1
        
        if (valid_count + skip_count) % 100 == 0:
            print(f"  处理进度: {valid_count} 有效, {skip_count} 跳过")
    
    print(f"\n数据集构建完成:")
    print(f"  有效样本: {valid_count}")
    print(f"  跳过样本: {skip_count}")
    
    df = pd.DataFrame(dataset_rows)
    return df


def cross_validate_model(args, ds_init, data, device, subject_id=""):
    """交叉验证函数"""
    # 使用与eeg_audio_fusion_classify_hanglei.py一致的折划分方法
    folds, shuffled_indices = build_consistent_folds(len(data), seed=args.seed, k=args.n_folds)
    accuracies = []

    for fold_idx, val_idx in enumerate(folds, 1):
        print(f"\n=== Subject: {subject_id} | Fold {fold_idx}/{args.n_folds} ===")
        
        # 使用与eeg_audio_fusion_classify_hanglei.py一致的划分方式
        train_idx = np.setdiff1d(shuffled_indices, val_idx, assume_unique=True)
        
        # 创建模型
        eeg_model = EEGEncoder(args, device)
        audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        model = CLIPModel(eeg_model, audio_model)
        
        # 加载预训练权重
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            model.load_pretrained_weights(args.pretrained_model)
        else:
            print("Warning: No pretrained model found, training from scratch")

        # 准备数据
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        train_dataset = DownstreamDataset(train_data)
        val_dataset = DownstreamDataset(val_data)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem
        )

        # 训练模型
        trained_model, train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, args, device
        )

        # 最终评估
        clip_loss_fn = ClipLoss(initial_temp=args.temperature)
        clip_loss_fn.to(device)
        _, final_accuracy = evaluate_model(trained_model, val_loader, device, clip_loss_fn)
        accuracies.append(final_accuracy)
        
        # 保存当前fold的结果
        fold_results = {
            'subject_id': subject_id,
            'fold': fold + 1,
            'final_accuracy': final_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        fold_save_path = os.path.join(args.output_dir, f'{subject_id}_fold_{fold+1}_results.json')
        with open(fold_save_path, 'w') as f:
            json.dump(fold_results, f, indent=2)

    # 计算总体结果
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = f"=== Subject: {subject_id} Cross-Validation Results ===\n"
    results += f"Fold Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n"
    results += f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n"
    results += f"Best Fold Accuracy: {max(accuracies):.4f}\n\n"

    print(results)

    # 保存结果到txt文件
    results_file = os.path.join(args.output_dir, "finetune_results.txt")
    with open(results_file, "a") as f:
        f.write(results)
    
    return mean_accuracy, std_accuracy, accuracies


def main():
    """主函数"""
    args, ds_init = get_args()
    
    # 设置hanglei特定的参数
    args.output_dir = OUTPUT_DIR
    args.n_folds = N_FOLDS
    args.seed = CV_SEED
    
    set_random_seed(args.seed)
    
    # 初始化分布式训练（如果需要）
    init_distributed_mode(args)

    if ds_init is not None:
        create_ds_config(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载EEG数据
    print(f"\n{'='*60}")
    print("步骤1: 加载EEG数据")
    print(f"{'='*60}")
    eeg_data, labels, meta = load_eeg_data()
    print(f"EEG数据形状: {eeg_data.shape}")
    
    # 2. 降采样EEG数据（250Hz -> 200Hz）
    print(f"\n{'='*60}")
    print("步骤2: 降采样EEG数据")
    print(f"{'='*60}")
    eeg_data_downsampled = downsample_eeg_to_200hz(eeg_data, original_fs=SAMPLING_RATE, target_fs=TARGET_SAMPLING_RATE)
    print(f"降采样后EEG数据形状: {eeg_data_downsampled.shape}")
    
    # 3. 加载音频映射文件
    print(f"\n{'='*60}")
    print("步骤3: 加载音频映射文件")
    print(f"{'='*60}")
    mappings, folder_ranges = load_all_mappings()
    
    # 4. 加载trial信息
    print(f"\n{'='*60}")
    print("步骤4: 加载trial信息")
    print(f"{'='*60}")
    all_trials = load_hanglei_trials(
        benchmark_files=BENCHMARK_FILES,
        mappings=mappings,
        folder_ranges=folder_ranges
    )
    
    if len(all_trials) == 0:
        print("\n错误: 未能加载任何trial信息")
        return
    
    # 5. 构建数据集
    print(f"\n{'='*60}")
    print("步骤5: 构建数据集")
    print(f"{'='*60}")
    dataset_df = build_dataset(eeg_data_downsampled, all_trials, meta)
    
    if len(dataset_df) == 0:
        print("\n错误: 数据集为空")
        return
    
    print(f"数据集大小: {len(dataset_df)}")
    
    # 6. 交叉验证训练
    print(f"\n{'='*60}")
    print("步骤6: 开始交叉验证训练")
    print(f"{'='*60}")
    
    mean_acc, std_acc, fold_accs = cross_validate_model(
        args, ds_init, dataset_df, device, subject_id="hanglei"
    )
    
    # 保存所有结果
    all_results = {
        'hanglei': {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_accs
        }
    }
    
    final_results_path = os.path.join(args.output_dir, 'all_subjects_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 计算总体统计
    summary = f"\n{'='*60}\n"
    summary += f"OVERALL RESULTS SUMMARY\n"
    summary += f"{'='*60}\n"
    summary += f"Subject: hanglei\n"
    summary += f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
    summary += f"Best fold accuracy: {max(fold_accs):.4f}\n"
    summary += f"Worst fold accuracy: {min(fold_accs):.4f}\n"
    summary += f"{'='*60}\n"
    
    print(summary)
    
    # 保存总结到文件
    with open(os.path.join(args.output_dir, "finetune_results.txt"), "a") as f:
        f.write(summary)
    
    print("Fine-tuning completed!")


if __name__ == "__main__":
    main()

