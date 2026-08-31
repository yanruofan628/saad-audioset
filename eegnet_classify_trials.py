import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# 数据配置
DATA_NAME = "\jiachen1017"
SAMPLING_RATE = 250
MFF_DIRECTORY = r"D:\D\research\数据采集\\jiachen1017_1_20251017_041306.mff"
TRIAL_TIMES_FILE = os.path.join(MFF_DIRECTORY, "log_jiachen1017_1_20251017_041306.txt")  # Jiachen的trial时间文件在MFF文件夹内
BENCHMARK_FILE = os.path.join(MFF_DIRECTORY, "benchmark_1_10-1-1.txt")  # 标签文件
PROCESSED_DATA_PATH = os.path.join("../standard_data_onlyeeg", DATA_NAME)
PROCESSED_DATA_PATH = 'A:/standard_data_noica/jiachen1017'

@dataclass
class DataMeta:
    sampling_rate_hz: float
    num_channels: int
    samples_per_trial: Optional[int] = None


def find_data_files(data_dir: Path) -> Tuple[Path, Path, Path]:
    trials = list(data_dir.glob("*trials.npy"))
    labels = list(data_dir.glob("*labels.csv"))
    info = list(data_dir.glob("*trial_info.json"))
    if len(trials) != 1 or len(labels) != 1 or len(info) != 1:
        raise FileNotFoundError(
            f"期望在 {data_dir} 找到各1个 trials.npy/labels.csv/trial_info.json，实际数量: "
            f"trials={len(trials)}, labels={len(labels)}, info={len(info)}"
        )
    return trials[0], labels[0], info[0]


def load_meta(info_path: Path, override_fs: Optional[float], override_channels: Optional[int]) -> DataMeta:
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    # 常见字段名兼容
    fs_candidates = [
        info.get("fs"),
        info.get("sampling_rate"),
        info.get("sampling_rate_hz"),
        info.get("sfreq"),
    ]
    ch_candidates = [
        info.get("n_channels"),
        info.get("num_channels"),
        info.get("channels"),
    ]
    spt_candidates = [
        info.get("samples_per_trial"),
        info.get("n_samples_per_trial"),
    ]

    fs = override_fs if override_fs is not None else next((x for x in fs_candidates if x is not None), None)
    num_channels = override_channels if override_channels is not None else next(
        (x for x in ch_candidates if x is not None), None)
    # 若从 info 中取到的是通道名称列表，转为长度
    if isinstance(num_channels, list):
        num_channels = len(num_channels)
    samples_per_trial = next((x for x in spt_candidates if x is not None), None)

    if fs is None or num_channels is None:
        raise ValueError("trial_info.json 未提供采样率或通道数，且未通过命令行覆盖。")

    return DataMeta(float(fs), int(num_channels), int(samples_per_trial) if samples_per_trial is not None else None)


def reshape_trials(trials_2d: np.ndarray, num_channels: int) -> np.ndarray:
    if trials_2d.ndim == 3:
        # (n_trials, n_channels, n_samples)
        return trials_2d
    if trials_2d.ndim != 2:
        raise ValueError(f"期望 2D 或 3D 数组，得到 {trials_2d.ndim}D")

    num_trials, flat_len = trials_2d.shape
    if flat_len % num_channels != 0:
        raise ValueError(
            f"无法按 {num_channels} 通道整除展平长度 {flat_len}，请检查 meta 或数据。"
        )
    samples_per_trial = flat_len // num_channels
    return trials_2d.reshape(num_trials, num_channels, samples_per_trial)


def exponential_moving_standardize(
    data: np.ndarray,
    alpha: float = 0.01,
    eps: float = 1e-8,
    init_block_size: Optional[int] = None
) -> np.ndarray:
    """
    对EEG数据应用指数移动标准化（Exponential Moving Standardize）
    
    该方法使用指数移动平均（EMA）和指数移动标准差（EMSD）对每个通道在时间维度上进行标准化。
    适用于非平稳信号，能够动态适应数据的变化。
    
    参数:
        data: 形状为 (n_samples, n_channels, n_timepoints) 或 (n_channels, n_timepoints) 的数组
        alpha: EMA的平滑参数，范围(0, 1)，默认0.01。较小的值更平滑，较大的值响应更快
        eps: 防止除零的小常数，默认1e-8
        init_block_size: 初始化块大小。如果提供，使用前init_block_size个时间点计算初始均值和标准差
                         如果不提供，使用第一个时间点初始化
    
    返回:
        标准化后的数据，形状与输入相同
    """
    original_shape = data.shape
    original_ndim = data.ndim
    
    # 统一处理为3D: (n_samples, n_channels, n_timepoints)
    if original_ndim == 2:
        data = data[np.newaxis, :, :]  # (1, n_channels, n_timepoints)
        was_2d = True
    elif original_ndim == 3:
        was_2d = False
    else:
        raise ValueError(f"数据必须是2D或3D，得到 {original_ndim}D")
    
    n_samples, n_channels, n_timepoints = data.shape
    standardized = np.zeros_like(data)
    
    # 对每个样本独立处理
    for sample_idx in range(n_samples):
        sample_data = data[sample_idx]  # (n_channels, n_timepoints)
        ema = np.zeros_like(sample_data)
        emsd = np.zeros_like(sample_data)
        
        # 初始化第一个时间点
        if init_block_size is not None and n_timepoints > init_block_size:
            # 使用前init_block_size个时间点计算初始均值和标准差
            init_mean = np.mean(sample_data[:, :init_block_size], axis=1, keepdims=True)
            init_std = np.std(sample_data[:, :init_block_size], axis=1, keepdims=True) + eps
            ema[:, 0] = init_mean[:, 0]
            emsd[:, 0] = init_std[:, 0] ** 2  # 存储方差
        else:
            # 使用第一个时间点初始化
            ema[:, 0] = sample_data[:, 0]
            emsd[:, 0] = eps ** 2  # 初始方差设为小值
        
        # 计算EMA和EMSD（指数移动标准差）
        for t in range(1, n_timepoints):
            # EMA: 指数移动平均
            ema[:, t] = alpha * sample_data[:, t] + (1 - alpha) * ema[:, t - 1]
            # EMSD: 指数移动标准差（先存储方差）
            squared_diff = (sample_data[:, t] - ema[:, t]) ** 2
            emsd[:, t] = alpha * squared_diff + (1 - alpha) * emsd[:, t - 1]
        
        # 将方差转换为标准差
        emsd = np.sqrt(emsd)
        
        # 标准化: (x - EMA) / (EMSD + eps)
        standardized[sample_idx] = (sample_data - ema) / (emsd + eps)
    
    # 恢复原始维度
    if was_2d:
        standardized = standardized[0]  # 移除第一个维度
    
    return standardized


def design_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError(
            f"带通频率不合法: low={low_hz}, high={high_hz}, Nyquist={fs / 2:.2f}"
        )
    return butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")


def select_last_seconds(data: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    last_n = int(round(seconds * fs))
    if last_n <= 0:
        raise ValueError("截取秒数需为正。")
    if data.shape[-1] < last_n:
        raise ValueError(
            f"每trial样本数 {data.shape[-1]} 小于所需最后 {last_n} 样本，请检查采样率或trial长度。"
        )
    return data[..., -last_n:]


def apply_baseline_correction(
    data: np.ndarray,
    fs: float,
    baseline_start: float = 0.0,
    baseline_end: float = 5.0
) -> np.ndarray:
    """
    对EEG数据进行基线矫正
    
    使用指定时间窗口的平均值作为基线，从整个trial中减去该基线。
    每个trial、每个通道独立进行基线矫正。
    
    参数:
        data: 形状为 (n_trials, n_channels, n_timepoints) 的数组
        fs: 采样率 (Hz)
        baseline_start: 基线窗口开始时间（秒），默认0.0
        baseline_end: 基线窗口结束时间（秒），默认5.0
    
    返回:
        基线矫正后的数据，形状与输入相同
    """
    n_trials, n_channels, n_timepoints = data.shape
    corrected_data = np.zeros_like(data)
    
    # 计算基线窗口的样本索引
    baseline_start_idx = int(round(baseline_start * fs))
    baseline_end_idx = int(round(baseline_end * fs))
    
    if baseline_start_idx < 0 or baseline_end_idx > n_timepoints:
        raise ValueError(
            f"基线窗口 [{baseline_start}, {baseline_end}] 超出数据范围 [0, {n_timepoints/fs:.2f}]"
        )
    if baseline_start_idx >= baseline_end_idx:
        raise ValueError(f"基线窗口开始时间必须小于结束时间")
    
    # 对每个trial、每个通道独立进行基线矫正
    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            # 提取基线窗口
            baseline_window = data[trial_idx, ch_idx, baseline_start_idx:baseline_end_idx]
            # 计算基线平均值
            baseline_mean = np.mean(baseline_window)
            # 从整个trial中减去基线平均值
            corrected_data[trial_idx, ch_idx, :] = data[trial_idx, ch_idx, :] - baseline_mean
    
    return corrected_data


def select_time_segment(
    data: np.ndarray,
    fs: float,
    mode: str = "full",
    segment_seconds: float = 2.0
) -> np.ndarray:
    """
    选择特定时间段的数据

    参数:
        data: (n_trials, n_channels, n_timepoints)
        fs: 采样率
        mode: 'full' | 'first2' | 'last2'
        segment_seconds: 选择的时长（秒），默认2s

    返回:
        裁剪后的数据
    """
    if mode == "full":
        return data

    seg_samples = int(round(segment_seconds * fs))
    if seg_samples <= 0 or seg_samples > data.shape[-1]:
        raise ValueError(f"segment_seconds 无效: {segment_seconds}")

    if mode == "first2":
        return data[:, :, :seg_samples]
    elif mode == "last2":
        return data[:, :, -seg_samples:]
    else:
        raise ValueError(f"未知的 mode: {mode}")


def split_trials_by_seconds(data: np.ndarray, labels: np.ndarray, fs: float, seconds_per_sample: float = 1.0):
    """
    将每个trial按指定秒数划分为多个样本

    Args:
        data: (n_trials, n_channels, n_samples) 原始数据
        labels: (n_trials,) 标签
        fs: 采样率
        seconds_per_sample: 每个样本的秒数

    Returns:
        X_split: (n_trials * n_samples_per_trial, n_channels, samples_per_sample)
        y_split: (n_trials * n_samples_per_trial,) 重复的标签
    """
    samples_per_sample = int(seconds_per_sample * fs)
    n_trials, n_channels, n_samples = data.shape

    # 计算每个trial能划分出多少个样本
    n_samples_per_trial = n_samples // samples_per_sample

    # 截取到能完整划分的长度
    truncated_samples = n_samples_per_trial * samples_per_sample
    data_truncated = data[..., :truncated_samples]

    # 重塑为 (n_trials, n_channels, n_samples_per_trial, samples_per_sample)
    X_split = data_truncated.reshape(n_trials, n_channels, n_samples_per_trial, samples_per_sample)
    X_split = X_split.reshape(n_trials * n_samples_per_trial, n_channels, samples_per_sample)

    # 重复标签
    y_split = np.repeat(labels, n_samples_per_trial)

    return X_split, y_split, n_samples_per_trial


def split_train_test_by_trials(X_split: np.ndarray, y_split: np.ndarray, n_samples_per_trial: int,
                               train_trials: int, total_trials: int, random_state: int = 42):
    """
    按trial数量划分训练/测试集（随机打乱trial后按比例划分）

    Args:
        X_split: 划分后的数据
        y_split: 划分后的标签
        n_samples_per_trial: 每个trial的样本数
        train_trials: 训练trial数量
        total_trials: 总trial数量
        random_state: 随机种子

    Returns:
        X_train, y_train, X_test, y_test
    """
    # 重塑数据为 (n_trials, n_samples_per_trial, n_channels, n_samples)
    X_reshaped = X_split.reshape(total_trials, n_samples_per_trial, X_split.shape[1], X_split.shape[2])
    y_reshaped = y_split.reshape(total_trials, n_samples_per_trial)
    
    # 生成随机打乱的trial索引
    np.random.seed(random_state)
    trial_indices = np.random.permutation(total_trials)
    
    # 按打乱的顺序重排trial
    X_shuffled = X_reshaped[trial_indices]
    y_shuffled = y_reshaped[trial_indices]
    
    # 重新展平为原始形状
    X_shuffled = X_shuffled.reshape(-1, X_split.shape[1], X_split.shape[2])
    y_shuffled = y_shuffled.reshape(-1)
    
    # 划分训练和测试集
    train_samples = train_trials * n_samples_per_trial
    
    X_train = X_shuffled[:train_samples]
    y_train = y_shuffled[:train_samples]
    X_test = X_shuffled[train_samples:]
    y_test = y_shuffled[train_samples:]

    return X_train, y_train, X_test, y_test


def split_train_test_by_samples_per_trial(X_split: np.ndarray, y_split: np.ndarray, n_samples_per_trial: int):
    """
    每个trial内划分：前N-1个样本训练，最后1个样本测试

    Args:
        X_split: 划分后的数据
        y_split: 划分后的标签
        n_samples_per_trial: 每个trial的样本数

    Returns:
        X_train, y_train, X_test, y_test
    """
    n_total_samples = X_split.shape[0]
    n_trials = n_total_samples // n_samples_per_trial

    train_indices = []
    test_indices = []

    for i in range(n_trials):
        start_idx = i * n_samples_per_trial
        # 前n_samples_per_trial-1个样本用于训练
        train_indices.extend(range(start_idx, start_idx + n_samples_per_trial - 1))
        # 最后1个样本用于测试
        test_indices.append(start_idx + n_samples_per_trial - 1)

    X_train = X_split[train_indices]
    y_train = y_split[train_indices]
    X_test = X_split[test_indices]
    y_test = y_split[test_indices]

    return X_train, y_train, X_test, y_test


def split_train_test_first1_per_trial(X_split: np.ndarray, y_split: np.ndarray, n_samples_per_trial: int):
    """
    每个trial内划分：第1个样本测试，其余样本训练
    """
    n_total_samples = X_split.shape[0]
    n_trials = n_total_samples // n_samples_per_trial
    
    train_indices = []
    test_indices = []
    
    for i in range(n_trials):
        start_idx = i * n_samples_per_trial
        # 第1个样本用于测试
        test_indices.append(start_idx)
        # 其余样本用于训练
        train_indices.extend(range(start_idx + 1, start_idx + n_samples_per_trial))
    
    X_train = X_split[train_indices]
    y_train = y_split[train_indices]
    X_test = X_split[test_indices]
    y_test = y_split[test_indices]
    
    return X_train, y_train, X_test, y_test


def split_train_test_mixed_4to1(X_split: np.ndarray, y_split: np.ndarray, random_state: int = 42):
    """
    将所有样本混合后按4:1比例随机划分训练/测试集
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_split, y_split, test_size=0.2, random_state=random_state, stratify=y_split
    )
    
    return X_train, y_train, X_test, y_test


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super(EEGNet, self).__init__()
        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, 20, (1,41), stride=(1, 1), bias=False, )

        # 独立成分分析
        self.conv_ica = nn.Conv2d(1, 8, (n_channels,1), stride=(1, 1),
                                  bias=False, )


        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5, )
        self.poolmean = nn.AvgPool2d(kernel_size=(1,210), stride=(1, 25))
        # 第二个卷积块：时间维卷积
        self.conv_class = nn.Conv2d(1,
                                    2,
                                    kernel_size=(20,8),
                                    stride=(1, 1),
                                    bias=True )
    def _get_conv_output_size(self, n_channels: int, n_samples: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.conv1(dummy)
            x = self.conv2(x)
            return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_ica(x)
        x = F.dropout(x, 0.15, training=self.training)  # 适度Dropout
        x=torch.permute(x,(0,2,1,3))
        x = self.conv_time(x)
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)  # 适度Dropout

        x = torch.mul(x, x)
        x = self.poolmean(x)
        x=torch.permute(x,(0,3,1,2))
        x=self.conv_class(x)
        x=x.squeeze()
        return x


def train_eegnet(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', class_weights: torch.Tensor = None, patience=30):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 适度正则化：weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=7e-4)
    # 学习率调度
    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)

    best_val_acc = 0
    best_val_balanced_acc = 0  # 最佳平衡准确率
    train_losses = []
    val_accuracies = []
    val_balanced_accuracies = []  # 平衡准确率历史
    patience_counter = 0  # 早停计数器
    best_model_state = None  # 保存最佳模型状态

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        # 计算训练集准确率（评估模式）
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
        train_acc = train_correct / max(1, train_total)
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(batch_y.cpu().numpy())

        val_acc = val_correct / val_total
        val_balanced_acc = balanced_accuracy_score(val_true_labels, val_predictions)
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_acc)
        val_balanced_accuracies.append(val_balanced_acc)

        # 早停检查：使用平衡准确率选择最佳模型（更适合类别不平衡的情况）
        if val_balanced_acc > best_val_balanced_acc:
            best_val_acc = val_acc  # 保存对应的普通准确率
            best_val_balanced_acc = val_balanced_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f} (最佳: {best_val_balanced_acc:.4f}, 耐心: {patience_counter}/{patience})')

        # 早停：如果验证平衡准确率连续patience个epoch没有改善，停止训练
        if patience_counter >= patience:
            print(f'\n早停触发！Epoch {epoch}, 最佳验证平衡准确率: {best_val_balanced_acc:.4f}')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)  # 恢复最佳模型
            break

        # 学习率调度步进
        scheduler.step()

    # 如果没有早停，也要恢复最佳模型
    if best_model_state is not None and patience_counter < patience:
        model.load_state_dict(best_model_state)
        print(f'\n训练完成，已恢复最佳模型（验证平衡准确率: {best_val_balanced_acc:.4f}）')

    return best_val_balanced_acc, train_losses, val_accuracies, val_balanced_accuracies


def main():
    # 直接设置所有参数，无需命令行
    data_dir_path = PROCESSED_DATA_PATH
    fs = SAMPLING_RATE
    channels = None  # 从trial_info.json读取
    keep_channels = 128
    last_seconds = 5.0
    train_ratio = 0.75  # 训练集占比（仅用于 by_trials 方法）
    random_state = 42  # 随机种子（用于 by_trials 和 mixed_4to1 方法）
    epochs = 300
    batch_size = 16
    lr = 0.0008  # 适中的学习率

    # 选择划分方式: 'by_trials' | 'by_samples_per_trial' | 'by_samples_per_trial_first_test' | 'mixed_4to1'
    split_method = 'by_trials'
    # 选择分类使用的时间段: 'full' 使用5-10s全部，'first2' 使用刺激期前2s，'last2' 使用刺激期后2s
    segment_mode = 'first2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"数据目录: {data_dir_path}")
    print(f"采样率: {fs} Hz")

    data_dir = Path(data_dir_path)
    if not data_dir.exists():
        print(f"目录不存在: {data_dir}", file=sys.stderr)
        print(f"请确保已处理的数据保存在: {PROCESSED_DATA_PATH}")
        sys.exit(1)

    trials_path, labels_path, info_path = find_data_files(data_dir)

    # 如果trial_info.json不存在或缺少信息，使用默认值
    try:
        meta = load_meta(info_path, fs, channels)
        print(
            f"从trial_info.json读取元信息: fs={meta.sampling_rate_hz}, 通道数={meta.num_channels}, 每trial样本={meta.samples_per_trial}")
    except (FileNotFoundError, ValueError) as e:
        print(f"警告: {e}")
        print("使用默认元信息...")
        # 从trials数据推断通道数
        trials = np.load(trials_path)
        if trials.ndim == 2:
            # 假设是 (n_trials, n_channels * n_samples) 格式
            # 需要推断通道数，这里假设是128通道
            inferred_channels = 128
        else:
            inferred_channels = trials.shape[1] if trials.ndim == 3 else 128

        meta = DataMeta(sampling_rate_hz=fs, num_channels=inferred_channels, samples_per_trial=None)
        print(f"使用默认元信息: fs={meta.sampling_rate_hz}, 通道数={meta.num_channels}")

    trials = np.load(trials_path)
    # 兼容读取 labels.csv（优先使用带表头的 'Label' 列）
    labels_df = pd.read_csv(labels_path)
    if 'Label' in labels_df.columns:
        labels = labels_df['Label'].to_numpy()
    else:
        # 无表头时，若有两列则取第二列（trial, label），否则取第一列
        if labels_df.shape[1] >= 2:
            labels = labels_df.iloc[:, 1].to_numpy()
        else:
            labels = labels_df.iloc[:, 0].to_numpy()

    # 形状处理
    trials_3d = reshape_trials(trials, meta.num_channels)
    print(f"原始 trials 形状: {trials_3d.shape}")
    print("注意: 数据在extract_multiple_subjects_eeg.py中已经进行了滤波和重参考（CAR），此处不再重复处理")

    # 截取最后10秒（0-10秒：前5秒基线期，后5秒刺激期）
    trial_duration = 10.0  # 需要10秒的数据（0-10秒）
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    print(f"截取0-10秒后形状: {data_full.shape}")
    
    # 基线矫正：使用前5秒（0-5秒）作为基线
    print("\n进行基线矫正...")
    print(f"基线窗口: 0-5秒")
    data_baseline_corrected = apply_baseline_correction(
        data_full,
        fs=meta.sampling_rate_hz,
        baseline_start=3.0,
        baseline_end=5.0
    )
    print("基线矫正完成")
    
    # 提取5-10秒用于分类（刺激期）
    stimulus_start_seconds = 5.0
    stimulus_duration = 5.0
    stimulus_start_idx = int(round(stimulus_start_seconds * meta.sampling_rate_hz))
    stimulus_end_idx = int(round((stimulus_start_seconds + stimulus_duration) * meta.sampling_rate_hz))
    data = data_baseline_corrected[:, :, stimulus_start_idx:stimulus_end_idx]
    print(f"提取5-10秒（刺激期）后形状: {data.shape}")

    # 根据 segment_mode 选择2秒片段
    data = select_time_segment(data, meta.sampling_rate_hz, mode=segment_mode, segment_seconds=2.0)
    print(f"时间段选择模式: {segment_mode} -> 形状: {data.shape}")

    # 确保数据和标签数量匹配
    if data.shape[0] != labels.shape[0]:
        # 取两者中较小的数量
        min_n = min(data.shape[0], labels.shape[0])
        print(f"警告: trial数量({data.shape[0]})与标签数量({labels.shape[0]})不匹配，将使用前{min_n}个")
        data = data[:min_n]
        labels = labels[:min_n]
    
    total_n = data.shape[0]  # 自动获取实际的trial数量
    train_n = int(total_n * train_ratio)  # 根据比例计算训练集大小
    print(f"实际使用 {total_n} 个trial，训练集 {train_n} 个trial")
    
    # 检查是否有足够的数据
    if total_n == 0:
        raise ValueError("没有可用的trial数据")
    if labels.shape[0] == 0:
        raise ValueError("没有可用的标签数据")

    # 使用函数将每个trial的片段按1秒划分
    X_split, y_split, n_samples_per_trial = split_trials_by_seconds(
        data, labels, meta.sampling_rate_hz, seconds_per_sample=1.0
    )

    print(f"按1秒划分后形状: {X_split.shape}")
    print(f"总样本数: {X_split.shape[0]} ({total_n} trials × {n_samples_per_trial} samples)")

    # 标签编码为0开始的整数
    unique_labels = np.unique(y_split)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in y_split])
    n_classes = len(unique_labels)
    print(f"类别数: {n_classes}, 类别: {unique_labels}")

    # 根据选择的划分方式进行训练/测试划分
    if split_method == 'by_trials':
        # 方式1：按trial数量划分（随机打乱后按比例划分）
        X_train, y_train, X_test, y_test = split_train_test_by_trials(
            X_split, y_encoded, n_samples_per_trial, train_n, total_n, random_state=random_state
        )
        print(f"划分方式: 按trial数量(随机打乱) - {train_n}个trial训练，{total_n - train_n}个trial测试")

    elif split_method == 'by_samples_per_trial':
        # 方式2：每个trial内划分（前4个样本训练，最后1个样本测试）
        X_train, y_train, X_test, y_test = split_train_test_by_samples_per_trial(
            X_split, y_encoded, n_samples_per_trial
        )
        print(f"划分方式: 每个trial内划分 - 前{n_samples_per_trial - 1}个样本训练，最后1个样本测试")
    elif split_method == 'by_samples_per_trial_first_test':
        # 方式3：每个trial内划分（第1个样本测试，其余样本训练）
        X_train, y_train, X_test, y_test = split_train_test_first1_per_trial(
            X_split, y_encoded, n_samples_per_trial
        )
        print(f"划分方式: 每个trial内划分 - 第1个样本测试，其余{n_samples_per_trial - 1}个样本训练")
    elif split_method == 'mixed_4to1':
        # 方式4：所有样本混合后按4:1比例随机划分
        X_train, y_train, X_test, y_test = split_train_test_mixed_4to1(
            X_split, y_encoded, random_state=random_state
        )
        print(f"划分方式: 混合随机划分 - 训练集80%，测试集20%")
        
    else:
        raise ValueError(f"未知的划分方式: {split_method}")

    print(f"训练集: {X_train.shape[0]} samples, 测试集: {X_test.shape[0]} samples")

    # 数据标准化：选择标准化方法
    # 'global': 全局StandardScaler标准化（默认）
    # 'exponential_moving': 指数移动标准化
    standardization_method = 'exponential_moving'  # 或 'global'
    
    n_train, n_channels, n_timepoints = X_train.shape
    n_test = X_test.shape[0]
    
    if standardization_method == 'global':
        # 全局StandardScaler标准化
        print("\n对数据进行全局标准化（StandardScaler）...")
        X_train_flat = X_train.reshape(n_train, -1)  # (n_train, n_channels * n_timepoints)
        X_test_flat = X_test.reshape(n_test, -1)  # (n_test, n_channels * n_timepoints)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # 恢复原始形状
        X_train_scaled = X_train_scaled.reshape(n_train, n_channels, n_timepoints)
        X_test_scaled = X_test_scaled.reshape(n_test, n_channels, n_timepoints)
        
        print(f"标准化完成 - 训练集均值: {X_train_scaled.mean():.6f}, 标准差: {X_train_scaled.std():.6f}")
        print(f"标准化完成 - 测试集均值: {X_test_scaled.mean():.6f}, 标准差: {X_test_scaled.std():.6f}")
        
    elif standardization_method == 'exponential_moving':
        # 指数移动标准化
        print("\n对数据进行指数移动标准化（Exponential Moving Standardize）...")
        print(f"参数: alpha=0.01, eps=1e-8")
        
        # 对训练集和测试集分别应用指数移动标准化
        # 每个样本独立处理，适用于分类任务
        X_train_scaled = exponential_moving_standardize(
            X_train, 
            alpha=0.01,  # 可以调整，较小的值更平滑
            eps=1e-8,
            init_block_size=None  # 使用第一个时间点初始化，可以改为整数使用前N个时间点
        )
        X_test_scaled = exponential_moving_standardize(
            X_test,
            alpha=0.01,
            eps=1e-8,
            init_block_size=None
        )
        
        print(f"标准化完成 - 训练集均值: {X_train_scaled.mean():.6f}, 标准差: {X_train_scaled.std():.6f}")
        print(f"标准化完成 - 测试集均值: {X_test_scaled.mean():.6f}, 标准差: {X_test_scaled.std():.6f}")
        
    else:
        raise ValueError(f"未知的标准化方法: {standardization_method}，请选择 'global' 或 'exponential_moving'")

    # 转换为PyTorch张量并添加通道维度
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # (n_samples, 1, n_channels, n_timepoints)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建EEGNet模型
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("开始训练...")
    # 计算类别权重（根据训练标签），传入损失函数以缓解类别不平衡
    # 注意：权重需在与模型相同的设备上
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    num_classes = len(unique_train)
    total_train = y_train.shape[0]
    weights_np = np.array([total_train / (num_classes * counts_train[np.where(unique_train == c)[0][0]]) for c in unique_train], dtype=np.float32)
    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

    best_val_balanced_acc, train_losses, val_accuracies, val_balanced_accuracies = train_eegnet(
        model, train_loader, test_loader,
        epochs=epochs, lr=lr, device=device, class_weights=class_weights, patience=30
    )

    # 最终测试
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())

    test_acc = test_correct / test_total
    # 计算平衡准确率（每个类别的召回率的平均值，更适合类别不平衡的情况）
    balanced_acc = balanced_accuracy_score(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, digits=4)
    cm = confusion_matrix(all_true_labels, all_predictions)

    print("\n=== EEGNet 最终结果 ===")
    print(f"测试准确率 (Accuracy): {test_acc:.4f}")
    print(f"平衡测试准确率 (Balanced Accuracy): {balanced_acc:.4f}")
    print(f"最佳验证平衡准确率: {best_val_balanced_acc:.4f}")
    print("\nClassification Report:\n" + report)
    print("Confusion Matrix:\n" + np.array2string(cm))

    # 保存结果
    out_path = data_dir / "eegnet_results.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"测试准确率 (Accuracy): {test_acc:.6f}\n")
        f.write(f"平衡测试准确率 (Balanced Accuracy): {balanced_acc:.6f}\n")
        f.write(f"最佳验证平衡准确率: {best_val_balanced_acc:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"结果已保存至: {out_path}")

    # 保存模型
    model_path = data_dir / "eegnet_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")


if __name__ == "__main__":
    main()


