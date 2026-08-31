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
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 注意：避免循环导入，相关函数将在需要时延迟导入

# 数据配置
DATA_NAME = "hanglei"
SAMPLING_RATE = 250
PROCESSED_DATA_PATH = 'A:/standard_data_noica/hanglei'

# Benchmark文件路径（用于筛选一致trials）
BENCHMARK_FILES = [
    r"A:\hanglei_1_20251224_060149.mff\benchmark_1_10-1-1.txt",
    r"A:\hanglei_2_20251224_063021.mff\benchmark_1_10-1-2.txt",
    r"A:\hanglei_3_20251224_065517.mff\benchmark_1_10-1-3.txt",
]

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
    baseline_end: float = 4.0
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
    segment_seconds: float = None
) -> np.ndarray:
    """
    选择特定时间段的数据

    参数:
        data: (n_trials, n_channels, n_timepoints)
        fs: 采样率
        mode: 'full' | 'first1' | 'first2' | 'last1' | 'last2'
        segment_seconds: 选择的时长（秒），如果为None则根据mode自动设置

    返回:
        裁剪后的数据
    """
    if mode == "full":
        return data

    # 根据mode自动设置segment_seconds
    if segment_seconds is None:
        if mode in ["first1", "last1"]:
            segment_seconds = 1.0
        elif mode in ["first2", "last2"]:
            segment_seconds = 2.0
        else:
            segment_seconds = 2.0  # 默认值

    seg_samples = int(round(segment_seconds * fs))
    if seg_samples <= 0 or seg_samples > data.shape[-1]:
        raise ValueError(f"segment_seconds 无效: {segment_seconds}")

    if mode == "first1":
        return data[:, :, :seg_samples]
    elif mode == "first2":
        return data[:, :, :seg_samples]
    elif mode == "last1":
        return data[:, :, -seg_samples:]
    elif mode == "last2":
        return data[:, :, -seg_samples:]
    else:
        raise ValueError(f"未知的 mode: {mode}")


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super(EEGNet, self).__init__()
        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, 20, (1,41), stride=(1, 1), bias=False, )

        # 独立成分分析
        self.conv_ica = nn.Conv2d(1, 8, (n_channels,1), stride=(1, 1),
                                  bias=False, )

        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5, )
        # 使用自适应池化，确保输出时间维度为1（适用于任意输入长度）
        # 原来的poolmean是为1秒数据设计的，现在改为自适应池化
        self.poolmean = nn.AdaptiveAvgPool2d((8, 1))  # 输出: (8, 1) 对应 (通道, 时间)
        # 第二个卷积块：时间维卷积
        self.conv_class = nn.Conv2d(1,
                                    n_classes,  # 直接输出类别数
                                    kernel_size=(20,8),
                                    stride=(1, 1),
                                    bias=True )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_ica(x)
        x = F.dropout(x, 0.15, training=self.training)  # 适度Dropout
        x=torch.permute(x,(0,2,1,3))
        x = self.conv_time(x)
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)  # 适度Dropout

        x = torch.mul(x, x)
        x = self.poolmean(x)  # 输出: (batch, 20, 8, 1)
        x=torch.permute(x,(0,3,1,2))  # 输出: (batch, 1, 20, 8)
        x=self.conv_class(x)  # 输出: (batch, n_classes, 1, 1)
        x=x.squeeze()  # 输出: (batch, n_classes)
        return x


def train_eegnet(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', class_weights: torch.Tensor = None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 适度正则化：weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=7e-4)
    # 学习率调度
    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)

    best_val_balanced_acc = 0  # 最佳验证平衡准确率
    train_losses = []
    val_accuracies = []
    val_balanced_accuracies = []  # 平衡准确率历史
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

        # 保存验证集上表现最好的模型
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f} (最佳: {best_val_balanced_acc:.4f})')

        # 学习率调度步进
        scheduler.step()

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\n训练完成，已恢复最佳模型（验证平衡准确率: {best_val_balanced_acc:.4f}）')

    return best_val_balanced_acc, train_losses, val_accuracies, val_balanced_accuracies


def filter_trials_by_choice_consistency(trials, labels, enable_filtering=True):
    """
    根据选择一致性筛选trials

    筛选逻辑：
    - AB版本和BA版本都选择了同一个音频的trials被保留
    - AB版本：左边放A，右边放B
    - BA版本：左边放B，右边放A
    - 情况1：AB选左（选A）且BA选右（选A）→ 一致，都选了A
    - 情况2：AB选右（选B）且BA选左（选B）→ 一致，都选了B
    - 其他情况：不一致，丢弃

    Parameters:
    -----------
    trials : np.ndarray
        EEG数据，形状为 (n_trials, n_channels, n_timepoints)
    labels : np.ndarray
        标签数据，形状为 (n_trials,)
    enable_filtering : bool
        是否启用选择一致性筛选

    Returns:
    --------
    filtered_trials : np.ndarray
        筛选后的EEG数据
    filtered_labels : np.ndarray
        筛选后的标签数据
    """
    if not enable_filtering:
        print("选择一致性筛选已关闭，使用全部数据")
        return trials, labels

    print("启用选择一致性筛选：只保留AB和BA版本都选择同一个音频的trials")

    # 如果无法进行复杂的筛选（缺少必要的数据），回退到简单筛选
    try:
        # 尝试导入复杂的筛选函数
        from eeg_audio_fusion_classify_hanglei import (
            load_all_mappings, load_hanglei_trials,
            filter_consistent_trials, BENCHMARK_FILES
        )

        print("使用复杂的选择一致性筛选...")

        # 这里需要数据目录路径来加载映射和trial信息
        # 由于这个函数只接收trials和labels，我们使用简单的方法
        # 只过滤掉NaN标签，但保留原有的复杂筛选逻辑在main函数中

    except ImportError:
        print("无法导入复杂筛选函数，使用简单NaN过滤")

    # 基础筛选：过滤掉NaN标签
    valid_mask = ~pd.isna(labels)

    if not valid_mask.any():
        print("警告：没有找到有效的标签，使用全部数据")
        return trials, labels

    filtered_trials = trials[valid_mask]
    filtered_labels = labels[valid_mask]

    print(f"基础筛选 - 筛选前：{len(trials)} 个trials")
    print(f"基础筛选 - 筛选后：{len(filtered_trials)} 个trials")
    print(f"基础筛选 - 过滤掉：{len(trials) - len(filtered_trials)} 个无效trials")

    return filtered_trials, filtered_labels


def main():
    # 直接设置所有参数，无需命令行
    data_dir_path = PROCESSED_DATA_PATH
    fs = SAMPLING_RATE
    channels = None  # 从trial_info.json读取
    keep_channels = 128
    last_seconds = 5.0
    train_ratio = 0.75  # 训练集占比
    random_state = 42  # 随机种子
    epochs = 300
    batch_size = 16
    lr = 0.0008  # 适中的学习率

    # ===== 新增：筛选开关 =====
    enable_filtering = False  # True: 筛选有效trials, False: 使用全部数据
    # ========================

    # 选择划分方式: 'by_trials' | 'random'
    split_method = 'by_trials'  # 'by_trials': 按trial划分, 'random': 随机划分
    # 选择分类使用的时间段: 'full' 使用全部数据，'first1' 使用前1s，'first2' 使用前2s，'last1' 使用后1s，'last2' 使用后2s
    segment_mode = 'last2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"数据目录: {data_dir_path}")
    print(f"采样率: {fs} Hz")
    print(f"每个trial作为完整样本（2秒，3-5秒刺激期）进行分类")

    data_dir = Path(data_dir_path)
    if not data_dir.exists():
        print(f"目录不存在: {data_dir}", file=sys.stderr)
        print(f"请确保已处理的数据保存在: {PROCESSED_DATA_PATH}")
        sys.exit(1)

    trials_path, labels_path, info_path = find_data_files(data_dir)

    # 如果trial_info.json不存在或缺少信息，使用默认值
    trial_duration = None
    try:
        meta = load_meta(info_path, fs, channels)
        print(
            f"从trial_info.json读取元信息: fs={meta.sampling_rate_hz}, 通道数={meta.num_channels}, 每trial样本={meta.samples_per_trial}")

        # 从trial_info.json读取实际的trial_duration
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        trial_duration = info.get("trial_duration")
        if trial_duration:
            print(f"从trial_info.json读取trial时长: {trial_duration}秒")
    except (FileNotFoundError, ValueError) as e:
        print(f"警告: {e}")
        print("使用默认元信息...")
        # 从trials数据推断通道数
        trials = np.load(trials_path)
        if trials.ndim == 2:
            inferred_channels = 128
        else:
            inferred_channels = trials.shape[1] if trials.ndim == 3 else 128

        meta = DataMeta(sampling_rate_hz=fs, num_channels=inferred_channels, samples_per_trial=None)
        print(f"使用默认元信息: fs={meta.sampling_rate_hz}, 通道数={meta.num_channels}")

    # 如果没从json读取到trial_duration，则从数据推断
    if trial_duration is None:
        trials = np.load(trials_path)
        trials_3d = reshape_trials(trials, meta.num_channels)
        inferred_trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz
        trial_duration = inferred_trial_duration
        print(f"从数据推断trial时长: {trial_duration:.2f}秒")

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

    # 筛选选择一致的trials（只保留AB和BA版本都选择同一个音频的trials）
    print("\n=== 筛选选择一致的trials ===")

    consistent_trials = None
    all_trials = None

    if enable_filtering:
        # 延迟导入以避免循环导入
        try:
            from eeg_audio_fusion_classify_hanglei import (
                load_all_mappings, load_hanglei_trials,
                filter_consistent_trials
            )

            # 1. 加载音频映射文件
            mappings, folder_ranges = load_all_mappings()

            # 2. 从映射文件中提取所有音频对名称（480个）
            all_original_names = set()
            for mapping in mappings:
                all_original_names.update(mapping.values())
            pair_names = sorted(list(all_original_names))
            print(f"从映射文件提取的音频对总数: {len(pair_names)}")

            # 3. 加载hanglei的trial信息（三个session）
            all_trials = load_hanglei_trials(
                benchmark_files=BENCHMARK_FILES,
                mappings=mappings,
                folder_ranges=folder_ranges
            )

            if len(all_trials) == 0:
                print("警告: 未能加载任何trial信息，将使用原始标签")
                consistent_trials = None
            else:
                # 4. 筛选一致的trials（只保留AB和BA版本都选择同一个音频的trials）
                print("进行选择一致性筛选...")
                print("筛选逻辑：")
                print("- AB版本：左边放A，右边放B")
                print("- BA版本：左边放B，右边放A")
                print("- 情况1：AB选左（选A）且BA选右（选A）→ 一致，都选了A")
                print("- 情况2：AB选右（选B）且BA选左（选B）→ 一致，都选了B")
                print("- 其他情况：不一致，丢弃")

                consistent_trials = filter_consistent_trials(all_trials, pair_names)

                if len(consistent_trials) == 0:
                    print("警告: 没有一致的trials，将使用原始标签")
                    consistent_trials = None
                else:
                    print(f"筛选结果：保留 {len(consistent_trials)} 个一致的trials")
        except ImportError as e:
            print(f"警告: 无法导入筛选函数 ({e})，将使用原始标签")
            consistent_trials = None
            all_trials = None
    else:
        print("选择一致性筛选已关闭，将使用全部trials")
    
    # 6. 对齐EEG数据和一致的trials（使用原始choice作为标签，不做平衡）
    if consistent_trials is not None and all_trials is not None:
        print("\n=== 对齐EEG数据和一致的trials ===")
        
        # 创建trial名称到索引的映射（在all_trials中）
        trial_name_to_idx = {}
        for idx, trial in enumerate(all_trials):
            pair_name = trial['original_name']
            if pair_name not in trial_name_to_idx:
                trial_name_to_idx[pair_name] = []
            trial_name_to_idx[pair_name].append(idx)
        
        # 提取一致的trials对应的EEG数据和标签（使用原始choice，不做平衡）
        consistent_indices = []
        consistent_labels_list = []
        matched_count = 0
        
        for trial in consistent_trials:
            pair_name = trial['pair_name']
            
            # 在all_trials中查找匹配的trial
            matching_trial_idx = None
            for idx, all_trial in enumerate(all_trials):
                if all_trial['original_name'] == pair_name:
                    matching_trial_idx = idx
                    break
            
            if matching_trial_idx is not None:
                consistent_indices.append(matching_trial_idx)
                # choice: 1=选左, 2=选右 -> 转换为0=选左, 1=选右
                consistent_labels_list.append(trial['choice'] - 1)
                matched_count += 1
        
        if matched_count > 0:
            print(f"成功匹配 {matched_count}/{len(consistent_trials)} 个trials")
            
            # 保存一致的索引和标签，稍后在数据预处理后使用
            filtered_indices = consistent_indices
            filtered_labels_array = np.array(consistent_labels_list)
            
            print(f"筛选后的标签分布（原始choice，未平衡）:")
            unique_labels, counts = np.unique(filtered_labels_array, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"  标签 {label}: {count} 个样本 ({count/len(filtered_labels_array)*100:.1f}%)")
        else:
            print("警告: 未能匹配任何trial，将使用原始数据")
            filtered_indices = None
            filtered_labels_array = None
    else:
        print("使用原始EEG数据和标签（未进行筛选）")
        filtered_indices = None
        filtered_labels_array = None
    
    # 形状处理
    trials_3d = reshape_trials(trials, meta.num_channels)
    print(f"原始 trials 形状: {trials_3d.shape}")
    print("注意: 数据在extract_multiple_subjects_eeg.py中已经进行了滤波和重参考（CAR），此处不再重复处理")

    # 截取完整trial数据
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    print(f"截取完整trial数据（{trial_duration}秒）后形状: {data_full.shape}")
    
    # 基线矫正：使用前3秒（0-3秒）作为基线
    print("\n进行基线矫正...")
    print(f"基线窗口: 0-3秒")
    data_baseline_corrected = apply_baseline_correction(
        data_full,
        fs=meta.sampling_rate_hz,
        baseline_start=0.0,
        baseline_end=3.0
    )
    print("基线矫正完成")

    # 提取3-5秒用于分类（刺激期，作为完整的2秒样本）
    # 检查数据长度是否足够（hanglei是6秒数据）
    if trial_duration < 5.0:
        raise ValueError(f"数据时长{trial_duration}秒不足以提取3-5秒的刺激期，请检查数据")

    stimulus_start_seconds = 3.0
    stimulus_duration = 2.0
    stimulus_start_idx = int(round(stimulus_start_seconds * meta.sampling_rate_hz))
    stimulus_end_idx = int(round((stimulus_start_seconds + stimulus_duration) * meta.sampling_rate_hz))
    data = data_full[:, :, stimulus_start_idx:stimulus_end_idx]
    print(f"提取3-5秒（刺激期，完整的2秒）后形状: {data.shape}  # (n_trials, n_channels, n_timepoints)")
    print(f"每个trial时长: {data.shape[2] / meta.sampling_rate_hz:.2f} 秒")

    # 根据 segment_mode 选择时间片段
    data = select_time_segment(data, meta.sampling_rate_hz, mode=segment_mode)
    print(f"时间段选择模式: {segment_mode} -> 形状: {data.shape}")

    # 如果使用了一致的trials，现在应用索引和标签（不做平衡）
    if filtered_indices is not None and filtered_labels_array is not None:
        print("\n=== 应用一致的trials索引 ===")
        if max(filtered_indices) < len(data):
            data = data[filtered_indices]
            labels = filtered_labels_array
            print(f"应用筛选索引后的数据形状: {data.shape}")
            print(f"应用筛选索引后的标签形状: {labels.shape}")
        else:
            print(f"警告: 筛选索引超出数据范围 (max={max(filtered_indices)}, data_len={len(data)})，使用原始数据")

    # 确保数据和标签数量匹配
    if data.shape[0] != labels.shape[0]:
        # 取两者中较小的数量
        min_n = min(data.shape[0], labels.shape[0])
        print(f"警告: trial数量({data.shape[0]})与标签数量({labels.shape[0]})不匹配，将使用前{min_n}个")
        data = data[:min_n]
        labels = labels[:min_n]
    
    total_n = data.shape[0]  # 总trial数量
    print(f"总trial数: {total_n}")
    
    # 检查是否有足够的数据
    if total_n == 0:
        raise ValueError("没有可用的trial数据")
    if labels.shape[0] == 0:
        raise ValueError("没有可用的标签数据")

    # 应用基础筛选（过滤NaN标签）
    print("\n=== 应用基础筛选 ===")
    data, labels = filter_trials_by_choice_consistency(data, labels, enable_filtering)
    total_n = data.shape[0]
    print(f"筛选后总trial数: {total_n}")

    # 标签编码为0开始的整数
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in labels])
    n_classes = len(unique_labels)
    print(f"类别数: {n_classes}, 类别: {unique_labels}")

    # 统计每个标签的样本数量
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    print("各标签样本数量:")
    for label in sorted(label_counts.keys()):
        print(f"  标签 {label}: {label_counts[label]} 个样本")

    # 数据划分：每个trial作为一个完整样本
    if split_method == 'by_trials':
        # 按trial数量划分（随机打乱后按比例划分）
        train_n = int(total_n * train_ratio)
        np.random.seed(random_state)
        indices = np.random.permutation(total_n)
        train_indices = indices[:train_n]
        test_indices = indices[train_n:]
        
        X_train = data[train_indices]
        y_train = y_encoded[train_indices]
        X_test = data[test_indices]
        y_test = y_encoded[test_indices]
        
        print(f"划分方式: 按trial数量(随机打乱) - {train_n}个trial训练，{total_n - train_n}个trial测试")
        
    elif split_method == 'random':
        # 随机划分（不保持trial完整性，但这里每个样本就是一个trial，所以效果相同）
        X_train, X_test, y_train, y_test = train_test_split(
            data, y_encoded, test_size=1-train_ratio, random_state=random_state, stratify=y_encoded
        )
        print(f"划分方式: 随机划分 - 训练集{len(X_train)}个trial，测试集{len(X_test)}个trial")
    else:
        raise ValueError(f"未知的划分方式: {split_method}，请选择 'by_trials' 或 'random'")

    print(f"训练集: {X_train.shape[0]} samples, 测试集: {X_test.shape[0]} samples")
    print(f"数据形状: (n_trials, n_channels={X_train.shape[1]}, n_timepoints={X_train.shape[2]})")

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

    # Channel Normalization (Z-score per channel)
    print("\n进行通道级Z-score标准化...")

    # 对训练集进行通道标准化
    X_train_channel_norm = np.zeros_like(X_train_scaled)
    train_channel_means = np.zeros(n_channels)
    train_channel_stds = np.zeros(n_channels)

    for ch in range(n_channels):
        channel_data = X_train_scaled[:, ch, :]  # (n_trials, n_timepoints)
        ch_mean = channel_data.mean()
        ch_std = channel_data.std()
        train_channel_means[ch] = ch_mean
        train_channel_stds[ch] = ch_std
        X_train_channel_norm[:, ch, :] = (channel_data - ch_mean) / (ch_std + 1e-8)

    # 对测试集使用训练集的统计量进行标准化
    X_test_channel_norm = np.zeros_like(X_test_scaled)
    for ch in range(n_channels):
        channel_data = X_test_scaled[:, ch, :]  # (n_trials, n_timepoints)
        X_test_channel_norm[:, ch, :] = (channel_data - train_channel_means[ch]) / (train_channel_stds[ch] + 1e-8)

    print(f"通道标准化完成")
    print(f"训练集 - 各通道均值范围: [{train_channel_means.min():.6f}, {train_channel_means.max():.6f}]")
    print(f"训练集 - 各通道标准差范围: [{train_channel_stds.min():.6f}, {train_channel_stds.max():.6f}]")
    print(f"标准化后训练集均值: {X_train_channel_norm.mean():.6f}, 标准差: {X_train_channel_norm.std():.6f}")
    print(f"标准化后测试集均值: {X_test_channel_norm.mean():.6f}, 标准差: {X_test_channel_norm.std():.6f}")

    # 直接使用指数移动标准化后的数据
    # X_train_channel_norm = X_train_scaled
    # X_test_channel_norm = X_test_scaled

    print("跳过通道级Z-score标准化，直接使用指数移动标准化后的数据")

    # 转换为PyTorch张量并添加通道维度
    X_train_tensor = torch.FloatTensor(X_train_channel_norm).unsqueeze(1)  # (n_samples, 1, n_channels, n_timepoints)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_channel_norm).unsqueeze(1)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建EEGNet模型
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]  # 现在是2秒的数据，约500个时间点
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入尺寸: (batch, 1, {n_channels} channels, {n_samples} timepoints)")

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
        epochs=epochs, lr=lr, device=device, class_weights=class_weights
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
    out_path = data_dir / "eegnet_results_3-5s.txt"
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
    model_path = data_dir / "eegnet_model_3-5s.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")


if __name__ == "__main__":
    main()

