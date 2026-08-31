import os
import numpy as np
import librosa
import soundfile as sf
import whisper
from whisper.audio import log_mel_spectrogram as whisper_log_mel_spectrogram
from whisper.audio import pad_or_trim as whisper_pad_or_trim
from whisper.audio import N_FRAMES as WHISPER_N_FRAMES
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random


def load_audio_with_correct_sr(audio_path, target_sr=16000, max_duration_seconds=60):
    """
    加载音频文件并确保采样率为16kHz（Whisper要求的采样率）
    只加载前max_duration_seconds秒的音频
    """
    print(f"正在加载音频文件: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=target_sr, duration=max_duration_seconds)
    print(f"音频长度: {len(audio) / sr:.2f}秒, 采样率: {sr}Hz")
    return audio, sr


def segment_audio_into_frames(audio, sr, frame_duration_ms=20):
    """
    将音频分割成指定时长的帧
    """
    frame_length_samples = int(sr * frame_duration_ms / 1000)  # 20ms对应的样本数
    frames = []

    # 确保audio是numpy数组
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    for i in range(0, len(audio), frame_length_samples):
        frame = audio[i:i + frame_length_samples]
        if len(frame) == frame_length_samples:  # 只保留完整长度的帧
            # 确保frame是numpy数组
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()
            # 确保frame是连续的numpy数组
            frame = np.ascontiguousarray(frame)
            frames.append(frame)

    print(f"音频被分割成 {len(frames)} 帧，每帧 {frame_duration_ms}ms")
    return frames


def calculate_whisper_surprise_per_frame(frames, model_name="base", min_context_frames=5, max_context_seconds=25):
    """
    使用Whisper计算每帧的惊喜度（预测误差 -log(P_t(Y_t))、熵和表征变化）
    使用历史信息进行预测，但限制历史长度以避免Whisper的30秒限制

    Args:
        frames: 音频帧列表
        model_name: Whisper模型名称
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过Whisper的30秒限制

    Returns:
        tuple: (prediction_error_scores, entropy_scores, representation_change_scores)
    """
    print(f"正在加载Whisper模型: {model_name}")
    model = whisper.load_model(model_name)

    # 获取tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

    prediction_error_scores = []
    entropy_scores = []
    representation_change_scores = []

    print(f"正在计算每帧的惊喜度... (使用最多{max_context_seconds}秒历史信息进行预测)")

    # 维护历史帧，但限制最大长度
    historical_frames = []
    frame_duration_seconds = 0.02  # 20ms = 0.02秒
    max_context_frames = int(max_context_seconds / frame_duration_seconds)

    for i, frame_data in enumerate(tqdm(frames, desc="处理帧")):
        try:
            # 将当前帧加入历史
            current_frame = np.array(frame_data, dtype=np.float32)
            current_frame = np.ascontiguousarray(current_frame)
            historical_frames.append(current_frame)

            # 保持历史长度在限制范围内
            if len(historical_frames) > max_context_frames:
                historical_frames.pop(0)

            # 对于前几帧，没有足够的历史信息
            if i < min_context_frames:
                prediction_error_scores.append(0.0)  # 前几帧没有足够历史，惊喜度为0
                entropy_scores.append(0.0)
                representation_change_scores.append(0.0)
                continue

            # 使用所有历史帧来预测当前帧
            if len(historical_frames) > 0:
                # 连接所有历史帧
                historical_audio = np.concatenate(historical_frames)

                # 将历史音频填充或修剪到合适的长度
                historical_audio_padded = whisper_pad_or_trim(historical_audio)

                # 将历史音频转换为mel频谱图
                historical_mel = whisper_log_mel_spectrogram(historical_audio_padded)

                # 确保mel频谱图有正确的形状
                if historical_mel.shape[0] != 80:  # Whisper期望80个mel bins
                    prediction_error_scores.append(0.0)
                    entropy_scores.append(0.0)
                    representation_change_scores.append(0.0)
                    continue

                # mel已经是tensor，只需要添加batch维度
                historical_mel_tensor = historical_mel.unsqueeze(0)

                # 使用Whisper进行编码和预测
                with torch.no_grad():
                    # 获取历史音频的编码器输出
                    encoder_output = model.encoder(historical_mel_tensor.to(model.device))

                    # 使用历史上下文来预测下一个token
                    tokens = torch.tensor([[tokenizer.sot]]).to(encoder_output.device)

                    # 获取decoder输出
                    logits = model.decoder(tokens, encoder_output)

                    # 计算概率分布
                    probs = torch.softmax(logits[0, -1, :], dim=-1)

                    # 计算预测误差作为惊喜度：-log(P_t(Y_t))
                    # 这里我们使用最大概率的负对数作为惊喜度
                    max_prob = torch.max(probs)
                    prediction_error = -torch.log(max_prob + 1e-10).item()

                    # 计算熵作为惊喜度：-sum(P * log(P))
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                    # 计算表征变化作为惊喜度（每帧计算一次）
                    representation_change = 0.0
                    if i >= 1:  # 需要至少1帧的历史数据
                        # 获取当前帧的编码器输出
                        current_audio = np.concatenate(historical_frames[-1:])  # 当前帧
                        current_audio_padded = whisper_pad_or_trim(current_audio)
                        current_mel = whisper_log_mel_spectrogram(current_audio_padded)

                        if current_mel.shape[0] == 80:  # 确保mel频谱图正确
                            current_mel_tensor = current_mel.unsqueeze(0)

                            with torch.no_grad():
                                current_encoder_output = model.encoder(current_mel_tensor.to(model.device))

                                # 计算与前一帧编码器输出的差异
                                # 使用L2距离作为表征变化的度量
                                if len(historical_frames) >= 2:  # 确保有足够的历史数据
                                    # 获取前一帧的编码器输出
                                    prev_audio = np.concatenate(historical_frames[-2:-1])  # 前一帧
                                    prev_audio_padded = whisper_pad_or_trim(prev_audio)
                                    prev_mel = whisper_log_mel_spectrogram(prev_audio_padded)

                                    if prev_mel.shape[0] == 80:
                                        prev_mel_tensor = prev_mel.unsqueeze(0)
                                        prev_encoder_output = model.encoder(prev_mel_tensor.to(model.device))

                                        # 计算编码器输出之间的L2距离
                                        representation_change = torch.norm(
                                            current_encoder_output - prev_encoder_output).item()

                    # 添加调试信息
                    if i % 100 == 0:  # 每100帧打印一次调试信息
                        print(
                            f"帧 {i}: 历史长度={len(historical_frames)}帧, 历史音频长度={len(historical_audio) / 16000:.2f}秒, 预测误差={prediction_error:.4f}, 熵={entropy:.4f}, 表征变化={representation_change:.4f}")

                    prediction_error_scores.append(prediction_error)
                    entropy_scores.append(entropy)
                    representation_change_scores.append(representation_change)
            else:
                prediction_error_scores.append(0.0)
                entropy_scores.append(0.0)
                representation_change_scores.append(0.0)


        except Exception as e:
            print(f"处理第 {i} 帧时出错: {e}")
            print(
                f"Frame type: {type(frame_data)}, Frame shape: {frame_data.shape if hasattr(frame_data, 'shape') else 'N/A'}")
            print(f"Frame dtype: {frame_data.dtype if hasattr(frame_data, 'dtype') else 'N/A'}")
            prediction_error_scores.append(0.0)  # 出错时使用0作为默认值
            entropy_scores.append(0.0)
            representation_change_scores.append(0.0)

            # 即使出错，也要将当前帧加入历史
            try:
                current_frame = np.array(frame_data, dtype=np.float32)
                current_frame = np.ascontiguousarray(current_frame)
                historical_frames.append(current_frame)
                # 保持历史长度在限制范围内
                if len(historical_frames) > max_context_frames:
                    historical_frames.pop(0)
            except:
                pass

    return prediction_error_scores, entropy_scores, representation_change_scores


def analyze_audio_surprise(audio_path, frame_duration_ms=20, model_name="base"):
    """
    分析音频的惊喜度
    """
    print("=" * 50)
    print("开始分析音频惊喜度")
    print("=" * 50)

    # 1. 加载音频
    audio, sr = load_audio_with_correct_sr(audio_path)

    # 2. 分割成帧
    frames = segment_audio_into_frames(audio, sr, frame_duration_ms)

    # 3. 计算每帧的惊喜度
    prediction_error_scores, entropy_scores, representation_change_scores = calculate_whisper_surprise_per_frame(frames,
                                                                                                                 model_name,
                                                                                                                 min_context_frames=5,
                                                                                                                 max_context_seconds=25)

    # 4. 创建时间轴
    time_axis = np.arange(len(prediction_error_scores)) * frame_duration_ms / 1000  # 转换为秒

    # 5. 保存结果
    results_df = pd.DataFrame({
        'frame_index': range(len(prediction_error_scores)),
        'time_seconds': time_axis,
        'prediction_error': prediction_error_scores,
        'entropy': entropy_scores,
        'representation_change': representation_change_scores
    })

    output_file = "audio_surprise_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")

    # 6. 绘制完整结果对比图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

    # 预测误差图
    ax1.plot(time_axis, prediction_error_scores, linewidth=1, alpha=0.7, color='blue', label='Prediction Error -log(P)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Prediction Error (-log P)')
    ax1.set_title('Audio Surprise Analysis - Prediction Error')
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    mean_pred_error = np.mean(prediction_error_scores)
    std_pred_error = np.std(prediction_error_scores)
    ax1.axhline(y=mean_pred_error, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_pred_error:.3f}')
    ax1.legend()

    # 熵图
    ax2.plot(time_axis, entropy_scores, linewidth=1, alpha=0.7, color='red', label='Entropy')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Audio Surprise Analysis - Entropy')
    ax2.grid(True, alpha=0.3)

    # 添加统计信息
    mean_entropy = np.mean(entropy_scores)
    std_entropy = np.std(entropy_scores)
    ax2.axhline(y=mean_entropy, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_entropy:.3f}')
    ax2.legend()

    # 表征变化图
    ax3.plot(time_axis, representation_change_scores, linewidth=1, alpha=0.7, color='green',
             label='Representation Change')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Representation Change (L2 Distance)')
    ax3.set_title('Audio Surprise Analysis - Representation Change')
    ax3.grid(True, alpha=0.3)

    # 添加统计信息
    mean_rep_change = np.mean(representation_change_scores)
    std_rep_change = np.std(representation_change_scores)
    ax3.axhline(y=mean_rep_change, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_rep_change:.3f}')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('audio_surprise_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 7. 绘制三种指标在同一张图上的对比
    plt.figure(figsize=(15, 8))

    # 标准化三种指标以便比较
    pred_error_norm = (prediction_error_scores - np.mean(prediction_error_scores)) / np.std(prediction_error_scores)
    entropy_norm = (entropy_scores - np.mean(entropy_scores)) / np.std(entropy_scores)
    rep_change_norm = (representation_change_scores - np.mean(representation_change_scores)) / np.std(
        representation_change_scores)

    plt.plot(time_axis, pred_error_norm, linewidth=1, alpha=0.7, color='blue', label='Prediction Error (Normalized)')
    plt.plot(time_axis, entropy_norm, linewidth=1, alpha=0.7, color='red', label='Entropy (Normalized)')
    plt.plot(time_axis, rep_change_norm, linewidth=1, alpha=0.7, color='green',
             label='Representation Change (Normalized)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Surprise')
    plt.title('Audio Surprise Analysis - Three Metrics Comparison (Normalized)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('audio_surprise_analysis_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 8. 绘制热力图显示惊喜度变化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # 将惊喜度数据重塑为2D数组用于热力图
    # 每行代表一段时间窗口
    window_size = 100  # 每行100个数据点
    n_windows = len(prediction_error_scores) // window_size

    if n_windows > 0:
        # 重塑数据
        pred_error_matrix = np.array(prediction_error_scores[:n_windows * window_size]).reshape(n_windows, window_size)
        entropy_matrix = np.array(entropy_scores[:n_windows * window_size]).reshape(n_windows, window_size)
        rep_change_matrix = np.array(representation_change_scores[:n_windows * window_size]).reshape(n_windows,
                                                                                                     window_size)

        # 创建时间标签
        time_labels = [f"{i * window_size * frame_duration_ms / 1000:.1f}s"
                       for i in range(0, n_windows, max(1, n_windows // 10))]

        # 绘制预测误差热力图
        sns.heatmap(pred_error_matrix, cmap='viridis', cbar=True, ax=ax1,
                    xticklabels=False, yticklabels=time_labels)
        ax1.set_xlabel('Frame Index (per 100 frames)')
        ax1.set_ylabel('Time Windows')
        ax1.set_title('Prediction Error Heatmap')

        # 绘制熵热力图
        sns.heatmap(entropy_matrix, cmap='plasma', cbar=True, ax=ax2,
                    xticklabels=False, yticklabels=time_labels)
        ax2.set_xlabel('Frame Index (per 100 frames)')
        ax2.set_ylabel('Time Windows')
        ax2.set_title('Entropy Heatmap')

        # 绘制表征变化热力图
        sns.heatmap(rep_change_matrix, cmap='coolwarm', cbar=True, ax=ax3,
                    xticklabels=False, yticklabels=time_labels)
        ax3.set_xlabel('Frame Index (per 100 frames)')
        ax3.set_ylabel('Time Windows')
        ax3.set_title('Representation Change Heatmap')

        plt.tight_layout()
        plt.savefig('audio_surprise_heatmap_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 9. 绘制移动平均线对比
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

    # 计算移动平均
    window_size_ma = 50  # 50帧的移动平均
    pred_error_ma = pd.Series(prediction_error_scores).rolling(window=window_size_ma, center=True).mean()
    entropy_ma = pd.Series(entropy_scores).rolling(window=window_size_ma, center=True).mean()
    rep_change_ma = pd.Series(representation_change_scores).rolling(window=window_size_ma, center=True).mean()

    # 预测误差移动平均
    ax1.plot(time_axis, prediction_error_scores, linewidth=0.5, alpha=0.3, color='lightblue', label='Raw Data')
    ax1.plot(time_axis, pred_error_ma, linewidth=2, color='blue', label=f'Moving Average ({window_size_ma} frames)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Prediction Error (-log P)')
    ax1.set_title('Prediction Error - Moving Average')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 熵移动平均
    ax2.plot(time_axis, entropy_scores, linewidth=0.5, alpha=0.3, color='lightcoral', label='Raw Data')
    ax2.plot(time_axis, entropy_ma, linewidth=2, color='red', label=f'Moving Average ({window_size_ma} frames)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy - Moving Average')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 表征变化移动平均
    ax3.plot(time_axis, representation_change_scores, linewidth=0.5, alpha=0.3, color='lightgreen', label='Raw Data')
    ax3.plot(time_axis, rep_change_ma, linewidth=2, color='green', label=f'Moving Average ({window_size_ma} frames)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Representation Change (L2 Distance)')
    ax3.set_title('Representation Change - Moving Average')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('audio_surprise_moving_avg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 10. 打印统计信息
    print("\n" + "=" * 60)
    print("分析结果统计:")
    print("=" * 60)
    print(f"总帧数: {len(prediction_error_scores)}")

    print("\n预测误差统计:")
    print(f"平均预测误差: {mean_pred_error:.4f}")
    print(f"预测误差标准差: {std_pred_error:.4f}")
    print(f"最大预测误差: {np.max(prediction_error_scores):.4f}")
    print(f"最小预测误差: {np.min(prediction_error_scores):.4f}")
    print(f"预测误差范围: {np.max(prediction_error_scores) - np.min(prediction_error_scores):.4f}")

    print("\n熵统计:")
    print(f"平均熵: {mean_entropy:.4f}")
    print(f"熵标准差: {std_entropy:.4f}")
    print(f"最大熵: {np.max(entropy_scores):.4f}")
    print(f"最小熵: {np.min(entropy_scores):.4f}")
    print(f"熵范围: {np.max(entropy_scores) - np.min(entropy_scores):.4f}")

    print("\n表征变化统计:")
    print(f"平均表征变化: {mean_rep_change:.4f}")
    print(f"表征变化标准差: {std_rep_change:.4f}")
    print(f"最大表征变化: {np.max(representation_change_scores):.4f}")
    print(f"最小表征变化: {np.min(representation_change_scores):.4f}")
    print(f"表征变化范围: {np.max(representation_change_scores) - np.min(representation_change_scores):.4f}")

    # 计算变化率
    pred_error_diff = np.diff(prediction_error_scores)
    entropy_diff = np.diff(entropy_scores)
    rep_change_diff = np.diff(representation_change_scores)

    print("\n预测误差变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(pred_error_diff)):.4f}")
    print(f"变化率标准差: {np.std(pred_error_diff):.4f}")
    print(f"最大正向变化: {np.max(pred_error_diff):.4f}")
    print(f"最大负向变化: {np.min(pred_error_diff):.4f}")

    print("\n熵变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(entropy_diff)):.4f}")
    print(f"变化率标准差: {np.std(entropy_diff):.4f}")
    print(f"最大正向变化: {np.max(entropy_diff):.4f}")
    print(f"最大负向变化: {np.min(entropy_diff):.4f}")

    print("\n表征变化变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(rep_change_diff)):.4f}")
    print(f"变化率标准差: {np.std(rep_change_diff):.4f}")
    print(f"最大正向变化: {np.max(rep_change_diff):.4f}")
    print(f"最大负向变化: {np.min(rep_change_diff):.4f}")

    # 计算三种指标的相关性
    corr_pred_entropy = np.corrcoef(prediction_error_scores, entropy_scores)[0, 1]
    corr_pred_rep = np.corrcoef(prediction_error_scores, representation_change_scores)[0, 1]
    corr_entropy_rep = np.corrcoef(entropy_scores, representation_change_scores)[0, 1]

    print(f"\n相关性分析:")
    print(f"预测误差与熵的相关系数: {corr_pred_entropy:.4f}")
    print(f"预测误差与表征变化的相关系数: {corr_pred_rep:.4f}")
    print(f"熵与表征变化的相关系数: {corr_entropy_rep:.4f}")

    return results_df


if __name__ == "__main__":
    # 设置音频文件路径
    audio_path = r"D:\D\research\audioset下载\指标 音频\女生演讲+警铃.m4a"

    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        print("请确认文件路径是否正确")
    else:
        # 运行分析
        results = analyze_audio_surprise(
            audio_path=audio_path,
            frame_duration_ms=20,  # 20ms一帧
            model_name="base"  # 可以使用 "tiny", "base", "small", "medium", "large"
        )
