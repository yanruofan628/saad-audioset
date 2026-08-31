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
    print(f"音频长度: {len(audio)/sr:.2f}秒, 采样率: {sr}Hz")
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
    使用Whisper计算每帧的惊喜度（通过预测下一帧的误差）
    使用历史信息预测下一帧，然后用真实下一帧的概率计算惊喜度
    
    Args:
        frames: 音频帧列表
        model_name: Whisper模型名称
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过Whisper的30秒限制
    """
    print(f"正在加载Whisper模型: {model_name}")
    model = whisper.load_model(model_name)
    
    # 获取tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    
    surprise_scores = []
    
    print(f"正在计算每帧的惊喜度... (使用最多{max_context_seconds}秒历史信息进行预测)")
    
    # 维护历史帧，但限制最大长度
    historical_frames = []
    frame_duration_seconds = 0.02  # 20ms = 0.02秒
    max_context_frames = int(max_context_seconds / frame_duration_seconds)
    
    for i, frame_data in enumerate(tqdm(frames, desc="处理帧")):
        try:
            # 对于前几帧，没有足够的历史信息进行预测
            if i < min_context_frames:
                # 将当前帧加入历史
                current_frame = np.array(frame_data, dtype=np.float32)
                current_frame = np.ascontiguousarray(current_frame)
                historical_frames.append(current_frame)
                surprise_scores.append(0.0)  # 前几帧没有足够历史，惊喜度为0
                continue
            
            # 使用历史帧预测当前帧
            if len(historical_frames) > 0:
                # 连接所有历史帧（不包括当前帧）
                historical_audio = np.concatenate(historical_frames)
                
                # 准备当前帧作为真实标签
                current_frame = np.array(frame_data, dtype=np.float32)
                current_frame = np.ascontiguousarray(current_frame)
                
                # 将历史音频填充或修剪到合适的长度
                historical_audio_padded = whisper_pad_or_trim(historical_audio)
                
                # 将历史音频转换为mel频谱图
                historical_mel = whisper_log_mel_spectrogram(historical_audio_padded)
                
                # 将当前帧也转换为mel频谱图（作为真实标签）
                current_audio_padded = whisper_pad_or_trim(current_frame)
                current_mel = whisper_log_mel_spectrogram(current_audio_padded)
                
                # 确保mel频谱图有正确的形状
                if historical_mel.shape[0] != 80 or current_mel.shape[0] != 80:
                    surprise_scores.append(0.0)
                    # 将当前帧加入历史
                    historical_frames.append(current_frame)
                    if len(historical_frames) > max_context_frames:
                        historical_frames.pop(0)
                    continue
                    
                # 使用Whisper进行编码和预测
                with torch.no_grad():
                    # 获取历史音频的编码器输出
                    historical_mel_tensor = historical_mel.unsqueeze(0)
                    encoder_output = model.encoder(historical_mel_tensor.to(model.device))
                    
                    # 获取当前帧的编码器输出（作为真实标签）
                    current_mel_tensor = current_mel.unsqueeze(0)
                    current_encoder_output = model.encoder(current_mel_tensor.to(model.device))
                    
                    # 使用历史上下文来预测
                    tokens = torch.tensor([[tokenizer.sot]]).to(encoder_output.device)
                    
                    # 获取decoder输出
                    logits = model.decoder(tokens, encoder_output)
                    
                    # 计算概率分布
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    
                    # 计算真实标签的惊喜度：-log(P(真实标签))
                    # 使用当前帧编码器输出与预测的相似度作为真实标签概率的代理
                    # 计算编码器输出的相似度
                    similarity = torch.cosine_similarity(
                        encoder_output.mean(dim=1),  # 历史编码器输出的平均
                        current_encoder_output.mean(dim=1),  # 当前帧编码器输出的平均
                        dim=-1
                    )
                    
                    # 将相似度转换为概率（使用sigmoid）
                    true_label_prob = torch.sigmoid(similarity).item()
                    
                    # 计算惊喜度：-log(P(真实标签))
                    surprise_score = -np.log(true_label_prob + 1e-10)
                    
                    # 添加调试信息
                    if i % 100 == 0:  # 每100帧打印一次调试信息
                        print(f"帧 {i}: 历史长度={len(historical_frames)}帧, 真实标签概率={true_label_prob:.4f}, 惊喜度={surprise_score:.4f}")
                    
                    surprise_scores.append(surprise_score)
                
                # 将当前帧加入历史
                historical_frames.append(current_frame)
                if len(historical_frames) > max_context_frames:
                    historical_frames.pop(0)
            else:
                surprise_scores.append(0.0)
                # 将当前帧加入历史
                current_frame = np.array(frame_data, dtype=np.float32)
                current_frame = np.ascontiguousarray(current_frame)
                historical_frames.append(current_frame)
                if len(historical_frames) > max_context_frames:
                    historical_frames.pop(0)
                
        except Exception as e:
            print(f"处理第 {i} 帧时出错: {e}")
            print(f"Frame type: {type(frame_data)}, Frame shape: {frame_data.shape if hasattr(frame_data, 'shape') else 'N/A'}")
            print(f"Frame dtype: {frame_data.dtype if hasattr(frame_data, 'dtype') else 'N/A'}")
            surprise_scores.append(0.0)  # 出错时使用0作为默认值
            
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
    
    return surprise_scores


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
    surprise_scores = calculate_whisper_surprise_per_frame(frames, model_name, min_context_frames=5, max_context_seconds=25)
    
    # 4. 创建时间轴
    time_axis = np.arange(len(surprise_scores)) * frame_duration_ms / 1000  # 转换为秒
    
    # 5. 保存结果
    results_df = pd.DataFrame({
        'frame_index': range(len(surprise_scores)),
        'time_seconds': time_axis,
        'surprise_score': surprise_scores
    })
    
    output_file = "whisper_surprise_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    # 6. 绘制完整结果
    plt.figure(figsize=(15, 8))
    plt.plot(time_axis, surprise_scores, linewidth=1, alpha=0.7)
    plt.xlabel('时间 (秒)')
    plt.ylabel('惊喜度 (-log P(真实标签))')
    plt.title('音频惊喜度分析 - Whisper真实标签概率')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_surprise = np.mean(surprise_scores)
    std_surprise = np.std(surprise_scores)
    plt.axhline(y=mean_surprise, color='r', linestyle='--', alpha=0.7, 
                label=f'平均惊喜度: {mean_surprise:.3f}')
    plt.axhline(y=mean_surprise + std_surprise, color='orange', linestyle=':', alpha=0.7,
                label=f'平均值+标准差: {mean_surprise + std_surprise:.3f}')
    plt.axhline(y=mean_surprise - std_surprise, color='orange', linestyle=':', alpha=0.7,
                label=f'平均值-标准差: {mean_surprise - std_surprise:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('whisper_surprise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 绘制30秒到31.15秒的放大图
    start_time = 30.0
    end_time = 31.15
    
    # 找到对应的时间索引
    start_idx = int(start_time * 1000 / frame_duration_ms)  # 转换为帧索引
    end_idx = int(end_time * 1000 / frame_duration_ms)
    
    # 确保索引在有效范围内
    start_idx = max(0, min(start_idx, len(time_axis)-1))
    end_idx = max(start_idx+1, min(end_idx, len(time_axis)))
    
    # 提取对应时间段的数据
    zoom_time_axis = time_axis[start_idx:end_idx]
    zoom_surprise_scores = surprise_scores[start_idx:end_idx]
    
    # 绘制放大图
    plt.figure(figsize=(15, 6))
    plt.plot(zoom_time_axis, zoom_surprise_scores, linewidth=2, alpha=0.8, color='blue')
    plt.xlabel('时间 (秒)')
    plt.ylabel('惊喜度 (-log P(真实标签))')
    plt.title(f'音频惊喜度分析 - 放大视图 ({start_time}s - {end_time}s)')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    zoom_mean = np.mean(zoom_surprise_scores)
    zoom_std = np.std(zoom_surprise_scores)
    plt.axhline(y=zoom_mean, color='r', linestyle='--', alpha=0.7, 
                label=f'局部平均惊喜度: {zoom_mean:.4f}')
    plt.axhline(y=zoom_mean + zoom_std, color='orange', linestyle=':', alpha=0.7,
                label=f'局部平均+标准差: {zoom_mean + zoom_std:.4f}')
    plt.axhline(y=zoom_mean - zoom_std, color='orange', linestyle=':', alpha=0.7,
                label=f'局部平均-标准差: {zoom_mean - zoom_std:.4f}')
    plt.legend()
    
    # 设置x轴刻度，显示更精确的时间
    plt.xticks(np.arange(start_time, end_time + 0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('whisper_surprise_analysis_zoom_30s.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印放大区域的统计信息
    print(f"\n放大区域统计 ({start_time}s - {end_time}s):")
    print(f"数据点数: {len(zoom_surprise_scores)}")
    print(f"平均惊喜度: {zoom_mean:.4f}")
    print(f"标准差: {zoom_std:.4f}")
    print(f"最大值: {np.max(zoom_surprise_scores):.4f}")
    print(f"最小值: {np.min(zoom_surprise_scores):.4f}")
    print(f"范围: {np.max(zoom_surprise_scores) - np.min(zoom_surprise_scores):.4f}")
    
    # 7. 打印统计信息
    print("\n" + "=" * 50)
    print("分析结果统计:")
    print("=" * 50)
    print(f"总帧数: {len(surprise_scores)}")
    print(f"平均惊喜度: {mean_surprise:.4f}")
    print(f"惊喜度标准差: {std_surprise:.4f}")
    print(f"最大惊喜度: {np.max(surprise_scores):.4f}")
    print(f"最小惊喜度: {np.min(surprise_scores):.4f}")
    print(f"惊喜度范围: {np.max(surprise_scores) - np.min(surprise_scores):.4f}")
    
    return results_df


if __name__ == "__main__":
    # 设置音频文件路径
    audio_path = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\audio_stimuli\full_sequence.wav"
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
    else:
        # 运行分析
        results = analyze_audio_surprise(
            audio_path=audio_path,
            frame_duration_ms=20,  # 20ms一帧
            model_name="base"  # 可以使用 "tiny", "base", "small", "medium", "large"
        )
