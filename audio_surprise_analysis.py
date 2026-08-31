import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor, Wav2Vec2Model
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
    加载音频文件并确保采样率为16kHz（wav2vec2要求的采样率）
    只加载前max_duration_seconds秒的音频
    """
    print(f"正在加载音频文件: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=target_sr, duration=max_duration_seconds)
    print(f"音频长度: {len(audio)/sr:.2f}秒, 采样率: {sr}Hz")
    return audio, sr


def segment_audio_into_frames(audio, sr, frame_duration_ms=25):
    """
    将音频分割成指定时长的帧
    wav2vec2在16kHz采样率下的最小输入长度约为25ms
    """
    frame_length_samples = int(sr * frame_duration_ms / 1000)  # 25ms对应的样本数
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


def calculate_wav2vec2_surprise_per_frame(frames, model_name="facebook/wav2vec2-base", min_context_frames=5, max_context_seconds=25):
    """
    使用wav2vec2计算每帧的惊喜度（真正的预测误差）
    使用wav2vec2的预测机制来预测当前帧，然后计算与真实当前帧的误差
    
    Args:
        frames: 音频帧列表
        model_name: wav2vec2模型名称
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒）
    
    Returns:
        tuple: (cosine_similarity_scores, mse_scores, prediction_scores, representation_change_scores)
    """
    print(f"正在加载wav2vec2模型: {model_name}")
    # 使用Wav2Vec2ForPreTraining来获得量化特征
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # 设置模型为评估模式
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    cosine_similarity_scores = []
    mse_scores = []
    prediction_scores = []
    representation_change_scores = []
    
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
                cosine_similarity_scores.append(0.0)  # 前几帧没有足够历史，惊喜度为0
                mse_scores.append(0.0)
                prediction_scores.append(0.0)
                representation_change_scores.append(0.0)
                continue
            
            # 使用历史帧预测当前帧
            if len(historical_frames) > 0:
                # 准备当前帧作为真实标签
                current_frame = np.array(frame_data, dtype=np.float32)
                current_frame = np.ascontiguousarray(current_frame)
                
                # 使用历史帧（不包括当前帧）来预测当前帧
                historical_audio = np.concatenate(historical_frames)
                
                # 将音频转换为wav2vec2的输入格式
                historical_input = processor(historical_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                current_input = processor(current_frame, sampling_rate=16000, return_tensors="pt", padding=True)
                
                # 使用wav2vec2进行编码和预测
                with torch.no_grad():
                    # 获取历史音频的完整输出（包括量化特征）
                    historical_output = model(**historical_input.to(device))
                    historical_projected_states = historical_output.projected_states
                    historical_projected_quantized_states = historical_output.projected_quantized_states
                    
                    # 获取当前帧的表示（真实标签）
                    current_output = model(**current_input.to(device))
                    current_projected_states = current_output.projected_states
                    current_projected_quantized_states = current_output.projected_quantized_states
                    
                    # 方法一：基于量化特征的预测误差
                    # 使用历史量化特征来预测当前量化特征
                    if historical_projected_quantized_states.size(1) > 0 and current_projected_quantized_states.size(1) > 0:
                        # 使用历史量化特征的平均值作为预测
                        pred_quantized = historical_projected_quantized_states.mean(dim=1, keepdim=True)
                        true_quantized = current_projected_quantized_states.mean(dim=1, keepdim=True)
                        
                        # 余弦相似度：衡量量化特征方向的相似性
                        cosine_sim = torch.cosine_similarity(pred_quantized, true_quantized, dim=-1).item()
                        cosine_surprise = 1 - cosine_sim  # 相似度越低，惊喜度越高
                        
                        # MSE：衡量量化特征数值的差异
                        mse = torch.nn.functional.mse_loss(pred_quantized, true_quantized).item()
                        
                        # 基于量化特征的预测误差
                        prediction_error = mse
                        
                        # 基于对比学习的预测
                        alignment_score = torch.cosine_similarity(pred_quantized, true_quantized, dim=-1).item()
                        prediction_surprise = 1 - alignment_score
                    else:
                        # 如果量化特征不可用，使用投影状态
                        pred_features = historical_projected_states.mean(dim=1, keepdim=True)
                        true_features = current_projected_states.mean(dim=1, keepdim=True)
                        
                        cosine_sim = torch.cosine_similarity(pred_features, true_features, dim=-1).item()
                        cosine_surprise = 1 - cosine_sim
                        
                        mse = torch.nn.functional.mse_loss(pred_features, true_features).item()
                        prediction_error = mse
                        
                        alignment_score = torch.cosine_similarity(pred_features, true_features, dim=-1).item()
                        prediction_surprise = 1 - alignment_score
                    
                    # 计算表征变化（相邻帧之间的差异）
                    representation_change = 0.0
                    if len(historical_frames) >= 2:  # 需要至少2帧的历史数据
                        # 获取前一帧的表示
                        prev_audio = np.concatenate(historical_frames[-2:-1])  # 前一帧
                        prev_input = processor(prev_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                        prev_output = model(**prev_input.to(device))
                        prev_projected_states = prev_output.projected_states
                        prev_projected_quantized_states = prev_output.projected_quantized_states
                        
                        # 计算当前帧与前一帧表示之间的L2距离
                        if current_projected_quantized_states.size(1) > 0 and prev_projected_quantized_states.size(1) > 0:
                            # 使用量化特征计算距离
                            curr_quantized_mean = current_projected_quantized_states.mean(dim=1)
                            prev_quantized_mean = prev_projected_quantized_states.mean(dim=1)
                            representation_change = torch.norm(curr_quantized_mean - prev_quantized_mean).item()
                        else:
                            # 使用投影状态计算距离
                            curr_features_mean = current_projected_states.mean(dim=1)
                            prev_features_mean = prev_projected_states.mean(dim=1)
                            representation_change = torch.norm(curr_features_mean - prev_features_mean).item()
                    
                    # 添加调试信息
                    if i % 100 == 0:  # 每100帧打印一次调试信息
                        print(f"帧 {i}: 历史长度={len(historical_frames)}帧, 余弦相似度惊喜度={cosine_surprise:.4f}, MSE={mse:.4f}, 预测惊喜度={prediction_surprise:.4f}, 表征变化={representation_change:.4f}")
                    
                    cosine_similarity_scores.append(cosine_surprise)
                    mse_scores.append(mse)
                    prediction_scores.append(prediction_surprise)
                    representation_change_scores.append(representation_change)
                
                # 将当前帧加入历史
                historical_frames.append(current_frame)
                if len(historical_frames) > max_context_frames:
                    historical_frames.pop(0)
            else:
                cosine_similarity_scores.append(0.0)
                mse_scores.append(0.0)
                prediction_scores.append(0.0)
                representation_change_scores.append(0.0)
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
            cosine_similarity_scores.append(0.0)  # 出错时使用0作为默认值
            mse_scores.append(0.0)
            prediction_scores.append(0.0)
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
    
    return cosine_similarity_scores, mse_scores, prediction_scores, representation_change_scores


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
    cosine_similarity_scores, mse_scores, prediction_scores, representation_change_scores = calculate_wav2vec2_surprise_per_frame(frames, model_name, min_context_frames=5, max_context_seconds=25)
    
    # 4. 创建时间轴
    time_axis = np.arange(len(cosine_similarity_scores)) * frame_duration_ms / 1000  # 转换为秒
    
    # 5. 保存结果
    results_df = pd.DataFrame({
        'frame_index': range(len(cosine_similarity_scores)),
        'time_seconds': time_axis,
        'cosine_similarity_surprise': cosine_similarity_scores,
        'mse_surprise': mse_scores,
        'prediction_surprise': prediction_scores,
        'representation_change': representation_change_scores
    })
    
    output_file = "audio_surprise_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    # 6. 绘制完整结果对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 余弦相似度惊喜度图
    ax1.plot(time_axis, cosine_similarity_scores, linewidth=1, alpha=0.7, color='blue', label='Cosine Similarity Surprise')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Cosine Similarity Surprise (1 - cos_sim)')
    ax1.set_title('Audio Surprise Analysis - Cosine Similarity Surprise')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_cosine_surprise = np.mean(cosine_similarity_scores)
    std_cosine_surprise = np.std(cosine_similarity_scores)
    ax1.axhline(y=mean_cosine_surprise, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_cosine_surprise:.3f}')
    ax1.legend()
    
    # MSE惊喜度图
    ax2.plot(time_axis, mse_scores, linewidth=1, alpha=0.7, color='orange', label='MSE Surprise')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('MSE Surprise')
    ax2.set_title('Audio Surprise Analysis - MSE Surprise')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_mse_surprise = np.mean(mse_scores)
    std_mse_surprise = np.std(mse_scores)
    ax2.axhline(y=mean_mse_surprise, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_mse_surprise:.3f}')
    ax2.legend()
    
    # 预测惊喜度图
    ax3.plot(time_axis, prediction_scores, linewidth=1, alpha=0.7, color='red', label='Prediction Surprise')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Prediction Surprise')
    ax3.set_title('Audio Surprise Analysis - Prediction Surprise')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_prediction = np.mean(prediction_scores)
    std_prediction = np.std(prediction_scores)
    ax3.axhline(y=mean_prediction, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_prediction:.3f}')
    ax3.legend()
    
    # 表征变化图
    ax4.plot(time_axis, representation_change_scores, linewidth=1, alpha=0.7, color='green', label='Representation Change')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Representation Change (L2 Distance)')
    ax4.set_title('Audio Surprise Analysis - Representation Change')
    ax4.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_rep_change = np.mean(representation_change_scores)
    std_rep_change = np.std(representation_change_scores)
    ax4.axhline(y=mean_rep_change, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_rep_change:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('audio_surprise_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 绘制四种指标在同一张图上的对比
    plt.figure(figsize=(15, 8))
    
    # 标准化四种指标以便比较
    cosine_norm = (cosine_similarity_scores - np.mean(cosine_similarity_scores)) / np.std(cosine_similarity_scores)
    mse_norm = (mse_scores - np.mean(mse_scores)) / np.std(mse_scores)
    prediction_norm = (prediction_scores - np.mean(prediction_scores)) / np.std(prediction_scores)
    rep_change_norm = (representation_change_scores - np.mean(representation_change_scores)) / np.std(representation_change_scores)
    
    plt.plot(time_axis, cosine_norm, linewidth=1, alpha=0.7, color='blue', label='Cosine Similarity Surprise (Normalized)')
    plt.plot(time_axis, mse_norm, linewidth=1, alpha=0.7, color='orange', label='MSE Surprise (Normalized)')
    plt.plot(time_axis, prediction_norm, linewidth=1, alpha=0.7, color='red', label='Prediction Surprise (Normalized)')
    plt.plot(time_axis, rep_change_norm, linewidth=1, alpha=0.7, color='green', label='Representation Change (Normalized)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Surprise')
    plt.title('Audio Surprise Analysis - Four Metrics Comparison (Normalized)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('audio_surprise_analysis_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. 绘制热力图显示惊喜度变化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 将惊喜度数据重塑为2D数组用于热力图
    # 每行代表一段时间窗口
    window_size = 100  # 每行100个数据点
    n_windows = len(cosine_similarity_scores) // window_size
    
    if n_windows > 0:
        # 重塑数据
        cosine_matrix = np.array(cosine_similarity_scores[:n_windows * window_size]).reshape(n_windows, window_size)
        mse_matrix = np.array(mse_scores[:n_windows * window_size]).reshape(n_windows, window_size)
        prediction_matrix = np.array(prediction_scores[:n_windows * window_size]).reshape(n_windows, window_size)
        rep_change_matrix = np.array(representation_change_scores[:n_windows * window_size]).reshape(n_windows, window_size)
        
        # 创建时间标签
        time_labels = [f"{i * window_size * frame_duration_ms / 1000:.1f}s" 
                      for i in range(0, n_windows, max(1, n_windows // 10))]
        
        # 绘制余弦相似度惊喜度热力图
        sns.heatmap(cosine_matrix, cmap='viridis', cbar=True, ax=ax1,
                   xticklabels=False, yticklabels=time_labels)
        ax1.set_xlabel('Frame Index (per 100 frames)')
        ax1.set_ylabel('Time Windows')
        ax1.set_title('Cosine Similarity Surprise Heatmap')
        
        # 绘制MSE惊喜度热力图
        sns.heatmap(mse_matrix, cmap='plasma', cbar=True, ax=ax2,
                   xticklabels=False, yticklabels=time_labels)
        ax2.set_xlabel('Frame Index (per 100 frames)')
        ax2.set_ylabel('Time Windows')
        ax2.set_title('MSE Surprise Heatmap')
        
        # 绘制预测惊喜度热力图
        sns.heatmap(prediction_matrix, cmap='coolwarm', cbar=True, ax=ax3,
                   xticklabels=False, yticklabels=time_labels)
        ax3.set_xlabel('Frame Index (per 100 frames)')
        ax3.set_ylabel('Time Windows')
        ax3.set_title('Prediction Surprise Heatmap')
        
        # 绘制表征变化热力图
        sns.heatmap(rep_change_matrix, cmap='magma', cbar=True, ax=ax4,
                   xticklabels=False, yticklabels=time_labels)
        ax4.set_xlabel('Frame Index (per 100 frames)')
        ax4.set_ylabel('Time Windows')
        ax4.set_title('Representation Change Heatmap')
        
        plt.tight_layout()
        plt.savefig('audio_surprise_heatmap_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 9. 绘制移动平均线对比
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 计算移动平均
    window_size_ma = 50  # 50帧的移动平均
    cosine_ma = pd.Series(cosine_similarity_scores).rolling(window=window_size_ma, center=True).mean()
    mse_ma = pd.Series(mse_scores).rolling(window=window_size_ma, center=True).mean()
    prediction_ma = pd.Series(prediction_scores).rolling(window=window_size_ma, center=True).mean()
    rep_change_ma = pd.Series(representation_change_scores).rolling(window=window_size_ma, center=True).mean()
    
    # 余弦相似度惊喜度移动平均
    ax1.plot(time_axis, cosine_similarity_scores, linewidth=0.5, alpha=0.3, color='lightblue', label='Raw Data')
    ax1.plot(time_axis, cosine_ma, linewidth=2, color='blue', label=f'Moving Average ({window_size_ma} frames)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Cosine Similarity Surprise')
    ax1.set_title('Cosine Similarity Surprise - Moving Average')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # MSE惊喜度移动平均
    ax2.plot(time_axis, mse_scores, linewidth=0.5, alpha=0.3, color='lightcoral', label='Raw Data')
    ax2.plot(time_axis, mse_ma, linewidth=2, color='orange', label=f'Moving Average ({window_size_ma} frames)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('MSE Surprise')
    ax2.set_title('MSE Surprise - Moving Average')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 预测惊喜度移动平均
    ax3.plot(time_axis, prediction_scores, linewidth=0.5, alpha=0.3, color='lightcoral', label='Raw Data')
    ax3.plot(time_axis, prediction_ma, linewidth=2, color='red', label=f'Moving Average ({window_size_ma} frames)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Prediction Surprise')
    ax3.set_title('Prediction Surprise - Moving Average')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 表征变化移动平均
    ax4.plot(time_axis, representation_change_scores, linewidth=0.5, alpha=0.3, color='lightgreen', label='Raw Data')
    ax4.plot(time_axis, rep_change_ma, linewidth=2, color='green', label=f'Moving Average ({window_size_ma} frames)')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Representation Change (L2 Distance)')
    ax4.set_title('Representation Change - Moving Average')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('audio_surprise_moving_avg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. 打印统计信息
    print("\n" + "=" * 60)
    print("分析结果统计:")
    print("=" * 60)
    print(f"总帧数: {len(cosine_similarity_scores)}")
    
    print("\n余弦相似度惊喜度统计:")
    print(f"平均余弦相似度惊喜度: {mean_cosine_surprise:.4f}")
    print(f"余弦相似度惊喜度标准差: {std_cosine_surprise:.4f}")
    print(f"最大余弦相似度惊喜度: {np.max(cosine_similarity_scores):.4f}")
    print(f"最小余弦相似度惊喜度: {np.min(cosine_similarity_scores):.4f}")
    print(f"余弦相似度惊喜度范围: {np.max(cosine_similarity_scores) - np.min(cosine_similarity_scores):.4f}")
    
    print("\nMSE惊喜度统计:")
    print(f"平均MSE惊喜度: {mean_mse_surprise:.4f}")
    print(f"MSE惊喜度标准差: {std_mse_surprise:.4f}")
    print(f"最大MSE惊喜度: {np.max(mse_scores):.4f}")
    print(f"最小MSE惊喜度: {np.min(mse_scores):.4f}")
    print(f"MSE惊喜度范围: {np.max(mse_scores) - np.min(mse_scores):.4f}")
    
    print("\n预测惊喜度统计:")
    print(f"平均预测惊喜度: {mean_prediction:.4f}")
    print(f"预测惊喜度标准差: {std_prediction:.4f}")
    print(f"最大预测惊喜度: {np.max(prediction_scores):.4f}")
    print(f"最小预测惊喜度: {np.min(prediction_scores):.4f}")
    print(f"预测惊喜度范围: {np.max(prediction_scores) - np.min(prediction_scores):.4f}")
    
    print("\n表征变化统计:")
    print(f"平均表征变化: {mean_rep_change:.4f}")
    print(f"表征变化标准差: {std_rep_change:.4f}")
    print(f"最大表征变化: {np.max(representation_change_scores):.4f}")
    print(f"最小表征变化: {np.min(representation_change_scores):.4f}")
    print(f"表征变化范围: {np.max(representation_change_scores) - np.min(representation_change_scores):.4f}")
    
    # 计算变化率
    cosine_diff = np.diff(cosine_similarity_scores)
    mse_diff = np.diff(mse_scores)
    prediction_diff = np.diff(prediction_scores)
    rep_change_diff = np.diff(representation_change_scores)
    
    print("\n余弦相似度惊喜度变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(cosine_diff)):.4f}")
    print(f"变化率标准差: {np.std(cosine_diff):.4f}")
    print(f"最大正向变化: {np.max(cosine_diff):.4f}")
    print(f"最大负向变化: {np.min(cosine_diff):.4f}")
    
    print("\nMSE惊喜度变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(mse_diff)):.4f}")
    print(f"变化率标准差: {np.std(mse_diff):.4f}")
    print(f"最大正向变化: {np.max(mse_diff):.4f}")
    print(f"最大负向变化: {np.min(mse_diff):.4f}")
    
    print("\n预测惊喜度变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(prediction_diff)):.4f}")
    print(f"变化率标准差: {np.std(prediction_diff):.4f}")
    print(f"最大正向变化: {np.max(prediction_diff):.4f}")
    print(f"最大负向变化: {np.min(prediction_diff):.4f}")
    
    print("\n表征变化变化率统计:")
    print(f"平均变化率: {np.mean(np.abs(rep_change_diff)):.4f}")
    print(f"变化率标准差: {np.std(rep_change_diff):.4f}")
    print(f"最大正向变化: {np.max(rep_change_diff):.4f}")
    print(f"最大负向变化: {np.min(rep_change_diff):.4f}")
    
    # 计算四种指标的相关性
    corr_cosine_mse = np.corrcoef(cosine_similarity_scores, mse_scores)[0, 1]
    corr_cosine_prediction = np.corrcoef(cosine_similarity_scores, prediction_scores)[0, 1]
    corr_cosine_rep = np.corrcoef(cosine_similarity_scores, representation_change_scores)[0, 1]
    corr_mse_prediction = np.corrcoef(mse_scores, prediction_scores)[0, 1]
    corr_mse_rep = np.corrcoef(mse_scores, representation_change_scores)[0, 1]
    corr_prediction_rep = np.corrcoef(prediction_scores, representation_change_scores)[0, 1]
    
    print(f"\n相关性分析:")
    print(f"余弦相似度惊喜度与MSE惊喜度的相关系数: {corr_cosine_mse:.4f}")
    print(f"余弦相似度惊喜度与预测惊喜度的相关系数: {corr_cosine_prediction:.4f}")
    print(f"余弦相似度惊喜度与表征变化的相关系数: {corr_cosine_rep:.4f}")
    print(f"MSE惊喜度与预测惊喜度的相关系数: {corr_mse_prediction:.4f}")
    print(f"MSE惊喜度与表征变化的相关系数: {corr_mse_rep:.4f}")
    print(f"预测惊喜度与表征变化的相关系数: {corr_prediction_rep:.4f}")
    
    return results_df


if __name__ == "__main__":
    # 设置音频文件路径
    audio_path = r"D:\D\research\audioset下载\指标 音频\规律重复.m4a"
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        print("请确认文件路径是否正确")
    else:
        # 运行分析
        results = analyze_audio_surprise(
            audio_path=audio_path,
            frame_duration_ms=25,  # 25ms一帧（wav2vec2最小支持长度）
            model_name="facebook/wav2vec2-base"  # 使用wav2vec2模型
        )
