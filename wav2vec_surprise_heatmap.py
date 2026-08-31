import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from datetime import datetime

warnings.filterwarnings("ignore")


class Wav2VecSurpriseCalculator:
    """使用Wav2Vec2计算音频惊喜度的类"""
    
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """
        初始化Wav2Vec2模型
        
        参数:
        model_name: Wav2Vec2模型名称
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载Wav2Vec2模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForPreTraining.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("Wav2Vec2ForPreTraining模型加载成功")
        except Exception as e:
            print(f"加载Wav2Vec2ForPreTraining模型失败: {e}")
            raise
    
    def load_stereo_audio(self, audio_path, target_sr=16000):
        """
        加载立体声音频文件
        
        参数:
        audio_path: 音频文件路径
        target_sr: 目标采样率
        
        返回:
        left_audio: 左声道音频数据
        right_audio: 右声道音频数据
        sr: 采样率
        """
        try:
            # 加载立体声音频
            audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=False)
            
            if audio_data.ndim != 2 or audio_data.shape[0] != 2:
                print(f"跳过 {audio_path}: 不是立体声音频")
                return None, None, None
            
            left_audio, right_audio = audio_data[0], audio_data[1]
            return left_audio, right_audio, sr
            
        except Exception as e:
            print(f"加载音频文件失败 {audio_path}: {e}")
            return None, None, None
    
    def segment_audio_into_frames(self, audio, sr, frame_duration_ms=25):
        """
        将音频分割成指定时长的帧
        wav2vec2在16kHz采样率下的最小输入长度约为25ms
        
        参数:
        audio: 音频数据
        sr: 采样率
        frame_duration_ms: 帧长度（毫秒）
        
        返回:
        frames: 音频帧列表
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
    
    def calculate_mse_surprise(self, frames, min_context_frames=5, max_context_seconds=25):
        """
        使用MSE损失计算惊喜度（参照audio_surprise_analysis.py的处理方式）
        
        参数:
        frames: 音频帧列表
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过模型限制
        
        返回:
        surprise_scores: 每帧的惊喜度分数
        """
        surprise_scores = []
        
        print(f"正在计算MSE惊喜度... (使用最多{max_context_seconds}秒历史信息进行预测)")
        
        # 维护历史帧，但限制最大长度
        historical_frames = []
        frame_duration_seconds = 0.025  # 25ms = 0.025秒
        max_context_frames = int(max_context_seconds / frame_duration_seconds)
        
        for i, frame_data in enumerate(tqdm(frames, desc="计算MSE惊喜度")):
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
                    surprise_scores.append(0.0)  # 前几帧没有足够历史，惊喜度为0
                    continue
                
                # 使用所有历史帧来预测当前帧
                if len(historical_frames) > 0:
                    # 连接所有历史帧
                    historical_audio = np.concatenate(historical_frames)
                    
                    # 将音频转换为wav2vec2的输入格式
                    historical_input = self.processor(historical_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    current_input = self.processor(current_frame, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    # 使用wav2vec2进行编码和预测
                    with torch.no_grad():
                        # 获取历史音频的完整输出（包括量化特征）
                        historical_output = self.model(**historical_input.to(self.device))
                        historical_projected_states = historical_output.projected_states
                        historical_projected_quantized_states = historical_output.projected_quantized_states
                        
                        # 获取当前帧的表示（真实标签）
                        current_output = self.model(**current_input.to(self.device))
                        current_projected_states = current_output.projected_states
                        current_projected_quantized_states = current_output.projected_quantized_states
                        
                        # 使用量化特征计算MSE
                        if historical_projected_quantized_states.size(1) > 0 and current_projected_quantized_states.size(1) > 0:
                            # 使用历史量化特征的平均值作为预测
                            pred_quantized = historical_projected_quantized_states.mean(dim=1, keepdim=True)
                            true_quantized = current_projected_quantized_states.mean(dim=1, keepdim=True)
                            
                            # 计算MSE
                            mse_loss = F.mse_loss(pred_quantized, true_quantized).item()
                        else:
                            # 如果量化特征不可用，使用投影状态
                            pred_features = historical_projected_states.mean(dim=1, keepdim=True)
                            true_features = current_projected_states.mean(dim=1, keepdim=True)
                            
                            # 计算MSE
                            mse_loss = F.mse_loss(pred_features, true_features).item()
                        
                        surprise_scores.append(mse_loss)
                        
                        # 添加调试信息
                        if i % 100 == 0:  # 每100帧打印一次调试信息
                            print(f"帧 {i}: 历史长度={len(historical_frames)}帧, 历史音频长度={len(historical_audio) / 16000:.2f}秒, MSE={mse_loss:.4f}")
                else:
                    surprise_scores.append(0.0)
                    
            except Exception as e:
                print(f"计算第{i}帧MSE惊喜度时出错: {e}")
                surprise_scores.append(0.0)
                
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
    
    def calculate_cosine_similarity_surprise(self, frames, min_context_frames=5, max_context_seconds=25):
        """
        使用余弦相似度计算惊喜度（参照audio_surprise_analysis.py的处理方式）
        
        参数:
        frames: 音频帧列表
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过模型限制
        
        返回:
        surprise_scores: 每帧的惊喜度分数
        """
        surprise_scores = []
        
        print(f"正在计算余弦相似度惊喜度... (使用最多{max_context_seconds}秒历史信息进行预测)")
        
        # 维护历史帧，但限制最大长度
        historical_frames = []
        frame_duration_seconds = 0.025  # 25ms = 0.025秒
        max_context_frames = int(max_context_seconds / frame_duration_seconds)
        
        for i, frame_data in enumerate(tqdm(frames, desc="计算余弦相似度惊喜度")):
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
                    surprise_scores.append(0.0)
                    continue
                
                # 使用所有历史帧来预测当前帧
                if len(historical_frames) > 0:
                    # 连接所有历史帧
                    historical_audio = np.concatenate(historical_frames)
                    
                    # 将音频转换为wav2vec2的输入格式
                    historical_input = self.processor(historical_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    current_input = self.processor(current_frame, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    # 使用wav2vec2进行编码和预测
                    with torch.no_grad():
                        # 获取历史音频的完整输出（包括量化特征）
                        historical_output = self.model(**historical_input.to(self.device))
                        historical_projected_states = historical_output.projected_states
                        historical_projected_quantized_states = historical_output.projected_quantized_states
                        
                        # 获取当前帧的表示（真实标签）
                        current_output = self.model(**current_input.to(self.device))
                        current_projected_states = current_output.projected_states
                        current_projected_quantized_states = current_output.projected_quantized_states
                        
                        # 使用量化特征计算余弦相似度
                        if historical_projected_quantized_states.size(1) > 0 and current_projected_quantized_states.size(1) > 0:
                            # 使用历史量化特征的平均值作为预测
                            pred_quantized = historical_projected_quantized_states.mean(dim=1, keepdim=True)
                            true_quantized = current_projected_quantized_states.mean(dim=1, keepdim=True)
                            
                            # 计算余弦相似度
                            cosine_sim = F.cosine_similarity(pred_quantized, true_quantized, dim=-1).item()
                            
                            # 将相似度转换为惊喜度 (1 - cosine_similarity)
                            surprise = 1 - cosine_sim
                        else:
                            # 如果量化特征不可用，使用投影状态
                            pred_features = historical_projected_states.mean(dim=1, keepdim=True)
                            true_features = current_projected_states.mean(dim=1, keepdim=True)
                            
                            # 计算余弦相似度
                            cosine_sim = F.cosine_similarity(pred_features, true_features, dim=-1).item()
                            
                            # 将相似度转换为惊喜度 (1 - cosine_similarity)
                            surprise = 1 - cosine_sim
                        
                        surprise_scores.append(surprise)
                        
                        # 添加调试信息
                        if i % 100 == 0:  # 每100帧打印一次调试信息
                            print(f"帧 {i}: 历史长度={len(historical_frames)}帧, 历史音频长度={len(historical_audio) / 16000:.2f}秒, 余弦相似度={cosine_sim:.4f}, 惊喜度={surprise:.4f}")
                else:
                    surprise_scores.append(0.0)
                    
            except Exception as e:
                print(f"计算第{i}帧余弦相似度惊喜度时出错: {e}")
                surprise_scores.append(0.0)
                
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
    
    def calculate_representation_change_surprise(self, frames):
        """
        使用表征变化计算惊喜度（参照audio_surprise_analysis.py的处理方式）
        
        参数:
        frames: 音频帧列表
        
        返回:
        surprise_scores: 每帧的惊喜度分数
        """
        surprise_scores = []
        
        print("正在计算表征变化惊喜度...")
        
        for i in tqdm(range(len(frames)), desc="计算表征变化惊喜度"):
            try:
                if i == 0:
                    # 第一帧没有前一帧
                    surprise_scores.append(0.0)
                    continue
                
                # 当前帧
                current_frame = frames[i]
                # 前一帧
                prev_frame = frames[i-1]
                
                # 将音频转换为wav2vec2的输入格式
                current_input = self.processor(current_frame, sampling_rate=16000, return_tensors="pt", padding=True)
                prev_input = self.processor(prev_frame, sampling_rate=16000, return_tensors="pt", padding=True)
                
                # 使用wav2vec2进行编码
                with torch.no_grad():
                    # 处理当前帧
                    current_output = self.model(**current_input.to(self.device))
                    current_projected_states = current_output.projected_states
                    current_projected_quantized_states = current_output.projected_quantized_states
                    
                    # 处理前一帧
                    prev_output = self.model(**prev_input.to(self.device))
                    prev_projected_states = prev_output.projected_states
                    prev_projected_quantized_states = prev_output.projected_quantized_states
                    
                    # 计算表征变化（L2距离）
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
                    
                    surprise_scores.append(representation_change)
                    
            except Exception as e:
                print(f"计算第{i}帧表征变化惊喜度时出错: {e}")
                surprise_scores.append(0.0)
        
        return surprise_scores
    
    def calculate_entropy_surprise(self, frames, min_context_frames=5, max_context_seconds=25):
        """
        使用熵计算惊喜度（参照audio_surprise_analysis.py的处理方式）
        
        参数:
        frames: 音频帧列表
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过模型限制
        
        返回:
        surprise_scores: 每帧的惊喜度分数
        """
        surprise_scores = []
        
        print(f"正在计算熵惊喜度... (使用最多{max_context_seconds}秒历史信息进行预测)")
        
        # 维护历史帧，但限制最大长度
        historical_frames = []
        frame_duration_seconds = 0.025  # 25ms = 0.025秒
        max_context_frames = int(max_context_seconds / frame_duration_seconds)
        
        for i, frame_data in enumerate(tqdm(frames, desc="计算熵惊喜度")):
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
                    surprise_scores.append(0.0)
                    continue
                
                # 使用所有历史帧来预测当前帧
                if len(historical_frames) > 0:
                    # 连接所有历史帧
                    historical_audio = np.concatenate(historical_frames)
                    
                    # 将音频转换为wav2vec2的输入格式
                    historical_input = self.processor(historical_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    current_input = self.processor(current_frame, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    # 使用wav2vec2进行编码和预测
                    with torch.no_grad():
                        # 获取历史音频的完整输出（包括量化特征）
                        historical_output = self.model(**historical_input.to(self.device))
                        historical_projected_states = historical_output.projected_states
                        historical_projected_quantized_states = historical_output.projected_quantized_states
                        
                        # 获取当前帧的表示（真实标签）
                        current_output = self.model(**current_input.to(self.device))
                        current_projected_states = current_output.projected_states
                        current_projected_quantized_states = current_output.projected_quantized_states
                        
                        # 使用量化特征计算熵
                        if historical_projected_quantized_states.size(1) > 0 and current_projected_quantized_states.size(1) > 0:
                            # 使用历史量化特征的平均值作为预测
                            pred_quantized = historical_projected_quantized_states.mean(dim=1, keepdim=True)
                            true_quantized = current_projected_quantized_states.mean(dim=1, keepdim=True)
                            
                            # 计算特征差异的分布
                            diff = torch.abs(pred_quantized - true_quantized)
                            
                            # 计算熵
                            # 将差异归一化为概率分布
                            diff_normalized = F.softmax(diff, dim=-1)
                            entropy = -torch.sum(diff_normalized * torch.log(diff_normalized + 1e-10), dim=-1).item()
                        else:
                            # 如果量化特征不可用，使用投影状态
                            pred_features = historical_projected_states.mean(dim=1, keepdim=True)
                            true_features = current_projected_states.mean(dim=1, keepdim=True)
                            
                            # 计算特征差异的分布
                            diff = torch.abs(pred_features - true_features)
                            
                            # 计算熵
                            # 将差异归一化为概率分布
                            diff_normalized = F.softmax(diff, dim=-1)
                            entropy = -torch.sum(diff_normalized * torch.log(diff_normalized + 1e-10), dim=-1).item()
                        
                        surprise_scores.append(entropy)
                        
                        # 添加调试信息
                        if i % 100 == 0:  # 每100帧打印一次调试信息
                            print(f"帧 {i}: 历史长度={len(historical_frames)}帧, 历史音频长度={len(historical_audio) / 16000:.2f}秒, 熵={entropy:.4f}")
                else:
                    surprise_scores.append(0.0)
                    
            except Exception as e:
                print(f"计算第{i}帧熵惊喜度时出错: {e}")
                surprise_scores.append(0.0)
                
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
    
    def parse_stereo_filename(self, filename):
        """
        解析立体声音频文件名，提取左右声道类别信息
        
        参数:
        filename: 音频文件名
        
        返回:
        left_category: 左声道类别
        right_category: 右声道类别
        """
        try:
            # 移除文件扩展名
            name_without_ext = os.path.splitext(filename)[0]
            
            # 检查是否以 'rn_' 或 'nn_' 开头
            if not (name_without_ext.startswith('rn_') or name_without_ext.startswith('nn_')):
                return None, None
            
            # 移除前缀
            name_without_prefix = name_without_ext[3:]
            
            # 按 '+' 分割左右声道
            if '+' not in name_without_prefix:
                return None, None
            
            left_part, right_part = name_without_prefix.split('+', 1)
            
            # 提取左声道类别
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None
            left_category = left_part[:left_underscore_pos]
            
            # 提取右声道类别
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None
            right_category = right_part[:right_underscore_pos]
            
            return left_category, right_category
            
        except Exception as e:
            print(f"解析文件名失败 {filename}: {e}")
            return None, None
    
    def get_main_category(self, sub_category):
        """
        根据子类别获取主类别
        
        参数:
        sub_category: 子类别名称
        
        返回:
        main_category: 主类别名称
        """
        category_mapping = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
        }
        for main_cat, sub_cats in category_mapping.items():
            if sub_category in sub_cats:
                return main_cat
        return 'Unknown'
    
    def process_stereo_audio_file(self, audio_path, frame_duration_ms=20, min_context_frames=5, max_context_seconds=25):
        """
        处理单个立体声音频文件，计算左右声道的惊喜度
        
        参数:
        audio_path: 音频文件路径
        frame_duration_ms: 帧长度（毫秒）
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过模型限制
        
        返回:
        result: 处理结果字典
        """
        print(f"\n正在处理音频文件: {os.path.basename(audio_path)}")
        
        # 加载立体声音频
        left_audio, right_audio, sr = self.load_stereo_audio(audio_path)
        if left_audio is None:
            return None
        
        # 分割成帧
        left_frames = self.segment_audio_into_frames(left_audio, sr, frame_duration_ms)
        right_frames = self.segment_audio_into_frames(right_audio, sr, frame_duration_ms)
        
        print(f"左声道: {len(left_frames)}帧, 右声道: {len(right_frames)}帧")
        
        # 计算各种惊喜度方法
        methods_results = {}
        
        # MSE惊喜度
        print("计算MSE惊喜度...")
        left_mse_surprise = self.calculate_mse_surprise(left_frames, min_context_frames, max_context_seconds)
        right_mse_surprise = self.calculate_mse_surprise(right_frames, min_context_frames, max_context_seconds)
        methods_results['mse'] = {
            'left_surprise': left_mse_surprise,
            'right_surprise': right_mse_surprise,
            'left_sum': np.sum(left_mse_surprise),
            'right_sum': np.sum(right_mse_surprise),
            'difference': np.sum(left_mse_surprise) - np.sum(right_mse_surprise)
        }
        
        # 余弦相似度惊喜度
        print("计算余弦相似度惊喜度...")
        left_cosine_surprise = self.calculate_cosine_similarity_surprise(left_frames, min_context_frames, max_context_seconds)
        right_cosine_surprise = self.calculate_cosine_similarity_surprise(right_frames, min_context_frames, max_context_seconds)
        methods_results['cosine'] = {
            'left_surprise': left_cosine_surprise,
            'right_surprise': right_cosine_surprise,
            'left_sum': np.sum(left_cosine_surprise),
            'right_sum': np.sum(right_cosine_surprise),
            'difference': np.sum(left_cosine_surprise) - np.sum(right_cosine_surprise)
        }
        
        # 表征变化惊喜度
        print("计算表征变化惊喜度...")
        left_rep_change_surprise = self.calculate_representation_change_surprise(left_frames)
        right_rep_change_surprise = self.calculate_representation_change_surprise(right_frames)
        methods_results['representation_change'] = {
            'left_surprise': left_rep_change_surprise,
            'right_surprise': right_rep_change_surprise,
            'left_sum': np.sum(left_rep_change_surprise),
            'right_sum': np.sum(right_rep_change_surprise),
            'difference': np.sum(left_rep_change_surprise) - np.sum(right_rep_change_surprise)
        }
        
        # 熵惊喜度
        print("计算熵惊喜度...")
        left_entropy_surprise = self.calculate_entropy_surprise(left_frames, min_context_frames, max_context_seconds)
        right_entropy_surprise = self.calculate_entropy_surprise(right_frames, min_context_frames, max_context_seconds)
        methods_results['entropy'] = {
            'left_surprise': left_entropy_surprise,
            'right_surprise': right_entropy_surprise,
            'left_sum': np.sum(left_entropy_surprise),
            'right_sum': np.sum(right_entropy_surprise),
            'difference': np.sum(left_entropy_surprise) - np.sum(right_entropy_surprise)
        }
        
        # 解析文件名获取类别信息
        left_category, right_category = self.parse_stereo_filename(os.path.basename(audio_path))
        
        # 构建结果
        result = {
            'audio_file': os.path.basename(audio_path),
            'audio_path': audio_path,
            'frame_duration_ms': frame_duration_ms,
            'sampling_rate': sr,
            'left_frames_count': len(left_frames),
            'right_frames_count': len(right_frames),
            'left_category': left_category,
            'right_category': right_category,
            'methods_results': methods_results,
            'processing_time': datetime.now().isoformat()
        }
        
        print(f"处理完成 - MSE差值: {methods_results['mse']['difference']:.4f}, "
              f"余弦相似度差值: {methods_results['cosine']['difference']:.4f}, "
              f"表征变化差值: {methods_results['representation_change']['difference']:.4f}, "
              f"熵差值: {methods_results['entropy']['difference']:.4f}")
        
        return result
    
    def plot_surprise_heatmaps(self, results, output_dir, prefix=""):
        """
        绘制惊喜度热力图
        
        参数:
        results: 处理结果列表
        output_dir: 输出目录
        prefix: 文件名前缀（nn或rn）
        """
        prefix_str = f"[{prefix.upper()}] " if prefix else ""
        print(f"=== {prefix_str}开始绘制惊喜度热力图 ===")
        
        # 定义标准类别顺序
        target_categories = [
            'Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry',  # 高生态效度
            'Computer keyboard', 'Helicopter', 'Chicken, rooster',  # 低生态效度
            'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',  # speech
            'Bass drum', 'Funny music', 'Sad music',  # music
            'Pulse', 'Whack, thwack', 'Crumpling, crinkling'  # 未知声源
        ]
        
        # 定义主类别组
        main_category_groups = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
        }
        
        # 定义主类别颜色
        main_category_colors = {
            'High Ecology': '#FF6B6B',  # 红色系
            'Low Ecology': '#DDA0DD',  # 紫色系
            'speech': '#45B7D1',  # 蓝色系
            'music': '#96CEB4',  # 绿色系
            'Unknown Source': '#FFEAA7',  # 黄色系
        }
        
        # 为每个子类别生成颜色
        category_colors = {}
        for cat in target_categories:
            # 找到该子类别所属的主类别
            main_cat = None
            for main_cat_name, sub_cats in main_category_groups.items():
                if cat in sub_cats:
                    main_cat = main_cat_name
                    break
            
            if main_cat:
                base_color = main_category_colors.get(main_cat, '#DDA0DD')
                # 根据在主类别中的位置调整亮度
                sub_cats = main_category_groups[main_cat]
                cat_index = sub_cats.index(cat)
                brightness = 0.4 + 0.6 * (cat_index / (len(sub_cats) - 1)) if len(sub_cats) > 1 else 0.7
                
                # 调整颜色亮度
                import matplotlib.colors as mcolors
                base_rgb = mcolors.to_rgb(base_color)
                adjusted_color = tuple(c * brightness for c in base_rgb)
                category_colors[cat] = adjusted_color
            else:
                category_colors[cat] = '#DDA0DD'
        
        # 检查是否有有效的结果数据
        valid_results = [r for r in results if r is not None and r.get('left_category') and r.get('right_category')]
        if not valid_results:
            print(f"{prefix_str}没有有效的结果数据，跳过热力图绘制")
            return
        
        print(f"{prefix_str}找到{len(valid_results)}个有效结果，开始绘制热力图")
        
        # 为每种惊喜度方法绘制热力图
        methods = ['mse', 'cosine', 'representation_change', 'entropy']
        method_names = ['MSE Loss', 'Cosine Similarity', 'Representation Change', 'Entropy']
        
        for method, method_name in zip(methods, method_names):
            print(f"{prefix_str}正在绘制{method_name}热力图...")
            
            # 创建选择矩阵
            n_categories = len(target_categories)
            selection_matrix = np.full((n_categories, n_categories), np.nan)
            color_matrix = np.full((n_categories, n_categories, 3), np.nan)
            
            # 统计每种配对的选择结果
            for result in valid_results:
                if result is None:
                    continue
                
                left_cat = result['left_category']
                right_cat = result['right_category']
                
                if left_cat in target_categories and right_cat in target_categories:
                    left_idx = target_categories.index(left_cat)
                    right_idx = target_categories.index(right_cat)
                    
                    # 获取该方法的结果
                    method_result = result['methods_results'].get(method)
                    if method_result is None:
                        continue
                    
                    # 根据差值决定选择哪个声道
                    difference = method_result['difference']
                    if difference > 0:
                        # 左声道惊喜度更大，选择左声道
                        selected_cat = left_cat
                    else:
                        # 右声道惊喜度更大，选择右声道
                        selected_cat = right_cat
                    
                    # 获取选中类别的颜色
                    if selected_cat in category_colors:
                        selected_color = category_colors[selected_cat]
                        if isinstance(selected_color, tuple):
                            color_matrix[left_idx, right_idx] = selected_color
                        else:
                            rgb_color = mcolors.to_rgb(selected_color)
                            color_matrix[left_idx, right_idx] = rgb_color
                        
                        # 设置标识值表示配对存在
                        selection_matrix[left_idx, right_idx] = 1
            
            # 绘制热力图
            fig, ax = plt.subplots(1, 1, figsize=(18, 16))
            
            # 先绘制背景（白色）
            ax.imshow(np.ones((n_categories, n_categories)), cmap='gray', vmin=0, vmax=1, alpha=0.3)
            
            # 为每个单元格绘制颜色
            for i in range(n_categories):
                for j in range(n_categories):
                    if not np.isnan(color_matrix[i, j, 0]):  # 如果有颜色数据
                        # 绘制彩色矩形
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=color_matrix[i, j],
                                           edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
            
            # 设置标签
            ax.set_xticks(range(n_categories))
            ax.set_yticks(range(n_categories))
            
            # 设置X轴标签（右声道类别）
            x_labels = ax.set_xticklabels(target_categories, rotation=45, ha='right', fontsize=9)
            for i, label in enumerate(x_labels):
                label.set_color(category_colors[target_categories[i]])
                label.set_fontweight('bold')
            
            # 设置Y轴标签（左声道类别）
            y_labels = ax.set_yticklabels(target_categories, fontsize=9)
            for i, label in enumerate(y_labels):
                label.set_color(category_colors[target_categories[i]])
                label.set_fontweight('bold')
            
            # 添加类别名称注释
            for i in range(n_categories):
                for j in range(n_categories):
                    if not np.isnan(color_matrix[i, j, 0]):  # 如果有配对
                        # 获取该配对选中的类别
                        for result in valid_results:
                            if result is None:
                                continue
                            left_cat = result['left_category']
                            right_cat = result['right_category']
                            if (left_cat == target_categories[i] and
                                    right_cat == target_categories[j]):
                                method_result = result['methods_results'].get(method)
                                if method_result is None:
                                    continue
                                
                                difference = method_result['difference']
                                if difference > 0:
                                    selected_cat = left_cat
                                else:
                                    selected_cat = right_cat
                                
                                # 显示选中类别的名称（简化版本）
                                short_name = selected_cat.split(',')[0]  # 取逗号前的部分
                                ax.text(j, i, short_name, ha='center', va='center',
                                        fontweight='bold', color='white', fontsize=6)
                                break
            
            title_prefix = f"[{prefix.upper()}] " if prefix else ""
            ax.set_title(f'{title_prefix}{method_name} Surprise Selection Results\n(Color = Selected Category)',
                         fontsize=16, fontweight='bold')
            ax.set_xlabel('Right Channel Category', fontsize=12, fontweight='bold')
            ax.set_ylabel('Left Channel Category', fontsize=12, fontweight='bold')
            
            # 添加网格线
            ax.set_xticks(np.arange(-0.5, n_categories, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_categories, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 设置坐标轴范围
            ax.set_xlim(-0.5, n_categories - 0.5)
            ax.set_ylim(-0.5, n_categories - 0.5)
            
            plt.tight_layout()
            
            # 保存图片
            safe_method_name = method_name.replace(' ', '_').lower()
            prefix_suffix = f"_{prefix}" if prefix else ""
            heatmap_path = os.path.join(output_dir, f"wav2vec_{safe_method_name}_surprise_heatmap{prefix_suffix}.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"{prefix_str}{method_name}热力图已保存: {heatmap_path}")
        
        print(f"\n{prefix_str}所有惊喜度热力图绘制完成！")
        print(f"  - MSE Loss热力图: 1张")
        print(f"  - Cosine Similarity热力图: 1张")
        print(f"  - Representation Change热力图: 1张")
        print(f"  - Entropy热力图: 1张")
        print(f"  - 总计: 4张热力图")
    
    def process_directory(self, input_dir, output_dir, frame_duration_ms=20, min_context_frames=5, max_context_seconds=25, max_files=50):
        """
        批量处理目录中的立体声音频文件，分别处理nn_和rn_开头的文件
        
        参数:
        input_dir: 输入目录
        output_dir: 输出目录
        frame_duration_ms: 帧长度（毫秒）
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒），避免超过模型限制
        max_files: 最大处理文件数
        """
        print(f"开始批量处理目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"帧长度: {frame_duration_ms}ms")
        print(f"最少上下文帧数: {min_context_frames}")
        print(f"最大历史音频长度: {max_context_seconds}秒")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找音频文件
        audio_path = Path(input_dir)
        audio_files = list(audio_path.glob("*.wav"))
        
        if not audio_files:
            print(f"在{input_dir}中未找到音频文件")
            return
        
        print(f"找到{len(audio_files)}个音频文件")
        
        # 分别筛选nn_和rn_开头的文件
        nn_files = [f for f in audio_files if f.name.startswith('nn_')]
        rn_files = [f for f in audio_files if f.name.startswith('rn_')]
        
        print(f"找到{len(nn_files)}个nn_开头的文件")
        print(f"找到{len(rn_files)}个rn_开头的文件")
        
        # 初始化结果变量
        nn_results = []
        rn_results = []
        
        # 处理nn_开头的文件
        if nn_files:
            print(f"\n=== 开始处理nn_开头的文件 ===")
            nn_results = self.process_file_group(nn_files, "nn", max_files, frame_duration_ms, min_context_frames, max_context_seconds, output_dir)
            
            if nn_results:
                print(f"\n=== 绘制nn_文件的热力图 ===")
                self.plot_surprise_heatmaps(nn_results, output_dir, prefix="nn")
                self.save_results(nn_results, output_dir, prefix="nn")
            else:
                print(f"[NN] 没有成功处理任何文件，跳过热力图绘制")
        
        # 处理rn_开头的文件
        if rn_files:
            print(f"\n=== 开始处理rn_开头的文件 ===")
            rn_results = self.process_file_group(rn_files, "rn", max_files, frame_duration_ms, min_context_frames, max_context_seconds, output_dir)
            
            if rn_results:
                print(f"\n=== 绘制rn_文件的热力图 ===")
                self.plot_surprise_heatmaps(rn_results, output_dir, prefix="rn")
                self.save_results(rn_results, output_dir, prefix="rn")
            else:
                print(f"[RN] 没有成功处理任何文件，跳过热力图绘制")
        
        print(f"\n=== 批量处理完成！===")
        print(f"nn_文件处理了: {len(nn_results)}个")
        print(f"rn_文件处理了: {len(rn_results)}个")
        print(f"总计处理了: {len(nn_results) + len(rn_results)}个文件")
        
        # 显示增量保存的文件信息
        print(f"\n=== 增量保存文件信息 ===")
        if nn_files:
            incremental_csv_nn = os.path.join(output_dir, "wav2vec_surprise_incremental_nn.csv")
            incremental_json_nn = os.path.join(output_dir, "wav2vec_surprise_incremental_nn.json")
            if os.path.exists(incremental_csv_nn):
                print(f"[NN] 增量CSV文件: {incremental_csv_nn}")
            if os.path.exists(incremental_json_nn):
                print(f"[NN] 增量JSON文件: {incremental_json_nn}")
        
        if rn_files:
            incremental_csv_rn = os.path.join(output_dir, "wav2vec_surprise_incremental_rn.csv")
            incremental_json_rn = os.path.join(output_dir, "wav2vec_surprise_incremental_rn.json")
            if os.path.exists(incremental_csv_rn):
                print(f"[RN] 增量CSV文件: {incremental_csv_rn}")
            if os.path.exists(incremental_json_rn):
                print(f"[RN] 增量JSON文件: {incremental_json_rn}")
    
    def process_file_group(self, audio_files, group_name, max_files, frame_duration_ms, min_context_frames, max_context_seconds, output_dir):
        """
        处理一组音频文件，每处理完一个文件就保存结果
        
        参数:
        audio_files: 音频文件列表
        group_name: 组名（nn或rn）
        max_files: 最大处理文件数
        frame_duration_ms: 帧长度（毫秒）
        min_context_frames: 最少需要多少帧历史信息才开始预测
        max_context_seconds: 最大历史音频长度（秒）
        output_dir: 输出目录
        
        返回:
        results: 处理结果列表
        """
        # 随机选择文件
        if len(audio_files) > max_files:
            audio_files = random.sample(audio_files, max_files)
            print(f"随机选择了{len(audio_files)}个{group_name}_开头的音频文件进行处理")
        else:
            print(f"使用所有{len(audio_files)}个{group_name}_开头的音频文件")
        
        # 处理音频文件
        results = []
        total_files = len(audio_files)
        
        # 创建增量保存文件路径
        prefix_suffix = f"_{group_name}"
        incremental_csv_path = os.path.join(output_dir, f"wav2vec_surprise_incremental{prefix_suffix}.csv")
        incremental_json_path = os.path.join(output_dir, f"wav2vec_surprise_incremental{prefix_suffix}.json")
        
        # 初始化CSV文件（写入表头）
        import pandas as pd
        csv_headers = ['audio_file', 'left_category', 'right_category', 'method', 'left_sum', 'right_sum', 'difference', 'selected_side', 'processing_time']
        pd.DataFrame(columns=csv_headers).to_csv(incremental_csv_path, index=False, encoding='utf-8-sig')
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{group_name.upper()}] 处理进度: {i}/{total_files} - {audio_file.name}")
            result = self.process_stereo_audio_file(
                str(audio_file), 
                frame_duration_ms=frame_duration_ms,
                min_context_frames=min_context_frames,
                max_context_seconds=max_context_seconds
            )
            if result is not None:
                results.append(result)
                print(f"[{group_name.upper()}] 成功处理: {audio_file.name}")
                
                # 立即保存这个文件的结果
                self.save_single_result(result, incremental_csv_path, incremental_json_path, group_name)
                print(f"[{group_name.upper()}] 结果已保存到增量文件")
            else:
                print(f"[{group_name.upper()}] 跳过: {audio_file.name}")
        
        print(f"\n[{group_name.upper()}] 组处理完成，成功处理了{len(results)}个文件")
        return results
    
    def save_single_result(self, result, csv_path, json_path, group_name):
        """
        保存单个音频文件的处理结果
        
        参数:
        result: 单个音频文件的处理结果
        csv_path: CSV文件路径
        json_path: JSON文件路径
        group_name: 组名（nn或rn）
        """
        import pandas as pd
        import json
        
        # 准备CSV数据
        csv_data = []
        methods_results = result['methods_results']
        for method, method_result in methods_results.items():
            csv_data.append({
                'audio_file': result['audio_file'],
                'left_category': result['left_category'],
                'right_category': result['right_category'],
                'method': method,
                'left_sum': method_result['left_sum'],
                'right_sum': method_result['right_sum'],
                'difference': method_result['difference'],
                'selected_side': 'Left' if method_result['difference'] > 0 else 'Right',
                'processing_time': result['processing_time']
            })
        
        # 追加到CSV文件
        df_single = pd.DataFrame(csv_data)
        df_single.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        # 更新JSON文件（读取现有数据，添加新结果，重新保存）
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        
        existing_data.append(result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2, default=str)
    
    def save_results(self, results, output_dir, prefix=""):
        """
        保存处理结果
        
        参数:
        results: 处理结果列表
        output_dir: 输出目录
        prefix: 文件名前缀（nn或rn）
        """
        # 保存详细结果
        detailed_data = []
        for result in results:
            if result is None:
                continue
            
            methods_results = result['methods_results']
            for method, method_result in methods_results.items():
                detailed_data.append({
                    'audio_file': result['audio_file'],
                    'left_category': result['left_category'],
                    'right_category': result['right_category'],
                    'method': method,
                    'left_sum': method_result['left_sum'],
                    'right_sum': method_result['right_sum'],
                    'difference': method_result['difference'],
                    'selected_side': 'Left' if method_result['difference'] > 0 else 'Right',
                    'processing_time': result['processing_time']
                })
        
        df_detailed = pd.DataFrame(detailed_data)
        prefix_suffix = f"_{prefix}" if prefix else ""
        detailed_path = os.path.join(output_dir, f"wav2vec_surprise_detailed{prefix_suffix}.csv")
        df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        
        prefix_str = f"[{prefix.upper()}] " if prefix else ""
        print(f"{prefix_str}详细结果已保存: {detailed_path}")
        
        # 保存原始结果数据（JSON格式）
        import json
        raw_results_path = os.path.join(output_dir, f"wav2vec_surprise_raw_results{prefix_suffix}.json")
        with open(raw_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"{prefix_str}原始结果数据已保存: {raw_results_path}")


def main():
    """主函数"""
    print("=== Wav2Vec2 Surprise Heatmap Tool ===")
    
    # 设置参数（参照audio_surprise_analysis.py的参数设置）
    model_name = "facebook/wav2vec2-base"  # 可以使用其他Wav2Vec2模型
    frame_duration_ms = 25  # 25ms per frame（wav2vec2最小支持长度）
    min_context_frames = 5  # 最少需要多少帧历史信息才开始预测
    max_context_seconds = 25  # 最大历史音频长度（秒），避免超过模型限制
    
    # 设置路径
    input_dir = r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120"  # 输入目录
    output_dir = r"D:\D\research\audioset下载\wav2vec_surprise_heatmaps"  # 输出目录
    
    # 创建惊喜度计算器
    calculator = Wav2VecSurpriseCalculator(model_name=model_name)
    
    # 批量处理立体声音频文件
    calculator.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        frame_duration_ms=frame_duration_ms,
        min_context_frames=min_context_frames,
        max_context_seconds=max_context_seconds,
        max_files=100  # 处理最多100个文件
    )


if __name__ == "__main__":
    main()
