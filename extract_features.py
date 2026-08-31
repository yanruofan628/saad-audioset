import pandas as pd
import os
import numpy as np
import librosa
import tqdm
import pickle
from pathlib import Path
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import shutil

def validate_folder(folder_path):
    """验证文件夹路径是否有效"""
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return True
    else:
        print(f"文件夹不存在: {folder_path}")
        return False

def extract_audio_features(audio_file, sr=22050, hop_length=176, n_fft=2048):
    """
    按照声学感知模型提取9个特征
    
    处理流程：
    1. 基础音频处理：加载音频，计算STFT
    2. 耳蜗滤波器组：28个梅尔滤波器模拟255Hz-10.3kHz的对数间隔滤波器
    3. 时间积分：8ms短期时间积分 (hop_length=176 ≈ 8ms at 22.05kHz)
    4. 特征提取：基于感知模型的9个声学特征
    """
    try:
        # === 步骤1: 基础音频处理 ===
        # 加载音频文件
        y, sample_rate = librosa.load(audio_file, sr=sr, res_type='kaiser_fast')
        
        # 计算短时傅里叶变换 (STFT)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # === 步骤2: 耳蜗滤波器组 ===
        # 使用28个梅尔滤波器模拟耳蜗滤波器
        # 频率范围：250Hz-12.5kHz (接近255Hz-10.3kHz)
        n_mels = 28  # 28个对数间隔滤波器
        mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, 
                                         fmin=250, fmax=12500)
        
        # 应用梅尔滤波器组，模拟耳蜗的频率分解
        mel_spectrogram = np.dot(mel_filters, magnitude)
        mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        
        # === 步骤3: 时间积分 ===
        # hop_length=176 在22.05kHz采样率下 ≈ 8ms
        # 这实现了8ms的短期时间积分
        
        # === 步骤4: 基于感知模型的特征提取 ===
        # 特征1: 频谱质心 (Spectral Centroid) - 亮度测量
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                             hop_length=hop_length, n_fft=n_fft)
        
        # 特征2: 频谱带宽 (Spectral Bandwidth) - 加权距离
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                               hop_length=hop_length, n_fft=n_fft)
        
        # 特征3: 频谱平坦度 (Spectral Flatness) - 几何平均/算术平均
        spectral_flatness = []
        for i in range(mel_spectrogram.shape[1]):
            frame = mel_spectrogram[:, i]
            # 避免零值
            frame = frame + 1e-10
            geometric_mean = np.exp(np.mean(np.log(frame)))
            arithmetic_mean = np.mean(frame)
            flatness = geometric_mean / arithmetic_mean
            spectral_flatness.append(flatness)
        spectral_flatness = np.array(spectral_flatness)
        
        # 特征4: 频谱不规则性 (Spectral Irregularity) - 相邻信道差异
        spectral_irregularity = []
        for i in range(mel_spectrogram.shape[1]):
            frame = mel_spectrogram[:, i]
            # 计算相邻信道之间的差异
            diff = np.diff(frame)
            irregularity = np.sum(np.abs(diff))
            spectral_irregularity.append(irregularity)
        spectral_irregularity = np.array(spectral_irregularity)
        
        # 特征5: 基频 (F0) - 使用YIN算法估计音高
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'), 
                                                    sr=sr, hop_length=hop_length)
        
        # 特征6: 谐波性 (Harmonicity) - 谐波与噪声的比率
        harmonic, percussive = librosa.effects.hpss(y)
        harmonicity = []
        for i in range(0, len(y), hop_length):
            end_idx = min(i + hop_length, len(y))
            h_seg = harmonic[i:end_idx]
            p_seg = percussive[i:end_idx]
            if len(h_seg) > 0 and len(p_seg) > 0:
                h_energy = np.sum(h_seg ** 2)
                p_energy = np.sum(p_seg ** 2)
                if p_energy > 0:
                    harm_ratio = h_energy / (h_energy + p_energy)
                else:
                    harm_ratio = 1.0
                harmonicity.append(harm_ratio)
        harmonicity = np.array(harmonicity)
        
        # 特征7: 时间调制 (Temporal Modulation) - 2-32Hz时间变化
        # 使用MFCC的delta特征模拟时间调制
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc)
        temporal_modulation = np.mean(np.abs(delta_mfcc), axis=0)
        
        # 特征8: 频谱调制 (Spectral Modulation) - 0.25-8 cycles/octave
        # 使用MFCC模拟频谱调制
        spectral_modulation = np.std(mfcc, axis=1)
        
        # 特征9: 响度 (Loudness) - 基于临界频段的响度
        # 使用梅尔频谱的响度感知模型
        loudness = []
        for i in range(mel_spectrogram.shape[1]):
            frame = mel_spectrogram[:, i]
            # 应用A-weighting类似的感知权重
            # 简化版本：使用对数压缩
            loudness_val = np.sum(20 * np.log10(frame + 1e-10))
            loudness.append(loudness_val)
        loudness = np.array(loudness)
        
        # === 步骤5: 统一时间维度并组织特征 ===
        # 确定统一的时间维度T（使用mel_spectrogram的时间维度作为参考）
        T = mel_spectrogram.shape[1]
        
        # 将所有特征统一到相同的时间维度T
        def ensure_time_dimension(feature_array, target_length):
            """确保特征数组具有目标时间长度"""
            if len(feature_array) == target_length:
                return feature_array
            elif len(feature_array) > target_length:
                return feature_array[:target_length]
            else:
                # 如果长度不足，用最后一个值填充
                padding = np.full(target_length - len(feature_array), feature_array[-1])
                return np.concatenate([feature_array, padding])
        
        # 统一所有特征的时间维度
        spectral_centroid_unified = ensure_time_dimension(spectral_centroid.flatten(), T)
        spectral_bandwidth_unified = ensure_time_dimension(spectral_bandwidth.flatten(), T)
        spectral_flatness_unified = ensure_time_dimension(spectral_flatness, T)
        spectral_irregularity_unified = ensure_time_dimension(spectral_irregularity, T)
        f0_unified = ensure_time_dimension(f0, T)
        harmonicity_unified = ensure_time_dimension(harmonicity, T)
        temporal_modulation_unified = ensure_time_dimension(temporal_modulation, T)
        # spectral_modulation 是频率维度，需要重复到时间维度
        spectral_modulation_unified = np.tile(spectral_modulation, (T, 1)).T
        loudness_unified = ensure_time_dimension(loudness, T)
        
        # 组织特征为 9 × T 的矩阵
        feature_matrix = np.array([
            spectral_centroid_unified,      # 特征1
            spectral_bandwidth_unified,     # 特征2
            spectral_flatness_unified,      # 特征3
            spectral_irregularity_unified,  # 特征4
            f0_unified,                     # 特征5
            harmonicity_unified,            # 特征6
            temporal_modulation_unified,    # 特征7
            spectral_modulation_unified.mean(axis=0),  # 特征8 (取平均)
            loudness_unified                # 特征9
        ])
        
        # === 步骤6: Z分数归一化 ===
        # 对每个特征进行Z分数归一化：(x - μ) / σ
        def z_score_normalize(feature_vector):
            """对特征向量进行Z分数归一化"""
            # 处理NaN值
            valid_mask = ~np.isnan(feature_vector)
            if np.sum(valid_mask) == 0:
                return feature_vector  # 如果全是NaN，返回原值
            
            valid_values = feature_vector[valid_mask]
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            
            if std_val == 0:
                # 如果标准差为0，返回零向量
                normalized = np.zeros_like(feature_vector)
            else:
                normalized = (feature_vector - mean_val) / std_val
            
            return normalized
        
        # 对每个特征进行Z分数归一化
        feature_matrix_normalized = np.array([
            z_score_normalize(feature_matrix[0]),  # 特征1
            z_score_normalize(feature_matrix[1]),  # 特征2
            z_score_normalize(feature_matrix[2]),  # 特征3
            z_score_normalize(feature_matrix[3]),  # 特征4
            z_score_normalize(feature_matrix[4]),  # 特征5
            z_score_normalize(feature_matrix[5]),  # 特征6
            z_score_normalize(feature_matrix[6]),  # 特征7
            z_score_normalize(feature_matrix[7]),  # 特征8
            z_score_normalize(feature_matrix[8])   # 特征9
        ])
        
        # 组织特征
        features = {
            'feature_matrix': feature_matrix,  # 9 × T 矩阵 (原始值)
            'feature_matrix_normalized': feature_matrix_normalized,  # 9 × T 矩阵 (Z分数归一化)
            'individual_features': {
                'spectral_centroid': spectral_centroid_unified,
                'spectral_bandwidth': spectral_bandwidth_unified,
                'spectral_flatness': spectral_flatness_unified,
                'spectral_irregularity': spectral_irregularity_unified,
                'f0': f0_unified,
                'harmonicity': harmonicity_unified,
                'temporal_modulation': temporal_modulation_unified,
                'spectral_modulation': spectral_modulation_unified,
                'loudness': loudness_unified
            },
            'individual_features_normalized': {
                'spectral_centroid': feature_matrix_normalized[0],
                'spectral_bandwidth': feature_matrix_normalized[1],
                'spectral_flatness': feature_matrix_normalized[2],
                'spectral_irregularity': feature_matrix_normalized[3],
                'f0': feature_matrix_normalized[4],
                'harmonicity': feature_matrix_normalized[5],
                'temporal_modulation': feature_matrix_normalized[6],
                'spectral_modulation': feature_matrix_normalized[7],
                'loudness': feature_matrix_normalized[8]
            },
            'mel_spectrogram': mel_spectrogram_db,
            'duration': len(y) / sr,
            'sample_rate': sr,
            'time_frames': T
        }
        
        return features, None
        
    except Exception as e:
        return None, str(e)

def get_audio_files(folder_path):
    """获取文件夹中的所有音频文件"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    audio_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(folder_path, file))
    
    return audio_files

def get_subfolders(parent_folder):
    """获取父文件夹下的所有子文件夹"""
    subfolders = []
    if os.path.exists(parent_folder) and os.path.isdir(parent_folder):
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
    return sorted(subfolders)

def select_categories(parent_folder, selected_categories=None):
    """选择要处理的类别文件夹"""
    subfolders = get_subfolders(parent_folder)
    
    if not subfolders:
        print("在指定路径下没有找到子文件夹")
        return []
    
    if selected_categories is None:
        # 如果没有指定类别，显示所有可用的类别
        print(f"\n在 {parent_folder} 下找到以下类别文件夹:")
        for i, folder in enumerate(subfolders, 1):
            print(f"{i}. {folder}")
        
        print("\n请选择要处理的类别:")
        print("输入方式:")
        print("- 输入数字，如: 1,3,5 (处理第1,3,5个类别)")
        print("- 输入类别名称，如: Music,Speech (处理Music和Speech类别)")
        print("- 输入 'all' 处理所有类别")
        
        choice = input("请选择: ").strip()
        
        if choice.lower() == 'all':
            return subfolders
        
        # 尝试解析数字选择
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [subfolders[i] for i in indices if 0 <= i < len(subfolders)]
            return selected
        except ValueError:
            # 如果不是数字，尝试按名称选择
            selected = [name.strip() for name in choice.split(',')]
            return [name for name in selected if name in subfolders]
    else:
        # 使用指定的类别
        return [cat for cat in selected_categories if cat in subfolders]

def save_features(features_data, output_path, folder_name):
    """保存提取的特征"""
    # 保存为pickle文件
    pickle_path = os.path.join(output_path, f"{folder_name}_acoustic_features.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(features_data, f)
    
    # 保存为CSV文件（只保存统计信息）
    csv_data = []
    for file_path, features, error in features_data:
        if features is not None:
            # 计算每个特征的统计信息
            feature_matrix = features['feature_matrix']  # 9 × T 矩阵 (原始值)
            feature_matrix_norm = features['feature_matrix_normalized']  # 9 × T 矩阵 (归一化)
            individual_features = features['individual_features']
            individual_features_norm = features['individual_features_normalized']
            
            csv_data.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'duration': features['duration'],
                'sample_rate': features['sample_rate'],
                'time_frames': features['time_frames'],
                'feature_matrix_shape': feature_matrix.shape,  # (9, T)
                'feature_matrix_normalized_shape': feature_matrix_norm.shape,  # (9, T)
                # 原始特征统计
                'spectral_centroid_mean': np.mean(individual_features['spectral_centroid']),
                'spectral_centroid_std': np.std(individual_features['spectral_centroid']),
                'spectral_bandwidth_mean': np.mean(individual_features['spectral_bandwidth']),
                'spectral_bandwidth_std': np.std(individual_features['spectral_bandwidth']),
                'spectral_flatness_mean': np.mean(individual_features['spectral_flatness']),
                'spectral_flatness_std': np.std(individual_features['spectral_flatness']),
                'spectral_irregularity_mean': np.mean(individual_features['spectral_irregularity']),
                'spectral_irregularity_std': np.std(individual_features['spectral_irregularity']),
                'f0_mean': np.nanmean(individual_features['f0']),
                'f0_std': np.nanstd(individual_features['f0']),
                'harmonicity_mean': np.mean(individual_features['harmonicity']),
                'harmonicity_std': np.std(individual_features['harmonicity']),
                'temporal_modulation_mean': np.mean(individual_features['temporal_modulation']),
                'temporal_modulation_std': np.std(individual_features['temporal_modulation']),
                'spectral_modulation_mean': np.mean(individual_features['spectral_modulation']),
                'spectral_modulation_std': np.std(individual_features['spectral_modulation']),
                'loudness_mean': np.mean(individual_features['loudness']),
                'loudness_std': np.std(individual_features['loudness']),
                # 归一化特征统计 (应该接近0和1)
                'spectral_centroid_norm_mean': np.mean(individual_features_norm['spectral_centroid']),
                'spectral_centroid_norm_std': np.std(individual_features_norm['spectral_centroid']),
                'spectral_bandwidth_norm_mean': np.mean(individual_features_norm['spectral_bandwidth']),
                'spectral_bandwidth_norm_std': np.std(individual_features_norm['spectral_bandwidth']),
                'spectral_flatness_norm_mean': np.mean(individual_features_norm['spectral_flatness']),
                'spectral_flatness_norm_std': np.std(individual_features_norm['spectral_flatness']),
                'spectral_irregularity_norm_mean': np.mean(individual_features_norm['spectral_irregularity']),
                'spectral_irregularity_norm_std': np.std(individual_features_norm['spectral_irregularity']),
                'f0_norm_mean': np.nanmean(individual_features_norm['f0']),
                'f0_norm_std': np.nanstd(individual_features_norm['f0']),
                'harmonicity_norm_mean': np.mean(individual_features_norm['harmonicity']),
                'harmonicity_norm_std': np.std(individual_features_norm['harmonicity']),
                'temporal_modulation_norm_mean': np.mean(individual_features_norm['temporal_modulation']),
                'temporal_modulation_norm_std': np.std(individual_features_norm['temporal_modulation']),
                'spectral_modulation_norm_mean': np.mean(individual_features_norm['spectral_modulation']),
                'spectral_modulation_norm_std': np.std(individual_features_norm['spectral_modulation']),
                'loudness_norm_mean': np.mean(individual_features_norm['loudness']),
                'loudness_norm_std': np.std(individual_features_norm['loudness']),
                'mel_spectrogram_shape': features['mel_spectrogram'].shape,
                'error': None
            })
        else:
            csv_data.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'duration': None,
                'sample_rate': None,
                'spectral_centroid_mean': None,
                'spectral_centroid_std': None,
                'spectral_bandwidth_mean': None,
                'spectral_bandwidth_std': None,
                'spectral_flatness_mean': None,
                'spectral_flatness_std': None,
                'spectral_irregularity_mean': None,
                'spectral_irregularity_std': None,
                'f0_mean': None,
                'f0_std': None,
                'harmonicity_mean': None,
                'harmonicity_std': None,
                'temporal_modulation_mean': None,
                'temporal_modulation_std': None,
                'spectral_modulation_mean': None,
                'spectral_modulation_std': None,
                'loudness_mean': None,
                'loudness_std': None,
                'mel_spectrogram_shape': None,
                'error': error
            })
    
    csv_path = os.path.join(output_path, f"{folder_name}_acoustic_features_summary.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return pickle_path, csv_path

def find_nearest_neighbor(query_features, target_features_list, target_files_list):
    """
    在目标类别中找到查询样本的最近邻
    
    参数:
    query_features: 查询样本的特征矩阵 (9, T)
    target_features_list: 目标类别的特征矩阵列表
    target_files_list: 对应的文件路径列表
    
    返回:
    nearest_file: 最近邻文件路径
    similarity: 相似度分数
    """
    if not target_features_list:
        return None, 0
    
    # 将查询特征和目标特征展平为一维向量
    query_flat = query_features.flatten()
    
    # 处理查询特征中的NaN值
    query_flat = np.nan_to_num(query_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    similarities = []
    for target_features in target_features_list:
        target_flat = target_features.flatten()
        
        # 处理目标特征中的NaN值
        target_flat = np.nan_to_num(target_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算余弦相似度
        similarity = cosine_similarity([query_flat], [target_flat])[0][0]
        similarities.append(similarity)
    
    # 找到最大相似度对应的文件
    max_similarity_idx = np.argmax(similarities)
    nearest_file = target_files_list[max_similarity_idx]
    max_similarity = similarities[max_similarity_idx]
    
    return nearest_file, max_similarity

def copy_files_to_output(query_file, nearest_file, output_folder, category1, category2):
    """
    将查询文件和最近邻文件复制到输出文件夹
    
    参数:
    query_file: 查询文件路径
    nearest_file: 最近邻文件路径
    output_folder: 输出文件夹路径
    category1: 第一个类别名称
    category2: 第二个类别名称
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 复制查询文件
    query_filename = f"query_{category1}_{os.path.basename(query_file)}"
    query_dest = os.path.join(output_folder, query_filename)
    shutil.copy2(query_file, query_dest)
    
    # 复制最近邻文件
    nearest_filename = f"nearest_{category2}_{os.path.basename(nearest_file)}"
    nearest_dest = os.path.join(output_folder, nearest_filename)
    shutil.copy2(nearest_file, nearest_dest)
    
    return query_dest, nearest_dest

def aggregate_features(feature_matrix, method='statistical'):
    """
    对特征矩阵进行聚合，减少维度
    
    参数:
    feature_matrix: 9 × T 的特征矩阵
    method: 聚合方法 ('statistical', 'pca', 'mean')
    
    返回:
    aggregated_features: 聚合后的特征向量
    """
    # 首先处理NaN值
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == 'statistical':
        # 统计聚合：每个特征计算均值、标准差、最大值、最小值
        aggregated = []
        for i in range(feature_matrix.shape[0]):  # 对每个特征
            feature_vector = feature_matrix[i, :]
            # 计算统计量
            mean_val = np.mean(feature_vector)
            std_val = np.std(feature_vector)
            max_val = np.max(feature_vector)
            min_val = np.min(feature_vector)
            # 添加统计量
            aggregated.extend([mean_val, std_val, max_val, min_val])
        return np.array(aggregated)  # 9 × 4 = 36维
    
    elif method == 'mean':
        # 简单均值聚合：每个特征取均值
        return np.mean(feature_matrix, axis=1)  # 9维
    
    elif method == 'pca':
        # PCA降维：保留95%的方差
        pca = PCA(n_components=0.95)
        # 转置以便PCA处理
        reduced = pca.fit_transform(feature_matrix.T)
        print(f"PCA降维: {feature_matrix.shape} → {reduced.shape} (保留95%方差)")
        return reduced.flatten()  # 展平为一维
    
    else:
        # 默认返回原始特征
        return feature_matrix.flatten()

def optimize_feature_comparison(features_list, files_list, method='statistical'):
    """
    优化的特征比较，使用聚合后的特征
    
    参数:
    features_list: 特征矩阵列表
    files_list: 对应的文件路径列表
    method: 聚合方法
    
    返回:
    optimized_features: 优化后的特征列表
    """
    optimized_features = []
    for features in features_list:
        aggregated = aggregate_features(features, method)
        optimized_features.append(aggregated)
    return optimized_features

def main(parent_folder, selected_categories=None, output_path=None, show_examples=True, 
         feature_optimization='statistical', query_file_id=None):
    """
    音频特征提取主函数 - 最近邻搜索版本
    
    参数:
    parent_folder: 包含类别子文件夹的父文件夹路径
    selected_categories: 要处理的类别列表，必须包含2个类别
    output_path: 输出路径，默认为当前目录下的"extracted_features"文件夹
    show_examples: 是否显示示例特征信息，默认为True
    feature_optimization: 特征优化方法 ('statistical', 'pca', 'mean', 'none')
    query_file_id: 指定查询文件的ID（如'2QcOD8uCu0E'），如果为None则随机选择
    """
    print("=== 音频特征提取工具 - 最近邻搜索 ===")
    print(f"特征优化方法: {feature_optimization}")
    
    # 验证父文件夹路径
    if not validate_folder(parent_folder):
        return None
    
    # 验证类别数量
    if not selected_categories or len(selected_categories) != 2:
        print("错误：必须选择恰好2个类别进行最近邻搜索")
        return None
    
    category1, category2 = selected_categories
    print(f"\n选择的类别: {category1} (查询类别) 和 {category2} (目标类别)")
    
    # 设置输出目录
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "extracted_features")
    os.makedirs(output_path, exist_ok=True)
    
    # 处理第一个类别（查询类别）
    print(f"\n=== 处理查询类别: {category1} ===")
    category1_path = os.path.join(parent_folder, category1)
    category1_files = get_audio_files(category1_path)
    
    if not category1_files:
        print(f"在 {category1} 文件夹中没有找到音频文件")
        return None
    
    print(f"找到 {len(category1_files)} 个音频文件")
    
    # 选择查询文件
    if query_file_id is not None:
        # 查找指定的查询文件
        query_file = None
        for file_path in category1_files:
            if query_file_id in os.path.basename(file_path):
                query_file = file_path
                break
        
        if query_file is None:
            print(f"错误：在 {category1} 类别中找不到包含 '{query_file_id}' 的文件")
            print(f"可用的文件示例: {[os.path.basename(f) for f in category1_files[:5]]}")
            return None
        
        print(f"指定选择的查询文件: {os.path.basename(query_file)}")
    else:
        # 随机选择一个查询文件
        query_file = random.choice(category1_files)
        print(f"随机选择的查询文件: {os.path.basename(query_file)}")
    
    # 提取查询文件的特征
    print("提取查询文件特征...")
    query_features, query_error = extract_audio_features(query_file)
    if query_features is None:
        print(f"查询文件特征提取失败: {query_error}")
        return None
    
    print(f"查询文件特征提取成功，特征矩阵形状: {query_features['feature_matrix_normalized'].shape}")
    
    # 优化查询特征
    if feature_optimization != 'none':
        query_optimized = aggregate_features(query_features['feature_matrix_normalized'], feature_optimization)
        print(f"查询特征优化后维度: {len(query_optimized)} (原始: {query_features['feature_matrix_normalized'].size})")
    else:
        query_optimized = query_features['feature_matrix_normalized']
    
    # 处理第二个类别（目标类别）
    print(f"\n=== 处理目标类别: {category2} ===")
    category2_path = os.path.join(parent_folder, category2)
    category2_files = get_audio_files(category2_path)
    
    if not category2_files:
        print(f"在 {category2} 文件夹中没有找到音频文件")
        return None
    
    print(f"找到 {len(category2_files)} 个音频文件")
    
    # 提取目标类别的所有特征
    print("提取目标类别所有文件特征...")
    target_features_list = []
    target_files_list = []
    successful_extractions = 0
    failed_extractions = 0
    
    for audio_file in tqdm.tqdm(category2_files, desc=f"提取{category2}特征"):
        features, error = extract_audio_features(audio_file)
        if features is not None:
            target_features_list.append(features['feature_matrix_normalized'])
            target_files_list.append(audio_file)
            successful_extractions += 1
        else:
            failed_extractions += 1
            print(f"提取失败: {os.path.basename(audio_file)} - {error}")
    
    print(f"目标类别特征提取完成: 成功 {successful_extractions} 个，失败 {failed_extractions} 个")
    
    if not target_features_list:
        print("目标类别没有成功提取任何特征")
        return None
    
    # 优化目标特征
    if feature_optimization != 'none':
        print(f"优化目标特征...")
        target_optimized_list = optimize_feature_comparison(target_features_list, target_files_list, feature_optimization)
        print(f"目标特征优化后维度: {len(target_optimized_list[0])} (原始: {target_features_list[0].size})")
    else:
        target_optimized_list = target_features_list
    
    # 寻找最近邻
    print(f"\n=== 寻找最近邻 ===")
    nearest_file, similarity = find_nearest_neighbor(
        query_optimized, 
        target_optimized_list, 
        target_files_list
    )
    
    print(f"找到最近邻文件: {os.path.basename(nearest_file)}")
    print(f"相似度分数: {similarity:.4f}")
    
    # 复制文件到输出文件夹
    print(f"\n=== 复制文件到输出文件夹 ===")
    nearest_neighbor_folder = os.path.join(output_path, "nearest_neighbor_results")
    query_dest, nearest_dest = copy_files_to_output(
        query_file, nearest_file, nearest_neighbor_folder, category1, category2
    )
    
    print(f"查询文件已复制到: {query_dest}")
    print(f"最近邻文件已复制到: {nearest_dest}")
    
    # 保存结果信息
    result_info = {
        'query_file': query_file,
        'query_category': category1,
        'nearest_file': nearest_file,
        'nearest_category': category2,
        'similarity': similarity,
        'query_dest': query_dest,
        'nearest_dest': nearest_dest,
        'target_files_processed': successful_extractions,
        'feature_optimization': feature_optimization,
        'original_feature_dim': query_features['feature_matrix_normalized'].size,
        'optimized_feature_dim': len(query_optimized) if feature_optimization != 'none' else query_features['feature_matrix_normalized'].size
    }
    
    # 保存结果到文件
    result_file = os.path.join(nearest_neighbor_folder, "nearest_neighbor_info.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=== 最近邻搜索结果 ===\n")
        f.write(f"查询类别: {category1}\n")
        f.write(f"查询文件: {os.path.basename(query_file)}\n")
        f.write(f"目标类别: {category2}\n")
        f.write(f"最近邻文件: {os.path.basename(nearest_file)}\n")
        f.write(f"相似度分数: {similarity:.4f}\n")
        f.write(f"目标类别处理文件数: {successful_extractions}\n")
        f.write(f"特征优化方法: {feature_optimization}\n")
        f.write(f"原始特征维度: {query_features['feature_matrix_normalized'].size}\n")
        f.write(f"优化后特征维度: {len(query_optimized) if feature_optimization != 'none' else query_features['feature_matrix_normalized'].size}\n")
        f.write(f"查询文件路径: {query_dest}\n")
        f.write(f"最近邻文件路径: {nearest_dest}\n")
    
    print(f"结果信息已保存到: {result_file}")
    
    # 显示特征信息
    if show_examples:
        print(f"\n=== 特征信息对比 ===")
        print(f"查询文件 ({os.path.basename(query_file)}):")
        print(f"  时长: {query_features['duration']:.2f} 秒")
        print(f"  时间帧数: {query_features['time_frames']}")
        print(f"  特征矩阵形状: {query_features['feature_matrix_normalized'].shape}")
        
        # 获取最近邻文件的特征信息
        nearest_features, _ = extract_audio_features(nearest_file)
        if nearest_features is not None:
            print(f"最近邻文件 ({os.path.basename(nearest_file)}):")
            print(f"  时长: {nearest_features['duration']:.2f} 秒")
            print(f"  时间帧数: {nearest_features['time_frames']}")
            print(f"  特征矩阵形状: {nearest_features['feature_matrix_normalized'].shape}")
    
    return result_info

if __name__ == "__main__":
    # 在这里设置要处理的文件夹路径
    parent_folder = r"D:\D\research\audioset下载\classified_audio"  # 修改为你的音频文件夹路径

    # 指定要处理的类别（必须恰好2个类别：查询类别和目标类别）
    selected_categories = ["Canidae, dogs, wolves", "Chicken, rooster"]  # 修改为你想要的类别

    # 可选：设置输出路径（如果不设置，会使用默认路径）
    output_path = None  # 或者指定具体路径，如: r"D:\output\features"
    
    # 特征优化方法选择：
    #     # 'statistical': 统计聚合 (推9维)
    # 'pca': PCA降维 - 保ne则随机选择留9荐) - 每个特征计算均值、标准差、最大值、最小值 (36维)
    #     # 'mean': 简单均值 - 每个特征取均值 (5%方差 维度可变()
    #     #     # 'none': 不优化 - 使用原始特征 (11,250维 for 10s用频)
    feature_optimization = 'statistical'  # 推荐使统计聚合

    query_file_id = "dZQ1gziNOc"  # 修改为你想要的查询文件ID，或设为None随机选择
    
    # 运行主函数J
    result = main(parent_folder, selected_categories, output_path, 
                 feature_optimization=feature_optimization, query_file_id=query_file_id)
    
    if result is None:
        print("最近邻搜索失败！")
    else:
        print(f"最近邻搜索完成！")
        print(f"查询文件: {os.path.basename(result['query_file'])}")
        print(f"最近邻文件: {os.path.basename(result['nearest_file'])}")
        print(f"相似度: {result['similarity']:.4f}")
        print(f"特征优化: {result['feature_optimization']}")
        print(f"维度压缩: {result['original_feature_dim']} → {result['optimized_feature_dim']}")
        print(f"结果保存在: {result['query_dest']} 和 {result['nearest_dest']}")
