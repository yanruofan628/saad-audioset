import pandas as pd
import os
import numpy as np
import librosa
import tqdm
from pathlib import Path
from datetime import datetime

def extract_audio_features(audio_file, sr=16000, hop_length=128, n_fft=2048):
    """
    按照声学感知模型提取9个特征
    
    处理流程：
    1. 基础音频处理：加载音频，计算STFT
    2. 耳蜗滤波器组：28个梅尔滤波器模拟255Hz-10.3kHz的对数间隔滤波器
    3. 时间积分：8ms短期时间积分 (hop_length=128 ≈ 8ms at 16kHz)
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
        # hop_length=128 在16kHz采样率下 ≈ 8ms
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

def load_existing_features(output_path):
    """检查已存在的特征数据"""
    # 扫描输出目录，查找已处理的类别
    existing_categories = []
    if os.path.exists(output_path):
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isdir(item_path) and not item.endswith('_summary'):
                # 检查是否包含numpy文件
                numpy_files = [f for f in os.listdir(item_path) if f.endswith('.npy')]
                if numpy_files:
                    existing_categories.append(item)
    
    if existing_categories:
        print(f"发现已存在的特征数据，包含 {len(existing_categories)} 个已处理的类别: {', '.join(existing_categories)}")
        return {'categories': existing_categories}
    return None

def save_single_file_features(features, audio_file, output_path, category_name):
    """保存单个文件的特征数据为numpy格式"""
    try:
        if features is None:
            return False
            
        # 创建类别目录
        category_dir = os.path.join(output_path, category_name)
        os.makedirs(category_dir, exist_ok=True)
        
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        # 保存归一化特征矩阵 (9 × T)
        feature_matrix_path = os.path.join(category_dir, f"{file_name}_features.npy")
        np.save(feature_matrix_path, features['feature_matrix_normalized'])
        
        # 保存原始特征矩阵 (9 × T)
        feature_matrix_raw_path = os.path.join(category_dir, f"{file_name}_features_raw.npy")
        np.save(feature_matrix_raw_path, features['feature_matrix'])
        
        # 保存梅尔频谱图
        mel_spectrogram_path = os.path.join(category_dir, f"{file_name}_mel_spectrogram.npy")
        np.save(mel_spectrogram_path, features['mel_spectrogram'])
        
        # 保存元数据为numpy数组
        metadata = np.array([
            features['duration'],
            features['sample_rate'],
            features['time_frames'],
            features['feature_matrix_normalized'].shape[0],  # 特征数量
            features['feature_matrix_normalized'].shape[1],  # 时间帧数
            features['mel_spectrogram'].shape[0],  # 梅尔滤波器数量
            features['mel_spectrogram'].shape[1]   # 梅尔频谱时间帧数
        ])
        
        metadata_path = os.path.join(category_dir, f"{file_name}_metadata.npy")
        np.save(metadata_path, metadata)
        
        return True
    except Exception as e:
        print(f"✗ 保存 {audio_file} 特征失败: {e}")
        return False

def save_category_metadata(all_features_dict, output_path, category_name):
    """保存类别元数据为numpy格式"""
    try:
        # 保存当前类别的特征摘要（CSV格式，便于查看）
        if category_name in all_features_dict['categories']:
            category_data = all_features_dict['categories'][category_name]
            if category_data['statistics']['successful_extractions'] > 0:
                summary_data = []
                for file_info in category_data['features']:
                    if file_info['features'] is not None:
                        features = file_info['features']
                        
                        # 计算variation_sum（L2距离的累积和）
                        feature_matrix = features['feature_matrix_normalized']
                        variation_sum = 0
                        if feature_matrix.shape[1] > 1:  # 确保有多个时间帧
                            for t in range(feature_matrix.shape[1] - 1):
                                diff = feature_matrix[:, t+1] - feature_matrix[:, t]
                                variation_sum += np.linalg.norm(diff)
                        
                        summary_data.append({
                            'audio_file': file_info['file_name'],
                            'duration': features['duration'],
                            'time_frames': features['time_frames'],
                            'variation_sum': variation_sum,
                            'spectral_centroid_mean': np.mean(features['individual_features']['spectral_centroid']),
                            'spectral_bandwidth_mean': np.mean(features['individual_features']['spectral_bandwidth']),
                            'spectral_flatness_mean': np.mean(features['individual_features']['spectral_flatness']),
                            'spectral_irregularity_mean': np.mean(features['individual_features']['spectral_irregularity']),
                            'f0_mean': np.nanmean(features['individual_features']['f0']),
                            'harmonicity_mean': np.mean(features['individual_features']['harmonicity']),
                            'temporal_modulation_mean': np.mean(features['individual_features']['temporal_modulation']),
                            'loudness_mean': np.mean(features['individual_features']['loudness'])
                        })
                
                # 保存为CSV文件
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(output_path, f"{category_name}_features_summary.csv")
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
                print(f"✓ {category_name} 特征摘要已保存为CSV: {summary_path}")
        
        return True
    except Exception as e:
        print(f"✗ 保存 {category_name} 类别元数据失败: {e}")
        return False

def select_categories_to_process(all_categories, selected_categories=None):
    """
    选择要处理的类别
    
    参数:
    all_categories: 所有可用的类别列表
    selected_categories: 要处理的类别列表，None表示处理所有类别
    
    返回:
    categories_to_process: 要处理的类别列表
    """
    if selected_categories is None:
        return all_categories
    
    # 检查选择的类别是否都存在
    available_categories = set(all_categories)
    selected_set = set(selected_categories)
    
    # 找到存在的类别
    valid_categories = list(selected_set.intersection(available_categories))
    invalid_categories = list(selected_set - available_categories)
    
    if invalid_categories:
        print(f"警告: 以下类别不存在，将被忽略: {', '.join(invalid_categories)}")
    
    if not valid_categories:
        print("错误: 没有找到有效的类别")
        return []
    
    return valid_categories

def extract_all_categories_features(parent_folder, output_path=None, max_files_per_category=None, 
                                  resume=True, skip_existing=True, selected_categories=None):
    """
    提取所有类别文件夹的特征并保存为字典（支持断点续传 + 类别选择）
    
    参数:
    parent_folder: 包含类别子文件夹的父文件夹路径
    output_path: 输出路径，默认为当前目录下的"all_categories_features"文件夹
    max_files_per_category: 每个类别最大处理文件数，None表示处理所有文件
    resume: 是否启用断点续传，True表示跳过已处理的类别
    skip_existing: 是否跳过已存在的类别，True表示不重新处理已完成的类别
    selected_categories: 要处理的类别列表，None表示处理所有类别
    
    返回:
    all_features_dict: 包含所有类别特征的字典
    """
    print("=== 批量提取所有类别特征（支持断点续传 + 类别选择）===")
    
    # 设置输出目录
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "all_categories_features")
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有类别文件夹
    all_categories = get_subfolders(parent_folder)
    if not all_categories:
        print("在指定路径下没有找到子文件夹")
        return None
    
    print(f"找到 {len(all_categories)} 个类别文件夹: {', '.join(all_categories)}")
    
    # 选择要处理的类别
    categories_to_process = select_categories_to_process(all_categories, selected_categories)
    if not categories_to_process:
        print("没有选择任何类别进行处理")
        return None
    
    print(f"选择的类别 ({len(categories_to_process)} 个): {', '.join(categories_to_process)}")
    
    # 尝试加载已存在的特征数据
    all_features_dict = None
    if resume:
        all_features_dict = load_existing_features(output_path)
    
    # 如果没有已存在的数据，创建新的字典
    if all_features_dict is None:
        all_features_dict = {
            'metadata': {
                'extraction_time': datetime.now().isoformat(),
                'parent_folder': parent_folder,
                'total_categories': len(all_categories),
                'all_categories': all_categories,
                'selected_categories': categories_to_process,
                'resume_mode': True
            },
            'categories': {}
        }
    else:
        # 更新元数据
        all_features_dict['metadata']['parent_folder'] = parent_folder
        all_features_dict['metadata']['total_categories'] = len(all_categories)
        all_features_dict['metadata']['all_categories'] = all_categories
        all_features_dict['metadata']['selected_categories'] = categories_to_process
        all_features_dict['metadata']['resume_mode'] = True
        all_features_dict['metadata']['last_update'] = datetime.now().isoformat()
    
    # 确定需要处理的类别（考虑断点续传）
    if skip_existing and resume:
        existing_categories = set(all_features_dict['categories'].keys())
        categories_to_process = [cat for cat in categories_to_process if cat not in existing_categories]
        skipped_categories = [cat for cat in categories_to_process if cat in existing_categories]
        
        if skipped_categories:
            print(f"跳过已处理的类别 ({len(skipped_categories)} 个): {', '.join(skipped_categories)}")
        
        if not categories_to_process:
            print("所有选择的类别都已处理完成！")
            return all_features_dict
    else:
        skipped_categories = []
    
    print(f"需要处理的类别 ({len(categories_to_process)} 个): {', '.join(categories_to_process)}")
    
    # 统计信息
    total_files_processed = 0
    total_successful = 0
    total_failed = 0
    
    # 处理每个类别
    for i, category in enumerate(categories_to_process, 1):
        category_path = os.path.join(parent_folder, category)
        print(f"\n=== 处理类别 {i}/{len(categories_to_process)}: {category} ===")
        print(f"路径: {category_path}")
        
        # 获取该类别下的音频文件
        audio_files = get_audio_files(category_path)
        if not audio_files:
            print(f"在 {category} 文件夹中没有找到音频文件")
            all_features_dict['categories'][category] = {
                'files': [],
                'features': [],
                'statistics': {
                    'total_files': 0,
                    'successful_extractions': 0,
                    'failed_extractions': 0
                }
            }
            # 保存空类别
            save_incremental_features(all_features_dict, output_path, category)
            continue
        
        # 限制每个类别的文件数量
        if max_files_per_category is not None and len(audio_files) > max_files_per_category:
            audio_files = audio_files[:max_files_per_category]
            print(f"限制处理文件数量: {max_files_per_category} (总共 {len(get_audio_files(category_path))} 个)")
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        # 提取特征
        category_features = []
        successful_extractions = 0
        failed_extractions = 0
        
        print(f"开始提取 {category} 的特征...")
        for audio_file in tqdm.tqdm(audio_files, desc=f"提取{category}特征"):
            features, error = extract_audio_features(audio_file)
            
            file_info = {
                'file_path': audio_file,
                'file_name': os.path.basename(audio_file),
                'features': features,
                'error': error
            }
            
            category_features.append(file_info)
            
            if features is not None:
                successful_extractions += 1
                # 立即保存单个文件的特征
                save_success = save_single_file_features(features, audio_file, output_path, category)
                if save_success:
                    print(f"✓ 已保存: {os.path.basename(audio_file)}")
                else:
                    print(f"✗ 保存失败: {os.path.basename(audio_file)}")
            else:
                failed_extractions += 1
                print(f"提取失败: {os.path.basename(audio_file)} - {error}")
        
        # 存储该类别的特征
        all_features_dict['categories'][category] = {
            'files': [f['file_name'] for f in category_features],
            'features': category_features,
            'statistics': {
                'total_files': len(audio_files),
                'successful_extractions': successful_extractions,
                'failed_extractions': failed_extractions
            }
        }
        
        total_files_processed += len(audio_files)
        total_successful += successful_extractions
        total_failed += failed_extractions
        
        print(f"{category} 处理完成: 成功 {successful_extractions} 个，失败 {failed_extractions} 个")
        
        # 保存当前类别的元数据
        save_success = save_category_metadata(all_features_dict, output_path, category)
        if not save_success:
            print(f"警告: {category} 类别元数据保存失败，但处理继续")
        
        # 显示进度
        remaining_categories = len(categories_to_process) - i
        print(f"进度: {i}/{len(categories_to_process)} 类别完成，剩余 {remaining_categories} 个类别")
    
    # 更新总体统计信息
    all_features_dict['metadata']['total_files_processed'] = total_files_processed
    all_features_dict['metadata']['total_successful'] = total_successful
    all_features_dict['metadata']['total_failed'] = total_failed
    all_features_dict['metadata']['completion_time'] = datetime.now().isoformat()
    
    # 最终保存（所有类别都已完成）
    print(f"\n=== 最终保存 ===")
    final_save_success = save_category_metadata(all_features_dict, output_path, "FINAL")
    if final_save_success:
        print("✓ 所有元数据已最终保存")
    else:
        print("✗ 最终保存失败，但已保存的numpy文件仍然可用")
    
    # 输出总体统计信息
    print(f"\n=== 所有类别处理完成 ===")
    print(f"总共处理: {total_files_processed} 个文件")
    print(f"成功提取: {total_successful} 个文件")
    print(f"失败: {total_failed} 个文件")
    print(f"处理的类别: {', '.join(categories)}")
    print(f"特征数据保存位置: {output_path}")
    
    # 显示每个类别的文件信息
    print(f"\n各类别处理结果:")
    for category, data in all_features_dict['categories'].items():
        stats = data['statistics']
        print(f"  {category}: 成功 {stats['successful_extractions']} 个，失败 {stats['failed_extractions']} 个")
    
    return all_features_dict

if __name__ == "__main__":
    # 在这里设置要处理的文件夹路径
    parent_folder = r"D:\D\research\audioset下载\classified_audio"  # 修改为你的音频文件夹路径
    
    # 可选：设置输出路径（如果不设置，会使用默认路径）
    output_path = r"D:\D\research\audioset下载\特征"  # 或者指定具体路径，如: r"D:\output\all_features"
    
    # 可选：限制每个类别处理的最大文件数（用于测试，None表示处理所有文件）
    max_files_per_category = None  # 或者设置为数字，如: 10
    
    # 断点续传设置
    resume = True  # 是否启用断点续传
    skip_existing = True  # 是否跳过已存在的类别
    
    # 类别选择设置
    # 方法1: 处理所有类别
    # selected_categories = None  # None表示处理所有类别
    
    # 方法2: 选择特定类别（取消注释下面的行）
    # selected_categories = ['Alarm','Bass drum','Telephone bell ringing',
    #                        'Computer keyboard','Crying, sobbing','Fireworks','Canidae, dogs, wolves',
    #                        'Chicken, rooster','Helicopter','Male speech, man speaking','Female speech, woman speaking',
    #                        'Child speech, kid speaking','Funny music','Sad music',
    #                        'Ambulance (Siren)']  # 只处理这些类别
    # selected_categories=['Pulse','Whack, thwack','Crumpling, crinkling']
    selected_categories = ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry',
                           'Computer keyboard', 'Helicopter', 'Chicken, rooster',
                           'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',
                           'Bass drum', 'Funny music', 'Sad music',
                           'Pulse', 'Whack, thwack', 'Crumpling, crinkling',
                           'Tick','Subway, metro, underground','toilet flush']
    # 方法3: 选择多个类别（取消注释下面的行）
    # selected_categories = [
    #     "Canidae, dogs, wolves", 
    #     "Chicken, rooster", 
    #     "Crying, sobbing",
    #     "Children playing"
    # ]qi
    
    # 运行主函数
    result = extract_all_categories_features(
        parent_folder=parent_folder, 
        output_path=output_path, 
        max_files_per_category=max_files_per_category,
        resume=resume,
        skip_existing=skip_existing,
        selected_categories=selected_categories
    )
    
    if result is None:
        print("特征提取失败！")
    else:
        print(f"特征提取完成！")
        print(f"总共处理 {result['metadata']['total_successful']} 个文件")
        print(f"涉及 {result['metadata']['total_categories']} 个类别")
        print(f"选择的类别: {', '.join(result['metadata']['selected_categories'])}")
        print(f"断点续传模式: {'启用' if result['metadata'].get('resume_mode', False) else '禁用'}")
