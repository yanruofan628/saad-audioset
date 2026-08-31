import pandas as pd
import os
import pickle
import numpy as np
import librosa
import tqdm
import shutil
import soundfile as sf
PATH='D:/D/research/audioset下载/archive'
OUTPUT_PATH='D:/D/research/audioset下载/classified_audio'

metadata = pd.read_csv(os.path.join(PATH, 'valid.csv'))
label_df = pd.read_csv(os.path.join(PATH, 'class_labels_indices.csv'))
# 处理多个标签的情况
metadata['positive_labels'] = metadata['positive_labels'].apply(lambda x: x.split(','))
# 展开多标签数据，每个标签创建一行
expanded_data = []
for _, row in metadata.iterrows():
    for label in row['positive_labels']:
        expanded_data.append({
            'YTID': row['YTID'],
            'positive_labels': label,
            'start_seconds': row['start_seconds'],
            'end_seconds': row['end_seconds']
        })

expanded_df = pd.DataFrame(expanded_data)
temp_df = pd.merge(expanded_df, label_df, left_on='positive_labels', right_on='mid')


def preprocess_audio(input_file, output_file, target_sr=16000):
    """
    音频预处理函数：
    1. 降采样到 16,000 Hz (适配Whisper和Wav2Vec)
    2. 处理立体声文件（只保留左声道）
    3. 基于最响1%的RMS能量进行归一化
    """
    try:
        # 读取音频文件
        audio_data, original_sr = sf.read(input_file)
        
        # 处理立体声文件（只保留左声道）
        is_stereo = False
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data[:, 0]  # 只保留左声道
            is_stereo = True
            preprocessing_stats['stereo_files'] += 1
        
        # 降采样到目标采样率
        is_resampled = False
        if original_sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
            is_resampled = True
            preprocessing_stats['resampled_files'] += 1
        
        # 基于最响1%的RMS能量进行归一化
        # 检查音频数据是否有效
        if len(audio_data) == 0:
            return False, "音频数据为空"
        
        # 找到最响1%的样本
        sorted_indices = np.argsort(np.abs(audio_data))[::-1]
        top_1_percent_count = max(1, int(len(audio_data) * 0.01))
        top_1_percent_samples = audio_data[sorted_indices[:top_1_percent_count]]
        
        # 计算最响1%的RMS能量，避免空数组警告
        if len(top_1_percent_samples) > 0:
            top_1_percent_rms = np.sqrt(np.mean(top_1_percent_samples**2))
        else:
            top_1_percent_rms = 0
        
        # 归一化到目标RMS水平（这里使用0.1作为目标RMS）
        target_rms = 0.1
        is_normalized = False
        if top_1_percent_rms > 0:
            normalization_factor = target_rms / top_1_percent_rms
            audio_data = audio_data * normalization_factor
            is_normalized = True
            preprocessing_stats['normalized_files'] += 1
        
        # 确保音频数据在有效范围内
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # 保存处理后的音频文件
        sf.write(output_file, audio_data, target_sr, subtype='PCM_16')
        
        # 构建处理信息
        processing_info = []
        if is_stereo:
            processing_info.append("立体声→单声道")
        if is_resampled:
            processing_info.append(f"{original_sr}→{target_sr}Hz")
        if is_normalized:
            processing_info.append("RMS归一化")
        
        return True, f"处理成功: {', '.join(processing_info)}"
        
    except Exception as e:
        return False, f"处理失败: {str(e)}"


def create_class_folder(class_name, output_path):
    """创建类别文件夹，处理特殊字符"""
    # 替换文件夹名称中的特殊字符
    safe_class_name = class_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    class_folder = os.path.join(output_path, safe_class_name)
    os.makedirs(class_folder, exist_ok=True)
    return class_folder

def copy_audio_to_class(source_file, class_folder, ytid):
    """将音频文件预处理后复制到对应的类别文件夹"""
    if os.path.exists(source_file):
        # 创建目标文件名
        dest_file = os.path.join(class_folder, f"{ytid}.wav")
        try:
            # 先进行音频预处理，然后保存
            success, message = preprocess_audio(source_file, dest_file)
            if success:
                return True
            else:
                print(f"预处理失败 {source_file}: {message}")
                return False
        except Exception as e:
            print(f"处理文件失败 {source_file}: {e}")
            return False
    else:
        print(f"源文件不存在: {source_file}")
        return False

# 创建输出根目录
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 统计信息
class_counts = {}
successful_copies = 0
failed_copies = 0
multi_label_count = 0  # 多标签音频文件数量
preprocessing_stats = {
    'stereo_files': 0,  # 立体声文件数量
    'resampled_files': 0,  # 需要重采样的文件数量
    'normalized_files': 0  # 归一化的文件数量
}

print("开始按类别分类音频文件...")

# 按YTID分组处理，避免重复处理同一个文件
processed_files = set()

# 获取所有唯一的YTID
unique_ytids = temp_df['YTID'].unique()

print(f"总共需要处理 {len(unique_ytids)} 个唯一的音频文件")

# 遍历所有唯一的音频文件
for ytid in tqdm.tqdm(unique_ytids, desc="分类音频文件"):
    # 构建源文件路径
    source_file = os.path.join(PATH, 'valid_wav', ytid + '.wav')
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"源文件不存在: {source_file}")
        failed_copies += 1
        continue
    
    # 获取这个文件的所有标签
    file_labels = temp_df[temp_df['YTID'] == ytid]
    
    # 统计多标签情况
    if len(file_labels) > 1:
        multi_label_count += 1
    
    # 为每个标签复制文件到对应的类别文件夹（多标签文件需要复制多次）
    for _, label_row in file_labels.iterrows():
        class_label = label_row['display_name']
        
        # 创建类别文件夹
        class_folder = create_class_folder(class_label, OUTPUT_PATH)
        
        # 复制文件到对应类别文件夹
        if copy_audio_to_class(source_file, class_folder, ytid):
            successful_copies += 1
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        else:
            failed_copies += 1

# 输出统计信息
print(f"\n分类完成！")
print(f"成功复制: {successful_copies} 个文件")
print(f"失败: {failed_copies} 个文件")
print(f"总共处理了 {len(class_counts)} 个不同的类别")
print(f"多标签音频文件数量: {multi_label_count} 个")

print(f"\n音频预处理统计:")
print(f"立体声文件处理: {preprocessing_stats['stereo_files']} 个")
print(f"重采样文件: {preprocessing_stats['resampled_files']} 个")
print(f"RMS归一化文件: {preprocessing_stats['normalized_files']} 个")

# 显示每个类别的文件数量
print("\n各类别文件数量:")
for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {count} 个文件")

print(f"\n分类后的音频文件保存在: {OUTPUT_PATH}")
