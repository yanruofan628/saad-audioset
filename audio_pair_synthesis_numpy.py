import pandas as pd
import os
import numpy as np
import librosa
import tqdm
import random
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import soundfile as sf
from scipy.io import wavfile
import shutil

def load_features_data_numpy(features_path):
    """从各个子类别目录加载numpy格式的特征数据"""
    features_data = {'categories': {}}
    
    # 获取所有子类别
    category_mapping = get_category_mapping()
    all_subcategories = []
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append(sub_cat)
    
    print(f"开始从各子类别目录加载numpy格式特征数据...")
    print(f"特征数据路径: {features_path}")
    print(f"要处理的类别: {all_subcategories}")
    
    for sub_cat in all_subcategories:
        # 构建类别目录路径
        category_dir = os.path.join(features_path, sub_cat)
        print(f"\n检查类别目录: {category_dir}")
        
        if os.path.exists(category_dir):
            try:
                # 列出目录中的所有文件
                all_files = os.listdir(category_dir)
                print(f"  目录中的文件: {all_files}")
                
                # 扫描类别目录中的所有numpy文件
                numpy_files = []
                for file_name in all_files:
                    if file_name.endswith('_features.npy'):
                        print(f"  找到特征文件: {file_name}")
                        # 提取音频文件名（去掉_features.npy后缀）
                        audio_file_name = file_name.replace('_features.npy', '')
                        # 查找对应的音频文件（支持多种格式）
                        # 音频文件在classified_audio目录中，特征文件在特征目录中
                        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
                        audio_file_path = None
                        
                        # 首先在特征目录中查找
                        for ext in audio_extensions:
                            potential_audio_path = os.path.join(category_dir, audio_file_name + ext)
                            if os.path.exists(potential_audio_path):
                                audio_file_path = potential_audio_path
                                print(f"  找到对应音频文件: {audio_file_name + ext}")
                                break
                        
                        # 如果在特征目录中没找到，尝试在classified_audio目录中查找
                        if audio_file_path is None:
                            # 构建classified_audio目录路径
                            classified_audio_dir = os.path.join(os.path.dirname(features_path), "classified_audio", sub_cat)
                            for ext in audio_extensions:
                                potential_audio_path = os.path.join(classified_audio_dir, audio_file_name + ext)
                                if os.path.exists(potential_audio_path):
                                    audio_file_path = potential_audio_path
                                    print(f"  在classified_audio目录找到对应音频文件: {audio_file_name + ext}")
                                    break
                        
                        if audio_file_path:
                            # 加载特征矩阵
                            feature_matrix_path = os.path.join(category_dir, file_name)
                            feature_matrix = np.load(feature_matrix_path)
                            
                            # 加载元数据
                            metadata_path = os.path.join(category_dir, audio_file_name + '_metadata.npy')
                            metadata = None
                            if os.path.exists(metadata_path):
                                metadata = np.load(metadata_path)
                            
                            # 计算variation_sum（L2距离的累积和）
                            variation_sum = 0
                            if feature_matrix.shape[1] > 1:  # 确保有多个时间帧
                                for t in range(feature_matrix.shape[1] - 1):
                                    diff = feature_matrix[:, t+1] - feature_matrix[:, t]
                                    variation_sum += np.linalg.norm(diff)
                            
                            # 构建特征信息
                            feature_info = {
                                'file_name': os.path.basename(audio_file_path),
                                'file_path': audio_file_path,
                                'features': {
                                    'feature_matrix_normalized': feature_matrix,
                                    'variation_sum': variation_sum,
                                    'metadata': metadata
                                }
                            }
                            numpy_files.append(feature_info)
                            print(f"  ✓ 成功加载: {audio_file_name}")
                        else:
                            print(f"  ✗ 未找到对应音频文件: {audio_file_name}")
                
                features_data['categories'][sub_cat] = {
                    'features': numpy_files,
                    'count': len(numpy_files)
                }
                print(f"成功加载 {sub_cat} 的特征数据，包含 {len(numpy_files)} 个文件")
                
            except Exception as e:
                print(f"加载 {sub_cat} 特征数据失败: {e}")
                features_data['categories'][sub_cat] = {'features': [], 'count': 0}
        else:
            print(f"未找到 {sub_cat} 的目录: {category_dir}")
            features_data['categories'][sub_cat] = {'features': [], 'count': 0}
    
    total_categories = len([cat for cat in features_data['categories'].values() if cat['count'] > 0])
    print(f"numpy格式特征数据加载完成，包含 {total_categories} 个有效类别")
    return features_data

def get_category_mapping():
    """获取类别映射关系"""
    category_mapping = {
        '高生态效度': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
        '低生态效度': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
        'music': ['Bass drum', 'Funny music', 'Sad music'],
        '未知声源': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
    }
    return category_mapping

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
    
    elif method == 'pca':
        # PCA降维：保留95%的方差
        pca = PCA(n_components=0.95)
        # 转置以便PCA处理
        reduced = pca.fit_transform(feature_matrix.T)
        return reduced.flatten()  # 展平为一维
    
    elif method == 'mean':
        # 简单均值聚合：每个特征取均值
        return np.mean(feature_matrix, axis=1)  # 9维
    
    else:
        # 默认返回统计聚合特征
        return aggregate_features(feature_matrix, 'statistical')

def find_nearest_neighbor(query_features, target_features_list, target_files_list):
    """
    在目标类别中找到查询样本的最近邻
    
    参数:
    query_features: 查询样本的特征向量
    target_features_list: 目标类别的特征向量列表
    target_files_list: 对应的文件路径列表
    
    返回:
    nearest_file: 最近邻文件路径
    similarity: 相似度分数
    """
    if not target_features_list:
        return None, 0
    
    # 处理查询特征中的NaN值
    query_features = np.nan_to_num(query_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    similarities = []
    for target_features in target_features_list:
        target_features = np.nan_to_num(target_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算余弦相似度
        similarity = cosine_similarity([query_features], [target_features])[0][0]
        similarities.append(similarity)
    
    # 找到最大相似度对应的文件
    max_similarity_idx = np.argmax(similarities)
    nearest_file = target_files_list[max_similarity_idx]
    max_similarity = similarities[max_similarity_idx]
    
    return nearest_file, max_similarity

def load_audio_file(audio_path, target_sr=16000):
    """加载音频文件，重采样到16kHz，确保为单声道"""
    try:
        # 使用librosa加载，重采样到16kHz
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)  # 重采样到16kHz
        
        # 检查是否为单声道
        if len(y.shape) > 1:
            print(f"警告: {os.path.basename(audio_path)} 是多声道，已转换为单声道")
            # 如果还是多声道，取第一个声道
            y = y[:, 0] if len(y.shape) > 1 else y
        
        return y, sr
    except Exception as e:
        print(f"加载音频文件失败 {audio_path}: {e}")
        return None, None

def create_stereo_audio(left_audio, right_audio, output_path, sr=16000, target_duration=10.0):
    """
    创建左右声道分离的双声道音频，确保两个声道都是10秒
    
    参数:
    left_audio: 左声道音频数据（第一个音频，单声道）
    right_audio: 右声道音频数据（第二个音频，单声道）
    output_path: 输出文件路径
    sr: 采样率
    target_duration: 目标时长（秒），默认10秒
    """
    try:
        # 确保输入是单声道
        if len(left_audio.shape) > 1:
            left_audio = left_audio.flatten()
        if len(right_audio.shape) > 1:
            right_audio = right_audio.flatten()
        
        # 计算目标样本数（10秒）
        target_samples = int(target_duration * sr)
        
        # 处理左声道
        left_len = len(left_audio)
        if left_len > target_samples:
            # 如果超过10秒，截断到10秒
            left_audio = left_audio[:target_samples]
            print(f"  左声道截断到 {target_duration} 秒 ({target_samples} 个样本)")
        elif left_len < target_samples:
            # 如果不足10秒，用零填充
            left_audio = np.pad(left_audio, (0, target_samples - left_len), mode='constant')
            print(f"  左声道填充到 {target_duration} 秒 ({target_samples} 个样本)")
        else:
            print(f"  左声道已经是 {target_duration} 秒 ({target_samples} 个样本)")
        
        # 处理右声道
        right_len = len(right_audio)
        if right_len > target_samples:
            # 如果超过10秒，截断到10秒
            right_audio = right_audio[:target_samples]
            print(f"  右声道截断到 {target_duration} 秒 ({target_samples} 个样本)")
        elif right_len < target_samples:
            # 如果不足10秒，用零填充
            right_audio = np.pad(right_audio, (0, target_samples - right_len), mode='constant')
            print(f"  右声道填充到 {target_duration} 秒 ({target_samples} 个样本)")
        else:
            print(f"  右声道已经是 {target_duration} 秒 ({target_samples} 个样本)")
        
        # 创建左右声道分离的立体声数组
        # 左声道：第一个音频，右声道：第二个音频
        stereo_audio = np.column_stack((left_audio, right_audio))
        
        # 保存为WAV文件
        sf.write(output_path, stereo_audio, sr)
        
        return True
    except Exception as e:
        print(f"创建立体声音频失败: {e}")
        return False

def get_available_audio_files(features_data, category_name, used_files):
    """获取指定类别中未使用的音频文件"""
    if category_name not in features_data['categories']:
        return []
    
    category_data = features_data['categories'][category_name]
    available_files = []
    
    for file_info in category_data['features']:
        # 检查文件是否存在且未被使用
        if (file_info['file_path'] and 
            os.path.exists(file_info['file_path']) and 
            file_info['file_name'] not in used_files):
            available_files.append(file_info)
    
    return available_files


def nearest_neighbor_matching(features_data, category_mapping, used_files, output_dir):
    """
    最近邻匹配阶段：每个子类别随机取15条音频，分别与15个子类别的最相似音频匹配
    总共15×15=225个配对
    """
    print("=== 开始最近邻匹配阶段 ===")

    # 获取所有子类别
    all_subcategories = []
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append((main_cat, sub_cat))

    print(f"总共有 {len(all_subcategories)} 个子类别")
    pairs_created = []
    nn_used_audios = {}  # 记录最近邻匹配阶段使用的音频，按类别分组

    # 为每个子类别创建音频队列（排除已使用的）
    audio_queues = {}
    for main_cat, sub_cat in all_subcategories:
        available_files = get_available_audio_files(features_data, sub_cat, used_files)
        audio_queues[(main_cat, sub_cat)] = available_files.copy()
        print(f"{main_cat} ({sub_cat}): {len(available_files)} 个可用音频")

    # 为每个类别进行匹配
    for left_idx, (left_main_cat, left_sub_cat) in enumerate(all_subcategories):
        print(f"\n--- {left_main_cat} ({left_sub_cat}) 进行匹配 ---")
        
        # 检查左声道类别是否有足够的音频（需要15条）
        if len(audio_queues[(left_main_cat, left_sub_cat)]) < 15:
            print(f"错误: {left_main_cat} ({left_sub_cat}) 可用音频不足15条，只有 {len(audio_queues[(left_main_cat, left_sub_cat)])} 条")
            continue

        # 从左声道类别随机取出15条音频
        left_audios = random.sample(audio_queues[(left_main_cat, left_sub_cat)], 15)
        # 从队列中移除这15条音频
        for left_audio in left_audios:
            audio_queues[(left_main_cat, left_sub_cat)].remove(left_audio)
            used_files.add(left_audio['file_name'])
        print(f"  取出15条左声道音频: {[audio['file_name'] for audio in left_audios]}")
        
        # 记录左声道音频用于后续补充
        if left_sub_cat not in nn_used_audios:
            nn_used_audios[left_sub_cat] = []
        nn_used_audios[left_sub_cat].extend(left_audios)

        # 为每条左声道音频在对应的右声道类别中寻找最相似的音频
        for right_idx, (right_main_cat, right_sub_cat) in enumerate(all_subcategories):
            left_file_info = left_audios[right_idx]
            print(f"    第{right_idx + 1}条音频 {left_file_info['file_name']} 与 {right_main_cat} ({right_sub_cat}) 匹配")

            # 检查右声道类别是否有可用音频
            if not audio_queues[(right_main_cat, right_sub_cat)]:
                print(f"    ✗ 没有找到可用的右声道音频")
                continue

            # 获取左声道音频的特征矩阵
            left_feature_matrix = left_file_info['features']['feature_matrix_normalized']
            
            # 聚合左声道特征
            left_aggregated = aggregate_features(left_feature_matrix, 'statistical')

            # 在右声道类别中寻找最相似的音频
            best_right_file = None
            best_similarity = -1
            
            for right_file_info in audio_queues[(right_main_cat, right_sub_cat)]:
                # 获取右声道音频的特征矩阵
                right_feature_matrix = right_file_info['features']['feature_matrix_normalized']
                
                # 聚合右声道特征
                right_aggregated = aggregate_features(right_feature_matrix, 'statistical')
                
                # 计算余弦相似度
                similarity = cosine_similarity([left_aggregated], [right_aggregated])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_right_file = right_file_info

            if best_right_file is None:
                print(f"    ✗ 没有找到可用的右声道音频")
                continue

            # 从队列中移除选中的右声道音频
            audio_queues[(right_main_cat, right_sub_cat)].remove(best_right_file)
            if best_right_file['file_name'] in used_files:
                print(f"    警告: 右声道音频 {best_right_file['file_name']} 已经在used_files中")
            used_files.add(best_right_file['file_name'])
            print(f"    添加右声道音频到used_files: {best_right_file['file_name']}")

            # 创建配对记录
            pair_info = {
                'type': 'nearest_neighbor',
                'left_category': left_main_cat,
                'left_subcategory': left_sub_cat,
                'left_file': left_file_info['file_name'],
                'right_category': right_main_cat,
                'right_subcategory': right_sub_cat,
                'right_file': best_right_file['file_name'],
                'similarity': best_similarity
            }
            pairs_created.append(pair_info)

            # 立即合成音频
            left_audio_path = left_file_info['file_path']
            right_audio_path = best_right_file['file_path']

            # 加载音频（统一16kHz采样率）
            left_audio, sr = load_audio_file(left_audio_path)
            right_audio, sr = load_audio_file(right_audio_path)

            if left_audio is not None and right_audio is not None:
                # 创建输出文件名
                left_category = left_sub_cat
                right_category = right_sub_cat
                left_audio_id = os.path.splitext(os.path.basename(left_file_info['file_path']))[0]
                right_audio_id = os.path.splitext(os.path.basename(best_right_file['file_path']))[0]

                # 直接使用原始类别名称和音频ID
                output_filename = f"nn_{left_category}_{left_audio_id}+{right_category}_{right_audio_id}.wav"
                output_path = os.path.join(output_dir, "nearest_neighbor_pairs", output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 合成双声道音频（确保10秒长度）
                success = create_stereo_audio(left_audio, right_audio, output_path, sr, target_duration=10.0)
                if success:
                    print(f"    ✓ 创建配对: {left_main_cat}({left_sub_cat}) ↔ {right_main_cat}({right_sub_cat}) (相似度: {best_similarity:.4f})")
                else:
                    print(f"    ✗ 音频合成失败")
            else:
                print(f"    ✗ 音频加载失败")

    print(f"最近邻匹配完成: {len(pairs_created)} 个配对")
    print(f"最近邻匹配后已使用音频文件: {len(used_files)} 个")
    return pairs_created, nn_used_audios


def random_pairing(features_data, category_mapping, used_files, output_dir, nn_used_audios=None):
    """
    随机配对阶段：每个子类别随机取15条音频，分别与15个子类别的随机音频匹配
    总共15×15=225个配对
    如果某个类别可用音频不足15条，从最近邻匹配阶段已用过的音频中随机补充
    """
    print(f"\n=== 开始随机配对阶段 ===")

    # 获取所有子类别
    all_subcategories = []
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append((main_cat, sub_cat))

    print(f"总共有 {len(all_subcategories)} 个子类别")
    pairs_created = []

    # 为每个子类别创建音频队列（排除已使用的）
    audio_queues = {}
    for main_cat, sub_cat in all_subcategories:
        available_files = get_available_audio_files(features_data, sub_cat, used_files)
        audio_queues[(main_cat, sub_cat)] = available_files.copy()
        print(f"{main_cat} ({sub_cat}): {len(available_files)} 个可用音频")

    # 为每个类别进行匹配
    for left_idx, (left_main_cat, left_sub_cat) in enumerate(all_subcategories):
        print(f"\n--- {left_main_cat} ({left_sub_cat}) 进行匹配 ---")
        
        # 检查左声道类别是否有足够的音频（需要15条）
        current_available = len(audio_queues[(left_main_cat, left_sub_cat)])
        if current_available < 15:
            print(f"警告: {left_main_cat} ({left_sub_cat}) 可用音频不足15条，只有 {current_available} 条")
            
            # 尝试从最近邻匹配阶段已用过的音频中补充
            if nn_used_audios and left_sub_cat in nn_used_audios:
                available_nn_audios = nn_used_audios[left_sub_cat]  # 最近邻匹配阶段用过的音频
                needed = 15 - current_available
                if len(available_nn_audios) >= needed:
                    # 随机选择需要的音频进行补充
                    selected_audios = random.sample(available_nn_audios, needed)
                    audio_queues[(left_main_cat, left_sub_cat)].extend(selected_audios)
                    print(f"  从最近邻匹配阶段补充了 {needed} 条音频")
                else:
                    print(f"  最近邻匹配阶段可补充的音频不足，只有 {len(available_nn_audios)} 条")
                    if len(available_nn_audios) > 0:
                        audio_queues[(left_main_cat, left_sub_cat)].extend(available_nn_audios)
                        print(f"  补充了 {len(available_nn_audios)} 条音频")
            
            # 再次检查是否足够15条
            if len(audio_queues[(left_main_cat, left_sub_cat)]) < 15:
                print(f"错误: {left_main_cat} ({left_sub_cat}) 补充后仍不足15条，只有 {len(audio_queues[(left_main_cat, left_sub_cat)])} 条")
                continue

        # 从左声道类别随机取出15条音频
        left_audios = random.sample(audio_queues[(left_main_cat, left_sub_cat)], 15)
        # 从队列中移除这15条音频
        for left_audio in left_audios:
            audio_queues[(left_main_cat, left_sub_cat)].remove(left_audio)
            used_files.add(left_audio['file_name'])
        print(f"  取出15条左声道音频: {[audio['file_name'] for audio in left_audios]}")

        # 为每条左声道音频在对应的右声道类别中随机选择一个音频
        for right_idx, (right_main_cat, right_sub_cat) in enumerate(all_subcategories):
            left_file_info = left_audios[right_idx]
            print(f"    第{right_idx + 1}条音频 {left_file_info['file_name']} 与 {right_main_cat} ({right_sub_cat}) 匹配")

            # 检查右声道类别是否有可用音频
            if not audio_queues[(right_main_cat, right_sub_cat)]:
                print(f"    警告: {right_main_cat} ({right_sub_cat}) 没有可用的右声道音频")
                
                # 尝试从最近邻匹配阶段已用过的音频中补充
                if nn_used_audios and right_sub_cat in nn_used_audios:
                    available_nn_audios = nn_used_audios[right_sub_cat]  # 最近邻匹配阶段用过的音频
                    if available_nn_audios:
                        # 随机选择一个音频进行补充
                        selected_audio = random.choice(available_nn_audios)
                        audio_queues[(right_main_cat, right_sub_cat)].append(selected_audio)
                        print(f"    从最近邻匹配阶段补充了1条右声道音频: {selected_audio['file_name']}")
                    else:
                        print(f"    最近邻匹配阶段也没有可用的 {right_sub_cat} 音频")
                        continue
                else:
                    print(f"    无法补充 {right_sub_cat} 的右声道音频")
                    continue

            # 随机选择一个右声道音频
            right_file_info = random.choice(audio_queues[(right_main_cat, right_sub_cat)])
            
            # 从队列中移除选中的右声道音频
            audio_queues[(right_main_cat, right_sub_cat)].remove(right_file_info)
            if right_file_info['file_name'] in used_files:
                print(f"    警告: 右声道音频 {right_file_info['file_name']} 已经在used_files中")
            used_files.add(right_file_info['file_name'])
            print(f"    添加右声道音频到used_files: {right_file_info['file_name']}")

            # 创建配对记录
            pair_info = {
                'type': 'random',
                'left_category': left_main_cat,
                'left_subcategory': left_sub_cat,
                'left_file': left_file_info['file_name'],
                'right_category': right_main_cat,
                'right_subcategory': right_sub_cat,
                'right_file': right_file_info['file_name'],
                'similarity': None
            }
            pairs_created.append(pair_info)

            # 立即合成音频
            left_audio_path = left_file_info['file_path']
            right_audio_path = right_file_info['file_path']

            # 加载音频（统一16kHz采样率）
            left_audio, sr = load_audio_file(left_audio_path)
            right_audio, sr = load_audio_file(right_audio_path)

            if left_audio is not None and right_audio is not None:
                # 创建输出文件名
                left_category = left_sub_cat
                right_category = right_sub_cat
                left_audio_id = os.path.splitext(os.path.basename(left_file_info['file_path']))[0]
                right_audio_id = os.path.splitext(os.path.basename(right_file_info['file_path']))[0]

                # 直接使用原始类别名称和音频ID
                output_filename = f"rn_{left_category}_{left_audio_id}+{right_category}_{right_audio_id}.wav"
                output_path = os.path.join(output_dir, "random_pairs", output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 合成双声道音频（确保10秒长度）
                success = create_stereo_audio(left_audio, right_audio, output_path, sr, target_duration=10.0)
                if success:
                    print(f"    ✓ 创建配对: {left_main_cat}({left_sub_cat}) ↔ {right_main_cat}({right_sub_cat})")
                else:
                    print(f"    ✗ 音频合成失败")
            else:
                print(f"    ✗ 音频加载失败")

    print(f"随机配对完成: {len(pairs_created)} 个配对")
    print(f"随机配对后已使用音频文件: {len(used_files)} 个")
    return pairs_created

def save_pairing_results(pairs_created, output_dir):
    """保存配对结果"""
    # 保存为JSON文件
    json_path = os.path.join(output_dir, "pairing_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_created, f, ensure_ascii=False, indent=2)
    
    # 保存为CSV文件
    csv_data = []
    for pair in pairs_created:
        csv_data.append({
            'type': pair['type'],
            'round': pair.get('round', ''),
            'left_category': pair['left_category'],
            'left_subcategory': pair['left_subcategory'],
            'left_file': pair['left_file'],
            'right_category': pair['right_category'],
            'right_subcategory': pair['right_subcategory'],
            'right_file': pair['right_file'],
            'similarity': pair.get('similarity', '')
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "pairing_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n配对结果已保存:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")

def main(features_path, output_dir):
    """
    主函数：音频配对合成（numpy格式）
    
    参数:
    features_path: 特征数据路径（包含numpy文件的目录）
    output_dir: 输出目录
    """
    print("=== 音频配对合成工具（numpy格式）===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载特征数据
    features_data = load_features_data_numpy(features_path)
    if features_data is None:
        print("无法加载特征数据，程序退出")
        return
    
    # 获取类别映射
    category_mapping = get_category_mapping()
    print(f"类别映射: {category_mapping}")
    
    # 初始化已使用文件集合
    used_files = set()
    
    # 阶段1: 最近邻匹配
    nn_pairs, nn_used_audios = nearest_neighbor_matching(features_data, category_mapping, used_files, output_dir)

    # 阶段2: 随机配对
    random_pairs = random_pairing(features_data, category_mapping, used_files, output_dir, nn_used_audios)
    
    # 合并所有配对结果
    all_pairs = nn_pairs + random_pairs
    
    # 保存结果
    save_pairing_results(all_pairs, output_dir)
    
    # 输出统计信息
    print(f"\n=== 配对完成统计 ===")
    print(f"最近邻配对: {len(nn_pairs)} 对")
    print(f"随机配对: {len(random_pairs)} 对")
    print(f"总配对: {len(all_pairs)} 对")
    print(f"已使用音频文件: {len(used_files)} 个")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 设置路径
    features_path = r"D:\D\research\audioset下载\特征"  # 包含各子类别numpy文件的路径
    output_dir = r"D:\D\research\audioset下载\音频配对合成_numpy"  # 输出目录
    
    # 运行主函数
    main(features_path, output_dir)
