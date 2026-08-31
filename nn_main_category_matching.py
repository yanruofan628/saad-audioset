import pandas as pd
import os
import numpy as np
import librosa
import random
from pathlib import Path
import json
from datetime import datetime
import soundfile as sf
from itertools import combinations_with_replacement
from sklearn.metrics.pairwise import cosine_similarity


def get_category_mapping():
    """获取类别映射关系（仅4个大类）"""
    category_mapping = {
        '高生态效度': ['Telephone bell ringing', 'Baby cry, infant cry'],
        '低生态效度': ['Computer keyboard', 'Helicopter'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking'],
        'music': ['Bass drum', 'Sad music'],
    }
    return category_mapping


def load_features_data_numpy_5s(features_path, clap_audio_path):
    """从各个子类别目录加载numpy格式的特征数据，适配5秒音频"""
    features_data = {'categories': {}}
    
    # 获取所有子类别
    category_mapping = get_category_mapping()
    all_subcategories = []
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append((main_cat, sub_cat))
    
    print(f"开始从各子类别目录加载numpy格式特征数据（5秒音频版本）...")
    print(f"特征数据路径: {features_path}")
    print(f"5秒音频路径: {clap_audio_path}")
    
    for main_cat, sub_cat in all_subcategories:
        # 构建类别目录路径
        category_dir = os.path.join(features_path, sub_cat)
        clap_category_dir = os.path.join(clap_audio_path, sub_cat)
        print(f"\n检查类别目录: {category_dir}")
        print(f"检查5秒音频目录: {clap_category_dir}")
        
        if os.path.exists(category_dir) and os.path.exists(clap_category_dir):
            try:
                # 扫描类别目录中的所有numpy文件
                numpy_files = []
                all_files = os.listdir(category_dir)
                clap_files = os.listdir(clap_category_dir)
                
                for file_name in all_files:
                    if file_name.endswith('_features.npy'):
                        # 提取原始音频文件名（去掉_features.npy后缀）
                        original_audio_name = file_name.replace('_features.npy', '')
                        
                        # 查找对应的5秒音频文件
                        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
                        audio_files = []
                        
                        for ext in audio_extensions:
                            # 查找 _1 结尾的文件（前5秒）
                            audio_1_name = original_audio_name + '_1' + ext
                            audio_1_path = os.path.join(clap_category_dir, audio_1_name)
                            if os.path.exists(audio_1_path):
                                audio_files.append({
                                    'name': audio_1_name,
                                    'path': audio_1_path,
                                    'type': 'first_half'
                                })
                            
                            # 查找 _2 结尾的文件（后5秒）
                            audio_2_name = original_audio_name + '_2' + ext
                            audio_2_path = os.path.join(clap_category_dir, audio_2_name)
                            if os.path.exists(audio_2_path):
                                audio_files.append({
                                    'name': audio_2_name,
                                    'path': audio_2_path,
                                    'type': 'second_half'
                                })
                        
                        if audio_files:
                            # 加载特征矩阵
                            feature_matrix_path = os.path.join(category_dir, file_name)
                            feature_matrix = np.load(feature_matrix_path)
                            
                            # 为每个5秒音频文件创建特征信息
                            for audio_info in audio_files:
                                # 根据音频类型选择特征
                                if audio_info['type'] == 'first_half':
                                    # 前5秒，取特征矩阵的前一半
                                    half_point = feature_matrix.shape[1] // 2
                                    audio_feature_matrix = feature_matrix[:, :half_point]
                                else:  # second_half
                                    # 后5秒，取特征矩阵的后一半
                                    half_point = feature_matrix.shape[1] // 2
                                    audio_feature_matrix = feature_matrix[:, half_point:]
                                
                                # 构建特征信息
                                feature_info = {
                                    'file_name': audio_info['name'],
                                    'file_path': audio_info['path'],
                                    'original_audio_name': original_audio_name,
                                    'audio_type': audio_info['type'],
                                    'main_category': main_cat,
                                    'sub_category': sub_cat,
                                    'features': {
                                        'feature_matrix_normalized': audio_feature_matrix
                                    }
                                }
                                numpy_files.append(feature_info)
                
                features_data['categories'][sub_cat] = {
                    'features': numpy_files,
                    'count': len(numpy_files),
                    'main_category': main_cat
                }
                print(f"成功加载 {sub_cat} 的特征数据，包含 {len(numpy_files)} 个5秒音频文件")
                
            except Exception as e:
                print(f"加载 {sub_cat} 特征数据失败: {e}")
                features_data['categories'][sub_cat] = {'features': [], 'count': 0, 'main_category': main_cat}
        else:
            print(f"未找到 {sub_cat} 的目录")
            features_data['categories'][sub_cat] = {'features': [], 'count': 0, 'main_category': main_cat}
    
    return features_data


def aggregate_features(feature_matrix, method='statistical'):
    """
    对特征矩阵进行聚合，减少维度
    
    参数:
    feature_matrix: 特征矩阵
    method: 聚合方法 ('statistical', 'mean')
    
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
        return np.array(aggregated)
    
    elif method == 'mean':
        # 简单均值聚合：每个特征取均值
        return np.mean(feature_matrix, axis=1)
    
    else:
        # 默认返回统计聚合特征
        return aggregate_features(feature_matrix, 'statistical')


def load_audio_file(audio_path, target_sr=16000):
    """加载音频文件，重采样到16kHz，确保为单声道"""
    try:
        # 使用librosa加载，重采样到16kHz
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 检查是否为单声道
        if len(y.shape) > 1:
            print(f"警告: {os.path.basename(audio_path)} 是多声道，已转换为单声道")
            y = y[:, 0] if len(y.shape) > 1 else y
        
        return y, sr
    except Exception as e:
        print(f"加载音频文件失败 {audio_path}: {e}")
        return None, None


def create_stereo_audio(left_audio, right_audio, output_path, sr=16000, target_duration=5.0):
    """
    创建左右声道分离的双声道音频，确保两个声道都是5秒
    
    参数:
    left_audio: 左声道音频数据（第一个音频，单声道）
    right_audio: 右声道音频数据（第二个音频，单声道）
    output_path: 输出文件路径
    sr: 采样率
    target_duration: 目标时长（秒），默认5秒
    """
    try:
        # 确保输入是单声道
        if len(left_audio.shape) > 1:
            left_audio = left_audio.flatten()
        if len(right_audio.shape) > 1:
            right_audio = right_audio.flatten()
        
        # 计算目标样本数（5秒）
        target_samples = int(target_duration * sr)
        
        # 处理左声道
        left_len = len(left_audio)
        if left_len > target_samples:
            left_audio = left_audio[:target_samples]
        elif left_len < target_samples:
            left_audio = np.pad(left_audio, (0, target_samples - left_len), mode='constant')
        
        # 处理右声道
        right_len = len(right_audio)
        if right_len > target_samples:
            right_audio = right_audio[:target_samples]
        elif right_len < target_samples:
            right_audio = np.pad(right_audio, (0, target_samples - right_len), mode='constant')
        
        # 创建左右声道分离的立体声数组
        stereo_audio = np.column_stack((left_audio, right_audio))
        
        # 保存为WAV文件
        sf.write(output_path, stereo_audio, sr)
        
        return True
    except Exception as e:
        print(f"创建立体声音频失败: {e}")
        return False


def get_all_audios_by_main_category(features_data, main_cat):
    """获取指定大类别下的所有音频"""
    category_mapping = get_category_mapping()
    sub_cats = category_mapping[main_cat]
    
    all_audios = []
    for sub_cat in sub_cats:
        if sub_cat in features_data['categories']:
            all_audios.extend(features_data['categories'][sub_cat]['features'])
    
    return all_audios


def sample_audios_balanced(features_data, main_cat, num_samples):
    """
    从指定大类别中平衡地抽取音频
    确保每个子类别被抽取的次数相同
    
    参数:
    features_data: 特征数据
    main_cat: 大类别名称
    num_samples: 需要抽取的总数
    
    返回:
    sampled_audios: 抽取的音频列表
    """
    category_mapping = get_category_mapping()
    sub_cats = category_mapping[main_cat]
    
    # 计算每个子类别需要抽取的数量
    samples_per_subcat = num_samples // len(sub_cats)
    
    sampled_audios = []
    for sub_cat in sub_cats:
        if sub_cat not in features_data['categories']:
            print(f"  警告: 子类别 {sub_cat} 不存在")
            continue
            
        available_audios = features_data['categories'][sub_cat]['features']
        
        if len(available_audios) < samples_per_subcat:
            print(f"  警告: {sub_cat} 可用音频不足 {samples_per_subcat} 个，只有 {len(available_audios)} 个，使用重复抽样")
            sampled = random.choices(available_audios, k=samples_per_subcat)
        else:
            sampled = random.sample(available_audios, samples_per_subcat)
        
        sampled_audios.extend(sampled)
        print(f"  从 {sub_cat} 抽取了 {len(sampled)} 条音频")
    
    # 打乱顺序
    random.shuffle(sampled_audios)
    
    return sampled_audios


def find_nearest_neighbor_by_features(query_audio_info, candidate_audios):
    """
    根据特征找到最近邻音频（排除查询音频本身）
    
    参数:
    query_audio_info: 查询音频的信息（包含特征）
    candidate_audios: 候选音频列表（包含特征）
    
    返回:
    nearest_audio_info: 最近邻音频信息
    similarity: 相似度分数
    """
    if not candidate_audios:
        return None, 0
    
    # 获取查询音频的文件名，用于排除自己
    query_file_name = query_audio_info['file_name']
    
    # 获取查询音频的特征矩阵
    query_feature_matrix = query_audio_info['features']['feature_matrix_normalized']
    
    # 聚合查询特征
    query_aggregated = aggregate_features(query_feature_matrix, 'statistical')
    query_aggregated = np.nan_to_num(query_aggregated, nan=0.0, posinf=0.0, neginf=0.0)
    
    best_similarity = -1
    best_audio = None
    
    for candidate_audio in candidate_audios:
        # 跳过查询音频本身（通过文件名判断）
        if candidate_audio['file_name'] == query_file_name:
            continue
        
        # 获取候选音频的特征矩阵
        candidate_feature_matrix = candidate_audio['features']['feature_matrix_normalized']
        
        # 聚合候选特征
        candidate_aggregated = aggregate_features(candidate_feature_matrix, 'statistical')
        candidate_aggregated = np.nan_to_num(candidate_aggregated, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算余弦相似度
        similarity = cosine_similarity([query_aggregated], [candidate_aggregated])[0][0]
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_audio = candidate_audio
    
    return best_audio, best_similarity


def nn_main_category_matching(features_data, output_dir):
    """
    大类别最近邻匹配阶段
    10种组合（包括自身），每种重复6次，每次生成2条音频
    总共120条音频
    """
    print("\n=== 开始大类别最近邻匹配阶段 ===")
    
    category_mapping = get_category_mapping()
    main_categories = list(category_mapping.keys())
    
    # 生成所有大类别组合（包括自身）
    all_combinations = list(combinations_with_replacement(main_categories, 2))
    print(f"大类别组合数: {len(all_combinations)}")
    print(f"组合列表: {all_combinations}")
    
    pairs_created = []
    total_audios = 0
    
    # 为每种组合创建配对
    for combo_idx, (cat1, cat2) in enumerate(all_combinations):
        print(f"\n--- 组合 {combo_idx + 1}/{len(all_combinations)}: {cat1} × {cat2} ---")
        
        # 从cat1平衡抽取6条音频（每个子类3条）
        audios_cat1_sampled = sample_audios_balanced(features_data, cat1, 6)
        
        # 获取cat2的所有音频（用于最近邻搜索）
        audios_cat2_all = get_all_audios_by_main_category(features_data, cat2)
        
        print(f"  从 {cat1} 平衡抽取了 {len(audios_cat1_sampled)} 条音频")
        print(f"  {cat2} 共有 {len(audios_cat2_all)} 条音频可供匹配")
        
        if len(audios_cat1_sampled) == 0 or len(audios_cat2_all) == 0:
            print(f"  错误: 类别音频不足，跳过")
            continue
        
        # 为抽取的6条音频，每条在cat2中找最近邻，生成2条音频（左右互换）
        for i, audio_A in enumerate(audios_cat1_sampled):
            # 在cat2中找到最近邻音频
            audio_B, similarity = find_nearest_neighbor_by_features(audio_A, audios_cat2_all)
            
            if audio_B is None:
                print(f"  配对 {i+1}: 未找到最近邻音频")
                continue
            
            # 加载音频
            audio_A_data, sr = load_audio_file(audio_A['file_path'])
            audio_B_data, sr = load_audio_file(audio_B['file_path'])
            
            if audio_A_data is None or audio_B_data is None:
                print(f"  配对 {i+1}: 音频加载失败")
                continue
            
            # 生成文件名ID（去掉扩展名）
            audio_A_id = os.path.splitext(audio_A['file_name'])[0]
            audio_B_id = os.path.splitext(audio_B['file_name'])[0]
            
            # 配对1: A左B右
            output_filename_1 = f"nn_main_{audio_A['sub_category']}_{audio_A_id}+{audio_B['sub_category']}_{audio_B_id}.wav"
            output_path_1 = os.path.join(output_dir, "nn_main_category_pairs", output_filename_1)
            os.makedirs(os.path.dirname(output_path_1), exist_ok=True)
            
            success_1 = create_stereo_audio(audio_A_data, audio_B_data, output_path_1, sr)
            if success_1:
                pairs_created.append({
                    'type': 'nn_main_category',
                    'combination': f"{cat1} × {cat2}",
                    'left_main_category': cat1,
                    'left_sub_category': audio_A['sub_category'],
                    'left_file': audio_A['file_name'],
                    'right_main_category': cat2,
                    'right_sub_category': audio_B['sub_category'],
                    'right_file': audio_B['file_name'],
                    'similarity': similarity,
                    'output_file': output_filename_1
                })
                total_audios += 1
                print(f"  ✓ 配对 {i+1}-1: {audio_A['sub_category']}(左) + {audio_B['sub_category']}(右) [相似度: {similarity:.4f}]")
            
            # 配对2: B左A右
            output_filename_2 = f"nn_main_{audio_B['sub_category']}_{audio_B_id}+{audio_A['sub_category']}_{audio_A_id}.wav"
            output_path_2 = os.path.join(output_dir, "nn_main_category_pairs", output_filename_2)
            
            success_2 = create_stereo_audio(audio_B_data, audio_A_data, output_path_2, sr)
            if success_2:
                pairs_created.append({
                    'type': 'nn_main_category',
                    'combination': f"{cat1} × {cat2}",
                    'left_main_category': cat2,
                    'left_sub_category': audio_B['sub_category'],
                    'left_file': audio_B['file_name'],
                    'right_main_category': cat1,
                    'right_sub_category': audio_A['sub_category'],
                    'right_file': audio_A['file_name'],
                    'similarity': similarity,
                    'output_file': output_filename_2
                })
                total_audios += 1
                print(f"  ✓ 配对 {i+1}-2: {audio_B['sub_category']}(左) + {audio_A['sub_category']}(右) [相似度: {similarity:.4f}]")
    
    print(f"\n大类别最近邻匹配完成: 创建了 {total_audios} 条音频")
    return pairs_created


def nn_sub_category_matching(features_data, output_dir):
    """
    子类别最近邻匹配阶段
    同一大类别内的2个子类别配对，4种组合，每种重复3次，每次生成2条音频
    总共24条音频
    """
    print("\n=== 开始子类别最近邻匹配阶段 ===")
    
    category_mapping = get_category_mapping()
    pairs_created = []
    total_audios = 0
    
    # 为每个大类别内的子类别进行配对
    for main_cat, sub_cats in category_mapping.items():
        print(f"\n--- 大类别: {main_cat} ---")
        print(f"  子类别: {sub_cats}")
        
        if len(sub_cats) != 2:
            print(f"  警告: {main_cat} 的子类别数量不是2，跳过")
            continue
        
        sub_cat_1, sub_cat_2 = sub_cats
        
        # 获取两个子类别的所有音频
        if sub_cat_1 not in features_data['categories'] or sub_cat_2 not in features_data['categories']:
            print(f"  错误: 子类别数据不存在，跳过")
            continue
        
        audios_1 = features_data['categories'][sub_cat_1]['features']
        audios_2 = features_data['categories'][sub_cat_2]['features']
        
        print(f"  {sub_cat_1} 共有 {len(audios_1)} 条音频")
        print(f"  {sub_cat_2} 共有 {len(audios_2)} 条音频")
        
        if len(audios_1) == 0 or len(audios_2) == 0:
            print(f"  错误: 子类别音频不足，跳过")
            continue
        
        # 从sub_cat_1随机抽取3条音频
        if len(audios_1) < 3:
            print(f"  警告: {sub_cat_1} 音频不足3个，使用重复抽样")
            audios_1_sampled = random.choices(audios_1, k=3)
        else:
            audios_1_sampled = random.sample(audios_1, 3)
        
        print(f"  从 {sub_cat_1} 抽取了 {len(audios_1_sampled)} 条音频")
        
        # 为抽取的3条音频，每条在sub_cat_2中找最近邻，生成2条音频（左右互换）
        for i, audio_A in enumerate(audios_1_sampled):
            # 在sub_cat_2中找到最近邻音频
            audio_B, similarity = find_nearest_neighbor_by_features(audio_A, audios_2)
            
            if audio_B is None:
                print(f"  配对 {i+1}: 未找到最近邻音频")
                continue
            
            # 加载音频
            audio_A_data, sr = load_audio_file(audio_A['file_path'])
            audio_B_data, sr = load_audio_file(audio_B['file_path'])
            
            if audio_A_data is None or audio_B_data is None:
                print(f"  配对 {i+1}: 音频加载失败")
                continue
            
            # 生成文件名ID（去掉扩展名）
            audio_A_id = os.path.splitext(audio_A['file_name'])[0]
            audio_B_id = os.path.splitext(audio_B['file_name'])[0]
            
            # 配对1: A左B右
            output_filename_1 = f"nn_sub_{sub_cat_1}_{audio_A_id}+{sub_cat_2}_{audio_B_id}.wav"
            output_path_1 = os.path.join(output_dir, "nn_sub_category_pairs", output_filename_1)
            os.makedirs(os.path.dirname(output_path_1), exist_ok=True)
            
            success_1 = create_stereo_audio(audio_A_data, audio_B_data, output_path_1, sr)
            if success_1:
                pairs_created.append({
                    'type': 'nn_sub_category',
                    'main_category': main_cat,
                    'left_sub_category': sub_cat_1,
                    'left_file': audio_A['file_name'],
                    'right_sub_category': sub_cat_2,
                    'right_file': audio_B['file_name'],
                    'similarity': similarity,
                    'output_file': output_filename_1
                })
                total_audios += 1
                print(f"  ✓ 配对 {i+1}-1: {sub_cat_1}(左) + {sub_cat_2}(右) [相似度: {similarity:.4f}]")
            
            # 配对2: B左A右
            output_filename_2 = f"nn_sub_{sub_cat_2}_{audio_B_id}+{sub_cat_1}_{audio_A_id}.wav"
            output_path_2 = os.path.join(output_dir, "nn_sub_category_pairs", output_filename_2)
            
            success_2 = create_stereo_audio(audio_B_data, audio_A_data, output_path_2, sr)
            if success_2:
                pairs_created.append({
                    'type': 'nn_sub_category',
                    'main_category': main_cat,
                    'left_sub_category': sub_cat_2,
                    'left_file': audio_B['file_name'],
                    'right_sub_category': sub_cat_1,
                    'right_file': audio_A['file_name'],
                    'similarity': similarity,
                    'output_file': output_filename_2
                })
                total_audios += 1
                print(f"  ✓ 配对 {i+1}-2: {sub_cat_2}(左) + {sub_cat_1}(右) [相似度: {similarity:.4f}]")
    
    print(f"\n子类别最近邻匹配完成: 创建了 {total_audios} 条音频")
    return pairs_created


def save_pairing_results(pairs_created, output_dir):
    """保存配对结果"""
    # 保存为JSON文件
    json_path = os.path.join(output_dir, "nn_pairing_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_created, f, ensure_ascii=False, indent=2)
    
    # 保存为CSV文件
    df = pd.DataFrame(pairs_created)
    csv_path = os.path.join(output_dir, "nn_pairing_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n配对结果已保存:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def main(features_path, clap_audio_path, output_dir):
    """
    主函数：大类别和子类别最近邻匹配音频配对合成
    
    参数:
    features_path: 特征数据路径（包含各子类别numpy文件）
    clap_audio_path: 音频文件路径（包含各子类别目录）
    output_dir: 输出目录
    """
    print("=== 最近邻匹配音频配对合成工具 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载特征数据
    print("\n正在加载特征数据...")
    features_data = load_features_data_numpy_5s(features_path, clap_audio_path)
    
    # 检查特征数据是否成功加载
    category_mapping = get_category_mapping()
    print(f"\n=== 特征数据加载统计 ===")
    for main_cat, sub_cats in category_mapping.items():
        print(f"{main_cat}:")
        for sub_cat in sub_cats:
            count = features_data['categories'].get(sub_cat, {}).get('count', 0)
            print(f"  {sub_cat}: {count} 个音频文件")
    
    # 阶段1: 大类别最近邻匹配
    main_pairs = nn_main_category_matching(features_data, output_dir)
    
    # 阶段2: 子类别最近邻匹配
    sub_pairs = nn_sub_category_matching(features_data, output_dir)
    
    # 合并所有配对结果
    all_pairs = main_pairs + sub_pairs
    
    # 保存结果
    save_pairing_results(all_pairs, output_dir)
    
    # 输出统计信息
    print(f"\n=== 配对完成统计 ===")
    print(f"大类别最近邻匹配: {len(main_pairs)} 对")
    print(f"子类别最近邻匹配: {len(sub_pairs)} 对")
    print(f"总配对: {len(all_pairs)} 对")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    # 设置路径
    features_path = r"D:\D\research\audioset下载\特征"  # 特征数据路径
    clap_audio_path = r"D:\D\research\audioset下载\clap_select"  # 音频文件路径
    output_dir = r"D:\D\research\audioset下载\最近邻匹配_大类小类"  # 输出目录
    
    # 运行主函数
    main(features_path, clap_audio_path, output_dir)

