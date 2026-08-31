import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def parse_audio_filename(filename):
    """
    解析音频文件名，提取左右声道信息
    
    示例：
    nn_Ambulance (siren)_1LA8BxMlXSY_2.wav+Helicopter_DSLO9BKHSEk_1.wav.wav
    rn_Ambulance (siren)_1LA8BxMlXSY_2.wav+Helicopter_DSLO9BKHSEk_1.wav.wav
    """
    # 只处理以 nn_ 或 rn_ 开头的文件
    if filename.startswith('nn_'):
        pair_type = 'nn'
        filename = filename[3:]  # 去掉 'nn_' 前缀
    elif filename.startswith('rn_'):
        pair_type = 'rn'
        filename = filename[3:]  # 去掉 'rn_' 前缀
    else:
        # 没有 nn_ 或 rn_ 前缀，跳过
        return None
    
    # 去掉 .wav 后缀
    filename = filename.replace('.wav', '')
    
    # 按 '+' 分割左右声道
    parts = filename.split('+')
    if len(parts) != 2:
        print(f"警告: 无法解析文件名 {filename}")
        return None
    
    left_part = parts[0].strip()
    right_part = parts[1].strip()
    
    # 解析左声道：类别名_音频ID
    # 需要找到最后一个下划线前的部分作为类别，最后的部分作为ID
    left_tokens = left_part.rsplit('_', 2)  # 从右边分割，最多分2次
    if len(left_tokens) >= 2:
        # 最后两个token是 videoID_segment（如 1LA8BxMlXSY_2）
        left_category = '_'.join(left_tokens[:-2]) if len(left_tokens) > 2 else left_tokens[0]
        left_audio_id = '_'.join(left_tokens[-2:])
    else:
        print(f"警告: 无法解析左声道 {left_part}")
        return None
    
    # 解析右声道
    right_tokens = right_part.rsplit('_', 2)
    if len(right_tokens) >= 2:
        right_category = '_'.join(right_tokens[:-2]) if len(right_tokens) > 2 else right_tokens[0]
        right_audio_id = '_'.join(right_tokens[-2:])
    else:
        print(f"警告: 无法解析右声道 {right_part}")
        return None
    
    return {
        'pair_type': pair_type,
        'left_category': left_category,
        'left_audio_id': left_audio_id,
        'right_category': right_category,
        'right_audio_id': right_audio_id
    }

def normalize_category_name(category_name):
    """
    标准化类别名称，处理特殊字符
    将文件名中的类别名转换为目录名格式
    """
    # 映射关系
    name_mapping = {
        'Ambulance_siren': 'Ambulance (siren)',
        'Ambulance (siren)': 'Ambulance (siren)',
        'Baby_cry_infant_cry': 'Baby cry, infant cry',
        'Baby cry, infant cry': 'Baby cry, infant cry',
        'Telephone_bell_ringing': 'Telephone bell ringing',
        'Telephone bell ringing': 'Telephone bell ringing',
        'Computer_keyboard': 'Computer keyboard',
        'Computer keyboard': 'Computer keyboard',
        'Helicopter': 'Helicopter',
        'Chicken_rooster': 'Chicken, rooster',
        'Chicken, rooster': 'Chicken, rooster',
        'Male_speech_man_speaking': 'Male speech, man speaking',
        'Male speech, man speaking': 'Male speech, man speaking',
        'Female_speech_woman_speaking': 'Female speech, woman speaking',
        'Female speech, woman speaking': 'Female speech, woman speaking',
        'Child_speech_kid_speaking': 'Child speech, kid speaking',
        'Child speech, kid speaking': 'Child speech, kid speaking',
        'Bass_drum': 'Bass drum',
        'Bass drum': 'Bass drum',
        'Funny_music': 'Funny music',
        'Funny music': 'Funny music',
        'Sad_music': 'Sad music',
        'Sad music': 'Sad music',
        'Pulse': 'Pulse',
        'Whack_thwack': 'Whack, thwack',
        'Whack, thwack': 'Whack, thwack',
        'Crumpling_crinkling': 'Crumpling, crinkling',
        'Crumpling, crinkling': 'Crumpling, crinkling'
    }
    
    return name_mapping.get(category_name, category_name)

def load_feature_from_file(features_base_path, category, audio_id):
    """
    从特征文件中加载指定音频的特征
    
    参数：
    features_base_path: 特征文件根目录
    category: 音频类别
    audio_id: 音频ID（如 1LA8BxMlXSY_2）
    
    返回：
    feature_matrix: 特征矩阵，如果未找到返回None
    
    注意：
    - 如果 audio_id 以 _1 结尾，表示前5秒，取特征矩阵的前一半
    - 如果 audio_id 以 _2 结尾，表示后5秒，取特征矩阵的后一半
    """
    # 标准化类别名
    category = normalize_category_name(category)
    
    # 判断是前5秒还是后5秒
    if audio_id.endswith('_1'):
        half_type = 'first_half'  # 前5秒
        # 去掉 _1，获取原始音频ID
        original_audio_id = audio_id[:-2]
    elif audio_id.endswith('_2'):
        half_type = 'second_half'  # 后5秒
        # 去掉 _2，获取原始音频ID
        original_audio_id = audio_id[:-2]
    else:
        # 没有 _1 或 _2 后缀，使用完整特征
        half_type = 'full'
        original_audio_id = audio_id
    
    # 构建特征文件路径（使用原始音频ID）
    feature_filename = f"{original_audio_id}_features.npy"
    feature_path = os.path.join(features_base_path, category, feature_filename)
    
    if not os.path.exists(feature_path):
        print(f"  警告: 特征文件不存在 {feature_path}")
        return None
    
    try:
        # 加载特征矩阵
        feature_matrix = np.load(feature_path)
        
        # 根据 half_type 选择特征的前一半或后一半
        if half_type == 'first_half':
            # 前5秒，取特征矩阵的前一半
            half_point = feature_matrix.shape[1] // 2
            feature_matrix = feature_matrix[:, :half_point]
            print(f"  → 使用前一半特征 (前5秒), 形状: {feature_matrix.shape}")
        elif half_type == 'second_half':
            # 后5秒，取特征矩阵的后一半
            half_point = feature_matrix.shape[1] // 2
            feature_matrix = feature_matrix[:, half_point:]
            print(f"  → 使用后一半特征 (后5秒), 形状: {feature_matrix.shape}")
        else:
            # 使用完整特征
            print(f"  → 使用完整特征, 形状: {feature_matrix.shape}")
        
        return feature_matrix
    except Exception as e:
        print(f"  错误: 加载特征文件失败 {feature_path}: {e}")
        return None

def aggregate_features(feature_matrix, method='statistical'):
    """
    对特征矩阵进行聚合（从audio_pair_synthesis_5s.py复制）
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

def calculate_feature_difference(left_feature, right_feature):
    """
    计算两个特征向量之间的差异
    
    返回：
    {
        'cosine_similarity': 余弦相似度,
        'euclidean_distance': 欧氏距离,
        'cosine_distance': 余弦距离 (1 - 余弦相似度)
    }
    """
    # 处理NaN
    left_feature = np.nan_to_num(left_feature, nan=0.0, posinf=0.0, neginf=0.0)
    right_feature = np.nan_to_num(right_feature, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 余弦相似度
    cos_sim = cosine_similarity([left_feature], [right_feature])[0][0]
    
    # 欧氏距离
    euclidean_dist = np.linalg.norm(left_feature - right_feature)
    
    # 余弦距离
    cos_dist = 1 - cos_sim
    
    return {
        'cosine_similarity': cos_sim,
        'euclidean_distance': euclidean_dist,
        'cosine_distance': cos_dist
    }

def analyze_audio_pairs(audio_dir, features_base_path):
    """
    分析音频配对的左右声道特征差异
    """
    print("=== 开始分析音频配对的左右声道特征差异 ===\n")
    
    # 获取所有wav文件
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    # 分别存储nn和rn的分析结果
    nn_results = []
    rn_results = []
    skipped_count = 0  # 跳过的文件数
    
    # 分析每个音频文件
    for filename in audio_files:
        print(f"分析: {filename}")
        
        # 解析文件名
        parsed = parse_audio_filename(filename)
        if parsed is None:
            print(f"  ⊗ 跳过（没有 nn_ 或 rn_ 前缀）\n")
            skipped_count += 1
            continue
        
        pair_type = parsed['pair_type']
        left_category = parsed['left_category']
        left_audio_id = parsed['left_audio_id']
        right_category = parsed['right_category']
        right_audio_id = parsed['right_audio_id']
        
        print(f"  类型: {pair_type}")
        print(f"  左声道: {left_category} - {left_audio_id}")
        print(f"  右声道: {right_category} - {right_audio_id}")
        
        # 加载左声道特征
        left_feature_matrix = load_feature_from_file(features_base_path, left_category, left_audio_id)
        if left_feature_matrix is None:
            print(f"  ✗ 跳过（左声道特征未找到）\n")
            continue
        
        # 加载右声道特征
        right_feature_matrix = load_feature_from_file(features_base_path, right_category, right_audio_id)
        if right_feature_matrix is None:
            print(f"  ✗ 跳过（右声道特征未找到）\n")
            continue
        
        # 聚合特征
        left_aggregated = aggregate_features(left_feature_matrix, 'statistical')
        right_aggregated = aggregate_features(right_feature_matrix, 'statistical')
        
        # 计算特征差异
        difference = calculate_feature_difference(left_aggregated, right_aggregated)
        
        # 存储结果
        result = {
            'filename': filename,
            'pair_type': pair_type,
            'left_category': left_category,
            'right_category': right_category,
            'cosine_similarity': difference['cosine_similarity'],
            'euclidean_distance': difference['euclidean_distance'],
            'cosine_distance': difference['cosine_distance']
        }
        
        if pair_type == 'nn':
            nn_results.append(result)
        else:
            rn_results.append(result)
        
        print(f"  ✓ 余弦相似度: {difference['cosine_similarity']:.4f}")
        print(f"  ✓ 欧氏距离: {difference['euclidean_distance']:.4f}")
        print(f"  ✓ 余弦距离: {difference['cosine_distance']:.4f}\n")
    
    print(f"\n--- 分析统计 ---")
    print(f"NN组分析成功: {len(nn_results)} 个")
    print(f"RN组分析成功: {len(rn_results)} 个")
    print(f"跳过的文件: {skipped_count} 个")
    print(f"总计: {len(nn_results) + len(rn_results) + skipped_count} 个\n")
    
    return nn_results, rn_results

def statistical_analysis(nn_results, rn_results):
    """
    对nn和rn组进行统计分析
    """
    print("\n=== 统计分析 ===\n")
    
    # 转换为DataFrame
    df_nn = pd.DataFrame(nn_results)
    df_rn = pd.DataFrame(rn_results)
    
    print(f"NN组（最近邻匹配）: {len(df_nn)} 对")
    print(f"RN组（随机匹配）: {len(df_rn)} 对\n")
    
    if len(df_nn) == 0 or len(df_rn) == 0:
        print("警告: 某组数据为空，无法进行统计分析")
        return
    
    # 统计描述
    print("--- NN组统计描述 ---")
    print(df_nn[['cosine_similarity', 'euclidean_distance', 'cosine_distance']].describe())
    print("\n--- RN组统计描述 ---")
    print(df_rn[['cosine_similarity', 'euclidean_distance', 'cosine_distance']].describe())
    
    # t检验
    print("\n--- 独立样本t检验 ---")
    
    # 余弦相似度检验
    t_stat_cos, p_value_cos = ttest_ind(df_nn['cosine_similarity'], df_rn['cosine_similarity'])
    print(f"\n余弦相似度:")
    print(f"  NN组均值: {df_nn['cosine_similarity'].mean():.4f} ± {df_nn['cosine_similarity'].std():.4f}")
    print(f"  RN组均值: {df_rn['cosine_similarity'].mean():.4f} ± {df_rn['cosine_similarity'].std():.4f}")
    print(f"  t统计量: {t_stat_cos:.4f}")
    print(f"  p值: {p_value_cos:.6e}")  # 使用科学计数法
    if p_value_cos < 0.001:
        print(f"  *** 差异极显著 (p < 0.001)")
    elif p_value_cos < 0.01:
        print(f"  ** 差异非常显著 (p < 0.01)")
    elif p_value_cos < 0.05:
        print(f"  * 差异显著 (p < 0.05)")
    else:
        print(f"  差异不显著 (p >= 0.05)")
    
    # 欧氏距离检验
    t_stat_euc, p_value_euc = ttest_ind(df_nn['euclidean_distance'], df_rn['euclidean_distance'])
    print(f"\n欧氏距离:")
    print(f"  NN组均值: {df_nn['euclidean_distance'].mean():.4f} ± {df_nn['euclidean_distance'].std():.4f}")
    print(f"  RN组均值: {df_rn['euclidean_distance'].mean():.4f} ± {df_rn['euclidean_distance'].std():.4f}")
    print(f"  t统计量: {t_stat_euc:.4f}")
    print(f"  p值: {p_value_euc:.6e}")  # 使用科学计数法
    if p_value_euc < 0.001:
        print(f"  *** 差异极显著 (p < 0.001)")
    elif p_value_euc < 0.01:
        print(f"  ** 差异非常显著 (p < 0.01)")
    elif p_value_euc < 0.05:
        print(f"  * 差异显著 (p < 0.05)")
    else:
        print(f"  差异不显著 (p >= 0.05)")
    
    # 余弦距离检验
    t_stat_cosd, p_value_cosd = ttest_ind(df_nn['cosine_distance'], df_rn['cosine_distance'])
    print(f"\n余弦距离:")
    print(f"  NN组均值: {df_nn['cosine_distance'].mean():.4f} ± {df_nn['cosine_distance'].std():.4f}")
    print(f"  RN组均值: {df_rn['cosine_distance'].mean():.4f} ± {df_rn['cosine_distance'].std():.4f}")
    print(f"  t统计量: {t_stat_cosd:.4f}")
    print(f"  p值: {p_value_cosd:.6e}")  # 使用科学计数法
    if p_value_cosd < 0.001:
        print(f"  *** 差异极显著 (p < 0.001)")
    elif p_value_cosd < 0.01:
        print(f"  ** 差异非常显著 (p < 0.01)")
    elif p_value_cosd < 0.05:
        print(f"  * 差异显著 (p < 0.05)")
    else:
        print(f"  差异不显著 (p >= 0.05)")
    
    return df_nn, df_rn

def visualize_results(df_nn, df_rn, output_dir):
    """
    可视化分析结果
    """
    print("\n=== 生成可视化图表 ===\n")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 余弦相似度对比箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 余弦相似度
    data_cos = [df_nn['cosine_similarity'], df_rn['cosine_similarity']]
    axes[0].boxplot(data_cos, labels=['NN (最近邻)', 'RN (随机)'])
    axes[0].set_ylabel('余弦相似度', fontsize=12)
    axes[0].set_title('左右声道余弦相似度对比', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 欧氏距离
    data_euc = [df_nn['euclidean_distance'], df_rn['euclidean_distance']]
    axes[1].boxplot(data_euc, labels=['NN (最近邻)', 'RN (随机)'])
    axes[1].set_ylabel('欧氏距离', fontsize=12)
    axes[1].set_title('左右声道欧氏距离对比', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 余弦距离
    data_cosd = [df_nn['cosine_distance'], df_rn['cosine_distance']]
    axes[2].boxplot(data_cosd, labels=['NN (最近邻)', 'RN (随机)'])
    axes[2].set_ylabel('余弦距离', fontsize=12)
    axes[2].set_title('左右声道余弦距离对比', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'stereo_feature_difference_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"保存箱线图: {boxplot_path}")
    plt.close()
    
    # 2. 分布直方图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 余弦相似度分布
    axes[0, 0].hist(df_nn['cosine_similarity'], bins=20, alpha=0.7, label='NN', color='blue', edgecolor='black')
    axes[0, 0].axvline(df_nn['cosine_similarity'].mean(), color='blue', linestyle='--', linewidth=2, label=f'NN均值: {df_nn["cosine_similarity"].mean():.3f}')
    axes[0, 0].set_xlabel('余弦相似度', fontsize=11)
    axes[0, 0].set_ylabel('频数', fontsize=11)
    axes[0, 0].set_title('NN组 - 余弦相似度分布', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].hist(df_rn['cosine_similarity'], bins=20, alpha=0.7, label='RN', color='orange', edgecolor='black')
    axes[1, 0].axvline(df_rn['cosine_similarity'].mean(), color='orange', linestyle='--', linewidth=2, label=f'RN均值: {df_rn["cosine_similarity"].mean():.3f}')
    axes[1, 0].set_xlabel('余弦相似度', fontsize=11)
    axes[1, 0].set_ylabel('频数', fontsize=11)
    axes[1, 0].set_title('RN组 - 余弦相似度分布', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 欧氏距离分布
    axes[0, 1].hist(df_nn['euclidean_distance'], bins=20, alpha=0.7, label='NN', color='blue', edgecolor='black')
    axes[0, 1].axvline(df_nn['euclidean_distance'].mean(), color='blue', linestyle='--', linewidth=2, label=f'NN均值: {df_nn["euclidean_distance"].mean():.3f}')
    axes[0, 1].set_xlabel('欧氏距离', fontsize=11)
    axes[0, 1].set_ylabel('频数', fontsize=11)
    axes[0, 1].set_title('NN组 - 欧氏距离分布', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].hist(df_rn['euclidean_distance'], bins=20, alpha=0.7, label='RN', color='orange', edgecolor='black')
    axes[1, 1].axvline(df_rn['euclidean_distance'].mean(), color='orange', linestyle='--', linewidth=2, label=f'RN均值: {df_rn["euclidean_distance"].mean():.3f}')
    axes[1, 1].set_xlabel('欧氏距离', fontsize=11)
    axes[1, 1].set_ylabel('频数', fontsize=11)
    axes[1, 1].set_title('RN组 - 欧氏距离分布', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 余弦距离分布
    axes[0, 2].hist(df_nn['cosine_distance'], bins=20, alpha=0.7, label='NN', color='blue', edgecolor='black')
    axes[0, 2].axvline(df_nn['cosine_distance'].mean(), color='blue', linestyle='--', linewidth=2, label=f'NN均值: {df_nn["cosine_distance"].mean():.3f}')
    axes[0, 2].set_xlabel('余弦距离', fontsize=11)
    axes[0, 2].set_ylabel('频数', fontsize=11)
    axes[0, 2].set_title('NN组 - 余弦距离分布', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].hist(df_rn['cosine_distance'], bins=20, alpha=0.7, label='RN', color='orange', edgecolor='black')
    axes[1, 2].axvline(df_rn['cosine_distance'].mean(), color='orange', linestyle='--', linewidth=2, label=f'RN均值: {df_rn["cosine_distance"].mean():.3f}')
    axes[1, 2].set_xlabel('余弦距离', fontsize=11)
    axes[1, 2].set_ylabel('频数', fontsize=11)
    axes[1, 2].set_title('RN组 - 余弦距离分布', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'stereo_feature_difference_histogram.png')
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"保存直方图: {hist_path}")
    plt.close()
    
    # 3. 小提琴图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 准备数据
    data_combined = pd.concat([
        df_nn[['cosine_similarity', 'euclidean_distance', 'cosine_distance']].assign(group='NN'),
        df_rn[['cosine_similarity', 'euclidean_distance', 'cosine_distance']].assign(group='RN')
    ])
    
    # 余弦相似度小提琴图
    parts = axes[0].violinplot([df_nn['cosine_similarity'], df_rn['cosine_similarity']], 
                                positions=[1, 2], showmeans=True, showmedians=True)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['NN (最近邻)', 'RN (随机)'])
    axes[0].set_ylabel('余弦相似度', fontsize=12)
    axes[0].set_title('左右声道余弦相似度分布', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 欧氏距离小提琴图
    parts = axes[1].violinplot([df_nn['euclidean_distance'], df_rn['euclidean_distance']], 
                                positions=[1, 2], showmeans=True, showmedians=True)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['NN (最近邻)', 'RN (随机)'])
    axes[1].set_ylabel('欧氏距离', fontsize=12)
    axes[1].set_title('左右声道欧氏距离分布', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 余弦距离小提琴图
    parts = axes[2].violinplot([df_nn['cosine_distance'], df_rn['cosine_distance']], 
                                positions=[1, 2], showmeans=True, showmedians=True)
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(['NN (最近邻)', 'RN (随机)'])
    axes[2].set_ylabel('余弦距离', fontsize=12)
    axes[2].set_title('左右声道余弦距离分布', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    violin_path = os.path.join(output_dir, 'stereo_feature_difference_violin.png')
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    print(f"保存小提琴图: {violin_path}")
    plt.close()

def save_results(nn_results, rn_results, df_nn, df_rn, output_dir):
    """
    保存分析结果到CSV和JSON
    """
    print("\n=== 保存分析结果 ===\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    df_all = pd.concat([df_nn.assign(group='NN'), df_rn.assign(group='RN')])
    csv_path = os.path.join(output_dir, 'stereo_feature_difference_details.csv')
    df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"保存详细结果: {csv_path}")
    
    # 保存统计摘要
    summary_data = {
        'NN组': {
            '样本数': len(df_nn),
            '余弦相似度_均值': df_nn['cosine_similarity'].mean(),
            '余弦相似度_标准差': df_nn['cosine_similarity'].std(),
            '余弦相似度_中位数': df_nn['cosine_similarity'].median(),
            '欧氏距离_均值': df_nn['euclidean_distance'].mean(),
            '欧氏距离_标准差': df_nn['euclidean_distance'].std(),
            '欧氏距离_中位数': df_nn['euclidean_distance'].median(),
            '余弦距离_均值': df_nn['cosine_distance'].mean(),
            '余弦距离_标准差': df_nn['cosine_distance'].std(),
            '余弦距离_中位数': df_nn['cosine_distance'].median(),
        },
        'RN组': {
            '样本数': len(df_rn),
            '余弦相似度_均值': df_rn['cosine_similarity'].mean(),
            '余弦相似度_标准差': df_rn['cosine_similarity'].std(),
            '余弦相似度_中位数': df_rn['cosine_similarity'].median(),
            '欧氏距离_均值': df_rn['euclidean_distance'].mean(),
            '欧氏距离_标准差': df_rn['euclidean_distance'].std(),
            '欧氏距离_中位数': df_rn['euclidean_distance'].median(),
            '余弦距离_均值': df_rn['cosine_distance'].mean(),
            '余弦距离_标准差': df_rn['cosine_distance'].std(),
            '余弦距离_中位数': df_rn['cosine_distance'].median(),
        }
    }
    
    df_summary = pd.DataFrame(summary_data).T
    summary_path = os.path.join(output_dir, 'stereo_feature_difference_summary.csv')
    df_summary.to_csv(summary_path, encoding='utf-8-sig')
    print(f"保存统计摘要: {summary_path}")

def main():
    """主函数"""
    # 设置路径
    audio_dir = r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120"
    features_base_path = r"D:\D\research\audioset下载\特征"
    output_dir = r"D:\D\research\audioset下载\立体声特征差异分析"
    
    print("音频目录:", audio_dir)
    print("特征目录:", features_base_path)
    print("输出目录:", output_dir)
    print()
    
    # 分析音频配对
    nn_results, rn_results = analyze_audio_pairs(audio_dir, features_base_path)
    
    if len(nn_results) == 0 and len(rn_results) == 0:
        print("\n错误: 没有成功分析任何音频文件")
        return
    
    # 统计分析
    df_nn, df_rn = statistical_analysis(nn_results, rn_results)
    
    # 可视化
    visualize_results(df_nn, df_rn, output_dir)
    
    # 保存结果
    save_results(nn_results, rn_results, df_nn, df_rn, output_dir)
    
    print("\n=== 分析完成 ===")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()

