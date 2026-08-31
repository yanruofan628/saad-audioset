#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Wav2Vec2惊喜度指标建立线性回归模型预测人类选择概率

建立8个模型：
1. MSE sum差值模型
2. MSE max差值模型
3. Cosine sum差值模型
4. Cosine max差值模型
5. Representation_change sum差值模型
6. Representation_change max差值模型
7. Entropy sum差值模型
8. Entropy max差值模型

每个模型形式：P(选择左) = β₀ + β₁ × 差值（包含截距）
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import shutil
warnings.filterwarnings("ignore")

WAV2VEC_FEATURE_NAMES = [
    'mse_sum_diff',
    'mse_max_diff',
    'cosine_sum_diff',
    'cosine_max_diff',
    'representation_change_sum_diff',
    'representation_change_max_diff',
    'entropy_sum_diff',
    'entropy_max_diff'
]

# 尝试使用HuggingFace镜像（如果网络连接有问题）
# 如果遇到连接超时，可以取消下面的注释使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入复用函数
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from linear_regression_selected_features import (
    parse_pair_name,
    merge_balanced_pairs_by_type,
    parse_experiment_data_txt,
    parse_experiment_data_csv,
    parse_audio_mapping,
    load_all_human_decisions,
    calculate_selection_probability_144
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def check_wav2vec_results(output_dir):
    """
    检查wav2vec结果文件是否存在（支持多种前缀格式）
    
    参数:
        output_dir: wav2vec结果输出目录
    
    返回:
        results: 如果存在则返回结果列表，否则返回None
    """
    results = []
    
    # 查找所有可能的结果文件（支持多种前缀）
    possible_prefixes = ['nn_main', 'nn_sub', 'main', 'sub', 'nn', 'rn']
    
    for prefix in possible_prefixes:
        json_path = os.path.join(output_dir, f'wav2vec_surprise_raw_results_{prefix}.json')
        if os.path.exists(json_path):
            print(f"找到{prefix}结果文件: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.extend(data)
                print(f"  加载了 {len(data)} 个{prefix}结果")
    
    if len(results) > 0:
        print(f"总共找到 {len(results)} 个wav2vec结果")
        return results
    else:
        print("未找到wav2vec结果文件")
        return None


def parse_stereo_pair_name_for_audio(pair_name, base_dir):
    """
    解析pair_name并找到对应的左右音频文件路径（参照pair_feature_diff_extraction.py）
    
    参数:
        pair_name: 音频对名称，如 nn_main_Helicopter_W3u2hj1x7gY_2+Telephone bell ringing__xuq9rBndUE_1.wav
        base_dir: 基础目录（按类别组织的音频文件，如clap_select）
    
    返回:
        (left_path, right_path, left_cat, left_id, right_cat, right_id) 或 None
    """
    # 去掉扩展名
    if pair_name.lower().endswith('.wav'):
        filename = pair_name[:-4]
    else:
        filename = pair_name
    
    # 切分左右
    if '+' not in filename:
        return None
    
    left_part, right_part = filename.split('+', 1)
    
    # 去掉前缀
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break
    
    # 基于文件系统解析类别和ID（参照pair_feature_diff_extraction.py的逻辑）
    def split_category_and_id_fs(part):
        us_positions = [i for i, ch in enumerate(part) if ch == '_']
        for pos in us_positions:
            category = part[:pos]
            file_id = part[pos + 1:]
            candidate = os.path.join(base_dir, category, f"{file_id}.wav")
            if os.path.exists(candidate):
                return category, file_id, candidate
        return None, None, None
    
    left_cat, left_id, left_path = split_category_and_id_fs(left_part)
    right_cat, right_id, right_path = split_category_and_id_fs(right_part)
    
    if left_path and right_path:
        return (left_path, right_path, left_cat, left_id, right_cat, right_id)
    return None


def create_stereo_audio_from_pair(left_path, right_path, output_path):
    """
    从左右单声道音频创建立体声音频文件
    
    参数:
        left_path: 左声道音频路径
        right_path: 右声道音频路径
        output_path: 输出立体声音频路径
    """
    import librosa
    import soundfile as sf
    
    # 加载左右音频
    left_audio, sr_left = librosa.load(left_path, sr=16000, mono=True)
    right_audio, sr_right = librosa.load(right_path, sr=16000, mono=True)
    
    # 确保长度一致
    min_len = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_len]
    right_audio = right_audio[:min_len]
    
    # 创建立体声（2通道）
    stereo_audio = np.stack([left_audio, right_audio], axis=0)
    
    # 保存为立体声WAV文件
    sf.write(output_path, stereo_audio.T, sr_left, format='WAV')
    
    return output_path


def save_sample_stereo_audios(pair_names, base_dir, sample_output_dir, sample_count=2):
    """保存部分音频对的临时立体声文件，便于试听"""
    os.makedirs(sample_output_dir, exist_ok=True)

    saved = 0
    for pair_name in pair_names:
        if saved >= sample_count:
            break

        parse_result = parse_stereo_pair_name_for_audio(pair_name, base_dir)
        if not parse_result:
            continue

        left_path, right_path, *_ = parse_result

        safe_name = os.path.splitext(pair_name)[0]
        safe_name = re.sub(r'[\\/:*?"<>|]+', '_', safe_name)
        output_path = os.path.join(sample_output_dir, f"sample_{saved + 1}_{safe_name}.wav")

        try:
            create_stereo_audio_from_pair(left_path, right_path, output_path)
            print(f"示例立体声已保存: {output_path}")
            saved += 1
        except Exception as exc:
            print(f"保存示例立体声失败 {pair_name}: {exc}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

    if saved == 0:
        print("警告: 未能保存任何示例立体声文件，请检查音频路径")

    return saved


def load_loudness_std_diff_map(project_root):
    """加载每个音频对的响度标准差差值（左-右）映射"""
    raw_path = os.path.join(project_root, 'pair_feature_diff_raw.npy')
    list_path = os.path.join(project_root, 'pair_list.npy')

    if not os.path.exists(raw_path) or not os.path.exists(list_path):
        print("警告: 未找到 pair_feature_diff_raw.npy 或 pair_list.npy，无法进行响度差异相关分析")
        return {}

    try:
        diff_raw = np.load(raw_path)
        pair_list = np.load(list_path, allow_pickle=True)
    except Exception as exc:
        print(f"警告: 加载响度差异数据失败: {exc}")
        return {}

    if diff_raw.ndim != 2 or diff_raw.shape[0] <= 11:
        print("警告: 特征矩阵形状异常，无法提取响度std差值")
        return {}

    loudness_std_index = 11  # 第12个特征：Loudness (std)
    if diff_raw.shape[1] != len(pair_list):
        print("警告: 特征矩阵与pair_list长度不一致，可能影响映射精度")

    loudness_std_diff = diff_raw[loudness_std_index, :]
    loudness_map = {}
    for idx, name in enumerate(pair_list):
        try:
            loudness_map[str(name)] = float(loudness_std_diff[idx])
        except (ValueError, TypeError, IndexError):
            continue

    return loudness_map


def analyze_wav2vec_loudness_correlation(df, feature_names, output_path):
    """计算Wav2Vec特征与响度标准差差值之间的皮尔逊相关"""
    if 'loudness_std_diff' not in df.columns:
        print("警告: 数据中缺少 loudness_std_diff 列，跳过相关性分析")
        return None

    results = []
    for feature in feature_names:
        if feature not in df.columns:
            print(f"  警告: 特征 {feature} 缺失，跳过相关分析")
            continue

        subset = df[[feature, 'loudness_std_diff']].dropna()
        n = len(subset)
        if n < 3:
            print(f"  警告: 特征 {feature} 有效样本数不足3，跳过相关分析")
            results.append({
                'feature_name': feature,
                'n_samples': n,
                'pearson_r': np.nan,
                'p_value': np.nan
            })
            continue

        r, p = pearsonr(subset[feature], subset['loudness_std_diff'])
        results.append({
            'feature_name': feature,
            'n_samples': n,
            'pearson_r': float(r),
            'p_value': float(p)
        })

    if not results:
        print("警告: 未生成任何相关性结果")
        return None

    result_df = pd.DataFrame(results)
    try:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Wav2Vec特征与响度std差值相关性结果已保存到: {output_path}")
    except Exception as exc:
        print(f"警告: 保存相关性结果失败: {exc}")

    return result_df


def analyze_feature_probability_correlation(df, feature_names, probability_column, output_path):
    """计算各特征与人类选择概率之间的相关性"""
    if probability_column not in df.columns:
        print(f"警告: 数据中缺少 {probability_column} 列，跳过相关性分析")
        return None

    results = []
    for feature in feature_names:
        if feature not in df.columns:
            print(f"  警告: 特征 {feature} 缺失，跳过概率相关分析")
            continue

        subset = df[[feature, probability_column]].dropna()
        n = len(subset)
        if n < 3:
            print(f"  警告: 特征 {feature} 与概率的有效样本数不足3，跳过")
            results.append({
                'feature_name': feature,
                'n_samples': n,
                'pearson_r': np.nan,
                'p_value': np.nan
            })
            continue

        r, p = pearsonr(subset[feature], subset[probability_column])
        results.append({
            'feature_name': feature,
            'n_samples': n,
            'pearson_r': float(r),
            'p_value': float(p)
        })

    if not results:
        print("警告: 未生成任何特征-概率相关性结果")
        return None

    result_df = pd.DataFrame(results)
    try:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"特征与选择概率相关性结果已保存到: {output_path}")
    except Exception as exc:
        print(f"警告: 保存特征-概率相关性结果失败: {exc}")

    return result_df


def compute_wav2vec_results_from_pair_names(pair_names, base_dir, output_dir):
    """
    根据pair_names重新计算wav2vec结果
    参照pair_feature_diff_extraction.py的逻辑，分别查找左右音频文件
    
    参数:
        pair_names: 音频对名称列表
        base_dir: 基础目录（按类别组织的音频文件，如clap_select）
        output_dir: 输出目录
    """
    from wav2vec_surprise_heatmap import Wav2VecSurpriseCalculator
    
    print("\n=== 根据pair_names重新计算wav2vec惊喜度 ===")
    print(f"找到 {len(pair_names)} 个音频对")
    print(f"基础目录: {base_dir}")
    
    model_name = "facebook/wav2vec2-base"
    frame_duration_ms = 25
    min_context_frames = 5
    max_context_seconds = 25
    
    print(f"\n正在加载Wav2Vec2模型: {model_name}")
    print("提示: 如果网络连接失败，请检查：")
    print("  1. 网络连接是否正常")
    print("  2. 是否需要使用代理/VPN")
    print("  3. 模型是否已下载到本地缓存（transformers会自动使用缓存）")
    print("  4. 可以尝试设置环境变量：HF_ENDPOINT=https://hf-mirror.com (使用镜像)")
    
    try:
        calculator = Wav2VecSurpriseCalculator(model_name=model_name)
    except Exception as e:
        print(f"\n错误: 无法加载Wav2Vec2模型: {e}")
        print("\n可能的解决方案：")
        print("1. 检查网络连接，或使用VPN/代理")
        print("2. 设置HuggingFace镜像（在代码开头添加）：")
        print("   import os")
        print("   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'")
        print("3. 手动下载模型到本地，然后指定本地路径")
        raise
    
    results = []
    processed_count = 0
    not_found_count = 0
    
    # 检查是否有已保存的中间结果
    checkpoint_file = os.path.join(output_dir, 'wav2vec_checkpoint.json')
    if os.path.exists(checkpoint_file):
        print(f"发现已保存的中间结果: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                loaded_results = checkpoint.get('results', [])
                processed_count = checkpoint.get('processed_count', 0)
                already_processed = set(checkpoint.get('processed_names', []))
                print(f"  已处理 {len(loaded_results)} 个音频对")
                results = loaded_results
        except Exception as e:
            print(f"  警告: 加载中间结果失败: {e}，从头开始")
            results = []
            processed_count = 0
            already_processed = set()
    else:
        already_processed = set()
    
    # 创建临时目录存放立体声音频
    temp_dir = os.path.join(output_dir, 'temp_stereo_audio')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存中间结果的频率（每处理多少个就保存一次）
    save_interval = 10
    
    try:
        for i, pair_name in enumerate(pair_names):
            # 如果已经处理过，跳过
            if pair_name in already_processed:
                continue
            if i < 3:  # 前3个打印详细信息
                print(f"\n处理 {i+1}/{len(pair_names)}: {pair_name}")
            else:
                if i % 10 == 0:  # 每10个打印一次进度
                    print(f"\n处理进度: {i+1}/{len(pair_names)}")
            
            # 解析pair_name并找到左右音频文件
            parse_result = parse_stereo_pair_name_for_audio(pair_name, base_dir)
            
            if not parse_result:
                if i < 3:
                    print(f"  警告: 无法解析pair_name或找不到音频文件")
                not_found_count += 1
                continue
            
            left_path, right_path, left_cat, left_id, right_cat, right_id = parse_result
            
            if i < 3:
                print(f"  找到左音频: {left_path}")
                print(f"  找到右音频: {right_path}")
            
            # 创建临时立体声音频文件
            temp_stereo_path = os.path.join(temp_dir, f"temp_{i:04d}.wav")
            try:
                create_stereo_audio_from_pair(left_path, right_path, temp_stereo_path)
            except Exception as e:
                if i < 3:
                    print(f"  错误: 创建立体声音频失败: {e}")
                not_found_count += 1
                continue
            
            # 使用wav2vec计算惊喜度
            try:
                result = calculator.process_stereo_audio_file(
                    temp_stereo_path,
                    frame_duration_ms=frame_duration_ms,
                    min_context_frames=min_context_frames,
                    max_context_seconds=max_context_seconds
                )
                
                if result:
                    # 更新文件名以匹配pair_name
                    result['audio_file'] = pair_name
                    results.append(result)
                    processed_count += 1
                    already_processed.add(pair_name)
                    if i < 3:
                        print(f"  ✓ 成功处理")
                    
                    # 定期保存中间结果
                    if processed_count % save_interval == 0:
                        checkpoint = {
                            'results': results,
                            'processed_count': processed_count,
                            'processed_names': list(already_processed)
                        }
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)
                        print(f"  [已保存中间结果: {processed_count}个]")
                else:
                    if i < 3:
                        print(f"  ✗ 处理失败")
                    not_found_count += 1
            except Exception as e:
                if i < 3:
                    print(f"  ✗ wav2vec处理失败: {e}")
                not_found_count += 1
                continue
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n清理临时文件: {temp_dir}")
        
        # 最后保存一次中间结果
        if results:
            checkpoint = {
                'results': results,
                'processed_count': processed_count,
                'processed_names': list(already_processed)
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n成功处理 {processed_count}/{len(pair_names)} 个音频对")
    print(f"未找到文件: {not_found_count} 个")
    
    # 保存结果
    if results:
        # 按所有前缀分组（nn_main, nn_sub, main, sub, nn_, rn_等）
        grouped_results = {}
        
        for result in results:
            audio_file = result.get('audio_file', '')
            # 确定分组前缀
            group_prefix = None
            prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_', 'nn_', 'rn_']
            for prefix in prefixes:
                if audio_file.startswith(prefix):
                    group_prefix = prefix.rstrip('_')
                    break
            
            if group_prefix is None:
                group_prefix = 'other'
            
            if group_prefix not in grouped_results:
                grouped_results[group_prefix] = []
            grouped_results[group_prefix].append(result)
        
        # 保存每个分组的结果
        for group_prefix, group_results in grouped_results.items():
            json_path = os.path.join(output_dir, f'wav2vec_surprise_raw_results_{group_prefix}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(group_results, f, ensure_ascii=False, indent=2, default=str)
            print(f"保存{group_prefix}结果: {len(group_results)} 个 -> {json_path}")
        
        print(f"结果已保存到: {output_dir}")
        
        # 返回所有结果（用于后续处理，不需要重新加载）
        return results
    else:
        print("警告: 没有成功处理任何音频对")
        return None


def extract_wav2vec_features(wav2vec_results):
    """
    从wav2vec结果中提取8个特征（4种方法×2种差值计算方式）
    
    参数:
        wav2vec_results: wav2vec结果列表
    
    返回:
        features_dict: 字典，键为音频文件名，值为特征字典
    """
    print("\n=== 提取wav2vec特征 ===")
    
    features_dict = {}
    methods = ['mse', 'cosine', 'representation_change', 'entropy']
    
    for result in wav2vec_results:
        audio_file = result.get('audio_file', '')
        if not audio_file:
            continue
        
        methods_results = result.get('methods_results', {})
        if not methods_results:
            continue
        
        features = {}
        
        for method in methods:
            if method not in methods_results:
                continue
            
            method_result = methods_results[method]
            left_surprise = method_result.get('left_surprise', [])
            right_surprise = method_result.get('right_surprise', [])
            
            if not left_surprise or not right_surprise:
                continue
            
            # 转换为numpy数组
            left_arr = np.array(left_surprise)
            right_arr = np.array(right_surprise)
            
            # 计算sum差值（已存在）
            left_sum = method_result.get('left_sum', np.sum(left_arr))
            right_sum = method_result.get('right_sum', np.sum(right_arr))
            sum_diff = left_sum - right_sum
            
            # 计算max差值
            left_max = np.max(left_arr)
            right_max = np.max(right_arr)
            max_diff = left_max - right_max
            
            features[f'{method}_sum_diff'] = sum_diff
            features[f'{method}_max_diff'] = max_diff
        
        if features:
            features_dict[audio_file] = features
    
    print(f"成功提取 {len(features_dict)} 个音频对的特征")
    return features_dict


def normalize_audio_filename(filename):
    """
    规范化音频文件名，用于匹配
    
    参数:
        filename: 音频文件名
    
    返回:
        normalized_name: 规范化后的名称
    """
    # 去掉可能的路径
    filename = os.path.basename(filename)
    
    # 去掉扩展名（可能有多重.wav）
    while filename.lower().endswith('.wav'):
        filename = filename[:-4]
    
    return filename


def parse_wav2vec_filename(filename):
    """
    专门解析wav2vec文件名格式
    格式示例: nn_Ambulance (siren)_1LA8BxMlXSY_2.wav+Helicopter_DSLO9BKHSEk_1.wav.wav
    
    参数:
        filename: wav2vec文件名
    
    返回:
        (left_cat, left_id, right_cat, right_id) 或 None
    """
    name = normalize_audio_filename(filename)
    
    # 去掉nn_或rn_前缀
    if name.startswith('nn_'):
        name = name[3:]
    elif name.startswith('rn_'):
        name = name[3:]
    
    # 按+分割
    if '+' not in name:
        return None
    
    left_part, right_part = name.split('+', 1)
    
    # 解析左侧：类别_ID.wav或类别_ID
    # 找到最后一个下划线，之前是类别，之后是ID
    left_last_us = left_part.rfind('_')
    if left_last_us == -1:
        return None
    left_cat = left_part[:left_last_us]
    left_id = left_part[left_last_us+1:]
    # 去掉.wav
    if left_id.endswith('.wav'):
        left_id = left_id[:-4]
    
    # 解析右侧：类别_ID.wav或类别_ID
    right_last_us = right_part.rfind('_')
    if right_last_us == -1:
        return None
    right_cat = right_part[:right_last_us]
    right_id = right_part[right_last_us+1:]
    # 去掉.wav
    if right_id.endswith('.wav'):
        right_id = right_id[:-4]
    
    return (left_cat.strip(), left_id.strip(), right_cat.strip(), right_id.strip())


def extract_category_and_id_from_pair_name(pair_name):
    """
    从pair_name中提取类别和ID（忽略实验类型前缀）
    格式示例: nn_main_Helicopter_W3u2hj1x7gY_2+Telephone bell ringing__xuq9rBndUE_1.wav
    
    参数:
        pair_name: 音频对名称
    
    返回:
        (left_cat, left_id, right_cat, right_id) 或 None
    """
    name = normalize_audio_filename(pair_name)
    
    # 去掉所有可能的前缀：nn_main_, nn_sub_, main_, sub_, nn_, rn_
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_', 'nn_', 'rn_']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    # 按+分割
    if '+' not in name:
        return None
    
    left_part, right_part = name.split('+', 1)
    
    # 解析左侧：类别_ID或类别_ID.wav
    # 找到最后一个下划线，之前是类别，之后是ID
    left_last_us = left_part.rfind('_')
    if left_last_us == -1:
        return None
    
    # 可能需要找倒数第二个下划线（如果ID包含下划线）
    # 但通常ID是最后一个下划线之后的内容
    left_cat = left_part[:left_last_us]
    left_id = left_part[left_last_us+1:]
    # 去掉.wav
    if left_id.endswith('.wav'):
        left_id = left_id[:-4]
    
    # 解析右侧：类别_ID或类别_ID.wav
    right_last_us = right_part.rfind('_')
    if right_last_us == -1:
        return None
    right_cat = right_part[:right_last_us]
    right_id = right_part[right_last_us+1:]
    # 去掉.wav
    if right_id.endswith('.wav'):
        right_id = right_id[:-4]
    
    return (left_cat.strip(), left_id.strip(), right_cat.strip(), right_id.strip())


def create_match_key(filename):
    """
    创建用于匹配的键（标准化格式，提取核心类别和ID信息，忽略所有前缀）
    
    参数:
        filename: 音频文件名
    
    返回:
        match_key: 用于匹配的键（格式：类别1_ID1+类别2_ID2，排序后）
    """
    # 去掉扩展名和路径
    name = normalize_audio_filename(filename)
    
    # 尝试多种解析方式
    result = None
    
    # 方式1: 尝试wav2vec格式解析（nn_类别_ID.wav+类别_ID.wav）
    wav2vec_result = parse_wav2vec_filename(filename)
    if wav2vec_result:
        result = wav2vec_result
    
    # 方式2: 尝试从pair_name提取（去掉所有前缀）
    if not result:
        pair_result = extract_category_and_id_from_pair_name(filename)
        if pair_result:
            result = pair_result
    
    # 方式3: 尝试标准parse_pair_name
    if not result:
        parse_result = parse_pair_name(name)
        if parse_result and parse_result[0]:
            left_cat, left_id, right_cat, right_id, exp_type = parse_result
            left_id_clean = left_id.replace('.wav', '').strip()
            right_id_clean = right_id.replace('.wav', '').strip()
            result = (left_cat.strip(), left_id_clean, right_cat.strip(), right_id_clean)
    
    # 如果解析成功，创建标准化匹配键
    if result:
        left_cat, left_id, right_cat, right_id = result
        # 标准化类别名称（去掉空格、括号、逗号，统一格式）
        left_cat_clean = left_cat.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
        right_cat_clean = right_cat.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
        # 标准化ID（去掉.wav，trim）
        left_id_clean = left_id.replace('.wav', '').strip()
        right_id_clean = right_id.replace('.wav', '').strip()
        
        # 返回主键（排序后的版本以保持一致性，不区分左右）
        parts = sorted([f"{left_cat_clean}_{left_id_clean}", f"{right_cat_clean}_{right_id_clean}"])
        return f"{parts[0]}+{parts[1]}".lower()
    
    # 如果都失败，返回规范化后的名称
    # 去掉所有可能的前缀
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_', 'nn_', 'rn_']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    return name.lower()


def match_wav2vec_with_probabilities(wav2vec_features, pair_names, probabilities):
    """
    匹配wav2vec特征数据与人类选择概率数据
    
    参数:
        wav2vec_features: wav2vec特征字典（键为音频文件名）
        pair_names: 音频对名称列表
        probabilities: 选择概率列表
    
    返回:
        matched_data: DataFrame，包含匹配后的数据
    """
    print("\n=== 匹配wav2vec特征与选择概率 ===")
    
    # 打印一些示例文件名用于调试
    print("\n调试信息：")
    if len(wav2vec_features) > 0:
        sample_wav2vec_names = list(wav2vec_features.keys())[:5]
        print(f"wav2vec文件名示例（前5个）:")
        for name in sample_wav2vec_names:
            print(f"  - {name}")
        # 尝试解析第一个文件名
        first_name = sample_wav2vec_names[0]
        parse_result = parse_pair_name(first_name)
        if parse_result and parse_result[0]:
            left_cat, left_id, right_cat, right_id, exp_type = parse_result
            print(f"  解析结果: {left_cat}_{left_id}+{right_cat}_{right_id}, exp_type={exp_type}")
        else:
            print(f"  无法解析文件名格式")
    
    if len(pair_names) > 0:
        print(f"\npair_names示例（前5个）:")
        for name in pair_names[:5]:
            print(f"  - {name}")
        # 尝试解析第一个pair_name
        first_name = pair_names[0]
        parse_result = parse_pair_name(first_name)
        if parse_result and parse_result[0]:
            left_cat, left_id, right_cat, right_id, exp_type = parse_result
            print(f"  解析结果: {left_cat}_{left_id}+{right_cat}_{right_id}, exp_type={exp_type}")
        else:
            print(f"  无法解析文件名格式")
    
    # 创建wav2vec特征的匹配键字典（使用多种键格式）
    wav2vec_keys = {}
    
    for filename, features in wav2vec_features.items():
        # 尝试多种匹配键格式
        keys_to_try = []
        
        # 1. 使用create_match_key（标准化键）
        match_key = create_match_key(filename)
        keys_to_try.append(match_key)
        
        # 2. 解析文件名，创建多种可能的键
        # 尝试多种解析方式
        result = None
        
        # 方式1: wav2vec格式
        wav2vec_result = parse_wav2vec_filename(filename)
        if wav2vec_result:
            result = wav2vec_result
        
        # 方式2: 标准parse_pair_name
        if not result:
            parse_result = parse_pair_name(filename)
            if parse_result and parse_result[0]:
                left_cat, left_id, right_cat, right_id, exp_type = parse_result
                left_id_clean = left_id.replace('.wav', '').strip()
                right_id_clean = right_id.replace('.wav', '').strip()
                result = (left_cat.strip(), left_id_clean, right_cat.strip(), right_id_clean)
        
        if result:
            left_cat, left_id, right_cat, right_id = result
            # 标准化类别名称（处理空格、括号、逗号）
            left_cat_clean = left_cat.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
            right_cat_clean = right_cat.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
            left_id_clean = left_id.replace('.wav', '').strip()
            right_id_clean = right_id.replace('.wav', '').strip()
            
            # 创建多种键格式
            # 格式1: 标准格式（排序后，标准化类别名）
            parts = sorted([f"{left_cat_clean}_{left_id_clean}", f"{right_cat_clean}_{right_id_clean}"])
            keys_to_try.append(f"{parts[0]}+{parts[1]}".lower())
            
            # 格式2: 原始顺序
            keys_to_try.append(f"{left_cat_clean}_{left_id_clean}+{right_cat_clean}_{right_id_clean}".lower())
            keys_to_try.append(f"{right_cat_clean}_{right_id_clean}+{left_cat_clean}_{left_id_clean}".lower())
            
            # 格式3: 带原始类别名称（保留空格和括号，但标准化ID）
            parts_orig = sorted([f"{left_cat}_{left_id_clean}", f"{right_cat}_{right_id_clean}"])
            keys_to_try.append(f"{parts_orig[0]}+{parts_orig[1]}".lower())
            
            keys_to_try.append(f"{left_cat}_{left_id_clean}+{right_cat}_{right_id_clean}".lower())
            keys_to_try.append(f"{right_cat}_{right_id_clean}+{left_cat}_{left_id_clean}".lower())
        
        # 将所有键都添加到字典中
        for key in keys_to_try:
            if key and key not in wav2vec_keys:
                wav2vec_keys[key] = (filename, features)
    
    print(f"wav2vec特征数量: {len(wav2vec_features)}")
    print(f"创建的匹配键数量: {len(wav2vec_keys)}")
    
    matched_records = []
    unmatched_count = 0
    unmatched_samples = []  # 保存未匹配的示例用于调试
    
    for i, pair_name in enumerate(pair_names):
        if i >= len(probabilities):
            break
        
        prob = probabilities[i]
        if np.isnan(prob):
            continue
        
        # 尝试多种匹配方式
        matched = False
        
        # 1. 使用create_match_key匹配
        match_key = create_match_key(pair_name)
        if match_key in wav2vec_keys:
            original_filename, features = wav2vec_keys[match_key]
            record = {
                'pair_name': pair_name,
                'wav2vec_filename': original_filename,
                'probability': prob,
                **features
            }
            matched_records.append(record)
            matched = True
        else:
            # 2. 解析后创建多种匹配键
            # 尝试多种解析方式
            result = None
            
            # 方式1: 从pair_name提取（去掉所有前缀）
            pair_result = extract_category_and_id_from_pair_name(pair_name)
            if pair_result:
                result = pair_result
            
            # 方式2: 标准parse_pair_name
            if not result:
                parse_result = parse_pair_name(pair_name)
                if parse_result and parse_result[0]:
                    left_cat, left_id, right_cat, right_id, exp_type = parse_result
                    left_id_clean = left_id.replace('.wav', '').strip()
                    right_id_clean = right_id.replace('.wav', '').strip()
                    result = (left_cat.strip(), left_id_clean, right_cat.strip(), right_id_clean)
            
            if result:
                left_cat, left_id, right_cat, right_id = result
                
                # 标准化类别名称和ID
                left_cat_clean = left_cat.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
                right_cat_clean = right_cat.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
                left_id_clean = left_id.replace('.wav', '').strip()
                right_id_clean = right_id.replace('.wav', '').strip()
                
                # 创建多种可能的匹配键
                possible_keys = []
                
                # 格式1: 排序后的标准格式（标准化类别名）
                parts = sorted([f"{left_cat_clean}_{left_id_clean}", f"{right_cat_clean}_{right_id_clean}"])
                possible_keys.append(f"{parts[0]}+{parts[1]}".lower())
                
                # 格式2: 原始顺序
                possible_keys.extend([
                    f"{left_cat_clean}_{left_id_clean}+{right_cat_clean}_{right_id_clean}".lower(),
                    f"{right_cat_clean}_{right_id_clean}+{left_cat_clean}_{left_id_clean}".lower(),
                ])
                
                # 格式3: 带原始类别名称
                parts_orig = sorted([f"{left_cat}_{left_id_clean}", f"{right_cat}_{right_id_clean}"])
                possible_keys.append(f"{parts_orig[0]}+{parts_orig[1]}".lower())
                
                possible_keys.extend([
                    f"{left_cat}_{left_id_clean}+{right_cat}_{right_id_clean}".lower(),
                    f"{right_cat}_{right_id_clean}+{left_cat}_{left_id_clean}".lower(),
                ])
                
                # 尝试匹配
                for possible_key in possible_keys:
                    if possible_key in wav2vec_keys:
                        original_filename, features = wav2vec_keys[possible_key]
                        record = {
                            'pair_name': pair_name,
                            'wav2vec_filename': original_filename,
                            'probability': prob,
                            **features
                        }
                        matched_records.append(record)
                        matched = True
                        break
        
        if not matched:
            unmatched_count += 1
            if len(unmatched_samples) < 5:  # 只保存前5个未匹配的示例
                unmatched_samples.append(pair_name)
    
    matched_df = pd.DataFrame(matched_records)
    print(f"\n成功匹配 {len(matched_df)} 个音频对")
    print(f"未匹配 {unmatched_count} 个音频对")
    
    if len(unmatched_samples) > 0:
        print(f"\n未匹配示例（前5个）:")
        for sample in unmatched_samples:
            print(f"  {sample}")
    
    if len(pair_names) > 0:
        print(f"\n匹配率: {len(matched_df) / len(pair_names) * 100:.2f}%")
    
    if len(matched_df) > 0:
        feature_cols = [col for col in matched_df.columns if col not in ['pair_name', 'wav2vec_filename', 'probability']]
        print(f"特征列: {feature_cols}")
    else:
        print("\n警告: 没有匹配到任何数据！")
        print("请检查文件名格式是否一致。")
        print("\n建议：")
        print("1. 检查wav2vec结果中的文件名格式")
        print("2. 检查pair_names中的文件名格式")
        print("3. 可能需要手动调整文件名匹配逻辑")
    
    return matched_df


def calculate_phi_coefficient(contingency_table):
    """
    计算Phi系数（2x2列联表的相关系数）
    
    参数:
        contingency_table: 2x2列联表 [[a, b], [c, d]]
                            其中 a=模型选A且人类选A, b=模型选A且人类选B
                                 c=模型选B且人类选A, d=模型选B且人类选B
    
    返回:
        phi: Phi系数值
    """
    a, b = contingency_table[0][0], contingency_table[0][1]
    c, d = contingency_table[1][0], contingency_table[1][1]
    
    # Phi系数计算公式: φ = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    
    if denominator == 0:
        return np.nan
    
    phi = numerator / denominator
    return phi


def check_human_preference_significance(selected_left_count, total_count, alpha=0.05):
    """
    检验人类对某个音频对的选择是否有显著偏好
    使用已统计的counts进行二项检验
    
    参数:
        selected_left_count: 选择左边的次数
        total_count: 总试验次数（左右平衡，每个被试2次，共9个被试=18次）
        alpha: 显著性水平（默认0.05）
    
    返回:
        (is_significant, p_value) 或 None
    """
    from scipy.stats import binomtest
    
    if total_count < 2:  # 至少需要2次选择才能做检验
        return None
    
    # 执行二项检验（双尾）：检验是否显著偏离0.5
    result = binomtest(selected_left_count, total_count, p=0.5, alternative='two-sided')
    p_value = result.pvalue
    is_significant = p_value < alpha
    
    return (is_significant, p_value)


def analyze_model_human_agreement(feature_values, probabilities, feature_name, sample_size=None, random_state=42):
    """
    分析模型与人类选择的一致性
    
    参数:
        feature_values: 特征差值数组 (左-右)
        probabilities: 人类选择左音频的概率数组
        feature_name: 特征名称
    
    返回:
        results_dict: 包含一致性和Phi系数的字典
    """
    # 去除NaN值
    valid_mask = ~(np.isnan(feature_values) | np.isnan(probabilities))
    feature_clean = feature_values[valid_mask]
    prob_clean = probabilities[valid_mask]
    
    if len(feature_clean) < 10:
        return None
    
    # 模型决策：差值 > 0 选左边(A)，差值 < 0 选右边(B)
    # 差值 = 0 的情况将被排除（平局）
    model_choose_left = feature_clean > 0
    model_choose_right = feature_clean < 0
    
    # 人类决策：probability > 0.5 选左边(A)，probability < 0.5 选右边(B)
    # probability = 0.5 的情况将被排除（平局）
    human_choose_left = prob_clean > 0.5
    human_choose_right = prob_clean < 0.5
    
    # 排除所有平局情况：
    # - 模型平局：差值 = 0（此时 model_choose_left 和 model_choose_right 都是 False）
    # - 人类平局：probability = 0.5（此时 human_choose_left 和 human_choose_right 都是 False）
    # 只保留模型和人类都有明确选择的样本
    valid_decision_mask = (model_choose_left | model_choose_right) & (human_choose_left | human_choose_right)
    available_indices = np.where(valid_decision_mask)[0]
    available_count = len(available_indices)
    model_decisions = model_choose_left[valid_decision_mask]
    human_decisions = human_choose_left[valid_decision_mask]
    sampled = False

    if sample_size is not None and len(model_decisions) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(model_decisions), size=sample_size, replace=False)
        model_decisions = model_decisions[sample_indices]
        human_decisions = human_decisions[sample_indices]
        sampled = True
    
    if len(model_decisions) < 10:
        return None
    
    # 构建2x2列联表
    #               人类偏爱A(左)    人类偏爱B(右)
    # 模型偏爱A(左)      a              b
    # 模型偏爱B(右)      c              d
    a = np.sum(model_decisions & human_decisions)  # 模型选左且人类选左
    b = np.sum(model_decisions & ~human_decisions)  # 模型选左且人类选右
    c = np.sum(~model_decisions & human_decisions)  # 模型选右且人类选左
    d = np.sum(~model_decisions & ~human_decisions)  # 模型选右且人类选右
    
    contingency_table = [[a, b], [c, d]]
    
    # 计算一致率
    total = a + b + c + d
    agreement_rate = (a + d) / total if total > 0 else 0.0
    
    # 计算Phi系数
    phi = calculate_phi_coefficient(contingency_table)
    
    # 计算卡方检验（用于Phi系数的显著性检验）
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    results_dict = {
        'feature_name': feature_name,
        'result_scope': 'sample' if sampled else 'full',
        'sample_size_requested': int(sample_size) if sample_size is not None else None,
        'sampled': bool(sampled),
        'sample_size_used': int(len(model_decisions)),
        'n_available': int(available_count),
        'n_total': total,
        'n_excluded': int(len(feature_clean) - available_count),
        'n_sampled_excluded': int(available_count - total) if sampled else 0,
        'contingency_table': contingency_table,
        'a': int(a),  # 模型选左且人类选左
        'b': int(b),  # 模型选左且人类选右
        'c': int(c),  # 模型选右且人类选左
        'd': int(d),  # 模型选右且人类选右
        'agreement_rate': float(agreement_rate),
        'phi_coefficient': float(phi),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }
    
    return results_dict


def fit_single_model(X, y, feature_name):
    """
    拟合单个OLS模型
    
    参数:
        X: 特征向量 (n_samples,)
        y: 目标变量 (n_samples,)
        feature_name: 特征名称
    
    返回:
        model: 拟合的OLS模型
        stats_dict: 统计信息字典
    """
    # 添加截距项
    X_with_const = sm.add_constant(X.reshape(-1, 1))
    
    # 拟合模型
    model = sm.OLS(y, X_with_const).fit()
    
    # 提取统计信息
    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()
    
    # 安全地访问参数（兼容numpy数组和pandas Series）
    intercept_idx = 0
    coef_idx = 1
    
    # 转换为numpy数组以确保兼容性
    if hasattr(params, 'values'):
        params = params.values
    if hasattr(pvalues, 'values'):
        pvalues = pvalues.values
    if hasattr(conf_int, 'values'):
        conf_int = conf_int.values
    
    stats_dict = {
        'feature_name': feature_name,
        'intercept': float(params[intercept_idx]),
        'intercept_pvalue': float(pvalues[intercept_idx]),
        'intercept_ci_lower': float(conf_int[intercept_idx, 0]),
        'intercept_ci_upper': float(conf_int[intercept_idx, 1]),
        'coefficient': float(params[coef_idx]),
        'coefficient_pvalue': float(pvalues[coef_idx]),
        'coefficient_ci_lower': float(conf_int[coef_idx, 0]),
        'coefficient_ci_upper': float(conf_int[coef_idx, 1]),
        'rsquared': float(model.rsquared),
        'rsquared_adj': float(model.rsquared_adj),
        'aic': float(model.aic),
        'bic': float(model.bic),
        'fvalue': float(model.fvalue),
        'f_pvalue': float(model.f_pvalue),
        'nobs': int(model.nobs)
    }
    
    return model, stats_dict


def cross_validate_model(X, y, feature_name, n_folds=5):
    """
    对模型进行5折交叉验证
    
    参数:
        X: 特征向量 (n_samples,)
        y: 目标变量 (n_samples,)
        feature_name: 特征名称
        n_folds: 折数
    
    返回:
        cv_results: 交叉验证结果字典
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_r2_scores = []
    cv_mse_scores = []
    cv_mae_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        X_train_const = sm.add_constant(X_train.reshape(-1, 1))
        X_test_const = sm.add_constant(X_test.reshape(-1, 1))
        
        model = sm.OLS(y_train, X_train_const).fit()
        
        # 预测
        y_pred = model.predict(X_test_const)
        
        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_r2_scores.append(r2)
        cv_mse_scores.append(mse)
        cv_mae_scores.append(mae)
    
    cv_results = {
        'feature_name': feature_name,
        'r2_mean': np.mean(cv_r2_scores),
        'r2_std': np.std(cv_r2_scores),
        'mse_mean': np.mean(cv_mse_scores),
        'mse_std': np.std(cv_mse_scores),
        'mae_mean': np.mean(cv_mae_scores),
        'mae_std': np.std(cv_mae_scores),
        'r2_scores': cv_r2_scores,
        'mse_scores': cv_mse_scores,
        'mae_scores': cv_mae_scores
    }
    
    return cv_results


def plot_predicted_vs_actual(y_true, y_pred, feature_name, output_path):
    """绘制预测值vs实际值散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    ax.set_xlabel('实际选择概率', fontsize=12)
    ax.set_ylabel('预测选择概率', fontsize=12)
    ax.set_title(f'{feature_name}\nR² = {r2:.4f}, r = {corr:.4f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_vs_predicted(y_pred, residuals, feature_name, output_path):
    """绘制残差vs预测值图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    ax.set_xlabel('预测选择概率', fontsize=12)
    ax.set_ylabel('残差', fontsize=12)
    ax.set_title(f'{feature_name} - 残差分析\nMean={residual_mean:.4f}, Std={residual_std:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    print("=== Wav2Vec2惊喜度回归分析 ===")
    
    # 设置路径
    project_root = os.getcwd()
    wav2vec_output_dir = r"D:\D\research\audioset下载\wav2vec_surprise_heatmaps"
    wav2vec_input_dir = r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120"
    regression_output_dir = os.path.join(project_root, 'wav2vec_regression_results')
    os.makedirs(regression_output_dir, exist_ok=True)
    
    # 1. 加载音频对列表（需要在计算wav2vec之前）
    print("\n步骤1: 加载音频对列表")
    pair_list_path = os.path.join(project_root, 'pair_list.npy')
    if not os.path.exists(pair_list_path):
        print(f"错误: 未找到pair_list.npy文件: {pair_list_path}")
        return
    
    pair_names = np.load(pair_list_path, allow_pickle=True)
    merged_results = merge_balanced_pairs_by_type(pair_names)
    
    # 合并所有类型（main, sub, nn_main, nn_sub）
    all_pair_names = []
    pair_experiment_type_map = {}
    for exp_type in ['main', 'sub', 'nn_main', 'nn_sub']:
        if exp_type in merged_results:
            for pair_name in merged_results[exp_type]['pair_names']:
                all_pair_names.append(pair_name)
                if pair_name not in pair_experiment_type_map:
                    pair_experiment_type_map[pair_name] = exp_type
    
    print(f"总共 {len(all_pair_names)} 个音频对（所有类型）")
    
    # 2. 直接根据pair_names重新计算wav2vec结果
    print("\n步骤2: 根据pair_names重新计算wav2vec惊喜度")
    
    # 基础目录（按类别组织的音频文件，参照pair_feature_diff_extraction.py）
    audio_base_dir = r"D:\D\research\audioset下载\clap_select"
    
    if not os.path.exists(audio_base_dir):
        print(f"错误: 音频基础目录不存在: {audio_base_dir}")
        print("请检查路径是否正确")
        return
    
    sample_audio_dir = os.path.join(regression_output_dir, 'sample_stereo_audio')
    saved_samples = save_sample_stereo_audios(all_pair_names, audio_base_dir, sample_audio_dir, sample_count=2)
    if saved_samples > 0:
        print(f"示例立体声音频已输出 {saved_samples} 个 -> {sample_audio_dir}")

    # 重新计算wav2vec结果（返回计算的结果，而不是从文件加载）
    wav2vec_results = compute_wav2vec_results_from_pair_names(all_pair_names, audio_base_dir, wav2vec_output_dir)
    if wav2vec_results is None or len(wav2vec_results) == 0:
        print("错误: 无法获取wav2vec结果")
        return
    
    # 3. 提取wav2vec特征
    print("\n步骤3: 提取wav2vec特征")
    wav2vec_features = extract_wav2vec_features(wav2vec_results)
    
    # 保存特征到文件（避免重新计算）
    features_file = os.path.join(regression_output_dir, 'wav2vec_features.json')
    with open(features_file, 'w', encoding='utf-8') as f:
        json.dump({k: {k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2 
                       for k2, v2 in v.items()} 
                  for k, v in wav2vec_features.items()}, 
                 f, ensure_ascii=False, indent=2)
    print(f"特征已保存到: {features_file}")
    
    if len(wav2vec_features) == 0:
        print("错误: 未能提取任何特征")
        return
    
    # 4. 加载人类决策数据
    print("\n步骤4: 加载人类决策数据")
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
    ]
    
    csv_files = [
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai2.csv",
        r"D:\D\research\audioset下载\experiment_output2\lironghua.csv",
        r"D:\D\research\audioset下载\experiment_output2\lironghua2.csv",
        r"D:\D\research\audioset下载\experiment_output2\mayunmiao_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\mayunmiao2.csv",
        r"D:\D\research\audioset下载\experiment_output2\ShangZiyang.csv",
        r"D:\D\research\audioset下载\experiment_output2\ShangZiyang1.csv",
        r"D:\D\research\audioset下载\experiment_output2\wjy1.csv",
        r"D:\D\research\audioset下载\experiment_output2\wjy_2.csv",
        r"D:\D\research\audioset下载\experiment_output2\LiuYaorui_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\Liu Yaorui2.csv"
    ]
    
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
    ] * 9
    
    all_trials = load_all_human_decisions(txt_files, csv_files, mapping_files)
    
    # 使用所有trials（不进行类型过滤）
    print(f"\n步骤4.5: 使用所有trials（所有实验类型）")
    print(f"trials数量: {len(all_trials)}")
    
    # 5. 计算选择概率和统计信息（用于显著性检验）
    print("\n步骤5: 计算选择概率和统计信息")
    probabilities = []
    selection_counts = []  # 存储(selected_left_count, total_count)对
    
    for pair_name in all_pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            probabilities.append(np.nan)
            selection_counts.append(None)
            continue
        
        left_cat, left_id, right_cat, right_id, exp_type = parse_result
        if pair_name not in pair_experiment_type_map and exp_type is not None:
            pair_experiment_type_map[pair_name] = exp_type
        audio_left = f"{left_cat}_{left_id}"
        audio_right = f"{right_cat}_{right_id}"
        
        # 统计：audio_left（pair_name中的左侧音频）被选择的次数和出现次数
        # 左右平衡：每个音频对会出现两次
        #   情况1: A左B右（trial_audio_left=A, trial_audio_right=B）
        #   情况2: B左A右（trial_audio_left=B, trial_audio_right=A）
        # 需要统计选择A的次数（不管A在左边还是右边）
        selected_audio_left_count = 0
        total_count = 0
        
        for trial in all_trials:
            trial_name = trial.get('original_name', '')
            trial_parse_result = parse_pair_name(trial_name)
            if trial_parse_result is None or trial_parse_result[0] is None:
                continue
            
            trial_left_cat, trial_left_id, trial_right_cat, trial_right_id, _ = trial_parse_result
            trial_audio_left = f"{trial_left_cat}_{trial_left_id}"
            trial_audio_right = f"{trial_right_cat}_{trial_right_id}"
            
            # 检查这个trial是否为AB组合对（两种情况都要统计）
            is_match = False
            audio_left_in_trial_left = False  # audio_left是否在trial的左边
            
            if trial_audio_left == audio_left and trial_audio_right == audio_right:
                # 情况1: A左B右（pair_name中的A在左边）
                is_match = True
                audio_left_in_trial_left = True
            elif trial_audio_left == audio_right and trial_audio_right == audio_left:
                # 情况2: B左A右（pair_name中的A在右边）
                is_match = True
                audio_left_in_trial_left = False
            
            if is_match:
                total_count += 1
                # 统计是否选择了audio_left（需要注意方向）
                # 情况1（A左B右）: response==1表示选左边，即选择了A
                # 情况2（B左A右）: response==2表示选右边，即选择了A
                if audio_left_in_trial_left:
                    # A在左边，response==1表示选择了A
                    if trial['response'] == 1:
                        selected_audio_left_count += 1
                else:
                    # A在右边，response==2表示选择了A
                    if trial['response'] == 2:
                        selected_audio_left_count += 1
        
        prob = selected_audio_left_count / total_count if total_count > 0 else 0.5
        probabilities.append(prob)
        selection_counts.append((selected_audio_left_count, total_count))
    
    probabilities = np.array(probabilities)
    print(f"计算完成: {len(probabilities)} 个概率值")
    print(f"概率范围: [{np.nanmin(probabilities):.3f}, {np.nanmax(probabilities):.3f}]")
    print(f"概率均值: {np.nanmean(probabilities):.3f}")
    
    # 6. 匹配数据（wav2vec结果已经使用正确的pair_name，直接匹配即可）
    print("\n步骤6: 匹配wav2vec特征与选择概率")
    
    # 由于wav2vec结果已经使用正确的pair_name，直接匹配
    matched_records = []
    for i, pair_name in enumerate(all_pair_names):
        if i >= len(probabilities):
            break
        
        prob = probabilities[i]
        if np.isnan(prob):
            continue
        
        # 获取该音频对的统计信息
        counts = selection_counts[i]
        if counts is None:
            continue
        selected_audio_left_count, total_count = counts

        experiment_type = pair_experiment_type_map.get(pair_name)
        if experiment_type is None:
            parsed = parse_pair_name(pair_name)
            if parsed and parsed[4]:
                experiment_type = parsed[4]
        
        # 直接使用pair_name匹配（wav2vec结果中的audio_file字段就是pair_name）
        if pair_name in wav2vec_features:
            features = wav2vec_features[pair_name]
            record = {
                'pair_name': pair_name,
                'experiment_type': experiment_type,
                'probability': prob,
                'selected_left_count': selected_audio_left_count,
                'total_count': total_count,
                **features
            }
            matched_records.append(record)
        else:
            # 如果直接匹配失败，尝试规范化后的名称
            pair_name_normalized = normalize_audio_filename(pair_name)
            if pair_name_normalized in wav2vec_features:
                features = wav2vec_features[pair_name_normalized]
                record = {
                    'pair_name': pair_name,
                    'experiment_type': experiment_type,
                    'probability': prob,
                    'selected_left_count': selected_audio_left_count,
                    'total_count': total_count,
                    **features
                }
                matched_records.append(record)
    
    matched_df = pd.DataFrame(matched_records)
    print(f"成功匹配 {len(matched_df)} 个音频对")

    loudness_std_map = load_loudness_std_diff_map(project_root)
    if loudness_std_map:
        matched_df['loudness_std_diff'] = matched_df['pair_name'].map(lambda x: loudness_std_map.get(str(x)))
        missing_loudness = matched_df['loudness_std_diff'].isna().sum()
        if missing_loudness > 0:
            print(f"警告: 有 {missing_loudness} 个音频对缺少响度std差值")
    else:
        matched_df['loudness_std_diff'] = np.nan

    matched_df_all = matched_df.copy()

    # 保存匹配后的数据（包含响度std差值）
    matched_data_file = os.path.join(regression_output_dir, 'matched_data.csv')
    matched_df.to_csv(matched_data_file, index=False, encoding='utf-8-sig')
    print(f"匹配数据已保存到: {matched_data_file}")

    main_sub_types = {'main', 'sub'}
    matched_df_main_sub_all = matched_df_all[matched_df_all['experiment_type'].isin(main_sub_types)].copy()
    if not matched_df_main_sub_all.empty:
        main_sub_all_file = os.path.join(regression_output_dir, 'matched_data_main_sub.csv')
        matched_df_main_sub_all.to_csv(main_sub_all_file, index=False, encoding='utf-8-sig')
        print(f"main+sub匹配数据已保存到: {main_sub_all_file}")
    else:
        print("警告: main+sub 匹配数据为空，检查实验类型标记是否正确")

    nn_types = {'nn_main', 'nn_sub'}
    matched_df_nn_all = matched_df_all[matched_df_all['experiment_type'].isin(nn_types)].copy()
    if not matched_df_nn_all.empty:
        nn_all_file = os.path.join(regression_output_dir, 'matched_data_nn_main_sub.csv')
        matched_df_nn_all.to_csv(nn_all_file, index=False, encoding='utf-8-sig')
        print(f"nn_main+nn_sub匹配数据已保存到: {nn_all_file}")
    else:
        print("警告: nn_main+nn_sub 匹配数据为空")

    correlation_output = os.path.join(regression_output_dir, 'wav2vec_loudness_std_correlations.csv')
    print("\n附加分析: 计算Wav2Vec特征与响度std差值的相关性（使用所有音频对）")
    analyze_wav2vec_loudness_correlation(matched_df_all, WAV2VEC_FEATURE_NAMES, correlation_output)

    prob_corr_output = os.path.join(regression_output_dir, 'wav2vec_probability_correlations.csv')
    print("附加分析: 计算Wav2Vec特征与选择概率的相关性（使用所有音频对）")
    analyze_feature_probability_correlation(matched_df_all, WAV2VEC_FEATURE_NAMES, 'probability', prob_corr_output)
    
    if len(matched_df) == 0:
        print("错误: 未能匹配任何数据")
        print("检查wav2vec结果中的audio_file字段是否与pair_names一致")
        if len(wav2vec_features) > 0:
            sample_keys = list(wav2vec_features.keys())[:3]
            print(f"wav2vec特征键示例: {sample_keys}")
            print(f"pair_names示例: {all_pair_names[:3]}")
        return
    
    # 6.5. 筛选：只保留人类选择有显著偏好的音频对（选择比例 > 60%）
    print("\n步骤6.5: 筛选人类选择有显著偏好的音频对（选择比例 > 60%）")
    significant_pair_names = []
    significant_results = []
    
    for i, row in matched_df.iterrows():
        pair_name = row['pair_name']
        prob = row['probability']
        selected_left_count = row.get('selected_left_count', 0)
        total_count = row.get('total_count', 0)
        
        # 筛选条件：选择比例 > 60% 或 < 40%
        # 即：选择左边的比例 > 60% 或 < 40%（选择右边的比例 > 60%）
        if prob > 0.6 or prob < 0.4:
            significant_pair_names.append(pair_name)
            
            # 可选：计算p值用于参考（不用于筛选）
            test_result = check_human_preference_significance(selected_left_count, total_count, alpha=0.05)
            p_value = test_result[1] if test_result else np.nan
            
            significant_results.append({
                'pair_name': pair_name,
                'probability': prob,
                'selected_left': selected_left_count,
                'total_trials': total_count,
                'p_value': p_value,
                'preference_direction': 'left' if prob > 0.6 else 'right'
            })
    
    print(f"原始音频对数量: {len(matched_df)}")
    print(f"人类选择有显著偏好的音频对数量: {len(significant_pair_names)} (选择比例 > 60% 或 < 40%)")
    print(f"筛选比例: {len(significant_pair_names)/len(matched_df)*100:.1f}%")
    
    # 只保留显著偏好的音频对
    matched_df = matched_df[matched_df['pair_name'].isin(significant_pair_names)].copy()
    print(f"筛选后剩余音频对数量: {len(matched_df)}")
    
    # 保存筛选结果
    significant_df = pd.DataFrame(significant_results)
    significant_file = os.path.join(regression_output_dir, 'significant_human_preferences.csv')
    significant_df.to_csv(significant_file, index=False, encoding='utf-8-sig')
    print(f"显著偏好结果已保存到: {significant_file}")
    
    if len(matched_df) == 0:
        print("错误: 筛选后没有剩余的音频对")
        print("可能原因：")
        print("  1. 人类选择没有显著偏好（p >= 0.05）")
        print("  2. 样本量太小，无法进行统计检验")
        return
    
    # 保存筛选后的匹配数据
    filtered_matched_file = os.path.join(regression_output_dir, 'matched_data_filtered.csv')
    matched_df.to_csv(filtered_matched_file, index=False, encoding='utf-8-sig')
    print(f"筛选后的匹配数据已保存到: {filtered_matched_file}")

    matched_df_main_sub_filtered = matched_df[matched_df['experiment_type'].isin(main_sub_types)].copy()
    if not matched_df_main_sub_filtered.empty:
        filtered_main_sub_file = os.path.join(regression_output_dir, 'matched_data_main_sub_filtered.csv')
        matched_df_main_sub_filtered.to_csv(filtered_main_sub_file, index=False, encoding='utf-8-sig')
        print(f"main+sub筛选后的匹配数据已保存到: {filtered_main_sub_file}")
    else:
        print("警告: main+sub 筛选后的匹配数据为空")

    matched_df_nn_filtered = matched_df[matched_df['experiment_type'].isin(nn_types)].copy()
    if not matched_df_nn_filtered.empty:
        filtered_nn_file = os.path.join(regression_output_dir, 'matched_data_nn_main_sub_filtered.csv')
        matched_df_nn_filtered.to_csv(filtered_nn_file, index=False, encoding='utf-8-sig')
        print(f"nn_main+nn_sub筛选后的匹配数据已保存到: {filtered_nn_file}")
    else:
        print("警告: nn_main+nn_sub 筛选后的匹配数据为空")

    if not matched_df_main_sub_filtered.empty:
        significant_df_main_sub = significant_df[significant_df['pair_name'].isin(matched_df_main_sub_filtered['pair_name'])].copy()
        if not significant_df_main_sub.empty:
            significant_main_sub_file = os.path.join(regression_output_dir, 'significant_human_preferences_main_sub.csv')
            significant_df_main_sub.to_csv(significant_main_sub_file, index=False, encoding='utf-8-sig')
            print(f"main+sub显著偏好结果已保存到: {significant_main_sub_file}")

    if not matched_df_nn_filtered.empty:
        significant_df_nn = significant_df[significant_df['pair_name'].isin(matched_df_nn_filtered['pair_name'])].copy()
        if not significant_df_nn.empty:
            significant_nn_file = os.path.join(regression_output_dir, 'significant_human_preferences_nn_main_sub.csv')
            significant_df_nn.to_csv(significant_nn_file, index=False, encoding='utf-8-sig')
            print(f"nn_main+nn_sub显著偏好结果已保存到: {significant_nn_file}")
    
    # 7. 使用特征差值拟合选择概率（筛选后的数据）
    print("\n步骤7: 使用Wav2Vec2特征差值拟合选择概率（筛选后的数据）")
    
    feature_names = WAV2VEC_FEATURE_NAMES
    
    y = matched_df['probability'].values
    all_regression_results = []
    
    for feature_name in feature_names:
        if feature_name not in matched_df.columns:
            print(f"警告: 特征 {feature_name} 不存在，跳过")
            continue
        
        print(f"\n拟合模型: {feature_name}")
        X = matched_df[feature_name].values
        
        # 去除NaN值
        valid_mask = ~(np.isnan(X) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 10:
            print(f"警告: {feature_name} 有效样本数太少 ({len(X_clean)} < 10)，跳过")
            continue
        
        # 标准化特征（Z-score标准化）
        X_mean = np.mean(X_clean)
        X_std = np.std(X_clean)
        if X_std == 0:
            print(f"警告: {feature_name} 标准差为0，跳过")
            continue
        X_standardized = (X_clean - X_mean) / X_std
        
        print(f"  特征统计: mean={X_mean:.4f}, std={X_std:.4f}")
        
        # 拟合OLS模型（使用标准化后的特征）
        X_df = pd.DataFrame(X_standardized.reshape(-1, 1), columns=[feature_name])
        X_with_const = sm.add_constant(X_df)
        model = sm.OLS(y_clean, X_with_const).fit()
        
        # 提取统计信息
        params = model.params
        pvalues = model.pvalues
        conf_int = model.conf_int()
        
        # 转换为numpy数组以确保兼容性
        if hasattr(params, 'values'):
            params = params.values
        if hasattr(pvalues, 'values'):
            pvalues = pvalues.values
        if hasattr(conf_int, 'values'):
            conf_int = conf_int.values
        
        intercept_idx = 0
        coef_idx = 1
        
        result_dict = {
            'feature_name': feature_name,
            'n_samples': len(y_clean),
            'feature_mean': float(X_mean),
            'feature_std': float(X_std),
            'intercept': float(params[intercept_idx]),
            'intercept_pvalue': float(pvalues[intercept_idx]),
            'intercept_ci_lower': float(conf_int[intercept_idx, 0]),
            'intercept_ci_upper': float(conf_int[intercept_idx, 1]),
            'coefficient': float(params[coef_idx]),
            'coefficient_pvalue': float(pvalues[coef_idx]),
            'coefficient_ci_lower': float(conf_int[coef_idx, 0]),
            'coefficient_ci_upper': float(conf_int[coef_idx, 1]),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj),
            'aic': float(model.aic),
            'bic': float(model.bic),
            'fvalue': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue)
        }
        
        all_regression_results.append(result_dict)
        
        print(f"  R² = {result_dict['rsquared']:.4f}")
        print(f"  系数（标准化后）= {result_dict['coefficient']:.4f}, p = {result_dict['coefficient_pvalue']:.4f}")
    
    # 保存回归结果
    if all_regression_results:
        regression_df = pd.DataFrame(all_regression_results)
        regression_path = os.path.join(regression_output_dir, 'wav2vec_regression_results.csv')
        regression_df.to_csv(regression_path, index=False, encoding='utf-8-sig')
        print(f"\n回归结果已保存: {regression_path}")
    
    # 8. 分析模型与人类选择的一致性
    print("\n步骤8: 分析模型与人类选择的一致性")
    
    feature_names = [
        'mse_sum_diff',
        'mse_max_diff',
        'cosine_sum_diff',
        'cosine_max_diff',
        'representation_change_sum_diff',
        'representation_change_max_diff',
        'entropy_sum_diff',
        'entropy_max_diff'
    ]
    
    dataset_candidates = [
        ('main_sub_all', matched_df_main_sub_all),
        ('main_sub_filtered', matched_df_main_sub_filtered),
        ('nn_main_sub_all', matched_df_nn_all),
        ('nn_main_sub_filtered', matched_df_nn_filtered),
    ]
    analysis_datasets = [(label, df_scope) for label, df_scope in dataset_candidates if df_scope is not None and not df_scope.empty]

    all_agreement_results = []

    def print_agreement_details(result):
        scope_label = result.get('analysis_scope', 'unknown')
        print(f"  [范围: {scope_label}] 列联表:")
        print(f"           人类偏爱左(>A)    人类偏爱右(>B)")
        print(f"  模型偏爱左(>A)    {result['a']:4d}            {result['b']:4d}")
        print(f"  模型偏爱右(>B)    {result['c']:4d}            {result['d']:4d}")
        print(f"  使用样本数: {result['n_total']} (可用: {result['n_available']})")
        print(f"  排除平局样本数: {result['n_excluded']}")
        print(f"  一致率: {result['agreement_rate']:.4f} ({result['a']+result['d']}/{result['n_total']})")
        print(f"  Phi系数: {result['phi_coefficient']:.4f}")
        print(f"  卡方检验: χ² = {result['chi2']:.4f}, p = {result['p_value']:.4f}")
        print(f"  显著性: {'是' if result['significant'] else '否'} (p < 0.05)")
    
    if not analysis_datasets:
        print("警告: 未找到可用的数据集进行一致性分析")
    else:
        for scope_label, df_scope in analysis_datasets:
            probabilities_scope = df_scope['probability'].values
            print(f"\n=== 分析范围: {scope_label} ===")
            for feature_name in feature_names:
                if feature_name not in df_scope.columns:
                    print(f"警告: 特征 {feature_name} 不存在，跳过")
                    continue
                
                print(f"\n处理特征: {feature_name}")
                feature_values = df_scope[feature_name].values
                
                result = analyze_model_human_agreement(feature_values, probabilities_scope, feature_name)
                if result is None:
                    print(f"  警告: {feature_name} 有效样本数不足，跳过")
                    continue
                
                result['analysis_scope'] = scope_label
                result['result_scope'] = 'full'
                all_agreement_results.append(result)
                print_agreement_details(result)
            
            if all_agreement_results:
                results_df = pd.DataFrame(all_agreement_results)
                results_path = os.path.join(regression_output_dir, 'model_human_agreement_results.csv')
                results_df_save = results_df.copy()
                results_df_save['contingency_table'] = results_df_save['contingency_table'].apply(str)
                results_df_save.to_csv(results_path, index=False, encoding='utf-8-sig')
                print(f"  [已保存中间结果: {scope_label}]")
    
    # 9. 保存最终结果
    print("\n步骤9: 保存最终结果")
    
    if len(all_agreement_results) == 0:
        print("错误: 没有成功分析任何特征")
        return
    
    results_df = pd.DataFrame(all_agreement_results)
    results_path = os.path.join(regression_output_dir, 'model_human_agreement_results.csv')
    # 将列联表转换为字符串以便保存
    results_df_save = results_df.copy()
    results_df_save['contingency_table'] = results_df_save['contingency_table'].apply(str)
    results_df_save.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"一致性分析结果已保存: {results_path}")
    
    # 保存文本摘要
    summary_path = os.path.join(regression_output_dir, 'model_human_agreement_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Wav2Vec2模型与人类选择一致性分析结果 ===\n\n")
        f.write(f"总共分析了 {len(all_agreement_results)} 个特征\n")
        f.write(f"匹配的音频对数量: {len(matched_df)}\n\n")
        f.write("决策规则:\n")
        f.write("  模型: 特征差值(左-右) > 0 选左边，< 0 选右边\n")
        f.write("  人类: 选择概率 > 0.5 选左边，< 0.5 选右边\n\n")
        
        for i, result in enumerate(all_agreement_results):
            f.write(f"\n特征 {i+1}: {result['feature_name']}\n")
            scope_label = '随机45样本' if result.get('result_scope') == 'sample' else '全量样本'
            f.write(f"  分析范围: {scope_label}\n")
            f.write(f"  列联表:\n")
            f.write(f"               人类偏爱左(>A)    人类偏爱右(>B)\n")
            f.write(f"  模型偏爱左(>A)    {result['a']:4d}            {result['b']:4d}\n")
            f.write(f"  模型偏爱右(>B)    {result['c']:4d}            {result['d']:4d}\n")
            f.write(f"  使用样本数: {result['n_total']} (可用: {result['n_available']}, 额外抽样排除: {result['n_sampled_excluded']})\n")
            f.write(f"  排除平局样本数: {result['n_excluded']}\n")
            f.write(f"  一致率: {result['agreement_rate']:.4f} ({result['a']+result['d']}/{result['n_total']})\n")
            f.write(f"  Phi系数: {result['phi_coefficient']:.4f}\n")
            f.write(f"  卡方检验: χ² = {result['chi2']:.4f}, p = {result['p_value']:.4f}\n")
            f.write(f"  显著性: {'是' if result['significant'] else '否'} (p < 0.05)\n")
    
    print(f"文本摘要已保存: {summary_path}")
    
    # 打印汇总
    print("\n=== 汇总结果 ===")
    print(f"{'特征名称':<30} {'范围':<10} {'一致率':<10} {'Phi系数':<10} {'p值':<10} {'显著性':<10}")
    print("-" * 70)
    for result in all_agreement_results:
        sig_mark = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        scope_label = result.get('analysis_scope', 'unknown')
        print(f"{result['feature_name']:<30} {scope_label:<20} {result['agreement_rate']:<10.4f} {result['phi_coefficient']:<10.4f} {result['p_value']:<10.4f} {sig_mark}")
    
    scope_display_mapping = {
        'main_sub_all': 'main+sub 全部',
        'main_sub_filtered': 'main+sub 筛选后',
        'nn_main_sub_all': 'nn_main+nn_sub 全部',
        'nn_main_sub_filtered': 'nn_main+nn_sub 筛选后'
    }

    summary_records = []
    for result in all_agreement_results:
        scope_key = result.get('analysis_scope', 'unknown')
        summary_records.append({
            '特征名称': result['feature_name'],
            '范围': scope_display_mapping.get(scope_key, scope_key),
            '一致率': round(result['agreement_rate'], 4) if result['agreement_rate'] is not None else np.nan,
            'Phi系数': round(result['phi_coefficient'], 4) if result['phi_coefficient'] is not None else np.nan,
            'p值': round(result['p_value'], 4) if result['p_value'] is not None else np.nan,
            '显著性': '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else ''
        })

    if summary_records:
        summary_df = pd.DataFrame(summary_records, columns=['特征名称', '范围', '一致率', 'Phi系数', 'p值', '显著性'])
        summary_csv_path = os.path.join(regression_output_dir, 'model_human_agreement_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"一致性分析摘要表已保存: {summary_csv_path}")

    print("\n=== 分析完成 ===")


if __name__ == '__main__':
    main()

