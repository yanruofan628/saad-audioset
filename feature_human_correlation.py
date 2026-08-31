#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征与人类决策相关性分析
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict
from scipy.stats import ttest_1samp

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_or_extract_features(project_root):
    """检测并加载已有特征数据，若不存在则提示"""
    raw_path = os.path.join(project_root, 'pair_feature_diff_raw.npy')
    zscore_path = os.path.join(project_root, 'pair_feature_diff_zscore.npy')
    list_path = os.path.join(project_root, 'pair_list.npy')
    
    if os.path.exists(raw_path) and os.path.exists(zscore_path) and os.path.exists(list_path):
        print("检测到已有特征文件，正在加载...")
        diff_raw = np.load(raw_path)
        diff_zscore = np.load(zscore_path)
        pair_names = np.load(list_path, allow_pickle=True)
        print(f"成功加载: {diff_raw.shape[1]} 个音频对, {diff_raw.shape[0]} 个特征")
        return diff_raw, diff_zscore, pair_names
    else:
        raise FileNotFoundError("未找到特征文件，请先运行 pair_feature_diff_extraction.py")


def load_rating_scores(rating_csv_path):
    """加载评分数据，创建子类别到评分的映射字典"""
    if not os.path.exists(rating_csv_path):
        print(f"警告: 评分文件不存在: {rating_csv_path}")
        return None
    
    try:
        df = pd.read_csv(rating_csv_path, encoding='utf-8-sig')
        print(f"成功加载评分数据: {len(df)} 个子类别")
        
        # 创建子类别名称到评分的字典
        rating_dict = {}
        for _, row in df.iterrows():
            subcategory = str(row['subcategory']).strip()
            if pd.isna(subcategory) or subcategory == '':
                continue
            
            rating_dict[subcategory] = {
                'familiarity': float(row['familiarity']) if not pd.isna(row['familiarity']) else np.nan,
                'urgency': float(row['urgency']) if not pd.isna(row['urgency']) else np.nan,
                'liking': float(row['liking']) if not pd.isna(row['liking']) else np.nan
            }
        
        print(f"成功创建评分字典: {len(rating_dict)} 个有效子类别")
        return rating_dict
    except Exception as e:
        print(f"加载评分数据失败: {e}")
        return None


def extract_rating_features(pair_names, rating_dict):
    """提取评分特征：通过解析音频对获取左右子类别，查找评分并计算差值（左-右）"""
    if rating_dict is None:
        print("警告: 评分字典为空，无法提取评分特征")
        return None
    
    n_pairs = len(pair_names)
    rating_features = np.full((3, n_pairs), np.nan, dtype=float)  # 3个特征：familiarity, urgency, liking
    
    missing_count = 0
    found_count = 0
    
    for idx, pair_name in enumerate(pair_names):
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            missing_count += 1
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        
        # 查找左右子类别的评分
        left_ratings = rating_dict.get(left_cat, None)
        right_ratings = rating_dict.get(right_cat, None)
        
        if left_ratings is None or right_ratings is None:
            missing_count += 1
            continue
        
        # 计算差值：left - right
        familiarity_diff = left_ratings['familiarity'] - right_ratings['familiarity']
        urgency_diff = left_ratings['urgency'] - right_ratings['urgency']
        liking_diff = left_ratings['liking'] - right_ratings['liking']
        
        rating_features[0, idx] = familiarity_diff
        rating_features[1, idx] = urgency_diff
        rating_features[2, idx] = liking_diff
        
        found_count += 1
    
    print(f"评分特征提取完成: {found_count} 个成功, {missing_count} 个缺失")
    return rating_features


def zscore_per_feature(matrix):
    """对矩阵的每一行进行z-score标准化"""
    out = matrix.copy().astype(float)
    for i in range(out.shape[0]):
        row = out[i]
        mask = ~np.isnan(row)
        if not np.any(mask):
            continue
        m = np.mean(row[mask])
        s = np.std(row[mask])
        if s == 0:
            out[i, mask] = 0.0
        else:
            out[i, mask] = (row[mask] - m) / s
    return out


def parse_pair_name(filename):
    """解析音频对名称，提取左右音频标识和实验类型"""
    # 去掉扩展名
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]
    
    # 切分左右
    if '+' not in filename:
        return None, None, None, None, None
    
    left_part, right_part = filename.split('+', 1)
    
    # 提取实验类型
    experiment_type = None
    prefixes = ['nn_main_', 'nn_sub_', 'main_', 'sub_']
    for p in prefixes:
        if left_part.startswith(p) or right_part.startswith(p):
            experiment_type = p.rstrip('_')
            break
    
    # 去掉前缀
    for p in prefixes:
        if left_part.startswith(p):
            left_part = left_part[len(p):]
            break
    for p in prefixes:
        if right_part.startswith(p):
            right_part = right_part[len(p):]
            break
    
    # 提取类别和ID
    def extract_cat_id(part):
        # 找到最后一个下划线之前的第二个下划线
        last_us = part.rfind('_')
        if last_us == -1:
            return None, None
        second_last = part.rfind('_', 0, last_us)
        if second_last == -1:
            return None, None
        cat = part[:second_last]
        audio_id = part[second_last+1:]
        return cat, audio_id
    
    left_cat, left_id = extract_cat_id(left_part)
    right_cat, right_id = extract_cat_id(right_part)
    
    if left_cat is None or right_cat is None:
        return None, None, None, None, None
    
    return left_cat, left_id, right_cat, right_id, experiment_type


def merge_balanced_pairs_by_type(pair_names):
    """按实验类型分组合并左右平衡设计的音频对（288→144）"""
    print("\n=== 按实验类型合并左右平衡设计的音频对 ===")
    
    # 按实验类型分组
    type_groups = {'nn_main': [], 'nn_sub': [], 'main': [], 'sub': []}
    
    for idx, pair_name in enumerate(pair_names):
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[4] is None:
            print(f"警告: 无法解析实验类型 {pair_name}")
            continue
        
        left_cat, left_id, right_cat, right_id, experiment_type = parse_result
        type_groups[experiment_type].append((idx, pair_name))
    
    # 对每个类型进行合并
    merged_results = {}
    
    for exp_type, pairs in type_groups.items():
        if len(pairs) == 0:
            continue
            
        print(f"\n处理 {exp_type} 类型: {len(pairs)} 个音频对")
        
        unique_pairs = {}
        merged_indices_list = []
        merged_pair_names_list = []
        
        for idx, pair_name in pairs:
            left_cat, left_id, right_cat, right_id, _ = parse_pair_name(pair_name)
            
            # 创建规范化键（排序确保唯一性）
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            key = tuple(sorted([audio_a, audio_b]))
            
            if key not in unique_pairs:
                # 第一次出现，记录
                unique_pairs[key] = idx
                merged_indices_list.append(idx)
                merged_pair_names_list.append(pair_name)
            else:
                # 第二次出现（平衡配对），跳过
                pass
        
        print(f"  合并后唯一对数: {len(unique_pairs)}")
        
        merged_results[exp_type] = {
            'indices': merged_indices_list,
            'pair_names': merged_pair_names_list,
            'unique_pairs': unique_pairs
        }
    
    return merged_results


def parse_experiment_data_txt(file_path):
    """解析TXT格式的实验数据文件"""
    trials = []
    encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法读取文件: {file_path}")
    
    for i, line in enumerate(lines):
        if line.strip().startswith('Level: 3') and i+1 < len(lines) and '*** LogFrame Start ***' in lines[i+1]:
            trial_data = {}
            for j in range(i+1, min(i+20, len(lines))):
                current_line = lines[j].strip()
                if 'wavfile:' in current_line:
                    trial_data['wavfile'] = current_line.split('wavfile:')[1].strip()
                elif 'ImageDisplay1.RESP:' in current_line:
                    trial_data['response'] = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                elif '*** LogFrame End ***' in current_line:
                    break
            if 'wavfile' in trial_data and 'response' in trial_data:
                trials.append(trial_data)
    
    return trials


def parse_experiment_data_csv(file_path):
    """解析CSV格式的实验数据文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取CSV失败 {file_path}: {e}")
        return []
    
    trials = []
    for _, row in df.iterrows():
        audio_file = row['音频文件']
        choice = row['选择']
        
        if choice.lower() == 'left':
            response = 1
        elif choice.lower() == 'right':
            response = 2
        else:
            continue
        
        trials.append({
            'wavfile': audio_file,
            'response': response
        })
    
    return trials


def parse_audio_mapping(file_path):
    """解析音频映射文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        for _, row in df.iterrows():
            mapping[row['new_name']] = row['original_name']
        return mapping
    except Exception as e:
        print(f"读取映射文件失败 {file_path}: {e}")
        return {}


def load_all_human_decisions(txt_files, csv_files, mapping_files):
    """加载所有人类决策数据，并标注被试ID"""
    print("\n=== 加载人类决策数据 ===")
    
    all_trials = []
    
    # 处理TXT文件
    for txt_file, mapping_file in zip(txt_files, mapping_files[:len(txt_files)]):
        if not os.path.exists(txt_file):
            print(f"TXT文件不存在: {txt_file}")
            continue
        
        # 从文件名提取被试ID
        subject_id = os.path.basename(os.path.dirname(txt_file)).split('_')[0]
        
        trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)
        
        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)
            
            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'subject_id': subject_id,
                    'original_name': original_name,
                    'response': trial['response']
                })
    
    # 处理CSV文件
    csv_mapping_files = mapping_files[len(txt_files):]
    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        if not os.path.exists(csv_file):
            print(f"CSV文件不存在: {csv_file}")
            continue
        
        # 从文件名提取被试ID
        # 需要合并同一被试的不同session（如aiwenkai_1和aiwenkai2应该是同一个被试）
        base_name = os.path.basename(csv_file).replace('.csv', '')
        
        # 先处理特殊情况：有空格的文件名（如"Liu Yaorui2"）
        base_name = base_name.replace(' ', '')
        
        # 去掉末尾的所有数字、下划线等
        # 例如：aiwenkai_1 -> aiwenkai, aiwenkai2 -> aiwenkai, wjy1 -> wjy
        subject_id = re.sub(r'[\d_]+$', '', base_name).strip('_').strip()
        
        # 如果去掉数字后为空，尝试提取字母部分
        if not subject_id or subject_id.isdigit():
            # 提取所有字母和中文（允许中间有数字，如"jiachen1017"）
            match = re.match(r'^([a-zA-Z\u4e00-\u9fa5]+(?:\d+)?[a-zA-Z\u4e00-\u9fa5]*)', base_name)
            if match:
                subject_id = match.group(1)
            else:
                subject_id = base_name  # 最后备选：使用原名称
        
        # 统一处理已知的变体（大小写、空格等）
        name_mapping = {
            'aiwenkai': 'aiwenkai',
            'lironghua': 'lironghua',
            'mayunmiao': 'mayunmiao',
            'ShangZiyang': 'ShangZiyang',
            'shangziyang': 'ShangZiyang',
            'wjy': 'wjy',
            'LiuYaorui': 'LiuYaorui',
            'liuyaorui': 'LiuYaorui',
        }
        
        # 尝试匹配和规范化
        subject_lower = subject_id.lower()
        for key, normalized in name_mapping.items():
            if subject_lower == key or subject_lower.startswith(key):
                subject_id = normalized
                break
        
        trials = parse_experiment_data_csv(csv_file)
        mapping = parse_audio_mapping(mapping_file)
        
        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)
            
            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'subject_id': subject_id,
                    'original_name': original_name,
                    'response': trial['response']
                })
    
    print(f"总共加载 {len(all_trials)} 个trial")
    unique_subjects = set(t['subject_id'] for t in all_trials)
    print(f"被试数量: {len(unique_subjects)}")
    print(f"被试ID列表: {sorted(unique_subjects)}")
    
    # 统计每个被试的trial数量
    from collections import Counter
    subject_counts = Counter(t['subject_id'] for t in all_trials)
    print("\n各被试的trial数量:")
    for subject_id, count in sorted(subject_counts.items()):
        print(f"  {subject_id}: {count} trials")
    
    return all_trials


def load_individual_rating_scores(excel_path):
    """加载每个被试的评分数据（每个sheet对应一个被试）"""
    if not os.path.exists(excel_path):
        print(f"警告: 评分Excel文件不存在: {excel_path}")
        return None
    
    try:
        excel_file = pd.ExcelFile(excel_path)
        subject_ratings = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # 尝试识别列
            category_col = df.columns[0]
            familiarity_col = None
            urgency_col = None
            liking_col = None
            
            for col in df.columns:
                col_str = str(col).strip()
                if '熟悉度' in col_str or 'familiarity' in col_str.lower():
                    familiarity_col = col
                elif '紧急度' in col_str or 'urgency' in col_str.lower():
                    urgency_col = col
                elif '喜爱度' in col_str or 'liking' in col_str.lower():
                    liking_col = col
            
            if familiarity_col is None and len(df.columns) > 1:
                familiarity_col = df.columns[1]
            if urgency_col is None and len(df.columns) > 2:
                urgency_col = df.columns[2]
            if liking_col is None and len(df.columns) > 3:
                liking_col = df.columns[3]
            
            rating_dict = {}
            for _, row in df.iterrows():
                category = str(row[category_col]).strip()
                if pd.isna(category) or category == '' or category == 'nan':
                    continue
                
                try:
                    familiarity = float(row[familiarity_col]) if familiarity_col and not pd.isna(row[familiarity_col]) else np.nan
                    urgency = float(row[urgency_col]) if urgency_col and not pd.isna(row[urgency_col]) else np.nan
                    liking = float(row[liking_col]) if liking_col and not pd.isna(row[liking_col]) else np.nan
                    
                    rating_dict[category] = {
                        'familiarity': familiarity,
                        'urgency': urgency,
                        'liking': liking
                    }
                except (ValueError, TypeError):
                    continue
            
            subject_ratings[sheet_name] = rating_dict
        
        print(f"成功加载 {len(subject_ratings)} 个被试的评分数据")
        return subject_ratings
    except Exception as e:
        print(f"加载评分数据失败: {e}")
        return None


def calculate_subject_rating_correlations(merged_pair_names, all_trials, subject_ratings, subject_id_mapping=None):
    """方法1：被试内分析 - 计算每个被试的评分差值与选A比例的相关性
    
    参数:
        merged_pair_names: 合并后的唯一音频对列表
        all_trials: 所有trial数据（包含subject_id, original_name, response）
        subject_ratings: 每个被试的评分字典 {sheet_name: {category: {familiarity, urgency, liking}}}
        subject_id_mapping: 被试ID映射字典，将实验数据中的subject_id映射到评分数据的sheet_name
    
    返回:
        包含每个被试相关性结果的DataFrame
    """
    print("\n=== 方法1：被试内分析 - 计算评分差值与选择比例的相关性 ===")
    
    if subject_ratings is None:
        print("警告: 没有评分数据，跳过被试内分析")
        return None
    
    # 如果没有提供映射，尝试自动匹配（sheet_name就是subject_id）
    if subject_id_mapping is None:
        subject_id_mapping = {sid: sid for sid in set(t['subject_id'] for t in all_trials)}
    
    all_results = []
    
    # 对于每个被试
    unique_subjects = set(t['subject_id'] for t in all_trials)
    
    for subject_id in unique_subjects:
        # 找到该被试的评分数据
        rating_sheet = subject_id_mapping.get(subject_id, subject_id)
        if rating_sheet not in subject_ratings:
            print(f"警告: 被试 {subject_id} 的评分数据不存在（sheet: {rating_sheet}）")
            continue
        
        subject_rating_dict = subject_ratings[rating_sheet]
        
        # 获取该被试的所有trial
        subject_trials = [t for t in all_trials if t['subject_id'] == subject_id]
        
        # 对于每个音频对，计算评分差值和选A比例
        pair_data = []
        
        for pair_name in merged_pair_names:
            parse_result = parse_pair_name(pair_name)
            if parse_result is None or parse_result[0] is None:
                continue
            
            left_cat, left_id, right_cat, right_id, _ = parse_result
            
            # 查找该被试对左右子类别的评分
            left_ratings = subject_rating_dict.get(left_cat, None)
            right_ratings = subject_rating_dict.get(right_cat, None)
            
            if left_ratings is None or right_ratings is None:
                continue
            
            # 计算评分差值（A - B，即left - right）
            familiarity_diff = left_ratings['familiarity'] - right_ratings['familiarity']
            urgency_diff = left_ratings['urgency'] - right_ratings['urgency']
            liking_diff = left_ratings['liking'] - right_ratings['liking']
            
            # 创建音频对的规范化键
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            pair_key = tuple(sorted([audio_a, audio_b]))
            
            # 找到该被试对这个音频对的所有trial（A-B和B-A）
            pair_trials = []
            for trial in subject_trials:
                trial_parse = parse_pair_name(trial['original_name'])
                if trial_parse is None or trial_parse[0] is None:
                    continue
                
                trial_left_cat, trial_left_id, trial_right_cat, trial_right_id, _ = trial_parse
                trial_audio_a = f"{trial_left_cat}_{trial_left_id}"
                trial_audio_b = f"{trial_right_cat}_{trial_right_id}"
                trial_pair_key = tuple(sorted([trial_audio_a, trial_audio_b]))
                
                if trial_pair_key == pair_key:
                    # 判断是否选择了audio_a（pair_name中定义的左侧音频）
                    # 注意：trial可能是A-B（audio_a在左）或B-A（audio_b在左）
                    selected_a = False
                    
                    # 情况1：trial是A-B（audio_a在左，audio_b在右）
                    if trial_audio_a == audio_a and trial_audio_b == audio_b:
                        if trial['response'] == 1:  # 选择了左侧，即audio_a
                            selected_a = True
                    # 情况2：trial是B-A（audio_b在左，audio_a在右）
                    elif trial_audio_a == audio_b and trial_audio_b == audio_a:
                        if trial['response'] == 2:  # 选择了右侧，即audio_a
                            selected_a = True
                    
                    pair_trials.append({
                        'trial_name': trial['original_name'],
                        'selected_a': selected_a
                    })
            
            # 计算选A的比例
            if len(pair_trials) == 0:
                continue
            
            select_a_count = sum(1 for t in pair_trials if t['selected_a'])
            select_a_ratio = select_a_count / len(pair_trials)
            
            # 记录数据
            pair_data.append({
                'pair_name': pair_name,
                'familiarity_diff': familiarity_diff,
                'urgency_diff': urgency_diff,
                'liking_diff': liking_diff,
                'select_a_ratio': select_a_ratio,
                'n_trials': len(pair_trials)
            })
        
        if len(pair_data) < 3:
            print(f"警告: 被试 {subject_id} 的有效数据点不足（{len(pair_data)} < 3），跳过")
            continue
        
        # 转换为DataFrame
        df_subject = pd.DataFrame(pair_data)
        
        # 计算三个维度与选A比例的相关性
        for dim_name, diff_col in [('Familiarity', 'familiarity_diff'),
                                    ('Urgency', 'urgency_diff'),
                                    ('Liking', 'liking_diff')]:
            # 去除NaN
            mask = ~(np.isnan(df_subject[diff_col]) | np.isnan(df_subject['select_a_ratio']))
            if np.sum(mask) < 3:
                all_results.append({
                    'subject_id': subject_id,
                    'dimension': dim_name,
                    'r': np.nan,
                    'p': np.nan,
                    'n': np.sum(mask)
                })
                continue
            
            x = df_subject[diff_col][mask]
            y = df_subject['select_a_ratio'][mask]
            
            r, p = pearsonr(x, y)
            all_results.append({
                'subject_id': subject_id,
                'dimension': dim_name,
                'r': r,
                'p': p,
                'n': np.sum(mask)
            })
        
        print(f"被试 {subject_id}: {len(pair_data)} 个有效音频对")
    
    if len(all_results) == 0:
        print("警告: 没有有效的相关性结果")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # 汇总统计
    print(f"\n=== 汇总统计 ===")
    for dim in ['Familiarity', 'Urgency', 'Liking']:
        dim_results = results_df[results_df['dimension'] == dim]
        valid_r = dim_results['r'].dropna()
        
        if len(valid_r) > 0:
            mean_r = valid_r.mean()
            std_r = valid_r.std()
            sig_count = (dim_results['p'] < 0.05).sum()
            total_count = len(valid_r)
            
            print(f"{dim}: r_mean={mean_r:.4f}, r_std={std_r:.4f}, "
                  f"显著数={sig_count}/{total_count} ({sig_count/total_count*100:.1f}%)")
            
            # 单样本t检验（检验相关系数是否显著不同于0）
            if len(valid_r) >= 2:
                t_stat, p_val = ttest_1samp(valid_r, 0)
                print(f"  单样本t检验: t={t_stat:.4f}, p={p_val:.4f}")
    
    return results_df


def calculate_selection_probability_144(merged_pair_names, all_trials):
    """计算144对的左侧音频选择概率"""
    print("\n=== 计算144对的选择概率 ===")
    
    probabilities = []
    
    for pair_name in merged_pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            probabilities.append(np.nan)
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        
        # 左侧音频标识
        audio_left = f"{left_cat}_{left_id}"
        audio_right = f"{right_cat}_{right_id}"
        
        # 统计：左侧音频被选择的次数和出现次数
        selected_count = 0
        total_count = 0
        
        for trial in all_trials:
            trial_name = trial['original_name']
            trial_parse_result = parse_pair_name(trial_name)
            if trial_parse_result is None or trial_parse_result[0] is None:
                continue
            
            trial_left_cat, trial_left_id, trial_right_cat, trial_right_id, _ = trial_parse_result
            
            trial_audio_left = f"{trial_left_cat}_{trial_left_id}"
            trial_audio_right = f"{trial_right_cat}_{trial_right_id}"
            
            # 检查这个trial是否为AB组合对（必须同时包含audio_left和audio_right）
            # 情况1: trial是 (audio_left, audio_right) 的组合
            if (trial_audio_left == audio_left and trial_audio_right == audio_right) or \
               (trial_audio_left == audio_right and trial_audio_right == audio_left):
                # 这是一个AB对
                total_count += 1
                # 检查是否选择了audio_left
                if (trial_audio_left == audio_left and trial['response'] == 1) or \
                   (trial_audio_right == audio_left and trial['response'] == 2):
                    selected_count += 1
        
        prob = selected_count / total_count if total_count > 0 else 0.5
        probabilities.append(prob)
    
    probabilities = np.array(probabilities)
    print(f"计算完成: {len(probabilities)} 个概率值")
    print(f"概率范围: [{np.nanmin(probabilities):.3f}, {np.nanmax(probabilities):.3f}]")
    print(f"概率均值: {np.nanmean(probabilities):.3f}")
    
    return probabilities


def calculate_correlations(feature_diff_144, probabilities_144):
    """计算21个特征与概率的pearson相关（18个音频特征 + 3个评分特征）"""
    print("\n=== 计算特征与概率的相关性 ===")
    
    feature_names = [
        'Spectral Centroid (mean)',
        'Spectral Centroid (std)',
        'F0 (mean)',
        'F0 (std)',
        'Harmonicity (mean)',
        'Harmonicity (std)',
        'Spectral Modulation Centroid (mean)',
        'Spectral Modulation Centroid (std)',
        'Spectral Irregularity (mean)',
        'Spectral Irregularity (std)',
        'Loudness (mean)',
        'Loudness (std)',
        'Spectral Flatness (mean)',
        'Spectral Flatness (std)',
        'Spectral Bandwidth (mean)',
        'Spectral Bandwidth (std)',
        'Temporal Modulation (mean)',
        'Temporal Modulation (std)',
        'Familiarity',
        'Urgency',
        'Liking'
    ]
    
    n_features = feature_diff_144.shape[0]
    results = []
    
    for i in range(n_features):
        feature_values = feature_diff_144[i, :]
        
        # 获取特征名称（如果索引超出范围，使用默认名称）
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i+1}'
        
        # 去除NaN
        mask = ~(np.isnan(feature_values) | np.isnan(probabilities_144))
        if np.sum(mask) < 3:
            results.append({
                'feature_idx': i + 1,
                'feature_name': feature_name,
                'r': np.nan,
                'p': np.nan,
                'n': np.sum(mask),
                'significant': False
            })
            continue
        
        r, p = pearsonr(feature_values[mask], probabilities_144[mask])
        is_sig = p < 0.05
        results.append({
            'feature_idx': i + 1,
            'feature_name': feature_name,
            'r': r,
            'p': p,
            'n': np.sum(mask),
            'significant': is_sig
        })
        
        sig_mark = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"特征{i+1} ({feature_name}): r={r:.4f}, p={p:.4f} {sig_mark} (n={np.sum(mask)})")
    
    return pd.DataFrame(results)


def plot_within_subject_correlations(results_df, output_dir):
    """绘制被试内分析结果的汇总图表"""
    print("\n=== 绘制被试内分析结果图表 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 绘制每个维度的相关系数分布（小提琴图+箱线图）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    dimensions = ['Familiarity', 'Urgency', 'Liking']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (dim, color) in enumerate(zip(dimensions, colors)):
        ax = axes[i]
        dim_data = results_df[results_df['dimension'] == dim]['r'].dropna()
        
        if len(dim_data) > 0:
            # 小提琴图
            parts = ax.violinplot([dim_data], positions=[0], widths=0.6, showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            # 箱线图
            bp = ax.boxplot([dim_data], positions=[0], widths=0.3, patch_artist=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.3)
            
            # 散点图（显示每个被试的值）
            y_jitter = np.random.normal(0, 0.05, len(dim_data))
            ax.scatter(y_jitter, dim_data, c=color, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # 添加均值线和置信区间
            mean_r = dim_data.mean()
            std_r = dim_data.std()
            ax.axhline(y=mean_r, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_r:.3f}')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-1.1, 1.1)
            ax.set_ylabel('Correlation Coefficient (r)', fontsize=11)
            ax.set_title(f'{dim}\n(n={len(dim_data)})', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=9)
        else:
            ax.text(0, 0, 'No data', ha='center', va='center')
            ax.set_title(dim, fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'within_subject_correlations_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"小提琴图已保存: {output_path}")
    
    # 2. 绘制每个被试的相关系数条形图
    if len(results_df) > 0:
        # 透视表：被试 x 维度
        pivot_df = results_df.pivot(index='subject_id', columns='dimension', values='r')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(pivot_df.index))
        width = 0.25
        
        for i, dim in enumerate(dimensions):
            if dim in pivot_df.columns:
                values = pivot_df[dim].fillna(0)
                ax.bar(x + i*width, values, width, label=dim, color=colors[i], alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Subject ID', fontsize=11)
        ax.set_ylabel('Correlation Coefficient (r)', fontsize=11)
        ax.set_title('Within-Subject Correlations by Subject', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha='right', fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'within_subject_correlations_by_subject.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"条形图已保存: {output_path}")


def plot_correlations(feature_diff_144, probabilities_144, output_dir):
    """绘制特征-概率相关性散点图"""
    print("\n=== 绘制相关性散点图 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_names = [
        'Spectral Centroid (mean)',
        'Spectral Centroid (std)',
        'F0 (mean)',
        'F0 (std)',
        'Harmonicity (mean)',
        'Harmonicity (std)',
        'Spectral Modulation Centroid (mean)',
        'Spectral Modulation Centroid (std)',
        'Spectral Irregularity (mean)',
        'Spectral Irregularity (std)',
        'Loudness (mean)',
        'Loudness (std)',
        'Spectral Flatness (mean)',
        'Spectral Flatness (std)',
        'Spectral Bandwidth (mean)',
        'Spectral Bandwidth (std)',
        'Temporal Modulation (mean)',
        'Temporal Modulation (std)',
        'Familiarity',
        'Urgency',
        'Liking'
    ]
    
    n_features = feature_diff_144.shape[0]
    # 调整子图布局：21个特征（18个音频特征+3个评分特征），使用5行5列（25个位置）
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        feature_values = feature_diff_144[i, :]
        
        # 去除NaN
        mask = ~(np.isnan(feature_values) | np.isnan(probabilities_144))
        x = feature_values[mask]
        y = probabilities_144[mask]
        
        if len(x) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i+1}')
            continue
        
        # 散点图
        ax.scatter(x, y, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # 拟合线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)
        
        # 相关系数
        r, p_val = pearsonr(x, y)
        sig_mark = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i+1}'
        ax.set_title(f'{feature_name}\nr={r:.3f}{sig_mark}, p={p_val:.4f}', fontsize=10)
        ax.set_xlabel('Feature Difference (L-R)', fontsize=9)
        ax.set_ylabel('Selection Probability', fontsize=9)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图（如果有21个特征，第22-25个子图需要隐藏）
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_probability_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 不使用show()避免阻塞
    
    print(f"散点图已保存: {output_path}")


def main():
    print("=== 开始特征-人类决策相关性分析（按实验类型分组） ===")
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, 'correlation_results_by_type')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载特征数据
    print("\n步骤1: 加载特征数据")
    diff_raw, diff_zscore, pair_names = load_or_extract_features(project_root)
    
    # 1.5. 加载评分特征并整合到特征矩阵中
    print("\n步骤1.5: 加载并整合评分特征")
    rating_csv_path = r"D:\D\research\audioset下载\展示的图片\rating_subcategories.csv"
    rating_dict = load_rating_scores(rating_csv_path)
    
    if rating_dict is not None:
        # 提取评分特征
        rating_features_raw = extract_rating_features(pair_names, rating_dict)
        
        if rating_features_raw is not None:
            # 对评分特征进行z-score标准化
            rating_features_zscore = zscore_per_feature(rating_features_raw)
            
            # 将评分特征追加到原始特征矩阵
            diff_raw = np.vstack([diff_raw, rating_features_raw])
            diff_zscore = np.vstack([diff_zscore, rating_features_zscore])
            
            print(f"特征矩阵已扩展: 18个音频特征 + 3个评分特征 = {diff_raw.shape[0]} 个特征")
        else:
            print("警告: 无法提取评分特征，继续使用原始18个特征")
    else:
        print("警告: 评分数据加载失败，继续使用原始18个特征")
    
    # 2. 按实验类型分组合并左右平衡配对
    merged_results = merge_balanced_pairs_by_type(pair_names)
    
    # 3. 加载人类决策数据
    print("\n步骤3: 加载人类决策数据")
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
    
    # 3.5. 加载每个被试的评分数据（用于方法1：被试内分析）
    print("\n步骤3.5: 加载每个被试的评分数据")
    rating_excel_path = r"D:\D\research\数据采集\rating.xlsx"
    subject_ratings = load_individual_rating_scores(rating_excel_path)
    
    # 执行方法1：被试内分析（对所有类型合并后的数据）
    if subject_ratings is not None:
        print("\n" + "="*60)
        print("执行方法1：被试内分析（所有实验类型合并）")
        print("="*60)
        
        # 合并所有类型的音频对名称
        all_merged_pair_names = []
        for exp_type, merged_data in merged_results.items():
            all_merged_pair_names.extend(merged_data['pair_names'])
        
        # 创建被试ID映射（将实验数据中的subject_id映射到Excel中的sheet_name）
        # 实验数据中的ID -> Excel中的sheet名称
        subject_id_mapping = {
            'liyanchen': 'yanchen',
            'LiuYaorui': 'liuyaorui',
            'ShangZiyang': 'shangziyang',
            'aiwenkai': 'aikaiwen',
            'jiachen1017': 'jiachen',
            # 其他的如果名称一致就不需要映射（会在calculate_subject_rating_correlations中自动匹配）
            'shimin': 'shimin',
            'lironghua': 'lironghua',
            'mayunmiao': 'mayunmiao',
            'wjy': 'wjy',
        }
        
        print("\n被试ID映射关系:")
        for exp_id, sheet_name in sorted(subject_id_mapping.items()):
            print(f"  实验数据: {exp_id} -> Excel Sheet: {sheet_name}")
        
        within_subject_results = calculate_subject_rating_correlations(
            all_merged_pair_names, 
            all_trials, 
            subject_ratings, 
            subject_id_mapping
        )
        
        if within_subject_results is not None:
            # 保存被试内分析结果
            within_subject_output_path = os.path.join(output_dir, 'within_subject_rating_correlations.csv')
            within_subject_results.to_csv(within_subject_output_path, index=False, encoding='utf-8-sig')
            print(f"\n被试内分析结果已保存: {within_subject_output_path}")
            
            # 绘制汇总图表
            plot_within_subject_correlations(within_subject_results, output_dir)
    
    # 4. 对每个实验类型进行分析
    all_results_summary = []
    
    for exp_type, merged_data in merged_results.items():
        print(f"\n\n{'='*60}")
        print(f"分析实验类型: {exp_type.upper()}")
        print(f"{'='*60}")
        
        merged_indices = merged_data['indices']
        merged_pair_names = merged_data['pair_names']
        
        # 提取该类型的特征差
        feature_diff = diff_zscore[:, merged_indices]
        print(f"特征差矩阵形状: {feature_diff.shape}")
        
        # 计算选择概率
        probabilities = calculate_selection_probability_144(merged_pair_names, all_trials)
        
        # 计算相关性
        correlation_results = calculate_correlations(feature_diff, probabilities)
        
        # 添加实验类型列
        correlation_results['experiment_type'] = exp_type
        all_results_summary.append(correlation_results)
        
        # 保存该类型的结果
        type_dir = os.path.join(output_dir, exp_type)
        os.makedirs(type_dir, exist_ok=True)
        
        np.save(os.path.join(type_dir, f'{exp_type}_pairs.npy'), merged_pair_names)
        np.save(os.path.join(type_dir, f'{exp_type}_probabilities.npy'), probabilities)
        np.save(os.path.join(type_dir, f'{exp_type}_feature_diff.npy'), feature_diff)
        correlation_results.to_csv(os.path.join(type_dir, f'{exp_type}_correlation.csv'), 
                                   index=False, encoding='utf-8-sig')
        
        # 绘制该类型的散点图
        plot_correlations(feature_diff, probabilities, type_dir)
        
        print(f"\n{exp_type} 类型结果已保存到: {type_dir}")
    
    # 5. 额外的合并分析
    additional_analyses = {
        'merged_main_sub': {
            'name': 'MAIN+SUB合并',
            'types': ['main', 'sub'],
            'display_name': 'main_sub'
        },
        'merged_nn_main_nn_sub': {
            'name': 'NN_MAIN+NN_SUB合并',
            'types': ['nn_main', 'nn_sub'],
            'display_name': 'nn_main_nn_sub'
        },
        'merged_all': {
            'name': '所有类型合并',
            'types': ['nn_main', 'nn_sub', 'main', 'sub'],
            'display_name': 'all'
        }
    }
    
    for merge_key, merge_config in additional_analyses.items():
        print(f"\n\n{'='*60}")
        print(f"分析合并类型: {merge_config['name']}")
        print(f"{'='*60}")
        
        # 合并指定类型的数据
        merged_indices_combined = []
        merged_pair_names_combined = []
        
        for exp_type in merge_config['types']:
            if exp_type in merged_results:
                merged_indices_combined.extend(merged_results[exp_type]['indices'])
                merged_pair_names_combined.extend(merged_results[exp_type]['pair_names'])
        
        if len(merged_indices_combined) == 0:
            print("没有数据可以合并")
            continue
        
        # 提取合并后的特征差
        feature_diff_combined = diff_zscore[:, merged_indices_combined]
        print(f"合并后特征差矩阵形状: {feature_diff_combined.shape}")
        
        # 计算选择概率
        probabilities_combined = calculate_selection_probability_144(merged_pair_names_combined, all_trials)
        
        # 计算相关性
        correlation_results_combined = calculate_correlations(feature_diff_combined, probabilities_combined)
        
        # 添加合并类型标识
        correlation_results_combined['experiment_type'] = merge_config['display_name']
        all_results_summary.append(correlation_results_combined)
        
        # 保存合并类型的结果
        merge_dir = os.path.join(output_dir, merge_config['display_name'])
        os.makedirs(merge_dir, exist_ok=True)
        
        np.save(os.path.join(merge_dir, f'{merge_config["display_name"]}_pairs.npy'), merged_pair_names_combined)
        np.save(os.path.join(merge_dir, f'{merge_config["display_name"]}_probabilities.npy'), probabilities_combined)
        np.save(os.path.join(merge_dir, f'{merge_config["display_name"]}_feature_diff.npy'), feature_diff_combined)
        correlation_results_combined.to_csv(os.path.join(merge_dir, f'{merge_config["display_name"]}_correlation.csv'), 
                                           index=False, encoding='utf-8-sig')
        
        # 绘制散点图
        plot_correlations(feature_diff_combined, probabilities_combined, merge_dir)
        
        print(f"{merge_config['name']} 结果已保存到: {merge_dir}")
    
    # 6. 汇总所有类型的结果（包括合并类型）
    print(f"\n\n{'='*60}")
    print("最终汇总结果")
    print(f"{'='*60}")
    
    all_results_df = pd.concat(all_results_summary, ignore_index=True)
    all_results_df.to_csv(os.path.join(output_dir, 'all_types_correlation_summary.csv'), 
                         index=False, encoding='utf-8-sig')
    
    print(f"\n汇总结果已保存: {os.path.join(output_dir, 'all_types_correlation_summary.csv')}")
    print("\n=== 分析完成 ===")
    
    # 打印汇总表格
    print("\n所有类型相关性结果汇总:")
    print(all_results_df.to_string())


if __name__ == '__main__':
    main()

