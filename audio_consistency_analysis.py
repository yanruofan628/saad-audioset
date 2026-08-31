#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频对左右位置互换时人类选择的一致性分析
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


COMBINED_GROUPS = {
    'random': ['main', 'sub'],
    'nn': ['nn_main', 'nn_sub']
}

CATEGORY_COLOR_MAP = {
    'main': '#264653',
    'sub': '#2A9D8F',
    'random': '#4C78A8',
    'nn_main': '#E9C46A',
    'nn_sub': '#F4A261',
    'nn': '#E76F51'
}

CATEGORY_ORDER = ['random', 'nn', 'main', 'nn_main', 'sub', 'nn_sub']


def add_combined_categories(df):
    """Append combined experiment groups (random, nn) based on base categories."""
    if df is None or df.empty or 'experiment_type' not in df.columns:
        return df

    combined_frames = []
    for new_label, members in COMBINED_GROUPS.items():
        subset = df[df['experiment_type'].isin(members)]
        if not subset.empty:
            temp = subset.copy()
            temp['experiment_type'] = new_label
            combined_frames.append(temp)

    if combined_frames:
        df = pd.concat([df] + combined_frames, ignore_index=True)

    return df


def get_category_palette(order):
    """Return a color palette aligned with the preferred category order."""
    palette = []
    for label in order:
        if label in CATEGORY_COLOR_MAP:
            palette.append(CATEGORY_COLOR_MAP[label])

    if len(palette) < len(order):
        remaining_labels = [label for label in order if label not in CATEGORY_COLOR_MAP]
        palette.extend(sns.color_palette('husl', len(remaining_labels)))

    return palette


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


def load_all_human_decisions_with_subject(txt_files, csv_files, mapping_files):
    """加载所有人类决策数据（带被试ID）"""
    print("\n=== 加载人类决策数据（带被试ID） ===")
    
    all_trials = []
    
    # 处理TXT文件
    for txt_file, mapping_file in zip(txt_files, mapping_files[:len(txt_files)]):
        if not os.path.exists(txt_file):
            print(f"TXT文件不存在: {txt_file}")
            continue
        
        # 从文件名提取被试ID
        subject_id = os.path.basename(txt_file).replace('.txt', '')
        
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
        subject_id = os.path.basename(csv_file).replace('.csv', '')
        
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
    return all_trials


def load_all_human_decisions(txt_files, csv_files, mapping_files):
    """加载所有人类决策数据"""
    print("\n=== 加载人类决策数据 ===")
    
    all_trials = []
    
    # 处理TXT文件
    for txt_file, mapping_file in zip(txt_files, mapping_files[:len(txt_files)]):
        if not os.path.exists(txt_file):
            print(f"TXT文件不存在: {txt_file}")
            continue
        
        trials = parse_experiment_data_txt(txt_file)
        mapping = parse_audio_mapping(mapping_file)
        
        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)
            
            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'original_name': original_name,
                    'response': trial['response']
                })
    
    # 处理CSV文件
    csv_mapping_files = mapping_files[len(txt_files):]
    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        if not os.path.exists(csv_file):
            print(f"CSV文件不存在: {csv_file}")
            continue
        
        trials = parse_experiment_data_csv(csv_file)
        mapping = parse_audio_mapping(mapping_file)
        
        for trial in trials:
            wavfile = trial['wavfile']
            filename = os.path.basename(wavfile)
            
            if filename in mapping:
                original_name = mapping[filename]
                all_trials.append({
                    'original_name': original_name,
                    'response': trial['response']
                })
    
    print(f"总共加载 {len(all_trials)} 个trial")
    return all_trials


def calculate_consistency_scores(pair_names, all_trials):
    """
    计算每个音频对的一致性得分（基于音频吸引力）
    
    对每个音频对(A,B)：
    - P₁ = A左B右时，所有被试选择右耳（即选B）的概率
    - P₂ = A右B左时，所有被试选择左耳（即选B）的概率
    
    # 方法一：基于音频吸引力的一致性
    - P_A_left = A在左时被选概率 = 1 - P₁ (选左=选A)
    - P_A_right = A在右时被选概率 = 1 - P₂ (选右=选A)
    - 一致性得分 = 1 - |P_A_left - P_A_right|
    
    返回：字典，包含每个实验类型的详细结果
    """
    print("\n=== 计算一致性得分 ===")
    
    # 按实验类型分组音频对
    type_groups = defaultdict(list)
    
    for idx, pair_name in enumerate(pair_names):
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[4] is None:
            print(f"警告: 无法解析实验类型 {pair_name}")
            continue
        
        left_cat, left_id, right_cat, right_id, experiment_type = parse_result
        type_groups[experiment_type].append((idx, pair_name, left_cat, left_id, right_cat, right_id))
    
    results = {}
    
    for exp_type, pairs in type_groups.items():
        print(f"\n处理 {exp_type} 类型: {len(pairs)} 个音频对")
        
        pair_results = []
        
        for idx, pair_name, left_cat, left_id, right_cat, right_id in pairs:
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            
            # 构建两种排列的音频对名称
            # A左B右: {prefix}{left_cat}_{left_id}+{right_cat}_{right_id}.wav
            # A右B左: {prefix}{right_cat}_{right_id}+{left_cat}_{left_id}.wav
            prefix = f"{exp_type}_"
            pair_ab = f"{prefix}{left_cat}_{left_id}+{right_cat}_{right_id}.wav"
            pair_ba = f"{prefix}{right_cat}_{right_id}+{left_cat}_{left_id}.wav"
            
            # 统计A左B右的情况
            trials_ab = []
            for trial in all_trials:
                if trial['original_name'] == pair_ab:
                    trials_ab.append(trial)
            
            # 统计A右B左的情况
            trials_ba = []
            for trial in all_trials:
                if trial['original_name'] == pair_ba:
                    trials_ba.append(trial)
            
            # 计算P₁: A左B右时选择右耳(即选B)的概率
            if len(trials_ab) > 0:
                right_count_ab = sum(1 for t in trials_ab if t['response'] == 2)
                left_count_ab = len(trials_ab) - right_count_ab
                p1 = right_count_ab / len(trials_ab)
            else:
                p1 = np.nan
                right_count_ab = 0
                left_count_ab = 0
            
            # 计算P₂: A右B左时选择左耳(即选B)的概率
            if len(trials_ba) > 0:
                left_count_ba = sum(1 for t in trials_ba if t['response'] == 1)
                right_count_ba = len(trials_ba) - left_count_ba
                p2 = left_count_ba / len(trials_ba)
            else:
                p2 = np.nan
                left_count_ba = 0
                right_count_ba = 0
            
            # 计算一致性得分（基于音频吸引力）
            if not (np.isnan(p1) or np.isnan(p2)):
                # A在两种位置下的被选概率
                p_a_left = 1 - p1  # A在左时被选概率（选左=选A）
                p_a_right = 1 - p2  # A在右时被选概率（选右=选A）
                
                # 一致性得分：两种位置下A被选概率的差异越小，一致性越高
                consistency = 1 - abs(p_a_left - p_a_right)
            else:
                consistency = np.nan
            
            total_trials = len(trials_ab) + len(trials_ba)
            if total_trials > 0:
                count_a = left_count_ab + right_count_ba  # A左时选左 + A右时选右
                count_b = right_count_ab + left_count_ba  # 其余对应B
                majority_support = max(count_a, count_b) / total_trials
                prefer_label = 'A' if count_a >= count_b else 'B'
            else:
                majority_support = np.nan
                prefer_label = None

            pair_results.append({
                'pair_name': pair_name,
                'audio_a': audio_a,
                'audio_b': audio_b,
                'p1': p1,
                'p2': p2,
                'consistency': consistency,
                'n_ab': len(trials_ab),
                'n_ba': len(trials_ba),
                'majority_support': majority_support,
                'majority_preference': prefer_label,
                'count_a': left_count_ab + right_count_ba,
                'count_b': right_count_ab + left_count_ba,
                'total_trials': total_trials
            })
            
            if idx % 20 == 0:
                print(f"  已处理 {idx}/{len(pairs)}")
        
        results[exp_type] = pair_results
        print(f"{exp_type} 完成: {len(pair_results)} 个音频对")
    
    return results


def calculate_individual_consistency_and_consensus(pair_names, all_trials_with_subject):
    """
    计算个体一致性和群体共识度
    
    对每个音频对(A,B)：
    1. 找出两种位置条件都参与过的被试
    2. 对每个被试，检查一致性：都选A或都选B
    3. 计算个体一致性率 = 一致被试数 / 总被试数
    4. 在一致被试中，计算群体共识度 = max(选A比例, 选B比例)
    
    返回：字典，包含每个实验类型的结果
    """
    print("\n=== 计算个体一致性和群体共识度 ===")
    
    # 按实验类型分组音频对
    type_groups = defaultdict(list)
    
    for idx, pair_name in enumerate(pair_names):
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[4] is None:
            continue
        
        left_cat, left_id, right_cat, right_id, experiment_type = parse_result
        type_groups[experiment_type].append((idx, pair_name, left_cat, left_id, right_cat, right_id))
    
    results = {}
    
    for exp_type, pairs in type_groups.items():
        print(f"\n处理 {exp_type} 类型: {len(pairs)} 个音频对")
        
        pair_results = []
        
        for idx, pair_name, left_cat, left_id, right_cat, right_id in pairs:
            audio_a = f"{left_cat}_{left_id}"
            audio_b = f"{right_cat}_{right_id}"
            
            # 构建两种排列的音频对名称
            prefix = f"{exp_type}_"
            pair_ab = f"{prefix}{left_cat}_{left_id}+{right_cat}_{right_id}.wav"
            pair_ba = f"{prefix}{right_cat}_{right_id}+{left_cat}_{left_id}.wav"
            
            # 按被试分组统计数据
            subject_data = defaultdict(lambda: {'ab': None, 'ba': None})
            
            for trial in all_trials_with_subject:
                if trial['original_name'] == pair_ab:
                    subject_data[trial['subject_id']]['ab'] = trial['response']
                elif trial['original_name'] == pair_ba:
                    subject_data[trial['subject_id']]['ba'] = trial['response']
            
            # 找出两种条件都参与过的被试
            complete_subjects = []
            for subj_id, data in subject_data.items():
                if data['ab'] is not None and data['ba'] is not None:
                    complete_subjects.append({
                        'subject_id': subj_id,
                        'ab': data['ab'],
                        'ba': data['ba']
                    })
            
            if len(complete_subjects) == 0:
                pair_results.append({
                    'pair_name': pair_name,
                    'audio_a': audio_a,
                    'audio_b': audio_b,
                    'individual_consistency': np.nan,
                    'group_consensus': np.nan,
                    'n_complete_subjects': 0,
                    'n_consistent': 0,
                    'n_prefer_a': 0,
                    'n_prefer_b': 0
                })
                continue
            
            # 计算个体一致性
            # 一致：都选A(1)或都选B(2)，或者在不同位置下选择都指向同一个音频
            n_consistent = 0
            n_prefer_a = 0  # 一致且偏好A的被试数
            n_prefer_b = 0  # 一致且偏好B的被试数
            
            for subj in complete_subjects:
                # A左B右时的选择
                choice_ab_left = True if subj['ab'] == 1 else False  # 1=左，选A
                choice_ab_right = True if subj['ab'] == 2 else False  # 2=右，选B
                
                # A右B左时的选择
                choice_ba_left = True if subj['ba'] == 1 else False  # 1=左，选B(因为此时B在左)
                choice_ba_right = True if subj['ba'] == 2 else False  # 2=右，选A(因为此时A在右)
                
                # 判断一致性：无论位置如何，是否都选A或都选B
                # 如果A左时选了A，A右时也选了A → 一致偏好A
                # 如果A左时选了B，A右时也选了B → 一致偏好B
                if choice_ab_left and choice_ba_right:
                    # A左时选A，A右时也选A → 一致偏好A
                    n_consistent += 1
                    n_prefer_a += 1
                elif choice_ab_right and choice_ba_left:
                    # A左时选B，A右时也选B → 一致偏好B
                    n_consistent += 1
                    n_prefer_b += 1
            
            individual_consistency = n_consistent / len(complete_subjects)
            
            # 在一致被试中计算群体共识度
            if n_consistent > 0:
                group_consensus = max(n_prefer_a, n_prefer_b) / n_consistent
            else:
                group_consensus = np.nan
            
            pair_results.append({
                'pair_name': pair_name,
                'audio_a': audio_a,
                'audio_b': audio_b,
                'individual_consistency': individual_consistency,
                'group_consensus': group_consensus,
                'n_complete_subjects': len(complete_subjects),
                'n_consistent': n_consistent,
                'n_prefer_a': n_prefer_a,
                'n_prefer_b': n_prefer_b
            })
            
            if len(pair_results) % 20 == 0:
                print(f"  已处理 {len(pair_results)}/{len(pairs)}")
        
        results[exp_type] = pair_results
        print(f"{exp_type} 完成: {len(pair_results)} 个音频对")
    
    return results


def plot_violin_consistency(all_results, output_dir):
    """Plot consistency score boxplots with all experiment types on one chart."""
    print("\n=== 绘制一致性得分箱线图 ===")

    plot_records = []
    for exp_type, pair_results in all_results.items():
        for pair_result in pair_results:
            consistency = pair_result['consistency']
            if not np.isnan(consistency):
                plot_records.append({
                    'experiment_type': exp_type,
                    'consistency': consistency
                })

    if len(plot_records) == 0:
        print("警告: 没有可用于绘图的一致性数据")
        return

    plot_df = pd.DataFrame(plot_records)

    # 按实验类型输出统计信息
    summary = plot_df.groupby('experiment_type')['consistency'].agg(['count', 'mean', 'std', 'median'])
    for exp_type, row in summary.iterrows():
        print(f"{exp_type}: n={int(row['count'])}, mean={row['mean']:.4f}, std={row['std']:.4f}, median={row['median']:.4f}")

    for combined_label, members in COMBINED_GROUPS.items():
        subset = plot_df[plot_df['experiment_type'].isin(members)]
        if not subset.empty:
            stats = subset['consistency'].agg(['count', 'mean', 'std', 'median'])
            print(f"{combined_label}: n={int(stats['count'])}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, median={stats['median']:.4f}")

    plot_df_combined = add_combined_categories(plot_df.copy())

    if plot_df_combined is None or plot_df_combined.empty:
        print("警告: 没有可用于绘图的一致性数据（组合后）")
        return

    unique_types = list(plot_df_combined['experiment_type'].unique())
    order = [label for label in CATEGORY_ORDER if label in unique_types]
    remaining = [label for label in unique_types if label not in order]
    order.extend(sorted(remaining))

    palette = get_category_palette(order)

    plt.figure(figsize=(12, 7))

    min_val = plot_df_combined['consistency'].min()
    max_val = plot_df_combined['consistency'].max()
    lower_margin = 0.08 if min_val > 0.1 else 0.04
    upper_margin = 0.06
    y_min = max(-0.02, min_val - lower_margin)
    y_max = min(1.05, max_val + upper_margin)
    if y_min >= y_max:
        y_min = max(-0.02, min_val - 0.02)
        y_max = min(1.05, max_val + 0.03)

    box_kw = dict(
        width=0.55,
        linewidth=1.8,
        fliersize=0,
        boxprops=dict(alpha=0.85, edgecolor='#1B1F3B', linewidth=1.6),
        medianprops=dict(color='#1B1F3B', linewidth=2.2),
        whiskerprops=dict(color='#1B1F3B', linewidth=1.6),
        capprops=dict(color='#1B1F3B', linewidth=1.6)
    )

    sns.boxplot(
        data=plot_df_combined,
        x='experiment_type',
        y='consistency',
        order=order,
        palette=palette,
        **box_kw
    )
    sns.stripplot(
        data=plot_df_combined,
        x='experiment_type',
        y='consistency',
        order=order,
        color='#2E4057',
        alpha=0.45,
        jitter=0.18,
        size=4.2,
        linewidth=0.4,
        edgecolor='#F7F9F9'
    )

    plt.title('Consistency Scores by Experiment Type', fontsize=22, fontweight='bold')
    plt.xlabel('Experiment Type', fontsize=26, fontweight='bold')
    plt.ylabel('Consistency Score', fontsize=26, fontweight='bold')
    plt.xticks(rotation=0, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.ylim([y_min, y_max])
    plt.grid(axis='y', color='#D7DBDD', alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'consistency_scores_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")
    print("箱线图绘制完成")


def plot_individual_analysis(individual_results, output_dir):
    """Plot separate boxplots for individual consistency and group consensus, including combined groups."""
    print("\n=== 绘制个体一致性与群体共识度箱线图 ===")

    individual_records = []
    group_records = []
    for exp_type, pair_results in individual_results.items():
        for pair_result in pair_results:
            indiv_cons = pair_result['individual_consistency']
            group_cons = pair_result['group_consensus']
            if not np.isnan(indiv_cons):
                individual_records.append({
                    'experiment_type': exp_type,
                    'score': indiv_cons
                })
            if not np.isnan(group_cons):
                group_records.append({
                    'experiment_type': exp_type,
                    'score': group_cons
                })

    if len(individual_records) == 0 and len(group_records) == 0:
        print("警告: 没有可用于绘制箱线图的个体一致性或群体共识度数据")
        return

    individual_df = pd.DataFrame(individual_records) if individual_records else pd.DataFrame(columns=['experiment_type', 'score'])
    group_df = pd.DataFrame(group_records) if group_records else pd.DataFrame(columns=['experiment_type', 'score'])

    individual_aug = add_combined_categories(individual_df.copy())
    group_aug = add_combined_categories(group_df.copy())

    def _print_stats(label, df):
        if df is None or df.empty:
            print(f"Warning: no data available for {label} boxplot")
            return
        stats = df.groupby('experiment_type')['score'].agg(['count', 'mean', 'std', 'median'])
        for exp_type, row in stats.iterrows():
            print(f"{label} - {exp_type}: n={int(row['count'])}, mean={row['mean']:.4f}, std={row['std']:.4f}, median={row['median']:.4f}")

    def _draw_boxplot(df, title, ylabel, filename):
        if df is None or df.empty:
            print(f"Warning: no data available for {title.lower()}")
            return

        unique_types = list(df['experiment_type'].unique())
        order = [label for label in CATEGORY_ORDER if label in unique_types]
        remaining = [label for label in unique_types if label not in order]
        order.extend(sorted(remaining))
        palette = get_category_palette(order)

        min_val = df['score'].min()
        max_val = df['score'].max()
        lower_margin = 0.08 if min_val > 0.1 else 0.04
        upper_margin = 0.06
        y_min = max(-0.02, min_val - lower_margin)
        y_max = min(1.05, max_val + upper_margin)
        if y_min >= y_max:
            y_min = max(-0.02, min_val - 0.02)
            y_max = min(1.05, max_val + 0.03)

        box_kw = dict(
            width=0.55,
            linewidth=1.8,
            fliersize=0,
            boxprops=dict(alpha=0.78, edgecolor='#1B1F3B', linewidth=1.6),
            medianprops=dict(color='#1B1F3B', linewidth=2.2),
            whiskerprops=dict(color='#1B1F3B', linewidth=1.6),
            capprops=dict(color='#1B1F3B', linewidth=1.6)
        )

        plt.figure(figsize=(12, 7))
        sns.boxplot(
            data=df,
            x='experiment_type',
            y='score',
            order=order,
            palette=palette,
            **box_kw
        )
        sns.stripplot(
            data=df,
            x='experiment_type',
            y='score',
            order=order,
            color='#2E4057',
            alpha=0.45,
            jitter=0.18,
            size=4.2,
            linewidth=0.4,
            edgecolor='#F7F9F9'
        )

        plt.title(title, fontsize=22, fontweight='bold')
        plt.xlabel('Experiment Type', fontsize=26, fontweight='bold')
        plt.ylabel(ylabel, fontsize=26, fontweight='bold')
        plt.ylim([y_min, y_max])
        plt.xticks(rotation=0, fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.grid(axis='y', color='#D7DBDD', alpha=0.5, linestyle='--', linewidth=1)
        plt.tight_layout()

        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_path}")

    _print_stats('Individual Consistency', individual_aug)
    _draw_boxplot(individual_aug, 'Individual Consistency by Experiment Type', 'Individual Consistency', 'individual_consistency_boxplot.png')

    _print_stats('Group Consensus', group_aug)
    _draw_boxplot(group_aug, 'Group Consensus by Experiment Type', 'Group Consensus', 'group_consensus_boxplot.png')

    print("个体分析和共识度箱线图绘制完成")


def plot_majority_support(all_results, output_dir):
    """Plot majority support proportion for each pair by experiment type (including combined groups)."""
    print("\n=== 绘制多数支持比例箱线图 ===")

    records = []
    for exp_type, pair_results in all_results.items():
        for pair_result in pair_results:
            majority_support = pair_result.get('majority_support')
            if majority_support is not None and not np.isnan(majority_support):
                records.append({
                    'experiment_type': exp_type,
                    'majority_support': majority_support
                })

    if len(records) == 0:
        print("警告: 没有可用于绘制多数支持比例的有效数据")
        return

    base_df = pd.DataFrame(records)
    combined_df = add_combined_categories(base_df.copy())

    unique_types = list(combined_df['experiment_type'].unique())
    order = [label for label in CATEGORY_ORDER if label in unique_types]
    remaining = [label for label in unique_types if label not in order]
    order.extend(sorted(remaining))
    palette = get_category_palette(order)

    stats = combined_df.groupby('experiment_type')['majority_support'].agg(['count', 'mean', 'std', 'median'])
    for exp_type, row in stats.iterrows():
        print(f"majority_support - {exp_type}: n={int(row['count'])}, mean={row['mean']:.4f}, std={row['std']:.4f}, median={row['median']:.4f}")

    min_val = combined_df['majority_support'].min()
    max_val = combined_df['majority_support'].max()
    lower_margin = 0.08 if min_val > 0.1 else 0.04
    upper_margin = 0.06
    y_min = max(-0.02, min_val - lower_margin)
    y_max = min(1.05, max_val + upper_margin)
    if y_min >= y_max:
        y_min = max(-0.02, min_val - 0.02)
        y_max = min(1.05, max_val + 0.03)

    box_kw = dict(
        width=0.55,
        linewidth=1.8,
        fliersize=0,
        boxprops=dict(alpha=0.85, edgecolor='#1B1F3B', linewidth=1.6),
        medianprops=dict(color='#1B1F3B', linewidth=2.2),
        whiskerprops=dict(color='#1B1F3B', linewidth=1.6),
        capprops=dict(color='#1B1F3B', linewidth=1.6)
    )

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=combined_df,
        x='experiment_type',
        y='majority_support',
        order=order,
        palette=palette,
        **box_kw
    )
    sns.stripplot(
        data=combined_df,
        x='experiment_type',
        y='majority_support',
        order=order,
        color='#2E4057',
        alpha=0.45,
        jitter=0.18,
        size=4.2,
        linewidth=0.4,
        edgecolor='#F7F9F9'
    )

    plt.title('Majority Support by Experiment Type', fontsize=22, fontweight='bold')
    plt.xlabel('Experiment Type', fontsize=26, fontweight='bold')
    plt.ylabel('Majority Support Proportion', fontsize=26, fontweight='bold')
    plt.ylim([y_min, y_max])
    plt.xticks(rotation=0, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.grid(axis='y', color='#D7DBDD', alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'majority_support_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")
    print("多数支持比例箱线图绘制完成")


def save_results(all_results, output_dir):
    """保存详细结果到CSV文件"""
    print("\n=== 保存详细结果 ===")
    
    # 保存每个实验类型的结果
    for exp_type, pair_results in all_results.items():
        df = pd.DataFrame(pair_results)
        output_path = os.path.join(output_dir, f'{exp_type}_consistency_detail.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  已保存: {output_path}")
    
    # 保存汇总统计
    summary_data = []
    plot_configs = [
        {'name': 'main_sub', 'types': ['main', 'sub']},
        {'name': 'nn_main_nn_sub', 'types': ['nn_main', 'nn_sub']},
        {'name': 'all', 'types': ['nn_main', 'nn_sub', 'main', 'sub']}
    ]
    
    for config in plot_configs:
        all_scores = []
        for exp_type in config['types']:
            if exp_type in all_results:
                for pair_result in all_results[exp_type]:
                    consistency = pair_result['consistency']
                    if not np.isnan(consistency):
                        all_scores.append(consistency)
        
        if len(all_scores) > 0:
            summary_data.append({
                'Group': config['name'],
                'N': len(all_scores),
                'Mean': np.mean(all_scores),
                'Std': np.std(all_scores),
                'Median': np.median(all_scores),
                'Min': np.min(all_scores),
                'Max': np.max(all_scores)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'consistency_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  已保存汇总: {summary_path}")
    
    # 收集所有音频对的详细信息
    all_pairs_info = []
    for exp_type, pair_results in all_results.items():
        for pair_result in pair_results:
            consistency = pair_result['consistency']
            if not np.isnan(consistency):
                all_pairs_info.append({
                    'experiment_type': exp_type,
                    'pair_name': pair_result['pair_name'],
                    'audio_a': pair_result['audio_a'],
                    'audio_b': pair_result['audio_b'],
                    'consistency': consistency,
                    'p1': pair_result['p1'],
                    'p2': pair_result['p2'],
                    'n_ab': pair_result['n_ab'],
                    'n_ba': pair_result['n_ba']
                })
    
    # 找出一致性得分小于0.5的音频对
    low_consistency_pairs = [p for p in all_pairs_info if p['consistency'] < 0.5]
    low_consistency_pairs.sort(key=lambda x: x['consistency'])
    
    # 找出得分最高的10个音频对
    high_consistency_pairs = sorted(all_pairs_info, key=lambda x: x['consistency'], reverse=True)[:10]
    
    # 保存详细信息到文本文件
    detail_path = os.path.join(output_dir, 'consistency_details.txt')
    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("一致性得分分析详情\n")
        f.write("=" * 80 + "\n\n")
        
        # 一致性得分小于0.5的音频对
        f.write(f"一致性得分 < 0.5 的音频对 (共 {len(low_consistency_pairs)} 个):\n")
        f.write("-" * 80 + "\n")
        if len(low_consistency_pairs) > 0:
            for i, pair in enumerate(low_consistency_pairs, 1):
                f.write(f"{i}. {pair['pair_name']}\n")
                f.write(f"   实验类型: {pair['experiment_type']}\n")
                f.write(f"   音频A: {pair['audio_a']}\n")
                f.write(f"   音频B: {pair['audio_b']}\n")
                f.write(f"   一致性得分: {pair['consistency']:.4f}\n")
                f.write(f"   P1 (A左B右选右): {pair['p1']:.4f} (n={pair['n_ab']})\n")
                f.write(f"   P2 (A右B左选左): {pair['p2']:.4f} (n={pair['n_ba']})\n")
                f.write("\n")
        else:
            f.write("无\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"一致性得分最高的前10个音频对:\n")
        f.write("-" * 80 + "\n")
        if len(high_consistency_pairs) > 0:
            for i, pair in enumerate(high_consistency_pairs, 1):
                f.write(f"{i}. {pair['pair_name']}\n")
                f.write(f"   实验类型: {pair['experiment_type']}\n")
                f.write(f"   音频A: {pair['audio_a']}\n")
                f.write(f"   音频B: {pair['audio_b']}\n")
                f.write(f"   一致性得分: {pair['consistency']:.4f}\n")
                f.write(f"   P1 (A左B右选右): {pair['p1']:.4f} (n={pair['n_ab']})\n")
                f.write(f"   P2 (A右B左选左): {pair['p2']:.4f} (n={pair['n_ba']})\n")
                f.write("\n")
        else:
            f.write("无\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("统计信息:\n")
        f.write("-" * 80 + "\n")
        f.write(f"总音频对数: {len(all_pairs_info)}\n")
        if len(all_pairs_info) > 0:
            f.write(f"一致性得分 < 0.5: {len(low_consistency_pairs)} ({len(low_consistency_pairs)/len(all_pairs_info)*100:.2f}%)\n")
        else:
            f.write(f"一致性得分 < 0.5: {len(low_consistency_pairs)}\n")
        if len(all_pairs_info) > 0:
            all_scores = [p['consistency'] for p in all_pairs_info]
            f.write(f"平均一致性得分: {np.mean(all_scores):.4f}\n")
            f.write(f"中位数一致性得分: {np.median(all_scores):.4f}\n")
            f.write(f"标准差: {np.std(all_scores):.4f}\n")
            f.write(f"最小值: {np.min(all_scores):.4f}\n")
            f.write(f"最大值: {np.max(all_scores):.4f}\n")
    
    print(f"  已保存详细信息: {detail_path}")
    print(f"  一致性得分 < 0.5 的音频对: {len(low_consistency_pairs)} 个")
    print(f"  一致性得分最高的10个音频对已列出")
    
    # 保存到CSV文件便于查看
    if len(low_consistency_pairs) > 0:
        low_df = pd.DataFrame(low_consistency_pairs)
        low_csv_path = os.path.join(output_dir, 'low_consistency_pairs.csv')
        low_df.to_csv(low_csv_path, index=False, encoding='utf-8-sig')
        print(f"  已保存低一致性音频对CSV: {low_csv_path}")
    
    if len(high_consistency_pairs) > 0:
        high_df = pd.DataFrame(high_consistency_pairs)
        high_csv_path = os.path.join(output_dir, 'top10_consistency_pairs.csv')
        high_df.to_csv(high_csv_path, index=False, encoding='utf-8-sig')
        print(f"  已保存最高一致性音频对CSV: {high_csv_path}")


def main():
    print("=== 开始音频对左右位置一致性分析 ===")
    
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, 'consistency_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载音频对列表
    print("\n步骤1: 加载音频对列表")
    list_path = os.path.join(project_root, 'pair_list.npy')
    
    if os.path.exists(list_path):
        pair_names = np.load(list_path, allow_pickle=True)
        print(f"成功加载: {len(pair_names)} 个音频对")
    else:
        raise FileNotFoundError("未找到 pair_list.npy，请先运行 pair_feature_diff_extraction.py")
    
    # 2. 加载人类决策数据
    print("\n步骤2: 加载人类决策数据")
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
    
    # 3. 计算一致性得分
    print("\n步骤3: 计算一致性得分")
    all_results = calculate_consistency_scores(pair_names, all_trials)
    
    # 4. 绘制提琴图
    plot_violin_consistency(all_results, output_dir)
    
    # 5. 保存结果
    save_results(all_results, output_dir)
    
    # 6. 个体一致性和群体共识度分析
    print("\n步骤6: 个体一致性和群体共识度分析")
    all_trials_with_subject = load_all_human_decisions_with_subject(txt_files, csv_files, mapping_files)
    individual_results = calculate_individual_consistency_and_consensus(pair_names, all_trials_with_subject)
    
    # 保存个体一致性结果
    for exp_type, pair_results in individual_results.items():
        df = pd.DataFrame(pair_results)
        output_path = os.path.join(output_dir, f'{exp_type}_individual_consistency.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  已保存: {output_path}")
    
    # 7. 绘制个体一致性和群体共识度可视化
    plot_individual_analysis(individual_results, output_dir)

    # 8. 绘制多数支持比例
    plot_majority_support(all_results, output_dir)
    
    print("\n=== 分析完成 ===")
    print(f"结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()

