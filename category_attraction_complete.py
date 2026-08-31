#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from collections import defaultdict
from scipy import stats
import statsmodels.api as sm
from scipy.stats import binomtest
import math

print("=== 大类吸引力分析工具 ===")
print("开始执行...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CATEGORY_COLOR_MAP = {
    'music': '#FF6B6B',
    'speech': '#FFC857',
    'High Ecology': '#45B7D1',
    'Low Ecology': '#96CEB4'
}
DEFAULT_CATEGORY_COLOR = '#8D99AE'

def parse_experiment_data_txt(file_path):
    """解析TXT格式的实验数据文件"""
    print(f"正在解析TXT实验数据: {os.path.basename(file_path)}")
    
    trials = []
    # 尝试不同的编码方式
    encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"成功使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法使用任何编码读取文件: {file_path}")
    
    # 找到trial数据开始的位置
    for i, line in enumerate(lines):
        if line.strip().startswith('Level: 3') and i+1 < len(lines) and '*** LogFrame Start ***' in lines[i+1]:
            # 这是一个trial的开始
            trial_data = {}
            
            # 解析trial信息
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
    
    print(f"成功解析 {len(trials)} 个trial")
    return trials

def parse_experiment_data_csv(file_path):
    """解析CSV格式的实验数据文件"""
    print(f"正在解析CSV实验数据: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []
    
    trials = []
    for _, row in df.iterrows():
        # 从CSV中提取信息
        audio_file = row['音频文件']
        choice = row['选择']
        
        # 将选择转换为数字：left=1, right=2
        if choice.lower() == 'left':
            response = 1
        elif choice.lower() == 'right':
            response = 2
        else:
            continue  # 跳过无效选择
        
        trials.append({
            'wavfile': audio_file,
            'response': response
        })
    
    print(f"成功解析 {len(trials)} 个trial")
    return trials

def parse_audio_mapping(file_path):
    """解析音频映射"""
    print(f"解析映射文件: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        for _, row in df.iterrows():
            new_name = row['new_name']
            original_name = row['original_name']
            mapping[new_name] = original_name
        
        print(f"成功解析 {len(mapping)} 个文件映射")
        return mapping
    except Exception as e:
        print(f"读取映射文件失败: {e}")
        return {}

def parse_audio_filename(filename):
    """解析音频文件名"""
    try:
        # 移除文件扩展名
        name_without_ext = filename.replace('.wav', '')
        
        # 检查实验类型
        if name_without_ext.startswith('nn_main_'):
            experiment_type = 'nn_main'
            name_part = name_without_ext[8:]  # 移除nn_main_前缀
        elif name_without_ext.startswith('nn_sub_'):
            experiment_type = 'nn_sub'
            name_part = name_without_ext[7:]  # 移除nn_sub_前缀
        elif name_without_ext.startswith('main_'):
            experiment_type = 'main'
            name_part = name_without_ext[5:]  # 移除main_前缀
        elif name_without_ext.startswith('sub_'):
            experiment_type = 'sub'
            name_part = name_without_ext[4:]  # 移除sub_前缀
        else:
            return None, None, None
        
        # 按+分割左右声道
        if '+' not in name_part:
            return None, None, None
        
        left_part, right_part = name_part.split('+', 1)
        
        # 提取左声道类别（第一个_前的部分）
        left_underscore_pos = left_part.find('_')
        if left_underscore_pos == -1:
            return None, None, None
        left_category = left_part[:left_underscore_pos]
        
        # 提取右声道类别（第一个_前的部分）
        right_underscore_pos = right_part.find('_')
        if right_underscore_pos == -1:
            return None, None, None
        right_category = right_part[:right_underscore_pos]
        
        return experiment_type, left_category, right_category
        
    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return None, None, None

def get_main_category(sub_category):
    """根据子类别获取主类别"""
    main_categories = {
        'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
        'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
        'music': ['Bass drum', 'Funny music', 'Sad music']
    }
    
    for main_cat, sub_cats in main_categories.items():
        if sub_category in sub_cats:
            return main_cat
    return None

def analyze_session(data_file, mapping_file, file_type='txt'):
    """分析单个session的数据"""
    print(f"\n=== 分析Session: {os.path.basename(data_file)} ===")
    
    # 提取被试ID（规范化：去除数字后缀、空格、下划线）
    def extract_subject_id(path, file_type):
        try:
            if file_type == 'csv':
                base = os.path.splitext(os.path.basename(path))[0]
                # 先去除下划线后的数字部分（如 _1, _2），然后取第一部分
                raw_id = base.split('_')[0]
                # 如果文件名是 "Liu Yaorui2" 这样的格式（没有下划线），需要处理空格和数字
                if '_' not in base:
                    # 直接使用整个文件名作为 raw_id，后面会处理空格和数字
                    raw_id = base
            else:
                # txt 的被试名在父目录名
                parent = os.path.basename(os.path.dirname(path))
                raw_id = parent.split('_')[0]
            
            # 规范化ID：去除空格、去除数字后缀
            import re
            # 先去除空格（处理 "Liu Yaorui2" -> "LiuYaorui2"）
            cleaned_id = raw_id.replace(' ', '')
            # 去除所有数字后缀（如 jiachen1017 -> jiachen, aiwenkai2 -> aiwenkai, LiuYaorui2 -> LiuYaorui）
            cleaned_id = re.sub(r'\d+$', '', cleaned_id)
            return cleaned_id
        except Exception:
            return 'unknown'

    subject_id = extract_subject_id(data_file, file_type)

    # 解析数据
    if file_type == 'txt':
        trials = parse_experiment_data_txt(data_file)
    else:
        trials = parse_experiment_data_csv(data_file)
    
    mapping = parse_audio_mapping(mapping_file)
    
    # 分析每个trial
    session_results = []
    
    for trial in trials:
        wavfile = trial['wavfile']
        response = trial['response']
        
        # 从wavfile中提取文件名
        filename = os.path.basename(wavfile)
        
        # 获取原始文件名
        if filename not in mapping:
            continue
        
        original_filename = mapping[filename]
        
        # 解析文件名
        experiment_type, left_category, right_category = parse_audio_filename(original_filename)
        
        if experiment_type is None:
            continue
        
        # 获取主类别（用于main和nn_main类型）
        left_main = get_main_category(left_category)
        right_main = get_main_category(right_category)
        
        # 记录结果
        result_data = {
            'experiment_type': experiment_type,
            'left_category': left_category,
            'right_category': right_category,
            'response': response,
            'original_filename': original_filename,
            'subject_id': subject_id
        }
        
        # 添加主类别信息（大类对比和小类对比都需要）
        if left_main is None or right_main is None:
            # 如果是小类但没有主类别信息，跳过
            if experiment_type in ['sub', 'nn_sub']:
                continue
        else:
            result_data['left_main'] = left_main
            result_data['right_main'] = right_main
        
        # 对于小类对比，确保左右类别属于同一大类（同一大类下的两个小类配对）
        if experiment_type in ['sub', 'nn_sub']:
            if left_main != right_main:
                continue  # 跳过不同大类的小类配对
        
        session_results.append(result_data)
    
    print(f"成功分析 {len(session_results)} 个trial")
    return session_results

def calculate_attraction_with_counts(all_results, experiment_type):
    """计算吸引力并返回原始计数数据"""
    print(f"\n=== 计算{experiment_type}类型吸引力和原始计数 ===")
    
    # 只处理指定实验类型的数据
    filtered_results = [r for r in all_results if r['experiment_type'] == experiment_type]
    
    print(f"找到 {len(filtered_results)} 个{experiment_type}类型的trial")
    
    if len(filtered_results) == 0:
        return [], {}
    
    # 收集每个类别的原始选择数据
    category_counts = defaultdict(lambda: {'total': 0, 'selected': 0})
    
    for result in filtered_results:
        if experiment_type in ['main', 'nn_main']:
            # 大类对比，使用主类别
            left_category = result['left_main']
            right_category = result['right_main']
            # 排除自配对（左右同一大类）
            if left_category == right_category:
                continue
        else:
            # 小类对比，使用子类别
            left_category = result['left_category']
            right_category = result['right_category']
        
        response = result['response']
        
        # 对于每个类别，记录它被选择的次数和总出现次数
        if response == 1:  # 选择左声道
            category_counts[left_category]['total'] += 1
            category_counts[left_category]['selected'] += 1
            category_counts[right_category]['total'] += 1
            category_counts[right_category]['selected'] += 0
        else:  # 选择右声道
            category_counts[left_category]['total'] += 1
            category_counts[left_category]['selected'] += 0
            category_counts[right_category]['total'] += 1
            category_counts[right_category]['selected'] += 1
    
    # 计算吸引力和统计
    attraction_data = []
    
    for category, counts in category_counts.items():
        total = counts['total']
        selected = counts['selected']
        attraction = selected / total if total > 0 else 0.5
        
        attraction_data.append({
            'category': category,
            'attraction': attraction,
            'total_opportunities': total,
            'selected_times': selected
        })
    
    print(f"计算了 {len(attraction_data)} 个类别的吸引力")
    return attraction_data, dict(category_counts)

def calculate_attraction(all_results, experiment_type):
    """计算吸引力（用于小提琴图）；每个点包含其配对大类，且排除自配对
    额外返回被试层面的平均得分 subject_points: [{category, subject_id, subject_score}]"""
    print(f"\n=== 计算{experiment_type}类型吸引力 ===")
    
    # 只处理指定实验类型的数据
    filtered_results = [r for r in all_results if r['experiment_type'] == experiment_type]
    
    print(f"找到 {len(filtered_results)} 个{experiment_type}类型的trial")
    
    if len(filtered_results) == 0:
        return []
    
    # 按刺激对分组
    stimulus_pairs = defaultdict(list)
    
    for result in filtered_results:
        if experiment_type in ['main', 'nn_main']:
            # 大类对比，使用主类别
            left_category = result['left_main']
            right_category = result['right_main']
        else:
            # 小类对比，使用子类别
            left_category = result['left_category']
            right_category = result['right_category']
        
        response = result['response']
        
        # 创建刺激对标识符（排序确保一致性）
        pair_key = tuple(sorted([left_category, right_category]))
        
        # 记录选择结果
        stimulus_pairs[pair_key].append({
            'left_category': left_category,
            'right_category': right_category,
            'response': response,
            'subject_id': result.get('subject_id', 'unknown'),
        })
    
    # 对于小提琴图，我们需要收集每个类别在所有刺激对中的选择数据
    # 同时记录该数据点对应的“配对类别”，并排除自配对
    attraction_points = []
    # 被试-大类聚合：统计每位被试在每个大类的选择比例（排除自配对）
    subject_category_counts = defaultdict(lambda: {'selected': 0, 'total': 0})
    
    for pair_key, trials in stimulus_pairs.items():
        cat1, cat2 = pair_key
        
        # 计算cat1的吸引力
        cat1_left_trials = [t for t in trials if t['left_category'] == cat1]
        cat1_right_trials = [t for t in trials if t['right_category'] == cat1]
        
        cat1_attraction_left = sum(1 for t in cat1_left_trials if t['response'] == 1) / len(cat1_left_trials) if cat1_left_trials else 0.5
        cat1_attraction_right = sum(1 for t in cat1_right_trials if t['response'] == 2) / len(cat1_right_trials) if cat1_right_trials else 0.5
        
        cat1_attraction = (cat1_attraction_left + cat1_attraction_right) / 2
        
        # 计算cat2的吸引力
        cat2_left_trials = [t for t in trials if t['left_category'] == cat2]
        cat2_right_trials = [t for t in trials if t['right_category'] == cat2]
        
        cat2_attraction_left = sum(1 for t in cat2_left_trials if t['response'] == 1) / len(cat2_left_trials) if cat2_left_trials else 0.5
        cat2_attraction_right = sum(1 for t in cat2_right_trials if t['response'] == 2) / len(cat2_right_trials) if cat2_right_trials else 0.5
        
        cat2_attraction = (cat2_attraction_left + cat2_attraction_right) / 2
        
        # 为每个类别添加一个数据点，并记录其配对的大类；排除自配对
        if cat1 != cat2:
            attraction_points.append({
                'category': cat1,
                'attraction': cat1_attraction,
                'pair_category': cat2
            })
            attraction_points.append({
                'category': cat2,
                'attraction': cat2_attraction,
                'pair_category': cat1
            })

            # 统计被试层面的数据
            for t in trials:
                subj = t.get('subject_id', 'unknown')
                # 对 cat1 作为目标类的计数
                if t['left_category'] == cat1:
                    subject_category_counts[(subj, cat1)]['total'] += 1
                    if t['response'] == 1:
                        subject_category_counts[(subj, cat1)]['selected'] += 1
                if t['right_category'] == cat1:
                    subject_category_counts[(subj, cat1)]['total'] += 1
                    if t['response'] == 2:
                        subject_category_counts[(subj, cat1)]['selected'] += 1
                # 对 cat2 作为目标类的计数
                if t['left_category'] == cat2:
                    subject_category_counts[(subj, cat2)]['total'] += 1
                    if t['response'] == 1:
                        subject_category_counts[(subj, cat2)]['selected'] += 1
                if t['right_category'] == cat2:
                    subject_category_counts[(subj, cat2)]['total'] += 1
                    if t['response'] == 2:
                        subject_category_counts[(subj, cat2)]['selected'] += 1
    
    # 直接返回包含配对信息的数据点
    attraction_data = attraction_points
    # 汇总被试层面的平均分
    subject_points = []
    for (subj, cat), cnt in subject_category_counts.items():
        if cnt['total'] > 0:
            subject_points.append({
                'subject_id': subj,
                'category': cat,
                'subject_score': cnt['selected'] / cnt['total']
            })
    
    print(f"生成了 {len(attraction_data)} 个吸引力数据点（用于小提琴图），{len(subject_points)} 个被试散点")
    return attraction_data, subject_points

def perform_statistical_tests(attraction_data_with_counts):
    """执行统计检验"""
    results = {
        'binomial_tests': [],
        'pairwise_comparisons': []
    }
    
    # 1. 二项检验：每个类别 vs 0.5
    for item in attraction_data_with_counts:
        category = item['category']
        selected = item['selected_times']
        total = item['total_opportunities']
        attraction = item['attraction']
        
        # 单侧二项检验
        if attraction > 0.5:
            result = binomtest(selected, total, p=0.5, alternative='greater')
        else:
            result = binomtest(selected, total, p=0.5, alternative='less')
        
        p_value = result.pvalue
        
        results['binomial_tests'].append({
            'category': category,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    # 2. 成对比较：两个独立比例的检验
    n = len(attraction_data_with_counts)
    for i in range(n):
        for j in range(i+1, n):
            cat1 = attraction_data_with_counts[i]
            cat2 = attraction_data_with_counts[j]
            
            # 两个独立比例的z检验
            n1 = cat1['total_opportunities']
            k1 = cat1['selected_times']
            n2 = cat2['total_opportunities']
            k2 = cat2['selected_times']
            
            p1 = k1 / n1
            p2 = k2 / n2
            
            # 计算z统计量
            p_combined = (k1 + k2) / (n1 + n2)
            se = math.sqrt(p_combined * (1 - p_combined) * (1/n1 + 1/n2))
            
            if se > 0:
                z = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # 双侧检验
            else:
                p_value = 1.0
            
            # 不使用Bonferroni校正，直接使用p<0.05
            is_significant = p_value < 0.05
            
            results['pairwise_comparisons'].append({
                'category1': cat1['category'],
                'category2': cat2['category'],
                'p_value': p_value,
                'significant': is_significant
            })
    
    return results

def plot_results_with_stats(attraction_data_with_counts, output_dir, experiment_type):
    """绘制带统计检验的柱状图"""
    print(f"\n=== 绘制{experiment_type}类型结果（带统计检验） ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if not attraction_data_with_counts:
        print("没有数据可以绘制")
        return
    
    # 执行统计检验
    stats_results = perform_statistical_tests(attraction_data_with_counts)
    
    # 打印所有成对比较结果
    print(f"\n成对比较结果：")
    print(f"总比较数: {len(stats_results['pairwise_comparisons'])}")
    significant_count = sum(1 for c in stats_results['pairwise_comparisons'] if c['significant'])
    print(f"显著比较数: {significant_count}")
    
    # 转换为DataFrame
    df = pd.DataFrame(attraction_data_with_counts)
    
    # 设置图形风格
    sns.set_theme(style="whitegrid")
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    
    categories = df['category'].values
    attractions = df['attraction'].values
    
    ax = plt.gca()
    bar_colors = [CATEGORY_COLOR_MAP.get(cat, DEFAULT_CATEGORY_COLOR) for cat in categories]
    bars = ax.bar(categories, attractions, 
                  color=bar_colors,
                  alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签和统计显著性
    y_max = 0
    for i, (bar, category, attraction) in enumerate(zip(bars, categories, attractions)):
        height = bar.get_height()
        y_max = max(y_max, height)
        label_y = height + 0.012
        label_x = bar.get_x() + bar.get_width() / 2.
        
        # 数值标签
        ax.text(label_x, label_y,
                f'{height:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # 二项检验显著性
        binom_test = stats_results['binomial_tests'][i]
        if binom_test['significant']:
            # 在数值标签右侧添加灰色星号
            ax.text(label_x + bar.get_width() * 0.3, label_y - 0.01,
                    '*', ha='left', va='bottom', fontsize=20, color='gray', fontweight='bold')
    
    # 添加成对比较的连线和显著性（括号形，防止重叠）
    n = len(categories)
    significant_comparisons = [comp for comp in stats_results['pairwise_comparisons'] if comp['significant']]
    
    max_annotation_height = y_max
    if significant_comparisons:
        # 为每条连线设置不同的高度（防止重叠）
        line_count = len(significant_comparisons)
        base_height = y_max + 0.06
        # 根据数量自适应步长
        step = 0.045 if line_count <= 4 else 0.035
        tick_len = 0.015  # 两端向下的短竖线长度

        max_comparison_height = base_height + (line_count - 1) * step + 0.02
        max_annotation_height = max(max_annotation_height, max_comparison_height)

        for i, comparison in enumerate(significant_comparisons):
            cat1 = comparison['category1']
            cat2 = comparison['category2']
            
            idx1 = list(categories).index(cat1)
            idx2 = list(categories).index(cat2)
            
            x1 = bars[idx1].get_x() + bars[idx1].get_width() / 2
            x2 = bars[idx2].get_x() + bars[idx2].get_width() / 2
            if x1 > x2:
                x1, x2 = x2, x1
            
            # 每条连线使用递增的高度
            height = base_height + i * step
            
            # 画括号形：中间水平线 + 两端向下短竖线
            ax.plot([x1, x2], [height, height], color='black', linewidth=1.5)
            ax.plot([x1, x1], [height, height - tick_len], color='black', linewidth=1.5)
            ax.plot([x2, x2], [height, height - tick_len], color='black', linewidth=1.5)
            
            # 连线正中添加红色星号
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, height - 0.01,
                    '*', ha='center', va='bottom', fontsize=16, color='red', fontweight='bold')
    
    # 添加Chance Level参考线
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
    
    # 美化图表
    plt.title('Average Attraction by Category', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Average Attraction', fontsize=14, fontweight='bold')

    upper_limit = max(max_annotation_height + 0.08, y_max + 0.12)
    ax.set_ylim(0.2, upper_limit)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0, ha='center')
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_category_attraction_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"柱状图已保存: {output_path}")

def plot_results(attraction_data_with_pair, output_dir, experiment_type):
    
    # 将输入数据转换为DataFrame
    # attraction_data_with_pair 仅含配对点；同时尝试从同名函数返回中获取被试散点
    if isinstance(attraction_data_with_pair, tuple):
        attraction_data, subject_points = attraction_data_with_pair
    else:
        attraction_data = attraction_data_with_pair
        subject_points = []
    df = pd.DataFrame(attraction_data)
    # 图B：刺激对吸引力分布小提琴图
    # 根据类别数量调整图形大小
    unique_categories = df['category'].nunique()
    if unique_categories > 8:
        fig_width = max(18, unique_categories * 1.2)
        fig_height = 10
    else:
        fig_width = 14
        fig_height = 9

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    
    # 创建小提琴图
    category_palette = {cat: CATEGORY_COLOR_MAP.get(cat, DEFAULT_CATEGORY_COLOR) for cat in df['category'].unique()}
    vp = sns.violinplot(data=df, x='category', y='attraction', 
                       inner='box', palette=category_palette, saturation=0.9, ax=ax)
    
    # 定义配对大类的颜色映射（与柱状图一致）
    pair_category_colors = {cat: CATEGORY_COLOR_MAP.get(cat, DEFAULT_CATEGORY_COLOR) for cat in CATEGORY_COLOR_MAP}
    
    # 为每个点设置颜色
    if 'pair_category' in df.columns:
        df['point_color'] = df['pair_category'].map(pair_category_colors).fillna(DEFAULT_CATEGORY_COLOR)
    else:
        df['point_color'] = DEFAULT_CATEGORY_COLOR  # 默认颜色
    
    # 在小提琴图上叠加原始数据点，按配对大类着色
    sns.stripplot(data=df, x='category', y='attraction', hue='pair_category',
                  palette=pair_category_colors, alpha=0.85, jitter=True, dodge=False, 
                  size=6, linewidth=1.5, edgecolor='black', legend=False, ax=ax)

    # 注意：被试散点已移至独立图，此处不再叠加
    
    # 添加Chance Level参考线
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
    
    # 美化图表
    ax.set_title('Stimulus Pair Attraction Distribution', fontsize=16, fontweight='bold', pad=24)
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attraction', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12, loc='upper right', frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.25)
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_category_attraction_violin.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"小提琴图已保存: {output_path}")
    
    # 返回subject_points供独立图使用
    return subject_points

def plot_subject_level_results(subject_points, output_dir, experiment_type):
    """绘制被试水平分析图"""
    if len(subject_points) == 0:
        print("没有被试水平数据")
        return
    
    df_subjects = pd.DataFrame(subject_points)
    df_subjects['subject_id'] = df_subjects['subject_id'].astype(str)
    
    # 定义9位被试的固定配色
    subject_colors = {
        'liyanchen': '#1f77b4',
        'shimin': '#ff7f0e',
        'jiachen': '#2ca02c',
        'aiwenkai': '#d62728',
        'lironghua': '#9467bd',
        'mayunmiao': '#8c564b',
        'ShangZiyang': '#e377c2',
        'wjy': '#7f7f7f',
        'LiuYaorui': '#bcbd22',
        'Liu Yaorui': '#bcbd22'
    }
    
    plt.figure(figsize=(12, 8))
    
    # 1. 背景：被试水平的总体分布（箱线图）
    sns.boxplot(data=df_subjects, x='category', y='subject_score', 
               width=0.7, palette="pastel", saturation=0.4, fliersize=0,
               showcaps=True, whis=[0, 100],
               boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2},
               medianprops={'color': 'black', 'linewidth': 2},
               whiskerprops={'color': 'black', 'linewidth': 2},
               capprops={'color': 'black', 'linewidth': 2})
    
    # 2. 个体轨迹线
    sns.lineplot(data=df_subjects, x='category', y='subject_score',
                units='subject_id', estimator=None,
                hue='subject_id', palette=subject_colors,
                alpha=0.4, linewidth=1)
    
    # 3. 个体数据点
    sns.stripplot(data=df_subjects, x='category', y='subject_score',
                 hue='subject_id', palette=subject_colors,
                 jitter=True, size=5, alpha=0.7, edgecolor='black', linewidth=0.8)
    
    # 4. 总体趋势线（基于被试均值）
    sns.lineplot(data=df_subjects, x='category', y='subject_score',
                estimator='mean', errorbar='sd',
                color='blue', linewidth=3, marker='o', markersize=8)
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.title('Individual Patterns and Group-Level Effects\n(All analysis at subject level)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Subject-Level Attraction Score', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0, ha='center')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_category_attraction_subjects.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"被试水平图已保存: {output_path}")

def calculate_subcategory_attraction(all_results, experiment_type):
    """计算小类吸引力（sub或nn_sub），返回包含主类别信息的数据"""
    print(f"\n=== 计算{experiment_type}类型小类吸引力 ===")
    
    # 只处理指定实验类型的数据
    filtered_results = [r for r in all_results if r['experiment_type'] == experiment_type]
    
    print(f"找到 {len(filtered_results)} 个{experiment_type}类型的trial")
    
    if len(filtered_results) == 0:
        return [], []
    
    # 收集每个小类的原始选择数据（包含主类别信息）
    subcategory_counts = defaultdict(lambda: {'total': 0, 'selected': 0})
    # 被试-小类聚合
    subject_subcategory_counts = defaultdict(lambda: {'selected': 0, 'total': 0})
    
    for result in filtered_results:
        # 小类对比，使用子类别
        left_subcategory = result['left_category']
        right_subcategory = result['right_category']
        left_main = result.get('left_main')
        right_main = result.get('right_main')
        subject_id = result.get('subject_id', 'unknown')
        response = result['response']
        
        # 对于每个小类，记录它被选择的次数和总出现次数
        if response == 1:  # 选择左声道
            subcategory_counts[left_subcategory]['total'] += 1
            subcategory_counts[left_subcategory]['selected'] += 1
            subcategory_counts[right_subcategory]['total'] += 1
            subcategory_counts[right_subcategory]['selected'] += 0
            
            # 被试水平统计
            subject_subcategory_counts[(subject_id, left_subcategory)]['total'] += 1
            subject_subcategory_counts[(subject_id, left_subcategory)]['selected'] += 1
            subject_subcategory_counts[(subject_id, right_subcategory)]['total'] += 1
        else:  # 选择右声道
            subcategory_counts[left_subcategory]['total'] += 1
            subcategory_counts[left_subcategory]['selected'] += 0
            subcategory_counts[right_subcategory]['total'] += 1
            subcategory_counts[right_subcategory]['selected'] += 1
            
            # 被试水平统计
            subject_subcategory_counts[(subject_id, left_subcategory)]['total'] += 1
            subject_subcategory_counts[(subject_id, right_subcategory)]['total'] += 1
            subject_subcategory_counts[(subject_id, right_subcategory)]['selected'] += 1
    
    # 计算吸引力和统计
    attraction_data = []
    for subcategory, counts in subcategory_counts.items():
        total = counts['total']
        selected = counts['selected']
        attraction = selected / total if total > 0 else 0.5
        
        # 获取该小类所属的主类别
        main_category = get_main_category(subcategory)
        
        attraction_data.append({
            'sub_category': subcategory,
            'main_category': main_category,
            'attraction_score': attraction,
            'total_opportunities': total,
            'selected_times': selected
        })
    
    # 汇总被试层面的平均分
    subject_points = []
    for (subj, subcat), cnt in subject_subcategory_counts.items():
        if cnt['total'] > 0:
            main_cat = get_main_category(subcat)
            subject_points.append({
                'subject_id': subj,
                'sub_category': subcat,
                'main_category': main_cat,
                'subject_score': cnt['selected'] / cnt['total']
            })
    
    print(f"计算了 {len(attraction_data)} 个小类的吸引力")
    return attraction_data, subject_points

def perform_subcategory_statistical_tests(attraction_data):
    """对小类吸引力数据执行统计检验（按主类别分组）"""
    results = {}
    
    df = pd.DataFrame(attraction_data)
    df = df[df['main_category'].notna()].copy()
    
    # 按主类别分组进行检验
    for main_cat in df['main_category'].unique():
        df_main = df[df['main_category'] == main_cat].copy()
        main_results = {
            'binomial_tests': [],
            'pairwise_comparisons': []
        }
        
        # 1. 二项检验：每个小类 vs 0.5
        for _, row in df_main.iterrows():
            subcategory = row['sub_category']
            selected = row['selected_times']
            total = row['total_opportunities']
            attraction = row['attraction_score']
            
            # 单侧二项检验
            if attraction > 0.5:
                result = binomtest(selected, total, p=0.5, alternative='greater')
            else:
                result = binomtest(selected, total, p=0.5, alternative='less')
            
            p_value = result.pvalue
            
            main_results['binomial_tests'].append({
                'sub_category': subcategory,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        # 2. 成对比较：同一主类别内的小类之间进行比较
        n = len(df_main)
        for i in range(n):
            for j in range(i+1, n):
                subcat1 = df_main.iloc[i]
                subcat2 = df_main.iloc[j]
                
                # 两个独立比例的z检验
                n1 = subcat1['total_opportunities']
                k1 = subcat1['selected_times']
                n2 = subcat2['total_opportunities']
                k2 = subcat2['selected_times']
                
                p1 = k1 / n1
                p2 = k2 / n2
                
                # 计算z统计量
                p_combined = (k1 + k2) / (n1 + n2)
                se = math.sqrt(p_combined * (1 - p_combined) * (1/n1 + 1/n2))
                
                if se > 0:
                    z = (p1 - p2) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # 双侧检验
                else:
                    p_value = 1.0
                
                is_significant = p_value < 0.05
                
                main_results['pairwise_comparisons'].append({
                    'sub_category1': subcat1['sub_category'],
                    'sub_category2': subcat2['sub_category'],
                    'p_value': p_value,
                    'significant': is_significant
                })
        
        results[main_cat] = main_results
    
    return results

def plot_subcategory_barplot(attraction_data, output_dir, experiment_type):
    """绘制小类吸引分数分面柱状图（带统计检验）"""
    print(f"\n=== 绘制{experiment_type}类型小类吸引分数分面柱状图（带统计检验） ===")
    
    if len(attraction_data) == 0:
        print("没有数据可以绘制")
        return
    
    df = pd.DataFrame(attraction_data)
    
    # 确保主类别存在
    df = df[df['main_category'].notna()].copy()
    
    if len(df) == 0:
        print("没有包含主类别信息的数据")
        return
    
    # 执行统计检验
    stats_results = perform_subcategory_statistical_tests(attraction_data)
    
    # 按主类别创建分面图
    main_categories = df['main_category'].unique()
    n_main_cats = len(main_categories)
    
    # 设置图形大小（根据主类别数量调整，为统计标注留出更多空间）
    fig_width = max(12, n_main_cats * 3)
    fig, axes = plt.subplots(1, n_main_cats, figsize=(fig_width, 7), sharey=True)
    
    if n_main_cats == 1:
        axes = [axes]
    
    for idx, main_cat in enumerate(main_categories):
        ax = axes[idx]
        df_main = df[df['main_category'] == main_cat].copy()
        df_main = df_main.sort_values('attraction_score', ascending=False)
        bar_color = CATEGORY_COLOR_MAP.get(main_cat, DEFAULT_CATEGORY_COLOR)
        
        # 绘制柱状图
        bars = ax.bar(df_main['sub_category'], df_main['attraction_score'],
                      color=bar_color, alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # 获取该主类别的统计结果
        main_stats = stats_results.get(main_cat, {'binomial_tests': [], 'pairwise_comparisons': []})
        binom_map = {item['sub_category']: item for item in main_stats.get('binomial_tests', [])}
        
        # 添加数值标签
        y_max = 0
        max_annotation_height = 0
        short_labels = []

        for bar, (_, row) in zip(bars, df_main.iterrows()):
            height = bar.get_height()
            y_max = max(y_max, height)
            label_x = bar.get_x() + bar.get_width() / 2
            label_y = height + 0.012
            short_label = row['sub_category'].split(' ')[0] if isinstance(row['sub_category'], str) else row['sub_category']
            short_labels.append(short_label)
            
            # 数值标签
            ax.text(label_x, label_y,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            max_annotation_height = max(max_annotation_height, label_y)
        
        # 添加Chance Level参考线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
        
        ax.set_title(main_cat, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Subcategory', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Attraction Score', fontsize=12, fontweight='bold')
        upper_limit = max(max_annotation_height + 0.08, y_max + 0.12)
        ax.set_ylim(0.2, upper_limit)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis='x', rotation=0)
        ax.set_xticklabels(short_labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(16)
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{experiment_type.upper()} Subcategory Attraction by Main Category', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_subcategory_attraction_facet.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"分面柱状图已保存: {output_path}")

def plot_subcategory_subjects(subject_points, output_dir, experiment_type):
    """绘制小类被试轨迹分面图"""
    print(f"\n=== 绘制{experiment_type}类型小类被试轨迹分面图 ===")
    
    if len(subject_points) == 0:
        print("没有被试水平数据")
        return
    
    df_subjects = pd.DataFrame(subject_points)
    df_subjects = df_subjects[df_subjects['main_category'].notna()].copy()
    df_subjects['subject_id'] = df_subjects['subject_id'].astype(str)
    
    if len(df_subjects) == 0:
        print("没有包含主类别信息的被试数据")
        return
    
    # 定义被试的固定配色
    subject_colors = {
        'liyanchen': '#1f77b4',
        'shimin': '#ff7f0e',
        'jiachen': '#2ca02c',
        'aiwenkai': '#d62728',
        'lironghua': '#9467bd',
        'mayunmiao': '#8c564b',
        'ShangZiyang': '#e377c2',
        'wjy': '#7f7f7f',
        'LiuYaorui': '#bcbd22',
        'Liu Yaorui': '#bcbd22'
    }
    
    # 按主类别创建分面图
    main_categories = sorted(df_subjects['main_category'].unique())
    n_main_cats = len(main_categories)
    
    # 设置图形大小
    fig_width = max(12, n_main_cats * 4)
    fig, axes = plt.subplots(1, n_main_cats, figsize=(fig_width, 7), sharey=True)
    
    if n_main_cats == 1:
        axes = [axes]
    
    for idx, main_cat in enumerate(main_categories):
        ax = axes[idx]
        df_main = df_subjects[df_subjects['main_category'] == main_cat].copy()
        
        # 1. 背景：被试水平的总体分布（箱线图）
        sns.boxplot(data=df_main, x='sub_category', y='subject_score', 
                   width=0.7, palette="pastel", saturation=0.4, fliersize=0, ax=ax,
                   showcaps=True, whis=[0, 100],
                   boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2},
                   medianprops={'color': 'black', 'linewidth': 2},
                   whiskerprops={'color': 'black', 'linewidth': 2},
                   capprops={'color': 'black', 'linewidth': 2})
        
        # 2. 个体轨迹线
        sns.lineplot(data=df_main, x='sub_category', y='subject_score',
                    units='subject_id', estimator=None,
                    hue='subject_id', palette=subject_colors,
                    alpha=0.4, linewidth=1, ax=ax, legend=False)
        
        # 3. 个体数据点（去掉图例）
        sns.stripplot(data=df_main, x='sub_category', y='subject_score',
                     hue='subject_id', palette=subject_colors,
                     jitter=True, size=4, alpha=0.7, edgecolor='black', linewidth=0.8, ax=ax, legend=False)
        
        # 4. 总体趋势线
        sns.lineplot(data=df_main, x='sub_category', y='subject_score',
                    estimator='mean', errorbar='sd',
                    color='blue', linewidth=2.5, marker='o', markersize=6, ax=ax, legend=False)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_title(main_cat, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Subcategory', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Subject-Level Attraction Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=0)
    short_xticks = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        short_text = text.split(' ')[0] if isinstance(text, str) else text
        short_xticks.append(short_text)
    ax.set_xticklabels(short_xticks)
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{experiment_type.upper()} Individual Patterns by Main Category\n(All analysis at subject level)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_subcategory_subjects_facet.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"分面被试轨迹图已保存: {output_path}")

def main():
    """主函数"""
    print("=== 大类吸引力分析工具 ===")
    print("开始执行...")
    
    # 设置文件路径
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
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        # CSV文件的映射文件
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    ]
    
    output_dir = r"D:\D\research\audioset下载\category_attraction_results"
    
    print(f"TXT文件数量: {len(txt_files)}")
    print(f"CSV文件数量: {len(csv_files)}")
    print(f"映射文件数量: {len(mapping_files)}")
    
    # 分析所有数据
    all_results = []
    
    # 处理TXT文件
    for txt_file, mapping_file in zip(txt_files, mapping_files[:len(txt_files)]):
        results = analyze_session(txt_file, mapping_file, 'txt')
        all_results.extend(results)
    
    # 处理CSV文件
    csv_mapping_files = mapping_files[len(txt_files):]
    for csv_file, mapping_file in zip(csv_files, csv_mapping_files):
        results = analyze_session(csv_file, mapping_file, 'csv')
        all_results.extend(results)
    
    print(f"总共处理了 {len(all_results)} 个trial")
    
    if not all_results:
        print("没有找到有效数据")
        return
    
    # 处理不同类型的实验（先只画main和nn_main）
    main_experiment_types = ['main', 'nn_main']
    
    for exp_type in main_experiment_types:
        # 计算带计数的吸引力数据（用于统计检验）
        attraction_data_with_counts, _ = calculate_attraction_with_counts(all_results, exp_type)
        
        if len(attraction_data_with_counts) > 0:
            # 绘制带统计检验的柱状图
            plot_results_with_stats(attraction_data_with_counts, output_dir, exp_type)
            
            # 保存结果
            df = pd.DataFrame(attraction_data_with_counts)
            csv_path = os.path.join(output_dir, f'{exp_type}_category_attraction_data.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"数据已保存: {csv_path}")
        else:
            print(f"没有{exp_type}类型的数据")
        
        # 计算并绘制小提琴图（返回配对点和被试散点）
        attraction_data, subject_points = calculate_attraction(all_results, exp_type)
        if len(attraction_data) > 0:
            plot_results((attraction_data, subject_points), output_dir, exp_type)
            
            # 绘制被试水平分析图
            plot_subject_level_results(subject_points, output_dir, exp_type)
        else:
            print(f"没有{exp_type}类型的数据用于小提琴图")
    
    # 处理小类实验类型（sub和nn_sub）
    sub_experiment_types = ['sub', 'nn_sub']
    
    for exp_type in sub_experiment_types:
        # 计算小类吸引力数据（包含主类别信息）
        attraction_data, subject_points = calculate_subcategory_attraction(all_results, exp_type)
        
        if len(attraction_data) > 0:
            # 绘制分面柱状图
            plot_subcategory_barplot(attraction_data, output_dir, exp_type)
            
            # 保存结果
            df = pd.DataFrame(attraction_data)
            csv_path = os.path.join(output_dir, f'{exp_type}_subcategory_attraction_data.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"数据已保存: {csv_path}")
        else:
            print(f"没有{exp_type}类型的数据")
        
        # 绘制分面被试轨迹图
        if len(subject_points) > 0:
            plot_subcategory_subjects(subject_points, output_dir, exp_type)
        else:
            print(f"没有{exp_type}类型的被试水平数据")
    
    print(f"\n分析完成！结果保存到: {output_dir}")

if __name__ == "__main__":
    main()

