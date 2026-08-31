#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析音频对的选择一致性

1. 个体一致性分析：
   - 对于左右平衡的试次（AB, BA），找出被试自己选择一致的（AB选A，BA选A）和不一致的（AB选A，BA选B）
   - 标记有多少人选择一致/不一致
   - 分别挑选出几个例子

2. 群体一致性分析：
   - 找出大部分人选择一致并且选同一个选项的音频对
   - 找出选择非常随机的音频对
   - 列出来让用户看
"""
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set


# ============== 复用数据加载函数 ==============

def parse_pair_name(filename):
    """解析音频对名称，提取左右音频标识和实验类型"""
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

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

    def extract_cat_id(part):
        first_underscore_pos = part.find('_')
        if first_underscore_pos == -1:
            return None, None
        cat = part[:first_underscore_pos]
        audio_id = part[first_underscore_pos + 1:]
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
        if line.strip().startswith('Level: 3') and i + 1 < len(lines) and '*** LogFrame Start ***' in lines[i + 1]:
            trial_data = {}
            for j in range(i + 1, min(i + 20, len(lines))):
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

        base_name = os.path.basename(csv_file).replace('.csv', '')
        base_name = base_name.replace(' ', '')
        subject_id = re.sub(r'[\d_]+$', '', base_name).strip('_').strip()

        if not subject_id or subject_id.isdigit():
            match = re.match(r'^([a-zA-Z\u4e00-\u9fa5]+(?:\d+)?[a-zA-Z\u4e00-\u9fa5]*)', base_name)
            if match:
                subject_id = match.group(1)
            else:
                subject_id = base_name

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

    return all_trials


def normalize_pair_key(left_cat, left_id, right_cat, right_id):
    """
    规范化音频对键，使得AB和BA被视为同一个对
    返回排序后的元组
    """
    audio_a = f"{left_cat}_{left_id}"
    audio_b = f"{right_cat}_{right_id}"
    return tuple(sorted([audio_a, audio_b]))


def get_pair_orientation(pair_name):
    """
    判断音频对的方向
    返回: ('A', 'B') 或 ('B', 'A')，其中A是排序后的第一个音频
    """
    parse_result = parse_pair_name(pair_name)
    if parse_result is None or parse_result[0] is None:
        return None, None
    
    left_cat, left_id, right_cat, right_id, _ = parse_result
    audio_a = f"{left_cat}_{left_id}"
    audio_b = f"{right_cat}_{right_id}"
    
    normalized_key = tuple(sorted([audio_a, audio_b]))
    first_audio = normalized_key[0]
    
    if audio_a == first_audio:
        return 'A', 'B'  # AB方向
    else:
        return 'B', 'A'  # BA方向


# ============== 主要分析函数 ==============

def analyze_individual_consistency(all_trials):
    """
    分析个体一致性：对于每个被试，找出AB和BA试次，看他们是否选择一致
    """
    print("\n" + "="*80)
    print("1. 个体一致性分析")
    print("="*80)
    
    # 按被试分组
    subject_trials = defaultdict(list)
    for trial in all_trials:
        subject_trials[trial['subject_id']].append(trial)
    
    # 存储每个被试的一致性信息
    individual_consistency = []
    
    for subject_id, trials in subject_trials.items():
        # 按音频对分组
        pair_trials = defaultdict(list)
        for trial in trials:
            pair_name = trial['original_name']
            parse_result = parse_pair_name(pair_name)
            if parse_result is None or parse_result[0] is None:
                continue
            
            left_cat, left_id, right_cat, right_id, _ = parse_result
            pair_key = normalize_pair_key(left_cat, left_id, right_cat, right_id)
            pair_trials[pair_key].append({
                'pair_name': pair_name,
                'response': trial['response'],
                'left_cat': left_cat,
                'left_id': left_id,
                'right_cat': right_cat,
                'right_id': right_id,
            })
        
        # 检查每个音频对是否有AB和BA两个试次
        for pair_key, trials_list in pair_trials.items():
            if len(trials_list) == 2:
                # 判断方向
                trial1_orientation = get_pair_orientation(trials_list[0]['pair_name'])
                trial2_orientation = get_pair_orientation(trials_list[1]['pair_name'])
                
                if trial1_orientation[0] != trial2_orientation[0]:  # 一个是AB，一个是BA
                    # 确定哪个是AB，哪个是BA
                    if trial1_orientation[0] == 'A':
                        ab_trial = trials_list[0]
                        ba_trial = trials_list[1]
                    else:
                        ab_trial = trials_list[1]
                        ba_trial = trials_list[0]
                    
                    # 提取前缀
                    ab_parse = parse_pair_name(ab_trial['pair_name'])
                    ba_parse = parse_pair_name(ba_trial['pair_name'])
                    ab_prefix = ab_parse[4] if ab_parse and ab_parse[4] else 'unknown'
                    ba_prefix = ba_parse[4] if ba_parse and ba_parse[4] else 'unknown'
                    
                    # 判断选择是否一致
                    # AB选1（左）=选A，BA选2（右）=选A，这样就是一致的
                    # 或者AB选2（右）=选B，BA选1（左）=选B，这样也是一致的
                    
                    # 简化：如果AB选左(1)且BA选右(2)，或者AB选右(2)且BA选左(1)，则一致
                    ab_response = ab_trial['response']
                    ba_response = ba_trial['response']
                    
                    # 判断一致性：如果AB选左(1)且BA选右(2)，说明都选了A（第一个音频）
                    # 如果AB选右(2)且BA选左(1)，说明都选了B（第二个音频）
                    is_consistent = (ab_response == 1 and ba_response == 2) or (ab_response == 2 and ba_response == 1)
                    
                    # 确定选择了哪个音频（基于normalized_key）
                    audio_a, audio_b = pair_key
                    if is_consistent:
                        if ab_response == 1:  # AB选左，说明选的是A
                            selected_audio = audio_a
                        else:  # AB选右，说明选的是B
                            selected_audio = audio_b
                    else:
                        selected_audio = None  # 不一致
                    
                    individual_consistency.append({
                        'subject_id': subject_id,
                        'pair_key': pair_key,
                        'audio_a': audio_a,
                        'audio_b': audio_b,
                        'ab_pair_name': ab_trial['pair_name'],
                        'ba_pair_name': ba_trial['pair_name'],
                        'ab_prefix': ab_prefix,
                        'ba_prefix': ba_prefix,
                        'ab_response': ab_response,
                        'ba_response': ba_response,
                        'is_consistent': is_consistent,
                        'selected_audio': selected_audio,
                    })
    
    # 统计
    consistent_count = sum(1 for item in individual_consistency if item['is_consistent'])
    inconsistent_count = len(individual_consistency) - consistent_count
    
    print(f"\n总共找到 {len(individual_consistency)} 个被试的平衡试次对（AB+BA）")
    print(f"  一致的选择: {consistent_count} ({consistent_count/len(individual_consistency)*100:.1f}%)")
    print(f"  不一致的选择: {inconsistent_count} ({inconsistent_count/len(individual_consistency)*100:.1f}%)")
    
    # 按被试统计
    subject_stats = defaultdict(lambda: {'consistent': 0, 'inconsistent': 0})
    for item in individual_consistency:
        if item['is_consistent']:
            subject_stats[item['subject_id']]['consistent'] += 1
        else:
            subject_stats[item['subject_id']]['inconsistent'] += 1
    
    print(f"\n各被试的一致性统计:")
    for subject_id, stats in sorted(subject_stats.items()):
        total = stats['consistent'] + stats['inconsistent']
        if total > 0:
            consistency_rate = stats['consistent'] / total * 100
            print(f"  {subject_id}: {stats['consistent']}/{total} 一致 ({consistency_rate:.1f}%)")
    
    # 挑选例子
    print(f"\n挑选例子（一致的选择）:")
    consistent_examples = [item for item in individual_consistency if item['is_consistent']]
    for i, item in enumerate(consistent_examples[:5], 1):
        print(f"  例子{i}:")
        print(f"    被试: {item['subject_id']}")
        print(f"    音频对: [{item['ab_prefix']}] {item['audio_a']} vs {item['audio_b']}")
        print(f"    AB试次: [{item['ab_prefix']}] {item['ab_pair_name']} -> 选择{'左' if item['ab_response']==1 else '右'}")
        print(f"    BA试次: [{item['ba_prefix']}] {item['ba_pair_name']} -> 选择{'左' if item['ba_response']==1 else '右'}")
        print(f"    一致选择: {item['selected_audio']}")
    
    print(f"\n挑选例子（不一致的选择）:")
    inconsistent_examples = [item for item in individual_consistency if not item['is_consistent']]
    for i, item in enumerate(inconsistent_examples[:5], 1):
        print(f"  例子{i}:")
        print(f"    被试: {item['subject_id']}")
        print(f"    音频对: [{item['ab_prefix']}] {item['audio_a']} vs {item['audio_b']}")
        print(f"    AB试次: [{item['ab_prefix']}] {item['ab_pair_name']} -> 选择{'左' if item['ab_response']==1 else '右'}")
        print(f"    BA试次: [{item['ba_prefix']}] {item['ba_pair_name']} -> 选择{'左' if item['ba_response']==1 else '右'}")
        print(f"    不一致：AB和BA选择了不同的音频")
    
    return individual_consistency


def analyze_group_consistency(all_trials):
    """
    分析群体一致性：找出大部分人选择一致的和非常随机的音频对
    """
    print("\n" + "="*80)
    print("2. 群体一致性分析")
    print("="*80)
    
    # 按音频对分组
    pair_responses = defaultdict(list)
    for trial in all_trials:
        pair_name = trial['original_name']
        parse_result = parse_pair_name(pair_name)
        if parse_result is None or parse_result[0] is None:
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        pair_key = normalize_pair_key(left_cat, left_id, right_cat, right_id)
        
        # 确定选择了哪个音频（基于normalized_key）
        audio_a, audio_b = pair_key
        pair_orientation = get_pair_orientation(pair_name)
        
        if pair_orientation[0] == 'A':  # AB方向
            # response=1选左，选的是A；response=2选右，选的是B
            selected_audio = audio_a if trial['response'] == 1 else audio_b
        else:  # BA方向
            # response=1选左，选的是B；response=2选右，选的是A
            selected_audio = audio_b if trial['response'] == 1 else audio_a
        
        # 提取前缀
        parse_result = parse_pair_name(pair_name)
        prefix = parse_result[4] if parse_result and parse_result[4] else 'unknown'
        
        pair_responses[pair_key].append({
            'subject_id': trial['subject_id'],
            'pair_name': pair_name,
            'prefix': prefix,
            'selected_audio': selected_audio,
            'response': trial['response'],
        })
    
    # 计算每个音频对的一致性
    group_consistency_results = []
    
    for pair_key, responses in pair_responses.items():
        audio_a, audio_b = pair_key
        total_responses = len(responses)
        
        if total_responses == 0:
            continue
        
        # 获取前缀（取最常见的那个）
        prefixes = [r['prefix'] for r in responses]
        most_common_prefix = max(set(prefixes), key=prefixes.count) if prefixes else 'unknown'
        
        # 统计选择A和B的次数
        count_a = sum(1 for r in responses if r['selected_audio'] == audio_a)
        count_b = sum(1 for r in responses if r['selected_audio'] == audio_b)
        
        # 计算一致性（选择次数多的那个的比例）
        max_count = max(count_a, count_b)
        consistency_rate = max_count / total_responses
        
        # 计算选择概率（选择A的概率）
        prob_a = count_a / total_responses
        
        # 判断是否随机（接近0.5）
        is_random = abs(prob_a - 0.5) < 0.15  # 在35%-65%之间认为是随机的
        
        # 判断是否高度一致（>80%选择同一个）
        is_highly_consistent = consistency_rate > 0.8
        
        group_consistency_results.append({
            'pair_key': pair_key,
            'audio_a': audio_a,
            'audio_b': audio_b,
            'prefix': most_common_prefix,
            'total_responses': total_responses,
            'count_a': count_a,
            'count_b': count_b,
            'prob_a': prob_a,
            'consistency_rate': consistency_rate,
            'is_random': is_random,
            'is_highly_consistent': is_highly_consistent,
            'majority_choice': 'A' if count_a > count_b else 'B' if count_b > count_a else 'Tie',
        })
    
    # 统计
    highly_consistent_count = sum(1 for r in group_consistency_results if r['is_highly_consistent'])
    random_count = sum(1 for r in group_consistency_results if r['is_random'])
    
    print(f"\n总共分析 {len(group_consistency_results)} 个音频对")
    print(f"  高度一致（>80%选择同一个）: {highly_consistent_count} ({highly_consistent_count/len(group_consistency_results)*100:.1f}%)")
    print(f"  随机选择（35%-65%选择A）: {random_count} ({random_count/len(group_consistency_results)*100:.1f}%)")
    
    # 排序：高度一致的按一致性率降序，随机的按接近0.5的程度排序
    highly_consistent = sorted(
        [r for r in group_consistency_results if r['is_highly_consistent']],
        key=lambda x: x['consistency_rate'],
        reverse=True
    )
    
    random_pairs = sorted(
        [r for r in group_consistency_results if r['is_random']],
        key=lambda x: abs(x['prob_a'] - 0.5)
    )
    
    print(f"\n高度一致的音频对（前10个）:")
    for i, result in enumerate(highly_consistent[:10], 1):
        print(f"  {i}. [{result['prefix']}] {result['audio_a']} vs {result['audio_b']}")
        print(f"     选择A: {result['count_a']}/{result['total_responses']} ({result['prob_a']*100:.1f}%)")
        print(f"     选择B: {result['count_b']}/{result['total_responses']} ({(1-result['prob_a'])*100:.1f}%)")
        print(f"     一致性: {result['consistency_rate']*100:.1f}%")
        print(f"     多数选择: {result['majority_choice']}")
    
    print(f"\n随机选择的音频对（前10个）:")
    for i, result in enumerate(random_pairs[:10], 1):
        print(f"  {i}. [{result['prefix']}] {result['audio_a']} vs {result['audio_b']}")
        print(f"     选择A: {result['count_a']}/{result['total_responses']} ({result['prob_a']*100:.1f}%)")
        print(f"     选择B: {result['count_b']}/{result['total_responses']} ({(1-result['prob_a'])*100:.1f}%)")
        print(f"     一致性: {result['consistency_rate']*100:.1f}%")
    
    return group_consistency_results


def analyze_position_bias(individual_consistency):
    """
    分析位置偏好：检查不一致的选择是否由位置偏好导致
    """
    print("\n" + "="*80)
    print("3. 位置偏好分析")
    print("="*80)
    
    # 分析不一致选择的位置模式
    inconsistent_items = [item for item in individual_consistency if not item['is_consistent']]
    
    position_bias_stats = defaultdict(lambda: {'left_preference': 0, 'right_preference': 0, 'total': 0})
    
    for item in inconsistent_items:
        subject_id = item['subject_id']
        # 如果AB选左(1)且BA选左(1)，说明偏好左
        # 如果AB选右(2)且BA选右(2)，说明偏好右
        if item['ab_response'] == 1 and item['ba_response'] == 1:
            position_bias_stats[subject_id]['left_preference'] += 1
            position_bias_stats[subject_id]['total'] += 1
        elif item['ab_response'] == 2 and item['ba_response'] == 2:
            position_bias_stats[subject_id]['right_preference'] += 1
            position_bias_stats[subject_id]['total'] += 1
    
    print(f"\n位置偏好统计（仅统计不一致的选择）:")
    for subject_id in sorted(position_bias_stats.keys()):
        stats = position_bias_stats[subject_id]
        if stats['total'] > 0:
            left_rate = stats['left_preference'] / stats['total'] * 100
            right_rate = stats['right_preference'] / stats['total'] * 100
            print(f"  {subject_id}:")
            print(f"    左偏好: {stats['left_preference']}/{stats['total']} ({left_rate:.1f}%)")
            print(f"    右偏好: {stats['right_preference']}/{stats['total']} ({right_rate:.1f}%)")
    
    # 总体统计
    total_left_pref = sum(s['left_preference'] for s in position_bias_stats.values())
    total_right_pref = sum(s['right_preference'] for s in position_bias_stats.values())
    total_inconsistent = len(inconsistent_items)
    
    print(f"\n总体位置偏好:")
    print(f"  左偏好: {total_left_pref}/{total_inconsistent} ({total_left_pref/total_inconsistent*100:.1f}%)")
    print(f"  右偏好: {total_right_pref}/{total_inconsistent} ({total_right_pref/total_inconsistent*100:.1f}%)")
    print(f"  其他模式: {total_inconsistent - total_left_pref - total_right_pref}/{total_inconsistent}")
    
    return position_bias_stats


def analyze_consistency_patterns(individual_consistency, group_consistency_results):
    """
    综合分析：找出个体一致性和群体一致性的关系
    """
    print("\n" + "="*80)
    print("4. 综合分析：个体一致性与群体一致性的关系")
    print("="*80)
    
    # 按音频对分组个体一致性
    pair_individual_consistency = defaultdict(list)
    pair_prefixes = {}  # 存储每个音频对的前缀
    for item in individual_consistency:
        pair_key = item['pair_key']
        pair_individual_consistency[pair_key].append(item['is_consistent'])
        # 保存前缀（取AB试次的前缀）
        if pair_key not in pair_prefixes:
            pair_prefixes[pair_key] = item.get('ab_prefix', 'unknown')
    
    # 合并群体一致性结果
    group_dict = {r['pair_key']: r for r in group_consistency_results}
    
    # 分析关系
    analysis_results = []
    for pair_key, consistency_list in pair_individual_consistency.items():
        if pair_key in group_dict:
            individual_consistency_rate = sum(consistency_list) / len(consistency_list)
            group_info = group_dict[pair_key]
            
            analysis_results.append({
                'pair_key': pair_key,
                'audio_a': group_info['audio_a'],
                'audio_b': group_info['audio_b'],
                'prefix': pair_prefixes.get(pair_key, group_info.get('prefix', 'unknown')),
                'individual_consistency_rate': individual_consistency_rate,
                'group_consistency_rate': group_info['consistency_rate'],
                'group_is_random': group_info['is_random'],
                'group_is_highly_consistent': group_info['is_highly_consistent'],
                'n_subjects': len(consistency_list),
            })
    
    # 分类分析
    highly_consistent_pairs = [r for r in analysis_results if r['group_is_highly_consistent']]
    random_pairs = [r for r in analysis_results if r['group_is_random']]
    
    if highly_consistent_pairs:
        avg_individual_consistency = np.mean([r['individual_consistency_rate'] for r in highly_consistent_pairs])
        print(f"\n高度一致的音频对（群体>80%）:")
        print(f"  数量: {len(highly_consistent_pairs)}")
        print(f"  平均个体一致性率: {avg_individual_consistency*100:.1f}%")
        print(f"  → 说明：群体高度一致时，个体也倾向于一致")
    
    if random_pairs:
        avg_individual_consistency = np.mean([r['individual_consistency_rate'] for r in random_pairs])
        print(f"\n随机选择的音频对（群体35%-65%）:")
        print(f"  数量: {len(random_pairs)}")
        print(f"  平均个体一致性率: {avg_individual_consistency*100:.1f}%")
        print(f"  → 说明：群体随机时，个体一致性也较低")
    
    # 按实验类型分析
    print(f"\n按实验类型分析:")
    prefix_stats = defaultdict(lambda: {'highly_consistent': 0, 'random': 0, 'total': 0, 'individual_consistency': []})
    
    for result in analysis_results:
        prefix = result['prefix']
        prefix_stats[prefix]['total'] += 1
        prefix_stats[prefix]['individual_consistency'].append(result['individual_consistency_rate'])
        if result['group_is_highly_consistent']:
            prefix_stats[prefix]['highly_consistent'] += 1
        if result['group_is_random']:
            prefix_stats[prefix]['random'] += 1
    
    for prefix in sorted(prefix_stats.keys()):
        stats = prefix_stats[prefix]
        if stats['total'] > 0:
            avg_ind_consistency = np.mean(stats['individual_consistency'])
            highly_rate = stats['highly_consistent'] / stats['total'] * 100
            random_rate = stats['random'] / stats['total'] * 100
            print(f"  {prefix}:")
            print(f"    总数: {stats['total']}")
            print(f"    高度一致: {stats['highly_consistent']} ({highly_rate:.1f}%)")
            print(f"    随机选择: {stats['random']} ({random_rate:.1f}%)")
            print(f"    平均个体一致性率: {avg_ind_consistency*100:.1f}%")
    
    return analysis_results


def save_results(individual_consistency, group_consistency_results, position_bias_stats, analysis_results, output_dir):
    """保存结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存个体一致性结果
    if individual_consistency:
        individual_df = pd.DataFrame(individual_consistency)
        individual_path = os.path.join(output_dir, 'individual_consistency.csv')
        individual_df.to_csv(individual_path, index=False, encoding='utf-8-sig')
        print(f"\n个体一致性结果已保存: {individual_path}")
    
    # 保存群体一致性结果
    if group_consistency_results:
        group_df = pd.DataFrame(group_consistency_results)
        group_path = os.path.join(output_dir, 'group_consistency.csv')
        group_df.to_csv(group_path, index=False, encoding='utf-8-sig')
        print(f"群体一致性结果已保存: {group_path}")
    
    # 保存综合分析结果
    if analysis_results:
        analysis_df = pd.DataFrame(analysis_results)
        analysis_path = os.path.join(output_dir, 'consistency_analysis.csv')
        analysis_df.to_csv(analysis_path, index=False, encoding='utf-8-sig')
        print(f"综合分析结果已保存: {analysis_path}")


def main():
    print("="*80)
    print("音频对选择一致性分析")
    print("="*80)
    
    # 数据文件路径
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt",
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
    
    # 加载数据
    all_trials = load_all_human_decisions(txt_files, csv_files, mapping_files)
    
    # 分析个体一致性
    individual_consistency = analyze_individual_consistency(all_trials)
    
    # 分析群体一致性
    group_consistency_results = analyze_group_consistency(all_trials)
    
    # 分析位置偏好
    position_bias_stats = analyze_position_bias(individual_consistency)
    
    # 综合分析
    analysis_results = analyze_consistency_patterns(individual_consistency, group_consistency_results)
    
    # 保存结果
    output_dir = os.path.join(os.getcwd(), 'selection_consistency_analysis')
    save_results(individual_consistency, group_consistency_results, position_bias_stats, analysis_results, output_dir)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()

