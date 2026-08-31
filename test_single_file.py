#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

print("=== 大类吸引力分析工具 ===")
print("开始执行...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        # 只处理main_开头的文件
        if not filename.startswith('main_'):
            return None, None, None
        
        # 移除文件扩展名和main_前缀
        name_without_ext = filename.replace('.wav', '')
        name_part = name_without_ext[5:]  # 移除main_前缀
        
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
        
        return 'main', left_category, right_category
        
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

def analyze_csv_file(csv_file, mapping_file):
    """分析CSV文件"""
    print(f"\n=== 分析CSV文件: {os.path.basename(csv_file)} ===")
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []
    
    mapping = parse_audio_mapping(mapping_file)
    if not mapping:
        return []
    
    results = []
    
    for _, row in df.iterrows():
        audio_file = row['音频文件']
        choice = row['选择']
        
        # 将选择转换为数字：left=1, right=2
        if choice.lower() == 'left':
            response = 1
        elif choice.lower() == 'right':
            response = 2
        else:
            continue  # 跳过无效选择
        
        # 获取原始文件名
        if audio_file not in mapping:
            continue
        
        original_filename = mapping[audio_file]
        
        # 解析文件名
        experiment_type, left_category, right_category = parse_audio_filename(original_filename)
        
        if experiment_type is None:
            continue
        
        # 获取主类别
        left_main = get_main_category(left_category)
        right_main = get_main_category(right_category)
        
        if left_main is None or right_main is None:
            continue
        
        # 记录结果
        results.append({
            'left_main': left_main,
            'right_main': right_main,
            'response': response,
            'original_filename': original_filename
        })
    
    print(f"成功分析 {len(results)} 个main类型trial")
    return results

def calculate_attraction(all_results):
    """计算吸引力"""
    print("\n=== 计算大类吸引力 ===")
    
    # 按刺激对分组
    stimulus_pairs = defaultdict(list)
    
    for result in all_results:
        left_main = result['left_main']
        right_main = result['right_main']
        response = result['response']
        
        # 创建刺激对标识符（排序确保一致性）
        pair_key = tuple(sorted([left_main, right_main]))
        
        # 记录选择结果
        stimulus_pairs[pair_key].append({
            'left_category': left_main,
            'right_category': right_main,
            'response': response,
        })
    
    # 计算每个大类的吸引力
    attraction_data = []
    
    for pair_key, trials in stimulus_pairs.items():
        if len(pair_key) == 1:
            # 同类别对比
            category = pair_key[0]
            total_trials = len(trials)
            selected_trials = sum(1 for t in trials if t['response'] == 1)  # 假设总是选择左声道
            attraction = selected_trials / total_trials if total_trials > 0 else 0.5
            
            attraction_data.append({
                'category': category,
                'attraction': attraction,
                'pair_type': f"{category} vs {category}",
                'total_trials': total_trials
            })
        else:
            # 跨类别对比
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
            
            # 添加两个类别的吸引力数据
            attraction_data.append({
                'category': cat1,
                'attraction': cat1_attraction,
                'pair_type': f"{cat1} vs {cat2}",
                'total_trials': len(trials)
            })
            
            attraction_data.append({
                'category': cat2,
                'attraction': cat2_attraction,
                'pair_type': f"{cat1} vs {cat2}",
                'total_trials': len(trials)
            })
    
    print(f"计算了 {len(attraction_data)} 个吸引力值")
    return attraction_data

def plot_results(attraction_data, output_dir):
    """绘制结果"""
    print("\n=== 绘制结果 ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(attraction_data)
    
    if df.empty:
        print("没有数据可以绘制")
        return
    
    # 设置图形风格
    sns.set_theme(style="whitegrid")
    
    # 图A：四大类吸引分数柱状图
    plt.figure(figsize=(10, 6))
    
    # 计算每个类别的平均吸引力
    category_means = df.groupby('category')['attraction'].mean().reset_index()
    
    # 创建柱状图
    bars = plt.bar(category_means['category'], category_means['attraction'], 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加Chance Level参考线
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
    
    # 美化图表
    plt.title('四大类吸引分数\n(整体上，哪个类别赢了？)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('类别', fontsize=14, fontweight='bold')
    plt.ylabel('平均吸引力', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'category_attraction_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"柱状图已保存: {output_path}")
    
    # 图B：刺激对吸引力分布小提琴图
    plt.figure(figsize=(12, 8))
    
    # 创建小提琴图
    vp = sns.violinplot(data=df, x='category', y='attraction', 
                       inner='box', palette="Set2", saturation=0.8)
    
    # 在小提琴图上叠加原始数据点
    sns.stripplot(data=df, x='category', y='attraction',
                  color='black', alpha=0.7, jitter=True, dodge=False, 
                  size=6, linewidth=1, edgecolor='white')
    
    # 添加Chance Level参考线
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
    
    # 美化图表
    plt.title('刺激对吸引力分布\n(这个结论有多稳健？)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('类别', fontsize=14, fontweight='bold')
    plt.ylabel('吸引力', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'category_attraction_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"小提琴图已保存: {output_path}")

# 测试单个文件
csv_file = r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv"
mapping_file = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv"
output_dir = r"D:\D\research\audioset下载\category_attraction_results"

print(f"测试文件: {os.path.basename(csv_file)}")
print(f"映射文件: {os.path.basename(mapping_file)}")

# 分析数据
results = analyze_csv_file(csv_file, mapping_file)

if results:
    print(f"找到 {len(results)} 个有效trial")
    
    # 计算吸引力
    attraction_data = calculate_attraction(results)
    
    if attraction_data:
        # 绘制图表
        plot_results(attraction_data, output_dir)
        
        # 保存结果
        df = pd.DataFrame(attraction_data)
        csv_path = os.path.join(output_dir, 'category_attraction_data.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n分析完成！结果保存到: {output_dir}")
    else:
        print("没有计算出吸引力数据")
else:
    print("没有找到有效数据")

print("脚本执行完成")

