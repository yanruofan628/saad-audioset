#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

print("=== 测试TXT文件读取 ===")

def parse_experiment_data_txt(file_path):
    """解析TXT格式的实验数据文件"""
    print(f"正在解析TXT实验数据: {file_path}")
    
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
    
    print(f"文件总行数: {len(lines)}")
    
    # 找到trial数据开始的位置
    trial_count = 0
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
                trial_count += 1
                if trial_count <= 3:  # 只打印前3个trial的信息
                    print(f"  Trial {trial_count}: {trial_data['wavfile']} -> {trial_data['response']}")
    
    print(f"成功解析 {len(trials)} 个trial")
    return trials

# 测试TXT文件
txt_file = r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt"

if os.path.exists(txt_file):
    print(f"文件存在: {os.path.basename(txt_file)}")
    try:
        trials = parse_experiment_data_txt(txt_file)
        print(f"解析成功，共 {len(trials)} 个trial")
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"文件不存在: {txt_file}")

print("测试完成")

