#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试断点续做功能
"""

import os
import csv
import json

# 配置参数
AUDIO_DIR = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1"
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"

def test_resume_logic(username):
    """测试断点续做逻辑"""
    
    print(f"=== 测试用户: {username} 的断点续做逻辑 ===")
    
    # 1. 获取所有音频文件
    if not os.path.exists(AUDIO_DIR):
        print(f"错误: 音频目录不存在: {AUDIO_DIR}")
        return
    
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.ogg'))]
    audio_files.sort()
    print(f"音频文件总数: {len(audio_files)}")
    
    # 2. 读取音频顺序文件
    order_file = os.path.join(OUTPUT_DIR, "audio_order.json")
    if not os.path.exists(order_file):
        print("错误: 音频顺序文件不存在")
        return
    
    with open(order_file, 'r', encoding='utf-8') as f:
        order_info = json.load(f)
    
    audio_order = order_info.get('shuffled_indices', [])
    print(f"音频顺序包含: {len(audio_order)} 个索引")
    
    # 3. 模拟断点续做逻辑
    safe_username = ''.join(c for c in username if c not in '\\/:*?"<>|').strip() or 'unknown'
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_username}.csv")
    
    completed_files = set()
    restored_trials = []
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header_skipped = False
                
                for row in reader:
                    if not header_skipped:
                        header_skipped = True
                        continue
                    if not row:
                        continue
                    
                    try:
                        trial_number = int(row[0]) if row[0] else 0
                        audio_file = row[1]
                        choice = row[2]
                        reaction_ms = float(row[3]) if row[3] else 0.0
                        participant_id = row[4]
                        timestamp = row[5]
                        
                        restored_trials.append({
                            'trial_number': trial_number,
                            'audio_file': audio_file,
                            'choice': choice,
                            'reaction_time': reaction_ms,
                            'participant_id': participant_id,
                            'timestamp': timestamp
                        })
                        
                        completed_files.add(audio_file)
                        
                    except Exception as e:
                        print(f"解析行失败: {row}, 错误: {e}")
                        continue
            
            print(f"用户已完成试验数: {len(restored_trials)}")
            print(f"用户已完成音频文件数: {len(completed_files)}")
            
        except Exception as e:
            print(f"读取CSV失败: {e}")
            return
    else:
        print("用户CSV文件不存在")
        return
    
    # 4. 计算剩余试验
    remaining_order = []
    for idx in audio_order:
        audio_file = audio_files[idx]
        if audio_file not in completed_files:
            remaining_order.append(idx)
    
    print(f"剩余待做音频文件数: {len(remaining_order)}")
    
    # 5. 计算试验偏移量
    if restored_trials:
        max_trial_number = max([t['trial_number'] for t in restored_trials])
        trial_offset = max_trial_number
        print(f"用户最大试验编号: {max_trial_number}")
    else:
        trial_offset = 0
        print("用户没有历史试验，从0开始")
    
    print(f"试验偏移量: {trial_offset}")
    
    # 6. 模拟新的试验编号计算
    print(f"\n=== 模拟新的试验编号计算 ===")
    for i, audio_idx in enumerate(remaining_order[:10]):  # 只显示前10个
        trial_number = trial_offset + i + 1
        current_audio = audio_files[audio_idx]
        print(f"剩余试验 {i+1}: 音频={current_audio}, 新编号={trial_number}")
    
    if len(remaining_order) > 10:
        print(f"... 还有 {len(remaining_order) - 10} 个试验")
    
    # 7. 检查是否有重复的试验编号
    if restored_trials:
        trial_numbers = [t['trial_number'] for t in restored_trials]
        if len(trial_numbers) != len(set(trial_numbers)):
            print(f"\n⚠️  警告: 发现重复的试验编号！")
            duplicates = [x for x in trial_numbers if trial_numbers.count(x) > 1]
            print(f"重复的试验编号: {set(duplicates)}")
        else:
            print(f"\n✅ 试验编号无重复")
    
    # 8. 检查试验编号的连续性
    if restored_trials:
        trial_numbers = sorted([t['trial_number'] for t in restored_trials])
        print(f"试验编号范围: {min(trial_numbers)} - {max(trial_numbers)}")
        
        # 检查是否有跳跃
        missing_numbers = []
        for i in range(1, max(trial_numbers) + 1):
            if i not in trial_numbers:
                missing_numbers.append(i)
        
        if missing_numbers:
            print(f"缺失的试验编号: {missing_numbers[:10]}...")  # 只显示前10个
        else:
            print("试验编号连续")
    
    # 9. 预测断点续做后的试验编号
    if len(remaining_order) > 0:
        print(f"\n=== 断点续做预测 ===")
        print(f"如果现在开始断点续做，试验编号将从 {trial_offset + 1} 开始")
        print(f"最后一个试验编号将是: {trial_offset + len(remaining_order)}")
        print(f"总共还需要完成 {len(remaining_order)} 个试验")
        
        # 检查是否会与现有编号冲突
        new_trial_numbers = list(range(trial_offset + 1, trial_offset + len(remaining_order) + 1))
        existing_trial_numbers = [t['trial_number'] for t in restored_trials]
        
        conflicts = set(new_trial_numbers) & set(existing_trial_numbers)
        if conflicts:
            print(f"❌ 警告: 新试验编号与现有编号冲突: {conflicts}")
        else:
            print("✅ 新试验编号与现有编号无冲突")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = "yanruofan"  # 默认测试用户
    
    test_resume_logic(username)


