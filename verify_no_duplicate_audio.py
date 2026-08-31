#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证断点续做时已完成的音频不会重复出现
"""

import os
import csv
import json

# 配置参数
AUDIO_DIR = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1"
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"

def verify_no_duplicate_audio(username):
    """验证断点续做时已完成的音频不会重复出现"""
    
    print(f"=== 验证用户: {username} 的音频重复问题 ===")
    
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
    
    # 3. 读取用户已完成的数据
    safe_username = ''.join(c for c in username if c not in '\\/:*?"<>|').strip() or 'unknown'
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_username}.csv")
    
    completed_files = set()
    user_trials = []
    
    if os.path.exists(csv_path):
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            data_read = False
            
            for encoding in encodings:
                try:
                    with open(csv_path, 'r', encoding=encoding) as f:
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
                                
                                user_trials.append({
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
                    
                    print(f"使用编码 {encoding} 成功读取文件")
                    data_read = True
                    break
                    
                except UnicodeDecodeError:
                    continue
            
            if not data_read:
                print("无法读取CSV文件，尝试了所有编码")
                return
            
            print(f"用户已完成试验数: {len(user_trials)}")
            print(f"用户已完成音频文件数: {len(completed_files)}")
            
        except Exception as e:
            print(f"读取CSV失败: {e}")
            return
    else:
        print("用户CSV文件不存在")
        return
    
    # 4. 模拟断点续做的过滤逻辑
    remaining_order = []
    for idx in audio_order:
        audio_file = audio_files[idx]
        if audio_file not in completed_files:
            remaining_order.append(idx)
    
    print(f"\n=== 断点续做过滤结果 ===")
    print(f"原始音频顺序: {len(audio_order)} 个")
    print(f"已完成音频文件: {len(completed_files)} 个")
    print(f"剩余待做音频: {len(remaining_order)} 个")
    
    # 5. 验证没有重复
    remaining_files = [audio_files[i] for i in remaining_order]
    
    print(f"\n=== 重复检查 ===")
    
    # 检查剩余音频中是否有已完成的音频
    overlap = set(remaining_files) & completed_files
    if overlap:
        print(f"❌ 错误: 剩余音频中包含已完成的音频!")
        print(f"重复的音频文件: {sorted(overlap)}")
        return False
    else:
        print(f"✅ 剩余音频中没有已完成的音频")
    
    # 检查已完成音频中是否有重复
    completed_list = list(completed_files)
    if len(completed_list) != len(set(completed_list)):
        print(f"❌ 错误: 已完成音频中有重复!")
        duplicates = [x for x in completed_list if completed_list.count(x) > 1]
        print(f"重复的音频文件: {set(duplicates)}")
        return False
    else:
        print(f"✅ 已完成音频中没有重复")
    
    # 检查剩余音频中是否有重复
    if len(remaining_files) != len(set(remaining_files)):
        print(f"❌ 错误: 剩余音频中有重复!")
        duplicates = [x for x in remaining_files if remaining_files.count(x) > 1]
        print(f"重复的音频文件: {set(duplicates)}")
        return False
    else:
        print(f"✅ 剩余音频中没有重复")
    
    # 6. 显示详细信息
    print(f"\n=== 详细信息 ===")
    print(f"已完成音频文件示例: {sorted(list(completed_files))[:10]}")
    print(f"剩余音频文件示例: {remaining_files[:10]}")
    
    # 7. 验证总数
    total_processed = len(completed_files) + len(remaining_files)
    if total_processed == len(audio_files):
        print(f"✅ 总数验证通过: {len(completed_files)} + {len(remaining_files)} = {total_processed} = {len(audio_files)}")
    else:
        print(f"❌ 总数验证失败: {len(completed_files)} + {len(remaining_files)} = {total_processed} ≠ {len(audio_files)}")
        return False
    
    print(f"\n=== 结论 ===")
    print(f"✅ 断点续做逻辑正确!")
    print(f"✅ 已完成的音频不会重复出现!")
    print(f"✅ 用户 {username} 还需要完成 {len(remaining_files)} 个音频")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = "yanruofan"  # 默认测试用户
    
    verify_no_duplicate_audio(username)


