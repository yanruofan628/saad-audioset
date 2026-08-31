#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终的重命名脚本：匹配音频内容并重新命名文件
"""

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil

def load_audio_mono(audio_path, sr=16000):
    """加载音频文件为单声道"""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"加载音频失败 {audio_path}: {e}")
        return None

def compare_audio_content(audio1, audio2, tolerance=1e-4):
    """比较两个音频内容是否相同"""
    if audio1 is None or audio2 is None:
        return False
    
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    mse = np.mean((audio1 - audio2) ** 2)
    
    # 调试信息
    if mse < 1e-2:  # 如果误差较小，打印调试信息
        print(f"    音频比较: MSE={mse:.8f}, 容差={tolerance:.0e}, 匹配={mse < tolerance}")
    
    return mse < tolerance

def separate_stereo_channels(stereo_audio_path):
    """分离立体声音频的左右声道"""
    try:
        stereo_audio, sr = sf.read(stereo_audio_path)
        
        print(f"    分离声道: 形状={stereo_audio.shape}, 采样率={sr}")
        
        if len(stereo_audio.shape) == 1:
            print(f"    音频是单声道，无法分离")
            return None, None
        
        # 确保是 (samples, channels) 格式
        if stereo_audio.shape[0] < stereo_audio.shape[1]:
            stereo_audio = stereo_audio.T
        
        left_channel = stereo_audio[:, 0]
        right_channel = stereo_audio[:, 1]
        
        print(f"    左声道长度: {len(left_channel)}, 右声道长度: {len(right_channel)}")
        
        return left_channel, right_channel
    except Exception as e:
        print(f"分离声道失败 {stereo_audio_path}: {e}")
        return None, None

def build_expected_pairs():
    """从pairing_results_5s.csv构建所有可能的配对"""
    csv_path = r"D:\D\research\audioset下载\音频配对合成_5s\pairing_results_5s.csv"
    
    if not os.path.exists(csv_path):
        print(f"错误: 找不到配对结果文件 {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    expected_pairs = []
    
    for _, row in df.iterrows():
        left_file = row['left_file']
        right_file = row['right_file']
        left_subcategory = row['left_subcategory']
        right_subcategory = row['right_subcategory']
        pair_type = row['type']
        
        if pair_type == 'nearest_neighbor':
            prefix = 'nn_'
        else:  # random
            prefix = 'rn_'
        
        expected_filename = f"{prefix}{left_subcategory}_{left_file}+{right_subcategory}_{right_file}.wav"
        
        expected_pairs.append({
            'expected_filename': expected_filename,
            'left_subcategory': left_subcategory,
            'left_file': left_file,
            'right_subcategory': right_subcategory,
            'right_file': right_file,
            'type': pair_type
        })
    
    print(f"构建了 {len(expected_pairs)} 个期望的配对")
    return expected_pairs

def generate_new_filename(match_info):
    """根据匹配信息生成新的文件名"""
    left_category = match_info['left_subcategory']
    left_audio_id = match_info['left_file'].replace('_1.wav', '').replace('_2.wav', '')
    left_segment = '1' if '_1.wav' in match_info['left_file'] else '2'
    
    right_category = match_info['right_subcategory']
    right_audio_id = match_info['right_file'].replace('_1.wav', '').replace('_2.wav', '')
    right_segment = '1' if '_1.wav' in match_info['right_file'] else '2'
    
    # 生成新文件名
    new_name = f"{left_category}_{left_audio_id}_{left_segment}+{right_category}_{right_audio_id}_{right_segment}.wav"
    
    # 替换文件名中的特殊字符，确保文件名合法
    new_name = new_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    
    return new_name

def match_and_rename():
    """匹配音频文件并重新命名"""
    # 设置路径
    selected_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\selected_120")
    clap_base_path = r"D:\D\research\audioset下载\clap"
    output_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120")
    
    # 检查文件夹是否存在
    if not selected_dir.exists():
        print(f"错误: 源文件夹 {selected_dir} 不存在!")
        return
    
    if not os.path.exists(clap_base_path):
        print(f"错误: clap文件夹 {clap_base_path} 不存在!")
        return
    
    # 创建输出文件夹
    output_dir.mkdir(exist_ok=True)
    
    # 构建期望的配对
    expected_pairs = build_expected_pairs()
    if not expected_pairs:
        print("无法构建期望配对，程序退出")
        return
    
    # 获取所有音频文件
    audio_files = list(selected_dir.glob("*.wav"))
    print(f"找到 {len(audio_files)} 个音频文件需要匹配")
    
    # 匹配和重命名结果
    rename_results = []
    unmatched_files = []
    
    # 处理每个文件
    for audio_file in audio_files:
        print(f"\n正在处理: {audio_file.name}")
        
        # 分离左右声道
        left_channel, right_channel = separate_stereo_channels(audio_file)
        
        if left_channel is None or right_channel is None:
            print(f"  无法分离声道或不是立体声")
            unmatched_files.append(audio_file.name)
            continue
        
        # 查找匹配
        matched_pair = None
        for expected_pair in expected_pairs:
            # 构建左右声道文件的路径
            left_file_path = os.path.join(clap_base_path, expected_pair['left_subcategory'], expected_pair['left_file'])
            right_file_path = os.path.join(clap_base_path, expected_pair['right_subcategory'], expected_pair['right_file'])
            
            # 检查文件是否存在
            if not os.path.exists(left_file_path) or not os.path.exists(right_file_path):
                continue
            
            # 加载期望的左右声道音频
            expected_left = load_audio_mono(left_file_path)
            expected_right = load_audio_mono(right_file_path)
            
            if expected_left is None or expected_right is None:
                continue
            
            # 比较左右声道
            print(f"    比较左声道: {expected_pair['left_subcategory']}/{expected_pair['left_file']}")
            left_match = compare_audio_content(left_channel, expected_left)
            print(f"    比较右声道: {expected_pair['right_subcategory']}/{expected_pair['right_file']}")
            right_match = compare_audio_content(right_channel, expected_right)
            
            if left_match and right_match:
                matched_pair = expected_pair
                print(f"    *** 找到完全匹配! ***")
                break
        
        if matched_pair:
            # 生成新文件名
            new_filename = generate_new_filename(matched_pair)
            
            # 复制并重命名文件
            output_path = output_dir / new_filename
            shutil.copy2(audio_file, output_path)
            
            # 记录结果
            rename_results.append({
                'original_name': audio_file.name,
                'new_name': new_filename,
                'left_category': matched_pair['left_subcategory'],
                'left_file': matched_pair['left_file'],
                'right_category': matched_pair['right_subcategory'],
                'right_file': matched_pair['right_file'],
                'type': matched_pair['type'],
                'expected_filename': matched_pair['expected_filename']
            })
            
            print(f"  ✓ 匹配成功并重命名!")
            print(f"  新文件名: {new_filename}")
            print(f"  左声道: {matched_pair['left_subcategory']} - {matched_pair['left_file']}")
            print(f"  右声道: {matched_pair['right_subcategory']} - {matched_pair['right_file']}")
            print(f"  类型: {matched_pair['type']}")
        else:
            print(f"  ✗ 未找到匹配")
            unmatched_files.append(audio_file.name)
    
    # 保存结果
    if rename_results:
        df_results = pd.DataFrame(rename_results)
        output_csv = output_dir.parent / "final_rename_results.csv"
        df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n重命名结果已保存到: {output_csv}")
        
        # 显示统计信息
        print(f"\n=== 重命名统计 ===")
        print(f"成功重命名: {len(rename_results)} 个文件")
        print(f"未匹配: {len(unmatched_files)} 个文件")
        print(f"输出目录: {output_dir}")
        
        if unmatched_files:
            print(f"\n未匹配的文件:")
            for filename in unmatched_files:
                print(f"  {filename}")
        
        # 按类型统计
        type_counts = df_results['type'].value_counts()
        print(f"\n按类型统计:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count} 个")
        
        # 按类别统计
        left_category_counts = df_results['left_category'].value_counts()
        print(f"\n左声道类别统计:")
        for category, count in left_category_counts.items():
            print(f"  {category}: {count} 个")
            
        right_category_counts = df_results['right_category'].value_counts()
        print(f"\n右声道类别统计:")
        for category, count in right_category_counts.items():
            print(f"  {category}: {count} 个")
    
    else:
        print("没有成功重命名任何文件")

def preview_rename():
    """预览重命名过程"""
    print("=== 重命名预览 ===")
    
    expected_pairs = build_expected_pairs()
    if not expected_pairs:
        return
    
    print(f"期望的配对数量: {len(expected_pairs)}")
    print(f"\n前5个期望配对示例:")
    for i, pair in enumerate(expected_pairs[:5], 1):
        new_name = generate_new_filename(pair)
        print(f"{i}. 原始期望: {pair['expected_filename']}")
        print(f"   重命名后: {new_name}")
        print(f"   左: {pair['left_subcategory']} - {pair['left_file']}")
        print(f"   右: {pair['right_subcategory']} - {pair['right_file']}")
        print(f"   类型: {pair['type']}")
        print()

if __name__ == "__main__":
    print("=== 最终音频重命名工具 ===")
    print()
    
    # 先预览
    preview_rename()
    
    print("\n" + "="*60)
    response = input("是否要执行实际的音频匹配和重命名操作？(y/n): ").strip().lower()
    if response == 'y':
        match_and_rename()
    else:
        print("操作已取消。")
