#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析selected_120文件夹中的音频文件，找出每个音频的合成关系并重新命名
根据pairing_results_5s.csv文件中的信息，重建文件名映射关系
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import re

def parse_audio_filename(filename):
    """
    解析音频文件名，提取类别和音频ID
    例如: EEhnuLFYehU_2.wav -> (EEhnuLFYehU, 2)
    """
    name_without_ext = os.path.splitext(filename)[0]
    if '_' in name_without_ext:
        parts = name_without_ext.rsplit('_', 1)
        audio_id = parts[0]
        segment = parts[1] if parts[1] in ['1', '2'] else None
        return audio_id, segment
    return name_without_ext, None

def get_category_mapping():
    """获取类别映射关系"""
    return {
        '高生态效度': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
        '低生态效度': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
        'music': ['Bass drum', 'Funny music', 'Sad music'],
        '未知声源': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
    }

def build_audio_mapping():
    """
    根据pairing_results_5s.csv构建音频映射关系
    返回: dict {audio_filename: (left_category, left_audio_id, left_segment, right_category, right_audio_id, right_segment, type)}
    """
    csv_path = r"D:\D\research\audioset下载\音频配对合成_5s\pairing_results_5s.csv"
    
    if not os.path.exists(csv_path):
        print(f"错误: 找不到配对结果文件 {csv_path}")
        return {}
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"从CSV文件读取到 {len(df)} 条配对记录")
    
    # 构建映射关系
    audio_mapping = {}
    pair_index = 1  # 用于生成a0001, a0002等编号
    
    for _, row in df.iterrows():
        # 获取左右声道信息
        left_file = row['left_file']
        right_file = row['right_file']
        left_subcategory = row['left_subcategory']
        right_subcategory = row['right_subcategory']
        pair_type = row['type']
        
        # 解析文件名
        left_audio_id, left_segment = parse_audio_filename(left_file)
        right_audio_id, right_segment = parse_audio_filename(right_file)
        
        # 生成标准文件名
        standard_filename = f"a{pair_index:04d}.wav"
        
        # 存储映射关系
        audio_mapping[standard_filename] = {
            'left_category': left_subcategory,
            'left_audio_id': left_audio_id,
            'left_segment': left_segment,
            'right_category': right_subcategory,
            'right_audio_id': right_audio_id,
            'right_segment': right_segment,
            'type': pair_type,
            'original_left_file': left_file,
            'original_right_file': right_file
        }
        
        pair_index += 1
    
    print(f"构建了 {len(audio_mapping)} 个音频文件的映射关系")
    return audio_mapping

def generate_new_filename(mapping_info):
    """
    根据映射信息生成新的文件名
    格式: 左类别_左声道的原音频名字_a+右类别_右声道的原音频名字_b
    其中a,b表示前5s(1)还是后5s(2)
    """
    left_category = mapping_info['left_category']
    left_audio_id = mapping_info['left_audio_id']
    left_segment = mapping_info['left_segment']
    right_category = mapping_info['right_category']
    right_audio_id = mapping_info['right_audio_id']
    right_segment = mapping_info['right_segment']
    
    # 生成新文件名
    new_name = f"{left_category}_{left_audio_id}_{left_segment}+{right_category}_{right_audio_id}_{right_segment}.wav"
    
    # 替换文件名中的特殊字符，确保文件名合法
    new_name = new_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    
    return new_name

def analyze_and_rename():
    """主函数：分析音频文件并重新命名"""
    # 设置路径
    selected_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\selected_120")
    output_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120")
    
    # 检查源文件夹是否存在
    if not selected_dir.exists():
        print(f"错误: 源文件夹 {selected_dir} 不存在!")
        return
    
    # 创建输出文件夹
    output_dir.mkdir(exist_ok=True)
    
    # 构建音频映射关系
    audio_mapping = build_audio_mapping()
    if not audio_mapping:
        print("无法构建音频映射关系，程序退出")
        return
    
    # 获取所有音频文件
    audio_files = list(selected_dir.glob("*.wav"))
    print(f"在 {selected_dir} 中找到 {len(audio_files)} 个音频文件")
    
    # 处理每个文件
    renamed_files = []
    mapping_data = []
    
    for audio_file in audio_files:
        filename = audio_file.name
        
        # 查找对应的映射信息
        if filename in audio_mapping:
            mapping_info = audio_mapping[filename]
            
            # 生成新文件名
            new_filename = generate_new_filename(mapping_info)
            
            # 复制并重命名文件
            output_path = output_dir / new_filename
            shutil.copy2(audio_file, output_path)
            
            # 记录重命名信息
            renamed_files.append({
                'original_name': filename,
                'new_name': new_filename,
                'left_category': mapping_info['left_category'],
                'left_audio_id': mapping_info['left_audio_id'],
                'left_segment': mapping_info['left_segment'],
                'right_category': mapping_info['right_category'],
                'right_audio_id': mapping_info['right_audio_id'],
                'right_segment': mapping_info['right_segment'],
                'type': mapping_info['type'],
                'original_left_file': mapping_info['original_left_file'],
                'original_right_file': mapping_info['original_right_file']
            })
            
            print(f"✓ {filename} -> {new_filename}")
            print(f"  左声道: {mapping_info['left_category']} - {mapping_info['left_audio_id']} ({mapping_info['left_segment']})")
            print(f"  右声道: {mapping_info['right_category']} - {mapping_info['right_audio_id']} ({mapping_info['right_segment']})")
            print(f"  类型: {mapping_info['type']}")
            print()
        else:
            print(f"✗ 未找到 {filename} 的映射信息")
    
    # 保存重命名结果到CSV
    if renamed_files:
        df_renamed = pd.DataFrame(renamed_files)
        csv_path = output_dir.parent / "audio_rename_mapping.csv"
        df_renamed.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n重命名结果已保存到: {csv_path}")
        
        # 显示统计信息
        print(f"\n=== 重命名统计 ===")
        print(f"成功重命名: {len(renamed_files)} 个文件")
        print(f"输出目录: {output_dir}")
        
        # 按类型统计
        type_counts = df_renamed['type'].value_counts()
        print(f"\n按类型统计:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count} 个")
        
        # 按类别统计
        left_category_counts = df_renamed['left_category'].value_counts()
        print(f"\n左声道类别统计:")
        for category, count in left_category_counts.items():
            print(f"  {category}: {count} 个")
            
        right_category_counts = df_renamed['right_category'].value_counts()
        print(f"\n右声道类别统计:")
        for category, count in right_category_counts.items():
            print(f"  {category}: {count} 个")
    
    else:
        print("没有成功重命名任何文件")

def preview_rename():
    """预览重命名结果"""
    # 构建音频映射关系
    audio_mapping = build_audio_mapping()
    if not audio_mapping:
        return
    
    print("=== 重命名预览 ===")
    print("原始文件名 -> 新文件名")
    print("-" * 80)
    
    for i, (original_name, mapping_info) in enumerate(list(audio_mapping.items())[:10], 1):
        new_filename = generate_new_filename(mapping_info)
        print(f"{i:2d}. {original_name}")
        print(f"    -> {new_filename}")
        print(f"    左: {mapping_info['left_category']} - {mapping_info['left_audio_id']} ({mapping_info['left_segment']})")
        print(f"    右: {mapping_info['right_category']} - {mapping_info['right_audio_id']} ({mapping_info['right_segment']})")
        print(f"    类型: {mapping_info['type']}")
        print()
    
    if len(audio_mapping) > 10:
        print(f"... 还有 {len(audio_mapping) - 10} 个文件")

if __name__ == "__main__":
    print("=== 音频文件分析重命名工具 ===")
    print()
    
    # 先预览
    preview_rename()
    
    print("\n" + "="*60)
    response = input("是否要执行实际的文件重命名操作？(y/n): ").strip().lower()
    if response == 'y':
        analyze_and_rename()
    else:
        print("操作已取消。")
