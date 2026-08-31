#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理音频配对合成文件夹中的音频文件
1. 获取所有音频文件并打乱顺序
2. 将文件一分为二，分到两个文件夹（folder1和folder2）
3. 每个文件夹中的文件重命名为a001.wav到a0xx.wav
4. 为每个文件夹生成CSV文件记录原文件名和新文件名的对应关系
"""

import os
import random
import shutil
import pandas as pd
from pathlib import Path

def split_and_rename_audio():
    # 源文件夹路径 - 使用168pairs文件夹，包含所有音频
    source_dir = Path(r"D:\D\research\audioset下载\音频配对合成_新版\all_288")
    
    # 检查文件夹是否存在
    if not source_dir.exists():
        print(f"错误: 源文件夹 {source_dir} 不存在!")
        return
    
    # 获取所有音频文件
    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac']
    all_files = []
    for ext in audio_extensions:
        all_files.extend(source_dir.glob(ext))
    
    print(f"在 {source_dir} 中找到 {len(all_files)} 个音频文件")
    
    if len(all_files) == 0:
        print("没有找到任何音频文件!")
        return
    
    # 打乱文件顺序
    random.shuffle(all_files)
    print(f"已打乱文件顺序")
    
    # 计算分割点
    mid_point = len(all_files) // 2
    folder1_files = all_files[:mid_point]
    folder2_files = all_files[mid_point:]
    
    print(f"将分配 {len(folder1_files)} 个文件到 folder1")
    print(f"将分配 {len(folder2_files)} 个文件到 folder2")
    
    # 创建目标文件夹
    output_base_dir = source_dir.parent / "分组音频"
    folder1_dir = output_base_dir / "folder1"
    folder2_dir = output_base_dir / "folder2"
    
    folder1_dir.mkdir(parents=True, exist_ok=True)
    folder2_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n创建输出文件夹:")
    print(f"  {folder1_dir}")
    print(f"  {folder2_dir}")
    
    # 处理 folder1
    print(f"\n开始处理 folder1...")
    csv_data_folder1 = process_folder(folder1_files, folder1_dir)
    
    # 处理 folder2
    print(f"\n开始处理 folder2...")
    csv_data_folder2 = process_folder(folder2_files, folder2_dir)
    
    # 生成CSV文件
    csv_path1 = folder1_dir / "file_mapping_folder1.csv"
    df1 = pd.DataFrame(csv_data_folder1)
    df1.to_csv(csv_path1, index=False, encoding='utf-8-sig')
    print(f"\n已生成CSV文件: {csv_path1}")
    
    csv_path2 = folder2_dir / "file_mapping_folder2.csv"
    df2 = pd.DataFrame(csv_data_folder2)
    df2.to_csv(csv_path2, index=False, encoding='utf-8-sig')
    print(f"已生成CSV文件: {csv_path2}")
    
    # 验证结果
    print(f"\n验证结果:")
    print(f"folder1 中的文件数: {len(list(folder1_dir.glob('*.wav')))}")
    print(f"folder2 中的文件数: {len(list(folder2_dir.glob('*.wav')))}")
    
    # 显示CSV文件内容预览
    print(f"\nfolder1 CSV文件内容预览（前5行）:")
    print(df1.head(5).to_string(index=False))
    
    print(f"\nfolder2 CSV文件内容预览（前5行）:")
    print(df2.head(5).to_string(index=False))
    
    print(f"\n处理完成!")
    print(f"folder1: {len(folder1_files)} 个文件已重命名为 a001.wav 到 a{len(folder1_files):03d}.wav")
    print(f"folder2: {len(folder2_files)} 个文件已重命名为 a001.wav 到 a{len(folder2_files):03d}.wav")

def process_folder(file_list, output_dir):
    """处理一个文件夹的文件"""
    csv_data = []
    
    for i, file_path in enumerate(file_list, 1):
        # 生成新的文件名 a001.wav, a002.wav, ...
        new_name = f"a{i:03d}.wav"
        
        # 记录原文件名和新文件名的对应关系
        csv_data.append({
            'original_name': file_path.name,
            'new_name': new_name,
            'original_path': str(file_path),
            'file_index': i
        })
        
        # 目标文件路径
        dest_path = output_dir / new_name
        
        # 直接复制文件
        shutil.copy2(file_path, dest_path)
        
        if i % 20 == 0:
            print(f"  已处理 {i}/{len(file_list)} 个文件")
    
    print(f"已处理所有 {len(file_list)} 个文件")
    
    return csv_data

def preview_split():
    """预览分割结果而不执行实际操作"""
    # 源文件夹路径
    source_dir = Path(r"D:\D\research\audioset下载\音频配对合成_新版\168pairs")
    
    # 检查文件夹是否存在
    if not source_dir.exists():
        print(f"错误: 源文件夹 {source_dir} 不存在!")
        return
    
    # 获取所有音频文件
    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac']
    all_files = []
    for ext in audio_extensions:
        all_files.extend(source_dir.glob(ext))
    
    print(f"在 {source_dir} 中找到 {len(all_files)} 个音频文件")
    
    if len(all_files) == 0:
        print("没有找到任何音频文件!")
        return
    
    # 打乱文件顺序（用于预览）
    random.shuffle(all_files)
    
    # 计算分割点
    mid_point = len(all_files) // 2
    folder1_files = all_files[:mid_point]
    folder2_files = all_files[mid_point:]
    
    print(f"\n分割预览:")
    print(f"总文件数: {len(all_files)}")
    print(f"folder1 将包含: {len(folder1_files)} 个文件 (a001.wav 到 a{len(folder1_files):03d}.wav)")
    print(f"folder2 将包含: {len(folder2_files)} 个文件 (a001.wav 到 a{len(folder2_files):03d}.wav)")
    
    print(f"\nfolder1 前5个文件示例:")
    for i, file_path in enumerate(folder1_files[:5], 1):
        new_name = f"a{i:03d}.wav"
        print(f"  {new_name} <- {file_path.name}")
    
    print(f"\nfolder2 前5个文件示例:")
    for i, file_path in enumerate(folder2_files[:5], 1):
        new_name = f"a{i:03d}.wav"
        print(f"  {new_name} <- {file_path.name}")
    
    print(f"\n预览完成!")
    print(f"输出文件夹将创建在: {source_dir.parent / '分组音频'}")

if __name__ == "__main__":
    try:
        # 先预览
        preview_split()
        
        # 询问用户是否要执行实际操作
        print("\n" + "="*60)
        response = input("是否要执行实际的文件分割和重命名操作？(y/n): ").strip().lower()
        if response == 'y':
            split_and_rename_audio()
        else:
            print("操作已取消。")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

