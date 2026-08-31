#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理selected_120文件夹中的音频文件
1. 将文件重命名为a0001.wav到a0120.wav
2. 确保所有文件都是WAV格式
3. 生成CSV文件记录原文件名和新文件名的对应关系
"""

import os
import random
import shutil
import pandas as pd
from pathlib import Path
import subprocess
import sys

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_to_wav(input_path, output_path):
    """使用ffmpeg将音频文件转换为WAV格式"""
    try:
        cmd = [
            'ffmpeg', '-i', str(input_path), 
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '44100',  # 44.1kHz采样率
            '-ac', '1',  # 单声道
            '-y',  # 覆盖输出文件
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败 {input_path}: {e}")
        return False

def process_selected_files():
    # selected_120文件夹路径
    selected_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\selected_120")
    
    # 检查文件夹是否存在
    if not selected_dir.exists():
        print(f"错误: selected_120文件夹 {selected_dir} 不存在!")
        print("请先运行分类脚本创建selected_120文件夹")
        return
    
    # 获取所有音频文件
    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac']
    all_files = []
    for ext in audio_extensions:
        all_files.extend(selected_dir.glob(ext))
    
    print(f"在 {selected_dir} 中找到 {len(all_files)} 个音频文件")
    
    if len(all_files) == 0:
        print("没有找到任何音频文件!")
        return
    
    # 检查ffmpeg是否可用
    if not check_ffmpeg():
        print("警告: 未找到ffmpeg，将尝试直接重命名文件而不进行格式转换")
        use_ffmpeg = False
    else:
        print("找到ffmpeg，将进行音频格式转换")
        use_ffmpeg = True
    
    # 准备CSV数据
    csv_data = []
    
    # 创建临时文件夹用于处理
    temp_dir = selected_dir.parent / "temp_processing"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\n开始处理文件...")
    
    # 处理每个文件
    for i, file_path in enumerate(all_files, 1):
        # 生成新的文件名 a001.wav, a002.wav, ..., a120.wav
        new_name = f"a{i:03d}.wav"
        
        # 记录原文件名和新文件名的对应关系
        csv_data.append({
            'original_name': file_path.name,
            'new_name': new_name,
            'original_path': str(file_path),
            'file_index': i
        })
        
        # 目标文件路径
        dest_path = temp_dir / new_name
        
        # 复制或转换文件
        if use_ffmpeg:
            success = convert_to_wav(file_path, dest_path)
            if not success:
                # 如果转换失败，直接复制并重命名
                shutil.copy2(file_path, dest_path)
                print(f"  转换失败，直接复制: {file_path.name} -> {new_name}")
        else:
            # 直接复制文件
            shutil.copy2(file_path, dest_path)
        
        if i % 20 == 0:
            print(f"  已处理 {i}/{len(all_files)} 个文件")
    
    print(f"已处理所有 {len(all_files)} 个文件")
    
    # 将处理好的文件移回原文件夹
    print(f"\n将重命名后的文件移回原文件夹...")
    for temp_file in temp_dir.glob("*.wav"):
        dest_path = selected_dir / temp_file.name
        shutil.move(str(temp_file), str(dest_path))
    
    # 删除临时文件夹
    temp_dir.rmdir()
    
    # 删除原始文件（现在已经被重命名的文件替代）
    print(f"清理原始文件...")
    for file_path in all_files:
        if file_path.exists():
            file_path.unlink()
    
    # 生成CSV文件
    csv_path = selected_dir.parent / "file_mapping.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n已生成CSV文件: {csv_path}")
    
    # 验证结果
    final_files = list(selected_dir.glob("*.wav"))
    print(f"\n验证结果:")
    print(f"selected_120文件夹中的文件数: {len(final_files)}")
    
    # 显示CSV文件内容预览
    print(f"\nCSV文件内容预览（前10行）:")
    print(df.head(10).to_string(index=False))
    
    print(f"\n处理完成!")
    print(f"所有文件已重命名为 a001.wav 到 a{len(all_files):03d}.wav")
    print(f"CSV文件记录了所有文件的对应关系")

def preview_processing():
    """预览处理结果而不执行实际重命名"""
    # selected_120文件夹路径
    selected_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\selected_120")
    
    # 检查文件夹是否存在
    if not selected_dir.exists():
        print(f"错误: selected_120文件夹 {selected_dir} 不存在!")
        print("请先运行分类脚本创建selected_120文件夹")
        return
    
    # 获取所有音频文件
    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac']
    all_files = []
    for ext in audio_extensions:
        all_files.extend(selected_dir.glob(ext))
    
    print(f"在 {selected_dir} 中找到 {len(all_files)} 个音频文件")
    
    if len(all_files) == 0:
        print("没有找到任何音频文件!")
        return
    
    # 显示重命名预览
    print(f"\n重命名预览（前10个文件）:")
    for i, file_path in enumerate(all_files[:10], 1):
        new_name = f"a{i:03d}.wav"
        print(f"  {i:2d}. {file_path.name} -> {new_name}")
    
    if len(all_files) > 10:
        print(f"  ... 还有 {len(all_files) - 10} 个文件")
    
    print(f"\n预览完成!")
    print(f"总共将重命名 {len(all_files)} 个文件为 a001.wav 到 a{len(all_files):03d}.wav")

if __name__ == "__main__":
    try:
        # 先预览
        preview_processing()
        
        # 询问用户是否要执行实际重命名
        print("\n" + "="*60)
        response = input("是否要执行实际的文件重命名和转换操作？(y/n): ").strip().lower()
        if response == 'y':
            process_selected_files()
        else:
            print("操作已取消。")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
