#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件分类、重命名和转换脚本
从remaining_all_mixed文件夹中：
1. 随机选择60个nearest_nn开头的文件
2. 随机选择60个random_rn开头的文件
3. 将这120个文件重命名为a0001.wav到a0120.wav
4. 确保所有文件都是WAV格式
5. 生成CSV文件记录原文件名和新文件名的对应关系
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

def classify_and_rename_audio_files():
    # 源文件夹路径
    source_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\remaining_all_mixed")
    
    # 目标文件夹路径
    selected_dir = source_dir.parent / "selected_120"
    remaining_dir = source_dir.parent / "remaining_210"
    
    # 创建目标文件夹
    selected_dir.mkdir(exist_ok=True)
    remaining_dir.mkdir(exist_ok=True)
    
    print(f"源文件夹: {source_dir}")
    print(f"选中文件将移动到: {selected_dir}")
    print(f"剩余文件将移动到: {remaining_dir}")
    
    # 检查源文件夹是否存在
    if not source_dir.exists():
        print(f"错误: 源文件夹 {source_dir} 不存在!")
        return
    
    # 检查ffmpeg是否可用
    if not check_ffmpeg():
        print("警告: 未找到ffmpeg，将尝试直接复制文件而不进行格式转换")
        use_ffmpeg = False
    else:
        print("找到ffmpeg，将进行音频格式转换")
        use_ffmpeg = True
    
    # 获取所有文件
    all_files = list(source_dir.glob("*.wav"))
    print(f"总共找到 {len(all_files)} 个音频文件")
    
    if len(all_files) == 0:
        print("没有找到任何音频文件!")
        return
    
    # 分类文件
    nearest_nn_files = [f for f in all_files if f.name.startswith("nearest_nn")]
    random_rn_files = [f for f in all_files if f.name.startswith("random_rn")]
    
    print(f"nearest_nn开头的文件: {len(nearest_nn_files)} 个")
    print(f"random_rn开头的文件: {len(random_rn_files)} 个")
    
    # 检查是否有足够的文件
    if len(nearest_nn_files) < 60:
        print(f"警告: nearest_nn文件只有 {len(nearest_nn_files)} 个，少于60个!")
        return
    
    if len(random_rn_files) < 60:
        print(f"警告: random_rn文件只有 {len(random_rn_files)} 个，少于60个!")
        return
    
    # 随机选择60个nearest_nn文件
    selected_nearest_nn = random.sample(nearest_nn_files, 60)
    print(f"随机选择了 {len(selected_nearest_nn)} 个nearest_nn文件")
    
    # 随机选择60个random_rn文件
    selected_random_rn = random.sample(random_rn_files, 60)
    print(f"随机选择了 {len(selected_random_rn)} 个random_rn文件")
    
    # 合并选中的文件
    selected_files = selected_nearest_nn + selected_random_rn
    print(f"总共选中 {len(selected_files)} 个文件")
    
    # 计算剩余文件
    remaining_files = [f for f in all_files if f not in selected_files]
    print(f"剩余 {len(remaining_files)} 个文件")
    
    # 准备CSV数据
    csv_data = []
    
    # 处理选中的文件 - 重命名并转换格式
    print(f"\n开始处理选中的文件...")
    for i, file_path in enumerate(selected_files, 1):
        # 生成新的文件名 a0001.wav, a0002.wav, ..., a0120.wav
        new_name = f"a{i:04d}.wav"
        dest_path = selected_dir / new_name
        
        # 记录原文件名和新文件名的对应关系
        csv_data.append({
            'original_name': file_path.name,
            'new_name': new_name,
            'file_type': 'nearest_nn' if file_path in selected_nearest_nn else 'random_rn',
            'original_path': str(file_path),
            'new_path': str(dest_path)
        })
        
        # 复制或转换文件
        if use_ffmpeg:
            success = convert_to_wav(file_path, dest_path)
            if not success:
                # 如果转换失败，直接复制
                shutil.copy2(file_path, dest_path)
                print(f"  转换失败，直接复制: {file_path.name} -> {new_name}")
        else:
            # 直接复制文件
            shutil.copy2(file_path, dest_path)
        
        if i % 20 == 0:
            print(f"  已处理 {i}/{len(selected_files)} 个选中文件")
    
    print(f"已处理所有 {len(selected_files)} 个选中文件")
    
    # 移动剩余文件（不重命名）
    print(f"\n开始移动剩余文件...")
    for i, file_path in enumerate(remaining_files, 1):
        dest_path = remaining_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        if i % 20 == 0:
            print(f"  已移动 {i}/{len(remaining_files)} 个剩余文件")
    
    print(f"已移动所有 {len(remaining_files)} 个剩余文件")
    
    # 生成CSV文件
    csv_path = source_dir.parent / "file_mapping.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n已生成CSV文件: {csv_path}")
    
    # 验证结果
    print(f"\n验证结果:")
    print(f"selected_120文件夹中的文件数: {len(list(selected_dir.glob('*.wav')))}")
    print(f"remaining_210文件夹中的文件数: {len(list(remaining_dir.glob('*.wav')))}")
    print(f"源文件夹中剩余文件数: {len(list(source_dir.glob('*.wav')))}")
    
    # 显示CSV文件内容预览
    print(f"\nCSV文件内容预览（前10行）:")
    print(df.head(10).to_string(index=False))
    
    print(f"\n分类和重命名完成!")
    print(f"选中文件已重命名为 a0001.wav 到 a{len(selected_files):04d}.wav")
    print(f"CSV文件记录了所有文件的对应关系")

def preview_classification():
    """预览分类结果而不执行实际移动"""
    # 源文件夹路径
    source_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\remaining_all_mixed")
    
    print(f"源文件夹: {source_dir}")
    
    # 检查源文件夹是否存在
    if not source_dir.exists():
        print(f"错误: 源文件夹 {source_dir} 不存在!")
        return
    
    # 获取所有文件
    all_files = list(source_dir.glob("*.wav"))
    print(f"总共找到 {len(all_files)} 个音频文件")
    
    if len(all_files) == 0:
        print("没有找到任何音频文件!")
        return
    
    # 分类文件
    nearest_nn_files = [f for f in all_files if f.name.startswith("nearest_nn")]
    random_rn_files = [f for f in all_files if f.name.startswith("random_rn")]
    
    print(f"nearest_nn开头的文件: {len(nearest_nn_files)} 个")
    print(f"random_rn开头的文件: {len(random_rn_files)} 个")
    
    # 检查是否有足够的文件
    if len(nearest_nn_files) < 60:
        print(f"警告: nearest_nn文件只有 {len(nearest_nn_files)} 个，少于60个!")
        return
    
    if len(random_rn_files) < 60:
        print(f"警告: random_rn文件只有 {len(random_rn_files)} 个，少于60个!")
        return
    
    # 随机选择文件
    selected_nearest_nn = random.sample(nearest_nn_files, 60)
    selected_random_rn = random.sample(random_rn_files, 60)
    selected_files = selected_nearest_nn + selected_random_rn
    
    # 显示重命名预览
    print(f"\n重命名预览（前10个文件）:")
    for i, file_path in enumerate(selected_files[:10], 1):
        new_name = f"a{i:04d}.wav"
        file_type = "nearest_nn" if file_path in selected_nearest_nn else "random_rn"
        print(f"  {i:2d}. {file_path.name} -> {new_name} ({file_type})")
    
    if len(selected_files) > 10:
        print(f"  ... 还有 {len(selected_files) - 10} 个文件")
    
    print(f"\n预览完成!")
    print(f"总共将重命名 {len(selected_files)} 个文件为 a0001.wav 到 a{len(selected_files):04d}.wav")

if __name__ == "__main__":
    # 设置随机种子以便结果可重现
    random.seed(42)
    
    try:
        # 先预览
        preview_classification()
        
        # 询问用户是否要执行实际移动
        print("\n" + "="*60)
        response = input("是否要执行实际的文件分类、重命名和转换操作？(y/n): ").strip().lower()
        if response == 'y':
            classify_and_rename_audio_files()
        else:
            print("操作已取消。")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
