#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为renamed_120文件夹中的音频文件添加正确的前缀
根据final_rename_results.csv中的expected_filename列来重命名文件
"""

import os
import shutil
import pandas as pd
from pathlib import Path

def add_prefix_to_renamed_files():
    """根据final_rename_results.csv为renamed_120中的文件添加正确的前缀，并删除没有前缀的文件"""
    # renamed_120文件夹路径
    renamed_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120")
    
    # final_rename_results.csv文件路径
    csv_path = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\final_rename_results.csv")
    
    # 检查文件夹和CSV文件是否存在
    if not renamed_dir.exists():
        print(f"错误: renamed_120文件夹 {renamed_dir} 不存在!")
        return
    
    if not csv_path.exists():
        print(f"错误: final_rename_results.csv文件 {csv_path} 不存在!")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"成功读取CSV文件，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查必要的列是否存在
    required_columns = ['new_name', 'expected_filename']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV文件缺少必要的列: {missing_columns}")
        print(f"可用的列: {list(df.columns)}")
        return
    
    # 获取当前文件夹中的所有wav文件
    current_files = list(renamed_dir.glob("*.wav"))
    print(f"在 {renamed_dir} 中找到 {len(current_files)} 个WAV文件")
    
    if len(current_files) == 0:
        print("没有找到任何WAV文件!")
        return
    
    # 创建映射字典：当前文件名 -> 期望的文件名
    rename_mapping = {}
    for _, row in df.iterrows():
        current_name = row['new_name']  # 当前文件名 (如 Ambulance_siren_1LA8BxMlXSY_2+Helicopter_DSLO9BKHSEk_1.wav)
        expected_name = row['expected_filename']  # 期望的文件名 (如 nn_Ambulance (siren)_1LA8BxMlXSY_2.wav+Helicopter_DSLO9BKHSEk_1.wav.wav)
        rename_mapping[current_name] = expected_name
    
    print(f"创建了 {len(rename_mapping)} 个重命名映射")
    
    # 创建临时文件夹用于处理
    temp_dir = renamed_dir.parent / "temp_prefix_processing"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\n开始处理文件...")
    
    # 处理每个文件
    processed_count = 0
    deleted_count = 0
    
    for current_file in current_files:
        current_name = current_file.name
        
        # 检查文件是否已经有正确的前缀
        if current_name.startswith(('nn_', 'rn_')):
            print(f"  跳过已有前缀的文件: {current_name}")
            continue
        
        if current_name in rename_mapping:
            expected_name = rename_mapping[current_name]
            
            # 复制文件到临时文件夹并重命名
            temp_path = temp_dir / expected_name
            shutil.copy2(current_file, temp_path)
            
            processed_count += 1
            if processed_count % 20 == 0:
                print(f"  已处理 {processed_count} 个文件")
        else:
            # 删除没有映射关系的文件
            print(f"  删除无映射关系的文件: {current_name}")
            current_file.unlink()
            deleted_count += 1
    
    print(f"已处理 {processed_count} 个文件，删除了 {deleted_count} 个文件")
    
    # 将重命名后的文件移回原文件夹
    print(f"\n将重命名后的文件移回原文件夹...")
    for temp_file in temp_dir.glob("*"):
        dest_path = renamed_dir / temp_file.name
        shutil.move(str(temp_file), str(dest_path))
    
    # 删除临时文件夹
    temp_dir.rmdir()
    
    # 验证结果
    final_files = list(renamed_dir.glob("*.wav"))
    print(f"\n验证结果:")
    print(f"renamed_120文件夹中的文件数: {len(final_files)}")
    
    # 统计有前缀和无前缀的文件
    prefixed_files = [f for f in final_files if f.name.startswith(('nn_', 'rn_'))]
    non_prefixed_files = [f for f in final_files if not f.name.startswith(('nn_', 'rn_'))]
    
    print(f"有前缀的文件: {len(prefixed_files)} 个")
    print(f"无前缀的文件: {len(non_prefixed_files)} 个")
    
    # 显示重命名结果预览
    print(f"\n重命名结果预览（前10个文件）:")
    for i, file_path in enumerate(final_files[:10], 1):
        print(f"  {i:2d}. {file_path.name}")
    
    if len(final_files) > 10:
        print(f"  ... 还有 {len(final_files) - 10} 个文件")
    
    print(f"\n处理完成!")
    print(f"已为 {processed_count} 个文件添加正确的前缀")
    print(f"已删除 {deleted_count} 个无映射关系的文件")

def preview_prefix_addition():
    """预览处理结果而不执行实际重命名"""
    # renamed_120文件夹路径
    renamed_dir = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\renamed_120")
    
    # final_rename_results.csv文件路径
    csv_path = Path(r"D:\D\research\audioset下载\随机选择配对音频_5s\final_rename_results.csv")
    
    # 检查文件夹和CSV文件是否存在
    if not renamed_dir.exists():
        print(f"错误: renamed_120文件夹 {renamed_dir} 不存在!")
        return
    
    if not csv_path.exists():
        print(f"错误: final_rename_results.csv文件 {csv_path} 不存在!")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"成功读取CSV文件，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查必要的列是否存在
    required_columns = ['new_name', 'expected_filename']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV文件缺少必要的列: {missing_columns}")
        print(f"可用的列: {list(df.columns)}")
        return
    
    # 获取当前文件夹中的所有wav文件
    current_files = list(renamed_dir.glob("*.wav"))
    print(f"在 {renamed_dir} 中找到 {len(current_files)} 个WAV文件")
    
    if len(current_files) == 0:
        print("没有找到任何WAV文件!")
        return
    
    # 创建映射字典：当前文件名 -> 期望的文件名
    rename_mapping = {}
    for _, row in df.iterrows():
        current_name = row['new_name']  # 当前文件名 (如 Ambulance_siren_1LA8BxMlXSY_2+Helicopter_DSLO9BKHSEk_1.wav)
        expected_name = row['expected_filename']  # 期望的文件名 (如 nn_Ambulance (siren)_1LA8BxMlXSY_2.wav+Helicopter_DSLO9BKHSEk_1.wav.wav)
        rename_mapping[current_name] = expected_name
    
    print(f"创建了 {len(rename_mapping)} 个重命名映射")
    
    # 分类文件
    files_to_rename = []
    files_to_skip = []
    files_to_delete = []
    
    for current_file in current_files:
        current_name = current_file.name
        
        if current_name.startswith(('nn_', 'rn_')):
            files_to_skip.append(current_file)
        elif current_name in rename_mapping:
            files_to_rename.append(current_file)
        else:
            files_to_delete.append(current_file)
    
    print(f"\n文件分类结果:")
    print(f"  已有前缀的文件: {len(files_to_skip)} 个 (将跳过)")
    print(f"  需要重命名的文件: {len(files_to_rename)} 个")
    print(f"  需要删除的文件: {len(files_to_delete)} 个")
    
    # 显示重命名预览
    if files_to_rename:
        print(f"\n重命名预览（前10个文件）:")
        for i, current_file in enumerate(files_to_rename[:10], 1):
            current_name = current_file.name
            expected_name = rename_mapping[current_name]
            print(f"  {i:2d}. {current_name}")
            print(f"      -> {expected_name}")
        
        if len(files_to_rename) > 10:
            print(f"  ... 还有 {len(files_to_rename) - 10} 个文件需要重命名")
    
    # 显示删除预览
    if files_to_delete:
        print(f"\n删除预览（前10个文件）:")
        for i, current_file in enumerate(files_to_delete[:10], 1):
            print(f"  {i:2d}. {current_file.name} (将删除)")
        
        if len(files_to_delete) > 10:
            print(f"  ... 还有 {len(files_to_delete) - 10} 个文件需要删除")
    
    print(f"\n预览完成!")
    print(f"总共将重命名 {len(files_to_rename)} 个文件")
    print(f"总共将删除 {len(files_to_delete)} 个文件")
    print(f"总共将跳过 {len(files_to_skip)} 个已有前缀的文件")

if __name__ == "__main__":
    try:
        # 先预览
        preview_prefix_addition()
        
        # 询问用户是否要执行实际重命名
        print("\n" + "="*60)
        response = input("是否要执行实际的文件重命名操作？(y/n): ").strip().lower()
        if response == 'y':
            add_prefix_to_renamed_files()
        else:
            print("操作已取消。")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
