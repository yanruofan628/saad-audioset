#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件分类脚本
从remaining_all_mixed文件夹中：
1. 随机选择60个nearest_nn开头的文件
2. 随机选择60个random_rn开头的文件
3. 将这120个文件移动到selected_120文件夹
4. 将剩余的210个文件移动到remaining_210文件夹
"""

import os
import random
import shutil
from pathlib import Path

def classify_audio_files():
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
    
    # 获取所有文件
    all_files = list(source_dir.glob("*.wav"))
    print(f"总共找到 {len(all_files)} 个音频文件")
    
    # 分类文件
    nearest_nn_files = [f for f in all_files if f.name.startswith("nearest_nn")]
    random_rn_files = [f for f in all_files if f.name.startswith("random_rn")]
    
    print(f"nearest_nn开头的文件: {len(nearest_nn_files)} 个")
    print(f"random_rn开头的文件: {len(random_rn_files)} 个")
    
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
    
    # 移动选中的文件
    print("\n开始移动选中的文件...")
    for i, file_path in enumerate(selected_files, 1):
        dest_path = selected_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        if i % 20 == 0:
            print(f"已移动 {i}/{len(selected_files)} 个选中文件")
    
    print(f"已将所有 {len(selected_files)} 个选中文件移动到 {selected_dir}")
    
    # 移动剩余文件
    print("\n开始移动剩余文件...")
    for i, file_path in enumerate(remaining_files, 1):
        dest_path = remaining_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        if i % 20 == 0:
            print(f"已移动 {i}/{len(remaining_files)} 个剩余文件")
    
    print(f"已将所有 {len(remaining_files)} 个剩余文件移动到 {remaining_dir}")
    
    # 验证结果
    print("\n验证结果:")
    print(f"selected_120文件夹中的文件数: {len(list(selected_dir.glob('*.wav')))}")
    print(f"remaining_210文件夹中的文件数: {len(list(remaining_dir.glob('*.wav')))}")
    print(f"源文件夹中剩余文件数: {len(list(source_dir.glob('*.wav')))}")
    
    # 统计选中文件中的类型分布
    selected_nearest_count = len([f for f in selected_dir.glob("nearest_nn*")])
    selected_random_count = len([f for f in selected_dir.glob("random_rn*")])
    
    print(f"\n选中文件类型分布:")
    print(f"  nearest_nn: {selected_nearest_count} 个")
    print(f"  random_rn: {selected_random_count} 个")
    print(f"  总计: {selected_nearest_count + selected_random_count} 个")
    
    print("\n分类完成！")

if __name__ == "__main__":
    # 设置随机种子以便结果可重现（可选）
    random.seed(42)
    
    try:
        classify_audio_files()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
