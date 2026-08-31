#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_main():
    """测试主函数"""
    print("=== 大类吸引力分析工具 ===")
    print("开始执行...")
    
    # 测试文件路径
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
    ]
    
    print(f"TXT文件数量: {len(txt_files)}")
    for i, f in enumerate(txt_files):
        exists = os.path.exists(f)
        print(f"  {i+1}. {os.path.basename(f)} - {'存在' if exists else '不存在'}")
    
    print("测试完成！")

if __name__ == "__main__":
    test_main()