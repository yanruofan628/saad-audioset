#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("脚本开始执行")

import os
import sys
import pandas as pd

print("导入库成功")

# 测试CSV文件读取
csv_file = r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv"
print(f"尝试读取CSV文件: {csv_file}")

try:
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"成功读取 {len(df)} 行数据")
    print(f"列名: {list(df.columns)}")
    print("前3行数据:")
    print(df.head(3))
except Exception as e:
    print(f"读取失败: {e}")

print("脚本执行完成")

