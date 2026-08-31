#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 强制刷新输出
sys.stdout.flush()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleAnalyzer:
    """简化版分析器"""
    
    def __init__(self):
        print("初始化分析器...")
        sys.stdout.flush()
        
        # 定义主类别映射
        self.main_categories = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music']
        }
        
        # 创建反向映射
        self.sub_to_main = {}
        for main_cat, sub_cats in self.main_categories.items():
            for sub_cat in sub_cats:
                self.sub_to_main[sub_cat] = main_cat
        
        print("分析器初始化完成")
        sys.stdout.flush()

    def parse_csv_data(self, file_path):
        """解析CSV数据"""
        print(f"解析CSV文件: {os.path.basename(file_path)}")
        sys.stdout.flush()
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"成功读取 {len(df)} 行数据")
            sys.stdout.flush()
            return df
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            sys.stdout.flush()
            return None

    def test_analysis(self):
        """测试分析"""
        print("开始测试分析...")
        sys.stdout.flush()
        
        # 测试CSV文件
        csv_file = r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv"
        df = self.parse_csv_data(csv_file)
        
        if df is not None:
            print(f"CSV文件列名: {list(df.columns)}")
            print(f"前5行数据:")
            print(df.head())
            sys.stdout.flush()
        
        print("测试分析完成")
        sys.stdout.flush()

def main():
    """主函数"""
    print("=== 简化版大类吸引力分析工具 ===")
    print("开始执行...")
    sys.stdout.flush()
    
    try:
        analyzer = SimpleAnalyzer()
        analyzer.test_analysis()
        print("分析成功完成！")
        sys.stdout.flush()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    main()

