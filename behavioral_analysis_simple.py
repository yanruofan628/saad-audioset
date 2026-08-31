#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版行为实验数据分析器 - 不依赖pandas和matplotlib
"""

import os
import csv
import json
from collections import defaultdict
import statistics

class SimpleBehavioralAnalyzer:
    """简化版行为实验数据分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 定义主类别映射
        self.main_categories = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music']
        }
        
        # 定义目标类别（根据我们的假设）
        self.target_categories = {
            # main类型的目标类别
            'main': {
                'High Ecology': 'High Ecology',
                'Low Ecology': 'Low Ecology', 
                'speech': 'speech',
                'music': 'music',
                'High Ecology vs Low Ecology': 'High Ecology',
                'High Ecology vs speech': 'High Ecology',
                'High Ecology vs music': 'High Ecology',
                'Low Ecology vs speech': 'speech',
                'Low Ecology vs music': 'music',
                'speech vs music': 'speech'
            },
            # sub类型的目标类别
            'sub': {
                'Baby cry, infant cry vs Telephone bell ringing': 'Baby cry, infant cry',
                'Helicopter vs Computer keyboard': 'Helicopter',
                'Male speech, man speaking vs Female speech, woman speaking': 'Male speech, man speaking',
                'Bass drum vs Sad music': 'Bass drum'
            }
        }

    def get_main_category(self, sub_category):
        """根据子类别获取主类别"""
        for main_cat, sub_cats in self.main_categories.items():
            if sub_category in sub_cats:
                return main_cat
        return None

    def parse_audio_filename(self, filename):
        """解析音频文件名"""
        try:
            name_without_ext = filename.replace('.wav', '')
            
            if name_without_ext.startswith('nn_main_'):
                experiment_type = 'nn_main'
                name_part = name_without_ext[8:]
            elif name_without_ext.startswith('nn_sub_'):
                experiment_type = 'nn_sub'
                name_part = name_without_ext[7:]
            elif name_without_ext.startswith('main_'):
                experiment_type = 'main'
                name_part = name_without_ext[5:]
            elif name_without_ext.startswith('sub_'):
                experiment_type = 'sub'
                name_part = name_without_ext[4:]
            else:
                return None, None, None
            
            if '+' not in name_part:
                return None, None, None
            
            left_part, right_part = name_part.split('+', 1)
            
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None, None
            left_category = left_part[:left_underscore_pos]
            
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None, None
            right_category = right_part[:right_underscore_pos]
            
            return experiment_type, left_category, right_category
            
        except Exception as e:
            print(f"解析文件名失败 {filename}: {e}")
            return None, None, None

    def determine_contrast_type(self, experiment_type, left_category, right_category):
        """确定对比类型"""
        if experiment_type in ['main', 'nn_main']:
            left_main = self.get_main_category(left_category)
            right_main = self.get_main_category(right_category)
            
            if left_main == right_main:
                return f"{left_main} vs {left_main}"
            else:
                return f"{left_main} vs {right_main}"
        
        elif experiment_type in ['sub', 'nn_sub']:
            return f"{left_category} vs {right_category}"
        
        return None

    def is_target_selected(self, contrast_type, experiment_type, left_category, right_category, response):
        """判断是否选择了目标类别"""
        try:
            if experiment_type in ['main', 'nn_main']:
                target_main = self.target_categories['main'].get(contrast_type)
                if target_main is None:
                    return False
                
                left_main = self.get_main_category(left_category)
                right_main = self.get_main_category(right_category)
                
                if left_main == target_main:
                    return response == 1
                elif right_main == target_main:
                    return response == 2
                    
            elif experiment_type in ['sub', 'nn_sub']:
                target_sub = self.target_categories['sub'].get(contrast_type)
                if target_sub is None:
                    return False
                
                if left_category == target_sub:
                    return response == 1
                elif right_category == target_sub:
                    return response == 2
            
            return False
            
        except Exception as e:
            print(f"判断目标选择失败: {e}")
            return False

    def parse_experiment_data(self, file_path):
        """解析实验数据文件"""
        print(f"正在解析实验数据: {file_path}")
        
        trials = []
        encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                print(f"成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法使用任何编码读取文件: {file_path}")
        
        # 找到trial数据
        for i, line in enumerate(lines):
            if line.strip().startswith('Level: 3') and i+1 < len(lines) and '*** LogFrame Start ***' in lines[i+1]:
                trial_data = {}
                
                for j in range(i+1, min(i+20, len(lines))):
                    current_line = lines[j].strip()
                    
                    if 'wavfile:' in current_line:
                        trial_data['wavfile'] = current_line.split('wavfile:')[1].strip()
                    elif 'ImageDisplay1.RESP:' in current_line:
                        trial_data['response'] = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                    elif '*** LogFrame End ***' in current_line:
                        break
                
                if 'wavfile' in trial_data and 'response' in trial_data:
                    trials.append(trial_data)
        
        print(f"成功解析 {len(trials)} 个trial")
        return trials

    def parse_audio_mapping(self, file_path):
        """解析音频文件映射"""
        print(f"正在解析音频映射: {file_path}")
        
        mapping = {}
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 处理BOM字符
                        original_name_key = 'original_name'
                        if '\ufefforiginal_name' in row:
                            original_name_key = '\ufefforiginal_name'
                        mapping[row['new_name']] = row[original_name_key]
                print(f"成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法使用任何编码读取映射文件: {file_path}")
        
        print(f"成功解析 {len(mapping)} 个文件映射")
        return mapping

    def analyze_session(self, data_file, mapping_file):
        """分析单个session的数据"""
        print(f"\n=== 分析Session: {os.path.basename(data_file)} ===")
        
        trials = self.parse_experiment_data(data_file)
        mapping = self.parse_audio_mapping(mapping_file)
        
        contrast_results = defaultdict(list)
        
        for trial in trials:
            wavfile = trial['wavfile']
            response = trial['response']
            
            # 从wavfile中提取文件名
            filename = os.path.basename(wavfile)
            
            if filename not in mapping:
                print(f"警告: 未找到映射 {filename} (原始: {wavfile})")
                continue
            
            original_filename = mapping[filename]
            
            # 解析文件名
            experiment_type, left_category, right_category = self.parse_audio_filename(original_filename)
            
            if experiment_type is None:
                print(f"警告: 无法解析文件名 {original_filename}")
                continue
            
            # 确定对比类型
            contrast_type = self.determine_contrast_type(experiment_type, left_category, right_category)
            
            if contrast_type is None:
                print(f"警告: 无法确定对比类型 {original_filename}")
                continue
            
            # 判断是否选择了目标类别
            is_target = self.is_target_selected(contrast_type, experiment_type, left_category, right_category, response)
            
            # 存储结果
            key = f"{experiment_type}_{contrast_type}"
            contrast_results[key].append(is_target)
        
        print(f"成功分析 {len(contrast_results)} 种对比类型")
        return contrast_results

    def calculate_statistics(self, all_results):
        """计算统计量"""
        print("\n=== 计算统计量 ===")
        
        stats_results = {}
        
        for contrast_key, values in all_results.items():
            if len(values) == 0:
                continue
            
            # 计算比例
            target_ratio = sum(values) / len(values)
            
            # 计算95%置信区间
            n = len(values)
            if n > 1:
                # 简化的置信区间计算
                std_err = statistics.stdev(values) / (n ** 0.5)
                ci_lower = target_ratio - 1.96 * std_err
                ci_upper = target_ratio + 1.96 * std_err
            else:
                ci_lower = target_ratio
                ci_upper = target_ratio
            
            stats_results[contrast_key] = {
                'mean': target_ratio,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n': n,
                'values': values
            }
        
        print(f"计算了 {len(stats_results)} 种对比类型的统计量")
        return stats_results

    def save_results(self, stats_results, output_dir):
        """保存分析结果"""
        print("\n=== 保存分析结果 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, 'behavioral_analysis_results.csv')
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['experiment_type', 'contrast_type', 'target_preference_mean', 
                           'ci_lower', 'ci_upper', 'sample_size'])
            
            for contrast_key, stats in stats_results.items():
                experiment_type, contrast_type = contrast_key.split('_', 1)
                writer.writerow([
                    experiment_type,
                    contrast_type,
                    stats['mean'],
                    stats['ci_lower'],
                    stats['ci_upper'],
                    stats['n']
                ])
        
        # 保存为JSON
        json_path = os.path.join(output_dir, 'behavioral_analysis_results.json')
        json_data = {}
        for contrast_key, stats in stats_results.items():
            experiment_type, contrast_type = contrast_key.split('_', 1)
            json_data[f"{experiment_type}_{contrast_type}"] = {
                'target_preference_mean': stats['mean'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'sample_size': stats['n'],
                'individual_values': stats['values']
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("行为实验分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总对比类型数: {len(stats_results)}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            
            for contrast_key, stats in stats_results.items():
                experiment_type, contrast_type = contrast_key.split('_', 1)
                f.write(f"\n{experiment_type} - {contrast_type}:\n")
                f.write(f"  目标偏好均值: {stats['mean']:.3f}\n")
                f.write(f"  95%置信区间: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]\n")
                f.write(f"  样本量: {stats['n']}\n")
        
        print(f"分析结果已保存:")
        print(f"  CSV文件: {csv_path}")
        print(f"  JSON文件: {json_path}")
        print(f"  报告文件: {report_path}")

    def analyze_all_sessions(self, data_files, mapping_files, output_dir):
        """分析所有session的数据"""
        print("=== 开始分析所有session数据 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录已创建: {output_dir}")
        
        # 合并所有session的结果
        all_results = defaultdict(list)
        
        for data_file, mapping_file in zip(data_files, mapping_files):
            session_results = self.analyze_session(data_file, mapping_file)
            
            # 合并结果
            for contrast_key, values in session_results.items():
                all_results[contrast_key].extend(values)
        
        # 计算统计量
        stats_results = self.calculate_statistics(all_results)
        
        # 保存结果
        self.save_results(stats_results, output_dir)
        
        print(f"\n分析完成！结果保存到: {output_dir}")
        
        return stats_results


def main():
    """主函数"""
    print("=== 简化版行为实验数据分析工具 ===")
    
    # 设置文件路径
    data_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt"
    ]
    
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv"
    ]
    
    output_dir = r"D:\D\research\audioset下载\behavioral_analysis_results"
    
    # 创建分析器
    analyzer = SimpleBehavioralAnalyzer()
    
    # 分析所有数据
    try:
        results = analyzer.analyze_all_sessions(data_files, mapping_files, output_dir)
        print("\n分析成功完成！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
