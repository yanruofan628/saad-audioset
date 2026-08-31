import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import warnings
from scipy import stats
import math

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BehavioralAnalyzer:
    """行为实验数据分析器"""
    
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
        
        # 定义对比类型
        self.contrast_types = {
            'main': [
                'High Ecology vs High Ecology',
                'Low Ecology vs Low Ecology',
                'speech vs speech', 
                'music vs music',
                'High Ecology vs Low Ecology',
                'High Ecology vs speech',
                'High Ecology vs music',
                'Low Ecology vs speech',
                'Low Ecology vs music',
                'speech vs music'
            ],
            'sub': [
                'Baby cry, infant cry vs Telephone bell ringing',
                'Helicopter vs Computer keyboard',
                'Male speech, man speaking vs Female speech, woman speaking',
                'Bass drum vs Sad music'
            ]
        }

    def parse_experiment_data(self, file_path):
        """
        解析实验数据文件
        
        参数:
        file_path: 实验数据文件路径
        
        返回:
        trials: 包含trial信息的列表
        """
        print(f"正在解析实验数据: {file_path}")
        
        trials = []
        # 尝试不同的编码方式
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
        
        # 找到trial数据开始的位置
        for i, line in enumerate(lines):
            if line.strip().startswith('Level: 3') and '*** LogFrame Start ***' in lines[i+1]:
                # 这是一个trial的开始
                trial_data = {}
                
                # 解析trial信息
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
        """
        解析音频文件映射
        
        参数:
        file_path: 映射文件路径
        
        返回:
        mapping: 文件名映射字典
        """
        print(f"正在解析音频映射: {file_path}")
        
        mapping = {}
        # 尝试不同的编码方式
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法使用任何编码读取映射文件: {file_path}")
        
        for _, row in df.iterrows():
            new_name = row['new_name']
            original_name = row['original_name']
            mapping[new_name] = original_name
        
        print(f"成功解析 {len(mapping)} 个文件映射")
        return mapping

    def parse_audio_filename(self, filename):
        """
        解析音频文件名，提取实验类型和类别信息
        
        参数:
        filename: 原始音频文件名
        
        返回:
        experiment_type: 实验类型 (main/sub/nn_main)
        left_category: 左声道类别
        right_category: 右声道类别
        """
        try:
            # 移除文件扩展名
            name_without_ext = filename.replace('.wav', '')
            
            # 检查实验类型
            if name_without_ext.startswith('nn_main_'):
                experiment_type = 'nn_main'
                # 移除nn_main_前缀
                name_part = name_without_ext[8:]
            elif name_without_ext.startswith('nn_sub_'):
                experiment_type = 'nn_sub'
                # 移除nn_sub_前缀  
                name_part = name_without_ext[7:]
            elif name_without_ext.startswith('main_'):
                experiment_type = 'main'
                # 移除main_前缀
                name_part = name_without_ext[5:]
            elif name_without_ext.startswith('sub_'):
                experiment_type = 'sub'
                # 移除sub_前缀
                name_part = name_without_ext[4:]
            else:
                return None, None, None
            
            # 按+分割左右声道
            if '+' not in name_part:
                return None, None, None
            
            left_part, right_part = name_part.split('+', 1)
            
            # 提取左声道类别（第一个_前的部分）
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None, None
            left_category = left_part[:left_underscore_pos]
            
            # 提取右声道类别（第一个_前的部分）
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None, None
            right_category = right_part[:right_underscore_pos]
            
            return experiment_type, left_category, right_category
            
        except Exception as e:
            print(f"解析文件名失败 {filename}: {e}")
            return None, None, None

    def get_main_category(self, sub_category):
        """
        根据子类别获取主类别
        
        参数:
        sub_category: 子类别名称
        
        返回:
        main_category: 主类别名称
        """
        for main_cat, sub_cats in self.main_categories.items():
            if sub_category in sub_cats:
                return main_cat
        return None

    def determine_contrast_type(self, experiment_type, left_category, right_category):
        """
        确定对比类型
        
        参数:
        experiment_type: 实验类型
        left_category: 左声道类别
        right_category: 右声道类别
        
        返回:
        contrast_type: 对比类型名称
        """
        if experiment_type in ['main', 'nn_main']:
            left_main = self.get_main_category(left_category)
            right_main = self.get_main_category(right_category)
            
            if left_main == right_main:
                # 同类别对比
                return f"{left_main} vs {left_main}"
            else:
                # 跨类别对比
                return f"{left_main} vs {right_main}"
        
        elif experiment_type in ['sub', 'nn_sub']:
            # 子类别对比
            return f"{left_category} vs {right_category}"
        
        return None

    def is_target_selected(self, contrast_type, experiment_type, left_category, right_category, response):
        """
        判断是否选择了目标类别
        
        参数:
        contrast_type: 对比类型
        experiment_type: 实验类型
        left_category: 左声道类别
        right_category: 右声道类别
        response: 选择结果 (1=左声道, 2=右声道)
        
        返回:
        is_target: 是否选择了目标类别
        """
        try:
            # 获取目标类别
            if experiment_type in ['main', 'nn_main']:
                target_main = self.target_categories['main'].get(contrast_type)
                if target_main is None:
                    return False
                
                left_main = self.get_main_category(left_category)
                right_main = self.get_main_category(right_category)
                
                # 判断目标类别在哪一侧
                if left_main == target_main:
                    return response == 1  # 选择左声道
                elif right_main == target_main:
                    return response == 2  # 选择右声道
                    
            elif experiment_type in ['sub', 'nn_sub']:
                target_sub = self.target_categories['sub'].get(contrast_type)
                if target_sub is None:
                    return False
                
                # 判断目标类别在哪一侧
                if left_category == target_sub:
                    return response == 1  # 选择左声道
                elif right_category == target_sub:
                    return response == 2  # 选择右声道
            
            return False
            
        except Exception as e:
            print(f"判断目标选择失败: {e}")
            return False

    def analyze_session(self, data_file, mapping_file):
        """
        分析单个session的数据
        
        参数:
        data_file: 实验数据文件路径
        mapping_file: 映射文件路径
        
        返回:
        session_results: session分析结果
        """
        print(f"\n=== 分析Session: {os.path.basename(data_file)} ===")
        
        # 解析数据
        trials = self.parse_experiment_data(data_file)
        mapping = self.parse_audio_mapping(mapping_file)
        
        # 分析每个trial
        contrast_results = {}
        
        for trial in trials:
            wavfile = trial['wavfile']
            response = trial['response']
            
            # 从wavfile中提取文件名（去掉audio/前缀）
            filename = os.path.basename(wavfile)  # 从"audio/a001.wav"提取"a001.wav"
            
            # 获取原始文件名
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
            if key not in contrast_results:
                contrast_results[key] = []
            
            contrast_results[key].append(is_target)
        
        print(f"成功分析 {len(contrast_results)} 种对比类型")
        return contrast_results

    def calculate_statistics(self, all_results):
        """
        计算统计量
        
        参数:
        all_results: 所有session的结果
        
        返回:
        stats_results: 统计结果
        """
        print("\n=== 计算统计量 ===")
        
        stats_results = {}
        
        for contrast_key, values in all_results.items():
            if len(values) == 0:
                continue
            
            # 计算比例
            target_ratio = np.mean(values)
            
            # 计算95%置信区间
            n = len(values)
            if n > 1:
                se = np.std(values, ddof=1) / np.sqrt(n)
                ci_lower = target_ratio - 1.96 * se
                ci_upper = target_ratio + 1.96 * se
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

    def plot_preference_distribution(self, stats_results, output_dir):
        """
        绘制偏好分布图
        
        参数:
        stats_results: 统计结果
        output_dir: 输出目录
        """
        print("\n=== 绘制偏好分布图 ===")
        
        # 检查是否有数据
        if not stats_results:
            print("警告: 没有数据可以绘制")
            return
        
        # 准备数据
        plot_data = []
        
        for contrast_key, stats in stats_results.items():
            experiment_type, contrast_type = contrast_key.split('_', 1)
            
            for value in stats['values']:
                plot_data.append({
                    'experiment_type': experiment_type,
                    'contrast_type': contrast_type,
                    'preference_ratio': value
                })
        
        if not plot_data:
            print("警告: 没有有效数据可以绘制")
            return
            
        df = pd.DataFrame(plot_data)
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('人类选择偏好分布分析', fontsize=16, fontweight='bold')
        
        # 为每个实验类型创建子图
        experiment_types = ['main', 'sub', 'nn_main']
        
        for i, exp_type in enumerate(experiment_types):
            if exp_type not in df['experiment_type'].values:
                axes[i].text(0.5, 0.5, f'没有{exp_type}类型数据', 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            exp_data = df[df['experiment_type'] == exp_type]
            
            # 获取对比类型列表
            contrast_types = exp_data['contrast_type'].unique()
            
            # 创建小提琴图
            if len(contrast_types) > 0:
                sns.violinplot(data=exp_data, x='contrast_type', y='preference_ratio', 
                             ax=axes[i], inner='box')
                
                # 添加散点图
                sns.stripplot(data=exp_data, x='contrast_type', y='preference_ratio',
                            ax=axes[i], color='black', alpha=0.6, size=3)
                
                # 添加随机水平线
                axes[i].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
                
                # 设置标签
                axes[i].set_title(f'{exp_type} 类型对比', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('对比类型', fontsize=12)
                axes[i].set_ylabel('偏好目标类别比例', fontsize=12)
                axes[i].set_ylim(0, 1)
                
                # 旋转x轴标签
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'preference_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"偏好分布图已保存: {output_path}")

    def plot_summary_statistics(self, stats_results, output_dir):
        """
        绘制汇总统计图
        
        参数:
        stats_results: 统计结果
        output_dir: 输出目录
        """
        print("\n=== 绘制汇总统计图 ===")
        
        # 检查是否有数据
        if not stats_results:
            print("警告: 没有数据可以绘制")
            return
        
        # 准备数据
        summary_data = []
        
        for contrast_key, stats in stats_results.items():
            experiment_type, contrast_type = contrast_key.split('_', 1)
            
            summary_data.append({
                'experiment_type': experiment_type,
                'contrast_type': contrast_type,
                'mean': stats['mean'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'n': stats['n']
            })
        
        if not summary_data:
            print("警告: 没有有效数据可以绘制")
            return
            
        df = pd.DataFrame(summary_data)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 为每个实验类型使用不同颜色
        colors = {'main': 'blue', 'sub': 'green', 'nn_main': 'red'}
        
        x_pos = 0
        x_labels = []
        x_positions = []
        
        for exp_type in ['main', 'sub', 'nn_main']:
            if exp_type not in df['experiment_type'].values:
                continue
            
            exp_data = df[df['experiment_type'] == exp_type].sort_values('contrast_type')
            
            for _, row in exp_data.iterrows():
                # 绘制均值点
                ax.errorbar(x_pos, row['mean'], 
                          yerr=[[row['mean'] - row['ci_lower']], 
                                [row['ci_upper'] - row['mean']]],
                          fmt='o', color=colors[exp_type], capsize=5, capthick=2,
                          markersize=8, label=exp_type)
                
                x_labels.append(row['contrast_type'])
                x_positions.append(x_pos)
                x_pos += 1
        
        # 添加随机水平线
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='随机水平')
        
        # 设置图形属性
        ax.set_xlabel('对比类型', fontsize=12, fontweight='bold')
        ax.set_ylabel('偏好目标类别比例', fontsize=12, fontweight='bold')
        ax.set_title('人类选择偏好汇总统计', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'summary_statistics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"汇总统计图已保存: {output_path}")

    def save_results(self, stats_results, output_dir):
        """
        保存分析结果
        
        参数:
        stats_results: 统计结果
        output_dir: 输出目录
        """
        print("\n=== 保存分析结果 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备保存数据
        results_data = []
        
        for contrast_key, stats in stats_results.items():
            experiment_type, contrast_type = contrast_key.split('_', 1)
            
            results_data.append({
                'experiment_type': experiment_type,
                'contrast_type': contrast_type,
                'target_preference_mean': stats['mean'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'sample_size': stats['n'],
                'individual_values': ','.join(map(str, stats['values']))
            })
        
        # 保存为CSV
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, 'behavioral_analysis_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("行为实验分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"总对比类型数: {len(stats_results)}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"\n{row['experiment_type']} - {row['contrast_type']}:\n")
                f.write(f"  目标偏好均值: {row['target_preference_mean']:.3f}\n")
                f.write(f"  95%置信区间: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]\n")
                f.write(f"  样本量: {row['sample_size']}\n")
        
        print(f"分析结果已保存:")
        print(f"  CSV文件: {csv_path}")
        print(f"  报告文件: {report_path}")

    def analyze_all_sessions(self, data_files, mapping_files, output_dir):
        """
        分析所有session的数据
        
        参数:
        data_files: 实验数据文件路径列表
        mapping_files: 映射文件路径列表
        output_dir: 输出目录
        """
        print("=== 开始分析所有session数据 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录已创建: {output_dir}")
        
        # 合并所有session的结果
        all_results = {}
        
        for data_file, mapping_file in zip(data_files, mapping_files):
            session_results = self.analyze_session(data_file, mapping_file)
            
            # 合并结果
            for contrast_key, values in session_results.items():
                if contrast_key not in all_results:
                    all_results[contrast_key] = []
                all_results[contrast_key].extend(values)
        
        # 计算统计量
        stats_results = self.calculate_statistics(all_results)
        
        # 绘制图表
        self.plot_preference_distribution(stats_results, output_dir)
        self.plot_summary_statistics(stats_results, output_dir)
        
        # 保存结果
        self.save_results(stats_results, output_dir)
        
        print(f"\n分析完成！结果保存到: {output_dir}")
        
        return stats_results


def main():
    """主函数"""
    print("=== 行为实验数据分析工具 ===")
    
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
    analyzer = BehavioralAnalyzer()
    
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
