import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BinomialTestAnalyzer:
    """二项假设检验分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.all_selections = []  # 存储所有选择数据
        self.participant_data = {}  # 存储每个被试者的数据
        
    def read_csv_files(self, csv_folder):
        """
        读取CSV文件夹中的所有CSV文件
        
        参数:
        csv_folder: CSV文件夹路径
        
        返回:
        csv_data: CSV数据列表
        """
        print(f"正在读取CSV文件夹: {csv_folder}")
        
        csv_files = [
            "aiwenkai_1.csv", "aiwenkai2.csv", "lironghua.csv", "lironghua2.csv",
            "mayunmiao_1.csv", "mayunmiao2.csv", "ShangZiyang.csv", "ShangZiyang1.csv",
            "wjy1.csv", "wjy_2.csv"
        ]
        
        csv_data = []
        
        for filename in csv_files:
            file_path = os.path.join(csv_folder, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    print(f"成功读取: {filename}, 数据行数: {len(df)}")
                    
                    # 提取选择数据
                    selections = df['选择'].tolist()
                    participant_name = filename.replace('.csv', '')
                    
                    # 转换为数值：left=0, right=1
                    numeric_selections = [1 if s == 'right' else 0 for s in selections]
                    
                    csv_data.append({
                        'participant': participant_name,
                        'selections': numeric_selections,
                        'left_count': selections.count('left'),
                        'right_count': selections.count('right'),
                        'total': len(selections)
                    })
                    
                except Exception as e:
                    print(f"读取文件失败 {filename}: {e}")
            else:
                print(f"文件不存在: {filename}")
        
        print(f"成功读取 {len(csv_data)} 个CSV文件")
        return csv_data
    
    def parse_txt_file(self, file_path):
        """
        解析TXT文件，提取response数据
        
        参数:
        file_path: TXT文件路径
        
        返回:
        selections: 选择数据列表
        """
        print(f"正在解析TXT文件: {file_path}")
        
        selections = []
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
                    
                    if 'ImageDisplay1.RESP:' in current_line:
                        response = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                        selections.append(response)
                        break
                    elif '*** LogFrame End ***' in current_line:
                        break
        
        print(f"成功解析 {len(selections)} 个trial")
        return selections
    
    def read_txt_files(self, txt_files):
        """
        读取TXT文件列表
        
        参数:
        txt_files: TXT文件路径列表
        
        返回:
        txt_data: TXT数据列表
        """
        print(f"正在读取 {len(txt_files)} 个TXT文件")
        
        txt_data = []
        
        for file_path in txt_files:
            if os.path.exists(file_path):
                try:
                    selections = self.parse_txt_file(file_path)
                    
                    # 转换为数值：1=left=0, 2=right=1
                    numeric_selections = [s - 1 for s in selections]  # 1->0, 2->1
                    
                    participant_name = os.path.basename(os.path.dirname(file_path))
                    
                    txt_data.append({
                        'participant': participant_name,
                        'selections': numeric_selections,
                        'left_count': numeric_selections.count(0),
                        'right_count': numeric_selections.count(1),
                        'total': len(numeric_selections)
                    })
                    
                except Exception as e:
                    print(f"读取TXT文件失败 {file_path}: {e}")
            else:
                print(f"TXT文件不存在: {file_path}")
        
        print(f"成功读取 {len(txt_data)} 个TXT文件")
        return txt_data
    
    def combine_all_data(self, csv_data, txt_data):
        """
        合并所有数据
        
        参数:
        csv_data: CSV数据列表
        txt_data: TXT数据列表
        
        返回:
        combined_data: 合并后的数据
        """
        print("\n=== 合并所有数据 ===")
        
        all_data = csv_data + txt_data
        
        # 统计总体数据
        total_left = sum([d['left_count'] for d in all_data])
        total_right = sum([d['right_count'] for d in all_data])
        total_selections = sum([d['total'] for d in all_data])
        
        # 合并所有选择数据
        all_selections = []
        for data in all_data:
            all_selections.extend(data['selections'])
        
        combined_data = {
            'participants': all_data,
            'total_left': total_left,
            'total_right': total_right,
            'total_selections': total_selections,
            'all_selections': all_selections,
            'left_proportion': total_left / total_selections if total_selections > 0 else 0,
            'right_proportion': total_right / total_selections if total_selections > 0 else 0
        }
        
        print(f"总被试者数: {len(all_data)}")
        print(f"总选择次数: {total_selections}")
        print(f"选择left次数: {total_left}")
        print(f"选择right次数: {total_right}")
        print(f"left比例: {combined_data['left_proportion']:.4f}")
        print(f"right比例: {combined_data['right_proportion']:.4f}")
        
        return combined_data
    
    def perform_binomial_test(self, combined_data):
        """
        执行二项假设检验
        
        参数:
        combined_data: 合并后的数据
        
        返回:
        test_results: 检验结果
        """
        print("\n=== 执行二项假设检验 ===")
        
        n = combined_data['total_selections']  # 总试验次数
        k = combined_data['total_right']  # 成功次数（选择right）
        p = 0.5  # 假设概率
        
        # 执行二项检验
        # 检验H0: p = 0.5 vs H1: p ≠ 0.5
        p_value_two_tailed = stats.binom_test(k, n, p, alternative='two-sided')
        
        # 检验H0: p = 0.5 vs H1: p > 0.5 (right偏好)
        p_value_right_tailed = stats.binom_test(k, n, p, alternative='greater')
        
        # 检验H0: p = 0.5 vs H1: p < 0.5 (left偏好)
        p_value_left_tailed = stats.binom_test(k, n, p, alternative='less')
        
        # 计算置信区间
        alpha = 0.05
        ci_lower = stats.binom.interval(1-alpha, n, k/n)[0] / n
        ci_upper = stats.binom.interval(1-alpha, n, k/n)[1] / n
        
        test_results = {
            'n_trials': n,
            'n_right': k,
            'n_left': n - k,
            'observed_proportion': k / n,
            'expected_proportion': 0.5,
            'p_value_two_tailed': p_value_two_tailed,
            'p_value_right_tailed': p_value_right_tailed,
            'p_value_left_tailed': p_value_left_tailed,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_two_tailed': p_value_two_tailed < 0.05,
            'significant_right_tailed': p_value_right_tailed < 0.05,
            'significant_left_tailed': p_value_left_tailed < 0.05
        }
        
        print(f"二项检验结果:")
        print(f"  总试验次数: {n}")
        print(f"  选择right次数: {k}")
        print(f"  选择left次数: {n-k}")
        print(f"  观察到的right比例: {test_results['observed_proportion']:.4f}")
        print(f"  期望比例: {test_results['expected_proportion']:.4f}")
        print(f"  双尾检验p值: {p_value_two_tailed:.6f}")
        print(f"  右尾检验p值: {p_value_right_tailed:.6f}")
        print(f"  左尾检验p值: {p_value_left_tailed:.6f}")
        print(f"  95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        if test_results['significant_two_tailed']:
            print(f"  结论: 拒绝H0，选择比例显著偏离0.5")
        else:
            print(f"  结论: 不拒绝H0，选择比例不显著偏离0.5")
        
        return test_results
    
    def create_visualization(self, combined_data, test_results, output_dir):
        """
        创建可视化图表
        
        参数:
        combined_data: 合并后的数据
        test_results: 检验结果
        output_dir: 输出目录
        """
        print("\n=== 创建可视化图表 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('二项假设检验分析结果', fontsize=16, fontweight='bold')
        
        # 1. 总体选择分布饼图
        ax1 = axes[0, 0]
        labels = ['Left', 'Right']
        sizes = [combined_data['total_left'], combined_data['total_right']]
        colors = ['lightblue', 'lightcoral']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'总体选择分布\n(总计: {combined_data["total_selections"]}次)', fontsize=12, fontweight='bold')
        
        # 2. 各被试者选择比例条形图
        ax2 = axes[0, 1]
        participants = [d['participant'] for d in combined_data['participants']]
        right_proportions = [d['right_count'] / d['total'] for d in combined_data['participants']]
        
        bars = ax2.bar(range(len(participants)), right_proportions, color='lightcoral', alpha=0.7)
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='期望比例 (0.5)')
        ax2.set_xlabel('被试者', fontsize=10)
        ax2.set_ylabel('Right选择比例', fontsize=10)
        ax2.set_title('各被试者Right选择比例', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(len(participants)))
        ax2.set_xticklabels(participants, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 选择次数对比条形图
        ax3 = axes[1, 0]
        categories = ['Left', 'Right']
        counts = [combined_data['total_left'], combined_data['total_right']]
        
        bars = ax3.bar(categories, counts, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax3.set_ylabel('选择次数', fontsize=12)
        ax3.set_title('Left vs Right选择次数对比', fontsize=12, fontweight='bold')
        
        # 在条形图上添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计检验结果展示
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 创建结果文本
        result_text = f"""
二项假设检验结果

总试验次数: {test_results['n_trials']}
Right选择次数: {test_results['n_right']}
Left选择次数: {test_results['n_left']}

观察到的Right比例: {test_results['observed_proportion']:.4f}
期望比例: {test_results['expected_proportion']:.4f}

p值 (双尾检验): {test_results['p_value_two_tailed']:.6f}
p值 (右尾检验): {test_results['p_value_right_tailed']:.6f}
p值 (左尾检验): {test_results['p_value_left_tailed']:.6f}

95%置信区间: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]

结论: {'拒绝H0，选择比例显著偏离0.5' if test_results['significant_two_tailed'] else '不拒绝H0，选择比例不显著偏离0.5'}
        """
        
        ax4.text(0.1, 0.9, result_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'binomial_test_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化图表已保存: {output_path}")
    
    def save_results(self, combined_data, test_results, output_dir):
        """
        保存分析结果
        
        参数:
        combined_data: 合并后的数据
        test_results: 检验结果
        output_dir: 输出目录
        """
        print("\n=== 保存分析结果 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_data = []
        for participant_data in combined_data['participants']:
            results_data.append({
                'participant': participant_data['participant'],
                'left_count': participant_data['left_count'],
                'right_count': participant_data['right_count'],
                'total': participant_data['total'],
                'right_proportion': participant_data['right_count'] / participant_data['total']
            })
        
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, 'binomial_test_detailed_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, 'binomial_test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("二项假设检验分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"总被试者数: {len(combined_data['participants'])}\n")
            f.write(f"总试验次数: {test_results['n_trials']}\n")
            f.write(f"Right选择次数: {test_results['n_right']}\n")
            f.write(f"Left选择次数: {test_results['n_left']}\n\n")
            
            f.write("统计检验结果:\n")
            f.write("-" * 30 + "\n")
            f.write(f"观察到的Right比例: {test_results['observed_proportion']:.4f}\n")
            f.write(f"期望比例: {test_results['expected_proportion']:.4f}\n")
            f.write(f"双尾检验p值: {test_results['p_value_two_tailed']:.6f}\n")
            f.write(f"右尾检验p值: {test_results['p_value_right_tailed']:.6f}\n")
            f.write(f"左尾检验p值: {test_results['p_value_left_tailed']:.6f}\n")
            f.write(f"95%置信区间: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]\n\n")
            
            f.write("结论:\n")
            if test_results['significant_two_tailed']:
                f.write("拒绝H0，选择比例显著偏离0.5\n")
            else:
                f.write("不拒绝H0，选择比例不显著偏离0.5\n")
        
        print(f"分析结果已保存:")
        print(f"  详细结果CSV: {csv_path}")
        print(f"  汇总报告: {report_path}")
    
    def analyze_all_data(self, csv_folder, txt_files, output_dir):
        """
        分析所有数据
        
        参数:
        csv_folder: CSV文件夹路径
        txt_files: TXT文件路径列表
        output_dir: 输出目录
        """
        print("=== 开始二项假设检验分析 ===")
        
        # 读取数据
        csv_data = self.read_csv_files(csv_folder)
        txt_data = self.read_txt_files(txt_files)
        
        # 合并数据
        combined_data = self.combine_all_data(csv_data, txt_data)
        
        # 执行二项检验
        test_results = self.perform_binomial_test(combined_data)
        
        # 创建可视化
        self.create_visualization(combined_data, test_results, output_dir)
        
        # 保存结果
        self.save_results(combined_data, test_results, output_dir)
        
        print(f"\n分析完成！结果保存到: {output_dir}")
        
        return combined_data, test_results


def main():
    """主函数"""
    print("=== 二项假设检验分析工具 ===")
    
    # 设置文件路径
    csv_folder = r"D:\D\research\audioset下载\experiment_output2"
    
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
    ]
    
    output_dir = r"D:\D\research\audioset下载\binomial_test_results"
    
    # 创建分析器
    analyzer = BinomialTestAnalyzer()
    
    # 分析所有数据
    try:
        combined_data, test_results = analyzer.analyze_all_data(csv_folder, txt_files, output_dir)
        print("\n分析成功完成！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
