import os
import csv
import math
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OverallPreferencePlotter:
    """整体偏好绘图器"""
    
    def __init__(self):
        """初始化绘图器"""
        pass
        
    def read_all_data(self, csv_folder, txt_files):
        """
        读取所有数据并计算整体统计
        
        参数:
        csv_folder: CSV文件夹路径
        txt_files: TXT文件路径列表
        
        返回:
        overall_stats: 整体统计数据
        """
        print("正在读取所有数据...")
        
        # 读取CSV数据
        person_files = {
            'aiwenkai': ['aiwenkai_1.csv', 'aiwenkai2.csv'],
            'lironghua': ['lironghua.csv', 'lironghua2.csv'],
            'mayunmiao': ['mayunmiao_1.csv', 'mayunmiao2.csv'],
            'ShangZiyang': ['ShangZiyang.csv', 'ShangZiyang1.csv'],
            'wjy': ['wjy1.csv', 'wjy_2.csv']
        }
        
        all_selections = []
        
        # 读取CSV文件
        for person_name, filenames in person_files.items():
            for filename in filenames:
                file_path = os.path.join(csv_folder, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        
                        selections = [row['选择'] for row in rows]
                        all_selections.extend(selections)
                        
                    except Exception as e:
                        print(f"读取文件失败 {filename}: {e}")
        
        # 读取TXT文件
        person_txt_files = {
            'liyanchen': [
                r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
                r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt"
            ],
            'shimin': [
                r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
                r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt"
            ],
            'jiachen': [
                r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
                r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
            ]
        }
        
        for person_name, file_paths in person_txt_files.items():
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        selections = self.parse_txt_file(file_path)
                        # 转换为left/right格式
                        txt_selections = ['left' if s == 1 else 'right' for s in selections]
                        all_selections.extend(txt_selections)
                        
                    except Exception as e:
                        print(f"读取TXT文件失败 {file_path}: {e}")
        
        # 计算统计量
        total_selections = len(all_selections)
        left_count = all_selections.count('left')
        right_count = all_selections.count('right')
        left_proportion = left_count / total_selections
        
        # 计算95%置信区间
        alpha = 0.05
        z_alpha_half = 1.96
        
        # Wilson置信区间
        p_hat = left_proportion
        n_total = total_selections
        ci_lower = (p_hat + z_alpha_half**2/(2*n_total) - z_alpha_half * math.sqrt((p_hat*(1-p_hat) + z_alpha_half**2/(4*n_total))/n_total)) / (1 + z_alpha_half**2/n_total)
        ci_upper = (p_hat + z_alpha_half**2/(2*n_total) + z_alpha_half * math.sqrt((p_hat*(1-p_hat) + z_alpha_half**2/(4*n_total))/n_total)) / (1 + z_alpha_half**2/n_total)
        
        # 确保置信区间在[0,1]范围内
        ci_lower = max(0, ci_lower)
        ci_upper = min(1, ci_upper)
        
        overall_stats = {
            'total_selections': total_selections,
            'left_count': left_count,
            'right_count': right_count,
            'left_proportion': left_proportion,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_selections': all_selections
        }
        
        print(f"总选择次数: {total_selections}")
        print(f"Left选择次数: {left_count}")
        print(f"Right选择次数: {right_count}")
        print(f"Left比例: {left_proportion:.4f}")
        print(f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return overall_stats
    
    def parse_txt_file(self, file_path):
        """
        解析TXT文件，提取response数据
        
        参数:
        file_path: TXT文件路径
        
        返回:
        selections: 选择数据列表
        """
        selections = []
        # 尝试不同的编码方式
        encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
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
        
        return selections
    
    def perform_binomial_test(self, overall_stats):
        """
        执行二项假设检验
        
        参数:
        overall_stats: 整体统计数据
        
        返回:
        test_results: 检验结果
        """
        print("\n=== 执行二项假设检验 ===")
        
        n = overall_stats['total_selections']
        k = overall_stats['right_count']  # 成功次数（选择right）
        p = 0.5  # 假设概率
        
        # 使用正态近似计算p值
        mean = n * p
        variance = n * p * (1 - p)
        std_dev = math.sqrt(variance)
        
        # 连续性修正
        z = (k + 0.5 - mean) / std_dev
        
        # 计算p值（双尾检验）
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        
        # 计算效应量（Cohen's h）
        observed_p = k / n
        effect_size = 2 * (math.asin(math.sqrt(observed_p)) - math.asin(math.sqrt(p)))
        
        test_results = {
            'n_trials': n,
            'n_right': k,
            'n_left': n - k,
            'observed_proportion': observed_p,
            'expected_proportion': p,
            'p_value': p_value,
            'effect_size': effect_size,
            'z_score': z
        }
        
        print(f"二项检验结果:")
        print(f"  总试验次数: {n}")
        print(f"  选择right次数: {k}")
        print(f"  选择left次数: {n-k}")
        print(f"  观察到的right比例: {test_results['observed_proportion']:.4f}")
        print(f"  期望比例: {test_results['expected_proportion']:.4f}")
        print(f"  双尾检验p值: {p_value:.6f}")
        print(f"  效应量 (Cohen's h): {effect_size:.4f}")
        
        return test_results
    
    def create_overall_preference_plot(self, overall_stats, test_results, output_dir):
        """
        创建整体偏好图 - 点图+误差线版本
        
        参数:
        overall_stats: 整体统计数据
        test_results: 检验结果
        output_dir: 输出目录
        """
        print("\n=== 创建整体偏好图 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图形
        plt.figure(figsize=(8, 6))
        
        # 数据
        left_prop = overall_stats['left_proportion']
        ci_lower = overall_stats['ci_lower']
        ci_upper = overall_stats['ci_upper']
        
        # 计算标准误
        n = overall_stats['total_selections']
        se = math.sqrt(left_prop * (1 - left_prop) / n)
        
        # 绘制整体效应点图
        plt.errorbar(left_prop, 0, xerr=se*1.96, 
                    fmt='o', capsize=5, capthick=2, markersize=10,
                    color='red', label='Overall Effect ±95% CI')
        
        # 添加个体数据点（使用真实的个体数据）
        # 这里我们使用之前分析得到的个体数据
        individual_data = [0.285, 0.333, 0.361, 0.438, 0.476, 0.517, 0.615, 0.642]
        participant_names = ['ShangZiyang', 'jiachen', 'lironghua', 'liyanchen', 
                           'aiwenkai', 'shimin', 'wjy', 'mayunmiao']
        
        # 为每个个体分配固定的y位置，按顺序排列
        y_positions = [i * 0.1 for i in range(len(individual_data))]
        
        plt.scatter(individual_data, y_positions,
                   alpha=0.7, s=60, color='blue', label='Individual Participants')
        
        # 添加个体标签
        for i, (x, y, name) in enumerate(zip(individual_data, y_positions, participant_names)):
            plt.annotate(name, (x, y), xytext=(5, 0), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # 参考线
        plt.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Chance Level')
        
        # 设置图形属性
        plt.xlim(0.2, 0.8)
        plt.ylim(-0.1, 0.8)
        plt.xlabel('Left Selection Proportion', fontsize=12, fontweight='bold')
        plt.ylabel('Participants', fontsize=12, fontweight='bold')
        plt.yticks(y_positions, participant_names, fontsize=9)
        plt.title('Left/Right Selection Preference:\nOverall Effect and Individual Differences', 
                 fontsize=14, fontweight='bold')
        
        # 添加统计结果文本
        stats_text = f"""Statistical Results:
p-value = {test_results['p_value']:.3f}
Effect Size = {test_results['effect_size']:.3f}
n = {test_results['n_trials']} trials"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 添加图例
        plt.legend(loc='upper right', fontsize=10)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'overall_preference_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"整体偏好图已保存: {output_path}")
    
    def analyze_and_plot(self, csv_folder, txt_files, output_dir):
        """
        分析并绘制整体偏好图
        
        参数:
        csv_folder: CSV文件夹路径
        txt_files: TXT文件路径列表
        output_dir: 输出目录
        """
        print("=== 开始整体偏好分析 ===")
        
        # 读取所有数据
        overall_stats = self.read_all_data(csv_folder, txt_files)
        
        # 执行二项检验
        test_results = self.perform_binomial_test(overall_stats)
        
        # 创建整体偏好图
        self.create_overall_preference_plot(overall_stats, test_results, output_dir)
        
        print(f"\n整体偏好分析完成！结果保存到: {output_dir}")
        
        return overall_stats, test_results


def main():
    """主函数"""
    print("=== 整体偏好绘图工具 ===")
    
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
    
    output_dir = r"D:\D\research\audioset下载\overall_preference_results"
    
    # 创建绘图器
    plotter = OverallPreferencePlotter()
    
    # 分析并绘图
    try:
        overall_stats, test_results = plotter.analyze_and_plot(csv_folder, txt_files, output_dir)
        print("\n分析成功完成！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
