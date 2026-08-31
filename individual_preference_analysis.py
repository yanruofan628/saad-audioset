import os
import csv
import math
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IndividualPreferenceAnalyzer:
    """个体偏好分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.participant_data = []
        
    def read_csv_files(self, csv_folder):
        """
        读取CSV文件夹中的所有CSV文件，按人合并数据
        
        参数:
        csv_folder: CSV文件夹路径
        
        返回:
        csv_data: CSV数据列表
        """
        print(f"正在读取CSV文件夹: {csv_folder}")
        
        # 按人分组的文件
        person_files = {
            'aiwenkai': ['aiwenkai_1.csv', 'aiwenkai2.csv'],
            'lironghua': ['lironghua.csv', 'lironghua2.csv'],
            'mayunmiao': ['mayunmiao_1.csv', 'mayunmiao2.csv'],
            'ShangZiyang': ['ShangZiyang.csv', 'ShangZiyang1.csv'],
            'wjy': ['wjy1.csv', 'wjy_2.csv'],
            'LiuYaorui': ['LiuYaorui_1.csv', 'Liu Yaorui2.csv']
        }
        
        csv_data = []
        
        for person_name, filenames in person_files.items():
            person_selections = []
            person_left_count = 0
            person_right_count = 0
            
            for filename in filenames:
                file_path = os.path.join(csv_folder, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        
                        print(f"成功读取: {filename}, 数据行数: {len(rows)}")
                        
                        # 提取选择数据
                        selections = [row['选择'] for row in rows]
                        person_selections.extend(selections)
                        person_left_count += selections.count('left')
                        person_right_count += selections.count('right')
                        
                    except Exception as e:
                        print(f"读取文件失败 {filename}: {e}")
                else:
                    print(f"文件不存在: {filename}")
            
            if person_selections:
                # 转换为数值：left=0, right=1
                numeric_selections = [1 if s == 'right' else 0 for s in person_selections]
                
                csv_data.append({
                    'participant': person_name,
                    'selections': numeric_selections,
                    'left_count': person_left_count,
                    'right_count': person_right_count,
                    'total': len(person_selections)
                })
        
        print(f"成功读取 {len(csv_data)} 个人的CSV数据")
        return csv_data
    
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
    
    def read_txt_files(self, txt_files):
        """
        读取TXT文件列表，按人合并数据
        
        参数:
        txt_files: TXT文件路径列表
        
        返回:
        txt_data: TXT数据列表
        """
        print(f"正在读取 {len(txt_files)} 个TXT文件")
        
        # 按人分组的文件
        person_files = {
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
        
        txt_data = []
        
        for person_name, file_paths in person_files.items():
            person_selections = []
            person_left_count = 0
            person_right_count = 0
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        selections = self.parse_txt_file(file_path)
                        print(f"成功解析: {os.path.basename(file_path)}, trial数: {len(selections)}")
                        
                        person_selections.extend(selections)
                        
                    except Exception as e:
                        print(f"读取TXT文件失败 {file_path}: {e}")
                else:
                    print(f"TXT文件不存在: {file_path}")
            
            if person_selections:
                # 转换为数值：1=left=0, 2=right=1
                numeric_selections = [s - 1 for s in person_selections]  # 1->0, 2->1
                person_left_count = numeric_selections.count(0)
                person_right_count = numeric_selections.count(1)
                
                txt_data.append({
                    'participant': person_name,
                    'selections': numeric_selections,
                    'left_count': person_left_count,
                    'right_count': person_right_count,
                    'total': len(numeric_selections)
                })
        
        print(f"成功读取 {len(txt_data)} 个人的TXT数据")
        return txt_data
    
    def create_tornado_chart(self, all_data, output_dir):
        """
        创建旋风图展示个体偏好
        
        参数:
        all_data: 所有被试者数据
        output_dir: 输出目录
        """
        print("\n=== 创建旋风图 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        data_for_plot = []
        for participant_data in all_data:
            left_prop = participant_data['left_count'] / participant_data['total']
            right_prop = participant_data['right_count'] / participant_data['total']
            
            data_for_plot.append({
                'participant': participant_data['participant'],
                'left_proportion': left_prop,
                'right_proportion': right_prop,
                'total': participant_data['total']
            })
        
        # 先计算整体平均和标准差（基于原始被试者数据）
        import numpy as np
        all_left_props = [d['left_proportion'] for d in data_for_plot]
        all_right_props = [d['right_proportion'] for d in data_for_plot]
        
        mean_left = np.mean(all_left_props)
        mean_right = np.mean(all_right_props)
        std_left = np.std(all_left_props)
        std_right = np.std(all_right_props)
        
        # 计算95%置信区间
        from scipy import stats
        n_participants = len(data_for_plot)  # 原始被试者数量
        se_left = std_left / np.sqrt(n_participants)
        se_right = std_right / np.sqrt(n_participants)
        t_critical = stats.t.ppf(0.975, n_participants - 1)  # 95%置信区间
        
        ci_left_lower = mean_left - t_critical * se_left
        ci_left_upper = mean_left + t_critical * se_left
        ci_right_lower = mean_right - t_critical * se_right
        ci_right_upper = mean_right + t_critical * se_right
        
        # 先按左耳比例排序（只对原始被试者数据排序）
        data_for_plot.sort(key=lambda x: x['left_proportion'])
        
        # 创建平均行和CI行数据
        avg_row = {
            'participant': 'Overall Average',
            'left_proportion': mean_left,
            'right_proportion': mean_right,
            'total': len(data_for_plot),
            'is_average': True,
            'std_left': std_left,
            'std_right': std_right
        }
        
        ci_row = {
            'participant': '95% CI',
            'left_proportion': mean_left,
            'right_proportion': mean_right,
            'total': len(data_for_plot),
            'is_ci': True,
            'ci_left_lower': ci_left_lower,
            'ci_left_upper': ci_left_upper,
            'ci_right_lower': ci_right_lower,
            'ci_right_upper': ci_right_upper
        }
        
        # 重新构建数据列表：先添加排序后的被试者数据，再添加平均行和CI行
        sorted_participants = data_for_plot.copy()
        data_for_plot = sorted_participants + [avg_row, ci_row]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # 设置y轴位置（包括平均行）
        y_pos = list(range(len(data_for_plot)))
        
        # 绘制左耳选择的条形（向左延伸，负值）
        left_props = [d['left_proportion'] for d in data_for_plot]
        bars_left = ax.barh(y_pos, [-lp for lp in left_props], 
                           color='skyblue', edgecolor='grey', alpha=0.8, label='Left Selection')
        
        # 绘制右耳选择的条形（向右延伸，正值）
        right_props = [d['right_proportion'] for d in data_for_plot]
        bars_right = ax.barh(y_pos, right_props, 
                            color='lightcoral', edgecolor='grey', alpha=0.8, label='Right Selection')
        
        # 为平均行添加误差线 - 平均行固定在倒数第二行
        avg_row_idx = len(data_for_plot) - 2
        avg_data = data_for_plot[avg_row_idx]
        
        # 左耳误差线（负值区域）
        ax.errorbar(-avg_data['left_proportion'], avg_row_idx, 
                   xerr=avg_data['std_left'], 
                   color='darkblue', linewidth=2, capsize=5, capthick=2)
        
        # 右耳误差线（正值区域）
        ax.errorbar(avg_data['right_proportion'], avg_row_idx, 
                   xerr=avg_data['std_right'], 
                   color='darkred', linewidth=2, capsize=5, capthick=2)
        
        # 为置信区间行添加置信区间条形 - 置信区间行固定在最后一行
        ci_row_idx = len(data_for_plot) - 1
        ci_data = data_for_plot[ci_row_idx]
        
        # 左耳置信区间条形（负值区域）
        ax.barh(ci_row_idx, -ci_data['ci_left_upper'], 
                height=0.3, color='lightblue', alpha=0.6, edgecolor='darkblue', linewidth=1)
        ax.barh(ci_row_idx, -ci_data['ci_left_lower'], 
                height=0.3, color='white', alpha=0.8, edgecolor='darkblue', linewidth=1)
        
        # 右耳置信区间条形（正值区域）
        ax.barh(ci_row_idx, ci_data['ci_right_upper'], 
                height=0.3, color='lightcoral', alpha=0.6, edgecolor='darkred', linewidth=1)
        ax.barh(ci_row_idx, ci_data['ci_right_lower'], 
                height=0.3, color='white', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # 添加数值标签
        for i, (left_val, right_val, participant) in enumerate(zip(left_props, right_props, [d['participant'] for d in data_for_plot])):
            # 检查是否是平均行或置信区间行
            is_average_row = i == avg_row_idx
            is_ci_row = i == ci_row_idx
            
            if is_average_row:
                # 平均行的标签位置要避开误差线
                # 左耳标签（负值区域，位置更高）
                ax.text(-left_val - 0.02, i + 0.5, f'{left_val:.1%}', va='center', ha='right', fontsize=16, fontweight='bold')
                # 右耳标签（正值区域，位置更高）
                ax.text(right_val + 0.02, i + 0.5, f'{right_val:.1%}', va='center', ha='left', fontsize=16, fontweight='bold')
            elif is_ci_row:
                # 置信区间行的标签位置 - 需要上移避开色块
                # 左耳标签（负值区域，位置更高）
                ax.text(-left_val - 0.02, i + 0.4, f'{left_val:.1%}', va='center', ha='right', fontsize=16, fontweight='bold')
                # 右耳标签（正值区域，位置更高）
                ax.text(right_val + 0.02, i + 0.4, f'{right_val:.1%}', va='center', ha='left', fontsize=16, fontweight='bold')
            else:
                # 普通行的标签位置
                # 左耳标签（负值区域）
                ax.text(-left_val - 0.02, i, f'{left_val:.1%}', va='center', ha='right', fontsize=16, fontweight='bold')
                # 右耳标签（正值区域）
                ax.text(right_val + 0.02, i, f'{right_val:.1%}', va='center', ha='left', fontsize=16, fontweight='bold')
        
        # 美化图形
        ax.axvline(0, color='black', linewidth=1.5, alpha=0.8)
        ax.axvline(-0.5, color='grey', linestyle='--', alpha=0.7, linewidth=1, label='Chance Level (50%)')
        ax.axvline(0.5, color='grey', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([d['participant'] for d in data_for_plot], fontsize=18)
        ax.set_xlabel('Selection Proportion', fontsize=20, fontweight='bold')
        ax.set_title('Individual Left/Right Selection Preference Distribution', fontsize=22, fontweight='bold')
        
        # 将图例放在上方横向排列，避免与标题重叠
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=16, frameon=True)
        
        # 设置x轴范围
        ax.set_xlim(-1.0, 1.0)
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='x')
        
        # 调整布局，为上方图例留出更多空间
        plt.tight_layout()
        plt.subplots_adjust(top=0.70)
        
        # 保存图片
        output_path = os.path.join(output_dir, 'individual_preference_tornado_chart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Tornado chart saved: {output_path}")
        print(f"Overall statistics: Left average={mean_left:.1%}±{std_left:.1%}, Right average={mean_right:.1%}±{std_right:.1%}")
        print(f"95% CI: Left [{ci_left_lower:.1%}, {ci_left_upper:.1%}], Right [{ci_right_lower:.1%}, {ci_right_upper:.1%}]")
        print(f"Average row and 95% CI row added to bottom of chart")
        
        return data_for_plot
    
    def create_preference_summary(self, data_for_plot, output_dir):
        """
        创建偏好汇总表
        
        参数:
        data_for_plot: 绘图数据
        output_dir: 输出目录
        """
        print("\n=== 创建偏好汇总表 ===")
        
        # 保存汇总数据
        csv_path = os.path.join(output_dir, 'individual_preference_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['participant', 'left_count', 'right_count', 'total', 'left_proportion', 'right_proportion', 'preference_direction', 'preference_strength'])
            
            for data in data_for_plot:
                left_count = int(data['left_proportion'] * data['total'])
                right_count = int(data['right_proportion'] * data['total'])
                
                # 确定偏好方向和强度
                if data['left_proportion'] > 0.5:
                    direction = 'Left偏好'
                    strength = abs(data['left_proportion'] - 0.5) * 2  # 转换为0-1的强度
                elif data['right_proportion'] > 0.5:
                    direction = 'Right偏好'
                    strength = abs(data['right_proportion'] - 0.5) * 2
                else:
                    direction = '无偏好'
                    strength = 0.0
                
                writer.writerow([
                    data['participant'],
                    left_count,
                    right_count,
                    data['total'],
                    f"{data['left_proportion']:.4f}",
                    f"{data['right_proportion']:.4f}",
                    direction,
                    f"{strength:.4f}"
                ])
        
        # 创建文本报告
        report_path = os.path.join(output_dir, 'individual_preference_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("个体偏好分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("各被试者偏好分析:\n")
            f.write("-" * 30 + "\n")
            
            for data in data_for_plot:
                left_count = int(data['left_proportion'] * data['total'])
                right_count = int(data['right_proportion'] * data['total'])
                
                if data['left_proportion'] > 0.5:
                    direction = 'Left偏好'
                    strength = abs(data['left_proportion'] - 0.5) * 2
                elif data['right_proportion'] > 0.5:
                    direction = 'Right偏好'
                    strength = abs(data['right_proportion'] - 0.5) * 2
                else:
                    direction = '无偏好'
                    strength = 0.0
                
                f.write(f"{data['participant']}:\n")
                f.write(f"  Left选择: {left_count}次 ({data['left_proportion']:.1%})\n")
                f.write(f"  Right选择: {right_count}次 ({data['right_proportion']:.1%})\n")
                f.write(f"  偏好方向: {direction}\n")
                f.write(f"  偏好强度: {strength:.1%}\n\n")
        
        print(f"偏好汇总已保存:")
        print(f"  CSV文件: {csv_path}")
        print(f"  报告文件: {report_path}")
    
    def analyze_individual_preferences(self, csv_folder, txt_files, output_dir):
        """
        分析个体偏好
        
        参数:
        csv_folder: CSV文件夹路径
        txt_files: TXT文件路径列表
        output_dir: 输出目录
        """
        print("=== 开始个体偏好分析 ===")
        
        # 读取数据
        csv_data = self.read_csv_files(csv_folder)
        txt_data = self.read_txt_files(txt_files)
        
        # 合并所有数据
        all_data = csv_data + txt_data
        
        print(f"\n总被试者数: {len(all_data)}")
        for data in all_data:
            print(f"{data['participant']}: Left={data['left_count']}, Right={data['right_count']}, 总计={data['total']}")
        
        # 创建旋风图
        data_for_plot = self.create_tornado_chart(all_data, output_dir)
        
        # 创建偏好汇总
        self.create_preference_summary(data_for_plot, output_dir)
        
        print(f"\n个体偏好分析完成！结果保存到: {output_dir}")
        
        return all_data, data_for_plot


def main():
    """主函数"""
    print("=== 个体偏好分析工具 ===")
    
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
    
    output_dir = r"D:\D\research\audioset下载\individual_preference_results"
    
    # 创建分析器
    analyzer = IndividualPreferenceAnalyzer()
    
    # 分析个体偏好
    try:
        all_data, data_for_plot = analyzer.analyze_individual_preferences(csv_folder, txt_files, output_dir)
        print("\n分析成功完成！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
