import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import re

warnings.filterwarnings("ignore")


class HumanSelectionHeatmapAnalyzer:
    """人类选择结果热力图分析器"""

    def __init__(self):
        """初始化分析器"""
        self.main_category_colors = {
            'High Ecology': '#FF6B6B',  # 红色系列
            'Low Ecology': '#DDA0DD',  # 紫色系列
            'speech': '#45B7D1',  # 蓝色系列
            'music': '#96CEB4',  # 绿色系列
            'Unknown Source': '#FFEAA7',  # 黄色系列
        }
        
        # 定义15个目标类别
        self.target_categories = [
            'Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry',  # 高生态效度
            'Computer keyboard', 'Helicopter', 'Chicken, rooster',  # 低生态效度
            'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',  # speech
            'Bass drum', 'Funny music', 'Sad music',  # music
            'Pulse', 'Whack, thwack', 'Crumpling, crinkling'  # 未知声源
        ]
        
        # 定义主类别分组
        self.main_category_groups = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
        }

    def parse_filename(self, filename):
        """
        解析音频文件名，提取左右声道类别信息
        
        参数:
        filename: 音频文件名，格式为 nn_左声道类别_左声道ID+右声道类别_右声道ID
        
        返回:
        left_category: 左声道类别
        right_category: 右声道类别
        """
        try:
            # 移除文件扩展名
            name_without_ext = os.path.splitext(filename)[0]
            
            # 检查是否以 'nn_' 或 'rn_' 开头
            if not (name_without_ext.startswith('nn_') or name_without_ext.startswith('rn_')):
                return None, None
            
            # 移除前缀
            name_without_prefix = name_without_ext[3:]
            
            # 按 '+' 分割左右声道
            if '+' not in name_without_prefix:
                return None, None
            
            left_part, right_part = name_without_prefix.split('+', 1)
            
            # 提取左声道类别（第一个下划线前的部分）
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None
            left_category = left_part[:left_underscore_pos]
            
            # 提取右声道类别（第一个下划线前的部分）
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None
            right_category = right_part[:right_underscore_pos]
            
            return left_category, right_category
            
        except Exception as e:
            print(f"解析文件名失败 {filename}: {e}")
            return None, None

    def get_main_category(self, sub_category):
        """
        根据子类别获取主类别
        
        参数:
        sub_category: 子类别名称
        
        返回:
        main_category: 主类别名称
        """
        for main_cat, sub_cats in self.main_category_groups.items():
            if sub_category in sub_cats:
                return main_cat
        return 'Unknown'

    def load_experiment_data(self, file_path):
        """
        加载实验数据
        
        参数:
        file_path: 实验数据文件路径
        
        返回:
        df: 包含实验数据的DataFrame
        """
        print(f"正在加载实验数据: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到试验详情开始的位置
        start_line = None
        for i, line in enumerate(lines):
            if "试验     音频文件" in line:
                start_line = i + 2  # 跳过分隔线
                break
        
        if start_line is None:
            print("未找到试验数据")
            return None
        
        # 解析数据行
        for i in range(start_line, len(lines)):
            line = lines[i].strip()
            if not line or "==" in line:
                break
            # 使用正则表达式解析每行数据
            match = re.match(r'(\d+)\s+(nn|rn)_([^_]+)_([^+]+)\+([^_]+)_([^\.]+)\.wav\s+(left|right)', line)
            if match:
                selection = match.group(7)  # 选择结果
                
                # 解析文件名获取左右声道类别
                left_category = match.group(3)  # 左声道类别
                right_category = match.group(5)  # 右声道类别
                
                if left_category and right_category:
                    data.append({
                        'left_category': left_category,
                        'right_category': right_category,
                        'selection': selection
                    })
        
        df = pd.DataFrame(data)
        print(f"成功加载 {len(df)} 条实验数据")
        return df

    def plot_human_selection_heatmap(self, df, output_dir):
        """
        绘制人类选择结果热力图
        
        参数:
        df: 实验数据DataFrame
        output_dir: 输出目录
        """
        print("=== 开始绘制人类选择结果热力图 ===")
        
        # 为每个子类别生成颜色（基于主类别）
        category_colors = {}
        for cat in self.target_categories:
            main_cat = self.get_main_category(cat)
            if main_cat in self.main_category_colors:
                base_color = self.main_category_colors[main_cat]
                # 根据在主类别中的位置调整亮度
                sub_cats = self.main_category_groups[main_cat]
                cat_index = sub_cats.index(cat) if cat in sub_cats else 0
                brightness = 0.4 + 0.6 * (cat_index / (len(sub_cats) - 1)) if len(sub_cats) > 1 else 0.7
                
                # 调整颜色亮度
                import matplotlib.colors as mcolors
                base_rgb = mcolors.to_rgb(base_color)
                adjusted_color = tuple(c * brightness for c in base_rgb)
                category_colors[cat] = adjusted_color
            else:
                category_colors[cat] = '#DDA0DD'
        
        # 创建配对选择矩阵
        n_categories = len(self.target_categories)
        selection_matrix = np.full((n_categories, n_categories), np.nan)
        color_matrix = np.full((n_categories, n_categories, 3), np.nan)
        
        # 统计每个配对的选择结果
        for _, row in df.iterrows():
            left_cat = row['left_category']
            right_cat = row['right_category']
            selection = row['selection']
            
            if left_cat in self.target_categories and right_cat in self.target_categories:
                left_idx = self.target_categories.index(left_cat)
                right_idx = self.target_categories.index(right_cat)
                
                # 根据选择结果确定颜色
                if selection == 'left':
                    # 选择左声道，使用左声道类别颜色
                    selected_color = category_colors[left_cat]
                else:
                    # 选择右声道，使用右声道类别颜色
                    selected_color = category_colors[right_cat]
                
                # 转换颜色为RGB值
                if isinstance(selected_color, tuple):
                    color_matrix[left_idx, right_idx] = selected_color
                else:
                    rgb_color = mcolors.to_rgb(selected_color)
                    color_matrix[left_idx, right_idx] = rgb_color
                
                # 设置标识值表示配对存在
                selection_matrix[left_idx, right_idx] = 1
        
        # 绘制热力图
        fig, ax = plt.subplots(1, 1, figsize=(20, 18))
        
        # 先绘制背景（白色）
        ax.imshow(np.ones((n_categories, n_categories)), cmap='gray', vmin=0, vmax=1, alpha=0.3)
        
        # 为每个单元格绘制颜色
        for i in range(n_categories):
            for j in range(n_categories):
                if not np.isnan(color_matrix[i, j, 0]):  # 如果有颜色数据
                    # 绘制彩色矩形
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                       facecolor=color_matrix[i, j], 
                                       edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
        
        # 设置标签
        ax.set_xticks(range(n_categories))
        ax.set_yticks(range(n_categories))
        
        # 设置X轴标签（右声道类别）
        x_labels = ax.set_xticklabels(self.target_categories, rotation=45, ha='right', fontsize=10)
        for i, label in enumerate(x_labels):
            label.set_color(category_colors[self.target_categories[i]])
            label.set_fontweight('bold')
        
        # 设置Y轴标签（左声道类别）
        y_labels = ax.set_yticklabels(self.target_categories, fontsize=10)
        for i, label in enumerate(y_labels):
            label.set_color(category_colors[self.target_categories[i]])
            label.set_fontweight('bold')
        
        # 添加选择结果标注
        for i in range(n_categories):
            for j in range(n_categories):
                if not np.isnan(color_matrix[i, j, 0]):  # 如果有配对
                    # 获取该配对的选择结果
                    for _, row in df.iterrows():
                        left_cat = row['left_category']
                        right_cat = row['right_category']
                        if (left_cat == self.target_categories[i] and 
                            right_cat == self.target_categories[j]):
                            selection = row['selection']
                            # 显示选择结果（简化版本）
                            if selection == 'left':
                                text = 'L'  # Left
                            else:
                                text = 'R'  # Right
                            ax.text(j, i, text, ha='center', va='center', 
                                   fontweight='bold', color='white', fontsize=8)
                            break
        
        ax.set_title('人类选择结果热力图\n(颜色 = 被选择的类别, L=左声道, R=右声道)',
                     fontsize=18, fontweight='bold')
        ax.set_xlabel('右声道类别', fontsize=14, fontweight='bold')
        ax.set_ylabel('左声道类别', fontsize=14, fontweight='bold')
        
        # 添加网格线
        ax.set_xticks(np.arange(-0.5, n_categories, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_categories, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 设置坐标轴范围
        ax.set_xlim(-0.5, n_categories - 0.5)
        ax.set_ylim(-0.5, n_categories - 0.5)
        
        plt.tight_layout()
        
        # 保存图片
        heatmap_path = os.path.join(output_dir, "human_selection_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"人类选择结果热力图已保存: {heatmap_path}")

    def plot_selection_statistics(self, df, output_dir):
        """
        绘制选择统计图表
        
        参数:
        df: 实验数据DataFrame
        output_dir: 输出目录
        """
        print("=== 开始绘制选择统计图表 ===")
        
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('人类选择结果统计分析', fontsize=16, fontweight='bold')
        
        # 1. 左右声道选择分布饼图
        selection_counts = df['selection'].value_counts()
        colors = ['lightcoral', 'lightgreen']
        axes[0, 0].pie(selection_counts.values, labels=selection_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('左右声道选择分布', fontsize=14, fontweight='bold')
        
        # 2. 类别选择分布
        all_categories = list(set(df['left_category'].tolist() + df['right_category'].tolist()))
        category_counts = {}
        for cat in all_categories:
            left_selected = len(df[(df['left_category'] == cat) & (df['selection'] == 'left')])
            right_selected = len(df[(df['right_category'] == cat) & (df['selection'] == 'right')])
            category_counts[cat] = left_selected + right_selected
        
        # 选择前10个最常见的类别
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        categories, counts = zip(*top_categories)
        
        axes[0, 1].bar(range(len(categories)), counts, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('各类别被选择次数', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('音频类别')
        axes[0, 1].set_ylabel('被选择次数')
        axes[0, 1].set_xticks(range(len(categories)))
        axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 按主类别的选择倾向
        df['left_main_category'] = df['left_category'].apply(self.get_main_category)
        df['right_main_category'] = df['right_category'].apply(self.get_main_category)
        
        # 计算每个主类别的选择率
        main_cat_stats = []
        for main_cat in self.main_category_groups.keys():
            left_selected = len(df[(df['left_main_category'] == main_cat) & (df['selection'] == 'left')])
            right_selected = len(df[(df['right_main_category'] == main_cat) & (df['selection'] == 'right')])
            total_left = len(df[df['left_main_category'] == main_cat])
            total_right = len(df[df['right_main_category'] == main_cat])
            
            if total_left > 0:
                left_rate = left_selected / total_left
            else:
                left_rate = 0
                
            if total_right > 0:
                right_rate = right_selected / total_right
            else:
                right_rate = 0
            
            main_cat_stats.append({
                'category': main_cat,
                'left_rate': left_rate,
                'right_rate': right_rate,
                'total_left': total_left,
                'total_right': total_right
            })
        
        main_cat_df = pd.DataFrame(main_cat_stats)
        
        x_pos = np.arange(len(main_cat_df))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, main_cat_df['left_rate'], width, 
                       label='作为左声道时的选择率', color='lightcoral', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, main_cat_df['right_rate'], width, 
                       label='作为右声道时的选择率', color='lightgreen', alpha=0.8)
        
        axes[1, 0].set_title('主类别选择倾向', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('主类别')
        axes[1, 0].set_ylabel('选择率')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(main_cat_df['category'], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 主类别选择对比
        main_cat_selection = {}
        for main_cat in self.main_category_groups.keys():
            left_as_left = len(df[(df['left_main_category'] == main_cat) & (df['selection'] == 'left')])
            right_as_right = len(df[(df['right_main_category'] == main_cat) & (df['selection'] == 'right')])
            total_appearances = len(df[(df['left_main_category'] == main_cat) | (df['right_main_category'] == main_cat)])
            if total_appearances > 0:
                selection_rate = (left_as_left + right_as_right) / total_appearances
                main_cat_selection[main_cat] = selection_rate
        
        main_cats = list(main_cat_selection.keys())
        selection_rates = list(main_cat_selection.values())
        
        bars = axes[1, 1].bar(main_cats, selection_rates, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 1].set_title('各主类别被选择率', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('主类别')
        axes[1, 1].set_ylabel('被选择率')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, rate in zip(bars, selection_rates):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        stats_path = os.path.join(output_dir, "human_selection_statistics.png")
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"选择统计图表已保存: {stats_path}")

    def save_results(self, df, output_dir):
        """
        保存分析结果
        
        参数:
        df: 实验数据DataFrame
        output_dir: 输出目录
        """
        print("=== 保存分析结果 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        detailed_path = os.path.join(output_dir, "human_selection_detailed.csv")
        df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        
        # 计算统计信息
        left_selected = len(df[df['selection'] == 'left'])
        right_selected = len(df[df['selection'] == 'right'])
        total_trials = len(df)
        
        # 选择统计
        selection_stats = df['selection'].value_counts()
        
        # 按主类别的统计
        df['left_main_category'] = df['left_category'].apply(self.get_main_category)
        df['right_main_category'] = df['right_category'].apply(self.get_main_category)
        
        main_cat_stats = []
        for main_cat in self.main_category_groups.keys():
            left_count = len(df[df['left_main_category'] == main_cat])
            right_count = len(df[df['right_main_category'] == main_cat])
            left_selected_count = len(df[(df['left_main_category'] == main_cat) & (df['selection'] == 'left')])
            right_selected_count = len(df[(df['right_main_category'] == main_cat) & (df['selection'] == 'right')])
            
            main_cat_stats.append({
                'main_category': main_cat,
                'as_left_count': left_count,
                'as_right_count': right_count,
                'left_selected': left_selected_count,
                'right_selected': right_selected_count,
                'left_selection_rate': left_selected_count / left_count if left_count > 0 else 0,
                'right_selection_rate': right_selected_count / right_count if right_count > 0 else 0
            })
        
        main_cat_df = pd.DataFrame(main_cat_stats)
        main_cat_path = os.path.join(output_dir, "main_category_statistics.csv")
        main_cat_df.to_csv(main_cat_path, index=False, encoding='utf-8-sig')
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, "human_selection_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("人类选择结果分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 总体统计\n")
            f.write(f"总试验数: {total_trials}\n")
            f.write(f"选择左声道: {left_selected} ({left_selected/total_trials*100:.1f}%)\n")
            f.write(f"选择右声道: {right_selected} ({right_selected/total_trials*100:.1f}%)\n\n")
            
            f.write("2. 选择分布统计\n")
            f.write(f"左声道选择次数: {selection_stats.get('left', 0)}\n")
            f.write(f"右声道选择次数: {selection_stats.get('right', 0)}\n")
            f.write(f"左声道选择率: {selection_stats.get('left', 0)/total_trials*100:.1f}%\n")
            f.write(f"右声道选择率: {selection_stats.get('right', 0)/total_trials*100:.1f}%\n\n")
            
            f.write("3. 主类别选择统计\n")
            for _, row in main_cat_df.iterrows():
                f.write(f"{row['main_category']}:\n")
                f.write(f"  作为左声道: {row['as_left_count']} 次, 被选择 {row['left_selected']} 次 ({row['left_selection_rate']*100:.1f}%)\n")
                f.write(f"  作为右声道: {row['as_right_count']} 次, 被选择 {row['right_selected']} 次 ({row['right_selection_rate']*100:.1f}%)\n\n")
        
        print(f"分析结果已保存:")
        print(f"  详细结果: {detailed_path}")
        print(f"  主类别统计: {main_cat_path}")
        print(f"  汇总报告: {report_path}")

    def analyze_human_selection(self, data_file, output_dir):
        """
        分析人类选择结果的主函数
        
        参数:
        data_file: 实验数据文件路径
        output_dir: 输出目录
        """
        print("=== 开始分析人类选择结果 ===")
        
        # 加载数据
        df = self.load_experiment_data(data_file)
        if df is None or len(df) == 0:
            print("无法加载实验数据")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制热力图
        self.plot_human_selection_heatmap(df, output_dir)
        
        # 绘制统计图表
        self.plot_selection_statistics(df, output_dir)
        
        # 保存结果
        self.save_results(df, output_dir)
        
        print(f"\n人类选择结果分析完成，结果保存到: {output_dir}")


def main():
    """主函数"""
    print("=== 人类选择结果热力图分析工具 ===")
    
    # 设置路径
    data_file = r"D:\D\research\audioset下载\experiment_output\experiment_results.txt"
    output_dir = r"D:\D\research\audioset下载\human_selection_analysis"
    
    # 创建分析器
    analyzer = HumanSelectionHeatmapAnalyzer()
    
    # 分析人类选择结果
    analyzer.analyze_human_selection(data_file, output_dir)


if __name__ == "__main__":
    main()
