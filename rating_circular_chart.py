#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

print("=== 评分数据环形图绘制工具 ===")
print("开始执行...")

# 设置中文字体支持（用于读取Excel中的中文列名）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_main_category(sub_category):
    """根据子类别获取主类别（与category_attraction_complete.py保持一致）"""
    main_categories = {
        'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
        'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
        'Speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
        'Music': ['Bass drum', 'Funny music', 'Sad music']
    }
    
    for main_cat, sub_cats in main_categories.items():
        if sub_category in sub_cats:
            return main_cat
    return None

def load_rating_data(excel_path):
    """加载所有sheet的评分数据并合并"""
    print(f"\n正在读取评分数据: {excel_path}")
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到文件: {excel_path}")
    
    # 读取所有sheet
    excel_file = pd.ExcelFile(excel_path)
    all_data = []
    
    for sheet_name in excel_file.sheet_names:
        print(f"  读取sheet: {sheet_name}")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 检查列名（应该是第一列是类别，后面三列是评分）
        # 根据图片，列名应该是：类别名、熟悉度、紧急度、喜爱度
        if len(df.columns) < 4:
            print(f"    警告: sheet {sheet_name} 列数不足，跳过")
            continue
        
        # 假设第一列是类别，后面三列是评分
        category_col = df.columns[0]
        
        # 尝试识别评分列（中文列名：熟悉度、紧急度、喜爱度）
        familiarity_col = None
        urgency_col = None
        liking_col = None
        
        for col in df.columns:
            col_str = str(col).strip()
            if '熟悉度' in col_str or 'familiarity' in col_str.lower():
                familiarity_col = col
            elif '紧急度' in col_str or 'urgency' in col_str.lower():
                urgency_col = col
            elif '喜爱度' in col_str or 'liking' in col_str.lower():
                liking_col = col
        
        # 如果没找到中文列名，就按位置假设（第2、3、4列）
        if familiarity_col is None and len(df.columns) > 1:
            familiarity_col = df.columns[1]
        if urgency_col is None and len(df.columns) > 2:
            urgency_col = df.columns[2]
        if liking_col is None and len(df.columns) > 3:
            liking_col = df.columns[3]
        
        # 提取数据
        for _, row in df.iterrows():
            category = str(row[category_col]).strip()
            if pd.isna(category) or category == '' or category == 'nan':
                continue
            
            try:
                # 获取评分值
                familiarity = row[familiarity_col] if familiarity_col else None
                urgency = row[urgency_col] if urgency_col else None
                liking = row[liking_col] if liking_col else None
                
                # 转换为数值
                try:
                    familiarity = float(familiarity)
                    urgency = float(urgency)
                    liking = float(liking)
                except (ValueError, TypeError):
                    continue
                
                all_data.append({
                    'subcategory': category,
                    'familiarity': familiarity,
                    'urgency': urgency,
                    'liking': liking
                })
            except Exception as e:
                print(f"    处理行时出错: {e}")
                continue
    
    if not all_data:
        raise ValueError("未能读取到任何有效数据")
    
    print(f"成功读取 {len(all_data)} 条评分记录")
    return pd.DataFrame(all_data)

def calculate_aggregated_ratings(df):
    """计算聚合后的评分（按类别计算均值）"""
    print("\n正在计算聚合评分...")
    
    # 按子类别计算均值
    subcategory_ratings = df.groupby('subcategory')[['familiarity', 'urgency', 'liking']].mean().reset_index()
    
    # 添加主类别信息
    subcategory_ratings['main_category'] = subcategory_ratings['subcategory'].apply(get_main_category)
    
    # 按主类别计算均值
    main_category_ratings = subcategory_ratings.groupby('main_category')[['familiarity', 'urgency', 'liking']].mean().reset_index()
    main_category_ratings = main_category_ratings[main_category_ratings['main_category'].notna()].copy()
    
    print(f"大类数量: {len(main_category_ratings)}")
    print(f"小类数量: {len(subcategory_ratings[subcategory_ratings['main_category'].notna()])}")
    
    return main_category_ratings, subcategory_ratings

def create_circular_barchart(df, title, figsize=(10, 10), group_colors=None, group_labels=None):
    """创建环形柱状图的通用函数
    - group_labels: 与类别等长的序列，表示每个类别所属组（用于给每个类别添加背景色块）
    - group_colors: dict 映射 组名 -> 颜色；若不提供则自动生成
    """
    # 提取类别名和三个维度的值
    category_col = df.columns[0]
    dim_cols = ['familiarity', 'urgency', 'liking']
    
    categories = df[category_col].values
    values = df[dim_cols].values.T  # 转置，每行是一个维度
    
    N = len(categories)
    if N == 0:
        print("没有数据可以绘制")
        return None, None
    
    # 计算角度
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # 计算柱状图宽度（每个维度一根柱子）
    n_dims = len(dim_cols)
    width = (2 * np.pi / N) / (n_dims + 1)  # 给每个维度之间留一点空隙
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # 定义颜色和维度名称（与category_attraction_complete.py保持一致）
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 对应 familiarity, urgency, liking
    dim_names = ['Familiarity', 'Urgency', 'Liking']
    
    # 定义半径层次（从内到外）
    inner_radius = 3.0  # 中心空白圆的半径（柱子从这个半径开始）
    bar_end_radius = inner_radius + 5.0  # 柱子结束的半径（最高评分5）
    arc_height = 0.4  # 弧形条的厚度
    outer_arc_radius = inner_radius + 5.0 - arc_height  # 调整大类弧形到评分5附近
    chart_outer_radius = inner_radius + 5.5  # 图表的最外层（为图例留空）
    
    # 设置径向轴范围（扩展到最外层）
    ax.set_ylim(0, chart_outer_radius)
    # 刻度标记：在柱子区域内（从inner_radius到inner_radius+5）
    tick_positions = [inner_radius + i for i in range(1, 6)]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Rating Scale (1-5)', fontsize=13, fontweight='bold', labelpad=20)
    ax.tick_params(axis='y', labelsize=11, width=1.6, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 绘制中心白色圆（最内层）
    theta_center = np.linspace(0, 2*np.pi, 100, endpoint=False)
    ax.bar(theta_center, [inner_radius] * 100, width=2*np.pi/100, bottom=0, 
           color='white', edgecolor='white', linewidth=0, zorder=4)
    
    # 准备分组信息
    sector_width = (2 * np.pi / N) * 0.98  # 每个类别占据的角度宽度
    if group_labels is None:
        group_labels = categories
    unique_groups = list(pd.unique(pd.Series(group_labels)))
    if group_colors is None:
        # 生成组颜色（柔和、可区分）
        default_palette = [
            '#CDE7BE', '#F6EAC2', '#CFE8F3', '#E4C1F9'  # High Ecology, Low Ecology, speech, music
        ]
        group_colors = {g: default_palette[i % len(default_palette)] for i, g in enumerate(unique_groups)}

    # 1. 最外层：绘制大类弧形颜色条（按连续的大类分组）
    # 找到每个大类对应的角度范围
    group_ranges = {}  # {group: [(start_idx, end_idx), ...]}
    current_group = None
    start_idx = 0
    
    for i, grp in enumerate(group_labels):
        if grp != current_group:
            if current_group is not None:
                if current_group not in group_ranges:
                    group_ranges[current_group] = []
                group_ranges[current_group].append((start_idx, i))
            current_group = grp
            start_idx = i
    # 处理最后一个组
    if current_group is not None:
        if current_group not in group_ranges:
            group_ranges[current_group] = []
        group_ranges[current_group].append((start_idx, N))
    
    # 绘制大类弧形颜色条（最外层）
    for group, ranges in group_ranges.items():
        group_color = group_colors.get(group, '#EEEEEE')
        for start_idx, end_idx in ranges:
            # 计算该组包含的类别数量
            if end_idx >= N:
                num_categories = N - start_idx
            else:
                num_categories = end_idx - start_idx
            
            # 计算该组占据的角度范围
            arc_width = (2 * np.pi / N) * num_categories
            arc_start = theta[start_idx]
            arc_center = arc_start + arc_width / 2
            
            # 确保中心角度在有效范围内
            if arc_center >= 2 * np.pi:
                arc_center -= 2 * np.pi
            
            # 绘制弧形条
            ax.bar(arc_center, arc_height, width=arc_width, bottom=outer_arc_radius,
                   color=group_color, alpha=0.8, edgecolor='black', linewidth=1.5, zorder=1)
    
    # 2. 类别名字层：沿着圆周弧度显示（在label_radius位置）
    # 取消类别标签文本

    # 不显示默认的xticklabels
    ax.set_xticks([])
    
    # 3. 为每个维度画柱状图（中间层）
    for i, (dim_vals, color, dim_name) in enumerate(zip(values, colors, dim_names)):
        offset = i * width
        bars = ax.bar(theta + offset, dim_vals, width=width, bottom=inner_radius, 
                     color=color, alpha=0.8, edgecolor='white', linewidth=1.2,
                     label=dim_name, zorder=3)
    
    # 4. 添加图例 - 维度（调整位置，避免与圆重合）
    legend_elements = [Patch(facecolor=color, alpha=0.8, label=dim_name) 
                      for color, dim_name in zip(colors, dim_names)]
    legend1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.23, 1.15), 
                        fontsize=13, title='Dimensions', title_fontsize=14, framealpha=0.95,
                        edgecolor='black', fancybox=True, shadow=True)

    # 4.2 大类颜色图例（右侧）
    category_legend_elements = []
    for group in unique_groups:
        if pd.isna(group):
            continue
        category_legend_elements.append(Patch(facecolor=group_colors.get(group, '#EEEEEE'), alpha=0.8, label=group))
    if category_legend_elements:
        legend2 = ax.legend(handles=category_legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5),
                             fontsize=13, title='Categories', title_fontsize=14, framealpha=0.95,
                             edgecolor='black', fancybox=True, shadow=True)
        ax.add_artist(legend1)
    
    plt.title(title, y=1.15, fontsize=16, fontweight='bold', pad=20)
    
    return fig, ax

def main():
    """主函数"""
    # 输入文件路径
    rating_excel_path = r"D:\D\research\数据采集\rating.xlsx"
    
    # 输出目录
    output_dir = r"D:\D\research\audioset下载\展示的图片"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    df_ratings = load_rating_data(rating_excel_path)
    
    # 2. 计算聚合评分
    main_category_ratings, subcategory_ratings = calculate_aggregated_ratings(df_ratings)
    
    # 3. 绘制大类环形图（图A）
    print("\n=== 绘制大类环形图（图A）===")
    
    # 确保大类顺序一致（与category_attraction_complete.py保持一致）
    category_order = ['High Ecology', 'Low Ecology', 'Speech', 'Music']
    main_category_ratings['main_category'] = pd.Categorical(
        main_category_ratings['main_category'], 
        categories=category_order, 
        ordered=True
    )
    main_category_ratings = main_category_ratings.sort_values('main_category')
    
    # 大类背景色按大类自身
    main_groups = main_category_ratings['main_category'].tolist()
    main_group_colors = {
        'High Ecology': '#CDE7BE',
        'Low Ecology': '#F6EAC2',
        'Speech': '#CFE8F3',
        'Music': '#E4C1F9'
    }
    fig1, ax1 = create_circular_barchart(
        main_category_ratings,
        'Psychological Profiles: Main Categories',
        figsize=(12, 12),
        group_colors=main_group_colors,
        group_labels=main_groups
    )
    
    if fig1:
        output_path1 = os.path.join(output_dir, 'rating_main_categories_circular.png')
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        print(f"大类环形图已保存: {output_path1}")
        plt.show()
    
    # 4. 绘制小类环形图（图B）
    print("\n=== 绘制小类环形图（图B）===")
    
    # 只保留有主类别信息的小类
    subcategory_ratings_valid = subcategory_ratings[subcategory_ratings['main_category'].notna()].copy()
    
    # 按主类别和小类名称排序，让图形更有条理
    category_order = ['High Ecology', 'Low Ecology', 'Speech', 'Music']
    subcategory_ordered = []
    for main_cat in category_order:
        subcats = subcategory_ratings_valid[subcategory_ratings_valid['main_category'] == main_cat]['subcategory'].tolist()
        subcats = sorted(subcats)  # 小类内部按字母排序
        subcategory_ordered.extend(subcats)
    
    # 按照指定顺序排序
    subcategory_ratings_valid['subcategory'] = pd.Categorical(
        subcategory_ratings_valid['subcategory'], 
        categories=subcategory_ordered, 
        ordered=True
    )
    subcategory_ratings_valid = subcategory_ratings_valid.sort_values('subcategory')
    
    # 准备小类数据（只保留需要的列）
    df_sub_for_plot = subcategory_ratings_valid[['subcategory', 'familiarity', 'urgency', 'liking']].copy()
    sub_groups = subcategory_ratings_valid['main_category'].tolist()
    sub_group_colors = {
        'High Ecology': '#CDE7BE',
        'Low Ecology': '#F6EAC2',
        'Speech': '#CFE8F3',
        'Music': '#E4C1F9'
    }
    
    fig2, ax2 = create_circular_barchart(
        df_sub_for_plot,
        'Psychological Profiles: All Subcategories',
        figsize=(16, 16),
        group_colors=sub_group_colors,
        group_labels=sub_groups
    )
    
    if fig2:
        output_path2 = os.path.join(output_dir, 'rating_subcategories_circular.png')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"小类环形图已保存: {output_path2}")
        plt.show()
    
    # 5. 保存数据为CSV
    main_csv_path = os.path.join(output_dir, 'rating_main_categories.csv')
    main_category_ratings.to_csv(main_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n大类数据已保存: {main_csv_path}")
    
    sub_csv_path = os.path.join(output_dir, 'rating_subcategories.csv')
    subcategory_ratings_valid[['subcategory', 'main_category', 'familiarity', 'urgency', 'liking']].to_csv(
        sub_csv_path, index=False, encoding='utf-8-sig')
    print(f"小类数据已保存: {sub_csv_path}")
    
    # 6. 保存数据为TXT格式
    print("\n=== 保存评分数据为TXT格式 ===")
    
    # 保存大类评分
    main_txt_path = os.path.join(output_dir, 'rating_main_categories.txt')
    with open(main_txt_path, 'w', encoding='utf-8') as f:
        f.write("=== 大类评分数据 ===\n\n")
        f.write(f"{'Category':<20} {'Familiarity':<15} {'Urgency':<15} {'Liking':<15}\n")
        f.write("-" * 65 + "\n")
        for _, row in main_category_ratings.iterrows():
            f.write(f"{row['main_category']:<20} {row['familiarity']:<15.3f} {row['urgency']:<15.3f} {row['liking']:<15.3f}\n")
    print(f"大类评分TXT已保存: {main_txt_path}")
    
    # 保存小类评分
    sub_txt_path = os.path.join(output_dir, 'rating_subcategories.txt')
    subcategory_output = subcategory_ratings_valid[['subcategory', 'main_category', 'familiarity', 'urgency', 'liking']].copy()
    with open(sub_txt_path, 'w', encoding='utf-8') as f:
        f.write("=== 小类评分数据 ===\n\n")
        f.write(f"{'Subcategory':<35} {'Main Category':<20} {'Familiarity':<15} {'Urgency':<15} {'Liking':<15}\n")
        f.write("-" * 100 + "\n")
        for _, row in subcategory_output.iterrows():
            f.write(f"{row['subcategory']:<35} {row['main_category']:<20} {row['familiarity']:<15.3f} {row['urgency']:<15.3f} {row['liking']:<15.3f}\n")
    print(f"小类评分TXT已保存: {sub_txt_path}")
    
    print(f"\n=== 完成！所有结果保存到: {output_dir} ===")

if __name__ == "__main__":
    main()

