#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""将评分结果绘制为普通柱状图（主类与子类）。"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== 评分数据柱状图绘制工具 ===")
print("开始执行...")

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_main_category(sub_category):
    """根据子类别获取主类别（与 rating_circular_chart.py 一致）。"""
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
    """加载评分数据。"""
    print(f"\n正在读取评分数据: {excel_path}")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到文件: {excel_path}")

    excel_file = pd.ExcelFile(excel_path)
    all_data = []

    for sheet_name in excel_file.sheet_names:
        print(f"  读取sheet: {sheet_name}")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        if len(df.columns) < 4:
            print(f"    警告: sheet {sheet_name} 列数不足，跳过")
            continue

        category_col = df.columns[0]

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

        if familiarity_col is None and len(df.columns) > 1:
            familiarity_col = df.columns[1]
        if urgency_col is None and len(df.columns) > 2:
            urgency_col = df.columns[2]
        if liking_col is None and len(df.columns) > 3:
            liking_col = df.columns[3]

        for _, row in df.iterrows():
            category = str(row[category_col]).strip()
            if pd.isna(category) or category in ('', 'nan'):
                continue

            try:
                familiarity = float(row[familiarity_col])
                urgency = float(row[urgency_col])
                liking = float(row[liking_col])
            except (ValueError, TypeError):
                continue

            all_data.append({
                'subcategory': category,
                'familiarity': familiarity,
                'urgency': urgency,
                'liking': liking
            })

    if not all_data:
        raise ValueError("未能读取到任何有效数据")

    print(f"成功读取 {len(all_data)} 条评分记录")
    return pd.DataFrame(all_data)


def calculate_aggregated_ratings(df):
    """按主类、子类计算均值。"""
    print("\n正在计算聚合评分...")

    subcategory_ratings = df.groupby('subcategory')[['familiarity', 'urgency', 'liking']].mean().reset_index()
    subcategory_ratings['main_category'] = subcategory_ratings['subcategory'].apply(get_main_category)

    main_category_ratings = subcategory_ratings.groupby('main_category')[['familiarity', 'urgency', 'liking']].mean().reset_index()
    main_category_ratings = main_category_ratings[main_category_ratings['main_category'].notna()].copy()

    print(f"大类数量: {len(main_category_ratings)}")
    print(f"小类数量: {len(subcategory_ratings[subcategory_ratings['main_category'].notna()])}")

    return main_category_ratings, subcategory_ratings


def plot_grouped_bar_chart(
    df,
    category_col,
    title,
    output_path,
    figsize=(15.06 / 2.54, 8.18 / 2.54),
    order=None,
    colors=None,
    label_transform=None,
    x_rotation=None,
    x_fontsize=18,
):
    """绘制普通分组柱状图。"""
    dim_cols = ['familiarity', 'urgency', 'liking']
    dim_names = ['Familiarity', 'Urgency', 'Liking']

    if order is not None:
        df = df.copy()
        df[category_col] = pd.Categorical(df[category_col], categories=order, ordered=True)
        df = df.sort_values(category_col)

    categories = df[category_col].astype(str).tolist()
    display_categories = [label_transform(cat) if label_transform else cat for cat in categories]
    values = df[dim_cols].values

    x = np.arange(len(categories))
    bar_width = 0.22

    if colors is None:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    fig, ax = plt.subplots(figsize=figsize)

    for i, (dim_name, color) in enumerate(zip(dim_names, colors)):
        ax.bar(x + (i - 1) * bar_width, values[:, i], width=bar_width, label=dim_name, color=color, alpha=0.85)

    ax.set_xticks(x)
    if x_rotation is None:
        rotation_angle = 30 if len(display_categories) > 6 else 0
    else:
        rotation_angle = x_rotation
    ax.set_xticklabels(display_categories, rotation=rotation_angle, ha='right' if rotation_angle else 'center', fontsize=x_fontsize, fontweight='bold')
    ax.set_ylabel('Rating (1-5)', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylim(0, 5.2)

    ax.tick_params(axis='y', labelsize=18, width=1.6, length=6)
    ax.tick_params(axis='x', width=1.6, length=6)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.legend(loc='upper right', fontsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"柱状图已保存: {output_path}")
    plt.show()


def main():
    # 使用你提供的新版评分文件（新增了一位受试者后需要重算）
    rating_excel_path = r"A:\ratings.xlsx"
    output_dir = r"D:\D\research\audioset下载\展示的图片"
    os.makedirs(output_dir, exist_ok=True)

    df_ratings = load_rating_data(rating_excel_path)
    main_category_ratings, subcategory_ratings = calculate_aggregated_ratings(df_ratings)

    category_order = ['High Ecology', 'Low Ecology', 'Speech', 'Music']
    main_category_ratings['main_category'] = pd.Categorical(
        main_category_ratings['main_category'], categories=category_order, ordered=True
    )
    main_category_ratings = main_category_ratings.sort_values('main_category')

    plot_grouped_bar_chart(
        main_category_ratings,
        category_col='main_category',
        title='Psychological Profiles: Main Categories (Bar Chart)',
        output_path=os.path.join(output_dir, 'rating_main_categories_bar.png'),
        figsize=(15.06 / 2.54, 8.18 / 2.54),
        order=category_order
    )

    subcategory_ratings_valid = subcategory_ratings[subcategory_ratings['main_category'].notna()].copy()

    subcategory_ratings_valid['main_category'] = pd.Categorical(
        subcategory_ratings_valid['main_category'], categories=category_order, ordered=True
    )
    subcategory_ratings_valid = subcategory_ratings_valid.sort_values(['main_category', 'subcategory'])

    subcategories = subcategory_ratings_valid['subcategory'].tolist()

    plot_grouped_bar_chart(
        subcategory_ratings_valid[['subcategory', 'familiarity', 'urgency', 'liking']],
        category_col='subcategory',
        title='Psychological Profiles: Subcategories (Bar Chart)',
        output_path=os.path.join(output_dir, 'rating_subcategories_bar.png'),
        figsize=(15.06 / 2.54, 8.18 / 2.54),
        order=subcategories,
        label_transform=lambda name: str(name).split()[0] if str(name).strip() else name,
        x_rotation=0,
        x_fontsize=18
    )


if __name__ == "__main__":
    main()


