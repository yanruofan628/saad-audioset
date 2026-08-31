import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


def _find_dim_columns(df: pd.DataFrame):
    """根据列名自动识别三维度列。若识别失败，按位置回退。"""
    familiarity_col = None
    urgency_col = None
    liking_col = None

    for col in df.columns:
        col_str = str(col).strip()
        if "熟悉度" in col_str or "familiarity" in col_str.lower():
            familiarity_col = col
        elif "紧急度" in col_str or "urgency" in col_str.lower():
            urgency_col = col
        elif "喜爱度" in col_str or "liking" in col_str.lower():
            liking_col = col

    # 回退：第 2/3/4 列
    if familiarity_col is None and len(df.columns) > 1:
        familiarity_col = df.columns[1]
    if urgency_col is None and len(df.columns) > 2:
        urgency_col = df.columns[2]
    if liking_col is None and len(df.columns) > 3:
        liking_col = df.columns[3]

    if familiarity_col is None or urgency_col is None or liking_col is None:
        raise ValueError(f"无法识别评分列：{list(df.columns)}")

    return familiarity_col, urgency_col, liking_col


def load_rating_records(excel_path: str) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到文件: {excel_path}")

    excel_file = pd.ExcelFile(excel_path)
    all_data: List[Dict[str, object]] = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        if df.shape[1] < 4:
            # 列数不足就跳过
            continue

        category_col = df.columns[0]
        familiarity_col, urgency_col, liking_col = _find_dim_columns(df)

        for _, row in df.iterrows():
            category = row.get(category_col, None)
            if pd.isna(category):
                continue
            category = str(category).strip()
            if category == "" or category.lower() == "nan":
                continue

            try:
                familiarity = float(row.get(familiarity_col))
                urgency = float(row.get(urgency_col))
                liking = float(row.get(liking_col))
            except (TypeError, ValueError):
                continue

            all_data.append(
                {
                    "subcategory": category,
                    "familiarity": familiarity,
                    "urgency": urgency,
                    "liking": liking,
                }
            )

    if not all_data:
        raise ValueError("未能读取到任何有效评分记录（请检查Excel格式/列名）。")

    return pd.DataFrame(all_data)


def main():
    excel_path = r"A:\ratings.xlsx"
    df = load_rating_records(excel_path)

    # 每个子类别：均值 + 标准差（跨所有sheet/所有人）
    grouped = df.groupby("subcategory")[["familiarity", "urgency", "liking"]].agg(["mean", "std"])

    # 将多层列摊平成普通列
    out = grouped.copy()
    out.columns = [f"{c}_{stat}" for c, stat in out.columns.to_flat_index()]
    out = out.reset_index()

    # 统一列顺序
    cols = [
        "subcategory",
        "familiarity_mean",
        "familiarity_std",
        "urgency_mean",
        "urgency_std",
        "liking_mean",
        "liking_std",
    ]
    for c in cols:
        if c not in out.columns:
            raise RuntimeError(f"缺少输出列: {c}")
    out = out[cols]

    # 为了“贴出来”更清晰：四舍五入到 3 位小数
    for c in out.columns:
        if c != "subcategory":
            out[c] = out[c].astype(float).round(3)

    # 输出为 Markdown 表格
    print(out.to_markdown(index=False))


if __name__ == "__main__":
    main()

