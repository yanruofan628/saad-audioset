#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 coefficients.csv 绘制回归系数棒棒糖图；y 轴标签去掉 d_ 前缀与 _mean 后缀。"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def format_feature_display(name: str) -> str:
    s = str(name).strip()
    if s.startswith("d_"):
        s = s[2:]
    if s.endswith("_mean"):
        s = s[: -len("_mean")]
    return s


def plot_lollipop_from_csv(csv_path: str, output_path: str | None = None) -> str:
    df = pd.read_csv(csv_path)
    required = {"feature", "coefficient", "pvalue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")

    df = df.copy()
    df["feature_display"] = df["feature"].map(format_feature_display)
    df = df.sort_values("coefficient").reset_index(drop=True)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    cm_to_inch = 1 / 2.54
    fig_w = 15.19 * cm_to_inch
    fig_h = 8.41 * cm_to_inch
    font_size = 18

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    y_positions = np.arange(len(df))

    sig_color = "#9555A3"
    ns_color = "#B8B8B8"
    stem_color = "#D8D8D8"

    for i, (_, row) in enumerate(df.iterrows()):
        is_sig = row["pvalue"] < 0.05
        point_color = sig_color if is_sig else ns_color
        coef = float(row["coefficient"])
        ax.plot([0, coef], [i, i], color=stem_color, linewidth=1.8, alpha=0.95, zorder=1)
        ax.scatter(coef, i, s=72, color=point_color, edgecolor="white", linewidth=0.8, zorder=3)
        p = float(row["pvalue"])
        sig_mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig_mark:
            ax.text(coef, i, f" {sig_mark}", ha="left", va="center", fontsize=font_size, color=point_color)

    ax.axvline(x=0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.9, zorder=0)
    ax.set_yticks(y_positions)
    ytick_font_size = 8.5
    ax.set_yticklabels(df["feature_display"], fontsize=ytick_font_size)
    ax.set_xlabel("Coefficient", fontsize=font_size)
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=ytick_font_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(ytick_font_size)
        tick.set_fontname("Arial")
    ax.grid(True, alpha=0.22, axis="x", linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if output_path is None:
        base, _ = os.path.splitext(csv_path)
        output_path = base + "_lollipop.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = os.path.splitext(output_path)[0] + ".svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"系数棒棒糖图已保存: {output_path}")
    print(f"系数棒棒糖图SVG已保存: {svg_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="从 coefficients.csv 画系数棒棒糖图（标签去掉 d_ 与 _mean）")
    parser.add_argument(
        "csv",
        nargs="?",
        default=r"d:\PycharmProjects\Attention_switch\pair_aggregate_ols_out\coefficients.csv",
        help="coefficients.csv 路径",
    )
    parser.add_argument("-o", "--output", default=None, help="输出 PNG 路径（默认同目录 *_lollipop.png）")
    args = parser.parse_args()
    plot_lollipop_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
