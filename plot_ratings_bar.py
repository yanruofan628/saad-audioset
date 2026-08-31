"""Paper figures: rating bar chart and GLM odds-ratio forest plot (95% CI)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GLM_CSV = _SCRIPT_DIR / "pair_aggregate_ols_out" / "glm_coefficients.csv"
FALLBACK_GLM_CSV = Path(
    r"d:\PycharmProjects\Attention_switch\pair_aggregate_ols_out\glm_coefficients.csv"
)

# OR forest plot style (fonts/colors aligned with coefficient lollipop figures)
OR_SIG_COLOR = "#9555A3"
OR_NS_COLOR = "#B8B8B8"
OR_REF_COLOR = "#666666"
OR_FIG_W_CM = 15.19
OR_FIG_H_CM = 8.41
OR_FONT_SIZE = 18
OR_YTICK_FONT_SIZE = 8.5
OR_DPI = 300

# Full names as in stats table -> one row: fam_m, fam_sd, urg_m, urg_sd, lik_m, lik_sd
_STATS = {
    "Baby cry, infant cry": [2.957, 1.022, 3.478, 0.846, 0.826, 0.984],
    "Bass drum": [2.652, 0.885, 0.652, 0.714, 2.913, 0.668],
    "Computer keyboard": [3.783, 0.518, 1.261, 1.010, 1.522, 0.898],
    "Female speech, woman speaking": [3.609, 0.656, 1.435, 0.896, 1.826, 0.778],
    "Helicopter": [2.000, 1.128, 2.304, 1.105, 0.870, 0.815],
    "Male speech, man speaking": [3.652, 0.647, 1.391, 0.891, 1.826, 0.717],
    "Sad music": [3.000, 0.798, 0.261, 0.619, 3.435, 0.662],
    "Telephone bell ringing": [3.174, 1.029, 3.348, 0.647, 0.696, 0.635],
}

X_LABELS = [
    "Sad music",
    "Bass Drum",
    "Female speech",
    "Male speech",
    "Telephone bell",
    "Baby Cry",
    "Keyboard",
    "Helicopter",
]
_KEYS = [
    "Sad music",
    "Bass drum",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Telephone bell ringing",
    "Baby cry, infant cry",
    "Computer keyboard",
    "Helicopter",
]

DATA = np.array([_STATS[k] for k in _KEYS])

LABELS = ["Familiarity", "Urgency", "Liking"]
METRIC_COLORS = ["#92D5D0", "#9594C0", "#D16552"]

_GROUP_INTRA = 1.0
_GROUP_INTER = 0.9
_HELICOPTER_X_NUDGE = 0.22


def prettify_term(term: str) -> str:
    t = str(term).strip()
    if t == "const":
        return "Intercept"
    if t.startswith("d_") and t.endswith("_mean"):
        return t[2:-5].replace("_", " ")
    return t.replace("_", " ")


def _apply_or_plot_rcparams() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def _p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_or_forest(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    *,
    include_intercept: bool = False,
) -> str:
    """从 glm_coefficients.csv 绘制 OR 森林图（95% CI，参考线 x=1，无标题）。"""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"term", "or", "or_ci_lower", "or_ci_upper", "pvalue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")

    d = df.copy()
    if not include_intercept:
        d = d[d["term"].astype(str) != "const"].copy()
    if d.empty:
        raise ValueError("无可用行：检查 CSV 或 include_intercept=True")

    d["label"] = d["term"].map(prettify_term)
    d = d.sort_values("or", ascending=True).reset_index(drop=True)

    est = d["or"].to_numpy(dtype=float)
    lo = d["or_ci_lower"].to_numpy(dtype=float)
    hi = d["or_ci_upper"].to_numpy(dtype=float)
    labels = d["label"].tolist()
    pvals = d["pvalue"].to_numpy(dtype=float)

    _apply_or_plot_rcparams()
    cm_to_inch = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(OR_FIG_W_CM * cm_to_inch, OR_FIG_H_CM * cm_to_inch)
    )

    y = np.arange(len(d))
    ref = 1.0
    xerr_lo = est - lo
    xerr_hi = hi - est

    for i in range(len(d)):
        color = OR_SIG_COLOR if pvals[i] < 0.05 else OR_NS_COLOR
        ax.errorbar(
            est[i],
            y[i],
            xerr=[[xerr_lo[i]], [xerr_hi[i]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
            linewidth=1.6,
            zorder=2,
        )

    ax.axvline(
        x=ref,
        color=OR_REF_COLOR,
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=OR_YTICK_FONT_SIZE)
    ax.set_xlabel("Odds ratio with 95% CI", fontsize=OR_FONT_SIZE)
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=OR_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=OR_YTICK_FONT_SIZE)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(OR_YTICK_FONT_SIZE)
        tick.set_fontname("Arial")
    ax.grid(True, alpha=0.22, axis="x", linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    span = float(np.nanmax(hi) - np.nanmin(lo))
    pad = 0.035 * span if span > 1e-12 else 0.05
    for i, (_, row) in enumerate(d.iterrows()):
        st = _p_to_star(float(row["pvalue"]))
        if st:
            color = OR_SIG_COLOR if float(row["pvalue"]) < 0.05 else OR_NS_COLOR
            ax.text(hi[i] + pad, y[i], st, va="center", fontsize=OR_FONT_SIZE, color=color)

    ax.margins(x=0.12)
    plt.tight_layout()

    if output_path is None:
        base, _ = os.path.splitext(str(csv_path))
        output_path = base + "_or_forest.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=OR_DPI, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"OR 森林图已保存: {output_path}")
    print(f"OR 森林图 SVG 已保存: {svg_path}")
    return str(output_path)


def _format_xtick_label(label: str) -> str:
    parts = label.split()
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return label


def plot_ratings_bar() -> None:
    """Grouped bar chart: Familiarity / Urgency / Liking by sound category."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    n_cat = len(X_LABELS)
    n_series = 3
    n_groups = n_cat // 2
    stride = _GROUP_INTRA + _GROUP_INTER
    x = np.empty(n_cat)
    for g in range(n_groups):
        base = g * stride
        x[2 * g] = base
        x[2 * g + 1] = base + _GROUP_INTRA
    x[-1] += _HELICOPTER_X_NUDGE
    group_dividers = [
        0.5 * ((g * stride + _GROUP_INTRA) + (g + 1) * stride) for g in range(n_groups - 1)
    ]
    width = 0.25

    cm_to_inch = 1 / 2.54
    fig_w = 15.64 * (n_cat / 6.0) * cm_to_inch
    fig_h = 8.18 * cm_to_inch
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for i in range(n_series):
        means = DATA[:, i * 2]
        stds = DATA[:, i * 2 + 1]
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=LABELS[i],
            color=METRIC_COLORS[i],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 1, "capthick": 1},
        )

    for xv in group_dividers:
        ax.axvline(
            xv,
            ymin=0,
            ymax=1,
            color="#888888",
            linewidth=1.2,
            linestyle="-",
            clip_on=False,
            zorder=0,
        )

    ax.set_ylabel("Rating", fontsize=16)
    ax.set_xticks(x)
    x_labels_wrapped = [_format_xtick_label(s) for s in X_LABELS]
    ax.set_xticklabels(x_labels_wrapped, rotation=0, ha="center", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    pad = 0.55
    ax.set_xlim(x.min() - pad, x.max() + pad)

    fig.tight_layout()
    fig.canvas.draw()
    ax_bb = ax.get_position()
    fig_w_in = fig.get_figwidth()
    fig_h_in = fig.get_figheight()
    dx_fig = (7.2 / 2.54) / fig_w_in
    dy_fig = (0.1 / 2.54) / fig_h_in
    fig_x_anchor = ax_bb.x0 + 0.519 * ax_bb.width
    fig_x_new = fig_x_anchor + dx_fig
    legend_x_axes = (fig_x_new - ax_bb.x0) / ax_bb.width
    fig_y_anchor = ax_bb.y0 + 1.017 * ax_bb.height
    fig_y_new = fig_y_anchor - dy_fig
    legend_y_axes = (fig_y_new - ax_bb.y0) / ax_bb.height

    ax.legend(
        frameon=True,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(legend_x_axes, legend_y_axes),
    )

    out = "ratings_bar_chart.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    out_svg = "ratings_bar_chart.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print(f"Saved {out_svg}")


def _resolve_glm_csv(path_arg: str) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.is_absolute():
            p = _SCRIPT_DIR / p
        return p
    if DEFAULT_GLM_CSV.is_file():
        return DEFAULT_GLM_CSV
    return FALLBACK_GLM_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="评分柱状图 / GLM OR 森林图（95% CI）")
    parser.add_argument(
        "--chart",
        choices=("or", "bar"),
        default="or",
        help="or=GLM 比值比森林图（默认）；bar=评分柱状图",
    )
    parser.add_argument(
        "--glm-csv",
        default="",
        help="glm_coefficients.csv 路径（--chart or 时使用）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="输出 PNG 路径（OR 图默认与 CSV 同目录 *_or_forest.png）",
    )
    parser.add_argument(
        "--include-intercept",
        action="store_true",
        help="OR 图中包含 const（默认排除）",
    )
    args = parser.parse_args()

    if args.chart == "bar":
        plot_ratings_bar()
        return

    csv_path = _resolve_glm_csv(args.glm_csv)
    if not csv_path.is_file():
        raise SystemExit(f"找不到: {csv_path}")

    out = args.output or None
    plot_or_forest(
        csv_path,
        out,
        include_intercept=args.include_intercept,
    )


if __name__ == "__main__":
    main()
