"""Visualization utilities for diagnostics and paper figures."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotStyle:
    """Global style settings for all figures."""

    font_family: str = "Arial"
    font_weight: str = "normal"
    font_size: float = 12.0
    title_size: float = 15.0
    label_size: float = 12.0
    tick_size: float = 10.5
    legend_size: float = 10.0

    dpi: int = 260
    line_width: float = 1.4
    marker_size: float = 18.0
    grid_alpha: float = 0.28
    grid_linestyle: str = "--"

    background_color: str = "#ffffff"
    axes_facecolor: str = "#ffffff"
    edge_color: str = "#111827"

    known_color: str = "#1f77b4"
    unknown_color: str = "#d62728"
    highlight_color: str = "#ff7f0e"
    collapse_colormap: str = "tab20"
    tsne_unknown_color: str = "#d62728"

    bubble_alpha: float = 0.72
    bubble_size_min: float = 140.0
    bubble_size_max: float = 2400.0
    collapse_active_threshold: float = 0.02


def _resolve_style(style: Optional[PlotStyle]) -> PlotStyle:
    return style if style is not None else PlotStyle()


def _apply_plot_style(style: PlotStyle) -> None:
    mpl.rcParams.update(
        {
            "font.family": [style.font_family, "DejaVu Sans", "sans-serif"],
            "font.size": style.font_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "axes.labelweight": style.font_weight,
            "axes.facecolor": style.axes_facecolor,
            "axes.edgecolor": style.edge_color,
            "figure.facecolor": style.background_color,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "legend.fontsize": style.legend_size,
            "grid.alpha": style.grid_alpha,
        }
    )


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict


def plot_dynamic_collapse(
        usage_history: np.ndarray,
        out_path: Path,
        style=None,  # 兼容你的 PlotStyle
        title: str = "Expert Routing Dynamics / Expert Consolidation",
        switch_epoch: int = 5,
        custom_colors: Dict[int, str] = None,  # 新增：手动设置专家颜色，例如 {2: '#ff0000', 5: 'blue'}
        custom_linewidths: Dict[int, float] = None,  # 新增：手动设置专家线宽，例如 {2: 3.0, 5: 1.5}
        active_line_color: str = "#9f9f9f",  # 新增：阶梯图颜色 (统一为黑色调)
        active_line_lw: float = 1.0  # 新增：阶梯图线宽
) -> None:
    """
    Plot expert usage trajectories across epochs with manual customization and IEEE TII standard formatting.
    """
    if usage_history.size == 0:
        return

    # 5. 全局强制设置 Arial 字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'

    usage = np.asarray(usage_history, dtype=np.float64)
    if usage.ndim != 2:
        return

    num_epochs = usage.shape[0]
    num_experts = usage.shape[1]
    epochs = np.arange(1, num_epochs + 1)

    if switch_epoch is None:
        switch_epoch = max(1, num_epochs // 2)

    # 初始化自定义字典
    if custom_colors is None: custom_colors = {}
    if custom_linewidths is None: custom_linewidths = {}

    fig, ax = plt.subplots(figsize=(9,5))

    # 删掉这两行：
    # cmap = plt.get_cmap("tab20")
    # default_colors = cmap(np.linspace(0.05, 0.95, num_experts))

    # 替换为你自己的颜色系列 (假设你需要 15 个颜色)：
    # 这里我提供一套顶级期刊常用的一种高级柔和/色盲友好配色卡作为例子
    my_color_palette = [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
        '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
        '#FF9F43', '#00CFD8', '#7367F0', '#28C76F', '#EA5455'
    ]

    # 添加背景色带
    ax.axvspan(1, switch_epoch, color='#eef2f5', alpha=0.7, zorder=0)
    # ax.axvspan(switch_epoch, num_epochs, color='#f5fcf6', alpha=0.7, zorder=0)

    # 1. 解决文字遮挡问题：
    # 采用 axes 相对坐标 (transform=ax.transAxes)，加入白色半透明底色框 (bbox)，即使压到线也能看清字
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)

    ax.text(0.1, 0.95, 'Soft Routing',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, color='#333333', bbox=bbox_props, zorder=5)

    ax.text(0.6, 0.95, 'Expert Consolidation',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, color='#333333', bbox=bbox_props, zorder=5)

    # 确定存活专家（用于图例和默认高亮）
    final_probs = usage[-1, :]
    threshold = 0.05
    survivor_handles = []

    for idx in range(num_experts):
        is_survivor = final_probs[idx] > threshold

        # 3. 手动颜色与线宽控制：优先读取字典，如果没有则使用默认逻辑
        # color = custom_colors.get(idx, default_colors[idx] if is_survivor else '#b0b0b0')
        # 如果外部没指定颜色，就在你自己的色卡里按顺序挑一个
        color = custom_colors.get(idx, my_color_palette[idx % len(my_color_palette)] if is_survivor else '#b0b0b0')
        lw = custom_linewidths.get(idx, 2.5 if is_survivor else 1.0)

        alpha = 0.95 if is_survivor else 0.3
        zorder = 4 if is_survivor else 2

        line, = ax.plot(
            epochs,
            usage[:, idx],
            linewidth=lw,
            color=color,
            alpha=alpha,
            marker="o" if (num_epochs <= 30 and is_survivor) else None,
            markersize=5,
            zorder=zorder,
            label=f"Expert {idx + 1}" if is_survivor else ""
        )
        if is_survivor:
            survivor_handles.append(line)

    ax.set_xlim(1, max(2, num_epochs))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training Epochs",  fontsize=14)
    ax.set_ylabel("Routing Probability",  fontsize=14)
    # ax.set_title(title, pad=15, fontweight='bold', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 右侧双 Y 轴 (活跃专家数)
    active_k = (usage > threshold).sum(axis=1)
    ax2 = ax.twinx()
    line2, = ax2.step(
        epochs,
        active_k,
        where='post',
        color=active_line_color,
        linestyle="--",
        linewidth=active_line_lw,
        label="Active Experts Count",
        zorder=3
    )

    # 4. 统一右侧坐标轴颜色为黑色
    ax2.set_ylabel("Number of Active Experts",  fontsize=14, color='black')
    ax2.tick_params(axis='y', colors='black', labelsize=12)
    ax2.spines['right'].set_color('black')

    # 将右侧 Y 轴的上限调高 15%，防止红虚线顶到天花板导致和文字重叠
    max_k = max(int(num_experts), 1)
    ax2.set_ylim(0, max_k * 1.15)

    # 2. 调整图例位置：放到左中偏下，避开曲线集中的右侧
    all_handles = survivor_handles + [line2]
    all_labels = [h.get_label() for h in survivor_handles] + [line2.get_label()]

    ax.legend(
        all_handles, all_labels,
        loc="center left",  # 核心修改：图例放左侧
        bbox_to_anchor=(0.02, 0.4),  # 微调左侧的锚点，让它正好卡在前半段平坦区域的上方
        frameon=True,
        facecolor='white',
        edgecolor='#9f9f9f',
        framealpha=0.9,
        fontsize=10
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _gaussian_kde(values: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n = values.size
    if n < 2:
        return np.zeros_like(grid)

    std = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    robust = iqr / 1.34 if iqr > 0 else 0.0
    sigma = min(v for v in [std, robust] if v > 0) if (std > 0 and robust > 0) else max(std, robust, 1e-6)
    if bandwidth is None or bandwidth <= 0:
        bandwidth = 1.06 * sigma * (n ** (-1.0 / 5.0))
    bandwidth = max(float(bandwidth), 1e-6)

    diff = (grid[:, None] - values[None, :]) / bandwidth
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth * n)
    return norm * np.exp(-0.5 * diff * diff).sum(axis=1)


def plot_open_set_hist(
    scores: np.ndarray,
    is_unknown: np.ndarray,
    out_path: Path,
    style: Optional[PlotStyle] = None,
    bins: int = 45,
    kde: bool = True,
    title: str = "OOD Score Histograms",
    score_label: str = "OOD Score",
) -> None:
    """
    Plot histogram/density of OOD scores for known and unknown samples.
    """
    if len(scores) == 0:
        return
    style = _resolve_style(style)
    _apply_plot_style(style)

    scores = np.asarray(scores, dtype=np.float64)
    flags = np.asarray(is_unknown, dtype=np.int64)
    known = scores[flags == 0]
    unknown = scores[flags == 1]
    if known.size == 0 and unknown.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    if known.size > 0:
        ax.hist(
            known,
            bins=bins,
            density=True,
            alpha=0.52,
            color=style.known_color,
            edgecolor="white",
            linewidth=0.6,
            label=f"Known (n={known.size})",
        )
    if unknown.size > 0:
        ax.hist(
            unknown,
            bins=bins,
            density=True,
            alpha=0.52,
            color=style.unknown_color,
            edgecolor="white",
            linewidth=0.6,
            label=f"Unknown (n={unknown.size})",
        )

    if kde and (known.size > 1 or unknown.size > 1):
        x_min = float(np.min(scores))
        x_max = float(np.max(scores))
        if np.isclose(x_min, x_max):
            x_min -= 1e-3
            x_max += 1e-3
        grid = np.linspace(x_min, x_max, 240)
        if known.size > 1:
            ax.plot(grid, _gaussian_kde(known, grid), color=style.known_color, linewidth=style.line_width + 0.4)
        if unknown.size > 1:
            ax.plot(grid, _gaussian_kde(unknown, grid), color=style.unknown_color, linewidth=style.line_width + 0.4)

    ax.set_xlabel(score_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, linestyle=style.grid_linestyle, alpha=style.grid_alpha)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def save_history_csv(history: List[Dict[str, float]], out_path: Path) -> None:
    """Save training history (list[dict]) into a CSV file."""
    if not history:
        return
    fields: List[str] = []
    for row in history:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_loss_curves(
    history: List[Dict[str, float]],
    out_path: Path,
    keys: Optional[Sequence[str]] = None,
    title: str = "Training Curves",
    style: Optional[PlotStyle] = None,
) -> None:
    """Plot curves from metrics history."""
    if not history or "epoch" not in history[0]:
        return
    style = _resolve_style(style)
    _apply_plot_style(style)

    epochs = [int(r["epoch"]) for r in history]
    if keys is None:
        keys = [k for k in history[0].keys() if k.startswith("train_")]
    valid_keys = [k for k in keys if k in history[0]]
    if not valid_keys:
        return

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for key in valid_keys:
        vals = np.array([float(r[key]) for r in history], dtype=np.float64)
        ax.plot(
            epochs,
            vals,
            marker="o" if len(epochs) <= 60 else None,
            markersize=max(style.marker_size * 0.18, 2.0),
            linewidth=style.line_width,
            label=key,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, linestyle=style.grid_linestyle, alpha=style.grid_alpha)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix using NumPy only."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = int(t)
        pi = int(p)
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            cm[ti, pi] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    style: Optional[PlotStyle] = None,
) -> None:
    """Plot confusion matrix as a heatmap."""
    if cm.size == 0:
        return
    style = _resolve_style(style)
    _apply_plot_style(style)

    data = cm.astype(np.float64)
    if normalize:
        row_sum = data.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        data = data / row_sum

    n = data.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    vmax = float(data.max()) if data.size else 1.0
    thresh = vmax * 0.62
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            txt = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=max(style.tick_size - 2.0, 7.0), color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def _sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def plot_tsne_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    class_names: Optional[Sequence[str]] = None,
    max_points: int = 3000,
    seed: int = 42,
    title: str = "t-SNE Feature Embeddings",
    style: Optional[PlotStyle] = None,
    point_size: float = 11.0,
    alpha: float = 0.78,
) -> None:
    """Plot 2-D t-SNE embeddings."""
    if features.size == 0 or len(features) != len(labels):
        return
    style = _resolve_style(style)
    _apply_plot_style(style)

    idx = _sample_indices(len(features), max_points=max_points, seed=seed)
    x = np.asarray(features[idx], dtype=np.float64)
    y = np.asarray(labels[idx], dtype=np.int64)
    if len(x) < 3:
        return

    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise RuntimeError("plot_tsne_embeddings requires scikit-learn. Please install sklearn first.") from e

    if x.shape[1] > 50:
        centered = x - x.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        x = u[:, :50] * s[:50]

    base_perp = max(5.0, min(30.0, float(len(x) // 35)))
    perplexity = min(base_perp, max(1.0, float(len(x) - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    emb = tsne.fit_transform(x)

    uniq = np.unique(y)
    known_labels = [int(v) for v in uniq.tolist() if int(v) >= 0]
    cmap = plt.get_cmap("tab20", max(len(known_labels), 1))
    color_map = {cls: cmap(i % cmap.N) for i, cls in enumerate(known_labels)}

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for cls in uniq:
        cls_i = int(cls)
        mask = y == cls_i
        if cls_i < 0:
            label = "Unknown"
            color = style.tsne_unknown_color
        else:
            if class_names is None or cls_i >= len(class_names):
                label = f"Class {cls_i}"
            else:
                label = str(class_names[cls_i])
            color = color_map.get(cls_i, None)
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=point_size,
            alpha=alpha,
            label=label,
            c=[color] if color is not None else None,
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, linestyle=style.grid_linestyle, alpha=style.grid_alpha * 0.9)
    if len(uniq) <= 18:
        ax.legend(markerscale=1.6, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def _scale_bubble_sizes(size_values: np.ndarray, style: PlotStyle) -> np.ndarray:
    s = np.asarray(size_values, dtype=np.float64)
    if s.size == 0:
        return s
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    if np.isclose(s_min, s_max):
        return np.full_like(s, (style.bubble_size_min + style.bubble_size_max) / 2.0, dtype=np.float64)
    ratio = (s - s_min) / (s_max - s_min)
    return style.bubble_size_min + ratio * (style.bubble_size_max - style.bubble_size_min)


def plot_bubble_chart(
    methods: Sequence[str],
    x_values: Sequence[float],
    y_values: Sequence[float],
    size_values: Sequence[float],
    out_path: Path,
    x_label: str = "Inference Latency (ms)",
    y_label: str = "Core Metric",
    size_label: str = "Params",
    title: str = "Performance-Efficiency Trade-off",
    style: Optional[PlotStyle] = None,
    highlight_method: str = "AERO",
    annotate: bool = True,
    invert_x: bool = False,
) -> None:
    """Plot a method-level bubble chart."""
    if len(methods) == 0:
        return
    style = _resolve_style(style)
    _apply_plot_style(style)

    m = np.array([str(v) for v in methods], dtype=object)
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    s_raw = np.asarray(size_values, dtype=np.float64)
    s = _scale_bubble_sizes(s_raw, style)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))

    for i in range(len(m)):
        is_hl = m[i].upper() == str(highlight_method).upper()
        color = style.highlight_color if is_hl else style.known_color
        alpha = min(style.bubble_alpha + 0.12, 0.95) if is_hl else style.bubble_alpha
        edge = "#111827" if is_hl else "#374151"
        ax.scatter(x[i], y[i], s=s[i], color=color, alpha=alpha, edgecolors=edge, linewidths=1.0, zorder=3 if is_hl else 2)
        if annotate:
            dx = 0.008 * (np.max(x) - np.min(x) + 1e-8)
            dy = 0.008 * (np.max(y) - np.min(y) + 1e-8)
            ax.text(x[i] + dx, y[i] + dy, str(m[i]), fontsize=style.legend_size, weight="bold" if is_hl else "normal")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=style.grid_linestyle, alpha=style.grid_alpha)
    if invert_x:
        ax.invert_xaxis()

    x_pad = 0.05 * (np.max(x) - np.min(x) + 1e-8)
    y_pad = 0.06 * (np.max(y) - np.min(y) + 1e-8)
    ax.set_xlim(np.min(x) - x_pad, np.max(x) + x_pad)
    ax.set_ylim(np.min(y) - y_pad, np.max(y) + y_pad)

    if s_raw.size >= 1:
        q = np.unique(np.quantile(s_raw, [0.2, 0.5, 0.8]))
        handles = []
        labels = []
        for val in q:
            handles.append(
                ax.scatter(
                    [],
                    [],
                    s=float(_scale_bubble_sizes(np.array([val]), style)[0]),
                    color="#9ca3af",
                    alpha=0.5,
                    edgecolors="#374151",
                    linewidths=0.8,
                )
            )
            labels.append(f"{val:.3g}")
        ax.legend(handles, labels, title=size_label, frameon=True, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def plot_dual_axis_bar(
    methods: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    out_path: Path,
    left_label: str = "Core Metric",
    right_label: str = "Inference Latency (ms)",
    title: str = "Dual-Axis Comparison",
    style: Optional[PlotStyle] = None,
    highlight_method: str = "AERO",
) -> None:
    """Plot a dual-axis bar/line chart."""
    if len(methods) == 0:
        return
    style = _resolve_style(style)
    _apply_plot_style(style)

    m = [str(v) for v in methods]
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    x = np.arange(len(m))

    fig, ax1 = plt.subplots(figsize=(9.0, 5.6))
    bar_colors = [style.highlight_color if n.upper() == str(highlight_method).upper() else style.known_color for n in m]
    ax1.bar(x, left, color=bar_colors, alpha=0.82, width=0.62)
    ax1.set_ylabel(left_label)
    ax1.set_xlabel("Method")
    ax1.set_xticks(x)
    ax1.set_xticklabels(m, rotation=22, ha="right")
    ax1.grid(True, axis="y", linestyle=style.grid_linestyle, alpha=style.grid_alpha)

    ax2 = ax1.twinx()
    ax2.plot(x, right, color="#111827", marker="o", linewidth=style.line_width + 0.2, label=right_label)
    ax2.set_ylabel(right_label)
    ax2.legend(loc="upper right", frameon=True)

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
