"""Step 24: dependency-free SVG figures for the repeated-CV experiment.

The project intentionally avoids a plotting dependency.  Every figure is
written as a standalone SVG by hand, which keeps the environment unchanged and
the figures diffable in git.

Aggregation rules used here
---------------------------
* ROC and PR curves are computed **within each repeat** from that repeat's
  out-of-fold predictions (every training row appears exactly once per repeat).
  The plotted line is the mean across the five repeats and the shaded band is
  +/- 1 SD across repeats.  Predictions are never pooled across repeats,
  because that would duplicate every row five times.
* Bar charts show fold-level means and SDs across the 25 folds.  Those folds
  are correlated (5 folds inside each of 5 repeats), so the error bars are
  descriptive, not confidence intervals.
* Confusion matrices show the mean count per fold across the 25 folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import METRIC_NAMES

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
FONT = "Segoe UI, Arial, sans-serif"
INK = "#222222"
MUTED = "#555555"
GRID = "#e6e6e6"
AXIS = "#444444"

MODEL_COLORS = {
    "dummy_prior": "#7f7f7f",
    "logistic_regression": "#1f77b4",
    "decision_tree": "#2ca02c",
    "random_forest": "#ff7f0e",
    "svc_rbf": "#9467bd",
}

METRIC_SHORT = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Avg precision",
    "recall_sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced acc",
    "brier_score": "Brier (lower)",
}


def _esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class Svg:
    """Tiny append-only SVG canvas."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        ]

    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str | None = None,
        stroke_width: float = 1,
        rx: float = 0,
        opacity: float = 1.0,
    ) -> None:
        extra = (
            f' stroke="{stroke}" stroke-width="{stroke_width}"'
            if stroke
            else ""
        )
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" opacity="{opacity}"{extra}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = AXIS,
        width: float = 1,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"'
            f"{dash_attr}/>"
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        stroke: str,
        width: float = 2,
        fill: str = "none",
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"'
            f"{dash_attr}/>"
        )

    def polygon(
        self,
        points: Sequence[tuple[float, float]],
        fill: str,
        opacity: float = 0.18,
    ) -> None:
        self.add(
            f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" '
            f'fill="{fill}" opacity="{opacity}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        size: float = 12,
        anchor: str = "start",
        fill: str = INK,
        weight: str = "normal",
        rotate: float | None = None,
    ) -> None:
        transform = (
            f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        )
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}"{transform}>{_esc(value)}</text>'
        )

    def save(self, path: Path) -> Path:
        self.parts.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts), encoding="utf-8")
        return path


class Frame:
    """Data-to-pixel mapping for one axes rectangle."""

    def __init__(
        self,
        svg: Svg,
        x0: float,
        y0: float,
        width: float,
        height: float,
        xlim: tuple[float, float] = (0.0, 1.0),
        ylim: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.svg = svg
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.xlim = xlim
        self.ylim = ylim

    def sx(self, value: float) -> float:
        return self.x0 + (value - self.xlim[0]) / (
            self.xlim[1] - self.xlim[0]
        ) * self.width

    def sy(self, value: float) -> float:
        return self.y0 + self.height - (value - self.ylim[0]) / (
            self.ylim[1] - self.ylim[0]
        ) * self.height

    def point(self, x: float, y: float) -> tuple[float, float]:
        return self.sx(x), self.sy(y)


def _draw_frame(
    svg: Svg,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    n_xticks: int = 5,
    n_yticks: int = 5,
    numeric_x_ticks: bool = True,
) -> Frame:
    """Draw title, subtitle, grid, axes and tick labels; return the mapping."""
    frame = Frame(svg, x0, y0, width, height, xlim=xlim, ylim=ylim)

    svg.text(svg.width / 2, 26, title, size=16, anchor="middle", weight="600")
    svg.text(
        svg.width / 2, 46, subtitle, size=11, anchor="middle", fill=MUTED
    )

    # Grid and tick labels.
    if numeric_x_ticks:
        for value in np.linspace(xlim[0], xlim[1], n_xticks):
            px = frame.sx(float(value))
            svg.line(px, y0, px, y0 + height, stroke=GRID, width=1)
            svg.text(px, y0 + height + 18, f"{value:.1f}", size=10, anchor="middle", fill=MUTED)
    for value in np.linspace(ylim[0], ylim[1], n_yticks):
        py = frame.sy(float(value))
        svg.line(x0, py, x0 + width, py, stroke=GRID, width=1)
        svg.text(x0 - 9, py + 3.5, f"{value:.2f}" if ylim[1] <= 1.01 else f"{value:.3g}", size=10, anchor="end", fill=MUTED)

    svg.rect(x0, y0, width, height, fill="none", stroke=AXIS, stroke_width=1.2)
    svg.text(x0 + width / 2, y0 + height + 44, xlabel, size=12, anchor="middle")
    svg.text(
        x0 - 52,
        y0 + height / 2,
        ylabel,
        size=12,
        anchor="middle",
        rotate=-90,
    )
    return frame


def _legend(
    svg: Svg,
    x: float,
    y: float,
    entries: Sequence[tuple[str, str]],
    *,
    title: str | None = None,
    width: float = 262,
    line_height: float = 22,
) -> None:
    height = 24 + line_height * len(entries) + (20 if title else 0)
    svg.rect(x, y, width, height, fill="#ffffff", stroke="#cccccc", stroke_width=1, rx=4)
    cursor = y + 18
    if title:
        svg.text(x + 12, cursor, title, size=11, weight="600")
        cursor += 18
    for color, label in entries:
        svg.line(x + 12, cursor - 4, x + 40, cursor - 4, stroke=color, width=3)
        svg.text(x + 48, cursor, label, size=10.5)
        cursor += line_height


# ---------------------------------------------------------------------------
# Curve preparation
# ---------------------------------------------------------------------------
def roc_curve_bundle(
    oof: pd.DataFrame, models: Sequence[str], n_points: int = 201
) -> dict[str, dict[str, np.ndarray | float]]:
    """Mean +/- SD ROC curve across repeats (no cross-repeat pooling)."""
    grid = np.linspace(0.0, 1.0, n_points)
    bundle: dict[str, dict[str, np.ndarray | float]] = {}
    for model in models:
        curves = []
        scores = []
        for _, group in oof[oof["model"] == model].groupby("repeat", sort=True):
            y_true = group["y_true"].to_numpy()
            y_proba = group["y_proba"].to_numpy()
            fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
            curves.append(np.interp(grid, fpr, tpr))
            scores.append(roc_auc_score(y_true, y_proba))
        stacked = np.vstack(curves)
        bundle[model] = {
            "x": grid,
            "mean": stacked.mean(axis=0),
            "sd": stacked.std(axis=0, ddof=1),
            "score_mean": float(np.mean(scores)),
            "score_sd": float(np.std(scores, ddof=1)),
        }
    return bundle


def pr_curve_bundle(
    oof: pd.DataFrame, models: Sequence[str], n_points: int = 201
) -> dict[str, dict[str, np.ndarray | float]]:
    """Mean +/- SD precision-recall curve across repeats.

    Each repeat's curve is interpolated onto a shared recall grid before
    averaging.  The reported score is average precision (step-wise), not a
    trapezoidal area.
    """
    grid = np.linspace(0.0, 1.0, n_points)
    bundle: dict[str, dict[str, np.ndarray | float]] = {}
    for model in models:
        curves = []
        scores = []
        for _, group in oof[oof["model"] == model].groupby("repeat", sort=True):
            y_true = group["y_true"].to_numpy()
            y_proba = group["y_proba"].to_numpy()
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            # recall is descending; make it ascending for np.interp.
            curves.append(np.interp(grid, recall[::-1], precision[::-1]))
            scores.append(average_precision_score(y_true, y_proba))
        stacked = np.vstack(curves)
        bundle[model] = {
            "x": grid,
            "mean": stacked.mean(axis=0),
            "sd": stacked.std(axis=0, ddof=1),
            "score_mean": float(np.mean(scores)),
            "score_sd": float(np.std(scores, ddof=1)),
        }
    return bundle


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_roc_figure(
    bundle: dict[str, dict[str, np.ndarray | float]],
    models: Sequence[str],
    path: Path,
) -> Path:
    plot_w, plot_h = 560, 500
    svg = Svg(width=960, height=690)
    frame = _draw_frame(
        svg,
        x0=80,
        y0=78,
        width=plot_w,
        height=plot_h,
        title="ROC curves - repeated stratified cross-validation",
        subtitle=(
            "Mean curve across 5 repeats; band = +/-1 SD across repeats. "
            "Each repeat's curve uses its own out-of-fold predictions (n=4,088)."
        ),
        xlabel="False positive rate (1 - specificity)",
        ylabel="True positive rate (sensitivity)",
    )

    svg.line(*frame.point(0, 0), *frame.point(1, 1), stroke="#999999", width=1.2, dash="5 4")

    for model in models:
        data = bundle[model]
        x = data["x"]
        mean = np.clip(data["mean"], 0.0, 1.0)
        sd = np.asarray(data["sd"], dtype=float)
        upper = np.clip(mean + sd, 0.0, 1.0)
        lower = np.clip(mean - sd, 0.0, 1.0)
        band = [frame.point(float(xi), float(yi)) for xi, yi in zip(x, upper)]
        band += [
            frame.point(float(xi), float(yi))
            for xi, yi in zip(x[::-1], lower[::-1])
        ]
        svg.polygon(band, MODEL_COLORS[model], opacity=0.15)
        svg.polyline(
            [frame.point(float(xi), float(yi)) for xi, yi in zip(x, mean)],
            stroke=MODEL_COLORS[model],
            width=2.2,
        )

    _legend(
        svg,
        x=80 + plot_w + 26,
        y=78,
        entries=[
            (
                MODEL_COLORS[model],
                f"{model}  (AUC {bundle[model]['score_mean']:.3f})",
            )
            for model in models
        ],
        title="Mean pooled-within-repeat AUC",
    )
    svg.text(
        80,
        690 - 22,
        "Chance line: dashed grey.  Curves are not pooled across repeats.",
        size=10,
        fill=MUTED,
    )
    return svg.save(path)


def save_pr_figure(
    bundle: dict[str, dict[str, np.ndarray | float]],
    models: Sequence[str],
    path: Path,
    prevalence: float,
) -> Path:
    plot_w, plot_h = 560, 500
    svg = Svg(width=960, height=690)
    frame = _draw_frame(
        svg,
        x0=80,
        y0=78,
        width=plot_w,
        height=plot_h,
        title="Precision-recall curves - repeated stratified cross-validation",
        subtitle=(
            "Mean curve across 5 repeats; band = +/-1 SD across repeats. "
            "Score = average precision (step-wise), not trapezoidal PR-AUC."
        ),
        xlabel="Recall / sensitivity",
        ylabel="Precision",
    )

    svg.line(
        *frame.point(0, prevalence),
        *frame.point(1, prevalence),
        stroke="#999999",
        width=1.2,
        dash="5 4",
    )
    svg.text(
        frame.sx(0.02),
        frame.sy(prevalence) - 7,
        f"no-skill = prevalence {prevalence:.4f}",
        size=10,
        fill=MUTED,
    )

    for model in models:
        data = bundle[model]
        x = data["x"]
        mean = np.clip(data["mean"], 0.0, 1.0)
        sd = np.asarray(data["sd"], dtype=float)
        upper = np.clip(mean + sd, 0.0, 1.0)
        lower = np.clip(mean - sd, 0.0, 1.0)
        band = [frame.point(float(xi), float(yi)) for xi, yi in zip(x, upper)]
        band += [
            frame.point(float(xi), float(yi))
            for xi, yi in zip(x[::-1], lower[::-1])
        ]
        svg.polygon(band, MODEL_COLORS[model], opacity=0.15)
        svg.polyline(
            [frame.point(float(xi), float(yi)) for xi, yi in zip(x, mean)],
            stroke=MODEL_COLORS[model],
            width=2.2,
        )

    _legend(
        svg,
        x=80 + plot_w + 26,
        y=78,
        entries=[
            (
                MODEL_COLORS[model],
                f"{model}  (AP {bundle[model]['score_mean']:.3f})",
            )
            for model in models
        ],
        title="Mean pooled-within-repeat AP",
    )
    svg.text(
        80,
        690 - 22,
        "Curves are not pooled across repeats; AP is step-wise, not trapezoidal.",
        size=10,
        fill=MUTED,
    )
    return svg.save(path)


def save_metric_comparison(
    summary: pd.DataFrame,
    models: Sequence[str],
    path: Path,
) -> Path:
    metrics = [
        "roc_auc",
        "average_precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]
    fold = summary[summary["scope"] == "fold_level"]

    x0, y0, plot_w, plot_h = 80, 100, 880, 350
    svg = Svg(width=1040, height=540)
    frame = _draw_frame(
        svg,
        x0=x0,
        y0=y0,
        width=plot_w,
        height=plot_h,
        title="Fold-level metric comparison (fixed threshold 0.50)",
        subtitle=(
            "Bars = mean across 25 folds; whiskers = +/-1 SD across folds. "
            "The 25 folds are correlated (5 repeats x 5 folds), so SD is descriptive only."
        ),
        xlabel="",
        ylabel="Score (higher is better)",
        numeric_x_ticks=False,
    )
    svg.rect(x0, y0, plot_w, plot_h, fill="none", stroke=AXIS, stroke_width=1.2)

    group_width = plot_w / len(metrics)
    bar_width = group_width * 0.78 / len(models)
    for metric_index, metric in enumerate(metrics):
        base_x = x0 + metric_index * group_width + group_width * 0.11
        for model_index, model in enumerate(models):
            row = fold[(fold["model"] == model) & (fold["metric"] == metric)]
            if row.empty:
                continue
            mean = float(row["mean"].iloc[0])
            sd = float(row["sd"].iloc[0])
            bx = base_x + model_index * bar_width
            svg.rect(
                bx,
                frame.sy(mean),
                bar_width * 0.86,
                frame.sy(0) - frame.sy(mean),
                fill=MODEL_COLORS[model],
                opacity=0.9,
            )
            top = frame.sy(min(1.0, mean + sd))
            bottom = frame.sy(max(0.0, mean - sd))
            center = bx + bar_width * 0.43
            svg.line(center, top, center, bottom, stroke="#333333", width=1)
            svg.line(center - 3.5, top, center + 3.5, top, stroke="#333333", width=1)
            svg.line(center - 3.5, bottom, center + 3.5, bottom, stroke="#333333", width=1)
        svg.text(
            x0 + metric_index * group_width + group_width / 2,
            y0 + plot_h + 20,
            METRIC_SHORT[metric],
            size=10.5,
            anchor="middle",
        )

    # Legend along the bottom, one row.
    legend_y = y0 + plot_h + 62
    cursor = x0
    for model in models:
        svg.rect(cursor, legend_y - 9, 12, 12, fill=MODEL_COLORS[model], rx=2)
        svg.text(cursor + 18, legend_y + 1, model, size=10.5)
        cursor += 22 + len(model) * 6.6
    return svg.save(path)


def save_brier_comparison(
    summary: pd.DataFrame, models: Sequence[str], path: Path
) -> Path:
    fold = summary[(summary["scope"] == "fold_level") & (summary["metric"] == "brier_score")]
    values = []
    for model in models:
        row = fold[fold["model"] == model]
        values.append((float(row["mean"].iloc[0]), float(row["sd"].iloc[0])))
    upper = max(mean + sd for mean, sd in values) * 1.25
    y_max = max(0.05, upper)

    x0, y0, plot_w, plot_h = 80, 100, 560, 330
    svg = Svg(width=920, height=520)
    frame = _draw_frame(
        svg,
        x0=x0,
        y0=y0,
        width=plot_w,
        height=plot_h,
        title="Brier score comparison (lower is better)",
        subtitle="Mean across 25 folds; whiskers = +/-1 SD across correlated folds.",
        xlabel="",
        ylabel="Brier score",
        ylim=(0.0, y_max),
        n_yticks=5,
        numeric_x_ticks=False,
    )
    group_width = plot_w / len(models)
    bar_width = group_width * 0.55
    for index, model in enumerate(models):
        mean, sd = values[index]
        bx = x0 + index * group_width + (group_width - bar_width) / 2
        svg.rect(
            bx,
            frame.sy(mean),
            bar_width,
            frame.sy(0) - frame.sy(mean),
            fill=MODEL_COLORS[model],
            opacity=0.9,
        )
        center = bx + bar_width / 2
        top = frame.sy(mean + sd)
        bottom = frame.sy(max(0.0, mean - sd))
        svg.line(center, top, center, bottom, stroke="#333333", width=1)
        svg.line(center - 4, top, center + 4, top, stroke="#333333", width=1)
        svg.line(center - 4, bottom, center + 4, bottom, stroke="#333333", width=1)
        svg.text(center, y0 + plot_h + 18, model, size=10, anchor="middle")
        svg.text(center, frame.sy(mean) - 6, f"{mean:.4f}", size=9.5, anchor="middle", fill=MUTED)

    svg.text(
        x0,
        520 - 20,
        "Brier score uses predicted probabilities and is not comparable to a thresholded accuracy.",
        size=10,
        fill=MUTED,
    )
    return svg.save(path)


def save_confusion_matrix_grid(
    fold_metrics: pd.DataFrame, models: Sequence[str], path: Path
) -> Path:
    panel_w, panel_h = 440, 240
    cols, rows = 2, 3
    svg = Svg(width=960, height=70 + rows * panel_h + 40)
    svg.text(
        svg.width / 2,
        28,
        "Confusion matrices - mean counts per fold (threshold 0.50)",
        size=16,
        anchor="middle",
        weight="600",
    )
    svg.text(
        svg.width / 2,
        48,
        "Averaged over 25 folds (5 repeats x 5 folds); row counts therefore do not sum to 4,088.",
        size=11,
        anchor="middle",
        fill=MUTED,
    )

    cell_w, cell_h = 92, 62
    for index, model in enumerate(models):
        col = index % cols
        row = index // cols
        ox = 20 + col * (panel_w + 20)
        oy = 62 + row * panel_h

        subset = fold_metrics[fold_metrics["model"] == model]
        means = {
            key: float(subset[key].mean())
            for key in ("tn", "fp", "fn", "tp")
        }
        sens = float(subset["recall_sensitivity"].mean())
        spec = float(subset["specificity"].mean())

        svg.rect(ox, oy, panel_w, panel_h - 8, fill="#fbfbfb", stroke="#dddddd", rx=6)
        svg.text(ox + 14, oy + 24, model, size=13, weight="600")

        matrix = [
            [means["tn"], means["fp"]],
            [means["fn"], means["tp"]],
        ]
        maximum = max(max(r) for r in matrix) or 1.0
        left = ox + 92
        top = oy + 52

        for c, label in enumerate(("Pred 0", "Pred 1")):
            svg.text(left + c * cell_w + cell_w / 2, top - 8, label, size=10.5, anchor="middle", fill=MUTED)
        for r, label in enumerate(("Actual 0", "Actual 1")):
            svg.text(left - 10, top + r * cell_h + cell_h / 2 + 4, label, size=10.5, anchor="end", fill=MUTED)

        for r in range(2):
            for c in range(2):
                value = matrix[r][c]
                intensity = (value / maximum) ** 0.5
                shade = int(246 - 135 * intensity)
                x = left + c * cell_w
                y = top + r * cell_h
                svg.rect(
                    x,
                    y,
                    cell_w - 3,
                    cell_h - 3,
                    fill=f"rgb({shade},{shade + 6},{shade + 16})",
                    stroke="#ffffff",
                    stroke_width=2,
                )
                svg.text(
                    x + (cell_w - 3) / 2,
                    y + (cell_h - 3) / 2 + 6,
                    f"{value:.1f}",
                    size=17,
                    anchor="middle",
                    weight="600",
                    fill="#ffffff" if intensity > 0.55 else INK,
                )

        svg.text(
            ox + 14,
            oy + panel_h - 30,
            f"mean sensitivity {sens:.3f} | mean specificity {spec:.3f}",
            size=10,
            fill=MUTED,
        )
        svg.text(
            ox + 14,
            oy + panel_h - 14,
            "TN / FP / FN / TP are fold means, not corpus totals.",
            size=9,
            fill="#888888",
        )

    return svg.save(path)
