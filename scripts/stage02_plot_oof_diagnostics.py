# -*- coding: utf-8 -*-
"""Publication-style OOF diagnostic figures for ZnBERT + XGBoost.

Each target receives one figure with:
  1. measured/predicted marginal distributions,
  2. OOF parity scatter, linear fit and 95% confidence band,
  3. residual-distribution inset,
  4. sample-wise signed relative error grouped by CV fold.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde, t
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


INPUT_XLSX = Path("Downstream_Compare_ZnBERTv2_metrics.xlsx")
INPUT_SHEET = "OOF_preds"
OUT_DIR = Path("outputs") / "ZnBERT_XGB_redrawn_diagnostics"
N_SPLITS = 5
RANDOM_SEED = 42

BLUE = "#4C9ED9"
BLUE_DARK = "#1479B8"
CORAL = "#F06A66"
CORAL_DARK = "#D94D48"
INK = "#172033"
MUTED = "#667085"
GRID = "#D9E2EC"
FOLD_SHADE = "#EAF4FB"

TARGETS = {
    "UTS": {"unit": "MPa", "rel_ylim": (-100, 100)},
    "YS": {"unit": "MPa", "rel_ylim": (-120, 120)},
    "EL": {"unit": "%", "rel_ylim": (-200, 200)},
}


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.2,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def kde_count_curve(values, grid, bin_width):
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) < 2:
        return np.zeros_like(grid)
    kde = gaussian_kde(values)
    return kde(grid) * len(values) * bin_width


def regression_with_ci(x, y, grid):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * grid
    residual = y - (intercept + slope * x)
    dof = max(len(x) - 2, 1)
    residual_se = np.sqrt(np.sum(residual**2) / dof)
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx <= 0:
        ci = np.full_like(grid, residual_se)
    else:
        se_mean = residual_se * np.sqrt(
            1 / len(x) + (grid - np.mean(x)) ** 2 / sxx
        )
        ci = t.ppf(0.975, dof) * se_mean
    return fitted, fitted - ci, fitted + ci


def add_distribution_top(ax, measured, predicted, bins):
    ax.hist(
        measured,
        bins=bins,
        color=BLUE,
        alpha=0.48,
        edgecolor=BLUE_DARK,
        linewidth=0.7,
        label="Measured",
    )
    ax.hist(
        predicted,
        bins=bins,
        color=CORAL,
        alpha=0.42,
        edgecolor=CORAL_DARK,
        linewidth=0.7,
        label="Predicted",
    )
    grid = np.linspace(bins[0], bins[-1], 400)
    bin_width = bins[1] - bins[0]
    ax.plot(grid, kde_count_curve(measured, grid, bin_width), color=BLUE_DARK, lw=2.2)
    ax.plot(grid, kde_count_curve(predicted, grid, bin_width), color=CORAL_DARK, lw=2.2)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linestyle=":", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2)


def add_distribution_right(ax, measured, predicted, bins):
    ax.hist(
        measured,
        bins=bins,
        orientation="horizontal",
        color=BLUE,
        alpha=0.48,
        edgecolor=BLUE_DARK,
        linewidth=0.7,
    )
    ax.hist(
        predicted,
        bins=bins,
        orientation="horizontal",
        color=CORAL,
        alpha=0.42,
        edgecolor=CORAL_DARK,
        linewidth=0.7,
    )
    grid = np.linspace(bins[0], bins[-1], 400)
    bin_width = bins[1] - bins[0]
    ax.plot(kde_count_curve(measured, grid, bin_width), grid, color=BLUE_DARK, lw=2.2)
    ax.plot(kde_count_curve(predicted, grid, bin_width), grid, color=CORAL_DARK, lw=2.2)
    ax.set_xlabel("Count")
    ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, linestyle=":", linewidth=0.8)


def add_residual_inset(ax, residual, unit):
    inset = inset_axes(
        ax,
        width="39%",
        height="31%",
        loc="lower right",
        borderpad=1.05,
    )
    bins = np.histogram_bin_edges(residual, bins="fd")
    if len(bins) < 7:
        bins = np.linspace(np.min(residual), np.max(residual), 10)
    inset.hist(
        residual,
        bins=bins,
        color=CORAL,
        alpha=0.42,
        edgecolor=CORAL_DARK,
        linewidth=0.7,
    )
    grid = np.linspace(bins[0], bins[-1], 300)
    curve = kde_count_curve(residual, grid, bins[1] - bins[0])
    inset.plot(grid, curve, color=CORAL_DARK, lw=1.7)
    inset.axvline(0, color=INK, linestyle="--", linewidth=1.0)
    inset.axvline(np.mean(residual), color=CORAL_DARK, linewidth=1.4)
    inset.set_xlabel(f"Residual ({unit})", fontsize=10, fontweight="bold")
    inset.set_ylabel("Count", fontsize=10, fontweight="bold")
    inset.tick_params(labelsize=8.5, direction="in")
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.text(
        0.04,
        0.92,
        f"Mean = {np.mean(residual):+.2f}",
        transform=inset.transAxes,
        va="top",
        color=CORAL_DARK,
        fontsize=9,
        fontweight="bold",
    )


def add_parity(ax, measured, predicted, target, unit):
    values = np.concatenate([measured, predicted])
    span = np.ptp(values)
    pad = max(span * 0.06, 1.0)
    low = min(values) - pad
    high = max(values) + pad
    grid = np.linspace(low, high, 300)

    ax.scatter(
        measured,
        predicted,
        s=26,
        color=BLUE,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    ax.plot(grid, grid, color=INK, linestyle="--", linewidth=1.7, label="1:1 line")
    fit, lower, upper = regression_with_ci(measured, predicted, grid)
    ax.fill_between(grid, lower, upper, color=CORAL, alpha=0.20, linewidth=0)
    ax.plot(grid, fit, color=CORAL_DARK, linewidth=2.3, label="Linear fit")

    r2 = r2_score(measured, predicted)
    rmse = np.sqrt(mean_squared_error(measured, predicted))
    mae = mean_absolute_error(measured, predicted)
    metric_text = (
        rf"$R^2$ = {r2:.3f}" + "\n"
        + f"RMSE = {rmse:.2f} {unit}" + "\n"
        + f"MAE = {mae:.2f} {unit}"
    )
    ax.text(
        0.045,
        0.955,
        metric_text,
        transform=ax.transAxes,
        va="top",
        color=BLUE_DARK,
        fontsize=13.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "alpha": 0.92},
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"Measured {target} ({unit})")
    ax.set_ylabel(f"Predicted {target} ({unit})")
    ax.grid(color=GRID, linestyle=":", linewidth=0.9)
    ax.set_axisbelow(True)
    add_residual_inset(ax, predicted - measured, unit)


def fold_assignments(n_samples):
    assignments = np.zeros(n_samples, dtype=int)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (_, test_idx) in enumerate(splitter.split(np.arange(n_samples)), start=1):
        assignments[test_idx] = fold
    return assignments


def add_relative_error_panel(ax, measured, predicted, folds, target, ylim):
    relative_error = (predicted - measured) / np.abs(measured) * 100.0
    order = np.argsort(folds, kind="stable")
    error_ordered = relative_error[order]
    fold_ordered = folds[order]
    x = np.arange(len(order))

    lower, upper = ylim
    displayed = np.clip(error_ordered, lower, upper)
    in_range = (error_ordered >= lower) & (error_ordered <= upper)
    above = error_ordered > upper
    below = error_ordered < lower

    starts = []
    ends = []
    cursor = 0
    for fold in range(1, N_SPLITS + 1):
        count = int(np.sum(fold_ordered == fold))
        start = cursor
        end = cursor + count - 1
        starts.append(start)
        ends.append(end)
        if fold % 2 == 1:
            ax.axvspan(start - 0.5, end + 0.5, color=FOLD_SHADE, alpha=0.72, zorder=0)
        if fold < N_SPLITS:
            ax.axvline(end + 0.5, color="#B8C4D2", linewidth=1.0)
        ax.text(
            (start + end) / 2,
            upper * 0.84,
            f"Fold {fold}",
            ha="center",
            va="top",
            color=MUTED,
            fontsize=13,
            fontweight="bold",
        )
        cursor += count

    ax.vlines(x[in_range], 0, displayed[in_range], color=BLUE, alpha=0.34, linewidth=0.8)
    ax.scatter(x[in_range], displayed[in_range], s=12, color=BLUE, alpha=0.82, edgecolors="none")
    ax.scatter(x[above], np.full(np.sum(above), upper * 0.965), marker="^", s=28, color=CORAL_DARK, zorder=4)
    ax.scatter(x[below], np.full(np.sum(below), lower * 0.965), marker="v", s=28, color=CORAL_DARK, zorder=4)

    mean_error = float(np.mean(relative_error))
    ax.axhline(0, color=INK, linestyle="--", linewidth=1.2)
    if lower < mean_error < upper:
        ax.axhline(mean_error, color=CORAL_DARK, linewidth=1.8)
    ax.text(
        0.012,
        0.08,
        f"Mean signed error = {mean_error:+.2f}%",
        transform=ax.transAxes,
        color=CORAL_DARK,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )
    clipped_count = int(np.sum(above) + np.sum(below))
    if clipped_count:
        ax.text(
            0.988,
            0.08,
            f"▲/▼  {clipped_count} values outside display range",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=10.5,
            ha="right",
            va="bottom",
        )

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("OOF samples grouped by fold")
    ax.set_ylabel("Signed relative error (%)")
    ax.set_title(f"{target}: sample-wise OOF relative error", pad=10)
    ax.grid(axis="y", color=GRID, linestyle=":", linewidth=0.9)
    ax.set_axisbelow(True)


def build_figure(df, target, unit, rel_ylim):
    measured = df[target].to_numpy(dtype=float)
    predicted = df[f"Pred_{target}_XGB"].to_numpy(dtype=float)
    folds = fold_assignments(len(df))

    fig = plt.figure(figsize=(10.4, 10.2), facecolor="white")
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[3.8, 1.55],
        hspace=0.34,
        top=0.93,
        bottom=0.08,
        left=0.095,
        right=0.955,
    )
    joint = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer[0],
        width_ratios=[4.4, 1.05],
        height_ratios=[1.05, 4.4],
        hspace=0.04,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(joint[0, 0])
    ax_scatter = fig.add_subplot(joint[1, 0])
    ax_right = fig.add_subplot(joint[1, 1], sharey=ax_scatter)
    ax_blank = fig.add_subplot(joint[0, 1])
    ax_blank.axis("off")
    ax_relative = fig.add_subplot(outer[1])

    combined = np.concatenate([measured, predicted])
    bin_edges = np.histogram_bin_edges(combined, bins="fd")
    if len(bin_edges) < 10:
        bin_edges = np.linspace(np.min(combined), np.max(combined), 12)

    add_distribution_top(ax_top, measured, predicted, bin_edges)
    add_parity(ax_scatter, measured, predicted, target, unit)
    add_distribution_right(ax_right, measured, predicted, bin_edges)
    add_relative_error_panel(ax_relative, measured, predicted, folds, target, rel_ylim)

    ax_top.set_xlim(ax_scatter.get_xlim())
    ax_right.set_ylim(ax_scatter.get_ylim())
    fig.suptitle(f"{target} — ZnBERT + XGBoost OOF diagnostics", y=0.985, fontsize=22, fontweight="bold", color=INK)
    return fig, folds


def main():
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)

    export_rows = []
    pdf_path = OUT_DIR / "ZnBERT_XGB_UTS_YS_EL_diagnostics_multipage.pdf"
    with PdfPages(pdf_path) as pdf:
        for target, meta in TARGETS.items():
            fig, folds = build_figure(df, target, meta["unit"], meta["rel_ylim"])
            stem = OUT_DIR / f"ZnBERT_XGB_{target}_OOF_diagnostic"
            fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            measured = df[target].to_numpy(dtype=float)
            predicted = df[f"Pred_{target}_XGB"].to_numpy(dtype=float)
            relative_error = (predicted - measured) / np.abs(measured) * 100.0
            for index in range(len(df)):
                export_rows.append(
                    {
                        "Target": target,
                        "Sample": index + 1,
                        "Fold": int(folds[index]),
                        "Measured": float(measured[index]),
                        "Predicted": float(predicted[index]),
                        "Residual_pred_minus_measured": float(predicted[index] - measured[index]),
                        "Signed_relative_error_pct": float(relative_error[index]),
                    }
                )

    pd.DataFrame(export_rows).to_csv(
        OUT_DIR / "ZnBERT_XGB_OOF_plot_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved figures and plot data to: {OUT_DIR}")


if __name__ == "__main__":
    main()
