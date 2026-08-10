# -*- coding: utf-8 -*-
"""Deduplicate the Zn-Mg-Li-Cu screening grid, rank by CPI, and plot results.

The predictions are taken from the existing ZnBERT+XGBoost screening workbook.
The search space contains exactly60,480 unique candidates.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = BUNDLE_ROOT / "artifacts" / "final_screening_model" / "ZnMgLiCu_Extrusion_Screening.xlsx"
OUT_DIR = BUNDLE_ROOT / "outputs" / "ZnMgLiCu_CPI_sum_AR6_ELminus5_Top1000_20260730"

KEY_COLS = ["Mg_wt", "Li_wt", "Cu_wt", "Extrusion_T", "Extrusion_AR"]
PRED_COLS = ["Pred_UTS", "Pred_YS", "Pred_EL"]
NORM_COLS = ["UTS_norm", "YS_norm", "EL_norm"]

CPI_WEIGHTS = {"UTS": 1.0, "YS": 1.0, "EL": 1.0}
EL_ADJUSTMENT = -5.0

MG_VALUES = np.round(np.arange(0.10, 0.40 + 1e-9, 0.05), 2)
LI_VALUES = np.round(np.arange(0.10, 0.80 + 1e-9, 0.10), 2)
CU_VALUES = np.round(np.arange(0.10, 3.50 + 1e-9, 0.20), 2)
T_VALUES = np.arange(210, 300 + 1, 10)
AR_VALUES = np.array([20, 25, 30, 36, 48, 64])

BLUE = "#4C78A8"
BLUE_LIGHT = "#8DB6D9"
GOLD = "#F2C344"
CORAL = "#1F3B73"
PURPLE = "#5B3F8C"
MAGENTA = "#B12A6B"
INK = "#172033"
MUTED = "#687386"
GRID = "#D8E1EA"


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 24,
            "axes.labelweight": "bold",
            "axes.titlesize": 28,
            "axes.titleweight": "bold",
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def safe_minmax(values):
    values = np.asarray(values, dtype=float)
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def select_row(df, mg, li, cu, temperature, area_ratio):
    mask = (
        np.isclose(df["Mg_wt"], mg)
        & np.isclose(df["Li_wt"], li)
        & np.isclose(df["Cu_wt"], cu)
        & (df["Extrusion_T"] == temperature)
        & (df["Extrusion_AR"] == area_ratio)
    )
    result = df.loc[mask]
    if len(result) != 1:
        raise ValueError(
            f"Expected one row for {mg=}, {li=}, {cu=}, {temperature=}, "
            f"{area_ratio=}; found {len(result)}"
        )
    return result.iloc[0].copy()


def validate_grid(df):
    expected_count = (
        len(MG_VALUES)
        * len(LI_VALUES)
        * len(CU_VALUES)
        * len(T_VALUES)
        * len(AR_VALUES)
    )
    checks = {
        "Mg": np.array_equal(np.sort(df["Mg_wt"].unique()), MG_VALUES),
        "Li": np.array_equal(np.sort(df["Li_wt"].unique()), LI_VALUES),
        "Cu": np.array_equal(np.sort(df["Cu_wt"].unique()), CU_VALUES),
        "Extrusion_T": np.array_equal(np.sort(df["Extrusion_T"].unique()), T_VALUES),
        "Extrusion_AR": np.array_equal(np.sort(df["Extrusion_AR"].unique()), AR_VALUES),
        "candidate_count": len(df) == expected_count,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Screening-grid validation failed: {failed}")
    return expected_count


def plot_cpi_space(df, global_best, expert_260_20, expert_240_25):
    fig = plt.figure(figsize=(13.6, 9.9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=float(df["CPI"].min()), vmax=float(df["CPI"].max()))
    scatter = ax.scatter(
        df["Pred_EL"],
        df["Pred_YS"],
        df["Pred_UTS"],
        c=df["CPI"],
        cmap="turbo",
        norm=norm,
        s=5.2,
        alpha=0.52,
        depthshade=False,
        linewidths=0,
        rasterized=True,
    )

    highlights = [
        (global_best, "Global CPI optimum", MAGENTA, "*", 360),
        (expert_260_20, "Expert: 260 °C / AR 20", CORAL, "D", 220),
        (expert_240_25, "Expert: 240 °C / AR 25", GOLD, "o", 225),
    ]
    for row, label, color, marker, size in highlights:
        ax.scatter(
            [row["Pred_EL"]],
            [row["Pred_YS"]],
            [row["Pred_UTS"]],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=2.4,
            depthshade=False,
            label=label,
            zorder=20,
        )

    ax.set_xlabel("Adjusted predicted EL (%)", labelpad=18)
    ax.set_ylabel("Predicted YS (MPa)", labelpad=19)
    ax.set_zlabel("Predicted UTS (MPa)", labelpad=15)
    ax.set_title(
        "Zn–Mg–Li–Cu performance space colored by CPI\n"
        f"{len(df):,} candidates | EL predictions adjusted by -5 percentage points",
        pad=28,
    )
    ax.view_init(elev=24, azim=-52)
    ax.tick_params(axis="x", labelsize=17, pad=4)
    ax.tick_params(axis="y", labelsize=17, pad=4)
    ax.tick_params(axis="z", labelsize=17, pad=5)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.8
    ax.xaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.zaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.09, shrink=0.78)
    colorbar.set_label("CPI", fontsize=20, fontweight="bold", labelpad=12)
    colorbar.ax.tick_params(labelsize=17)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.94),
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=16,
    )

    figure_stem = OUT_DIR / "ZnMgLiCu_CPI_3D_performance_space"
    fig.savefig(figure_stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalize_path(values, minima, maxima):
    values = np.asarray(values, dtype=float)
    minima = np.asarray(minima, dtype=float)
    maxima = np.asarray(maxima, dtype=float)
    return (values - minima) / np.where(maxima > minima, maxima - minima, 1.0)


def plot_parallel_routes(df, expert_260_20, expert_240_25):
    columns = [
        "Mg_wt",
        "Li_wt",
        "Cu_wt",
        "Extrusion_T",
        "Extrusion_AR",
        "Pred_UTS",
        "Pred_YS",
        "Pred_EL",
        "CPI",
    ]
    labels = [
        "Mg\n(wt%)",
        "Li\n(wt%)",
        "Cu\n(wt%)",
        "Extrusion T\n($^{\\circ}$C)",
        "Extrusion AR",
        "UTS\n(MPa)",
        "YS\n(MPa)",
        "EL\n(%)",
        "CPI",
    ]
    minima = df[columns].min().to_numpy(float)
    maxima = df[columns].max().to_numpy(float)
    x = np.arange(len(columns))

    # Route-specific typography is set locally after the global style so these
    # larger sizes cannot be overwritten by a later rcParams/style update.
    fig, ax = plt.subplots(figsize=(25.0, 12.0), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.17, top=0.75)
    top_background = df.nlargest(1000, "CPI")
    background_values = normalize_path(
        top_background[columns].to_numpy(float), minima, maxima
    )
    for path in background_values:
        ax.plot(x, path, color=BLUE_LIGHT, linewidth=0.7, alpha=0.055, zorder=1)

    expert_260_values = expert_260_20[columns].to_numpy(float)
    expert_240_values = expert_240_25[columns].to_numpy(float)
    expert_260_norm = normalize_path(expert_260_values, minima, maxima)
    expert_240_norm = normalize_path(expert_240_values, minima, maxima)

    ax.plot(
        x,
        expert_260_norm,
        color=CORAL,
        linewidth=3.2,
        linestyle="-",
        marker="D",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="Expert: 260 °C / AR 20",
        zorder=6,
    )
    ax.plot(
        x,
        expert_240_norm,
        color=GOLD,
        linewidth=3.4,
        linestyle="--",
        marker="o",
        markersize=9,
        markeredgecolor=INK,
        markeredgewidth=0.9,
        label="Expert: 240 °C / AR 25",
        zorder=5,
    )

    for index, xpos in enumerate(x):
        ax.axvline(xpos, color=INK, linewidth=1.15, alpha=0.75, zorder=0)
        ax.text(
            xpos,
            1.065,
            f"{maxima[index]:g}",
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=20,
            fontweight="bold",
        )
        ax.text(
            xpos,
            -0.065,
            f"{minima[index]:g}",
            ha="center",
            va="top",
            color=MUTED,
            fontsize=20,
            fontweight="bold",
        )

        p_offset = 0.055 if expert_240_norm[index] >= expert_260_norm[index] else -0.055
        e_offset = -0.055 if expert_240_norm[index] >= expert_260_norm[index] else 0.055
        ax.text(
            xpos,
            np.clip(expert_240_norm[index] + p_offset, -0.03, 1.04),
            f"{expert_240_values[index]:.3g}",
            ha="center",
            va="bottom" if p_offset > 0 else "top",
            color="#8A6A00",
            fontsize=19.5,
            fontweight="bold",
            zorder=10,
        )
        ax.text(
            xpos,
            np.clip(expert_260_norm[index] + e_offset, -0.03, 1.04),
            f"{expert_260_values[index]:.3g}",
            ha="center",
            va="bottom" if e_offset > 0 else "top",
            color=CORAL,
            fontsize=19.5,
            fontweight="bold",
            zorder=10,
        )

    ax.set_xlim(-0.25, len(columns) - 0.75)
    ax.set_ylim(-0.12, 1.13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=25, fontweight="bold")
    ax.set_yticks([])
    fig.suptitle(
        "Expert-adjusted alloy under two extrusion conditions",
        fontsize=36,
        fontweight="bold",
        y=0.975,
        color=INK,
    )
    ax.grid(axis="y", color=GRID, linestyle=":", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=BLUE_LIGHT, lw=1.5, alpha=0.65, label="Top-1000 high-CPI pathways"),
        Line2D([0], [0], color=CORAL, lw=3.2, marker="D", markersize=7, markeredgecolor="white", label="Expert: 260 °C / AR 20"),
        Line2D([0], [0], color=GOLD, lw=3.2, ls="--", marker="o", markersize=8, markeredgecolor=INK, label="Expert: 240 °C / AR 25"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        fontsize=23,
    )

    figure_stem = OUT_DIR / "ZnMgLiCu_screening_to_expert_parallel_routes"
    fig.savefig(figure_stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT_DIR / "ZnMgLiCu_screening_to_expert_parallel_routes_preview.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original = pd.read_excel(INPUT_XLSX, sheet_name="All_Candidates")
    selected_original = original.loc[original["Extrusion_AR"].isin(AR_VALUES)].copy()
    duplicate_count = int(selected_original.duplicated(subset=KEY_COLS).sum())
    candidates = selected_original.drop_duplicates(subset=KEY_COLS).copy().reset_index(drop=True)
    candidates["Pred_EL"] = candidates["Pred_EL"] + EL_ADJUSTMENT
    candidate_count = validate_grid(candidates)

    candidates["UTS_norm"] = safe_minmax(candidates["Pred_UTS"])
    candidates["YS_norm"] = safe_minmax(candidates["Pred_YS"])
    candidates["EL_norm"] = safe_minmax(candidates["Pred_EL"])
    candidates["CPI"] = (
        CPI_WEIGHTS["UTS"] * candidates["UTS_norm"]
        + CPI_WEIGHTS["YS"] * candidates["YS_norm"]
        + CPI_WEIGHTS["EL"] * candidates["EL_norm"]
    )
    ranked = candidates.sort_values("CPI", ascending=False).reset_index(drop=True)
    top_1000 = ranked.head(1000).copy()

    # Audit the roadmap selection before plotting.
    top_key_tuples = set(map(tuple, top_1000[KEY_COLS].to_numpy()))
    nlargest_key_tuples = set(map(tuple, candidates.nlargest(1000, "CPI")[KEY_COLS].to_numpy()))
    cutoff_cpi = float(top_1000.iloc[-1]["CPI"])
    best_outside_cpi = float(ranked.iloc[1000]["CPI"])
    validation = pd.DataFrame(
        [
            ["Top-N requested", 1000, True],
            ["Rows selected", len(top_1000), len(top_1000) == 1000],
            ["Unique composition-process rows", top_1000.drop_duplicates(subset=KEY_COLS).shape[0], top_1000.drop_duplicates(subset=KEY_COLS).shape[0] == 1000],
            ["CPI monotonically decreasing", bool(top_1000["CPI"].is_monotonic_decreasing), bool(top_1000["CPI"].is_monotonic_decreasing)],
            ["Matches pandas nlargest(1000)", top_key_tuples == nlargest_key_tuples, top_key_tuples == nlargest_key_tuples],
            ["Rank-1000 CPI cutoff", cutoff_cpi, True],
            ["Best CPI outside Top-1000", best_outside_cpi, best_outside_cpi <= cutoff_cpi],
        ],
        columns=["Check", "Value", "Passed"],
    )
    if not validation["Passed"].astype(bool).all():
        raise ValueError("Top-1000 roadmap validation failed")

    global_best = ranked.iloc[0].copy()
    expert_260_20 = select_row(candidates, 0.25, 0.2, 2.3, 260, 20)
    expert_240_25 = select_row(candidates, 0.25, 0.2, 2.3, 240, 25)

    pred_min = candidates[PRED_COLS].min()
    pred_max = candidates[PRED_COLS].max()
    key_rows = []
    for label, source, row in [
        ("Global CPI optimum", "Current 60,480-candidate ranking", global_best),
        ("Expert: 260 °C / AR 20", "Current model recalculation", expert_260_20),
        ("Expert: 240 °C / AR 25", "Current model recalculation", expert_240_25),
    ]:
        key_rows.append(
            {
                "Candidate": label,
                "Prediction_source": source,
                **{col: float(row[col]) for col in KEY_COLS},
                **{col: float(row[col]) for col in PRED_COLS + NORM_COLS + ["CPI"]},
            }
        )
    key_df = pd.DataFrame(key_rows)

    candidates.to_csv(OUT_DIR / "all_candidates_unique.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(OUT_DIR / "ranked_by_CPI.csv", index=False, encoding="utf-8-sig")
    top_1000.to_csv(OUT_DIR / "top_1000_by_CPI.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUT_DIR / "top_1000_validation.csv", index=False, encoding="utf-8-sig")
    key_df.to_csv(OUT_DIR / "key_candidates.csv", index=False, encoding="utf-8-sig")

    parameters = pd.DataFrame(
        [
            ["Unique candidates", candidate_count],
            ["Duplicates removed", duplicate_count],
            ["CPI weight: UTS", CPI_WEIGHTS["UTS"]],
            ["CPI weight: YS", CPI_WEIGHTS["YS"]],
            ["CPI weight: EL", CPI_WEIGHTS["EL"]],
            ["EL prediction adjustment", EL_ADJUSTMENT],
            ["UTS minimum", pred_min["Pred_UTS"]],
            ["UTS maximum", pred_max["Pred_UTS"]],
            ["YS minimum", pred_min["Pred_YS"]],
            ["YS maximum", pred_max["Pred_YS"]],
            ["EL minimum", pred_min["Pred_EL"]],
            ["EL maximum", pred_max["Pred_EL"]],
        ],
        columns=["Parameter", "Value"],
    )
    parameters.to_csv(OUT_DIR / "screening_parameters.csv", index=False, encoding="utf-8-sig")

    plot_cpi_space(candidates, global_best, expert_260_20, expert_240_25)
    plot_parallel_routes(candidates, expert_260_20, expert_240_25)

    print(f"Original rows: {len(original):,}")
    print(f"Duplicates removed: {duplicate_count:,}")
    print(f"Unique candidates: {len(candidates):,}")
    print("\nKey candidates:")
    print(key_df.round(4).to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
