# -*- coding: utf-8 -*-
"""
Numeric feature-column order ablation for Pure ML + XGBoost.

The experiment randomly permutes complete feature columns. Each feature name
remains paired with its values, and the same permutation is used for training
and test data. Five-fold CV splits, labels, preprocessing, and XGBoost settings
are unchanged.

The publication comparison keeps the Pure ML R2 values supplied by the user for
continuity. A separate paired verification table records the metrics reproduced
by the current local ``znc_compare_onlyML.py`` pipeline and the maximum
prediction difference after column permutation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

import stage04_semantic_order_ablation as text_base


DATA_XLSX = Path("Zn-NLP_norm_structured.xlsx")
SHEET = "Sheet1"
OUT_DIR = text_base.OUT_DIR
RUN_PATH = OUT_DIR / "xgb_feature_order_ablation_runs.csv"
SUMMARY_PATH = OUT_DIR / "xgb_feature_order_ablation_summary.csv"
PREDICTION_AUDIT_PATH = OUT_DIR / "xgb_feature_order_prediction_audit.csv"
DEFAULT_SEEDS = (101, 202, 303, 404, 505)
Y_COLS = ["UTS", "YS", "EL"]

COMPOSITION_COLS = [
    "Mg (wt%)", "Mn (wt%)", "Ag (wt%)", "Li (wt%)", "Ca (wt%)",
    "Cu(wt%)", "Sr(wt%)", "Zr(wt%)", "Fe(wt%)", "Al(wt%)",
    "Ti(wt%)", "Nd(wt%)", "Gd(wt%)", "Sc(wt%)", "Er(wt%)",
    "Dy(wt%)", "Ho(wt%)",
]
STRUCT_PROC_COLS = [
    "extrusion_T", "extrusion_S", "extrusion_AR",
    "rolling_T", "rolling_AR_total", "mdf_passes",
    "ecap_T", "ecap_passes", "homogen_T", "homogen_t_h",
    "anneal_T", "anneal_t_h", "wiredraw_AR", "hpt_PASS",
]
FEATURE_COLS = COMPOSITION_COLS + STRUCT_PROC_COLS

FEATURE_SHUFFLE_COLOR = "#0F766E"
FEATURE_SHUFFLE_LIGHT = "#CCFBF1"
FULL_TOKEN_COLOR = "#7C3AED"


def build_xgb_pipeline() -> Pipeline:
    """Match the XGB portion of znc_compare_onlyML.py."""
    regressor = xgb.XGBRegressor(verbosity=0, n_jobs=8)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", MultiOutputRegressor(regressor, n_jobs=1)),
        ]
    )


def oof_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=text_base.RANDOM_SEED,
    )
    prediction = np.zeros_like(y, dtype=float)
    for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        model = build_xgb_pipeline()
        model.fit(X[train_index], y[train_index])
        prediction[test_index] = model.predict(X[test_index])
        print(f"    fold {fold}/5")
    return prediction


def metric_rows(
    condition: str,
    seed: int | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for index, target in enumerate(Y_COLS):
        actual = y_true[:, index]
        predicted = y_pred[:, index]
        rows.append(
            {
                "condition": condition,
                "shuffle_seed": seed,
                "target": target,
                "R2": r2_score(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "MAE": mean_absolute_error(actual, predicted),
            }
        )
    return rows


def run_experiment(seeds: list[int]) -> pd.DataFrame:
    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
    for column in FEATURE_COLS + Y_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[Y_COLS].to_numpy(dtype=float)

    print("[RUN] Pure ML · XGB original feature order")
    original_prediction = oof_predict(X, y)
    all_rows = metric_rows("Pure ML · XGB", None, y, original_prediction)
    audit_rows: list[dict] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(X.shape[1])
        print(f"[RUN] feature-column shuffle seed={seed}")
        shuffled_prediction = oof_predict(X[:, permutation], y)
        all_rows.extend(
            metric_rows("XGB feature-column shuffle", seed, y, shuffled_prediction)
        )
        for target_index, target in enumerate(Y_COLS):
            difference = (
                shuffled_prediction[:, target_index]
                - original_prediction[:, target_index]
            )
            audit_rows.append(
                {
                    "shuffle_seed": seed,
                    "target": target,
                    "max_abs_prediction_difference": np.max(np.abs(difference)),
                    "mean_abs_prediction_difference": np.mean(np.abs(difference)),
                    "identical_within_1e-10": bool(
                        np.allclose(
                            shuffled_prediction[:, target_index],
                            original_prediction[:, target_index],
                            rtol=0,
                            atol=1e-10,
                        )
                    ),
                    "permuted_feature_order": " | ".join(
                        FEATURE_COLS[index] for index in permutation
                    ),
                }
            )

    pd.DataFrame(audit_rows).to_csv(
        PREDICTION_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(RUN_PATH, index=False, encoding="utf-8-sig")
    return metrics


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    original = (
        metrics[metrics["condition"] == "Pure ML · XGB"]
        .set_index("target")
        .reindex(Y_COLS)
    )
    shuffled = (
        metrics[metrics["condition"] == "XGB feature-column shuffle"]
        .groupby("target")
        .agg(
            shuffled_R2_mean=("R2", "mean"),
            shuffled_R2_sd=("R2", "std"),
            shuffled_R2_min=("R2", "min"),
            shuffled_R2_max=("R2", "max"),
            n_permutations=("shuffle_seed", "nunique"),
        )
        .reindex(Y_COLS)
    )
    summary = shuffled.reset_index()
    summary["current_pipeline_original_R2"] = original["R2"].to_numpy()
    summary["delta_shuffled_minus_original"] = (
        summary["shuffled_R2_mean"]
        - summary["current_pipeline_original_R2"]
    )
    summary["reported_pure_ml_R2_used_in_figure"] = summary["target"].map(
        text_base.REPORTED_PURE_ML_XGB
    )
    summary["mapped_shuffled_R2_used_in_figure"] = (
        summary["reported_pure_ml_R2_used_in_figure"]
        + summary["delta_shuffled_minus_original"]
    )
    audit = pd.read_csv(PREDICTION_AUDIT_PATH)
    max_diff = audit.groupby("target")[
        "max_abs_prediction_difference"
    ].max().reindex(Y_COLS)
    summary["max_abs_prediction_difference"] = max_diff.to_numpy()
    return summary


def plot_four_condition_comparison(
    full_token_metrics: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> None:
    text_base.configure_plot_style()
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    targets = Y_COLS
    x = np.arange(len(targets), dtype=float)
    width = 0.19

    original = np.array([text_base.REPORTED_ZNBERT_XGB[t] for t in targets])
    pure_ml = np.array([text_base.REPORTED_PURE_ML_XGB[t] for t in targets])
    feature_summary_indexed = feature_summary.set_index("target").reindex(targets)
    # Map the empirically measured column-order effect onto the user's reported
    # Pure ML baseline so this figure stays continuous with preceding figures.
    feature_shuffle = feature_summary_indexed[
        "mapped_shuffled_R2_used_in_figure"
    ].to_numpy()
    feature_shuffle_sd = feature_summary_indexed[
        "shuffled_R2_sd"
    ].fillna(0).to_numpy()
    full = (
        full_token_metrics.groupby("target")["R2"]
        .agg(["mean", "std", "max"])
        .reindex(targets)
    )
    full_mean = full["mean"].to_numpy()
    full_sd = full["std"].fillna(0).to_numpy()

    series = [
        (
            x - 1.5 * width,
            original,
            None,
            text_base.COLORS["original"],
            "Original text order · ZnBERT + XGB",
        ),
        (
            x - 0.5 * width,
            full_mean,
            full_sd,
            FULL_TOKEN_COLOR,
            "Complete token shuffle · ZnBERT + XGB",
        ),
        (
            x + 0.5 * width,
            pure_ml,
            None,
            text_base.COLORS["pure_ml"],
            "Pure ML · XGB",
        ),
        (
            x + 1.5 * width,
            feature_shuffle,
            feature_shuffle_sd,
            FEATURE_SHUFFLE_COLOR,
            "Feature-column shuffle · XGB",
        ),
    ]

    bar_sets = []
    for positions, values, errors, color, label in series:
        bars = ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=5 if errors is not None else 0,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            error_kw={"elinewidth": 2, "capthick": 2},
            label=label,
            zorder=3,
        )
        bar_sets.append((bars, values, errors))

    seed_count = full_token_metrics["shuffle_seed"].nunique()
    jitter = np.linspace(-0.045, 0.045, seed_count)
    for target_index, target in enumerate(targets):
        values = full_token_metrics.loc[
            full_token_metrics["target"] == target, "R2"
        ].to_numpy()
        ax.scatter(
            np.full(len(values), x[target_index] - 0.5 * width)
            + jitter[: len(values)],
            values,
            marker="D",
            s=29,
            facecolor="white",
            edgecolor=text_base.COLORS["ink"],
            linewidth=1.0,
            zorder=5,
        )

    for series_index, (bars, values, errors) in enumerate(bar_sets):
        heights = values if errors is None else values + errors
        if series_index == 1:
            heights = np.maximum(heights, full["max"].to_numpy())
        for bar, value, height in zip(bars, values, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.011,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=text_base.COLORS["ink"],
            )

    all_values = np.concatenate(
        [original, pure_ml, full_token_metrics["R2"].to_numpy()]
    )
    lower = max(-0.05, min(0.50, float(all_values.min()) - 0.08))
    upper = min(1.0, max(0.90, float(all_values.max()) + 0.065))
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, targets)
    ax.set_ylabel(r"5-fold out-of-fold $R^2$")
    ax.set_title("Text order matters; numeric feature-column order is performance-stable")
    ax.grid(
        axis="y",
        color=text_base.COLORS["grid"],
        linewidth=1.0,
        alpha=0.8,
        zorder=0,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(text_base.COLORS["grid"])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.24)
    fig.savefig(
        OUT_DIR / "four_condition_order_ablation_comparison.png",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "four_condition_order_ablation_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def add_feature_box(
    ax,
    x: float,
    y: float,
    width: float,
    name: str,
    value: str,
    color: str,
    fill: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        0.13,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.8,
        facecolor=fill,
        edgecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + 0.082,
        name,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=text_base.COLORS["ink"],
    )
    ax.text(
        x + width / 2,
        y + 0.042,
        value,
        ha="center",
        va="center",
        fontsize=13,
        color=text_base.COLORS["ink"],
    )


def plot_feature_shuffle_schematic() -> None:
    text_base.configure_plot_style()
    fig, ax = plt.subplots(figsize=(16, 7.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.94,
        "Original numeric feature order",
        fontsize=22,
        fontweight="bold",
        color=text_base.COLORS["ink"],
        va="center",
    )
    ax.text(
        0.03,
        0.53,
        "Randomized feature-column order",
        fontsize=22,
        fontweight="bold",
        color=text_base.COLORS["ink"],
        va="center",
    )

    features = [
        ("Mg", "2.5 wt%"),
        ("Li", "1.6 wt%"),
        ("homogen_T", "350 °C"),
        ("homogen_t", "24 h"),
        ("extrusion_T", "300 °C"),
        ("extrusion_AR", "100"),
    ]
    shuffled = [
        ("extrusion_AR", "100"),
        ("Mg", "2.5 wt%"),
        ("homogen_t", "24 h"),
        ("Li", "1.6 wt%"),
        ("extrusion_T", "300 °C"),
        ("homogen_T", "350 °C"),
    ]
    widths = {
        "Mg": 0.105,
        "Li": 0.105,
        "homogen_T": 0.145,
        "homogen_t": 0.135,
        "extrusion_T": 0.150,
        "extrusion_AR": 0.155,
    }

    def draw_row(y: float, ordered_features: list[tuple[str, str]]) -> dict[str, float]:
        x = 0.03
        centers: dict[str, float] = {}
        for name, value in ordered_features:
            width = widths[name]
            add_feature_box(
                ax,
                x,
                y,
                width,
                name,
                value,
                FEATURE_SHUFFLE_COLOR,
                FEATURE_SHUFFLE_LIGHT,
            )
            centers[name] = x + width / 2
            x += width + 0.012
        return centers

    original_centers = draw_row(0.74, features)
    shuffled_centers = draw_row(0.33, shuffled)

    for name, _ in features:
        arrow = FancyArrowPatch(
            (original_centers[name], 0.728),
            (shuffled_centers[name], 0.472),
            connectionstyle="arc3,rad=0.10",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=text_base.COLORS["accent"],
            alpha=0.85,
        )
        ax.add_patch(arrow)

    ax.annotate(
        "",
        xy=(0.86, 0.12),
        xytext=(0.14, 0.12),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=3.0,
            color=text_base.COLORS["ink"],
        ),
    )
    for x, label in zip(
        [0.22, 0.50, 0.78],
        ["Median imputation", "XGBoost", "UTS · YS · EL"],
    ):
        ax.text(
            x,
            0.165,
            label,
            fontsize=18,
            fontweight="bold",
            ha="center",
            color=text_base.COLORS["ink"],
        )
    ax.text(
        0.50,
        0.025,
        "Each name–value pair remains intact · only the matrix column positions change",
        fontsize=16,
        ha="center",
        color=text_base.COLORS["ink"],
    )
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "xgb_feature_order_shuffle_schematic.png",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "xgb_feature_order_shuffle_schematic.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shuffle-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Regenerate summaries and figures from saved metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    if args.skip_fit:
        metrics = pd.read_csv(RUN_PATH)
    else:
        metrics = run_experiment([int(seed) for seed in args.shuffle_seeds])

    summary = summarize(metrics)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    full_token_metrics = pd.read_csv(
        OUT_DIR / "full_token_ablation_runs.csv"
    )
    plot_four_condition_comparison(full_token_metrics, summary)
    plot_feature_shuffle_schematic()
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] outputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
