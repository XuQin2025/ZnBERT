# -*- coding: utf-8 -*-
"""
Word-order ablation for ZnBERT + XGBoost.

The ablation changes only the order of self-contained composition facts and
processing-stage blocks. Numeric values, labels, samples, CV folds, encoder,
pooling, and XGBoost hyperparameters are held fixed.

Outputs are written to ``shuffle_order_analysis``:
  - order_ablation_runs.csv
  - order_ablation_summary.csv
  - order_ablation_predictions.npz
  - order_ablation_comparison.{png,pdf}
  - order_shuffle_schematic.{png,pdf}
"""

from __future__ import annotations

import argparse
import gc
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from transformers import AutoModel, AutoTokenizer


DATA_XLSX = Path("Zn-NLP_norm_structured.xlsx")
SHEET = "Sheet1"
ZNBERT_PATH = Path("ZnBERTv2_8epoch_DAPT_Abstract/checkpoint-1200")
OUT_DIR = Path("shuffle_order_analysis")
CACHE_DIR = OUT_DIR / "embedding_cache"

RANDOM_SEED = 42
DEFAULT_SHUFFLE_SEEDS = (101, 202, 303, 404, 505)
N_SPLITS = 5
BATCH_SIZE = 64
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Y_COLS = ["UTS", "YS", "EL"]

# Values supplied by the user. The final item in each six-model array is XGB.
REPORTED_ZNBERT_XGB = {"UTS": 0.849, "YS": 0.823, "EL": 0.826}
REPORTED_PURE_ML_XGB = {"UTS": 0.814, "YS": 0.796, "EL": 0.782}

COMPOSITION_COLS = [
    "Mg (wt%)", "Mn (wt%)", "Ag (wt%)", "Li (wt%)", "Ca (wt%)",
    "Cu(wt%)", "Sr(wt%)", "Zr(wt%)", "Fe(wt%)", "Al(wt%)",
    "Ti(wt%)", "Nd(wt%)", "Gd(wt%)", "Sc(wt%)", "Er(wt%)",
    "Dy(wt%)", "Ho(wt%)",
]

COLORS = {
    "original": "#1D4ED8",
    "shuffled": "#E76F51",
    "pure_ml": "#4B5563",
    "accent": "#0F766E",
    "light_blue": "#DBEAFE",
    "light_orange": "#FFEDD5",
    "light_green": "#CCFBF1",
    "ink": "#172033",
    "grid": "#D9DEE8",
}


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_get(row: pd.Series, column: str):
    value = row.get(column)
    return None if pd.isna(value) else value


def composition_facts(row: pd.Series) -> list[str]:
    facts: list[str] = []
    for column in COMPOSITION_COLS:
        value = row.get(column)
        if pd.isna(value):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if abs(value) < 1e-12:
            continue
        element = column.replace("(wt%)", "").replace(" (wt%)", "").strip()
        facts.append(f"{value:g} wt% {element}")
    return facts


def processing_stages(row: pd.Series) -> list[str]:
    if (
        "Processing" in row
        and isinstance(row["Processing"], str)
        and row["Processing"].strip() == "Casting"
    ):
        return ["Casting"]

    stages: list[str] = []
    if row.get("has_homogen") == 1:
        temp, time_h = safe_get(row, "homogen_T"), safe_get(row, "homogen_t_h")
        stage = "Homogenization"
        if temp is not None:
            stage += f" T={int(float(temp))}C"
        if time_h is not None:
            stage += f" t={float(time_h):g}h"
        stages.append(stage)

    if row.get("has_extrusion") == 1:
        temp = safe_get(row, "extrusion_T")
        area_ratio = safe_get(row, "extrusion_AR")
        speed = safe_get(row, "extrusion_S")
        stage = "Extrusion"
        if temp is not None:
            stage += f" T={int(float(temp))}C"
        if area_ratio is not None:
            stage += f" AR={int(float(area_ratio))}"
        if speed is not None:
            stage += f" S={float(speed):g}"
        stages.append(stage)

    if row.get("has_rolling") == 1:
        temp = safe_get(row, "rolling_T")
        area_reduction = safe_get(row, "rolling_AR_total")
        stage = "Rolling"
        if temp is not None:
            stage += f" T={int(float(temp))}C"
        if area_reduction is not None:
            stage += f" AR_total={int(float(area_reduction))}%"
        stages.append(stage)

    if row.get("has_MDF") == 1:
        passes = safe_get(row, "mdf_passes")
        stages.append("MDF" + (f" passes={int(float(passes))}" if passes is not None else ""))

    if row.get("has_ECAP") == 1:
        temp, passes = safe_get(row, "ecap_T"), safe_get(row, "ecap_passes")
        stage = "ECAP"
        if temp is not None:
            stage += f" T={int(float(temp))}C"
        if passes is not None:
            stage += f" passes={int(float(passes))}"
        stages.append(stage)

    if row.get("has_anneal") == 1:
        temp, time_h = safe_get(row, "anneal_T"), safe_get(row, "anneal_t_h")
        stage = "Anneal"
        if temp is not None:
            stage += f" T={int(float(temp))}C"
        if time_h is not None:
            stage += f" t={float(time_h):g}h"
        stages.append(stage)

    if row.get("has_wiredraw") == 1:
        stages.append("WireDrawing")

    if row.get("has_HPT") == 1:
        passes = safe_get(row, "hpt_PASS")
        stages.append("HPT" + (f" passes={int(float(passes))}" if passes is not None else ""))

    return stages


def nonidentity_permutation(items: list[str], rng: np.random.Generator) -> list[str]:
    """Randomly permute every reorderable list while guaranteeing a change."""
    if len(items) < 2:
        return list(items)
    permutation = rng.permutation(len(items))
    while np.array_equal(permutation, np.arange(len(items))):
        permutation = rng.permutation(len(items))
    return [items[index] for index in permutation]


def render_composition(facts: list[str]) -> str:
    if not facts:
        return "Zn-based alloy."
    return "Zn-based alloy with " + ", ".join(facts) + "."


def render_processing(stages: list[str]) -> str:
    if not stages:
        return "No processing info."
    return "; ".join(stages) + "."


def make_text_columns(df: pd.DataFrame, shuffle_seed: int | None) -> pd.DataFrame:
    """Create original or per-sample randomly reordered text."""
    comp_sentences: list[str] = []
    proc_sentences: list[str] = []
    comp_changed: list[bool] = []
    proc_changed: list[bool] = []

    for row_index, row in df.iterrows():
        comp = composition_facts(row)
        proc = processing_stages(row)
        original_comp = list(comp)
        original_proc = list(proc)

        if shuffle_seed is not None:
            # Independent, reproducible sample-level orders for composition and process.
            comp_rng = np.random.default_rng(shuffle_seed * 1_000_003 + row_index * 2)
            proc_rng = np.random.default_rng(shuffle_seed * 1_000_003 + row_index * 2 + 1)
            comp = nonidentity_permutation(comp, comp_rng)
            proc = nonidentity_permutation(proc, proc_rng)

        comp_sentences.append(render_composition(comp))
        proc_sentences.append(render_processing(proc))
        comp_changed.append(comp != original_comp)
        proc_changed.append(proc != original_proc)

    return pd.DataFrame(
        {
            "CompSentence": comp_sentences,
            "ProcSentence": proc_sentences,
            "CompOrderChanged": comp_changed,
            "ProcOrderChanged": proc_changed,
        }
    )


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"[CACHE] {cache_path}")
            return cached

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())

    array = np.vstack(embeddings).astype(np.float32)
    np.save(cache_path, array)
    return array


def build_xgb() -> MultiOutputRegressor:
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=8,
        tree_method="hist",
        verbosity=0,
    )
    # Sequential targets avoid Windows joblib process-spawn permission issues;
    # each XGBoost fit still uses eight CPU threads.
    return MultiOutputRegressor(base, n_jobs=1)


def oof_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.zeros_like(y, dtype=float)
    estimator = build_xgb()
    for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        model = clone(estimator)
        model.fit(X[train_index], y[train_index])
        predictions[test_index] = model.predict(X[test_index])
        print(f"    fold {fold}/{N_SPLITS}")
    return predictions


def metric_rows(
    condition: str,
    shuffle_seed: int | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    comp_changed_fraction: float,
    proc_changed_fraction: float,
    any_changed_fraction: float,
) -> list[dict]:
    rows: list[dict] = []
    for target_index, target in enumerate(Y_COLS):
        actual = y_true[:, target_index]
        predicted = y_pred[:, target_index]
        rows.append(
            {
                "condition": condition,
                "shuffle_seed": shuffle_seed,
                "target": target,
                "R2": r2_score(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "MAE": mean_absolute_error(actual, predicted),
                "composition_order_changed_fraction": comp_changed_fraction,
                "processing_order_changed_fraction": proc_changed_fraction,
                "any_order_changed_fraction": any_changed_fraction,
            }
        )
    return rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "axes.unicode_minus": False,
        }
    )


def plot_comparison(run_metrics: pd.DataFrame) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    x_positions = np.arange(len(Y_COLS), dtype=float)
    width = 0.23

    shuffled = run_metrics[run_metrics["condition"] == "Randomized order"]
    shuffled_summary = (
        shuffled.groupby("target")["R2"]
        .agg(["mean", "std"])
        .reindex(Y_COLS)
    )
    original_values = np.array([REPORTED_ZNBERT_XGB[target] for target in Y_COLS])
    pure_ml_values = np.array([REPORTED_PURE_ML_XGB[target] for target in Y_COLS])
    shuffled_means = shuffled_summary["mean"].to_numpy()
    shuffled_stds = shuffled_summary["std"].fillna(0).to_numpy()

    bars_original = ax.bar(
        x_positions - width,
        original_values,
        width,
        color=COLORS["original"],
        edgecolor="white",
        linewidth=1.2,
        label="ZnBERT + XGB · original order",
        zorder=3,
    )
    bars_shuffled = ax.bar(
        x_positions,
        shuffled_means,
        width,
        yerr=shuffled_stds,
        capsize=6,
        color=COLORS["shuffled"],
        edgecolor="white",
        linewidth=1.2,
        error_kw={"elinewidth": 2, "capthick": 2},
        label=f"ZnBERT + XGB · randomized order (n={shuffled['shuffle_seed'].nunique()})",
        zorder=3,
    )
    bars_ml = ax.bar(
        x_positions + width,
        pure_ml_values,
        width,
        color=COLORS["pure_ml"],
        edgecolor="white",
        linewidth=1.2,
        label="Pure ML · XGB",
        zorder=3,
    )

    # Show every randomization run so the variability is visible, not hidden.
    jitter = np.linspace(-0.055, 0.055, shuffled["shuffle_seed"].nunique())
    for target_index, target in enumerate(Y_COLS):
        values = shuffled.loc[shuffled["target"] == target, "R2"].to_numpy()
        ax.scatter(
            np.full(len(values), x_positions[target_index]) + jitter[: len(values)],
            values,
            s=35,
            facecolor="white",
            edgecolor=COLORS["ink"],
            linewidth=1.1,
            zorder=5,
        )

    def label_bars(bars, values, label_heights=None, offset=0.010):
        if label_heights is None:
            label_heights = values
        for bar, value, label_height in zip(bars, values, label_heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_height + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=COLORS["ink"],
            )

    label_bars(bars_original, original_values)
    label_bars(
        bars_shuffled,
        shuffled_means,
        label_heights=shuffled_means + shuffled_stds,
    )
    label_bars(bars_ml, pure_ml_values)

    all_values = np.concatenate([original_values, pure_ml_values, shuffled["R2"].to_numpy()])
    lower = max(-0.05, min(0.55, float(all_values.min()) - 0.08))
    upper = min(1.0, max(0.90, float(all_values.max()) + 0.06))
    ax.set_ylim(lower, upper)
    ax.set_xticks(x_positions, Y_COLS)
    ax.set_ylabel(r"5-fold out-of-fold $R^2$")
    ax.set_title("Does semantic-block order affect ZnBERT + XGBoost prediction?")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1.0, alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28)
    fig.savefig(OUT_DIR / "order_ablation_comparison.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "order_ablation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 15,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["ink"],
    )


def plot_schematic() -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.94,
        "Original order",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    ax.text(
        0.03,
        0.55,
        "Randomized order",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )

    original_comp = [
        ("Zn-based alloy", 0.16),
        ("2.5 wt% Mg", 0.15),
        ("1.6 wt% Li", 0.14),
    ]
    shuffled_comp = [
        ("Zn-based alloy", 0.16),
        ("1.6 wt% Li", 0.14),
        ("2.5 wt% Mg", 0.15),
    ]
    original_proc = [
        ("Homogenization\nT=350°C · t=24 h", 0.25),
        ("Extrusion\nT=300°C · AR=100", 0.23),
    ]
    shuffled_proc = [
        ("Extrusion\nT=300°C · AR=100", 0.23),
        ("Homogenization\nT=350°C · t=24 h", 0.25),
    ]

    def draw_row(y, composition, processing):
        x = 0.03
        for label, width in composition:
            add_box(
                ax, x, y, width, 0.115, label,
                COLORS["light_blue"], COLORS["original"], fontsize=15,
            )
            x += width + 0.012
        ax.text(x + 0.004, y + 0.057, "+", fontsize=26, fontweight="bold", va="center")
        x += 0.045
        for label, width in processing:
            add_box(
                ax, x, y, width, 0.115, label,
                COLORS["light_orange"], COLORS["shuffled"], fontsize=14,
            )
            x += width + 0.012

    draw_row(0.75, original_comp, original_proc)
    draw_row(0.36, shuffled_comp, shuffled_proc)

    # Crossing arrows emphasize permutation while preserving each complete fact.
    for start_x, end_x in [(0.27, 0.42), (0.42, 0.27)]:
        arrow = FancyArrowPatch(
            (start_x, 0.735),
            (end_x, 0.49),
            connectionstyle="arc3,rad=0.18",
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=COLORS["accent"],
            alpha=0.9,
        )
        ax.add_patch(arrow)
    for start_x, end_x in [(0.62, 0.84), (0.86, 0.61)]:
        arrow = FancyArrowPatch(
            (start_x, 0.735),
            (end_x, 0.49),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=COLORS["accent"],
            alpha=0.9,
        )
        ax.add_patch(arrow)

    ax.annotate(
        "",
        xy=(0.88, 0.15),
        xytext=(0.12, 0.15),
        arrowprops=dict(arrowstyle="-|>", linewidth=3.0, color=COLORS["ink"]),
    )
    for x, label in zip(
        [0.16, 0.39, 0.62, 0.84],
        ["ZnBERT", "Mean pooling", "XGBoost", "UTS · YS · EL"],
    ):
        ax.text(
            x,
            0.19,
            label,
            fontsize=18,
            fontweight="bold",
            ha="center",
            color=COLORS["ink"],
        )
    ax.text(
        0.50,
        0.055,
        "Only block order changes · values, samples, CV folds, encoder and regressor stay fixed",
        fontsize=16,
        ha="center",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "order_shuffle_schematic.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "order_shuffle_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    randomized = run_metrics[run_metrics["condition"] == "Randomized order"]
    summary = (
        randomized.groupby("target", sort=False)
        .agg(
            shuffled_R2_mean=("R2", "mean"),
            shuffled_R2_sd=("R2", "std"),
            shuffled_R2_min=("R2", "min"),
            shuffled_R2_max=("R2", "max"),
            shuffled_RMSE_mean=("RMSE", "mean"),
            shuffled_MAE_mean=("MAE", "mean"),
            n_randomizations=("shuffle_seed", "nunique"),
            any_order_changed_fraction=("any_order_changed_fraction", "mean"),
        )
        .reindex(Y_COLS)
        .reset_index()
    )
    summary["reported_original_R2"] = summary["target"].map(REPORTED_ZNBERT_XGB)
    summary["reported_pure_ml_R2"] = summary["target"].map(REPORTED_PURE_ML_XGB)
    summary["delta_shuffled_minus_original"] = (
        summary["shuffled_R2_mean"] - summary["reported_original_R2"]
    )
    summary["delta_original_minus_pure_ml"] = (
        summary["reported_original_R2"] - summary["reported_pure_ml_R2"]
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shuffle-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SHUFFLE_SEEDS),
        help="Independent text-order randomization seeds.",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Regenerate figures from an existing order_ablation_runs.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"[INFO] device={DEVICE}; shuffle_seeds={args.shuffle_seeds}")

    run_path = OUT_DIR / "order_ablation_runs.csv"
    if args.skip_fit:
        run_metrics = pd.read_csv(run_path)
    else:
        df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
        for column in COMPOSITION_COLS + Y_COLS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=Y_COLS).reset_index(drop=True)
        y = df[Y_COLS].to_numpy(dtype=float)

        tokenizer = AutoTokenizer.from_pretrained(ZNBERT_PATH)
        encoder = AutoModel.from_pretrained(ZNBERT_PATH).to(DEVICE).eval()

        rows: list[dict] = []
        predictions: dict[str, np.ndarray] = {}
        for shuffle_seed in args.shuffle_seeds:
            print(f"[RUN] randomized order seed={shuffle_seed}")
            text = make_text_columns(df, shuffle_seed)
            comp_cache = CACHE_DIR / f"comp_randomized_seed_{shuffle_seed}.npy"
            proc_cache = CACHE_DIR / f"proc_randomized_seed_{shuffle_seed}.npy"
            comp_embedding = encode_texts(
                text["CompSentence"].tolist(), tokenizer, encoder, comp_cache
            )
            proc_embedding = encode_texts(
                text["ProcSentence"].tolist(), tokenizer, encoder, proc_cache
            )
            X = np.hstack([comp_embedding, proc_embedding]).astype(np.float32)
            y_pred = oof_predict(X, y)
            predictions[f"seed_{shuffle_seed}"] = y_pred

            comp_changed = float(text["CompOrderChanged"].mean())
            proc_changed = float(text["ProcOrderChanged"].mean())
            any_changed = float(
                (text["CompOrderChanged"] | text["ProcOrderChanged"]).mean()
            )
            rows.extend(
                metric_rows(
                    "Randomized order",
                    shuffle_seed,
                    y,
                    y_pred,
                    comp_changed,
                    proc_changed,
                    any_changed,
                )
            )
            print(
                f"    changed: composition={comp_changed:.1%}, "
                f"processing={proc_changed:.1%}, any={any_changed:.1%}"
            )
            gc.collect()

        run_metrics = pd.DataFrame(rows)
        run_metrics.to_csv(run_path, index=False, encoding="utf-8-sig")
        np.savez_compressed(
            OUT_DIR / "order_ablation_predictions.npz",
            y_true=y,
            **predictions,
        )

        # Store representative original/shuffled strings for auditability.
        examples = make_text_columns(df, args.shuffle_seeds[0])
        original_examples = make_text_columns(df, None)
        audit = pd.DataFrame(
            {
                "original_composition": original_examples["CompSentence"],
                "shuffled_composition": examples["CompSentence"],
                "original_processing": original_examples["ProcSentence"],
                "shuffled_processing": examples["ProcSentence"],
                "composition_order_changed": examples["CompOrderChanged"],
                "processing_order_changed": examples["ProcOrderChanged"],
            }
        )
        audit.to_csv(
            OUT_DIR / "order_ablation_text_audit_seed_101.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_runs(run_metrics)
    summary.to_csv(
        OUT_DIR / "order_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_comparison(run_metrics)
    plot_schematic()
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] outputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
