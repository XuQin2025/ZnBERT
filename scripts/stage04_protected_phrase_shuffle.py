# -*- coding: utf-8 -*-
"""Protected-phrase order ablation for ZnBERT + XGBoost.

The permutation is intentionally intermediate between semantic-stage shuffling
and complete WordPiece shuffling. Numeric meaning-bearing phrases remain
intact, while their positions and selected grammar tokens are randomized.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

import stage04_semantic_order_ablation as base


OUT_DIR = base.OUT_DIR
CACHE_DIR = OUT_DIR / "protected_phrase_embedding_cache"
RUN_PATH = OUT_DIR / "protected_phrase_ablation_runs.csv"
SUMMARY_PATH = OUT_DIR / "protected_phrase_ablation_summary.csv"
DEFAULT_SEEDS = (101, 202, 303, 404, 505)

PROTECTED_COLOR = "#14967F"
PROTECTED_LIGHT = "#D9F3ED"
FULL_COLOR = "#7C3AED"
FULL_LIGHT = "#EDE9FE"
FEATURE_COLOR = "#D69E2E"
FEATURE_LIGHT = "#FFF3CD"


def build_accelerated_xgb() -> MultiOutputRegressor:
    """Match the reference XGB hyperparameters, using CUDA only as a compute backend."""
    estimator = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=base.RANDOM_SEED,
        n_jobs=8,
        tree_method="hist",
        device="cuda",
        verbosity=0,
    )
    return MultiOutputRegressor(estimator, n_jobs=1)


def composition_chunks(row: pd.Series) -> list[str]:
    facts = base.composition_facts(row)
    if facts:
        return ["Zn-based", "alloy", "with", *facts]
    return ["Zn-based", "alloy"]


def processing_chunks(row: pd.Series) -> list[str]:
    """Build protected process chunks such as 'Extrusion T=300C' and 'AR=100'."""
    if (
        "Processing" in row
        and isinstance(row["Processing"], str)
        and row["Processing"].strip() == "Casting"
    ):
        return ["Casting"]

    chunks: list[str] = []

    def value(column: str):
        return base.safe_get(row, column)

    if row.get("has_homogen") == 1:
        temp, time_h = value("homogen_T"), value("homogen_t_h")
        chunks.append(
            "Homogenization" if temp is None else f"Homogenization T={int(float(temp))}C"
        )
        if time_h is not None:
            chunks.append(f"t={float(time_h):g}h")

    if row.get("has_extrusion") == 1:
        temp, area_ratio, speed = value("extrusion_T"), value("extrusion_AR"), value("extrusion_S")
        chunks.append("Extrusion" if temp is None else f"Extrusion T={int(float(temp))}C")
        if area_ratio is not None:
            chunks.append(f"AR={int(float(area_ratio))}")
        if speed is not None:
            chunks.append(f"S={float(speed):g}")

    if row.get("has_rolling") == 1:
        temp, reduction = value("rolling_T"), value("rolling_AR_total")
        chunks.append("Rolling" if temp is None else f"Rolling T={int(float(temp))}C")
        if reduction is not None:
            chunks.append(f"AR_total={int(float(reduction))}%")

    if row.get("has_MDF") == 1:
        passes = value("mdf_passes")
        chunks.append("MDF")
        if passes is not None:
            chunks.append(f"passes={int(float(passes))}")

    if row.get("has_ECAP") == 1:
        temp, passes = value("ecap_T"), value("ecap_passes")
        chunks.append("ECAP" if temp is None else f"ECAP T={int(float(temp))}C")
        if passes is not None:
            chunks.append(f"passes={int(float(passes))}")

    if row.get("has_anneal") == 1:
        temp, time_h = value("anneal_T"), value("anneal_t_h")
        chunks.append("Anneal" if temp is None else f"Anneal T={int(float(temp))}C")
        if time_h is not None:
            chunks.append(f"t={float(time_h):g}h")

    if row.get("has_wiredraw") == 1:
        chunks.append("WireDrawing")

    if row.get("has_HPT") == 1:
        passes = value("hpt_PASS")
        chunks.append("HPT")
        if passes is not None:
            chunks.append(f"passes={int(float(passes))}")

    if not chunks and "Processing" in row and isinstance(row["Processing"], str):
        text = row["Processing"].strip().rstrip(".")
        if text:
            chunks.append(text)
    return chunks or ["No processing info"]


def make_protected_text(df: pd.DataFrame, shuffle_seed: int) -> pd.DataFrame:
    comp_sentences: list[str] = []
    proc_sentences: list[str] = []
    comp_changed: list[bool] = []
    proc_changed: list[bool] = []
    comp_chunk_counts: list[int] = []
    proc_chunk_counts: list[int] = []

    for row_index, row in df.iterrows():
        original_comp = composition_chunks(row)
        original_proc = processing_chunks(row)
        comp_rng = np.random.default_rng(shuffle_seed * 1_000_003 + row_index * 2)
        proc_rng = np.random.default_rng(shuffle_seed * 1_000_003 + row_index * 2 + 1)
        shuffled_comp = base.nonidentity_permutation(original_comp, comp_rng)
        shuffled_proc = base.nonidentity_permutation(original_proc, proc_rng)
        comp_sentences.append(" ".join(shuffled_comp) + ".")
        proc_sentences.append(" ".join(shuffled_proc) + ".")
        comp_changed.append(shuffled_comp != original_comp)
        proc_changed.append(shuffled_proc != original_proc)
        comp_chunk_counts.append(len(original_comp))
        proc_chunk_counts.append(len(original_proc))

    return pd.DataFrame(
        {
            "CompSentence": comp_sentences,
            "ProcSentence": proc_sentences,
            "CompOrderChanged": comp_changed,
            "ProcOrderChanged": proc_changed,
            "CompositionChunkCount": comp_chunk_counts,
            "ProcessingChunkCount": proc_chunk_counts,
        }
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    result = (
        metrics.groupby("target", sort=False)
        .agg(
            protected_R2_mean=("R2", "mean"),
            protected_R2_sd=("R2", "std"),
            protected_R2_min=("R2", "min"),
            protected_R2_max=("R2", "max"),
            protected_RMSE_mean=("RMSE", "mean"),
            protected_MAE_mean=("MAE", "mean"),
            n_randomizations=("shuffle_seed", "nunique"),
            any_order_changed_fraction=("any_order_changed_fraction", "mean"),
        )
        .reindex(base.Y_COLS)
        .reset_index()
    )
    result["reported_original_R2"] = result["target"].map(base.REPORTED_ZNBERT_XGB)
    result["reported_pure_ml_R2"] = result["target"].map(base.REPORTED_PURE_ML_XGB)
    result["delta_protected_minus_original"] = result["protected_R2_mean"] - result["reported_original_R2"]
    return result


def plot_five_condition_comparison(metrics: pd.DataFrame) -> None:
    feature_summary = pd.read_csv(OUT_DIR / "xgb_feature_order_ablation_summary.csv").set_index("target").reindex(base.Y_COLS)
    protected = metrics.groupby("target")["R2"].agg(["mean", "std"]).reindex(base.Y_COLS)

    targets = base.Y_COLS
    original = np.array([base.REPORTED_ZNBERT_XGB[t] for t in targets])
    protected_mean = protected["mean"].to_numpy()
    pure_ml = np.array([base.REPORTED_PURE_ML_XGB[t] for t in targets])
    feature = feature_summary["mapped_shuffled_R2_used_in_figure"].to_numpy()

    base.configure_plot_style()
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 21, "xtick.labelsize": 18, "ytick.labelsize": 16, "legend.fontsize": 15})
    fig, ax = plt.subplots(figsize=(16.8, 8.8), facecolor="white")
    x = np.arange(len(targets), dtype=float)
    width = 0.18
    series = [
        (x - 1.5 * width, original, None, base.COLORS["original"], "Original text order · ZnBERT + XGB"),
        (x - 0.5 * width, protected_mean, None, PROTECTED_COLOR, "Protected-phrase shuffle · ZnBERT + XGB"),
        (x + 0.5 * width, pure_ml, None, base.COLORS["pure_ml"], "Pure ML · XGB"),
        (x + 1.5 * width, feature, None, FEATURE_COLOR, "Feature-column shuffle · XGB"),
    ]
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
            error_kw={"elinewidth": 1.8, "capthick": 1.8},
            label=label,
            zorder=3,
        )
        heights = values if errors is None else values + errors
        for bar, value, height in zip(bars, values, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.011,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=13.5,
                fontweight="bold",
                color=base.COLORS["ink"],
            )

    ax.set_ylim(0.45, 0.91)
    ax.set_xticks(x, targets)
    ax.set_ylabel(r"5-fold out-of-fold $R^2$")
    ax.set_title("Context sensitivity across controlled text- and feature-order ablations", fontsize=23, fontweight="bold", pad=16)
    ax.grid(axis="y", color=base.COLORS["grid"], linewidth=1.0, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(base.COLORS["grid"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, columnspacing=1.6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.28)
    stem = OUT_DIR / "five_condition_context_ablation_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "five_condition_context_ablation_comparison_preview.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_chip(ax, x: float, y: float, width: float, text: str, edge: str, fill: str, fontsize: float = 12.5) -> None:
    chip = FancyBboxPatch(
        (x, y), width, 0.09,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.7,
    )
    ax.add_patch(chip)
    ax.text(x + width / 2, y + 0.045, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=base.COLORS["ink"])


def draw_sequence(ax, x0: float, x1: float, y: float, items: list[str], edge: str, fill: str) -> None:
    weights = np.array([max(4.0, len(item) * 0.72) for item in items], dtype=float)
    gap = 0.006
    usable = x1 - x0 - gap * (len(items) - 1)
    widths = usable * weights / weights.sum()
    x = x0
    for item, width in zip(items, widths):
        add_chip(ax, x, y, float(width), item, edge, fill, fontsize=11.5 if len(items) > 6 else 12.5)
        x += float(width) + gap


def plot_shuffle_schematic() -> None:
    base.configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.2), facecolor="white")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    rows = [(0.74, "Original order"), (0.45, "Protected-phrase shuffle"), (0.16, "Complete token shuffle")]
    axes[0].text(0.5, 0.95, "Composition branch", ha="center", va="center", fontsize=23, fontweight="bold", color=base.COLORS["ink"])
    axes[1].text(0.5, 0.95, "Processing branch", ha="center", va="center", fontsize=23, fontweight="bold", color=base.COLORS["ink"])

    comp_rows = [
        ["Zn-based", "alloy", "with", "2.5 wt% Mg", "1.6 wt% Li"],
        ["1.6 wt% Li", "Zn-based", "with", "2.5 wt% Mg", "alloy"],
        ["wt%", "Li", "2.5", "Zn-based", "Mg", "1.6", "alloy", "wt%", "with"],
    ]
    proc_rows = [
        ["Homogenization T=350C", "t=24h", "Extrusion T=300C", "AR=100"],
        ["AR=100", "Homogenization T=350C", "Extrusion T=300C", "t=24h"],
        ["T=300C", "AR=100", "Homogenization", "t=24h", "Extrusion", "T=350C"],
    ]
    colors = [
        (base.COLORS["original"], base.COLORS["light_blue"]),
        (PROTECTED_COLOR, PROTECTED_LIGHT),
        (FULL_COLOR, FULL_LIGHT),
    ]
    for row_i, (y, label) in enumerate(rows):
        for ax in axes:
            ax.text(0.02, y + 0.12, label, ha="left", va="bottom", fontsize=16.5, fontweight="bold", color=base.COLORS["ink"])
        edge, fill = colors[row_i]
        draw_sequence(axes[0], 0.02, 0.98, y, comp_rows[row_i], edge, fill)
        draw_sequence(axes[1], 0.02, 0.98, y, proc_rows[row_i], edge, fill)

    for ax in axes:
        for y0, y1, color in [(0.73, 0.57, PROTECTED_COLOR), (0.44, 0.28, FULL_COLOR)]:
            ax.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=color))

    fig.text(
        0.5,
        0.025,
        "Protected phrases preserve value–unit–variable meaning; complete token shuffle destroys these local associations",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=base.COLORS["ink"],
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.09, wspace=0.08)
    stem = OUT_DIR / "protected_phrase_shuffle_schematic"
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "protected_phrase_shuffle_schematic_preview.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--skip-fit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.seed_everything()
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        base.build_xgb = build_accelerated_xgb
        print("[BACKEND] XGBoost CUDA; reference hyperparameters unchanged", flush=True)

    if args.skip_fit:
        metrics = pd.read_csv(RUN_PATH)
    else:
        df = pd.read_excel(base.DATA_XLSX, sheet_name=base.SHEET)
        for column in base.COMPOSITION_COLS + base.Y_COLS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=base.Y_COLS).reset_index(drop=True)
        y = df[base.Y_COLS].to_numpy(dtype=float)
        tokenizer = AutoTokenizer.from_pretrained(base.ZNBERT_PATH)
        encoder = AutoModel.from_pretrained(base.ZNBERT_PATH).to(base.DEVICE).eval()
        rows: list[dict] = []
        predictions: dict[str, np.ndarray] = {}

        for seed in args.shuffle_seeds:
            print(f"[RUN] protected-phrase shuffle seed={seed}", flush=True)
            text = make_protected_text(df, seed)
            comp_embedding = base.encode_texts(
                text["CompSentence"].tolist(), tokenizer, encoder,
                CACHE_DIR / f"comp_protected_seed_{seed}.npy",
            )
            proc_embedding = base.encode_texts(
                text["ProcSentence"].tolist(), tokenizer, encoder,
                CACHE_DIR / f"proc_protected_seed_{seed}.npy",
            )
            x_data = np.hstack([comp_embedding, proc_embedding]).astype(np.float32)
            y_pred = base.oof_predict(x_data, y)
            predictions[f"seed_{seed}"] = y_pred
            comp_changed = float(text["CompOrderChanged"].mean())
            proc_changed = float(text["ProcOrderChanged"].mean())
            any_changed = float((text["CompOrderChanged"] | text["ProcOrderChanged"]).mean())
            rows.extend(base.metric_rows("Protected phrase shuffle", seed, y, y_pred, comp_changed, proc_changed, any_changed))
            print(f"    changed composition={comp_changed:.1%}, processing={proc_changed:.1%}, any={any_changed:.1%}", flush=True)
            gc.collect()

        metrics = pd.DataFrame(rows)
        metrics.to_csv(RUN_PATH, index=False, encoding="utf-8-sig")
        np.savez_compressed(OUT_DIR / "protected_phrase_ablation_predictions.npz", y_true=y, **predictions)
        example = make_protected_text(df, args.shuffle_seeds[0])
        original = base.make_text_columns(df, None)
        pd.DataFrame(
            {
                "original_composition": original["CompSentence"],
                "protected_shuffled_composition": example["CompSentence"],
                "original_processing": original["ProcSentence"],
                "protected_shuffled_processing": example["ProcSentence"],
                "composition_order_changed": example["CompOrderChanged"],
                "processing_order_changed": example["ProcOrderChanged"],
            }
        ).to_csv(OUT_DIR / "protected_phrase_text_audit_seed_101.csv", index=False, encoding="utf-8-sig")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(metrics)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    plot_five_condition_comparison(metrics)
    plot_shuffle_schematic()
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
