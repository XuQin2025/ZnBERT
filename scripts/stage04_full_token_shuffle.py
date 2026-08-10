# -*- coding: utf-8 -*-
"""
Complete token-order ablation for ZnBERT + XGBoost.

Every non-special WordPiece token is randomly permuted inside the composition
branch and processing branch independently. The token multiset, numeric values,
special tokens, samples, CV folds, ZnBERT, pooling, and XGBoost are unchanged.

This script imports the shared data/model helpers from
``znbert_xgb_word_order_ablation.py`` and writes resumable outputs to
``shuffle_order_analysis``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from transformers import AutoModel, AutoTokenizer

import stage04_semantic_order_ablation as base


OUT_DIR = base.OUT_DIR
CACHE_DIR = OUT_DIR / "full_token_embedding_cache"
RUN_PATH = OUT_DIR / "full_token_ablation_runs.csv"
SUMMARY_PATH = OUT_DIR / "full_token_ablation_summary.csv"
DEFAULT_SEEDS = (101, 202, 303, 404, 505)

FULL_COLOR = "#7C3AED"
FULL_LIGHT = "#EDE9FE"
BLOCK_COLOR = base.COLORS["shuffled"]


def nonidentity_index_permutation(
    length: int, rng: np.random.Generator
) -> np.ndarray:
    if length < 2:
        return np.arange(length)
    permutation = rng.permutation(length)
    while np.array_equal(permutation, np.arange(length)):
        permutation = rng.permutation(length)
    return permutation


@torch.inference_mode()
def encode_fully_shuffled_tokens(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    cache_path: Path,
    shuffle_seed: int,
    branch_offset: int,
) -> np.ndarray:
    """Shuffle all content-token IDs while keeping special tokens fixed."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"[CACHE] {cache_path}")
            return cached

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), base.BATCH_SIZE):
        batch = texts[start : start + base.BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=base.MAX_LEN,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        input_ids = encoded["input_ids"].clone()
        attention_mask = encoded["attention_mask"]
        special_mask = encoded.pop("special_tokens_mask")

        for local_index in range(input_ids.shape[0]):
            global_index = start + local_index
            content_positions = torch.where(
                (attention_mask[local_index] == 1)
                & (special_mask[local_index] == 0)
            )[0]
            count = int(content_positions.numel())
            rng = np.random.default_rng(
                shuffle_seed * 1_000_003
                + global_index * 2
                + branch_offset
            )
            permutation = nonidentity_index_permutation(count, rng)
            original_ids = input_ids[local_index, content_positions].clone()
            input_ids[local_index, content_positions] = original_ids[
                torch.as_tensor(permutation, dtype=torch.long)
            ]

        encoded["input_ids"] = input_ids
        encoded = encoded.to(base.DEVICE)
        outputs = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())

    array = np.vstack(embeddings).astype(np.float32)
    np.save(cache_path, array)
    return array


def token_audit(
    texts: list[str],
    tokenizer: AutoTokenizer,
    shuffle_seed: int,
    branch_offset: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for index, text in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        rng = np.random.default_rng(
            shuffle_seed * 1_000_003 + index * 2 + branch_offset
        )
        permutation = nonidentity_index_permutation(len(tokens), rng)
        shuffled = [tokens[position] for position in permutation]
        rows.append(
            {
                "row_index": index,
                "original_text": text,
                "original_wordpieces": " | ".join(tokens),
                "fully_shuffled_wordpieces": " | ".join(shuffled),
                "token_count": len(tokens),
                "token_multiset_preserved": sorted(tokens) == sorted(shuffled),
            }
        )
    return pd.DataFrame(rows)


def completed_seeds(metrics: pd.DataFrame) -> set[int]:
    if metrics.empty:
        return set()
    counts = metrics.groupby("shuffle_seed")["target"].nunique()
    return {int(seed) for seed, count in counts.items() if count == len(base.Y_COLS)}


def full_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("target", sort=False)
        .agg(
            full_token_R2_mean=("R2", "mean"),
            full_token_R2_sd=("R2", "std"),
            full_token_R2_min=("R2", "min"),
            full_token_R2_max=("R2", "max"),
            full_token_RMSE_mean=("RMSE", "mean"),
            full_token_MAE_mean=("MAE", "mean"),
            n_randomizations=("shuffle_seed", "nunique"),
        )
        .reindex(base.Y_COLS)
        .reset_index()
    )
    summary["reported_original_R2"] = summary["target"].map(
        base.REPORTED_ZNBERT_XGB
    )
    summary["reported_pure_ml_R2"] = summary["target"].map(
        base.REPORTED_PURE_ML_XGB
    )
    summary["delta_full_minus_original"] = (
        summary["full_token_R2_mean"] - summary["reported_original_R2"]
    )
    summary["delta_full_minus_pure_ml"] = (
        summary["full_token_R2_mean"] - summary["reported_pure_ml_R2"]
    )
    return summary


def plot_four_way_comparison(
    block_metrics: pd.DataFrame, full_metrics: pd.DataFrame
) -> None:
    base.configure_plot_style()
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    targets = base.Y_COLS
    x = np.arange(len(targets), dtype=float)
    width = 0.19

    block = (
        block_metrics.groupby("target")["R2"]
        .agg(["mean", "std"])
        .reindex(targets)
    )
    full = (
        full_metrics.groupby("target")["R2"]
        .agg(["mean", "std"])
        .reindex(targets)
    )
    original = np.array([base.REPORTED_ZNBERT_XGB[t] for t in targets])
    pure_ml = np.array([base.REPORTED_PURE_ML_XGB[t] for t in targets])
    block_mean = block["mean"].to_numpy()
    block_sd = block["std"].fillna(0).to_numpy()
    full_mean = full["mean"].to_numpy()
    full_sd = full["std"].fillna(0).to_numpy()

    series = [
        (
            x - 1.5 * width,
            original,
            None,
            base.COLORS["original"],
            "Original order",
        ),
        (
            x - 0.5 * width,
            block_mean,
            block_sd,
            BLOCK_COLOR,
            "Semantic-block shuffle",
        ),
        (
            x + 0.5 * width,
            full_mean,
            full_sd,
            FULL_COLOR,
            "Complete WordPiece shuffle",
        ),
        (
            x + 1.5 * width,
            pure_ml,
            None,
            base.COLORS["pure_ml"],
            "Pure ML · XGB",
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

    full_seed_count = full_metrics["shuffle_seed"].nunique()
    full_jitter = np.linspace(-0.045, 0.045, full_seed_count)
    block_seed_count = block_metrics["shuffle_seed"].nunique()
    block_jitter = np.linspace(-0.04, 0.04, block_seed_count)
    for target_index, target in enumerate(targets):
        block_values = block_metrics.loc[
            block_metrics["target"] == target, "R2"
        ].to_numpy()
        ax.scatter(
            np.full(len(block_values), x[target_index] - 0.5 * width)
            + block_jitter[: len(block_values)],
            block_values,
            s=27,
            facecolor="white",
            edgecolor=base.COLORS["ink"],
            linewidth=1.0,
            zorder=5,
        )
        full_values = full_metrics.loc[
            full_metrics["target"] == target, "R2"
        ].to_numpy()
        ax.scatter(
            np.full(len(full_values), x[target_index] + 0.5 * width)
            + full_jitter[: len(full_values)],
            full_values,
            marker="D",
            s=29,
            facecolor="white",
            edgecolor=base.COLORS["ink"],
            linewidth=1.0,
            zorder=5,
        )

    for series_index, (bars, values, errors) in enumerate(bar_sets):
        heights = values if errors is None else values + errors
        if series_index == 2:
            run_maxima = (
                full_metrics.groupby("target")["R2"].max().reindex(targets).to_numpy()
            )
            heights = np.maximum(heights, run_maxima)
        for bar, value, height in zip(bars, values, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.010,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=base.COLORS["ink"],
            )

    all_values = np.concatenate(
        [original, pure_ml, block_metrics["R2"].to_numpy(), full_metrics["R2"].to_numpy()]
    )
    lower = max(-0.05, min(0.50, float(all_values.min()) - 0.08))
    upper = min(1.0, max(0.90, float(all_values.max()) + 0.065))
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, targets)
    ax.set_ylabel(r"5-fold out-of-fold $R^2$")
    ax.set_title("How much order destruction can ZnBERT + XGBoost tolerate?")
    ax.grid(axis="y", color=base.COLORS["grid"], linewidth=1.0, alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(base.COLORS["grid"])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.24)
    fig.savefig(OUT_DIR / "full_token_ablation_comparison.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "full_token_ablation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def add_token_box(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    light_color: str,
    width: float,
    fontsize: float = 13,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        0.105,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        facecolor=light_color,
        edgecolor=color,
        linewidth=1.7,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + 0.0525,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=base.COLORS["ink"],
    )


def plot_full_shuffle_schematic() -> None:
    base.configure_plot_style()
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03, 0.94, "Original token order",
        fontsize=22, fontweight="bold", color=base.COLORS["ink"], va="center",
    )
    ax.text(
        0.03, 0.54, "Complete random order",
        fontsize=22, fontweight="bold", color=base.COLORS["ink"], va="center",
    )

    comp_original = ["Zn-based", "alloy", "with", "2.5", "wt%", "Mg", "1.6", "wt%", "Li"]
    comp_shuffled = ["wt%", "Li", "2.5", "Zn-based", "Mg", "1.6", "alloy", "wt%", "with"]
    proc_original = ["Homogenization", "T=350°C", "t=24h", "Extrusion", "T=300°C", "AR=100"]
    proc_shuffled = ["T=300°C", "AR=100", "Homogenization", "t=24h", "Extrusion", "T=350°C"]

    def draw_lane(y: float, comp: list[str], proc: list[str]) -> None:
        x = 0.03
        comp_widths = {
            "Zn-based": 0.0656,
            "alloy": 0.0512,
            "with": 0.0440,
            "2.5": 0.0416,
            "wt%": 0.0416,
            "Mg": 0.0360,
            "1.6": 0.0416,
            "Li": 0.0320,
        }
        for token in comp:
            width = comp_widths[token]
            add_token_box(
                ax, x, y, token, base.COLORS["original"],
                base.COLORS["light_blue"], width,
            )
            x += width + 0.006
        ax.text(x + 0.006, y + 0.052, "+", fontsize=24, fontweight="bold", va="center")
        x += 0.038
        proc_widths = {
            "Homogenization": 0.0966,
            "T=350°C": 0.0630,
            "t=24h": 0.0538,
            "Extrusion": 0.0664,
            "T=300°C": 0.0630,
            "AR=100": 0.0563,
        }
        for token in proc:
            width = proc_widths[token]
            add_token_box(
                ax, x, y, token, FULL_COLOR, FULL_LIGHT, width, fontsize=12.5
            )
            x += width + 0.006

    draw_lane(0.75, comp_original, proc_original)
    draw_lane(0.35, comp_shuffled, proc_shuffled)

    for start_x, end_x in [
        (0.08, 0.25), (0.20, 0.05), (0.32, 0.39),
        (0.63, 0.82), (0.78, 0.59), (0.91, 0.72),
    ]:
        arrow = FancyArrowPatch(
            (start_x, 0.735),
            (end_x, 0.475),
            connectionstyle="arc3,rad=0.12",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=base.COLORS["accent"],
            alpha=0.85,
        )
        ax.add_patch(arrow)

    ax.annotate(
        "",
        xy=(0.88, 0.13),
        xytext=(0.12, 0.13),
        arrowprops=dict(arrowstyle="-|>", linewidth=3.0, color=base.COLORS["ink"]),
    )
    for x, label in zip(
        [0.16, 0.39, 0.62, 0.84],
        ["ZnBERT", "Mean pooling", "XGBoost", "UTS · YS · EL"],
    ):
        ax.text(
            x, 0.17, label, fontsize=18, fontweight="bold",
            ha="center", color=base.COLORS["ink"],
        )
    ax.text(
        0.50,
        0.035,
        "All non-special WordPiece tokens are permuted · token multiset and model pipeline stay fixed",
        fontsize=16,
        ha="center",
        color=base.COLORS["ink"],
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_token_shuffle_schematic.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "full_token_shuffle_schematic.pdf", bbox_inches="tight")
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
        help="Regenerate summaries and figures from existing run metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.seed_everything()
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    existing = pd.read_csv(RUN_PATH) if RUN_PATH.exists() else pd.DataFrame()
    done = completed_seeds(existing)
    requested = [int(seed) for seed in args.shuffle_seeds]
    pending = [] if args.skip_fit else [seed for seed in requested if seed not in done]
    print(f"[INFO] device={base.DEVICE}; completed={sorted(done)}; pending={pending}")

    if pending:
        df = pd.read_excel(base.DATA_XLSX, sheet_name=base.SHEET)
        for column in base.COMPOSITION_COLS + base.Y_COLS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=base.Y_COLS).reset_index(drop=True)
        y = df[base.Y_COLS].to_numpy(dtype=float)
        text = base.make_text_columns(df, None)

        tokenizer = AutoTokenizer.from_pretrained(base.ZNBERT_PATH)
        encoder = AutoModel.from_pretrained(base.ZNBERT_PATH).to(base.DEVICE).eval()

        metrics = existing.copy()
        for shuffle_seed in pending:
            print(f"[RUN] complete WordPiece shuffle seed={shuffle_seed}")
            comp_embedding = encode_fully_shuffled_tokens(
                text["CompSentence"].tolist(),
                tokenizer,
                encoder,
                CACHE_DIR / f"comp_full_token_seed_{shuffle_seed}.npy",
                shuffle_seed,
                branch_offset=0,
            )
            proc_embedding = encode_fully_shuffled_tokens(
                text["ProcSentence"].tolist(),
                tokenizer,
                encoder,
                CACHE_DIR / f"proc_full_token_seed_{shuffle_seed}.npy",
                shuffle_seed,
                branch_offset=1,
            )
            X = np.hstack([comp_embedding, proc_embedding]).astype(np.float32)
            prediction = base.oof_predict(X, y)
            np.save(
                OUT_DIR / f"full_token_predictions_seed_{shuffle_seed}.npy",
                prediction,
            )
            rows = base.metric_rows(
                "Complete WordPiece shuffle",
                shuffle_seed,
                y,
                prediction,
                1.0,
                1.0,
                1.0,
            )
            metrics = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)
            metrics.to_csv(RUN_PATH, index=False, encoding="utf-8-sig")
            print(
                "    "
                + ", ".join(
                    f"{row['target']} R2={row['R2']:.6f}" for row in rows
                )
            )
            gc.collect()

        comp_audit = token_audit(
            text["CompSentence"].tolist(), tokenizer, requested[0], branch_offset=0
        ).add_prefix("composition_")
        proc_audit = token_audit(
            text["ProcSentence"].tolist(), tokenizer, requested[0], branch_offset=1
        ).add_prefix("processing_")
        pd.concat([comp_audit, proc_audit], axis=1).to_csv(
            OUT_DIR / f"full_token_text_audit_seed_{requested[0]}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = pd.read_csv(RUN_PATH)
    metrics = metrics[metrics["shuffle_seed"].isin(requested)].copy()
    summary = full_summary(metrics)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    block_metrics = pd.read_csv(OUT_DIR / "order_ablation_runs.csv")
    plot_four_way_comparison(block_metrics, metrics)
    plot_full_shuffle_schematic()
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] outputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
