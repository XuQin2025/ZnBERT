# -*- coding: utf-8 -*-
"""
Build ZnBERT + XGB downstream models, draw parity plots, and estimate
prediction uncertainty for three requested Zn alloy candidates.

Uncertainty is now estimator/tree based:
  1. Train the same 5-fold OOF models used for validation.
  2. For every validation sample, compute every XGB tree's single-tree
     prediction as base_score + that tree's leaf contribution, then compute
     an uncalibrated tree-prediction sigma.
  3. Calibrate sigma with OOF residuals using sigma_cal = a * sigma_tree + b.
  4. Train the final best ensemble on all data and report calibrated
     estimator-based uncertainty for the three requested candidates.

The old 5-fold candidate prediction std is kept only as a reference column.
"""

import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - fallback for lean Python environments
    minimize = None


warnings.filterwarnings("ignore")

DATA_XLSX = "Zn-NLP_norm_structured.xlsx"
SHEET = "Sheet1"
ZNBERT_PATH = "ZnBERTv2_8epoch_DAPT_Abstract/checkpoint-1200"
CACHE_DIR = "cache_embeddings"
OUT_DIR = "ZnBERT_XGB_CV_Uncertainty"

RANDOM_SEED = 42
N_SPLITS = 5
PCA_COMPONENTS = 64
BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Y_COLS = ["UTS", "YS", "EL"]

COMPOSITION_COLS = [
    "Mg (wt%)", "Mn (wt%)", "Ag (wt%)", "Li (wt%)", "Ca (wt%)", "Cu(wt%)",
    "Sr(wt%)", "Zr(wt%)", "Fe(wt%)", "Al(wt%)", "Ti(wt%)", "Nd(wt%)",
    "Gd(wt%)", "Sc(wt%)", "Er(wt%)", "Dy(wt%)", "Ho(wt%)",
]

STRUCT_PROC_COLS = [
    "extrusion_T", "extrusion_AR", "extrusion_S", "rolling_T",
    "rolling_AR_total", "mdf_passes", "ecap_T", "ecap_passes",
    "homogen_T", "homogen_t_h", "anneal_T", "anneal_t_h", "wiredraw_AR",
    "hpt_PASS",
]

HAS_COLS = [
    "has_extrusion", "has_rolling", "has_ECAP", "has_MDF", "has_homogen",
    "has_anneal", "has_wiredraw", "has_HPT",
]

NUMERIC_FEATURE_COLS = COMPOSITION_COLS + STRUCT_PROC_COLS + HAS_COLS


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_comp_sentence(row, base="Zn"):
    parts = []
    for col in COMPOSITION_COLS:
        value = row.get(col)
        if pd.isna(value):
            continue
        value = float(value)
        if abs(value) < 1e-12:
            continue
        elem = col.replace("(wt%)", "").replace(" (wt%)", "").strip()
        parts.append(f"{value:g} wt% {elem}")
    if not parts:
        return f"{base}-based alloy."
    return f"{base}-based alloy with " + ", ".join(parts) + "."


def build_proc_sentence(row):
    if "Processing" in row and isinstance(row["Processing"], str):
        if row["Processing"].strip() == "Casting":
            return "Casting."

    def safe_get(col):
        value = row.get(col)
        return None if pd.isna(value) else value

    parts = []
    if row.get("has_homogen") == 1:
        temp, time_h = safe_get("homogen_T"), safe_get("homogen_t_h")
        seg = "Homogenization"
        if temp is not None and float(temp) != 0:
            seg += f" T={int(float(temp))}C"
        if time_h is not None and float(time_h) != 0:
            seg += f" t={float(time_h):g}h"
        parts.append(seg)

    if row.get("has_extrusion") == 1:
        temp, ratio, speed = safe_get("extrusion_T"), safe_get("extrusion_AR"), safe_get("extrusion_S")
        seg = "Extrusion"
        if temp is not None:
            seg += f" T={int(float(temp))}C"
        if ratio is not None:
            seg += f" AR={int(float(ratio))}"
        if speed is not None and abs(float(speed)) > 1e-12:
            seg += f" S={float(speed):g}"
        parts.append(seg)

    if row.get("has_rolling") == 1:
        temp, total_ar = safe_get("rolling_T"), safe_get("rolling_AR_total")
        seg = "Rolling"
        if temp is not None and float(temp) != 0:
            seg += f" T={int(float(temp))}C"
        if total_ar is not None and float(total_ar) != 0:
            seg += f" AR_total={int(float(total_ar))}%"
        parts.append(seg)

    if row.get("has_MDF") == 1:
        passes = safe_get("mdf_passes")
        parts.append("MDF" + (f" passes={int(float(passes))}" if passes else ""))

    if row.get("has_ECAP") == 1:
        temp, passes = safe_get("ecap_T"), safe_get("ecap_passes")
        seg = "ECAP"
        if temp is not None and float(temp) != 0:
            seg += f" T={int(float(temp))}C"
        if passes is not None and float(passes) != 0:
            seg += f" passes={int(float(passes))}"
        parts.append(seg)

    if row.get("has_anneal") == 1:
        temp, time_h = safe_get("anneal_T"), safe_get("anneal_t_h")
        seg = "Anneal"
        if temp is not None and float(temp) != 0:
            seg += f" T={int(float(temp))}C"
        if time_h is not None and float(time_h) != 0:
            seg += f" t={float(time_h):g}h"
        parts.append(seg)

    if row.get("has_wiredraw") == 1:
        parts.append("WireDrawing")

    if row.get("has_HPT") == 1:
        passes = safe_get("hpt_PASS")
        parts.append("HPT" + (f" passes={int(float(passes))}" if passes else ""))

    return "; ".join(parts) + "." if parts else "No processing info."


@torch.no_grad()
def encode_texts(texts):
    tokenizer = AutoTokenizer.from_pretrained(ZNBERT_PATH)
    model = AutoModel.from_pretrained(ZNBERT_PATH).to(DEVICE).eval()

    all_embs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def load_training_embeddings(df):
    comp_cache = os.path.join(CACHE_DIR, "comp_emb_ZnBERV2.npy")
    proc_cache = os.path.join(CACHE_DIR, "proc_emb_ZnBERTV2.npy")
    if os.path.exists(comp_cache) and os.path.exists(proc_cache):
        emb_comp = np.load(comp_cache)
        emb_proc = np.load(proc_cache)
        if len(emb_comp) == len(df) and len(emb_proc) == len(df):
            print("[INFO] Loaded cached ZnBERT embeddings.")
            return np.hstack([emb_comp, emb_proc]).astype(np.float32)

    print("[INFO] Cache missing or mismatched; encoding training texts.")
    emb_comp = encode_texts(df["CompSentence"].tolist())
    emb_proc = encode_texts(df["ProcSentence"].tolist())
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(comp_cache, emb_comp)
    np.save(proc_cache, emb_proc)
    return np.hstack([emb_comp, emb_proc]).astype(np.float32)


def make_candidate(name, mg, li, cu, mn, extrusion_t, extrusion_ar):
    row = {col: 0.0 for col in NUMERIC_FEATURE_COLS}
    row.update({
        "Alloy": name,
        "Mg (wt%)": mg,
        "Li (wt%)": li,
        "Cu(wt%)": cu,
        "Mn (wt%)": mn,
        "extrusion_T": extrusion_t,
        "extrusion_AR": extrusion_ar,
        "extrusion_S": 0.0,
        "has_extrusion": 1,
        "Processing": f"HotExtrusion (T={extrusion_t} degC; AR_total={extrusion_ar}_to_1)",
    })
    return row


def make_xgb(target):
    if target == "EL":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.035,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.5,
            min_child_weight=2,
            random_state=RANDOM_SEED,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
        )

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_weight=2,
        random_state=RANDOM_SEED,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


def get_sample_weight(df_part, target):
    if target != "EL":
        return None
    return (
        np.ones(len(df_part), dtype=float)
        + 1.5 * (df_part["EL"].values <= 15)
        + 0.8 * ((df_part["Mg (wt%)"].values > 0) & (df_part["Li (wt%)"].values > 0))
        + 0.5 * (df_part["has_extrusion"].values == 1)
    )


def transform_target(y, target):
    return np.log1p(y) if target == "EL" else y


def inverse_target(y_pred, target):
    return np.expm1(y_pred) if target == "EL" else y_pred


def build_fold_features(x_text, x_num, train_idx, test_idx, cand_text, cand_num):
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    imputer = SimpleImputer(strategy="median")

    train_text_scaled = scaler.fit_transform(x_text[train_idx])
    train_text_pca = pca.fit_transform(train_text_scaled)
    test_text_pca = pca.transform(scaler.transform(x_text[test_idx]))
    cand_text_pca = pca.transform(scaler.transform(cand_text))

    train_num = imputer.fit_transform(x_num[train_idx])
    test_num = imputer.transform(x_num[test_idx])
    cand_num_fold = imputer.transform(cand_num)

    x_train = np.hstack([train_text_pca, train_num]).astype(np.float32)
    x_test = np.hstack([test_text_pca, test_num]).astype(np.float32)
    x_cand = np.hstack([cand_text_pca, cand_num_fold]).astype(np.float32)
    return x_train, x_test, x_cand


def build_full_features(x_text, x_num, cand_text, cand_num):
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    imputer = SimpleImputer(strategy="median")

    train_text_scaled = scaler.fit_transform(x_text)
    train_text_pca = pca.fit_transform(train_text_scaled)
    cand_text_pca = pca.transform(scaler.transform(cand_text))

    train_num = imputer.fit_transform(x_num)
    cand_num_full = imputer.transform(cand_num)

    x_train = np.hstack([train_text_pca, train_num]).astype(np.float32)
    x_cand = np.hstack([cand_text_pca, cand_num_full]).astype(np.float32)
    return x_train, x_cand


def xgb_tree_contribution_matrix(model, x_data):
    """Return one leaf contribution per sample per boosted tree.

    For gradient-boosted XGB models, a tree is an additive residual component,
    not an independent full predictor. We convert each tree contribution to a
    single-tree prediction by adding the model base_score before taking std.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(x_data)
    leaves = booster.predict(dmat, pred_leaf=True)
    leaves = np.asarray(leaves)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)

    trees_df = booster.trees_to_dataframe()
    leaf_df = trees_df.loc[trees_df["Feature"] == "Leaf", ["Tree", "Node", "Gain"]]
    tree_maps = {
        int(tree_id): dict(zip(group["Node"].astype(int), group["Gain"].astype(float)))
        for tree_id, group in leaf_df.groupby("Tree")
    }

    contrib = np.zeros(leaves.shape, dtype=float)
    for tree_i in range(leaves.shape[1]):
        node_to_value = tree_maps.get(tree_i, {})
        contrib[:, tree_i] = [
            node_to_value.get(int(node_id), 0.0)
            for node_id in leaves[:, tree_i]
        ]
    return contrib


def xgb_base_score(model):
    """Read XGBoost base_score in raw-margin space."""
    config = json.loads(model.get_booster().save_config())
    base = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(base, str):
        base = base.strip()
        if base.startswith("["):
            return float(np.asarray(json.loads(base), dtype=float).ravel()[0])
        return float(base)
    return float(base)


def estimator_tree_sigma(model, x_data, target):
    """Compute uncalibrated std of per-tree prediction results."""
    contrib = xgb_tree_contribution_matrix(model, x_data)
    tree_pred_raw = xgb_base_score(model) + contrib
    if target == "EL":
        tree_pred = np.expm1(tree_pred_raw)
    else:
        tree_pred = tree_pred_raw
    sigma = tree_pred.std(axis=1, ddof=1)
    return sigma, tree_pred


def fit_sigma_calibration(y_true, y_pred, sigma_uncal):
    """Fit Palmer-style affine calibration sigma_cal = a*sigma_uncal + b."""
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    sigma_uncal = np.clip(np.asarray(sigma_uncal, dtype=float), 1e-9, None)
    rmse_value = float(np.sqrt(np.mean(residual ** 2)))

    def evaluate(a, b):
        sigma = np.clip(a * sigma_uncal + b, 1e-9, None)
        nll = 0.5 * np.mean(np.log(2 * np.pi * sigma ** 2) + (residual ** 2) / (sigma ** 2))
        r_stat = float(np.sqrt(np.mean((residual / sigma) ** 2)))
        coverage_1sigma = float(np.mean(np.abs(residual) <= sigma))
        coverage_95 = float(np.mean(np.abs(residual) <= 1.96 * sigma))
        return sigma, float(nll), r_stat, coverage_1sigma, coverage_95

    if minimize is None:
        a = rmse_value / max(float(np.median(sigma_uncal)), 1e-9)
        b = 0.0
        sigma, nll, r_stat, coverage_1sigma, coverage_95 = evaluate(a, b)
        return {
            "a": float(a),
            "b": float(b),
            "nll": nll,
            "r_stat": r_stat,
            "coverage_1sigma": coverage_1sigma,
            "coverage_95": coverage_95,
            "median_sigma_uncal": float(np.median(sigma_uncal)),
            "median_sigma_cal": float(np.median(sigma)),
            "rmse": rmse_value,
            "optimizer": "fallback_scale_only",
        }

    init_a = rmse_value / max(float(np.median(sigma_uncal)), 1e-9)
    init_b = max(0.05 * rmse_value, 1e-6)

    def objective(log_params):
        a, b = np.exp(log_params)
        _, nll, _, _, _ = evaluate(a, b)
        return nll

    result = minimize(
        objective,
        x0=np.log([max(init_a, 1e-6), init_b]),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
    )
    a, b = np.exp(result.x)
    sigma, nll, r_stat, coverage_1sigma, coverage_95 = evaluate(a, b)
    return {
        "a": float(a),
        "b": float(b),
        "nll": nll,
        "r_stat": r_stat,
        "coverage_1sigma": coverage_1sigma,
        "coverage_95": coverage_95,
        "median_sigma_uncal": float(np.median(sigma_uncal)),
        "median_sigma_cal": float(np.median(sigma)),
        "rmse": rmse_value,
        "optimizer": "scipy_nelder_mead",
    }


def plot_parity(y_true, y_pred, metrics_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    units = {"UTS": "MPa", "YS": "MPa", "EL": "%"}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=150)
    for i, target in enumerate(Y_COLS):
        ax = axes[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        row = metrics_df.loc[metrics_df["Target"] == target].iloc[0]

        ax.scatter(yt, yp, s=28, alpha=0.58, color="#4E79A7", edgecolors="white", linewidths=0.5)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.06 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.4, alpha=0.72)

        coef = np.polyfit(yt, yp, 1)
        xs = np.linspace(lo, hi, 120)
        ax.plot(xs, coef[0] * xs + coef[1], "--", color="#D62728", linewidth=2.0, alpha=0.9)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{target} - ZnBERT+XGB", fontweight="bold", fontsize=15)
        ax.set_xlabel(f"Measured {target} ({units[target]})")
        ax.set_ylabel(f"Predicted {target} ({units[target]})")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
        ax.text(
            0.05,
            0.94,
            f"R2={row['R2']:.3f}\nRMSE={row['RMSE']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.75", alpha=0.95),
        )

        single_fig = plt.figure(figsize=(5.2, 4.8), dpi=150)
        single_ax = single_fig.add_subplot(111)
        single_ax.scatter(yt, yp, s=30, alpha=0.6, color="#4E79A7", edgecolors="white", linewidths=0.5)
        single_ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.4, alpha=0.72)
        single_ax.plot(xs, coef[0] * xs + coef[1], "--", color="#D62728", linewidth=2.0, alpha=0.9)
        single_ax.set_xlim(lo, hi)
        single_ax.set_ylim(lo, hi)
        single_ax.set_title(f"{target} - ZnBERT+XGB", fontweight="bold", fontsize=15)
        single_ax.set_xlabel(f"Measured {target} ({units[target]})")
        single_ax.set_ylabel(f"Predicted {target} ({units[target]})")
        single_ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
        single_ax.text(
            0.05,
            0.94,
            f"R2={row['R2']:.3f}\nRMSE={row['RMSE']:.2f}",
            transform=single_ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.75", alpha=0.95),
        )
        single_fig.tight_layout()
        single_fig.savefig(os.path.join(out_dir, f"Parity_{target}_ZnBERT_XGB.png"), dpi=300, bbox_inches="tight")
        plt.close(single_fig)

    fig.tight_layout()
    combined = os.path.join(out_dir, "Best_ZnBERT_XGB_Parity_UTS_YS_EL.png")
    fig.savefig(combined, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return combined


def main():
    seed_everything(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] DEVICE = {DEVICE}")

    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
    for col in NUMERIC_FEATURE_COLS + Y_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    df["CompSentence"] = df.apply(build_comp_sentence, axis=1)
    df["ProcSentence"] = df.apply(build_proc_sentence, axis=1)

    x_text = load_training_embeddings(df)
    x_num = df[NUMERIC_FEATURE_COLS].values.astype(float)
    y = df[Y_COLS].values.astype(float)

    candidates = pd.DataFrame([
        make_candidate("Zn-0.25Mg-0.5Li (240C/25:1)", 0.25, 0.5, 0.0, 0.0, 240, 25),
        make_candidate("Zn-0.25Mg-0.2Li-2.3Cu (240C/25:1)", 0.25, 0.2, 2.3, 0.0, 240, 25),
        make_candidate("Zn-0.25Mg-0.2Li-0.8Mn (260C/20:1)", 0.25, 0.2, 0.0, 0.8, 260, 20),
    ])
    candidates["CompSentence"] = candidates.apply(build_comp_sentence, axis=1)
    candidates["ProcSentence"] = candidates.apply(build_proc_sentence, axis=1)
    cand_text = np.hstack([
        encode_texts(candidates["CompSentence"].tolist()),
        encode_texts(candidates["ProcSentence"].tolist()),
    ]).astype(np.float32)
    cand_num = candidates[NUMERIC_FEATURE_COLS].values.astype(float)

    oof = np.zeros_like(y, dtype=float)
    oof_tree_sigma_uncal = np.zeros_like(y, dtype=float)
    cand_fold_pred = {target: [] for target in Y_COLS}
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_text), start=1):
        print(f"[INFO] Fold {fold}/{N_SPLITS}")
        x_train, x_test, x_cand = build_fold_features(x_text, x_num, train_idx, test_idx, cand_text, cand_num)
        df_train = df.iloc[train_idx].reset_index(drop=True)

        for target_i, target in enumerate(Y_COLS):
            model = make_xgb(target)
            y_train_target = transform_target(y[train_idx, target_i], target)
            sample_weight = get_sample_weight(df_train, target)
            model.fit(x_train, y_train_target, sample_weight=sample_weight)
            pred_test = inverse_target(model.predict(x_test), target)
            pred_cand = inverse_target(model.predict(x_cand), target)
            sigma_test, _ = estimator_tree_sigma(model, x_test, target)
            oof[test_idx, target_i] = pred_test
            oof_tree_sigma_uncal[test_idx, target_i] = sigma_test
            cand_fold_pred[target].append(pred_cand)

    metrics = []
    for target_i, target in enumerate(Y_COLS):
        metrics.append({
            "Target": target,
            "R2": r2_score(y[:, target_i], oof[:, target_i]),
            "RMSE": rmse(y[:, target_i], oof[:, target_i]),
            "MAE": mean_absolute_error(y[:, target_i], oof[:, target_i]),
        })
    metrics_df = pd.DataFrame(metrics)
    print("\n[METRICS]")
    print(metrics_df.to_string(index=False))

    calibration = {}
    calibration_rows = []
    for target_i, target in enumerate(Y_COLS):
        cal = fit_sigma_calibration(y[:, target_i], oof[:, target_i], oof_tree_sigma_uncal[:, target_i])
        calibration[target] = cal
        calibration_rows.append({
            "Target": target,
            "calibration_formula": "sigma_cal = a * sigma_tree_uncal + b",
            "a": cal["a"],
            "b": cal["b"],
            "OOF_RMSE": cal["rmse"],
            "median_sigma_tree_uncal": cal["median_sigma_uncal"],
            "median_sigma_calibrated": cal["median_sigma_cal"],
            "r_stat_after_calibration": cal["r_stat"],
            "coverage_1sigma_after_calibration": cal["coverage_1sigma"],
            "coverage_95_after_calibration": cal["coverage_95"],
            "gaussian_NLL_after_calibration": cal["nll"],
            "optimizer": cal["optimizer"],
        })
    calibration_df = pd.DataFrame(calibration_rows)

    x_full, x_cand_full = build_full_features(x_text, x_num, cand_text, cand_num)

    uncertainty = candidates[[
        "Alloy", "Mg (wt%)", "Li (wt%)", "Cu(wt%)", "Mn (wt%)", "extrusion_T", "extrusion_AR",
        "CompSentence", "ProcSentence",
    ]].copy()

    fold_rows = []
    estimator_rows = []
    for target_i, target in enumerate(Y_COLS):
        pred_matrix = np.vstack(cand_fold_pred[target])
        fold_mean_pred = pred_matrix.mean(axis=0)
        fold_model_std = pred_matrix.std(axis=0, ddof=1)
        cv_rmse = float(metrics_df.loc[metrics_df["Target"] == target, "RMSE"].iloc[0])
        cal = calibration[target]

        full_model = make_xgb(target)
        y_full_target = transform_target(y[:, target_i], target)
        sample_weight = get_sample_weight(df, target)
        full_model.fit(x_full, y_full_target, sample_weight=sample_weight)
        best_pred = inverse_target(full_model.predict(x_cand_full), target)
        sigma_tree_uncal, tree_pred_matrix = estimator_tree_sigma(full_model, x_cand_full, target)
        sigma_tree_cal = cal["a"] * sigma_tree_uncal + cal["b"]

        uncertainty[f"Pred_{target}_best_ensemble"] = best_pred
        uncertainty[f"Pred_{target}_tree_std_uncalibrated"] = sigma_tree_uncal
        uncertainty[f"Pred_{target}_tree_std_calibrated_1sigma"] = sigma_tree_cal
        uncertainty[f"Pred_{target}_95CI_low"] = best_pred - 1.96 * sigma_tree_cal
        uncertainty[f"Pred_{target}_95CI_high"] = best_pred + 1.96 * sigma_tree_cal
        uncertainty[f"Pred_{target}_95CI_low_physical"] = np.maximum(
            0.0, uncertainty[f"Pred_{target}_95CI_low"].values
        )
        uncertainty[f"Pred_{target}_cv_fold_mean_reference"] = fold_mean_pred
        uncertainty[f"Pred_{target}_cv_fold_std_reference"] = fold_model_std
        uncertainty[f"Pred_{target}_cv_RMSE_reference"] = cv_rmse
        uncertainty[f"Pred_{target}_calibration_a"] = cal["a"]
        uncertainty[f"Pred_{target}_calibration_b"] = cal["b"]

        for fold_i in range(pred_matrix.shape[0]):
            for cand_i, alloy in enumerate(candidates["Alloy"]):
                fold_rows.append({
                    "Target": target,
                    "Fold": fold_i + 1,
                    "Alloy": alloy,
                    "Prediction": pred_matrix[fold_i, cand_i],
                })

        for cand_i, alloy in enumerate(candidates["Alloy"]):
            for tree_i, value in enumerate(tree_pred_matrix[cand_i], start=1):
                estimator_rows.append({
                    "Target": target,
                    "Alloy": alloy,
                    "Tree": tree_i,
                    "TreePrediction_target_units": value,
                    "BestEnsemblePrediction": best_pred[cand_i],
                    "TreeStdUncalibrated_target_units": sigma_tree_uncal[cand_i],
                    "TreeStdCalibrated_1sigma": sigma_tree_cal[cand_i],
                })

    oof_df = df[COMPOSITION_COLS + ["Processing", "CompSentence", "ProcSentence"] + Y_COLS].copy()
    for target_i, target in enumerate(Y_COLS):
        oof_df[f"OOF_Pred_{target}"] = oof[:, target_i]
        oof_df[f"OOF_AbsErr_{target}"] = np.abs(oof[:, target_i] - y[:, target_i])
        oof_df[f"OOF_TreeSigma_{target}_uncalibrated"] = oof_tree_sigma_uncal[:, target_i]
        cal = calibration[target]
        oof_df[f"OOF_TreeSigma_{target}_calibrated"] = (
            cal["a"] * oof_tree_sigma_uncal[:, target_i] + cal["b"]
        )

    combined_png = plot_parity(y, oof, metrics_df, OUT_DIR)
    out_xlsx = os.path.join(OUT_DIR, "ZnBERT_XGB_predictions_with_uncertainty.xlsx")
    notes = pd.DataFrame([
        {"item": "model", "value": "ZnBERT mean-pooled comp/proc embeddings + numeric composition/process features + XGB"},
        {"item": "validation", "value": f"{N_SPLITS}-fold KFold OOF"},
        {"item": "embedding_reduction", "value": f"Fold-local StandardScaler + PCA({PCA_COMPONENTS}) fitted only on each training fold"},
        {"item": "UTS_YS_targets", "value": "plain squared-error XGB"},
        {"item": "EL_target", "value": "log1p(EL) XGB with low-EL/Mg+Li/extrusion sample weights"},
        {"item": "uncertainty", "value": "Estimator/tree-based: compute each XGB tree's single-tree prediction as base_score + leaf contribution, compute tree_prediction_std_uncalibrated, then calibrate with OOF residuals as sigma_cal = a*tree_prediction_std+b; 95CI = best full-data ensemble prediction +/- 1.96*sigma_cal"},
        {"item": "fold_std_reference", "value": "5-fold candidate prediction std is retained only as a reference column and is not the primary uncertainty estimate"},
        {"item": "final_prediction_model", "value": "For candidate prediction, each target's best ensemble is retrained once on all available data using the same feature transform and target settings"},
        {"item": "candidate_extrusion_S", "value": "0/unknown; omitted from process sentence"},
        {"item": "combined_parity_png", "value": combined_png},
    ])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        uncertainty.to_excel(writer, sheet_name="candidate_uncertainty", index=False)
        metrics_df.to_excel(writer, sheet_name="cv_metrics", index=False)
        calibration_df.to_excel(writer, sheet_name="sigma_calibration", index=False)
        oof_df.to_excel(writer, sheet_name="oof_predictions", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="candidate_fold_preds", index=False)
        pd.DataFrame(estimator_rows).to_excel(writer, sheet_name="candidate_tree_predictions", index=False)
        notes.to_excel(writer, sheet_name="model_notes", index=False)

    print("\n[CANDIDATE UNCERTAINTY]")
    show_cols = ["Alloy"]
    for target in Y_COLS:
        show_cols += [f"Pred_{target}_best_ensemble", f"Pred_{target}_tree_std_calibrated_1sigma"]
    print(uncertainty[show_cols].to_string(index=False))
    print("\n[SIGMA CALIBRATION]")
    print(calibration_df.to_string(index=False))
    print(f"\n[OK] Saved Excel: {out_xlsx}")
    print(f"[OK] Saved parity: {combined_png}")


if __name__ == "__main__":
    main()
