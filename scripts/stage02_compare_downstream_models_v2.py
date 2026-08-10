# -*- coding: utf-8 -*-
"""
ZnBERT (all-text) + Downstream model comparison (6 models)
Upstream:
  - Composition -> text sentence -> ZnBERT embedding
  - Processing  -> text sentence -> ZnBERT embedding
  - Concatenate [emb_comp, emb_proc] -> downstream regressors -> predict UTS/YS/EL

Downstream models compared (6):
  XGB, RF, SVR, GBR, MLP, ET(ExtraTrees)

Outputs:
  1) metrics table (OOF) printed and exported to Excel
  2) ONLY parity plots for the best downstream model (OOF)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModel

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# =========================
# Config
# =========================
DATA_XLSX = "Zn-NLP_norm_structured.xlsx"
SHEET = "Sheet1"

# ZNBERT_PATH = "ZnSciBERT_MLM"   # your local ZnBERT directory
ZNBERT_PATH = "ZnBERTv2_8epoch_DAPT_Abstract/checkpoint-1200"
#ZNBERT_PATH = "alloymechanicalBERT/checkpoint-1200"
CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MAX_LEN = 128

Y_COLS = ["UTS", "YS", "EL"]

# Export
OUT_METRICS_XLSX = "Downstream_Compare_ZnBERTv2_metrics.xlsx"
OUT_BEST_PARITY_PNG = "Downstream_Compare_ALLOOYmBERT_Best_Parity.png"

# Fonts / style
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# Composition columns (wt.%)
COMPOSITION_COLS = [
    "Mg (wt%)", "Mn (wt%)", "Ag (wt%)", "Li (wt%)", "Ca (wt%)", "Cu(wt%)", "Sr(wt%)", "Zr(wt%)", "Fe(wt%)",
    "Al(wt%)", "Ti(wt%)", "Nd(wt%)", "Gd(wt%)", "Sc(wt%)", "Er(wt%)", "Dy(wt%)", "Ho(wt%)"
]

# =========================
# Helpers
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def build_comp_sentence(row, base="Zn"):
    parts = []
    for c in COMPOSITION_COLS:
        v = row.get(c)
        if pd.isna(v):
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if abs(v) < 1e-12:
            continue
        elem = c.replace("(wt%)", "").replace(" (wt%)", "").strip()
        parts.append(f"{v:g} wt% {elem}")
    if not parts:
        return f"{base}-based alloy."
    return f"{base}-based alloy with " + ", ".join(parts) + "."

def build_proc_sentence(row):
    if "Processing" in row and isinstance(row["Processing"], str) and row["Processing"].strip() == "Casting":
        return "Casting."
    parts = []

    def safe_get(c):
        v = row.get(c)
        return None if pd.isna(v) else v

    if row.get("has_homogen") == 1:
        T, t = safe_get("homogen_T"), safe_get("homogen_t_h")
        parts.append("Homogenization" + (f" T={int(T)}C" if T else "") + (f" t={t}h" if t else ""))
    if row.get("has_extrusion") == 1:
        T = safe_get("extrusion_T")
        AR = safe_get("extrusion_AR")
        S = safe_get("extrusion_S")  # ✅ NEW

        seg = "Extrusion"
        if T is not None:
            seg += f" T={int(float(T))}C"
        if AR is not None:
            seg += f" AR={int(float(AR))}"
        # ✅ NEW: extrusion speed (unit unknown, keep generic S=)
        # If your DB uses 0 to mean unknown, uncomment the 2 lines below:
        # if S is not None and abs(float(S)) < 1e-12:
        #     S = None
        if S is not None:
            seg += f" S={float(S):g}"
        parts.append(seg)

    if row.get("has_rolling") == 1:
        T, ARt = safe_get("rolling_T"), safe_get("rolling_AR_total")
        parts.append("Rolling" + (f" T={int(T)}C" if T else "") + (f" AR_total={int(ARt)}%" if ARt else ""))
    if row.get("has_MDF") == 1:
        p = safe_get("mdf_passes")
        if p:
            parts.append(f"MDF passes={int(p)}")
    if row.get("has_ECAP") == 1:
        T, p = safe_get("ecap_T"), safe_get("ecap_passes")
        parts.append("ECAP" + (f" T={int(T)}C" if T else "") + (f" passes={int(p)}" if p else ""))
    if row.get("has_anneal") == 1:
        T, t = safe_get("anneal_T"), safe_get("anneal_t_h")
        parts.append("Anneal" + (f" T={int(T)}C" if T else "") + (f" t={t}h" if t else ""))
    if row.get("has_wiredraw") == 1:
        parts.append("WireDrawing")
    if row.get("has_HPT") == 1:
        p = safe_get("hpt_PASS")
        parts.append("HPT" + (f" passes={int(p)}" if p else ""))

    return "; ".join(parts) + "." if parts else "No processing info."

@torch.no_grad()
def get_embeddings(texts, model_path, cache_path):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cache: {cache_path}")
        return np.load(cache_path)

    print(f"[INFO] Encoding texts with {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(emb.cpu().numpy())

    res = np.vstack(all_embs).astype(np.float32)
    np.save(cache_path, res)
    print(f"[INFO] Saved cache: {cache_path}")
    return res

def oof_predict(estimator, X, y, splitter):
    oof = np.zeros_like(y, dtype=float)
    for fold, (tr_idx, te_idx) in enumerate(splitter.split(X, np.zeros(len(X))), 1):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = est.predict(X[te_idx])
        print(f"[INFO] fold {fold} done.")
    return oof

def summarize_metrics(model_name, y_true, y_pred):
    rows = []
    r2s = []
    for i, tgt in enumerate(Y_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2v = r2_score(yt, yp)
        r2s.append(r2v)
        rows.append({
            "model": model_name,
            "target": tgt,
            "R2": r2v,
            "RMSE": rmse(yt, yp),
            "MAE": mean_absolute_error(yt, yp),
        })
    rows.append({
        "model": model_name,
        "target": "AVG",
        "R2": float(np.mean(r2s)),
        "RMSE": np.nan,
        "MAE": np.nan,
    })
    return pd.DataFrame(rows)

def metrics_to_wide(metrics_long: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide summary table:
      model | R2_UTS R2_YS R2_EL avgR2 | RMSE_UTS ... | MAE_UTS ...
    """
    df = metrics_long[metrics_long["target"].isin(Y_COLS)].copy()

    p_r2 = df.pivot(index="model", columns="target", values="R2").rename(columns=lambda c: f"R2_{c}")
    p_rm = df.pivot(index="model", columns="target", values="RMSE").rename(columns=lambda c: f"RMSE_{c}")
    p_ma = df.pivot(index="model", columns="target", values="MAE").rename(columns=lambda c: f"MAE_{c}")

    out = pd.concat([p_r2, p_rm, p_ma], axis=1)
    out["avgR2"] = out[[f"R2_{c}" for c in Y_COLS]].mean(axis=1)
    out = out.reset_index().sort_values("avgR2", ascending=False).reset_index(drop=True)
    return out

def plot_best_parity(y_true, y_pred, model_name, out_png):
    """
    Parity plots styled like your reference image:
      - blue transparent scatter with white edge
      - red dashed regression line + CI band
      - black dashed y=x line
      - R2/RMSE box at top-left
    Fix: use linewidths (NOT linewidth) to avoid matplotlib alias conflict.
    """
    units = {"UTS": "MPa", "YS": "MPa", "EL": "%"}

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=120)

    for i, tgt in enumerate(Y_COLS):
        ax = axes[i]
        yt = np.asarray(y_true[:, i], dtype=float)
        yp = np.asarray(y_pred[:, i], dtype=float)

        r2v = r2_score(yt, yp)
        rmsev = rmse(yt, yp)

        # regplot: scatter + regression + CI band
        sns.regplot(
            x=yt, y=yp, ax=ax,
            ci=95,
            scatter=True,
            scatter_kws=dict(
                s=28,
                alpha=0.55,
                color="#5DA5DA",
                edgecolor="white",
                linewidths=0.5,     # ✅ only linewidths
            ),
            line_kws=dict(
                color="red",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9
            )
        )

        # y=x dashed
        mn = float(min(yt.min(), yp.min()))
        mx = float(max(yt.max(), yp.max()))
        pad = 0.06 * (mx - mn + 1e-9)
        lo, hi = mn - pad, mx + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.4, alpha=0.7)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_title(f"{tgt} - {model_name}")
        ax.set_xlabel(f"Measured {tgt} ({units.get(tgt,'')})")
        ax.set_ylabel(f"Predicted {tgt} ({units.get(tgt,'')})")

        ax.text(
            0.05, 0.93,
            f"$R^2={r2v:.3f}$\n$RMSE={rmsev:.2f}$",
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95)
        )

        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved parity: {out_png}")
    plt.show()


# =========================
# Downstream models (6)
# =========================
def build_downstream_models():
    # XGB (wrap for multi-output)
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="gpu_hist" if DEVICE == "cuda" else "hist",
        verbosity=0,
    )
    m_xgb = MultiOutputRegressor(xgb_reg, n_jobs=-1)

    # RF (native multi-output)
    m_rf = RandomForestRegressor(
        n_estimators=900,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
    )

    # SVR (scale + wrap)
    m_svr = MultiOutputRegressor(
        Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=20.0, epsilon=0.05, kernel="rbf", gamma="scale"))
        ]),
        n_jobs=-1
    )

    # GBR (wrap)
    m_gbr = MultiOutputRegressor(
        GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            n_estimators=900,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
        ),
        n_jobs=-1
    )

    # MLP (native multi-output)
    m_mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=RANDOM_SEED,
        ))
    ])

    # ET (ExtraTrees) — replace LR
    m_et = ExtraTreesRegressor(
        n_estimators=1400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=False,
    )

    return {
        "XGB": m_xgb,
        "GBR": m_gbr,
        "MLP": m_mlp,
        "RF": m_rf,
        "SVR": m_svr,
        "ET": m_et,     # replaced LR
    }

# =========================
# Main
# =========================
def main():
    seed_everything(RANDOM_SEED)
    print(f"[INFO] DEVICE = {DEVICE}")

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)

    # Ensure numeric columns
    for c in COMPOSITION_COLS + Y_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep rows with labels
    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    print(f"[INFO] n_samples = {len(df)}")

    # -------------------------
    # Build text fields
    # -------------------------
    df["CompSentence"] = df.apply(build_comp_sentence, axis=1)
    df["ProcSentence"] = df.apply(build_proc_sentence, axis=1)

    y = df[Y_COLS].values.astype(float)

    # -------------------------
    # CV splitter
    # -------------------------
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # -------------------------
    # ZnBERT embeddings
    # -------------------------
    comp_cache = os.path.join(CACHE_DIR, "comp_emb_ZnBERV2.npy")
    proc_cache = os.path.join(CACHE_DIR, "proc_emb_ZnBERTV2.npy")

    emb_comp = get_embeddings(df["CompSentence"].tolist(), ZNBERT_PATH, comp_cache)
    emb_proc = get_embeddings(df["ProcSentence"].tolist(), ZNBERT_PATH, proc_cache)

    X_text = np.hstack([emb_comp, emb_proc]).astype(np.float32)
    print(f"[INFO] X_text shape = {X_text.shape}")

    # -------------------------
    # Only best downstream model: XGB
    # -------------------------
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="gpu_hist" if DEVICE == "cuda" else "hist",
        verbosity=0,
    )
    model = MultiOutputRegressor(xgb_reg, n_jobs=-1)

    print("\n[TRAIN] ZnBERT(all-text) + XGB (5-fold OOF) ...")
    oof_xgb = oof_predict(model, X_text, y, cv)

    # -------------------------
    # Metrics (XGB only)
    # -------------------------
    metrics_long = summarize_metrics("XGB", y, oof_xgb)
    print("\n========== Metrics (XGB OOF) ==========")
    print(metrics_long)

    # -------------------------
    # Plot parity only (XGB)
    # -------------------------
    plot_best_parity(y, oof_xgb, "XGBoost", OUT_BEST_PARITY_PNG)

    # -------------------------
    # Export: metrics + OOF preds
    # -------------------------
    with pd.ExcelWriter(OUT_METRICS_XLSX, engine="openpyxl") as w:
        metrics_long.to_excel(w, sheet_name="metrics_long", index=False)

        base_cols = COMPOSITION_COLS + ["CompSentence", "ProcSentence"] + Y_COLS
        out = df[base_cols].copy()

        for i, tgt in enumerate(Y_COLS):
            out[f"Pred_{tgt}_XGB"] = oof_xgb[:, i]
            out[f"AbsErr_{tgt}_XGB"] = np.abs(y[:, i] - oof_xgb[:, i])

        out.to_excel(w, sheet_name="OOF_preds", index=False)

    print(f"[INFO] Saved parity figure: {OUT_BEST_PARITY_PNG}")
    print(f"[INFO] Exported metrics + OOF preds to: {OUT_METRICS_XLSX}")


if __name__ == "__main__":
    main()
