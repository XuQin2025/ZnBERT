# -*- coding: utf-8 -*-
"""
ZnBERT(all-text) + XGB fixed
Explainability via Natural-Language Ablation:
  - Remove one structured composition/processing field from the sentence
  - Re-encode with ZnBERT
  - Use the SAME fold models trained on full features to predict OOF
  - Performance drop => feature importance

Outputs (Excel):
  1) baseline_metrics
  2) ablation_feature_importance (delta metrics)
  3) group_summary (composition vs processing)
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# Config
# =========================
DATA_XLSX = "Zn-NLP_norm_structured.xlsx"
SHEET = "Sheet1"

ZNBERT_PATH = "ZnBERTv2_8epoch_DAPT_Abstract/checkpoint-1200"
CACHE_DIR = "cache_embeddings_ablation"
os.makedirs(CACHE_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MAX_LEN = 128

Y_COLS = ["UTS", "YS", "EL"]

OUT_XLSX = "ZnBERT_XGB_Ablation_Importance.xlsx"

# -------------------------
# Structured features you want to explain (exactly your lists)
# -------------------------
COMPOSITION_COLS = [
    "Mg (wt%)", "Mn (wt%)", "Ag (wt%)", "Li (wt%)", "Ca (wt%)", "Cu(wt%)", "Sr(wt%)", "Zr(wt%)", "Fe(wt%)",
    "Al(wt%)", "Ti(wt%)", "Nd(wt%)", "Gd(wt%)", "Sc(wt%)", "Er(wt%)", "Dy(wt%)", "Ho(wt%)"
]

STRUCT_PROC_COLS = [
    "extrusion_T",
    "extrusion_AR",
    "extrusion_S",
    "rolling_T",
    "rolling_AR_total",
    "mdf_passes",
    "ecap_T",
    "ecap_passes",
    "homogen_T",
    "homogen_t_h",
    "anneal_T",
    "anneal_t_h",
    "wiredraw_AR",
    "hpt_PASS",
]

# If you want strict ablation (retrain model per feature), set True (much slower)
RETRAIN_FOR_EACH_FEATURE = False


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

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def build_comp_sentence(row, base="Zn", exclude_elem: str = None):
    """
    exclude_elem: one of COMPOSITION_COLS, e.g. "Mg (wt%)"
    """
    parts = []
    for c in COMPOSITION_COLS:
        if exclude_elem is not None and c == exclude_elem:
            continue
        v = safe_float(row.get(c))
        if v is None:
            continue
        if abs(v) < 1e-12:
            continue
        elem = c.replace("(wt%)", "").replace(" (wt%)", "").strip()
        parts.append(f"{v:g} wt% {elem}")
    if not parts:
        return f"{base}-based alloy."
    return f"{base}-based alloy with " + ", ".join(parts) + "."

def build_proc_sentence(row, exclude_proc: str = None):
    """
    exclude_proc: one of STRUCT_PROC_COLS, e.g. "extrusion_T"
    Build a processing sentence from has_* flags + numeric params,
    but allow removing one specific parameter (natural-language ablation).
    """
    if "Processing" in row and isinstance(row["Processing"], str) and row["Processing"].strip() == "Casting":
        return "Casting."

    def getv(c):
        v = row.get(c)
        return None if pd.isna(v) else v

    parts = []

    # Homogenization
    if row.get("has_homogen") == 1:
        seg = "Homogenization"
        T = getv("homogen_T")
        t = getv("homogen_t_h")
        if exclude_proc != "homogen_T" and T is not None:
            seg += f" T={int(float(T))}C"
        if exclude_proc != "homogen_t_h" and t is not None:
            seg += f" t={float(t):g}h"
        parts.append(seg)

    # Extrusion
    # Extrusion
    if row.get("has_extrusion") == 1:
        seg = "Extrusion"
        T = getv("extrusion_T")
        AR = getv("extrusion_AR")
        S = getv("extrusion_S")  # NEW

        if exclude_proc != "extrusion_T" and T is not None:
            seg += f" T={int(float(T))}C"
        if exclude_proc != "extrusion_AR" and AR is not None:
            seg += f" AR={int(float(AR))}"
        # NEW: extrusion speed (unit unknown in your DB -> keep generic "S=")
        if exclude_proc != "extrusion_S" and S is not None:
            seg += f" S={float(S):g}"
        parts.append(seg)


    # Rolling
    if row.get("has_rolling") == 1:
        seg = "Rolling"
        T = getv("rolling_T")
        ARt = getv("rolling_AR_total")
        if exclude_proc != "rolling_T" and T is not None:
            seg += f" T={int(float(T))}C"
        if exclude_proc != "rolling_AR_total" and ARt is not None:
            seg += f" AR_total={float(ARt):g}%"
        parts.append(seg)

    # MDF
    if row.get("has_MDF") == 1:
        p = getv("mdf_passes")
        if exclude_proc != "mdf_passes" and p is not None:
            parts.append(f"MDF passes={int(float(p))}")
        else:
            parts.append("MDF")

    # ECAP
    if row.get("has_ECAP") == 1:
        seg = "ECAP"
        T = getv("ecap_T")
        p = getv("ecap_passes")
        if exclude_proc != "ecap_T" and T is not None:
            seg += f" T={int(float(T))}C"
        if exclude_proc != "ecap_passes" and p is not None:
            seg += f" passes={int(float(p))}"
        parts.append(seg)

    # Anneal
    if row.get("has_anneal") == 1:
        seg = "Anneal"
        T = getv("anneal_T")
        t = getv("anneal_t_h")
        if exclude_proc != "anneal_T" and T is not None:
            seg += f" T={int(float(T))}C"
        if exclude_proc != "anneal_t_h" and t is not None:
            seg += f" t={float(t):g}h"
        parts.append(seg)

    # WireDrawing (add AR if available)
    if row.get("has_wiredraw") == 1:
        seg = "WireDrawing"
        AR = getv("wiredraw_AR")
        if exclude_proc != "wiredraw_AR" and AR is not None:
            seg += f" AR={float(AR):g}%"
        parts.append(seg)

    # HPT
    if row.get("has_HPT") == 1:
        p = getv("hpt_PASS")
        if exclude_proc != "hpt_PASS" and p is not None:
            parts.append(f"HPT passes={int(float(p))}")
        else:
            parts.append("HPT")

    return "; ".join(parts) + "." if parts else "No processing info."

@torch.no_grad()
def get_embeddings(texts, model_path, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(emb.cpu().numpy())

    res = np.vstack(all_embs).astype(np.float32)
    np.save(cache_path, res)
    return res

def fit_fold_models_and_oof(X, y, cv, xgb_params):
    """
    Train fold models on FULL features (baseline) and return:
      - baseline_oof
      - fold_models: list of (te_idx, fitted_model)
    """
    baseline_oof = np.zeros_like(y, dtype=float)
    fold_models = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, np.zeros(len(X))), 1):
        xgb_reg = xgb.XGBRegressor(**xgb_params)
        model = MultiOutputRegressor(xgb_reg, n_jobs=-1)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx])
        baseline_oof[te_idx] = pred
        fold_models.append((te_idx, model))
        print(f"[INFO] baseline fold {fold} done.")
    return baseline_oof, fold_models

def oof_predict_with_fixed_fold_models(X_new, fold_models, y_shape):
    """
    Use existing fold models (trained on baseline) to predict OOF on X_new.
    """
    oof = np.zeros(y_shape, dtype=float)
    for (te_idx, model) in fold_models:
        oof[te_idx] = model.predict(X_new[te_idx])
    return oof

def summarize_metrics(tag, y_true, y_pred):
    rows = []
    r2s = []
    for i, tgt in enumerate(Y_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2v = r2_score(yt, yp)
        r2s.append(r2v)
        rows.append({
            "name": tag,
            "target": tgt,
            "R2": r2v,
            "RMSE": rmse(yt, yp),
            "MAE": mean_absolute_error(yt, yp),
        })
    rows.append({
        "name": tag,
        "target": "AVG",
        "R2": float(np.mean(r2s)),
        "RMSE": np.nan,
        "MAE": np.nan,
    })
    return pd.DataFrame(rows)

def to_wide(metrics_df):
    df = metrics_df[metrics_df["target"].isin(Y_COLS)].copy()
    p_r2 = df.pivot(index="name", columns="target", values="R2").rename(columns=lambda c: f"R2_{c}")
    p_rm = df.pivot(index="name", columns="target", values="RMSE").rename(columns=lambda c: f"RMSE_{c}")
    out = pd.concat([p_r2, p_rm], axis=1)
    out["avgR2"] = out[[f"R2_{c}" for c in Y_COLS]].mean(axis=1)
    return out.reset_index()

# =========================
# Main
# =========================
def main():
    seed_everything(RANDOM_SEED)
    print(f"[INFO] DEVICE={DEVICE}")

    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)

    # ensure numeric (composition + y + structured proc)
    for c in COMPOSITION_COLS + STRUCT_PROC_COLS + Y_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    print(f"[INFO] n_samples={len(df)}")

    y = df[Y_COLS].values.astype(float)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # XGB params (same as your best)
    xgb_params = dict(
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

    # ========== Baseline embeddings ==========
    comp_text = df.apply(lambda r: build_comp_sentence(r, exclude_elem=None), axis=1).tolist()
    proc_text = df.apply(lambda r: build_proc_sentence(r, exclude_proc=None), axis=1).tolist()

    emb_comp = get_embeddings(comp_text, ZNBERT_PATH, os.path.join(CACHE_DIR, "emb_comp_BASE.npy"))
    emb_proc = get_embeddings(proc_text, ZNBERT_PATH, os.path.join(CACHE_DIR, "emb_proc_BASE.npy"))
    X_base = np.hstack([emb_comp, emb_proc]).astype(np.float32)
    print(f"[INFO] X_base shape={X_base.shape}")

    # ========== Train baseline fold models & baseline OOF ==========
    oof_base, fold_models = fit_fold_models_and_oof(X_base, y, cv, xgb_params)
    base_metrics = summarize_metrics("BASE(full)", y, oof_base)
    print("\n========== BASELINE (OOF) ==========")
    print(base_metrics)

    base_wide = to_wide(base_metrics)
    base_avgR2 = float(base_wide.loc[0, "avgR2"])

    # ========== Ablation: each composition/proc feature ==========
    records = []

    # --- composition ablation (remove one element from comp sentence) ---
    for feat in COMPOSITION_COLS:
        tag = f"drop_comp::{feat}"
        print(f"[ABLATE] {tag}")

        comp_text_drop = df.apply(lambda r: build_comp_sentence(r, exclude_elem=feat), axis=1).tolist()
        emb_comp_drop = get_embeddings(
            comp_text_drop,
            ZNBERT_PATH,
            os.path.join(CACHE_DIR, f"emb_comp_drop_{feat.replace(' ','').replace('(','').replace(')','').replace('%','').replace('/','')}.npy")
        )

        X_drop = np.hstack([emb_comp_drop, emb_proc]).astype(np.float32)

        if RETRAIN_FOR_EACH_FEATURE:
            # strict: retrain per ablation (slow)
            oof_drop, _ = fit_fold_models_and_oof(X_drop, y, cv, xgb_params)
        else:
            # fast & fair: use the same baseline fold models to predict OOF on ablated input
            oof_drop = oof_predict_with_fixed_fold_models(X_drop, fold_models, y.shape)

        m = summarize_metrics(tag, y, oof_drop)
        w = to_wide(m).iloc[0].to_dict()
        delta = base_avgR2 - float(w["avgR2"])

        records.append({
            "group": "COMPOSITION",
            "feature": feat,
            "delta_avgR2": delta,
            "delta_R2_UTS": float(base_wide.loc[0, "R2_UTS"]) - float(w["R2_UTS"]),
            "delta_R2_YS":  float(base_wide.loc[0, "R2_YS"])  - float(w["R2_YS"]),
            "delta_R2_EL":  float(base_wide.loc[0, "R2_EL"])  - float(w["R2_EL"]),
        })

    # --- processing ablation (remove one parameter from proc sentence) ---
    for feat in STRUCT_PROC_COLS:
        tag = f"drop_proc::{feat}"
        print(f"[ABLATE] {tag}")

        proc_text_drop = df.apply(lambda r: build_proc_sentence(r, exclude_proc=feat), axis=1).tolist()
        emb_proc_drop = get_embeddings(
            proc_text_drop,
            ZNBERT_PATH,
            os.path.join(CACHE_DIR, f"emb_proc_drop_{feat}.npy")
        )

        X_drop = np.hstack([emb_comp, emb_proc_drop]).astype(np.float32)

        if RETRAIN_FOR_EACH_FEATURE:
            oof_drop, _ = fit_fold_models_and_oof(X_drop, y, cv, xgb_params)
        else:
            oof_drop = oof_predict_with_fixed_fold_models(X_drop, fold_models, y.shape)

        m = summarize_metrics(tag, y, oof_drop)
        w = to_wide(m).iloc[0].to_dict()
        delta = base_avgR2 - float(w["avgR2"])

        records.append({
            "group": "PROCESSING",
            "feature": feat,
            "delta_avgR2": delta,
            "delta_R2_UTS": float(base_wide.loc[0, "R2_UTS"]) - float(w["R2_UTS"]),
            "delta_R2_YS":  float(base_wide.loc[0, "R2_YS"])  - float(w["R2_YS"]),
            "delta_R2_EL":  float(base_wide.loc[0, "R2_EL"])  - float(w["R2_EL"]),
        })

    imp_df = pd.DataFrame(records).sort_values("delta_avgR2", ascending=False).reset_index(drop=True)

    # group summary: sum / mean importance
    grp = imp_df.groupby("group", as_index=False).agg(
        n_features=("feature", "count"),
        sum_delta_avgR2=("delta_avgR2", "sum"),
        mean_delta_avgR2=("delta_avgR2", "mean"),
        median_delta_avgR2=("delta_avgR2", "median"),
        top5_sum=("delta_avgR2", lambda s: float(np.sum(np.sort(s.values)[-5:])) if len(s) >= 5 else float(np.sum(s.values)))
    )

    print("\n========== Feature importance by natural-language ablation ==========")
    print(imp_df.head(15))

    print("\n========== Group summary (composition vs processing) ==========")
    print(grp)

    # export
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        base_metrics.to_excel(w, sheet_name="baseline_metrics", index=False)
        imp_df.to_excel(w, sheet_name="ablation_importance", index=False)
        grp.to_excel(w, sheet_name="group_summary", index=False)

    print(f"\n[INFO] Saved: {OUT_XLSX}")
    print("[NOTE] delta_avgR2 > 0 means: removing this info hurts performance => more important")

if __name__ == "__main__":
    main()
