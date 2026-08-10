# -*- coding: utf-8 -*-
"""
Fixed downstream: XGBoost
Compare upstream encoders: ZnBERT vs SciBERT vs BERT vs MatBERT
All-text:
  - Composition -> sentence -> encoder embedding
  - Processing  -> sentence -> encoder embedding
  - concat -> XGB -> predict UTS/YS/EL

Outputs:
  - metrics_long (per target + AVG) -> Excel
  - metrics_wide (R2/RMSE/MAE summary) -> Excel
  - a publication-ready TABLE FIGURE (PNG) for encoder comparison
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# Config
# =========================
DATA_XLSX = "Zn-NLP_norm_structured.xlsx"
SHEET = "Sheet1"

CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MAX_LEN = 128

Y_COLS = ["UTS", "YS", "EL"]

# Upstream encoders to compare
ZNBERT_PATH = "ZnSciBERT_MLM"  # your local ZnBERT directory
ENCODERS = {
    "ZnBERT": ZNBERT_PATH,
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BERT": "bert-base-uncased",
    "MatBERT": "alan-yahya/MatBERT",  # HF upload of MatBERT
}

# Export
OUT_XLSX = "EncoderCompare_FixedXGB_metrics.xlsx"
OUT_TABLE_PNG = "EncoderCompare_FixedXGB_Table.png"

# Fonts
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

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
        T, AR = safe_get("extrusion_T"), safe_get("extrusion_AR")
        parts.append("Extrusion" + (f" T={int(T)}C" if T else "") + (f" AR={int(AR)}" if AR else ""))
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
    """Encode texts using mean pooling. Cache to .npy for speed."""
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
    """KFold OOF for multi-target regression."""
    oof = np.zeros_like(y, dtype=float)
    for fold, (tr_idx, te_idx) in enumerate(splitter.split(X, np.zeros(len(X))), 1):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = est.predict(X[te_idx])
        print(f"[INFO] fold {fold} done.")
    return oof

def summarize_metrics(tag, y_true, y_pred):
    rows = []
    r2s = []
    for i, tgt in enumerate(Y_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2v = r2_score(yt, yp)
        r2s.append(r2v)
        rows.append({
            "encoder": tag,
            "target": tgt,
            "R2": r2v,
            "RMSE": rmse(yt, yp),
            "MAE": mean_absolute_error(yt, yp),
        })
    rows.append({
        "encoder": tag,
        "target": "AVG",
        "R2": float(np.mean(r2s)),
        "RMSE": np.nan,
        "MAE": np.nan,
    })
    return pd.DataFrame(rows)

def metrics_to_wide(metrics_long: pd.DataFrame) -> pd.DataFrame:
    df = metrics_long[metrics_long["target"].isin(Y_COLS)].copy()

    p_r2 = df.pivot(index="encoder", columns="target", values="R2").rename(columns=lambda c: f"R2_{c}")
    p_rm = df.pivot(index="encoder", columns="target", values="RMSE").rename(columns=lambda c: f"RMSE_{c}")
    p_ma = df.pivot(index="encoder", columns="target", values="MAE").rename(columns=lambda c: f"MAE_{c}")

    out = pd.concat([p_r2, p_rm, p_ma], axis=1)
    out["avgR2"] = out[[f"R2_{c}" for c in Y_COLS]].mean(axis=1)
    out = out.reset_index().sort_values("avgR2", ascending=False).reset_index(drop=True)
    return out

def plot_table_figure(df_wide: pd.DataFrame, out_png: str, title: str):
    """
    Draw a clean table figure (PNG) for papers/slides.
    """
    # Choose what to show (you can add/remove columns here)
    show_cols = [
        "encoder",
        "avgR2",
        "R2_UTS", "R2_YS", "R2_EL",
        "RMSE_UTS", "RMSE_YS", "RMSE_EL",
    ]
    df = df_wide.copy()
    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[show_cols]

    # format values
    def fmt(v, kind):
        if pd.isna(v):
            return ""
        if kind == "r2":
            return f"{v:.3f}"
        if kind == "rmse":
            return f"{v:.2f}"
        return str(v)

    cell_text = []
    for _, r in df.iterrows():
        row = [
            str(r["encoder"]),
            fmt(r["avgR2"], "r2"),
            fmt(r["R2_UTS"], "r2"), fmt(r["R2_YS"], "r2"), fmt(r["R2_EL"], "r2"),
            fmt(r["RMSE_UTS"], "rmse"), fmt(r["RMSE_YS"], "rmse"), fmt(r["RMSE_EL"], "rmse"),
        ]
        cell_text.append(row)

    col_labels = ["Encoder", "avgR2", "R2(UTS)", "R2(YS)", "R2(EL)", "RMSE(UTS)", "RMSE(YS)", "RMSE(EL)"]

    # figure size adapts to rows
    nrows = len(df) + 1
    fig_w = 15
    fig_h = max(2.6, 0.55 + 0.38 * nrows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)

    # style header
    for j in range(len(col_labels)):
        cell = tbl[(0, j)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#F2F2F2")

    # highlight best row (first after sorting)
    # (data rows start at 1)
    if len(df) > 0:
        best_row_idx = 1
        for j in range(len(col_labels)):
            tbl[(best_row_idx, j)].set_text_props(weight="bold")
            tbl[(best_row_idx, j)].set_facecolor("#E8F4FF")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved table figure: {out_png}")
    plt.show()

def make_xgb_estimator():
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
    return MultiOutputRegressor(xgb_reg, n_jobs=-1)


# =========================
# Main
# =========================
def main():
    seed_everything(RANDOM_SEED)
    print(f"[INFO] DEVICE = {DEVICE}")

    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)

    # numeric cast
    for c in COMPOSITION_COLS + Y_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    print(f"[INFO] n_samples = {len(df)}")

    # build text
    df["CompSentence"] = df.apply(build_comp_sentence, axis=1)
    df["ProcSentence"] = df.apply(build_proc_sentence, axis=1)

    y = df[Y_COLS].values.astype(float)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    metrics_all = []
    oof_store = {}

    for enc_name, enc_path in ENCODERS.items():
        comp_cache = os.path.join(CACHE_DIR, f"comp_emb_{enc_name}.npy")
        proc_cache = os.path.join(CACHE_DIR, f"proc_emb_{enc_name}.npy")

        emb_comp = get_embeddings(df["CompSentence"].tolist(), enc_path, comp_cache)
        emb_proc = get_embeddings(df["ProcSentence"].tolist(), enc_path, proc_cache)
        X_text = np.hstack([emb_comp, emb_proc]).astype(np.float32)

        print(f"\n[TRAIN] Encoder={enc_name} + Fixed XGB (OOF) ...")
        model = make_xgb_estimator()
        oof = oof_predict(model, X_text, y, cv)

        oof_store[enc_name] = oof
        metrics_all.append(summarize_metrics(enc_name, y, oof))

    metrics_long = pd.concat(metrics_all, ignore_index=True)
    metrics_wide = metrics_to_wide(metrics_long)

    print("\n========== Metrics (LONG) ==========")
    print(metrics_long.sort_values(["target", "R2"], ascending=[True, False]))

    print("\n========== Metrics (WIDE summary) ==========")
    print(metrics_wide)

    # Plot table figure (sorted by avgR2)
    plot_table_figure(
        metrics_wide,
        OUT_TABLE_PNG,
        title="Encoder Comparison (Fixed XGBoost, 5-fold OOF)"
    )

    # Export Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        metrics_long.to_excel(w, sheet_name="metrics_long", index=False)
        metrics_wide.to_excel(w, sheet_name="metrics_wide", index=False)

        # Optional: save OOF predictions for each encoder
        base_cols = COMPOSITION_COLS + ["CompSentence", "ProcSentence"] + Y_COLS
        out = df[base_cols].copy()
        for enc_name, oof in oof_store.items():
            for i, tgt in enumerate(Y_COLS):
                out[f"Pred_{tgt}_{enc_name}"] = oof[:, i]
                out[f"AbsErr_{tgt}_{enc_name}"] = np.abs(y[:, i] - oof[:, i])
        out.to_excel(w, sheet_name="OOF_preds", index=False)

    print(f"[INFO] Exported results to: {OUT_XLSX}")
    print(f"[INFO] Table figure saved to: {OUT_TABLE_PNG}")


if __name__ == "__main__":
    main()
