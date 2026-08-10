# -*- coding: utf-8 -*-
"""
Pure ML baseline (no BERT):
Input = numeric composition + numeric structured processing features
Compare 6 models: XGB, RF, SVR, GBR, MLP, ET
Evaluation: 5-fold OOF for UTS/YS/EL
Outputs:
  - metrics_long + metrics_wide -> Excel
  - parity plot ONLY for best model (by avgR2)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

RANDOM_SEED = 42
N_SPLITS = 5

Y_COLS = ["UTS", "YS", "EL"]

COMPOSITION_COLS = [
    "Mg (wt%)","Mn (wt%)","Ag (wt%)","Li (wt%)","Ca (wt%)","Cu(wt%)","Sr(wt%)","Zr(wt%)","Fe(wt%)",
    "Al(wt%)","Ti(wt%)","Nd(wt%)","Gd(wt%)","Sc(wt%)","Er(wt%)","Dy(wt%)","Ho(wt%)"
]

STRUCT_PROC_COLS = [
    "extrusion_T","extrusion_S","extrusion_AR","rolling_T","rolling_AR_total","mdf_passes","ecap_T","ecap_passes",
    "homogen_T","homogen_t_h","anneal_T","anneal_t_h","wiredraw_AR","hpt_PASS"
]

FEATURE_COLS = COMPOSITION_COLS + STRUCT_PROC_COLS

# Export
OUT_METRICS_XLSX = "PureML_Compare_metrics.xlsx"
OUT_BEST_PARITY_PNG = "PureML_Best_Parity.png"

# Style (Arial)
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# 曲线
# =========================
# Helpers
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def oof_predict(estimator, X, y, splitter):
    """Generic KFold OOF for multi-target regression."""
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
    df = metrics_long[metrics_long["target"].isin(Y_COLS)].copy()

    p_r2 = df.pivot(index="model", columns="target", values="R2").rename(columns=lambda c: f"R2_{c}")
    p_rm = df.pivot(index="model", columns="target", values="RMSE").rename(columns=lambda c: f"RMSE_{c}")
    p_ma = df.pivot(index="model", columns="target", values="MAE").rename(columns=lambda c: f"MAE_{c}")

    out = pd.concat([p_r2, p_rm, p_ma], axis=1)
    out["avgR2"] = out[[f"R2_{c}" for c in Y_COLS]].mean(axis=1)
    out = out.reset_index().sort_values("avgR2", ascending=False).reset_index(drop=True)
    return out

def plot_best_parity(y_true, y_pred, model_name, out_png):
    """Parity plots like your reference (fix linewidth alias: use linewidths)."""
    units = {"UTS": "MPa", "YS": "MPa", "EL": "%"}

    plt.rcParams.update({
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

        sns.regplot(
            x=yt, y=yp, ax=ax,
            ci=95,
            scatter=True,
            scatter_kws=dict(
                s=28, alpha=0.55,
                color="#5DA5DA",
                edgecolor="white",
                linewidths=0.5,  # IMPORTANT
            ),
            line_kws=dict(
                color="red",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9
            )
        )

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
# Models (6)
# =========================
def build_models():
    # shared numeric preprocessing
    # - impute missing values (median)
    # - optionally scale (for SVR/MLP; keep consistent by putting scaler inside pipeline)
    imputer = ("imputer", SimpleImputer(strategy="median"))

    # XGB (wrap for multi-output)
    xgb_reg = xgb.XGBRegressor(

        verbosity=0,
    )
    m_xgb = Pipeline([
        imputer,
        ("xgb", MultiOutputRegressor(xgb_reg, n_jobs=-1))
    ])

    # RF
    m_rf = Pipeline([
        imputer,
        ("rf", RandomForestRegressor(
            n_estimators=900, random_state=RANDOM_SEED, n_jobs=-1,
            max_features="sqrt", min_samples_leaf=1
        ))
    ])

    # ET
    m_et = Pipeline([
        imputer,
        ("et", ExtraTreesRegressor(
            n_estimators=1400, random_state=RANDOM_SEED, n_jobs=-1,
            max_features="sqrt", min_samples_leaf=1, bootstrap=False
        ))
    ])

    # GBR (wrap)
    m_gbr = Pipeline([
        imputer,
        ("gbr", MultiOutputRegressor(
            GradientBoostingRegressor(
                random_state=RANDOM_SEED,
                n_estimators=900,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
            ),
            n_jobs=-1
        ))
    ])

    # SVR (scale + wrap)
    m_svr = Pipeline([
        imputer,
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(
            SVR(C=20.0, epsilon=0.05, kernel="rbf", gamma="scale"),
            n_jobs=-1
        ))
    ])

    # MLP (scale)
    m_mlp = Pipeline([
        imputer,
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128),
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

    return {
        "XGB": m_xgb,
        "GBR": m_gbr,
        "ET":  m_et,
        "MLP": m_mlp,
        "RF":  m_rf,
        "SVR": m_svr,
    }


# =========================
# Main
# =========================
def main():
    seed_everything(RANDOM_SEED)

    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)

    # numeric cast
    for c in FEATURE_COLS + Y_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    print(f"[INFO] n_samples = {len(df)}")
    print(f"[INFO] n_features = {len(FEATURE_COLS)}")

    X = df[FEATURE_COLS].values.astype(float)
    y = df[Y_COLS].values.astype(float)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    models = build_models()
    preds = {}
    metrics_all = []

    for name, est in models.items():
        print(f"\n[TRAIN] PureML + {name} (OOF) ...")
        oof = oof_predict(est, X, y, cv)
        preds[name] = oof
        metrics_all.append(summarize_metrics(name, y, oof))

    metrics_long = pd.concat(metrics_all, ignore_index=True)
    metrics_wide = metrics_to_wide(metrics_long)

    print("\n========== Metrics (LONG) ==========")
    print(metrics_long.sort_values(["target", "R2"], ascending=[True, False]))

    print("\n========== Metrics (WIDE summary) ==========")
    print(metrics_wide)

    best_name = metrics_wide.iloc[0]["model"]
    print(f"\n[BEST] Best model = {best_name} (by avgR2)")

    # only best parity plot
    plot_best_parity(y, preds[best_name], best_name, OUT_BEST_PARITY_PNG)

    # export
    with pd.ExcelWriter(OUT_METRICS_XLSX, engine="openpyxl") as w:
        metrics_long.to_excel(w, sheet_name="metrics_long", index=False)
        metrics_wide.to_excel(w, sheet_name="metrics_wide", index=False)

        out = df[FEATURE_COLS + Y_COLS].copy()
        for mname, oof in preds.items():
            for i, tgt in enumerate(Y_COLS):
                out[f"Pred_{tgt}_{mname}"] = oof[:, i]
                out[f"AbsErr_{tgt}_{mname}"] = np.abs(y[:, i] - oof[:, i])
        out.to_excel(w, sheet_name="OOF_preds", index=False)

    print(f"[INFO] Saved parity: {OUT_BEST_PARITY_PNG}")
    print(f"[INFO] Exported tables: {OUT_METRICS_XLSX}")


if __name__ == "__main__":
    main()
