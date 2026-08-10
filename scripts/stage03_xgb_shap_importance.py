# -*- coding: utf-8 -*-
"""
Feature importance analysis for multi-target (UTS/YS/EL) with XGBoost (Booster API).
Outputs:
  - Built-in importance (gain/weight/cover)
  - Permutation importance (RMSE increase) on CV folds
  - Optional SHAP summary (if shap installed)
  - Export all to Excel

Author: (you)
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# Config
# =========================
DATA_XLSX = "Zn-NLP_norm_structured.xlsx"
SHEET = "Sheet1"
OUT_XLSX = "Feature_Importance_XGB.xlsx"

RANDOM_SEED = 42
N_SPLITS = 5
STRATIFY_COL = "Route_main"  # 若不足以分层，会自动用KFold

Y_COLS = ["UTS", "YS", "EL"]

COMPOSITION_COLS = [
    "Mg (wt%)","Mn (wt%)","Ag (wt%)","Li (wt%)","Ca (wt%)","Cu(wt%)","Sr(wt%)","Zr(wt%)","Fe(wt%)",
    "Al(wt%)","Ti(wt%)","Nd(wt%)","Gd(wt%)","Sc(wt%)","Er(wt%)","Dy(wt%)","Ho(wt%)"
]
STRUCT_PROC_COLS = [
    "extrusion_T","extrusion_AR","extrusion_S","rolling_T","rolling_AR_total","mdf_passes","ecap_T","ecap_passes",
    "homogen_T","homogen_t_h","anneal_T","anneal_t_h","wiredraw_AR","hpt_PASS"
]
HAS_COLS = [
    "has_extrusion","has_rolling","has_ECAP","has_MDF","has_homogen","has_anneal","has_wiredraw","has_HPT"
]

FEAT_COLS = COMPOSITION_COLS + STRUCT_PROC_COLS


# =========================
# Helpers
# =========================
def can_stratify(labels, n_splits):
    vc = pd.Series(labels).value_counts(dropna=False)
    return (vc.min() >= n_splits)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def build_xgb_params(seed=42):
    # 你现在的结果很好，这里给一套“稳健默认”，你也可以替换成你当前最优超参
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "seed": seed,
    }

def train_booster(X_train, y_train, X_val, y_val, seed=42,
                  num_boost_round=10000, early_stopping_rounds=200):
    params = build_xgb_params(seed)
    dtr = xgb.DMatrix(X_train, label=y_train, feature_names=FEAT_COLS)
    dva = xgb.DMatrix(X_val, label=y_val, feature_names=FEAT_COLS)
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    return booster

def get_builtin_importance(booster, importance_type="gain"):
    """
    importance_type: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
    returns DataFrame with columns: feature, importance
    """
    score = booster.get_score(importance_type=importance_type)
    # ensure all features present
    imp = pd.DataFrame({"feature": FEAT_COLS})
    imp["importance"] = imp["feature"].map(score).fillna(0.0)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp

def permutation_importance_rmse(booster, X_test, y_test, n_repeats=5, seed=42):
    """
    Manual permutation importance:
      baseline_rmse computed first
      for each feature: shuffle column -> compute rmse increase
    Returns DataFrame: feature, delta_rmse_mean, delta_rmse_std
    """
    rng = np.random.default_rng(seed)
    dte = xgb.DMatrix(X_test, feature_names=FEAT_COLS)
    base_pred = booster.predict(dte)
    base = rmse(y_test, base_pred)

    deltas = []
    Xp = X_test.copy()

    for j, feat in enumerate(FEAT_COLS):
        incs = []
        for r in range(n_repeats):
            old = Xp[:, j].copy()
            rng.shuffle(Xp[:, j])
            pred = booster.predict(xgb.DMatrix(Xp, feature_names=FEAT_COLS))
            incs.append(rmse(y_test, pred) - base)
            Xp[:, j] = old  # restore
        deltas.append((feat, float(np.mean(incs)), float(np.std(incs)), base))

    out = pd.DataFrame(deltas, columns=["feature", "delta_rmse_mean", "delta_rmse_std", "baseline_rmse"])
    out = out.sort_values("delta_rmse_mean", ascending=False).reset_index(drop=True)
    return out

def add_group(feature):
    if feature in COMPOSITION_COLS:
        return "Composition"
    if feature in STRUCT_PROC_COLS:
        return "StructuredProc"
    if feature in HAS_COLS:
        return "HasFlags"
    return "Other"


# =========================
# Main
# =========================
def main():
    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET).copy()

    # numeric coercion
    for c in FEAT_COLS + Y_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing targets
    df = df.dropna(subset=Y_COLS).reset_index(drop=True)

    # impute features
    X_raw = df[FEAT_COLS].values
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_raw)
    y = df[Y_COLS].values

    # CV splitter
    if STRATIFY_COL in df.columns and df[STRATIFY_COL].notna().any() and can_stratify(df[STRATIFY_COL].astype(str), N_SPLITS):
        labels = df[STRATIFY_COL].astype(str).values
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X, labels)
        print(f"[INFO] StratifiedKFold by {STRATIFY_COL}")
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X)
        print("[INFO] KFold used (no valid stratify)")

    # collect permutation importance across folds
    perm_all = {t: [] for t in Y_COLS}

    # also train ONE final model per target (for built-in importance + SHAP)
    final_models = {}

    for fold, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # inner split for early stopping
        Xtr2, Xva, ytr2, yva = train_test_split(
            Xtr, ytr, test_size=0.15,
            random_state=RANDOM_SEED + fold, shuffle=True
        )

        print(f"\n================ Fold {fold}/{N_SPLITS} ================")

        for j, tgt in enumerate(Y_COLS):
            booster = train_booster(
                Xtr2, ytr2[:, j],
                Xva,  yva[:, j],
                seed=RANDOM_SEED + 100 * fold + j,
                num_boost_round=10000,
                early_stopping_rounds=200
            )

            # permutation importance on test fold
            perm = permutation_importance_rmse(
                booster, Xte, yte[:, j],
                n_repeats=5, seed=RANDOM_SEED + 999 * fold + j
            )
            perm["Fold"] = fold
            perm["Target"] = tgt
            perm_all[tgt].append(perm)

    # aggregate permutation importance across folds
    perm_summary_tables = {}
    for tgt in Y_COLS:
        all_df = pd.concat(perm_all[tgt], axis=0, ignore_index=True)
        g = all_df.groupby("feature").agg(
            delta_rmse_mean=("delta_rmse_mean", "mean"),
            delta_rmse_std=("delta_rmse_mean", "std"),
            baseline_rmse_mean=("baseline_rmse", "mean")
        ).reset_index()
        g["group"] = g["feature"].map(add_group)
        g = g.sort_values("delta_rmse_mean", ascending=False).reset_index(drop=True)
        perm_summary_tables[tgt] = g

    # train final models on full data (for built-in importance & SHAP)
    XtrF, XvaF, ytrF, yvaF = train_test_split(X, y, test_size=0.12, random_state=RANDOM_SEED, shuffle=True)
    for j, tgt in enumerate(Y_COLS):
        booster = train_booster(XtrF, ytrF[:, j], XvaF, yvaF[:, j], seed=RANDOM_SEED + 2025 + j)
        final_models[tgt] = booster

    # built-in importances
    builtin_tables = {}
    for tgt in Y_COLS:
        b = final_models[tgt]
        df_gain  = get_builtin_importance(b, "gain").rename(columns={"importance": "gain"})
        df_weight= get_builtin_importance(b, "weight").rename(columns={"importance": "weight"})
        df_cover = get_builtin_importance(b, "cover").rename(columns={"importance": "cover"})
        out = df_gain.merge(df_weight, on="feature").merge(df_cover, on="feature")
        out["group"] = out["feature"].map(add_group)
        builtin_tables[tgt] = out

    # optional SHAP
    shap_tables = {}
    try:
        import shap
        # sample a subset for speed
        n_sample = min(800, X.shape[0])
        idx = np.random.default_rng(RANDOM_SEED).choice(np.arange(X.shape[0]), size=n_sample, replace=False)
        Xs = X[idx]

        for tgt in Y_COLS:
            b = final_models[tgt]
            explainer = shap.TreeExplainer(b)
            shap_values = explainer.shap_values(pd.DataFrame(Xs, columns=FEAT_COLS))

            # global importance = mean(|shap|)
            imp = np.mean(np.abs(shap_values), axis=0)
            st = pd.DataFrame({"feature": FEAT_COLS, "mean_abs_shap": imp})
            st["group"] = st["feature"].map(add_group)
            st = st.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_tables[tgt] = st

            # plot (saved to png)
            shap.summary_plot(shap_values, pd.DataFrame(Xs, columns=FEAT_COLS), show=False)
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(f"SHAP_summary_{tgt}.png", dpi=300)
            plt.close()

            shap.summary_plot(shap_values, pd.DataFrame(Xs, columns=FEAT_COLS),
                              plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f"SHAP_bar_{tgt}.png", dpi=300)
            plt.close()

        print("[OK] SHAP plots saved: SHAP_summary_*.png and SHAP_bar_*.png")

    except Exception as e:
        print(f"[WARN] SHAP not available or failed: {e}")
        print("       If you want SHAP, run: pip install shap")

    # export to excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        # raw data snapshot
        df.to_excel(w, "data_used", index=False)

        for tgt in Y_COLS:
            perm_summary_tables[tgt].to_excel(w, f"perm_{tgt}", index=False)
            builtin_tables[tgt].to_excel(w, f"builtin_{tgt}", index=False)
            if tgt in shap_tables:
                shap_tables[tgt].to_excel(w, f"shap_{tgt}", index=False)

        # group-level contribution (from permutation importance)
        for tgt in Y_COLS:
            g = perm_summary_tables[tgt].groupby("group")["delta_rmse_mean"].sum().reset_index()
            g = g.sort_values("delta_rmse_mean", ascending=False)
            g.to_excel(w, f"group_perm_{tgt}", index=False)

    print(f"\n[OK] Exported importance tables to: {OUT_XLSX}")


if __name__ == "__main__":
    main()
