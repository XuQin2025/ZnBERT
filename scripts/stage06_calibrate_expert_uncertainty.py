# -*- coding: utf-8 -*-
"""Calibrated tree uncertainty for two Zn-Mg-Li-Cu extrusion conditions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from stage02_compare_downstream_models_v2 import ZNBERT_PATH, get_embeddings
from stage06_predict_two_alloys_uncertainty import fit_sigma_calibration


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = BUNDLE_ROOT / "artifacts" / "final_screening_model" / "ZnBERT_XGB_final_model.joblib"
DATA_XLSX = BUNDLE_ROOT / "data" / "Zn-NLP_norm_structured.xlsx"
TRAIN_COMP_EMB = BUNDLE_ROOT / "artifacts" / "final_screening_model" / "cache_embeddings" / "train_comp_emb.npy"
TRAIN_PROC_EMB = BUNDLE_ROOT / "artifacts" / "final_screening_model" / "cache_embeddings" / "train_proc_emb_extrusion_only.npy"
OUT_DIR = BUNDLE_ROOT / "outputs" / "znbxgb_expert_process_uncertainty"
RANDOM_SEED = 42
N_SPLITS = 5

COMP_SENTENCE = "Zn-based alloy with 0.25 wt% Mg, 0.2 wt% Li, 2.3 wt% Cu."
CANDIDATES = [
    {
        "Process": "260 °C / AR 25",
        "ProcSentence": "Extrusion T=260C AR=25.",
        "UTS": 410.6625061035156,
        "YS": 353.7416381835938,
        "EL": 25.63358306884766,
    },
    {
        "Process": "240 °C / AR 20",
        "ProcSentence": "Extrusion T=240C AR=20.",
        "UTS": 394.3329467773438,
        "YS": 356.6358032226562,
        "EL": 31.776962280273438,
    },
]


def tree_contribution_matrix(model, x_data):
    booster = model.get_booster()
    leaves = np.asarray(booster.predict(xgb.DMatrix(x_data), pred_leaf=True))
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    trees_df = booster.trees_to_dataframe()
    leaf_df = trees_df.loc[trees_df["Feature"] == "Leaf", ["Tree", "Node", "Gain"]]
    maps = {
        int(tree_id): dict(zip(group["Node"].astype(int), group["Gain"].astype(float)))
        for tree_id, group in leaf_df.groupby("Tree")
    }
    contrib = np.zeros(leaves.shape, dtype=float)
    for tree_i in range(leaves.shape[1]):
        node_values = maps.get(tree_i, {})
        contrib[:, tree_i] = [node_values.get(int(node_id), 0.0) for node_id in leaves[:, tree_i]]
    return contrib


def base_score(model):
    config = json.loads(model.get_booster().save_config())
    value = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(value, str) and value.strip().startswith("["):
        return float(np.asarray(json.loads(value), dtype=float).ravel()[0])
    return float(value)


def tree_sigma(model, x_data):
    tree_predictions = base_score(model) + tree_contribution_matrix(model, x_data)
    return tree_predictions.std(axis=1, ddof=1)


def make_xgb():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp_texts = [COMP_SENTENCE] * len(CANDIDATES)
    proc_texts = [row["ProcSentence"] for row in CANDIDATES]
    comp_emb = get_embeddings(
        comp_texts,
        ZNBERT_PATH,
        str(OUT_DIR / "candidate_comp_embeddings.npy"),
    )
    proc_emb = get_embeddings(
        proc_texts,
        ZNBERT_PATH,
        str(OUT_DIR / "candidate_proc_embeddings.npy"),
    )
    x_cand = np.hstack([comp_emb, proc_emb]).astype(np.float32)

    bundle = joblib.load(MODEL_PATH)
    multi_model = bundle["model"]
    targets = list(bundle["y_cols"])
    model_predictions = np.column_stack(
        [estimator.predict(x_cand) for estimator in multi_model.estimators_]
    ).astype(float)

    # The current reporting convention applies a -5 percentage-point correction to EL.
    model_predictions_adjusted = model_predictions.copy()
    model_predictions_adjusted[:, targets.index("EL")] -= 5.0

    # Recalibrate specifically for the current extrusion-only screening pipeline.
    train_df = pd.read_excel(DATA_XLSX, sheet_name="Sheet1")
    for target in targets:
        train_df[target] = pd.to_numeric(train_df[target], errors="coerce")
    train_df = train_df.dropna(subset=targets).reset_index(drop=True)
    y_train = train_df[targets].to_numpy(dtype=float)
    x_train = np.hstack([np.load(TRAIN_COMP_EMB), np.load(TRAIN_PROC_EMB)]).astype(np.float32)
    if len(x_train) != len(y_train):
        raise ValueError(f"Training row mismatch: embeddings={len(x_train)}, targets={len(y_train)}")

    previous_calibration = pd.read_csv(OUT_DIR / "current_pipeline_sigma_calibration.csv")
    calibration_map = {
        target: previous_calibration.loc[
            previous_calibration["Target"] == target
        ].iloc[0].to_dict()
        for target in ["UTS", "YS"]
    }
    el_i = targets.index("EL")
    oof_pred_el = np.zeros(len(y_train), dtype=float)
    oof_sigma_raw_el = np.zeros(len(y_train), dtype=float)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x_train), start=1):
        print(f"[INFO] Adjusted-EL calibration fold {fold}/{N_SPLITS}", flush=True)
        fold_model = make_xgb()
        fold_model.fit(x_train[train_idx], y_train[train_idx, el_i])
        oof_pred_el[test_idx] = fold_model.predict(x_train[test_idx]) - 5.0
        oof_sigma_raw_el[test_idx] = tree_sigma(fold_model, x_train[test_idx])

    el_cal = fit_sigma_calibration(
        y_train[:, el_i],
        oof_pred_el,
        oof_sigma_raw_el,
    )
    calibration_map["EL"] = el_cal
    calibration_rows = [
        calibration_map["UTS"],
        calibration_map["YS"],
        {"Target": "EL", **el_cal},
    ]
    pd.DataFrame(
        {
            "Measured_EL": y_train[:, el_i],
            "OOF_Pred_EL_adjusted_minus5": oof_pred_el,
            "OOF_TreeSigma_EL_uncalibrated": oof_sigma_raw_el,
        }
    ).to_csv(OUT_DIR / "adjusted_el_oof_calibration_data.csv", index=False, encoding="utf-8-sig")

    rows = []
    for target_i, target in enumerate(targets):
        estimator = multi_model.estimators_[target_i]
        sigma_raw = tree_sigma(estimator, x_cand)
        a = float(calibration_map[target]["a"])
        b = float(calibration_map[target]["b"])
        sigma_cal = a * sigma_raw + b
        for cand_i, candidate in enumerate(CANDIDATES):
            fixed_mean = float(candidate[target])
            model_mean = float(model_predictions_adjusted[cand_i, target_i])
            if not np.isclose(fixed_mean, model_mean, atol=0.06):
                raise ValueError(
                    f"Mean mismatch for {candidate['Process']} {target}: "
                    f"table={fixed_mean:.6f}, model={model_mean:.6f}"
                )
            low = fixed_mean - 1.96 * sigma_cal[cand_i]
            high = fixed_mean + 1.96 * sigma_cal[cand_i]
            rows.append(
                {
                    "Alloy": "Zn-0.25Mg-0.2Li-2.3Cu",
                    "Process": candidate["Process"],
                    "Target": target,
                    "Predict": fixed_mean,
                    "TreeStd_uncalibrated": float(sigma_raw[cand_i]),
                    "Calibrated_1sigma": float(sigma_cal[cand_i]),
                    "CI95_low": float(low),
                    "CI95_high": float(high),
                    "CI95_low_physical": float(max(0.0, low)),
                    "Calibration_a": a,
                    "Calibration_b": b,
                    "Model_mean_check": model_mean,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "expert_process_uncertainty.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(calibration_rows).to_csv(
        OUT_DIR / "current_pipeline_sigma_calibration.csv", index=False, encoding="utf-8-sig"
    )
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
