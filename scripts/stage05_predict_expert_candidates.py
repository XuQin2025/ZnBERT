# -*- coding: utf-8 -*-
"""Predict Zn-0.25Mg-0.2Li-2.3Cu at two extrusion conditions.

The model and uncertainty definition are kept identical to
``predict_two_alloys_znbxgb_uncertainty.py``:
  - ZnBERTv2 checkpoint-1200 mean-pooled composition/process embeddings
  - one full-data XGBRegressor per target
  - affine calibration of per-tree spread from the existing 5-fold OOF run
  - 95% interval = prediction +/- 1.96 * calibrated sigma
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stage06_predict_two_alloys_uncertainty import (
    estimator_tree_sigma,
    make_xgb,
    seed_everything,
)
from stage02_compare_downstream_models_v2 import (
    CACHE_DIR,
    COMPOSITION_COLS,
    DATA_XLSX,
    SHEET,
    Y_COLS,
    ZNBERT_PATH,
    build_comp_sentence,
    build_proc_sentence,
    get_embeddings,
)


RANDOM_SEED = 42
OUT_DIR = Path("outputs") / "znbxgb_zn025mg_02li_23cu_two_extrusions"
CALIBRATION_FILE = (
    Path("outputs")
    / "znbxgb_two_alloy_experiment_compare"
    / "sigma_calibration.csv"
)


def make_candidate(name, extrusion_t, extrusion_ar):
    row = {column: 0.0 for column in COMPOSITION_COLS}
    row.update(
        {
            "Alloy": name,
            "Mg (wt%)": 0.25,
            "Li (wt%)": 0.2,
            "Cu(wt%)": 2.3,
            "Processing": (
                f"HotExtrusion (T={extrusion_t} degC; "
                f"AR_total={extrusion_ar}_to_1)"
            ),
            "has_extrusion": 1,
            "extrusion_T": extrusion_t,
            "extrusion_AR": extrusion_ar,
            "extrusion_S": np.nan,
        }
    )
    return row


def main():
    seed_everything(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
    for column in COMPOSITION_COLS + Y_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=Y_COLS).reset_index(drop=True)
    df["CompSentence"] = df.apply(build_comp_sentence, axis=1)
    df["ProcSentence"] = df.apply(build_proc_sentence, axis=1)

    comp_cache = os.path.join(CACHE_DIR, "comp_emb_ZnBERV2.npy")
    proc_cache = os.path.join(CACHE_DIR, "proc_emb_ZnBERTV2.npy")
    emb_comp = get_embeddings(df["CompSentence"].tolist(), ZNBERT_PATH, comp_cache)
    emb_proc = get_embeddings(df["ProcSentence"].tolist(), ZNBERT_PATH, proc_cache)
    x_train = np.hstack([emb_comp, emb_proc]).astype(np.float32)
    y = df[Y_COLS].values.astype(float)

    candidates = pd.DataFrame(
        [
            make_candidate(
                "Zn-0.25Mg-0.2Li-2.3Cu (260C/20:1)",
                extrusion_t=260,
                extrusion_ar=20,
            ),
            make_candidate(
                "Zn-0.25Mg-0.2Li-2.3Cu (240C/25:1)",
                extrusion_t=240,
                extrusion_ar=25,
            ),
        ]
    )
    candidates["CompSentence"] = candidates.apply(build_comp_sentence, axis=1)
    candidates["ProcSentence"] = candidates.apply(build_proc_sentence, axis=1)

    cand_comp = get_embeddings(
        candidates["CompSentence"].tolist(),
        ZNBERT_PATH,
        str(OUT_DIR / "candidate_comp_embeddings.npy"),
    )
    cand_proc = get_embeddings(
        candidates["ProcSentence"].tolist(),
        ZNBERT_PATH,
        str(OUT_DIR / "candidate_proc_embeddings.npy"),
    )
    x_candidate = np.hstack([cand_comp, cand_proc]).astype(np.float32)

    calibration_df = pd.read_csv(CALIBRATION_FILE)
    calibration = calibration_df.set_index("Target")[["a", "b"]].to_dict("index")

    rows = []
    tree_rows = []
    for target_i, target in enumerate(Y_COLS):
        model = make_xgb(RANDOM_SEED)
        model.fit(x_train, y[:, target_i])

        prediction = model.predict(x_candidate)
        sigma_raw, tree_values = estimator_tree_sigma(model, x_candidate)
        sigma_cal = (
            calibration[target]["a"] * sigma_raw
            + calibration[target]["b"]
        )
        ci95_low = prediction - 1.96 * sigma_cal
        ci95_high = prediction + 1.96 * sigma_cal

        for candidate_i, alloy in enumerate(candidates["Alloy"]):
            rows.append(
                {
                    "Alloy": alloy,
                    "Target": target,
                    "Prediction": float(prediction[candidate_i]),
                    "TreeStd_raw": float(sigma_raw[candidate_i]),
                    "Uncertainty_1sigma_calibrated": float(
                        sigma_cal[candidate_i]
                    ),
                    "CI95_half_width": float(1.96 * sigma_cal[candidate_i]),
                    "CI95_low": float(ci95_low[candidate_i]),
                    "CI95_high": float(ci95_high[candidate_i]),
                }
            )
            for tree_i, tree_value in enumerate(
                tree_values[candidate_i], start=1
            ):
                tree_rows.append(
                    {
                        "Alloy": alloy,
                        "Target": target,
                        "Tree": tree_i,
                        "TreeValue": float(tree_value),
                    }
                )

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot(
        index="Alloy",
        columns="Target",
        values=[
            "Prediction",
            "Uncertainty_1sigma_calibrated",
            "CI95_half_width",
            "CI95_low",
            "CI95_high",
        ],
    )
    wide_df.columns = [
        f"{target}_{measure}" for measure, target in wide_df.columns
    ]
    wide_df = wide_df.reset_index()

    input_df = candidates[
        [
            "Alloy",
            "CompSentence",
            "ProcSentence",
            "extrusion_T",
            "extrusion_AR",
        ]
    ].copy()
    notes_df = pd.DataFrame(
        [
            {"item": "data", "value": DATA_XLSX},
            {"item": "training_rows", "value": len(df)},
            {"item": "znb_path", "value": ZNBERT_PATH},
            {
                "item": "model",
                "value": (
                    "ZnBERT composition/process mean-pooled embeddings "
                    "+ full-data XGBRegressor"
                ),
            },
            {
                "item": "uncertainty",
                "value": (
                    "Per-tree spread calibrated by existing 5-fold OOF "
                    "residuals; 95% interval = prediction +/- 1.96 sigma"
                ),
            },
            {
                "item": "candidate_assumption",
                "value": (
                    "Extrusion only; no homogenization/annealing/other "
                    "process; extrusion speed unknown and omitted"
                ),
            },
            {
                "item": "xgb_parameters",
                "value": json.dumps(make_xgb().get_params(), sort_keys=True),
            },
        ]
    )

    long_df.to_csv(
        OUT_DIR / "predictions_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wide_df.to_csv(
        OUT_DIR / "predictions_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(tree_rows).to_csv(
        OUT_DIR / "tree_values.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(
        OUT_DIR / "ZnBERT_XGB_Cu_two_extrusions_predictions.xlsx",
        engine="openpyxl",
    ) as writer:
        wide_df.to_excel(writer, sheet_name="predictions_wide", index=False)
        long_df.to_excel(writer, sheet_name="predictions_long", index=False)
        input_df.to_excel(writer, sheet_name="model_inputs", index=False)
        calibration_df.to_excel(
            writer, sheet_name="uncertainty_calibration", index=False
        )
        notes_df.to_excel(writer, sheet_name="notes", index=False)

    print("\n[MODEL INPUTS]")
    print(input_df.to_string(index=False))
    print("\n[PREDICTIONS]")
    print(long_df.round(3).to_string(index=False))
    print(f"\n[OK] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
