## ZnBERT-guided design of Zn alloys with optimized mechanical performance for biomedical applications


The complete pretrained ZnBERT is hosted at Hugging Face: https://huggingface.co/XuQin/ZnBERT.


## Example analyses

Run scripts from the repository root so that relative paths resolve consistently.

Compare downstream models:

```bash
python scripts/stage02_compare_downstream_models_v2.py
```

Calculate XGBoost SHAP importance:

```bash
python scripts/stage03_xgb_shap_importance.py
```

Test whether ZnBERT uses linguistic context and word order:

```bash
python scripts/stage04_semantic_order_ablation.py
python scripts/stage04_protected_phrase_shuffle.py
python scripts/stage04_xgb_feature_order_shuffle.py
```

Perform CPI-based screening:

```bash
python scripts/stage05_rank_cpi_and_plot.py
```

Estimate predictive uncertainty:

```bash
python scripts/stage06_cv_uncertainty.py
python scripts/stage06_calibrate_expert_uncertainty.py
```

Before running an analysis, review the paths and experiment settings near the beginning of the corresponding script. Some analyses are computationally intensive and cache embeddings to avoid repeated encoder inference.

## Semantic robustness analysis

Conventional tabular machine-learning models are invariant to the physical order of feature columns when the model is retrained with the same feature–value mapping. Language encoders, however, can respond to changes in token order because self-attention generates contextual representations.

The stage-4 experiments distinguish among:

1. the original composition–processing sentence;
2. complete token shuffling;
3. protected-phrase shuffling, which keeps physically meaningful units such as `0.25 wt% Mg`, `Extrusion T=260 C`, and `AR=20` intact;
4. structured XGBoost feature-column shuffling as a control.

This design tests whether ZnBERT's predictive contribution arises only from the presence of numerical and chemical tokens or also from their contextual organization.

## CPI-based alloy screening

Candidate alloys are ranked using a comprehensive performance index:

```text
CPI = UTS_norm + YS_norm + EL_norm
```

The screening scripts combine model predictions with composition/process constraints and expert feasibility assessment. Model-ranked candidates should always be validated experimentally before biomedical interpretation or application.

## Uncertainty

The uncertainty workflow extracts predictions from individual estimators/trees in the selected ensemble, calculates their dispersion, and calibrates the resulting standard deviation against cross-validated residuals. Results can be reported as:

```text
prediction ± calibrated 1σ (95% confidence interval)
```

The calibrated intervals quantify model uncertainty under the implemented validation setting; they do not replace experimental error analysis or guarantee reliability outside the training domain.


## Citation

The associated manuscript is in preparation:


