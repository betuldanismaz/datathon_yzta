# YZTA Datathon 2026 - Cognitive Performance Regression

This repository contains Team 37's final notebook for the YZTA Datathon 2026 regression task. The objective is to predict `bilissel_performans_skoru` using the provided training and test data. The final workflow is documented in `YZTA_Datathon_Final.ipynb`, which covers data loading, exploratory analysis, leakage-aware preprocessing, feature engineering, model validation, ensemble construction, and final submission validation.

## Final Notebook

`YZTA_Datathon_Final.ipynb` is the final readable notebook for the project. The best-scoring modeling workflow was completed on May 13, 2026; afterward, the notebook was organized and documented more clearly without changing the modeling code, prediction logic, or final submission approach.

## Best Public Score

**Public RMSE: 1.20242**

## Final Submission

Final submitted file:

`submission_T1_T_constrained_optimizer_penalty_0_0.csv`

## Method Summary

The final solution uses a validation-driven regression workflow with:

- CatBoost-based primary modeling
- Domain-focused feature engineering, especially sleep-related features
- Fold-safe preprocessing with target encoding and frequency encoding
- K-Fold cross-validation and out-of-fold prediction tracking
- Conservative calibration and blending experiments
- LightGBM and XGBoost diversity checks for ensemble support
- Final R-anchored constrained optimizer ensemble
- Submission validation for row count, ID alignment, missing values, finite predictions, and valid prediction range

## Notebook Workflow

The final notebook is organized into the following stages:

1. Environment setup and data loading
2. Dataset understanding and exploratory data analysis
3. Preprocessing strategy
4. Feature engineering
5. Baseline model and validation framework
6. Main CatBoost model search
7. Post-processing and calibration experiments
8. Stacking, blending, and optimizer experiments
9. Final submission generation and validation

## Repository Structure

```text
catboost_info/                         CatBoost training logs
data/yzta-2026-datathon/               Competition data
  train.csv
  test_x.csv
research/                              Analysis notes and strategy documents
submission/                            Generated submission files and sample output files
LEGACY NOTEBOOKS/                      Earlier notebook versions
BEST_SCORE_NOTEBOOK.ipynb              Original best-score notebook
YZTA_Datathon_Final.ipynb              Final documented notebook
submission_T1_T_constrained_optimizer_penalty_0_0.csv
                                        Final submitted file for the reported public score
```

## How to Reproduce

1. Open `YZTA_Datathon_Final.ipynb`.
2. Ensure the competition data files are available:
   - `data/yzta-2026-datathon/train.csv`
   - `data/yzta-2026-datathon/test_x.csv`
3. If running outside Google Colab, update the notebook `data_path` variable to point to `data/yzta-2026-datathon/`.
4. Run the notebook cells in order.
5. Validate and generate the final submission CSV.

The notebook is written to preserve the competition submission format and produces `submission_T1_T_constrained_optimizer_penalty_0_0.csv` as the final submitted file.
