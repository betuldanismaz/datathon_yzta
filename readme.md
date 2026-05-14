# YZTA Datathon 2026 - Cognitive Performance Regression

This repository contains Team 37's final documented notebook for the YZTA Datathon 2026 regression task. The objective is to predict the target variable `bilissel_performans_skoru` from sleep, lifestyle, demographic, and physiological features. The final workflow is implemented in `YZTA_Datathon_Final.ipynb` and covers the full modeling process from data loading and exploratory analysis through validation, model search, calibration, ensembling, and final submission checks.

## Final Notebook

`BEST_SCORE_NOTEBOOK.ipynb` contains the best-scoring modeling workflow and was created before May 13, 2026. The same workflow was later reorganized with clearer explanations and comments under the file name `YZTA_Datathon_Final.ipynb`. This final notebook is the documented, readable version of the best-score workflow; the modeling code, prediction logic, and final submission approach were not changed after May 13, 2026.

## Best Public Score

**Public RMSE: 1.20242**

This is the reported public leaderboard RMSE for the final submitted file in this repository. No leaderboard rank is claimed.

## Final Submission

Final submitted file:

`submission_T1_T_constrained_optimizer_penalty_0_0.csv`

The final notebook validates this file for submission format, row count, ID alignment, missing values, finite predictions, and valid prediction range.

## Problem Setup

- **Task type:** Supervised regression
- **Target variable:** `bilissel_performans_skoru`
- **Metric:** RMSE
- **ID column:** `id`
- **Main notebook:** `YZTA_Datathon_Final.ipynb`
- **Final submission:** `submission_T1_T_constrained_optimizer_penalty_0_0.csv`

The notebook treats `id` only as a submission identifier and does not use it as a model feature. The target column is used only for supervised training and validation.

## Method Summary

The final workflow is validation-driven and leakage-aware. The core modeling approach is based on CatBoost, supported by fold-safe preprocessing, domain-focused feature engineering, cross-validation, out-of-fold prediction tracking, conservative calibration, and constrained optimizer ensembling.

Main components:

- CatBoost-based primary regression models
- Sleep-related and domain-driven feature engineering
- Fold-safe target encoding and frequency encoding
- K-Fold cross-validation with out-of-fold predictions
- Multi-seed model stability checks
- Conservative calibration and residual correction experiments
- LightGBM and XGBoost diversity models for ensemble testing
- Anchored blending and constrained optimizer ensemble
- Final submission validation before output

## Detailed Notebook Walkthrough

### Step 1 - Environment Setup and Data Loading

This step prepares the notebook environment and loads the competition data. It is designed primarily for Google Colab, where the Kaggle API can be used to retrieve the data directly.

#### Step 1.1 - Kaggle Login

The notebook authenticates with Kaggle so the competition files can be downloaded from within the runtime. This step confirms that Kaggle credentials are available and valid.

#### Step 1.2 - Download Competition Data

The dataset is downloaded using `kagglehub.competition_download`. The downloaded path is stored for later file organization and loading.

#### Step 1.3 - File Organization

The downloaded CSV files are copied into a consistent working directory. This keeps file loading predictable and avoids path mismatches in later cells.

#### Step 1.4 - Verify and Load Data

The notebook checks that the required data files exist before continuing. This prevents later modeling cells from failing due to missing input files.

#### Step 1.5 - Load DataFrames

The files are loaded into pandas DataFrames:

- `train.csv` as `train`
- `test_x.csv` as `test`
- `sample_submission.csv` as `sample_submission`

These DataFrames are used throughout the notebook for analysis, training, validation, prediction, and final submission formatting.

#### Step 1.6 - Initial Structural Inspection

The notebook checks the shape of each DataFrame and verifies the basic structure of the train, test, and sample submission files. This confirms that the expected number of rows and columns are available before modeling begins.

#### Step 1.7 - Inspect Column Names

The column names of `train`, `test`, and `sample_submission` are printed. This confirms that all expected feature columns are present and that the target and ID columns are handled correctly.

#### Step 1.8 - Verify Target and ID Handling

The notebook verifies that:

- `bilissel_performans_skoru` exists in the training data
- `bilissel_performans_skoru` does not exist in the test feature data
- `id` exists where needed for submission mapping

This step reduces the risk of target leakage and submission ID mismatch.

### Step 2 - Dataset Understanding and Exploratory Data Analysis

This step inspects the data before modeling. The goal is to understand feature types, missing values, target distribution, and categorical behavior.

#### Step 2.1 - Dataset Structure Overview

The notebook runs structural checks such as `train.info()` to inspect column names, non-null counts, and data types. This helps identify which columns should be treated as numeric or categorical.

#### Step 2.2 - Target Distribution Analysis

The target variable `bilissel_performans_skoru` is summarized with descriptive statistics and visualized with a histogram. This helps understand its range, spread, and distribution before selecting models and post-processing strategies.

#### Step 2.3 - Feature Type Identification

Features are separated into numeric and categorical groups. This distinction is important because numeric features and categorical features require different preprocessing and modeling strategies.

#### Step 2.4 - Missing Value Analysis

Missing values are counted across train and test data. These checks guide imputation and model selection decisions, especially because missingness can affect validation stability.

#### Step 2.5 - Categorical Feature Summaries

The notebook reviews categorical feature cardinality and frequency distributions. This helps identify rare categories, high-cardinality columns, and possible train-test category mismatches.

#### Step 2 Summary

The exploratory analysis provides the basis for the preprocessing and feature engineering strategy. CatBoost is selected as a strong primary model because it can handle categorical features effectively and performs well in tabular regression settings.

### Step 3 - Preprocessing Strategy

This step defines reusable preprocessing logic. The key goal is to avoid leakage by ensuring that fold-specific statistics are computed only from the training portion of each validation fold.

#### Step 3.1 - Initial Categorical Data Preparation

Categorical columns are identified and prepared for consistent handling. This ensures that downstream encoders and CatBoost categorical feature indices remain aligned.

#### Step 3.2 - Categorical Variable Standardization

Categorical values are standardized so that equivalent categories are represented consistently. This reduces noise from formatting differences and improves encoding reliability.

#### Step 3.3 - Fold-Safe Frequency Encoding Helper

A helper function is defined for frequency encoding. Frequencies are computed inside each fold using only the fold's training data, preventing validation leakage.

#### Step 3.4 - Fold-Safe Target Encoding Helper

A target encoding helper is defined for categorical variables. Because target encoding can easily leak information, the notebook computes target statistics fold-by-fold instead of using the full training set during validation.

#### Step 3.5 - Unified Fold-Safe Feature Engineering Evaluator

The notebook defines a reusable evaluator that can apply target encoding, frequency encoding, and feature engineering inside each cross-validation fold. This allows preprocessing and feature choices to be compared fairly.

#### Step 3 Summary

The preprocessing framework is built around reproducibility and leakage prevention. These utilities are reused in feature engineering, model tuning, and final prediction generation.

### Step 4 - Feature Engineering

This step evaluates domain-driven features before model tuning. Since the target is cognitive performance, the notebook focuses especially on sleep and lifestyle-related signals.

#### Step 4.1 - Sleep-Related Domain Feature Evaluation

The notebook tests additional sleep-related features using the same cross-validation framework. This checks whether engineered features improve validation RMSE rather than adding them based only on intuition.

#### Step 4.2 - Combined Feature Set Comparison

Different feature set combinations are compared using validation results. The notebook avoids blindly keeping every engineered feature and instead selects the setup that performs best under cross-validation.

#### Step 4.3 - Final Feature Set Preparation for Tuning

The selected feature set is prepared for CatBoost tuning. This includes aligning feature columns, categorical feature indices, and encoding settings.

#### Step 4.4 - Feature Engineering Selection Check

The notebook verifies that the selected engineered features and categorical feature references are correctly aligned before model training continues.

#### Step 4 Summary

The final feature set is chosen based on validation performance. Sleep-related engineered features are retained because they provide useful predictive signal in the validated workflow.

### Step 5 - Baseline Model and Validation Framework

This step defines the validation framework used throughout the notebook. RMSE is the main metric, and cross-validation is used to produce more stable estimates than a single train-validation split.

#### Step 5.1 - RMSE Evaluation Function

The notebook defines a reusable RMSE evaluation function. It is used consistently across CatBoost models, calibration tests, blends, and optimizer candidates.

#### Step 5.2 - Core CatBoost Cross-Validation Function

A CatBoost cross-validation function is created to train and evaluate models across folds. It reports fold-level RMSE and supports consistent comparison between parameter configurations.

#### Step 5 Summary

This validation framework becomes the backbone of the notebook. Later modeling and ensembling decisions are based on OOF and cross-validation behavior rather than test data feedback alone.

### Step 6 - Main CatBoost Model Search

This step performs the main CatBoost model search after preprocessing and feature engineering decisions are established.

#### Step 6.1 - CatBoost Hyperparameter Screening

Several CatBoost configurations are evaluated through cross-validation. The search focuses on strong tabular regression settings while controlling overfitting through depth, learning rate, regularization, and early stopping behavior.

#### Step 6.2 - Best CatBoost Configuration Selection

The best-performing CatBoost configuration is selected based on validation RMSE. This selected configuration becomes the main model anchor for later experiments.

#### Step 6.3 - Champion Pipeline Review

The notebook reviews the current champion configuration and confirms that it is the correct anchor for downstream prediction, calibration, and blending.

#### Step 6.4 - OOF and Test Prediction Evaluator

A function is defined to generate both out-of-fold predictions and averaged test predictions. OOF predictions are required for calibration, blending, and optimizer experiments because they provide validation-aligned predictions for every training row.

#### Step 6.5 - Fold-Safe Encoded CatBoost Evaluator

The notebook applies fold-safe encoding inside the CatBoost evaluation process. This keeps target-sensitive transformations inside each fold and improves the reliability of validation results.

#### Step 6.6 - Additional CatBoost Regularization Tuning

The strongest CatBoost configurations are refined with additional regularization checks. The goal is to improve RMSE while limiting overfitting.

#### Step 6.7 - Multi-Seed Stability Confirmation

The selected CatBoost setup is evaluated across multiple random seeds. Averaging across seeds reduces prediction variance and helps confirm that the model is not dependent on a single lucky split or seed.

#### Step 6.8 - Depth5 Stable Test Prediction Generation

The stable CatBoost configuration generates averaged predictions for the test set. These predictions serve as a strong standalone candidate and as an anchor for later post-processing.

#### Step 6.9 - Depth5 Stable Submission File Creation

The notebook creates a valid submission file from the stable CatBoost predictions. It preserves the expected `id` and `bilissel_performans_skoru` columns.

#### Step 6.10 - Depth5 Stable Submission Validation

The generated CatBoost submission is validated for required columns, row count, ID alignment, missing values, duplicate IDs, and expected prediction range.

#### Step 6.11 - Save Prediction Artifacts for Later Experiments

OOF and test prediction artifacts are saved or kept available for calibration, residual correction, blending, and optimizer-based ensembling.

#### Step 6 Summary

Step 6 produces the main CatBoost anchor model and the prediction artifacts needed for the rest of the notebook.

### Step 7 - Post-Processing and Calibration Experiments

This step evaluates conservative post-processing around the CatBoost anchor. The intent is to improve validation behavior without making predictions unstable.

#### Step 7.1 - Safe OOF Calibration Search

The notebook searches for small calibration adjustments using OOF predictions. Calibration is intentionally conservative because aggressive adjustment can overfit local validation behavior.

#### Step 7.2 - Calibrated Submission Creation

The best supported calibration is applied to test predictions and converted into a submission candidate.

#### Step 7.3 - Calibration Report and Validation

The calibration result is saved and validated. The notebook compares the calibrated candidate against the CatBoost anchor to decide whether it is useful.

#### Step 7.4 - Residual Bin Correction Setup

OOF predictions, target values, and test predictions are prepared for residual-bin correction. This setup allows the notebook to inspect whether prediction errors vary systematically across prediction ranges.

#### Step 7.5 - Residual Bin Correction Search

The notebook searches for bin-level corrections based on OOF residuals. This attempts to reduce underprediction or overprediction patterns in specific prediction bands.

#### Step 7.6 - Bin-Corrected Prediction and Submission Creation

The selected residual-bin correction is applied to predictions and saved as a submission candidate.

#### Step 7.7 - Bin-Correction Report and Safety Check

The bin-corrected candidate is validated and documented. Safety checks ensure it remains a valid competition submission.

#### Step 7.8 - Conservative Blend with Bin-Corrected Candidate

The notebook tests a blend between the original CatBoost prediction and the bin-corrected prediction. This reduces risk by keeping the output close to the stronger anchor.

#### Step 7.9 - Conservative Blend with Global Calibration

The notebook tests a conservative blend between the original CatBoost prediction and the globally calibrated prediction. This creates another low-risk post-processing candidate.

#### Step 7 Summary

Step 7 documents calibration and residual-correction attempts. These candidates are evaluated with OOF RMSE and retained as part of the transparent model-selection process.

### Step 8 - Stacking, Blending, and Optimizer Experiments

This is the main ensemble construction stage. It collects validated prediction artifacts and tests whether combining models improves OOF RMSE while keeping predictions stable.

#### Step 8.1 - Candidate Artifact Setup for Blending

The notebook loads and validates candidate OOF and test predictions. This establishes a clean set of artifacts for blending.

#### Step 8.2 - Alternative CatBoost Diversity Candidate

An alternative CatBoost candidate is created or inspected to test whether it adds prediction diversity beyond the main CatBoost anchor.

#### Step 8.3 - Two-Way and Three-Way OOF Blend Search

The notebook searches blend weights between selected candidates. The search uses OOF RMSE so that blend selection is grounded in validation performance.

#### Step 8.4 - Blend Submission Creation and Report

The selected blend candidate is converted into a submission file, and blend results are saved for traceability.

#### Step 8.5 - LightGBM Diversity Model and Blend

LightGBM is tested as a different model family. Although it is not the primary model, it can add diversity to an ensemble when blended conservatively with CatBoost predictions.

#### Step 8.6 - Overnight Diagnostic and Feature-Set Experiments

The notebook performs broader diagnostic checks and feature-set screening. These experiments help determine whether additional signal exists beyond the current champion setup.

#### Step 8.7 - Overnight Stacking and Conservative Manual Blend

Available prediction artifacts are collected for stacking and manual blending. Conservative manual blends are used to avoid overly flexible stacking that may overfit OOF predictions.

#### Step 8.8 - Final-Day Residual and Calibration Experiments

The notebook tests additional residual modeling, calibration, and smoothing strategies around the current champion prediction. Only useful candidates are carried forward into later selection.

#### Step 8.9 - Final Candidate Artifact Collection

Completed prediction artifacts are audited. Weak, duplicate, or incomplete candidates are filtered out so the final optimizer uses only reliable sources.

#### Step 8.10 - Conservative Anchored Blend and Constrained Optimizer

The notebook performs anchored blend search and constrained optimization. The optimizer is restricted to keep predictions close to strong anchor candidates while searching for OOF-supported improvement.

#### Step 8.11 - Stability-Selected Final-Day Candidates

Candidate submissions are evaluated with stability considerations, not only raw OOF RMSE. This helps reduce the risk of selecting a candidate that improves validation by noise.

#### Step 8.12 - Robust CatBoost Anchored Blend

A robust CatBoost variant is evaluated and blended with the current stack anchor. This adds another controlled prediction source to the ensemble.

#### Step 8.13 - XGBoost Diversity Blend

XGBoost is tested as another non-CatBoost diversity model. It is blended only when OOF validation suggests it can improve the ensemble.

#### Step 8.14 - T-Family R-Anchored Final Optimizer

The notebook runs the T-family R-anchored final optimizer. This stage produces the final submitted file:

`submission_T1_T_constrained_optimizer_penalty_0_0.csv`

The selected T1 candidate has:

- **Public RMSE:** `1.20242`
- **OOF RMSE:** approximately `1.213209338`

#### Step 8.15 - U-Family Public-Validated Local Optimizer Check

The notebook tests local optimizer variants around the public-validated T1 candidate. These checks are documented, but the final submitted file remains the T1 constrained optimizer candidate.

#### Step 8 Summary

Step 8 combines CatBoost, LightGBM, XGBoost, calibration outputs, stack candidates, anchored blends, and constrained optimizer candidates. The final selected submission is the conservative T1 constrained optimizer file.

### Step 9 - Final Submission Generation

This step verifies the exact final submission file and prepares it for download.

#### Step 9.1 - Final Submission Validation and Download

The notebook checks that `submission_T1_T_constrained_optimizer_penalty_0_0.csv` exists and validates:

- required columns
- row count
- ID alignment with the test data
- duplicate IDs
- missing predictions
- finite prediction values
- prediction range

After validation, the notebook downloads the final CSV in the Colab environment.

#### Step 9 Summary

The final output is a validated submission CSV matching the required competition format.

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
   - `sample_submission.csv` in the notebook data path if running the full Colab workflow locally
3. If running outside Google Colab, update the notebook `data_path` variable to point to the local data directory.
4. Install or enable the required libraries used in the notebook, including `pandas`, `numpy`, `scikit-learn`, `catboost`, `lightgbm`, `xgboost`, `matplotlib`, and `seaborn`.
5. Run the notebook cells in order from Step 1 through Step 9.
6. Confirm that the final validation cell passes.
7. Use `submission_T1_T_constrained_optimizer_penalty_0_0.csv` as the final generated submission file.

## Notes

- The notebook is designed to avoid target leakage by keeping target-dependent preprocessing inside validation folds.
- Public leaderboard feedback is reported only as the achieved public RMSE for the final submitted file.
- Earlier notebooks are retained under `LEGACY NOTEBOOKS/` for project history, but `YZTA_Datathon_Final.ipynb` is the final documented workflow.
