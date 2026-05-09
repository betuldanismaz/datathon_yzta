# Model Benchmarks

## Setup

- CV scheme: 4-fold `KFold(shuffle=True, random_state=42)`
- Metric: RMSE
- `id` excluded from features in every run
- All CatBoost predictions clipped to `[0, 10]`

Benchmarked families:

- Dummy mean baseline
- Ridge with one-hot encoding
- HistGradientBoostingRegressor with one-hot encoding
- CatBoostRegressor with native categorical handling

## CV Results

| Rank | Model / Variant | RMSE mean | RMSE std |
|---|---|---:|---:|
| 1 | `CatBoost` + normalized categories + engineered features + missing flags + numeric median fill | `1.21686` | `0.01018` |
| 2 | `CatBoost` + normalized categories | `1.21706` | `0.01021` |
| 3 | `CatBoost` + raw categories | `1.21707` | `0.01054` |
| 4 | `CatBoost` + normalized categories + engineered features + missing flags | `1.21808` | `0.01014` |
| 5 | `HistGradientBoosting` + normalized categories + engineered features + missing flags | `1.23019` | `0.01093` |
| 6 | `HistGradientBoosting` + raw features | `1.23083` | `0.00950` |
| 7 | `Ridge` + normalized categories + engineered features + missing flags | `1.25146` | `0.01211` |
| 8 | `Ridge` + raw features | `1.25628` | `0.01210` |
| 9 | Dummy mean baseline | `2.23174` | `0.00847` |

## What Matters

### 1. Model family matters more than feature tinkering

- Best CatBoost beats best HistGradientBoosting by about `0.0133` RMSE
- Best CatBoost beats best Ridge by about `0.0346` RMSE
- Best non-trivial model beats the dummy baseline by about `1.0149` RMSE

Conclusion:

- Most of the gain comes from using a strong nonlinear tabular learner with categorical support.
- Trying to win this competition with a linear stack is the wrong optimization target if your only goal is minimum RMSE.

### 2. CatBoost is on a performance plateau across sensible variants

The top four CatBoost runs are extremely close:

- `1.21686`
- `1.21706`
- `1.21707`
- `1.21808`

Conclusion:

- Category normalization and engineered features help a little or stay neutral, but they do not change the leaderboard dramatically.
- This usually means the dataset signal is already easy for CatBoost to capture from the raw columns.

### 3. Feature engineering helps linear models more than CatBoost

Ridge improved from `1.25628` to `1.25146` after engineered features and missing flags.

Conclusion:

- Manual feature engineering is useful when the model is weak at learning interactions.
- For CatBoost, only keep engineered features that are domain-sensible and cheap to maintain.

### 4. Missing-value treatment is a low-impact decision here

CatBoost comparison:

- Native-style handling with normalized categories: `1.21706`
- Same plus engineered features and numeric median fill: `1.21686`

Difference: `0.00020` RMSE.

Conclusion:

- Missingness is too small for complex imputation to become a major differentiator.
- Simplicity is acceptable unless leaderboard feedback shows otherwise.

## Narrow CatBoost Sweep

I also ran a smaller 3-fold sweep around the best CatBoost region.

Best configuration from that sweep:

- `depth=6`
- `learning_rate=0.04`
- `l2_leaf_reg=5.0`
- `bagging_temperature=0.5`
- `random_strength=0.5`
- early stopping typically landed around `793` to `869` trees

Observed 3-fold RMSE for that tuned region:

- best mean from the sweep: `1.21810`
- other nearby settings stayed within about `0.0016` RMSE

Conclusion:

- You are not looking for a radically different CatBoost configuration.
- The useful search band is narrow; focus on depth `5-7`, learning rate `0.03-0.06`, and moderate regularization.

## Feature Importance From Full-Data CatBoost Fit

Top features from the final full-data fit:

| Feature | Importance |
|---|---:|
| `stres_skoru` | `18.92` |
| `ruh_sagligi_durumu` | `17.09` |
| `toplam_onarici_uyku_yuzdesi` | `15.38` |
| `gun_tipi` | `12.24` |
| `meslek` | `8.47` |
| `rem_yuzdesi` | `8.10` |
| `uyku_bolunme_indeksi` | `5.62` |
| `adim_stres_orani` | `2.50` |
| `gecelik_uyanma_sayisi` | `2.49` |
| `ruh_x_gun_tipi` | `2.06` |

Interpretation:

- Stress and mental-health context dominate the target.
- Sleep quality is better captured as a combined structure than as isolated columns.
- Weekday/weekend context interacts strongly with other lifestyle variables.

## Benchmark Takeaways

1. If you want the best RMSE with the least wasted effort, start with CatBoost.
2. Category cleanup is worth doing, but do not expect large gains from it alone.
3. Missing-value strategy is not the main bottleneck.
4. Domain features are useful, but mostly as second-pass refinements after the CatBoost baseline is stable.
