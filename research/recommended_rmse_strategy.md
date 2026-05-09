# Recommended RMSE Strategy

## Short Answer

If the main goal is the lowest possible RMSE, the best starting point from this analysis is:

1. Use `CatBoostRegressor`
2. Drop `id`
3. Normalize duplicated `ulke` labels
4. Keep categorical missing values as an explicit `"__MISSING__"` token
5. Either leave numeric missing values for CatBoost or median-fill them; in this dataset the difference was effectively zero
6. Clip final predictions to `[0, 10]`

## Best Practical Pipeline

### Data engineering

Apply these before modeling:

1. Remove `id` from the feature matrix.
2. Normalize duplicate country labels:
   - `Spain -> Ispanya`
   - `Sweden -> Isvec`
   - `South Korea -> Guney Kore`
3. Preserve all rows. Do not delete outliers aggressively.
4. Keep categorical values as strings and send missing categories to a dedicated `"__MISSING__"` bucket.

### Feature engineering

These are the highest-value derived features to try:

| Feature | Formula | Why it helps |
|---|---|---|
| `toplam_onarici_uyku_yuzdesi` | `rem_yuzdesi + derin_uyku_yuzdesi` | captures restorative sleep jointly |
| `uyku_bolunme_indeksi` | `uykuya_dalma_suresi_dk * (gecelik_uyanma_sayisi + 1)` | captures fragmentation burden |
| `stres_x_calisma` | `stres_skoru * gunluk_calisma_saati` | stress load is not independent of work time |
| `adim_stres_orani` | `gunluk_adim_sayisi / (stres_skoru + 1)` | steps appear more useful in low-stress contexts |
| `mutlak_hafta_sonu_uyku_farki` | `abs(hafta_sonu_uyku_farki_saat)` | inconsistency is more meaningful than sign |
| `kronotip_x_gun_tipi` | categorical cross | weekday/weekend effect likely depends on chronotype |
| `ruh_x_gun_tipi` | categorical cross | mental-health effect shifts by day context |

Important:

- These features improved weaker models more than they improved CatBoost.
- For CatBoost, keep only the engineered features that are interpretable and easy to maintain.

## Best Imputation Strategy

### For CatBoost

Best practical choice:

- Categorical columns: fill missing with `"__MISSING__"`
- Numeric columns: leave as `NaN` or median-fill

What the benchmark says:

- `CatBoost` with normalized categories: `1.21706`
- `CatBoost` with engineered features, missing flags, and numeric median fill: `1.21686`

That gain is only `0.00020` RMSE.

Recommendation:

- Do not spend early iteration time on KNN, iterative, or model-based imputation.
- On this dataset, complex imputation is almost certainly not where the leaderboard improvement is hiding.

### For sklearn pipelines

Best fallback choice:

- Numeric: median imputation
- Numeric missingness: add binary missing indicators
- Categorical: fill with `"__MISSING__"` and one-hot encode

Reason:

- Missingness is mildly informative.
- This is safer than mode-filling categories and pretending missingness carries no signal.

## Suggested CatBoost Starting Configuration

Use this as the first serious RMSE-chasing baseline:

```python
CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=2000,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=5.0,
    bagging_temperature=0.5,
    random_strength=0.5,
    grow_policy="SymmetricTree",
    od_type="Iter",
    od_wait=100,
    verbose=False,
    random_seed=42,
)
```

Practical note:

- In the local sweep, good runs usually stopped around `500` to `1100` trees.
- You should tune around this region, not far away from it.

## What To Prioritize Next

### Highest priority

1. CatBoost hyperparameter search around depth `5-7`, learning rate `0.03-0.06`, and moderate `l2_leaf_reg`
2. Multi-seed averaging of the top CatBoost configurations
3. Submission-time prediction clipping to `[0, 10]`

### Medium priority

1. Keep the best 3 to 6 engineered features only
2. Compare raw vs normalized categorical pipelines on the public leaderboard
3. Try fold-wise ensembling of 4 or 5 CatBoost models

### Low priority

1. Heavy outlier removal
2. Complex imputation
3. Making the final pipeline linear-model friendly

## What I Would Do First If I Were Optimizing This Competition

1. Build a CatBoost baseline on normalized raw data.
2. Tune it carefully with early stopping.
3. Add only `toplam_onarici_uyku_yuzdesi`, `uyku_bolunme_indeksi`, and `ruh_x_gun_tipi`.
4. Average predictions from 3 to 5 strong CatBoost seeds.
5. Keep a clean CV protocol and do not overreact to tiny leaderboard noise.

## Bottom Line

The dataset does not need exotic cleaning. The fastest path to a lower RMSE is a strong CatBoost pipeline with light category normalization, explicit handling of categorical missingness, and a small set of domain features centered on sleep quality, stress, and weekday/weekend context.
