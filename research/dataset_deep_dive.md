# Dataset Deep Dive

## Snapshot

- `train.csv`: 56,000 rows, 24 columns
- `test_x.csv`: 24,000 rows, 23 columns
- Target: `bilissel_performans_skoru`
- Modeling features: 15 numeric + 7 categorical after dropping `id`
- No duplicate `id` values in train or test
- No duplicate feature rows in train or test
- No unseen categorical levels in `test_x.csv`

## Target Behavior

- Mean: `5.9131`
- Std: `2.2318`
- Median: `6.0322`
- Range: `0` to `10`
- Skew: `-0.2885` (slightly left-skewed, but not severe)
- `0.341%` of rows are exactly `0`
- `1.409%` of rows are exactly `10`

Implication:

- This is a bounded regression target, so clipping final predictions to `[0, 10]` is reasonable.
- The target is not skewed enough to justify a complex target transform as a first move.

## Missingness

Missingness is low to moderate and very stable between train and test.

| Feature | Train missing % | Test missing % |
|---|---:|---:|
| `kronotip` | 3.51 | 3.47 |
| `vucut_kitle_indeksi` | 3.13 | 2.70 |
| `stres_skoru` | 3.06 | 3.19 |
| `uyku_oncesi_kafein_mg` | 2.61 | 2.90 |
| `meslek` | 2.46 | 2.59 |
| `ruh_sagligi_durumu` | 1.96 | 2.10 |

Missingness is not fully random:

- `meslek` missing rows have target mean `6.0229` vs `5.9103` when present
- `kronotip` missing rows have target mean `6.0165` vs `5.9093` when present
- `uyku_oncesi_kafein_mg` missing rows have target mean `5.8377` vs `5.9151` when present

Implication:

- Treat missingness as signal, not just nuisance.
- For sklearn pipelines, add explicit missing indicators or a categorical `"__MISSING__"` level.
- For CatBoost, preserving missing structure is enough; heavy imputation is not required.

## Strongest Numeric Signal

Top Pearson correlations with the target:

| Feature | Pearson r |
|---|---:|
| `stres_skoru` | `-0.5867` |
| `rem_yuzdesi` | `0.4430` |
| `gunluk_calisma_saati` | `-0.3421` |
| `derin_uyku_yuzdesi` | `0.2800` |
| `gecelik_uyanma_sayisi` | `-0.2703` |
| `uykuya_dalma_suresi_dk` | `-0.2073` |
| `gunluk_adim_sayisi` | `0.1340` |

Near-zero standalone linear signal:

- `hafta_sonu_uyku_farki_saat`: `-0.0016`
- `oda_sicakligi_celsius`: `-0.0120`
- `sekerleme_suresi_dk`: `-0.0237`

Implication:

- Stress, REM sleep, work hours, deep sleep, wakeups, and sleep latency are the core numeric signal.
- Low univariate correlation does not mean a feature is useless. Some weak standalone features can still matter through interactions in tree models.

## Strongest Categorical Signal

### `ruh_sagligi_durumu`

| Level | Count | Mean target |
|---|---:|---:|
| `Saglikli` | 38,365 | `6.4918` |
| `Anksiyete` | 8,270 | `5.0123` |
| `Depresyon` | 5,504 | `4.3944` |
| `Anksiyete ve depresyon` | 2,765 | `3.5902` |

### `gun_tipi`

| Level | Count | Mean target |
|---|---:|---:|
| `Hafta sonu` | 15,948 | `6.9933` |
| `Hafta ici` | 40,052 | `5.4830` |

### `kronotip`

| Level | Count | Mean target |
|---|---:|---:|
| `Sabah insani` | 13,500 | `5.9956` |
| `Notr` | 24,594 | `5.9345` |
| `Gece insani` | 15,938 | `5.7973` |

### `meslek`

Highest-target groups:

- `Emekli`: `7.9350`
- `Ev Hanimi`: `7.0560`
- `Serbest Calisan`: `6.9855`

Lower-target groups:

- `Yonetici`: `5.7298`
- `Satis ve Pazarlama Calisani`: `6.0103`

Implication:

- Mental health and weekday/weekend context are major drivers.
- Chronotype and occupation matter enough to keep and potentially interact with `gun_tipi`.

## Train/Test Shift

Numeric train/test drift is extremely small.

Largest standardized mean differences:

- `vucut_kitle_indeksi`: `-0.0181`
- `rem_yuzdesi`: `-0.0130`
- `dinlenik_nabiz_bpm`: `0.0119`
- `derin_uyku_yuzdesi`: `-0.0108`

All numeric shift magnitudes are below `0.02`.

Implication:

- Random K-fold CV should be representative.
- There is no obvious distribution-shift warning in the provided holdout/test feature file.

## Data Engineering Opportunities

### 1. Normalize duplicated country labels

`ulke` mixes Turkish and English labels for the same country:

| Raw label pair | Counts |
|---|---:|
| `Spain` / `Ispanya` | `1139 / 2823` |
| `Sweden` / `Isvec` | `566 / 2798` |
| `South Korea` / `Guney Kore` | `2189 / 4459` |

This should be normalized before modeling to avoid splitting the same concept across two categories.

### 2. Standardize categorical text

- `meslek` is mostly Turkish but includes `Lawyer` in English (`2742` rows)
- There is no clear duplicate occupation label for `Lawyer`, so it should stay as-is unless you build a full translation map

### 3. Do not delete outliers aggressively

Ranges look plausible:

- `uyku_oncesi_kafein_mg`: `0` to `400`
- `gunluk_adim_sayisi`: `500` to `20000`
- `gunluk_calisma_saati`: `0` to `18`
- `hafta_sonu_uyku_farki_saat`: `-1` to `3`

These are long-tailed in places, but not obviously invalid. Prefer log transforms for linear models and robust tree models for final RMSE chasing.

## Main Takeaways

1. The dataset is clean enough for direct modeling. The biggest wins will come from model choice, not heavy data cleaning.
2. The strongest signal lives in stress, sleep architecture, weekday/weekend context, and mental health.
3. Missingness is mild but informative, so preserve it.
4. Train/test alignment is strong, which makes CV trustworthy.
5. Category normalization is worth doing, but it is a second-order improvement compared with choosing the right model.
