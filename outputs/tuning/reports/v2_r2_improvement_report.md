# Second round (v2): reducing the R² collapse

## Diagnosis (validation 2024 only)
R² on the SAR scale is dominated by a handful of giant Commercial land parcels: about 100 of 247,088
validation rows carry over 90% of the squared error, and the trees over-predict such rows by orders of
magnitude by copying the few huge training prices. On the ln(price) scale the same models score about
0.70-0.72, so relative accuracy is reasonable; the SAR-scale R² is an outlier-driven metric.
Capping predictions at the 99.5th/99.9th training-price percentile was tried and rejected (it hurt,
because real 2024 prices go far above those levels).

## Change 1: minimum rows per leaf (XGBoost only, validation experiment)
| min_child_weight | trees_kept | val_mae_sar | val_rmse_sar | val_r2_sar | val_r2_ln_price | train_seconds |
|---|---|---|---|---|---|---|
| 50.0 | 1216.0 | 531,509 | 7,401,742 | 0.3129 | 0.7118 | 43 |
| 200.0 | 1406.0 | 523,325 | 7,198,783 | 0.3501 | 0.7090 | 65 |
| 1000.0 | 2684.0 | 541,422 | 7,646,554 | 0.2667 | 0.6989 | 114 |

`min_child_weight = 200` (v1: 5) gave the best validation MAE and R²; 50 and 1000 were both worse.

## Change 2: equal-weight ensemble
Ensemble v2 = geometric mean (average in ln space) of XGBoost v2 and the frozen v1 CatBoost, weights 0.5/0.5, not tuned.

## Validation 2024
| Model | Validation MAE (SAR) | Validation RMSE (SAR) | Validation R2 (SAR) | Validation R2 (ln price) | Median abs error (SAR) | MAE improvement vs baseline (%) |
|---|---|---|---|---|---|---|
| XGBoost v1 | 539,630 | 8,261,632 | 0.1440 | 0.7132 | 117,990 | 18.73 |
| CatBoost v1 | 554,411 | 7,638,329 | 0.2683 | 0.7019 | 130,865 | 16.50 |
| XGBoost v2 (min_child_weight=200) | 523,325 | 7,198,783 | 0.3501 | 0.7090 | 118,440 | 21.18 |
| Ensemble v2 (XGBoost v2 + CatBoost, 50/50) | 523,066 | 6,801,708 | 0.4198 | 0.7190 | 119,563 | 21.22 |

## Test 2025 (second look; v1 test results had already been seen)
| Model | Test MAE (SAR) | Test RMSE (SAR) | Test R2 (SAR) | Test R2 (ln price) | Median abs error (SAR) | MAE improvement vs test baseline (%) | Met 15% target |
|---|---|---|---|---|---|---|---|
| XGBoost v1 | 596,077 | 12,670,756 | -0.7500 | 0.6497 | 130,469 | 12.28 | No |
| CatBoost v1 | 554,819 | 7,609,560 | 0.3688 | 0.6445 | 132,480 | 18.35 | Yes |
| XGBoost v2 (min_child_weight=200) | 561,276 | 8,655,182 | 0.1835 | 0.6488 | 130,179 | 17.40 | Yes |
| Ensemble v2 (XGBoost v2 + CatBoost, 50/50) | 545,026 | 7,647,744 | 0.3625 | 0.6604 | 128,683 | 19.79 | Yes |

### By property type (test)
| Model | Property type | Test rows | MAE (SAR) | RMSE (SAR) | R2 |
|---|---|---|---|---|---|
| XGBoost v1 | Residential | 181,503 | 381,752 | 4,612,771 | 0.2346 |
| XGBoost v1 | Commercial | 12,180 | 3,363,116 | 47,703,108 | -1.2005 |
| XGBoost v1 | Agricultural | 6,255 | 1,427,106 | 9,126,991 | 0.0960 |
| CatBoost v1 | Residential | 181,503 | 388,364 | 4,602,817 | 0.2379 |
| CatBoost v1 | Commercial | 12,180 | 2,702,117 | 24,344,781 | 0.4269 |
| CatBoost v1 | Agricultural | 6,255 | 1,203,601 | 9,060,417 | 0.1091 |
| XGBoost v2 (min_child_weight=200) | Residential | 181,503 | 378,534 | 4,732,294 | 0.1945 |
| XGBoost v2 (min_child_weight=200) | Commercial | 12,180 | 2,876,027 | 29,184,539 | 0.1764 |
| XGBoost v2 (min_child_weight=200) | Agricultural | 6,255 | 1,356,572 | 9,282,328 | 0.0649 |
| Ensemble v2 (XGBoost v2 + CatBoost, 50/50) | Residential | 181,503 | 376,357 | 4,634,253 | 0.2275 |
| Ensemble v2 (XGBoost v2 + CatBoost, 50/50) | Commercial | 12,180 | 2,699,049 | 24,435,199 | 0.4226 |
| Ensemble v2 (XGBoost v2 + CatBoost, 50/50) | Agricultural | 6,255 | 1,244,943 | 9,148,730 | 0.0917 |

## Caveats
- All v2 decisions used validation only, but Test 2025 was already inspected for v1, so it is no longer a
  perfectly untouched set. Treat these test numbers as a second look, and prefer a fresh chronological
  split (or 2026 data) for any further decisions.
- Only min_child_weight (3 values) and one fixed 50/50 blend were tried. This is not a full tuning study.
- SAR-scale R² stays low and unstable across years because it depends on a few unpredictable giant parcels.
  Judge models on MAE, median error and ln-scale R² as well.
- The MLP was not revisited.
