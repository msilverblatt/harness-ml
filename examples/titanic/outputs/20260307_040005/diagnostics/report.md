# Backtest Report

## Top-Line Metrics

| Metric | Value |
|--------|-------|
| Brier | 0.2341 |
| Accuracy | 0.7733 |
| Ece | 0.2379 |
| Log Loss | 0.7871 |

## Per-Fold Breakdown

| Fold | Accuracy | Brier Score | Ece | Log Loss | Games |
|------|------|------|------|------|------|
| 1 | 0.7963 | 0.2493 | 0.3101 | 0.7280 | 216 |
| 2 | 0.8424 | 0.1806 | 0.1643 | 0.5451 | 184 |
| 3 | 0.7373 | 0.2474 | 0.2579 | 0.9038 | 491 |

## Meta-Learner Coefficients

| Model | Coefficient |
|-------|-------------|
| xgb_main | +2.2897 |
| lgb_main | +1.8443 |
| lr_baseline | +1.8367 |
| prior_diff | +0.0000 |

## Pick Analysis

- **Total Picks**: 891
- **Correct**: 689 (77.3%)
- **Avg Confidence**: 0.2687
- **Avg Model Agreement**: 95.2%
- **Avg Confidence (correct)**: 0.2439
- **Avg Confidence (incorrect)**: 0.3532
