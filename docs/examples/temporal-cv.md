# Example: Temporal Cross-Validation

A walkthrough of time-series prediction using temporal CV strategies — expanding window and sliding window.

---

## When to use temporal CV

Standard k-fold CV shuffles data randomly, which leaks future information into the training set for time-dependent data. Use temporal CV when:

- Your data has a natural time ordering (dates, seasons, quarters)
- Future observations should never appear in training data
- You want to simulate real-world prediction: train on past, predict the future

---

## Setup

```
configure(action="init",
  project_name="sales_forecast",
  task="regression",
  target_column="revenue",
  time_column="quarter"
)

data(action="add", data_path="data/quarterly_sales.csv")
```

---

## Strategy 1: Expanding window

Each fold trains on all data before the test period. The training set grows with each fold.

```
configure(action="backtest",
  cv_strategy="expanding",
  fold_column="year",
  fold_values=[2019, 2020, 2021, 2022, 2023, 2024],
  min_train_folds=2,
  metrics=["rmse", "mae", "r2"]
)
```

This produces:
- Fold 2021: train on 2019-2020, test on 2021
- Fold 2022: train on 2019-2021, test on 2022
- Fold 2023: train on 2019-2022, test on 2023
- Fold 2024: train on 2019-2023, test on 2024

`min_train_folds=2` means the first two years are training-only (no test fold for 2019 or 2020).

**When to use:** When older data is still relevant and more training data generally helps. Common for stable domains where the underlying distribution changes slowly.

---

## Strategy 2: Sliding window

Each fold trains on a fixed-size window of recent data. Older data drops off as the window moves forward.

```
configure(action="backtest",
  cv_strategy="sliding",
  fold_column="year",
  fold_values=[2019, 2020, 2021, 2022, 2023, 2024],
  min_train_folds=3,
  metrics=["rmse", "mae", "r2"]
)
```

This produces:
- Fold 2022: train on 2019-2021, test on 2022
- Fold 2023: train on 2020-2022, test on 2023
- Fold 2024: train on 2021-2023, test on 2024

**When to use:** When older data becomes stale or the distribution shifts over time. Common for markets, user behavior, and domains with concept drift.

---

## Strategy 3: Leave-One-Season-Out (LOSO)

Each fold holds out one time period and trains on all others. Useful when you want to test robustness across all periods.

```
configure(action="backtest",
  cv_strategy="loso",
  fold_column="season",
  fold_values=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
  metrics=["rmse", "mae", "r2"]
)
```

Note: LOSO does not enforce temporal ordering — it trains on future data too. Use it when temporal leakage is acceptable (e.g., evaluating feature robustness across periods) but prefer expanding or sliding for production forecasting.

---

## Adding models for temporal data

Temporal tasks often benefit from models that can capture trend and seasonality:

```
models(action="add", name="lgb_temporal", model_type="lightgbm",
  params={"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31})

models(action="add", name="xgb_temporal", model_type="xgboost",
  params={"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5})
```

## Engineering temporal features

Lag and rolling features are critical for temporal tasks:

```
features(action="add_batch", features=[
  {"name": "revenue_lag1", "formula": "revenue.shift(1)", "type": "instance"},
  {"name": "revenue_lag4", "formula": "revenue.shift(4)", "type": "instance"},
  {"name": "revenue_rolling4_mean", "type": "instance"}
])
```

Or use auto-search to discover temporal features:

```
features(action="auto_search",
  features=["revenue", "units_sold", "avg_price"],
  search_types=["lags", "rolling"]
)
```

## Run and evaluate

```
pipeline(action="run_backtest")
pipeline(action="diagnostics")
```

For temporal tasks, pay special attention to:
- **Per-fold metrics**: Performance should not degrade significantly in recent folds. If it does, the model may not be adapting to distribution shifts.
- **RMSE vs MAE**: Large gap between RMSE and MAE indicates sensitivity to outliers.
- **R-squared**: Below 0 means the model is worse than predicting the mean.
