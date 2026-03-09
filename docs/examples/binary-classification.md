# Example: Binary Classification

A walkthrough of a basic binary classification project — from initialization through backtest results.

---

## 1. Initialize the project

```
configure(action="init",
  project_name="churn_prediction",
  task="binary",
  target_column="churned"
)
```

## 2. Add a data source

```
data(action="add", data_path="data/customers.csv")
data(action="inspect")
```

Review the inspect output to confirm column names, types, and null counts. Verify the target column exists and contains binary values (0/1 or True/False).

## 3. Discover features

```
features(action="discover", top_n=15, method="xgboost")
```

This ranks columns by predictive importance. Add the ones that make domain sense:

```
features(action="add_batch", features=[
  {"name": "tenure_months", "type": "instance"},
  {"name": "monthly_charges", "type": "instance"},
  {"name": "total_charges", "type": "instance"},
  {"name": "contract_type", "type": "instance"},
  {"name": "num_support_tickets", "type": "instance"}
])
```

## 4. Add models

```
models(action="add", name="xgb_main", model_type="xgboost",
  params={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6})

models(action="add", name="lgb_main", model_type="lightgbm",
  params={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31})

models(action="add", name="cat_main", model_type="catboost",
  params={"iterations": 300, "learning_rate": 0.05, "depth": 6})
```

## 5. Configure backtest

```
configure(action="backtest",
  cv_strategy="kfold",
  fold_values=[1, 2, 3, 4, 5],
  metrics=["brier", "accuracy", "log_loss", "ece", "auc"]
)
```

## 6. Configure ensemble

```
configure(action="ensemble",
  method="stacked",
  calibration="spline"
)
```

## 7. Run the backtest

```
pipeline(action="run_backtest")
```

## 8. Interpret results

```
pipeline(action="diagnostics")
```

This returns:
- Per-model and ensemble metrics across all folds
- Calibration quality (ECE, reliability curve data)
- Feature importance rankings
- Model correlation matrix (to assess diversity)

Key things to look for:
- **Brier score**: lower is better. This is your primary probability metric for binary tasks.
- **ECE (Expected Calibration Error)**: below 0.05 is good calibration.
- **Model correlations**: if all models are >0.95 correlated, you need more model diversity.
- **Per-fold variance**: large swings across folds suggest the model is not robust.

## 9. Next steps

- Add engineered features: `features(action="auto_search", search_types=["interactions", "rolling"])`
- Run experiments to tune hyperparameters (see [feature engineering example](feature-engineering.md))
- Check workflow progress: `pipeline(action="progress")`
