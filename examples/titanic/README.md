# Titanic Survival Prediction

End-to-end example using HarnessML through the MCP tools.

## Setup

This example is meant to be run through an AI agent connected to HarnessML via MCP. The agent conversation below shows the complete workflow.

## Agent Workflow

```
# 1. Initialize project
configure(action="init", project_dir="examples/titanic")

# 2. Download data
data(action="fetch_url",
     data_path="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
     name="titanic.csv",
     project_dir="examples/titanic")

# 3. Ingest and profile
data(action="add", data_path="titanic.csv", project_dir="examples/titanic")
data(action="profile", project_dir="examples/titanic")

# 4. Configure pipeline
configure(action="update",
          target_column="Survived",
          fold_column="Pclass",
          project_dir="examples/titanic")

# 5. Handle nulls
data(action="fill_nulls", column="Age", strategy="median", project_dir="examples/titanic")
data(action="fill_nulls", column="Embarked", strategy="mode", project_dir="examples/titanic")

# 6. Add features
features(action="add", name="age", formula="Age", type="instance", project_dir="examples/titanic")
features(action="add", name="fare", formula="Fare", type="instance", project_dir="examples/titanic")
features(action="add", name="sibsp", formula="SibSp", type="instance", project_dir="examples/titanic")
features(action="add", name="parch", formula="Parch", type="instance", project_dir="examples/titanic")
features(action="add", name="sex_male", formula="(Sex == 'male').astype(int)", type="instance",
         project_dir="examples/titanic")

# 7. Add models
models(action="add", name="xgb_main", preset="xgboost_classifier",
       features=["age", "fare", "sibsp", "parch", "sex_male"],
       project_dir="examples/titanic")

models(action="add", name="lgb_main", preset="lightgbm_classifier",
       features=["age", "fare", "sibsp", "parch", "sex_male"],
       project_dir="examples/titanic")

models(action="add", name="lr_baseline", preset="logistic_regression",
       features=["age", "fare", "sibsp", "parch", "sex_male"],
       project_dir="examples/titanic")

# 8. Run backtest
pipeline(action="run_backtest", project_dir="examples/titanic")

# 9. Check results
pipeline(action="diagnostics", project_dir="examples/titanic")
```

## What happens automatically

- Cross-validation across Pclass folds (leave-one-out)
- Model training with early stopping where supported
- Stacked ensemble with calibration
- Metric computation (Brier, accuracy, log loss, ECE, AUROC)
- Run fingerprinting and logging
- Guardrail checks (leakage, temporal integrity)
