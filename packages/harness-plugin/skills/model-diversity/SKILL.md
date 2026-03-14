# HarnessML: Model Diversity

Use when evaluating your model ensemble — what's in it, what's missing, and whether the models are actually providing diverse perspectives on the problem.

---

## Why Diversity Matters

An ensemble of 6 models that all make the same predictions is effectively one model. The power of ensembling comes from combining models that make **different mistakes** — when one model is wrong, others compensate.

Diversity comes from three sources:
1. **Different model families** (tree vs linear vs neural) — different inductive biases
2. **Different feature sets** — models see different slices of the data
3. **Different hyperparameters** — models fit the data at different complexity levels

Source #1 is the strongest. Source #3 is the weakest.

---

## What Different Model Families Bring

### Linear Models (logistic_regression, elastic_net)

**Inductive bias:** Monotonic relationships, additive effects.

**Strengths:** Stable, interpretable, calibrates well, handles high-dimensional sparse data. Often the best-calibrated model in the ensemble even when not the most discriminative.

**Weaknesses:** Cannot learn interactions or non-linear relationships without explicit feature engineering.

**When they help the ensemble:** They anchor predictions in linear signal that trees might overfit. If your linear model has a high meta-learner coefficient, it means the trees are noisy on cases where the linear signal is clear.

### Gradient Boosted Trees (xgboost, lightgbm, catboost)

**Inductive bias:** Axis-aligned splits, interaction detection, sequential error correction.

**Strengths:** Handle mixed types, learn interactions natively, robust to outliers, built-in feature importance.

**Why try all three:**
- XGBoost: mature, well-understood regularization, good baseline
- LightGBM: faster, leaf-wise growth finds different splits than depth-wise
- CatBoost: native categorical handling, ordered boosting reduces overfitting

Two of the three often end up with >0.95 correlation and should be differentiated with different feature sets. But which two varies by dataset — try all three.

### Random Forest

**Inductive bias:** Bagging reduces variance through decorrelated trees.

**Strengths:** Naturally resistant to overfitting, gives calibrated probability estimates, useful as a diversity anchor against boosted models.

**When it helps the ensemble:** Its errors are decorrelated from boosted trees because it doesn't sequentially correct — it averages. This is exactly the diversity ensembles need.

### Neural Networks (mlp, tabnet)

**Inductive bias:** Learned representations, smooth decision boundaries.

**Strengths:** Can learn complex non-linear relationships, representation learning, flexible architecture.

**Weaknesses:** Need more data, harder to regularize, less interpretable, training sensitive to hyperparameters.

**When they help the ensemble:** On datasets with enough rows (10k+), neural networks learn different representations than trees. If your neural net has low correlation with tree models, it's capturing genuinely different signal.

### Other Models (svm, histgbm, gam, ngboost)

**SVM:** Different decision boundary geometry. Useful when classes are separable in high-dimensional space.

**HistGBM:** sklearn's histogram-based gradient boosting. Similar to LightGBM but integrates with sklearn pipeline. Good for when you want a second tree-based model with different implementation details.

**GAM (Generalized Additive Model):** Learns smooth non-linear relationships for each feature independently. Highly interpretable. Useful as a "step up from linear" that doesn't overfit like trees can.

**NGBoost:** Probabilistic predictions with uncertainty. If your use case values uncertainty quantification, NGBoost provides it natively.

---

## Evaluating Your Ensemble

### Check Prediction Correlations

```
features(action="diversity")
```

- **>0.95 correlation between two models:** They're making nearly identical predictions. One is redundant. Differentiate them (different features, different hyperparameters) or drop one.
- **0.7-0.9 correlation:** Healthy. Models agree on easy cases, disagree on hard cases.
- **<0.7 correlation:** Very diverse. Make sure both models are individually competent — low correlation from a bad model just adds noise.

### Check Meta-Learner Coefficients

```
pipeline(action="diagnostics")
```

- **High positive coefficient:** Model contributes strong unique signal.
- **Near-zero coefficient:** Model is redundant with others. The ensemble ignores it.
- **Negative coefficient:** Model actively hurts when combined. Its signal is anti-correlated with truth *given the other models*. Consider removing.

### Check LOMO (Leave-One-Model-Out) Impact

Remove each model from the ensemble one at a time and measure the impact. Models whose removal hurts performance are carrying their weight. Models whose removal is neutral or positive should be removed or replaced.

---

## Building Diversity

### Give Each Model Type a Fair Trial

Don't dismiss a model family after one configuration. A logistic regression with the wrong features looks terrible. A logistic regression with well-engineered features can be the best-calibrated model in the ensemble.

Minimum fair trial:
- 2-3 configurations with different feature sets or key structural parameters
- Evaluate both individual performance AND contribution to ensemble diversity
- A model with mediocre individual performance but low correlation with others may still improve the ensemble

### Differentiate with Features

The most powerful diversification lever. Give different models different feature sets:

```
models(action="update", name="lr_main", features=[...])  # linear-friendly
models(action="update", name="xgb_main", features=[...])  # all features
models(action="update", name="mlp_main", features=[...])  # normalized numerics
```

This creates genuine diversity because models literally see different data.

### When to Add vs Replace

- **Add** when prediction correlation with existing models is <0.9 and individual performance is reasonable
- **Replace** when a new model is better than an existing one with similar correlation profile
- **Remove** when meta-learner coefficient is near-zero or negative and LOMO shows no loss

---

## Common Traps

- **Model count as a goal:** 8 models with 0.98 pairwise correlation is worse than 4 diverse models. Measure diversity, not count.
- **Only trying gradient boosted trees:** XGBoost + LightGBM + CatBoost is three implementations of the same idea. You need different *families*.
- **Dismissing linear models:** "Logistic regression is too simple." It's not — it's your calibration anchor and the first model to tell you if your features carry signal.
- **Dismissing neural nets on small data:** MLP with dropout and batch norm can work on 5k rows. Try it before assuming.
- **Uniform feature sets:** Every model seeing every feature is a missed diversity opportunity.
