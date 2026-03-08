# Experiment Log

| ID | Hypothesis | Changes | Verdict | Notes |
|----|-----------|---------|---------|-------|
| exp-001 | 2026-03-07 | Establish baseline RMSE with a single XGBoost model using all 11 raw physicochemical features. Published benchmarks for this dataset range from 0.55-0.65 RMSE with tree-based methods. This gives us a reference point for all future experiments. |  | - | - | - | - | neutral |  |  |
| exp-002 | 2026-03-07 | Adding alcohol-volatile_acidity (corr=0.52, strongest single interaction) should improve RMSE. This feature captures the balance between the top positive predictor (alcohol) and top negative predictor (volatile acidity). However, since XGBoost can learn interactions natively, the improvement may be marginal. | {models: {xgb_base: {features: [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
        alcohol, alcohol_minus_va]}}}
 | - | - | - | - | improved |  |  |
| exp-003 | 2026-03-07 | Adding a second gradient boosting implementation (LightGBM) with the same features should provide modest ensemble improvement through algorithmic diversity. LightGBM uses histogram-based splits and leaf-wise growth vs XGBoost's level-wise, so predictions should differ enough to benefit averaging. | {models: {lgbm_base: {active: true, features: [fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol], include_in_ensemble: true, mode: regressor, params: {
        learning_rate: 0.05, max_depth: 6, n_estimators: 300, num_leaves: 31, subsample: 0.8},
      type: lightgbm}}}
 | - | - | - | - | improved |  |  |
| exp-004 | 2026-03-07 | Switching from average to stacked ensemble should recover the regression from exp-003. The meta-learner can learn to weight XGBoost more heavily if it's the stronger model, rather than giving equal weight to both. Expect RMSE to match or beat the single-model baseline of 0.5739. | {ensemble: {method: stacked}, models: {lgbm_base: {active: true, features: [fixed_acidity,
        volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol], include_in_ensemble: true,
      mode: regressor, params: {learning_rate: 0.05, max_depth: 6, n_estimators: 300,
        num_leaves: 31, subsample: 0.8}, type: lightgbm}}}
 | - | - | - | - | improved |  |  |
| exp-005 | 2026-03-07 | RF uses bagging + random feature subsets, a fundamentally different variance reduction strategy than XGBoost's boosting. This algorithmic diversity should produce less correlated predictions, making the stacked ensemble more effective than two boosting algorithms. Expect RMSE to beat the 0.5739 baseline. | {ensemble: {method: stacked}, models: {rf_base: {active: true, features: [fixed_acidity,
        volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol], include_in_ensemble: true,
      mode: regressor, params: {max_depth: 10, min_samples_leaf: 5, n_estimators: 300},
      type: random_forest}}}
 | - | - | - | - | regressed |  |  |
| exp-006 | 2026-03-07 | CatBoost uses ordered boosting (prevents target leakage in gradient estimation), a third distinct algorithm. With XGBoost, RF, and CatBoost, we have three different tree-building strategies. The stacked meta-learner should benefit from this tripartite diversity. Expect RMSE below current best of 0.5732. | {models: {catboost_reg: {active: true, features: [fixed_acidity, volatile_acidity,
        citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol], include_in_ensemble: true, mode: regressor,
      params: {depth: 6, iterations: 500, learning_rate: 0.05}, type: catboost}}}
 | - | - | - | - | improved |  |  |
| exp-007 | 2026-03-07 | MLP is a fundamentally different function approximator from tree-based models — it learns smooth, continuous decision boundaries via gradient descent on weighted sums. This should produce prediction patterns uncorrelated with XGBoost and RF, giving the stacked meta-learner genuine diversity. Expect RMSE improvement over the 2-model baseline of 0.5732. | {models: {mlp_reg: {active: true, features: [fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol], include_in_ensemble: true, mode: regressor, params: {
        batch_size: 64, early_stopping: true, epochs: 200, hidden_layers: [64, 32],
        learning_rate: 0.001, normalize: true}, type: mlp}}}
 | - | - | - | - | improved |  |  |
| exp-008 | 2026-03-07 | va_sulphate_ratio (corr=-0.44) captures the balance between off-flavor and preservation — a domain-meaningful ratio. Unlike alcohol_minus_va (exp-002), this ratio is harder for trees to learn natively since it requires multiplicative reasoning. Should improve RMSE below 0.5732. | {models: {rf_base: {features: [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
        alcohol, va_sulphate_ratio]}, xgb_base: {features: [fixed_acidity, volatile_acidity,
        citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol, va_sulphate_ratio]}}}
 | - | - | - | - | improved |  |  |
| exp-009 | 2026-03-07 | With only 1599 rows, weaker features (residual_sugar, pH, free_sulfur_dioxide, chlorides, fixed_acidity) may add noise. Removing them should reduce overfitting and improve generalization. The top 6 features account for ~85% of mutual information importance. Expect RMSE improvement through noise reduction. | {models: {rf_base: {features: [alcohol, volatile_acidity, sulphates, density, citric_acid,
        total_sulfur_dioxide]}, xgb_base: {features: [alcohol, volatile_acidity, sulphates,
        density, citric_acid, total_sulfur_dioxide]}}}
 | - | - | - | - | improved |  |  |
| exp-010 | 2026-03-07 | If both models use the same features they produce correlated predictions. By giving RF a different view — dropping the top 3 XGBoost-dominant features (alcohol, volatile_acidity, sulphates) and keeping the rest plus engineered features — RF should learn complementary patterns. The meta-learner can combine XGBoost's strength on dominant features with RF's strength on secondary features. | {models: {rf_base: {features: [fixed_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, alcohol_minus_va,
        va_sulphate_ratio]}}}
 | - | - | - | - | regressed |  |  |
| exp-011 | 2026-03-07 | free_so2_ratio (free SO2 / total SO2) measures preservation efficiency. Corr=0.19 with quality. Adding to RF only (which benefits from ratio features per exp-010). XGBoost stays unchanged. Should give RF another useful ratio the trees can't easily learn. | {models: {rf_base: {features: [fixed_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, alcohol_minus_va,
        va_sulphate_ratio, free_so2_ratio]}}}
 | - | - | - | - | regressed |  |  |
| exp-012 | 2026-03-07 | acid_balance (fixed_acidity - volatile_acidity, corr=0.16) captures wine acid health. Lower correlation than previous features but it's a subtraction RF struggles to learn. Following the pattern of feeding engineered features to RF only. Incremental RMSE improvement expected. | {models: {rf_base: {features: [fixed_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, alcohol_minus_va,
        va_sulphate_ratio, free_so2_ratio, acid_balance]}}}
 | - | - | - | - | regressed |  |  |
| exp-013 | 2026-03-07 | With only 1599 rows and 11 features, depth=5 may overfit. Reducing to depth=4 limits tree complexity and could improve out-of-fold generalization. The trade-off is potentially underfitting — but the dataset is small enough that regularization often wins. | {models: {xgb_base: {params: {colsample_bytree: 0.8, learning_rate: 0.05, max_depth: 4,
        n_estimators: 300, subsample: 0.8}}}}
 | - | - | - | - | improved |  |  |
| exp-014 | 2026-03-07 | Depth 4 underfit (exp-013). Testing depth 6 to see if more complexity captures additional non-linear patterns. Risk is overfitting on 1599 rows, but subsample=0.8 and colsample_bytree=0.8 provide regularization. | {models: {xgb_base: {params: {colsample_bytree: 0.8, learning_rate: 0.05, max_depth: 6,
        n_estimators: 300, subsample: 0.8}}}}
 | - | - | - | - | regressed |  |  |
| exp-015 | 2026-03-07 | More trees with a lower learning rate should produce a smoother fit with less variance. The current 300 trees at lr=0.05 may not fully converge. Doubling the budget while halving the step size is a standard technique for squeezing out residual gains. | {models: {xgb_base: {params: {colsample_bytree: 0.8, learning_rate: 0.03, max_depth: 6,
        n_estimators: 500, subsample: 0.8}}}}
 | - | - | - | - | regressed |  |  |
