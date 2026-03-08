# Experiment Log

| ID | Hypothesis | Changes | Verdict | Notes |
|----|-----------|---------|---------|-------|
| exp-001 | 2026-03-07 | Establish baseline with the 5 strongest predictors: OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF. These have correlations 0.61-0.79 with SalePrice. Simple model, simple features. Published Ames benchmarks range from RMSE 20K-35K depending on approach. |  | - | - | - | - | neutral |  |  |
| exp-002 | 2026-03-07 | Adding the next 10 strongest numeric features should substantially improve R² since we're only at 0.79 with 5 features. YearBuilt (corr 0.52), 1stFlrSF (0.61), FullBath (0.56), YearRemodAdd (0.51) all have strong individual signal. With 15 features and 1460 rows, overfitting risk is manageable at depth 3. | {models: {xgb_base: {features: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF,
        YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces, LotArea,
        MasVnrArea, BsmtFinSF1, 2ndFlrSF]}}}
 | - | - | - | - | regressed |  |  |
| exp-003 | 2026-03-07 | Total square footage (above grade + basement) is what buyers actually compare. GrLivArea and TotalBsmtSF are both in the model, but their sum gives the model a direct "total usable area" feature. At depth 3, XGBoost can't easily add two features — providing the sum frees up splits for other interactions. | {models: {xgb_base: {features: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF,
        YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces, LotArea,
        MasVnrArea, BsmtFinSF1, 2ndFlrSF, total_sf]}}}
 | - | - | - | - | regressed |  |  |
| exp-004 | 2026-03-07 | House age at time of sale is a more intuitive and normalized feature than raw YearBuilt. A house built in 1970 sold in 2006 is 36 years old — same as a house built in 1960 sold in 1996. The age feature removes the confound of when it was sold. Since YrSold and YearBuilt are both available, XGBoost could learn this, but the subtraction is hard at depth 3. | {models: {xgb_base: {features: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF,
        YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces, LotArea,
        MasVnrArea, BsmtFinSF1, 2ndFlrSF, total_sf, house_age]}}}
 | - | - | - | - | improved |  |  |
| exp-005 | 2026-03-07 | How recently a house was remodeled matters — a house remodeled last year is worth more than one remodeled 30 years ago. remodel_age directly captures this. Unlike house_age, remodel recency varies independently of when the house was built, so it should provide non-redundant signal. | {models: {xgb_base: {features: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF,
        YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces, LotArea,
        MasVnrArea, BsmtFinSF1, 2ndFlrSF, total_sf, remodel_age]}}}
 | - | - | - | - | improved |  |  |
| exp-006 | 2026-03-07 | Total bathroom count (weighting half baths at 0.5) is a standard real estate metric. It combines 4 separate bath columns into one actionable number. Only FullBath is currently in the model. This adds HalfBath, BsmtFullBath, and BsmtHalfBath signal indirectly. More bathrooms = higher value in real estate. | {models: {xgb_base: {features: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF,
        YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces, LotArea,
        MasVnrArea, BsmtFinSF1, 2ndFlrSF, total_sf, total_bath]}}}
 | - | - | - | - | regressed |  |  |
| exp-007 | 2026-03-07 | OverallQual and GrLivArea are the two strongest predictors (0.79, 0.71 correlation). Their product captures the "big and high quality" premium — a large high-quality house is worth disproportionately more than the sum of size and quality suggests. At depth 3, XGBoost uses 2 of 3 splits on these features, leaving no room for other interactions. The product feature frees up a split. | {models: {xgb_base: {features: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF,
        YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces, LotArea,
        MasVnrArea, BsmtFinSF1, 2ndFlrSF, total_sf, total_bath, qual_x_area]}}}
 | - | - | - | - | improved |  |  |
| exp-008 | 2026-03-07 | Proven pattern from wine quality and Titanic: XGBoost + RandomForest stacked ensemble provides the best algorithmic diversity. Same 17 features for both. The meta-learner can weight them optimally. Expect RMSE improvement from the ensemble averaging out model-specific errors. | {models: {rf_base: {active: true, features: [OverallQual, GrLivArea, GarageCars, GarageArea,
        TotalBsmtSF, YearBuilt, 1stFlrSF, FullBath, YearRemodAdd, TotRmsAbvGrd, Fireplaces,
        LotArea, MasVnrArea, BsmtFinSF1, 2ndFlrSF, total_sf, total_bath], include_in_ensemble: true,
      mode: regressor, params: {max_depth: 10, min_samples_leaf: 5, n_estimators: 300},
      type: random_forest}}}
 | - | - | - | - | regressed |  |  |
| exp-009 | 2026-03-07 | Proven cross-project pattern: give RF features XGBoost doesn't use to increase prediction diversity. RF gets house_age, remodel_age, qual_x_area (all failed for XGBoost) plus additional raw features (OverallCond, LotFrontage, WoodDeckSF, OpenPorchSF, BsmtUnfSF, GarageYrBlt, MSSubClass). XGBoost keeps the clean 17-feature set. | {models: {rf_base: {features: [OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt,
        1stFlrSF, FullBath, TotRmsAbvGrd, Fireplaces, LotArea, total_sf, total_bath,
        house_age, remodel_age, qual_x_area, OverallCond, LotFrontage, WoodDeckSF,
        OpenPorchSF, BsmtUnfSF, GarageYrBlt, MSSubClass]}}}
 | - | - | - | - | regressed |  |  |
| exp-010 | 2026-03-07 | With 17 features and 1460 rows, depth 3 only allows 3 interaction levels per tree. Housing price depends on complex feature interactions (quality × age × size × location proxy). Depth 5 allows richer interactions. Wine quality saw the biggest gain from depth increase. Ames has more features and similar row count, so more depth should help. | {models: {xgb_base: {params: {learning_rate: 0.1, max_depth: 5, n_estimators: 100}}}}
 | - | - | - | - | improved |  |  |
| exp-011 | 2026-03-07 | Depth 5 overfit, depth 3 may underfit. Depth 4 with subsampling regularization (0.8 row/column) should find the sweet spot — more interaction capacity than depth 3, but the stochastic regularization prevents the overfitting seen at depth 5. This combo worked well in wine quality. | {models: {xgb_base: {params: {colsample_bytree: 0.8, learning_rate: 0.05, max_depth: 4,
        n_estimators: 200, subsample: 0.8}}}}
 | - | - | - | - | improved |  |  |
| exp-012 | 2026-03-07 | XGBoost currently lacks OverallCond and GarageYrBlt which RF uses. Adding these raw features should improve XGBoost predictions on houses where condition and garage age matter, reducing ensemble RMSE below 31,382. | {models: {xgb_base: {append_features: [OverallCond, GarageYrBlt]}}}
 | - | - | - | - | regressed |  |  |
| exp-013 | 2026-03-07 | LotFrontage and BsmtUnfSF are currently only in RF. Adding them to XGBoost gives it more signal about lot size and basement finish level, potentially reducing RMSE below 30,900. | {models: {xgb_base: {append_features: [LotFrontage, BsmtUnfSF]}}}
 | - | - | - | - | improved |  |  |
| exp-015 | 2026-03-07 | LightGBM uses histogram-based splitting and leaf-wise growth, which may capture different patterns than XGBoost's level-wise approach. Adding it as a third ensemble member should improve diversity and reduce RMSE below 30,900. | {models: {lgbm_base: {active: true, features: [OverallQual, GrLivArea, GarageCars,
        GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd,
        MasVnrArea, Fireplaces, BsmtFinSF1, LotArea, 2ndFlrSF, total_sf, total_bath,
        OverallCond, GarageYrBlt], include_in_ensemble: true, mode: regressor, params: {
        learning_rate: 0.1, max_depth: 3, n_estimators: 100, num_leaves: 15}, type: lightgbm}}}
 | - | - | - | - | regressed |  |  |
| exp-016 | 2026-03-07 | Currently LightGBM and XGBoost share the same 19 features. Giving LightGBM differentiated features (adding house_age, remodel_age, qual_x_area like RF) should increase ensemble diversity and reduce RMSE below 30,690. | {models: {lgbm_base: {append_features: [house_age, remodel_age, qual_x_area]}}}
 | - | - | - | - | improved |  |  |
| exp-017 | 2026-03-07 | LightGBM with 100 trees may be underfitting. Doubling to 200 should allow it to capture more complex patterns without overfitting at depth 3, reducing ensemble RMSE below 30,690. | {models: {lgbm_base: {params: {n_estimators: 200}}}}
 | - | - | - | - | improved |  |  |
| exp-018 | 2026-03-07 | XGBoost at depth 3 with 100 trees may be slightly underfitting. Increasing to 200 trees gives more iterations for the gradient boosting to converge, potentially reducing ensemble RMSE below 30,690. | {models: {xgb_base: {params: {n_estimators: 200}}}}
 | - | - | - | - | regressed |  |  |
| exp-019 | 2026-03-07 | Halving the learning rate to 0.05 and doubling trees to 400 should allow XGBoost to converge more smoothly with less overfitting, reducing ensemble RMSE below 30,573. | {models: {xgb_base: {params: {learning_rate: 0.05, n_estimators: 400}}}}
 | - | - | - | - | improved |  |  |
| exp-020 | 2026-03-07 | RF benefits from engineered features. garage_age captures garage condition/age signal, total_porch captures total outdoor living space. Both should add predictive value to RF and reduce ensemble RMSE below 30,573. | {models: {rf_base: {append_features: [garage_age, total_porch]}}}
 | - | - | - | - | regressed |  |  |
| exp-021 | 2026-03-07 | CatBoost uses ordered boosting and symmetric trees which fundamentally differ from XGBoost/LightGBM approaches. Adding a fourth diverse model should improve the stacked ensemble and reduce RMSE below 30,573. | {models: {cat_base: {active: true, features: [OverallQual, GrLivArea, GarageCars,
        GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd,
        MasVnrArea, Fireplaces, BsmtFinSF1, LotArea, 2ndFlrSF, total_sf, total_bath,
        OverallCond, GarageYrBlt], include_in_ensemble: true, mode: regressor, params: {
        depth: 4, iterations: 200, learning_rate: 0.1}, type: catboost}}}
 | - | - | - | - | improved |  |  |
| exp-022 | 2026-03-07 | CatBoost at depth 4 slightly overfit. Depth 3 should match XGBoost's optimal depth and reduce variance, improving ensemble RMSE below 30,524. | {models: {cat_base: {active: true, features: [OverallQual, GrLivArea, GarageCars,
        GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd,
        MasVnrArea, Fireplaces, BsmtFinSF1, LotArea, 2ndFlrSF, total_sf, total_bath,
        OverallCond, GarageYrBlt], include_in_ensemble: true, mode: regressor, params: {
        depth: 3, iterations: 200, learning_rate: 0.1}, type: catboost}}}
 | - | - | - | - | improved |  |  |
| exp-023 | 2026-03-07 | A linear model (ElasticNet) captures fundamentally different patterns than tree models. Linear trends that XGBoost/LightGBM/RF approximate with step functions will be learned directly. This model diversity should reduce ensemble RMSE below 30,520. | {models: {enet_base: {active: true, features: [OverallQual, GrLivArea, GarageCars,
        GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd,
        MasVnrArea, Fireplaces, BsmtFinSF1, LotArea, 2ndFlrSF, total_sf, total_bath,
        OverallCond, GarageYrBlt, house_age, remodel_age, qual_x_area, total_porch,
        garage_age], include_in_ensemble: true, mode: regressor, params: {normalize: true},
      type: elastic_net}}}
 | - | - | - | - | improved |  |  |
| exp-024 | 2026-03-07 | RF with 300 trees may not have enough estimators for stable averaging across 24 features. Increasing to 500 should reduce variance in RF predictions, improving ensemble RMSE below 30,520. | {models: {rf_base: {params: {n_estimators: 500}}}}
 | - | - | - | - | regressed |  |  |
| exp-025 | 2026-03-07 | qual_squared captures the nonlinear quality premium (quality 10 vs 5 is more than 2x price). log_area compresses large living areas to reduce RF's sensitivity to outlier sizes. Both should help RF handle fold 4's extreme-value houses, reducing ensemble RMSE below 30,520. | {models: {rf_base: {append_features: [qual_squared, log_area]}}}
 | - | - | - | - | improved |  |  |
| exp-026 | 2026-03-07 | LightGBM with num_leaves=15 and depth=3 is heavily constrained. Increasing num_leaves to 31 (default) allows more complex leaf-wise splits while depth limit prevents overfitting. Should reduce ensemble RMSE below 30,520. | {models: {lgbm_base: {params: {num_leaves: 31}}}}
 | - | - | - | - | regressed |  |  |
| exp-027 | 2026-03-07 | All three boosted/tree models need different feature perspectives for maximum diversity. Adding MSSubClass, WoodDeckSF, and qual_squared to LightGBM (features not in XGBoost) gives it a unique view while the boosting algorithm differs from RF. Should reduce RMSE below 30,542. | {models: {lgbm_base: {append_features: [MSSubClass, WoodDeckSF, qual_squared]}}}
 | - | - | - | - | improved |  |  |
| exp-028 | 2026-03-07 | RF at max_depth=10 may be too constrained to split on important interactions in deep trees. Removing the depth limit while increasing min_samples_leaf to 10 allows deeper trees with statistical significance at leaves. Should improve RF's predictions on complex houses and reduce RMSE below 30,542. | {models: {rf_base: {params: {max_depth: null, min_samples_leaf: 10}}}}
 | - | - | - | - | improved |  |  |
| exp-029 | 2026-03-07 | XGBoost benefited from going 100→200 trees. Going to 300 may provide additional marginal gain as the boosting converges further at depth 3 with learning_rate 0.1. | {models: {xgb_base: {params: {n_estimators: 300}}}}
 | - | - | - | - | regressed |  |  |
| exp-030 | 2026-03-07 | XGBoost continues to benefit from more trees at depth 3 with LR 0.1. Going from 300 to 400 trees should provide further convergence benefit, reducing RMSE below 30,468. | {models: {xgb_base: {params: {n_estimators: 400}}}}
 | - | - | - | - | regressed |  |  |
| exp-031 | 2026-03-07 | XGBoost has shown consistent improvement from 100→200→300→400 trees. 500 trees should continue the diminishing returns curve and reduce RMSE below 30,381. | {models: {xgb_base: {params: {n_estimators: 500}}}}
 | - | - | - | - | regressed |  |  |
| exp-032 | 2026-03-07 | Earlier LightGBM 200 trees failed with num_leaves=15. Now with num_leaves=31, more leaves per tree means 200 trees won't overfit as easily. Should reduce RMSE below 30,381. | {models: {lgbm_base: {params: {n_estimators: 200}}}}
 | - | - | - | - | improved |  |  |
| exp-033 | 2026-03-07 | Adding colsample_bytree=0.8 makes XGBoost use random 80% of features per tree, reducing overfitting and increasing diversity within the model. This regularization should improve generalization and reduce RMSE below 30,381. | {models: {xgb_base: {params: {colsample_bytree: 0.8}}}}
 | - | - | - | - | improved |  |  |
| exp-034 | 2026-03-07 | Row subsampling (subsample=0.8) trains each tree on a random 80% of rows, introducing stochasticity that reduces overfitting. This is one of the most effective XGBoost regularization techniques and should reduce RMSE below 30,381. | {models: {xgb_base: {params: {subsample: 0.8}}}}
 | - | - | - | - | improved |  |  |
| exp-035 | 2026-03-07 | Basement finish ratio (BsmtFinSF1/TotalBsmtSF) captures value-add of finished basement space. This ratio isn't learnable from individual features alone. Adding to both boosted models should help price high-value finished basements and reduce RMSE below 30,381. | {models: {lgbm_base: {append_features: [bsmt_finish_ratio]}, xgb_base: {append_features: [
        bsmt_finish_ratio]}}}
 | - | - | - | - | improved |  |  |
| exp-036 | 2026-03-07 | min_child_weight=3 prevents XGBoost from creating leaves with fewer than 3 samples, reducing overfitting to rare cases. This should help with fold 4's extreme-value houses and reduce RMSE below 30,381. | {models: {xgb_base: {params: {min_child_weight: 3}}}}
 | - | - | - | - | improved |  |  |
| exp-037 | 2026-03-07 | BsmtFullBath and HalfBath capture additional bathroom value not in FullBath alone. Houses with basement bathrooms and half-baths command premiums. Adding to XGBoost should improve pricing accuracy and reduce RMSE below 30,381. | {models: {xgb_base: {append_features: [BsmtFullBath, HalfBath]}}}
 | - | - | - | - | regressed |  |  |
| exp-038 | 2026-03-07 | LightGBM at depth 3 is very similar to XGBoost depth 3. Giving LightGBM depth 4 creates more structural diversity between the two boosted models. LightGBM's leaf-wise growth handles depth 4 differently than XGBoost, reducing correlation and improving ensemble RMSE below 30,381. | {models: {lgbm_base: {params: {max_depth: 4}}}}
 | - | - | - | - | regressed |  |  |
| exp-039 | 2026-03-07 | LightGBM depth 4 improved RMSE by 90. Depth 5 may provide additional diversity benefit. LightGBM's leaf-wise growth with num_leaves=31 cap should prevent overfitting even at depth 5. | {models: {lgbm_base: {params: {max_depth: 5}}}}
 | - | - | - | - | regressed |  |  |
| exp-040 | 2026-03-07 | LightGBM improved at depth 4 and 5. Depth 6 with num_leaves=31 still has the leaf cap preventing runaway complexity. Should continue the depth benefit and reduce RMSE below 30,249. | {models: {lgbm_base: {params: {max_depth: 6}}}}
 | - | - | - | - | regressed |  |  |
| exp-041 | 2026-03-07 | At depth 5, num_leaves=31 may be the binding constraint on tree complexity (2^5=32, so 31 leaves nearly saturates depth 5). Increasing to 50 doesn't change depth 5 trees much. But if some trees benefit from asymmetric growth, it could help. Should reduce RMSE below 30,249. | {models: {lgbm_base: {params: {num_leaves: 50}}}}
 | - | - | - | - | regressed |  |  |
| exp-042 | 2026-03-07 | With depth 5 and num_leaves 50, each LightGBM tree has more capacity. 150 trees (vs 100) gives more iterations to converge with this richer tree structure. Should reduce RMSE below 30,259. | {models: {lgbm_base: {params: {n_estimators: 150}}}}
 | - | - | - | - | improved |  |  |
| exp-043 | 2026-03-07 | total_qual combines quality and condition into a single measure. Houses with high quality AND good condition command premiums that RF can capture with a single split on this combined metric. Should reduce ensemble RMSE below 30,259. | {models: {rf_base: {append_features: [total_qual]}}}
 | - | - | - | - | regressed |  |  |
| exp-044 | 2026-03-07 | LightGBM at depth 5 with 100 trees might benefit from slower learning. Halving the LR to 0.05 while keeping 100 trees forces smoother convergence that could improve generalization and reduce RMSE below 30,285. | {models: {lgbm_base: {params: {learning_rate: 0.05}}}}
 | - | - | - | - | improved |  |  |
| exp-045 | 2026-03-07 | RF at max_depth=10 may be slightly underfitting with 25 features. Increasing to 12 allows deeper interaction learning. With min_samples_leaf=5 as guard, should not overfit. May reduce RMSE below 30,285. | {models: {rf_base: {params: {max_depth: 12}}}}
 | - | - | - | - | improved |  |  |
| exp-046 | 2026-03-07 | Bedroom and kitchen counts directly impact home value. BedroomAbvGr helps RF distinguish 2-bed from 4-bed homes. KitchenAbvGr captures multi-kitchen homes (potential multi-unit). Should reduce RMSE below 30,285. | {models: {rf_base: {append_features: [BedroomAbvGr, KitchenAbvGr]}}}
 | - | - | - | - | regressed |  |  |
| exp-047 | 2026-03-07 | EnclosedPorch and ScreenPorch capture additional outdoor living value. In Ames, enclosed and screened porches add usable square footage that impacts price differently from open porches. Should reduce RMSE below 30,222. | {models: {rf_base: {append_features: [EnclosedPorch, ScreenPorch]}}}
 | - | - | - | - | regressed |  |  |
| exp-048 | 2026-03-07 | gamma=0.1 requires a minimum loss reduction of 0.1 for each split, pruning marginal splits that add variance without meaningful gain. Should help XGBoost generalize better and reduce RMSE below 30,267. | {models: {xgb_base: {params: {gamma: 0.1}}}}
 | - | - | - | - | regressed |  |  |
| exp-049 | 2026-03-07 | L1 regularization (reg_alpha=0.1) encourages sparsity in LightGBM leaf values, reducing overfitting. Similar to how gamma helped XGBoost, this should improve LightGBM's generalization and reduce ensemble RMSE below 30,232. | {models: {lgbm_base: {params: {reg_alpha: 0.1}}}}
 | - | - | - | - | regressed |  |  |
