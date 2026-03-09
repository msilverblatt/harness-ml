# Experiment Log

| ID | Hypothesis | Changes | Verdict | Notes |
|----|-----------|---------|---------|-------|
| exp-001 | 2026-03-08 | Simple logistic regression on the strongest differential features establishes the performance floor. Differential features (0.10-0.15 correlation) should outperform raw features (~0.07-0.11) for linear models. |  | 0.5706 (+0.0000) | - | - | 0.6798 (+0.0000) | neutral |  |  |
| exp-002 | 2026-03-08 | XGBoost can learn feature interactions internally from raw home/away features across both windows. Should beat logistic baseline since it can capture nonlinear relationships between team stats. | {models: {xgb_raw: {active: true, features: [home_win_pct, away_win_pct, home_run_diff_per_game,
        away_run_diff_per_game, home_team_era, away_team_era, home_starter_era, away_starter_era,
        home_bullpen_era, away_bullpen_era, home_team_k_rate, away_team_k_rate, home_team_whiff_rate,
        away_team_whiff_rate, home_team_barrel_rate_allowed, away_team_barrel_rate_allowed,
        home_team_bb_rate, away_team_bb_rate, home_14d_team_era, away_14d_team_era,
        home_14d_win_pct, away_14d_win_pct, home_14d_run_diff_per_game, away_14d_run_diff_per_game,
        home_14d_starter_era, away_14d_starter_era, games_in_window], include_in_ensemble: true,
      params: {colsample_bytree: 0.7, learning_rate: 0.05, max_depth: 4, n_estimators: 200,
        subsample: 0.8}, type: xgboost}}}
 | 0.5744 (+0.0038) | - | - | 0.6803 (+0.0005) | regressed |  |  |
| exp-003 | 2026-03-08 | LightGBM with leaf-wise growth on differential features should find different splits than XGBoost. Differential features reduce dimensionality and encode the matchup signal directly, which may suit LGBM's histogram-based splits better. | {models: {lgbm_diff: {active: true, features: [era_diff_30d, era_diff_14d, starter_era_diff,
        starter_era_diff_14d, win_pct_diff_30d, win_pct_diff_14d, run_diff_diff_30d,
        run_diff_diff_14d, k_rate_diff_30d, k_rate_diff_14d, bullpen_era_diff_30d,
        bullpen_era_diff_14d, whiff_rate_diff_30d, whiff_rate_diff_14d, games_in_window],
      include_in_ensemble: true, params: {learning_rate: 0.05, max_depth: 4, n_estimators: 200,
        num_leaves: 15, subsample: 0.8}, type: lightgbm}}}
 | 0.5680 (-0.0026) | - | - | 0.6800 (+0.0002) | regressed |  |  |
| exp-004 | 2026-03-08 | CatBoost's ordered boosting and symmetric trees handle overfitting differently than XGB/LGBM. A mixed feature set (raw + key differentials) gives it both the raw signal and pre-computed interactions, which should outperform either alone. | {models: {catboost_mix: {active: true, features: [home_win_pct, away_win_pct, home_run_diff_per_game,
        away_run_diff_per_game, home_team_era, away_team_era, home_starter_era, away_starter_era,
        home_bullpen_era, away_bullpen_era, home_14d_team_era, away_14d_team_era,
        home_14d_run_diff_per_game, away_14d_run_diff_per_game, era_diff_30d, era_diff_14d,
        win_pct_diff_30d, run_diff_diff_30d, starter_era_diff, games_in_window], include_in_ensemble: true,
      params: {depth: 4, iterations: 250, l2_leaf_reg: 3, learning_rate: 0.05}, type: catboost}}}
 | 0.5701 (-0.0005) | - | - | 0.6797 (-0.0001) | improved |  |  |
| exp-005 | 2026-03-08 | RF's bagging approach provides fundamentally different variance reduction than boosting. Using diff features keeps it focused on the matchup signal. Should add diversity since bagging decorrelates from boosted models. | {models: {rf_diff: {active: true, features: [era_diff_30d, era_diff_14d, starter_era_diff,
        starter_era_diff_14d, win_pct_diff_30d, win_pct_diff_14d, run_diff_diff_30d,
        run_diff_diff_14d, k_rate_diff_30d, k_rate_diff_14d, bullpen_era_diff_30d,
        bullpen_era_diff_14d, games_in_window], include_in_ensemble: true, params: {
        max_depth: 6, min_samples_leaf: 10, n_estimators: 300}, type: random_forest}}}
 | 0.5665 (-0.0036) | - | - | 0.6800 (+0.0003) | regressed |  |  |
| exp-007 | 2026-03-08 | MLP with 30d diffs only avoids the 14d NaN problem. The 30d features have ~5% null which tree models handle but MLP cannot. Restricting to 30d diffs should let MLP train successfully and add neural network diversity. | {models: {mlp_30d: {active: true, features: [era_diff_30d, starter_era_diff, win_pct_diff_30d,
        run_diff_diff_30d, k_rate_diff_30d, bullpen_era_diff_30d, whiff_rate_diff_30d,
        barrel_rate_diff_30d, games_in_window], include_in_ensemble: true, params: {
        batch_norm: true, early_stopping: true, hidden_layer_sizes: [64, 32], learning_rate: 0.001,
        max_iter: 200, normalize: true, weight_decay: 0.0005}, type: mlp, zero_fill_features: [
        era_diff_30d, starter_era_diff, win_pct_diff_30d, run_diff_diff_30d, k_rate_diff_30d,
        bullpen_era_diff_30d, whiff_rate_diff_30d, barrel_rate_diff_30d]}}}
 | 0.5637 (-0.0065) | - | - | 0.6795 (-0.0002) | improved |  |  |
| exp-008 | 2026-03-08 | Adding 14d differential features to logistic regression gives it multi-window signal. The 14d features capture recent form while 30d captures longer trends. With regularization (C=0.5) it should select the most informative features and improve over the 7-feature baseline. | {models: {lr_dual_window: {active: true, features: [era_diff_30d, era_diff_14d, starter_era_diff,
        starter_era_diff_14d, win_pct_diff_30d, win_pct_diff_14d, run_diff_diff_30d,
        run_diff_diff_14d, k_rate_diff_30d, k_rate_diff_14d, bullpen_era_diff_30d,
        bullpen_era_diff_14d, whiff_rate_diff_30d, whiff_rate_diff_14d, barrel_rate_diff_30d,
        barrel_rate_diff_14d], include_in_ensemble: true, params: {C: 0.5, max_iter: 1000},
      type: logistic_regression, zero_fill_features: [era_diff_14d, starter_era_diff_14d,
        win_pct_diff_14d, run_diff_diff_14d, k_rate_diff_14d, bullpen_era_diff_14d,
        whiff_rate_diff_14d, barrel_rate_diff_14d]}}}
 | 0.5668 (+0.0031) | - | - | 0.6796 (+0.0001) | regressed |  |  |
| exp-009 | 2026-03-08 | Previous XGBoost attempt used raw features only and regressed. A shallower tree (depth=3) with mixed raw + diff features and higher regularization should reduce overfitting. The mix of raw and diff gives it both perspectives. | {models: {xgb_mix: {active: true, features: [home_win_pct, away_win_pct, home_team_era,
        away_team_era, home_starter_era, away_starter_era, era_diff_30d, era_diff_14d,
        starter_era_diff, win_pct_diff_30d, run_diff_diff_30d, k_rate_diff_30d, bullpen_era_diff_30d,
        games_in_window], include_in_ensemble: true, params: {colsample_bytree: 0.6,
        learning_rate: 0.03, max_depth: 3, n_estimators: 150, reg_alpha: 0.1, reg_lambda: 1,
        subsample: 0.8}, type: xgboost}}}
 | 0.5630 (-0.0007) | - | - | 0.6792 (-0.0003) | improved |  |  |
| exp-011 | 2026-03-08 | Pythagorean expectation (RS^2/(RS^2+RA^2)) captures true team quality by filtering luck from W-L records. The differential should provide a cleaner quality signal than raw win_pct_diff. Correlation with target is +0.12. Expect modest Brier improvement. | {models: {catboost_mix: {append_features: [pyth_wpct_diff]}, lr_baseline: {append_features: [
        pyth_wpct_diff]}, mlp_30d: {append_features: [pyth_wpct_diff]}, xgb_mix: {
      append_features: [pyth_wpct_diff]}}}
 | 0.5700 (+0.0071) | - | - | 0.6789 (-0.0003) | improved |  |  |
| exp-012 | 2026-03-08 | Hard-hit rate allowed captures pitching quality beyond ERA — it measures quality of contact, a Statcast-era signal that ERA misses. Teams allowing fewer hard-hit balls should win more. Correlation is -0.08 (correct direction). May be partly redundant with ERA diff but captures a different dimension of pitching quality. | {models: {catboost_mix: {append_features: [hard_hit_diff_30d]}, lr_baseline: {append_features: [
        hard_hit_diff_30d]}, mlp_30d: {append_features: [hard_hit_diff_30d]}, xgb_mix: {
      append_features: [hard_hit_diff_30d]}}}
 | 0.5669 (-0.0031) | - | - | 0.6809 (+0.0020) | regressed |  |  |
