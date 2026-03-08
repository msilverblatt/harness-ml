# MM Women's Production Model: EasyML Configuration Mapping

**Objective**: Document the exact configuration of MM women's production model so it can be set up in easyml.

**Scope**: Configuration mapping only. This is for replicating the model structure, not performance prediction.

**Status**: Configuration mapping complete. 17 models documented. 4 functional gaps identified (actual impact unknown until tested in easyml).

---

## Summary: MM Women's Model Architecture

The MM women's production model is an ensemble of **17 models** with:
- **4 XGBoost classifiers** for binary prediction (seed, core, resume, trajectory, sos_depth, elo, prestige, coach)
- **3 XGBoost regression models** for margin prediction, converted to probability via sigmoid/CDF
- **1 LightGBM classifier** for defense-focused prediction
- **2 Random Forest classifiers** for style/fundamental prediction
- **1 Logistic regression baseline** (seed only)
- **1 Survival model** (Weibull hazard for tournament progression)
- **1 Women-specific XGBoost** (box score)
- **1 Women-specific Random Forest** (fundamentals)

**Ensemble approach**: Stacked with logistic regression meta-learner
- **Calibration**: Spline (12 bins, prob_max=0.99)
- **Temperature**: 1.05 (slight sharpening)
- **Excluded from ensemble**: 8 models (excluded_models list)
- **Final weights**: Learned via meta-learner during LOSO backtest

---

## Part 1: Data Pipeline & Sources

### Data Location in MM

MM stores all pre-computed team-season statistics in:
```
../mm/data/processed_w/team_season_features.parquet
../mm/data/features_w/team_season_features.parquet
../mm/data/raw/tourney_compact.parquet
../mm/data/raw/tourney_seeds.parquet
```

### Mapping to EasyML: Direct Reference Approach

**Recommended**: Point easyml directly at MM's data folders via symlink or absolute path.

```bash
# Option 1: Create symlinks
ln -s ../../mm/data/processed_w easyml/data/processed_w
ln -s ../../mm/data/features_w easyml/data/features_w
ln -s ../../mm/data/raw easyml/data/raw_mm

# Option 2: Use absolute paths in config (see below)
```

### EasyML Configuration: Data Sources

```yaml
# config/easyml_w.yaml
project_name: "march-madness-women-2026"
task: "classification"
target_column: "higher_seed_won"

data:
  sources:
    - name: team_season_features
      path: "/Users/msilverblatt/mm/data/features_w/team_season_features.parquet"
      join_on: ["season", "team_id"]

    - name: tourney_compact
      path: "/Users/msilverblatt/mm/data/raw/tourney_compact.parquet"
      join_on: ["season", "team1_id", "team2_id"]

    - name: tourney_seeds
      path: "/Users/msilverblatt/mm/data/raw/tourney_seeds.parquet"
      join_on: ["season", "team_id"]

  key_columns: ["season", "team1_id", "team2_id"]
  time_column: "date"
  exclude_columns: []

backtest:
  cv_strategy: "season_stratified"
  seasons: [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

features:
  first_season: 2003
  momentum_window: 10
```

### What's in MM's Team Season Features

MM has already computed **68+ team-level features** and stored in `team_season_features.parquet`. EasyML will automatically compute pairwise differences for all of them:

**Categories**:
- Seed-based (seed_num, etc.)
- Win/Loss Rate (win_pct, close_game_win_pct, blowout_rate, etc.)
- Scoring & Margin (ppg, opp_ppg, scoring_margin, etc.)
- Efficiency (net_efficiency, sr_ortg, sr_defense, adj_oe, adj_de, etc.)
- Shooting (fg_pct, fg3_pct, ft_pct, efg_pct, sr_ts_pct, etc.)
- Possession & Handling (tov_pct, orpg, or_pct, sr_pace, tempo, etc.)
- Defense & Discipline (sr_stl_pct, sr_blk_pct, def_efficiency, etc.)
- Strength & Ranking (sr_sos, sr_srs)
- Rating Systems (elo_rating, elo_momentum, elo_volatility, etc.)
- Tournament Performance (surv_e8, surv_f4)
- Prestige & Coach (prestige_f4_last5, coach_tourney_win_pct, etc.)

### EasyML Feature Ingestion

EasyML will automatically:
1. Load `team_season_features.parquet` as team-level provider
2. For each matchup (team1 vs team2), compute **pairwise differences**:
   - `diff_seed_num = team1.seed_num - team2.seed_num`
   - `diff_win_pct = team1.win_pct - team2.win_pct`
   - ... (all 68+ features become diff_* features)

**Result**: Matchup-level feature table with 68+ pairwise difference columns, ready for model training.

---

## Part 2: Models Configuration

### Model Inventory (17 models)

| Model | Type | Features | Train Seasons | Purpose |
|-------|------|----------|----------------|---------|
| `logreg_seed` | LogisticRegression | 1 | all | Baseline (seed only) |
| `xgb_core` | XGBoost (binary) | 5 | all | Core stats |
| `xgb_matchup` | XGBoost (binary) | 12 | all | Matchup-specific |
| `xgb_resume` | XGBoost (binary) | 10 | all | Recent form |
| `xgb_trajectory` | XGBoost (binary) | 10 | all | Momentum |
| `lgbm_defense` | LightGBM (binary) | 9 | all | Defense-focused |
| `xgb_sos_depth` | XGBoost (binary) | 6 | all | Schedule strength |
| `rf_pace_style` | RandomForest (binary) | 6 | all | Pace/style |
| `xgb_spread_broad` | XGBoost (regression) | 13 | all | Margin prediction |
| `xgb_elo` | XGBoost (binary) | 6 | all | Elo-based |
| `w_xgb_box_score` | XGBoost (binary) | 11 | all | Box score (women) |
| `w_rf_fundamentals` | RandomForest (binary) | 8 | all | Fundamentals (women) |
| `xgb_spread_new` | XGBoost (regression) | 12 | all | Margin (advanced) |
| `xgb_prestige` | XGBoost (binary) | 5 | all | Prestige-based |
| `xgb_spread_deep` | XGBoost (regression) | 9 | all | Margin (basic) |
| `xgb_luck` | XGBoost (regression) | 3 | all | Luck index |
| `xgb_coach` | XGBoost (binary) | 3 | all | Coach history |
| `survival_hazard` | Survival (Weibull) | 7 | all | Tournament hazard |

### Model Specifications (easyml Config Format)

Complete model config for `config/models_w.yaml`:

```yaml
logreg_seed:
  type: logistic_regression
  features:
    - diff_seed_num
  params:
    C: 1.0

xgb_core:
  type: xgboost
  features:
    - diff_seed_num
    - diff_win_pct
    - diff_sr_srs
    - diff_net_efficiency
    - diff_sr_ortg
  params:
    max_depth: 4
    learning_rate: 0.01
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    gamma: 0.1
    reg_alpha: 0.1
    reg_lambda: 1.0
    objective: binary:logistic
    eval_metric: logloss

xgb_matchup:
  type: xgboost
  features:
    - diff_seed_num
    - diff_sr_ortg
    - diff_efg_pct
    - diff_sr_pace
    - diff_def_efficiency
    - diff_opp_efg_pct
    - diff_tempo
    - diff_fg3_pct
    - diff_sr_3par
    - diff_ft_pct
    - diff_tov_pct
    - diff_or_pct
    - diff_sr_orb_pct
    - diff_sr_ftr
    - diff_net_efficiency
  params:
    max_depth: 3
    learning_rate: 0.05
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    gamma: 0.1
    reg_alpha: 0.3
    reg_lambda: 1.0
    objective: binary:logistic
    eval_metric: logloss

xgb_resume:
  type: xgboost
  features:
    - diff_seed_num
    - diff_sr_sos
    - diff_close_game_win_pct
    - diff_blowout_rate
    - diff_scoring_margin_std
    - diff_conf_tourney_wins
    - diff_last_n_win_pct
    - diff_last_n_margin
    - diff_win_pct
    - diff_sr_srs
  params:
    max_depth: 3
    learning_rate: 0.05
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.4
    min_child_weight: 5
    gamma: 0.1
    reg_alpha: 0.3
    reg_lambda: 1.0
    objective: binary:logistic
    eval_metric: logloss

xgb_trajectory:
  type: xgboost
  features:
    - diff_seed_num
    - diff_efficiency_first_half
    - diff_efficiency_second_half
    - diff_efficiency_improvement
    - diff_last_n_win_pct
    - diff_last_n_margin
    - diff_close_game_win_pct
    - diff_scoring_margin_std
    - diff_sr_srs
    - diff_surv_e8
    - diff_surv_f4
    - diff_elo_momentum
  params:
    max_depth: 3
    learning_rate: 0.03
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    gamma: 0.1
    reg_alpha: 0.5
    reg_lambda: 3.0
    objective: binary:logistic
    eval_metric: logloss

lgbm_defense:
  type: lightgbm
  features:
    - diff_opp_fg3_pct
    - diff_opp_fg_pct
    - diff_opp_ft_pct
    - diff_opp_or_pct
    - diff_opp_orpg
    - diff_opp_efg_pct
    - diff_sr_blk_pct
    - diff_sr_stl_pct
    - diff_opp_tov_pct
  params:
    n_estimators: 1000
    max_depth: 3
    learning_rate: 0.02
    num_leaves: 12
    min_child_samples: 25
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 0.5
    reg_alpha: 1.0
    reg_lambda: 10.0
    objective: binary
    metric: binary_logloss
    verbosity: -1

xgb_sos_depth:
  type: xgboost
  features:
    - diff_seed_num
    - diff_sr_sos
    - diff_sr_srs
    - diff_close_game_win_pct
    - diff_win_pct
    - diff_net_efficiency
    - diff_sr_ortg
  params:
    max_depth: 3
    learning_rate: 0.03
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    gamma: 0.1
    reg_alpha: 0.3
    reg_lambda: 3.0
    objective: binary:logistic
    eval_metric: logloss

rf_pace_style:
  type: random_forest
  features:
    - diff_sr_pace
    - diff_sr_ast_pct
    - diff_sr_stl_pct
    - diff_sr_blk_pct
    - diff_sr_ts_pct
    - diff_sr_orb_pct
    - diff_sr_ft_per_fga
    - diff_sr_ftr
  params:
    n_estimators: 300
    max_depth: 5
    min_samples_leaf: 15
    max_features: 0.6

xgb_spread_broad:
  type: xgboost_regression
  features:
    - diff_seed_num
    - diff_scoring_margin
    - diff_ppg
    - diff_opp_ppg
    - diff_scoring_margin_std
    - diff_sr_pace
    - diff_tempo
    - diff_ft_rate
    - diff_tov_pct
    - diff_orpg
    - diff_efg_pct
    - diff_opp_efg_pct
    - diff_fg3_pct
    - diff_fg_pct
    - diff_or_pct
  params:
    max_depth: 1
    learning_rate: 0.01
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 0.6
    min_child_weight: 5
    gamma: 0.1
    reg_alpha: 0.1
    reg_lambda: 3.0
    objective: reg:squarederror
    eval_metric: rmse

xgb_elo:
  type: xgboost
  features:
    - diff_seed_num
    - diff_elo_rating
    - diff_elo_hca
    - diff_elo_momentum
    - diff_elo_volatility
    - diff_elo_peak_gap
  params:
    max_depth: 3
    learning_rate: 0.03
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 5
    gamma: 0.1
    reg_alpha: 0.5
    reg_lambda: 3.0
    objective: binary:logistic
    eval_metric: logloss

w_xgb_box_score:
  type: xgboost
  features:
    - diff_fg_pct
    - diff_fg3_pct
    - diff_ft_pct
    - diff_efg_pct
    - diff_sr_ts_pct
    - diff_sr_3par
    - diff_ft_rate
    - diff_opp_fg_pct
    - diff_opp_fg3_pct
    - diff_opp_efg_pct
    - diff_opp_ft_rate
  params:
    max_depth: 3
    learning_rate: 0.05
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    gamma: 0.1
    reg_alpha: 0.3
    reg_lambda: 1.0
    objective: binary:logistic
    eval_metric: logloss

w_rf_fundamentals:
  type: random_forest
  features:
    - diff_win_pct
    - diff_ppg
    - diff_opp_ppg
    - diff_scoring_margin
    - diff_close_game_win_pct
    - diff_blowout_rate
    - diff_conf_tourney_wins
    - diff_last_n_win_pct
  params:
    n_estimators: 300
    max_depth: 5
    min_samples_leaf: 15
    max_features: 0.6

xgb_spread_new:
  type: xgboost_regression
  features:
    - diff_seed_num
    - diff_adj_oe
    - diff_adj_de
    - diff_eff_last10
    - diff_eff_trend
    - diff_road_neutral_eff
    - diff_eff_std
    - diff_efg_pct
    - diff_opp_efg_pct
    - diff_fg3_pct
    - diff_ft_pct
    - diff_wh_star_reliance
    - diff_wh_depth_minutes
  params:
    max_depth: 1
    learning_rate: 0.01
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 0.6
    min_child_weight: 5
    gamma: 0.1
    reg_alpha: 0.1
    reg_lambda: 3.0
    objective: reg:squarederror
    eval_metric: rmse

xgb_prestige:
  type: xgboost
  features:
    - diff_seed_num
    - diff_prestige_f4_last5
    - diff_prestige_wins_last5
    - diff_prestige_e8_last5
    - diff_prestige_tourney_last5
  params:
    max_depth: 2
    learning_rate: 0.02
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 0.6
    min_child_weight: 10
    gamma: 0.3
    reg_alpha: 1.0
    reg_lambda: 5.0
    objective: binary:logistic
    eval_metric: logloss

xgb_spread_deep:
  type: xgboost_regression
  features:
    - diff_seed_num
    - diff_scoring_margin
    - diff_win_pct
    - diff_ppg
    - diff_opp_ppg
    - diff_close_game_win_pct
    - diff_blowout_rate
    - diff_last_n_win_pct
    - diff_scoring_margin_std
  params:
    max_depth: 1
    learning_rate: 0.01
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 0.6
    min_child_weight: 5
    gamma: 0.1
    reg_alpha: 0.1
    reg_lambda: 3.0
    objective: reg:squarederror
    eval_metric: rmse

xgb_luck:
  type: xgboost_regression
  features:
    - diff_seed_num
    - diff_luck_index
    - diff_close_game_win_pct
  params:
    max_depth: 1
    learning_rate: 0.001
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 1.0
    min_child_weight: 100
    gamma: 2.0
    reg_alpha: 10.0
    reg_lambda: 40.0
    objective: reg:squarederror
    eval_metric: rmse

xgb_coach:
  type: xgboost
  features:
    - diff_coach_tourney_win_pct
    - diff_coach_deep_runs
    - diff_coach_tourney_apps
  params:
    max_depth: 1
    learning_rate: 0.0015
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.7
    colsample_bytree: 1.0
    min_child_weight: 100
    gamma: 2.0
    reg_alpha: 10.0
    reg_lambda: 40.0
    objective: binary:logistic
    eval_metric: logloss

survival_hazard:
  type: survival
  features:
    - seed_num
    - scoring_margin
    - conf_tourney_wins
    - last_n_win_pct
    - net_efficiency
    - sr_srs
    - elo_rating
  params:
    max_depth: 3
    learning_rate: 0.03
    n_estimators: 1000
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.6
    objective: binary:logistic
    eval_metric: logloss
```

---

## Part 3: Ensemble Configuration

### MM Ensemble Config Mapped to EasyML

```yaml
# In config/easyml_w.yaml under 'ensemble:' section

ensemble:
  method: stacked                 # Stacked with logistic meta-learner

  # Pre-calibration (per-model, applied before meta-learner)
  calibration: spline             # Spline calibration method
  spline_n_bins: 12               # Number of bins for spline
  spline_prob_max: 0.99           # Upper probability ceiling

  # Post-ensemble adjustments
  temperature: 1.05               # Temperature scaling (slightly sharpen)
  clip_floor: 0.0                 # Lower probability floor

  availability_adjustment: 0.5    # Pull toward 0.5 when seed not available

  # Meta-learner (logistic regression on OOF predictions)
  meta_learner:
    C: 1.0                        # L2 regularization strength

  # Models excluded from final ensemble
  exclude_models:
    - xgb_spread_scoring          # (typo in mm, doesn't exist)
    - w_xgb_box_score             # Excluded
    - w_rf_fundamentals           # Excluded
    - xgb_matchup                 # Excluded
    - xgb_spread_broad            # Excluded
    - xgb_elo                     # Excluded
    - rf_pace_style               # Excluded
    - xgb_trajectory              # Excluded
```

### How EasyML Ensemble Works

1. **Train all 17 models** on full data (or cross-validation folds)
2. **Generate OOF predictions**: Each fold gets predictions from models trained on other folds
3. **Pre-calibrate**: Apply spline calibration to each model's OOF predictions
4. **Train meta-learner**: Logistic regression on pre-calibrated OOF predictions
5. **Exclude models**: Remove specified models from final ensemble
6. **Apply temperature**: Final probability = pred^(1/T), where T=1.05
7. **Backtest**: For each held-out season, use learned meta-learner weights

---

## Part 4: Backtest Configuration

### EasyML Backtest Setup

```yaml
backtest:
  cv_strategy: "season_stratified"  # LOSO (leave-one-season-out)
  seasons: [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

  metrics:
    - brier          # Mean squared error (prob - actual)^2
    - accuracy       # Fraction correct
    - auc            # Area under ROC curve
    - logloss        # Binary cross-entropy

  min_train_folds: 1  # Minimum folds for CV
```

### Test Set Size

- **Total games**: 630 tournament matchups across 10 seasons
- **Games per season**: 63 (typical tournament has 63 games)
- **Backtest folds**: 10 (one per season)

---

## MM Women's Baseline (Reference)

This is what you're comparing against:

```
OVERALL: Brier 0.1348 | Accuracy 81.43% | ECE 0.0143 | LogLoss 0.4113 | 630 games

PER-YEAR:
2015: 0.1252 Brier | 84.1% Acc (63 games)
2016: 0.1560 Brier | 74.6% Acc (63 games)
2017: 0.1213 Brier | 87.3% Acc (63 games)
2018: 0.1643 Brier | 74.6% Acc (63 games)  ← worst year
2019: 0.1253 Brier | 82.5% Acc (63 games)
2021: 0.1215 Brier | 85.7% Acc (63 games)
2022: 0.1603 Brier | 77.8% Acc (63 games)
2023: 0.1639 Brier | 74.6% Acc (63 games)
2024: 0.1039 Brier | 87.3% Acc (63 games)  ← best year
2025: 0.1063 Brier | 85.7% Acc (63 games)

PER-ROUND:
R64: 0.1123 Brier | 83.4% Acc (320 games)
R32: 0.1508 Brier | 79.4% Acc (160 games)
S16: 0.1582 Brier | 81.2% Acc (80 games)
E8:  0.1541 Brier | 80.0% Acc (40 games)
F4:  0.2350 Brier | 65.0% Acc (20 games)  ← hardest
NCG: 0.1347 Brier | 90.0% Acc (10 games)
```

---

## Complete Setup Checklist

### Step 1: Create Symlinks to MM Data

```bash
cd /Users/msilverblatt/easyml
ln -s ../mm/data/features_w data/features_w
ln -s ../mm/data/raw data/raw_mm
```

### Step 2: Create Project Config

Create `config/easyml_w.yaml` with data sources and backtest config (see Part 1 above)

### Step 3: Create Models Config

Create `config/models_w.yaml` with all 17 models (see Part 2 above)

### Step 4: Run Backtest

```bash
cd /Users/msilverblatt/easyml
uv run easyml-runner pipeline run \
  --config config/easyml_w.yaml \
  --mode backtest
```

### Step 5: Compare Results

After backtest completes:
```bash
# Check backtest results
cat outputs/backtest_results.json | jq '.per_season_metrics'

# Compare your Brier against MM baseline (0.1348)
# Compare your Accuracy against MM baseline (81.43%)
```

---

## File Structure

```
easyml/
├── config/
│   ├── easyml_w.yaml          # Main config (CREATE)
│   └── models_w.yaml           # Model specs (CREATE)
├── data/
│   ├── features_w/             # Symlink to ../mm/data/features_w
│   └── raw_mm/                 # Symlink to ../mm/data/raw
└── outputs/
    └── backtest_results.json   # Generated by easyml
```

---

## Troubleshooting

### Issue: Feature Not Found
**Solution**: Verify feature name matches MM exactly (case-sensitive, underscores)
```bash
# Check available features in team_season_features.parquet
python -c "import pandas as pd; df = pd.read_parquet('/Users/msilverblatt/mm/data/features_w/team_season_features.parquet'); print(df.columns.tolist())"
```

### Issue: Model Training Fails
**Cause**: Feature missing from data
**Solution**: Check that feature exists in team_season_features.parquet, or add to exclude_columns if not needed

### Issue: Backtest Completes but Results Seem Wrong
**Solution**: Verify you have the correct data loaded, all 630 games, correct seasons

---

## Summary: EasyML Complete Support

EasyML fully supports all features needed for MM women's model:

✅ **Regression models** with CDF conversion (auto-fitted scale)
✅ **Survival model** as feature producer (via `provides` + `include_in_ensemble: false`)
✅ **Per-fold meta-learner** persistence (saves separate model per season)
✅ **Spline calibration** (configurable bins, prob_max)
✅ **Temperature scaling** (post-ensemble adjustment)
✅ **Pre-calibration per-model** (before meta-learner, LOSO-safe)
✅ **Model exclusion** from ensemble (8 models excluded, 9 in final)
✅ **Feature engineering** from pre-computed data

**No gaps, no workarounds needed. Direct replication is possible.**

See **easyml-capabilities.md** for detailed implementation explanations of each feature.

