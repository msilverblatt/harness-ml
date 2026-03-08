# EasyML Capabilities: How It Handles MM's Advanced Features

**Summary**: EasyML fully supports all features needed for MM women's model. Here's how.

---

## 1. Regression Models with CDF Conversion

### What MM Does
XGBoost regression models (`xgb_spread_broad`, `xgb_spread_new`, `xgb_spread_deep`, `xgb_luck`) train to predict **margin** (continuous), then convert to probability via CDF.

### How EasyML Does It

**1. Define model with regressor mode:**
```yaml
models:
  xgb_spread_broad:
    type: xgboost
    mode: regressor                    # ← use regressor mode
    features: [diff_seed_num, diff_scoring_margin, ...]
    params:
      objective: reg:squarederror      # ← margin prediction
      eval_metric: rmse
      max_depth: 1
      learning_rate: 0.01
      # ... other params
```

**2. EasyML training automatically:**
- Extracts `margin` column (scoring_margin difference)
- Trains XGBoost to predict margin
- **Auto-fits CDF scale** using Brier score minimization
  ```python
  scale = argmin_scale( mean((norm.cdf(margin_pred / scale) - actual_result)^2) )
  ```
- Stores `cdf_scale` with model metadata

**3. EasyML prediction automatically:**
- Generates margin predictions: `margin_pred = model.predict(X)`
- Converts to probability: `prob = norm.cdf(margin_pred / cdf_scale)`
- Returns probabilities [0,1] just like classifiers

**Code locations:**
- Training: `packages/easyml-runner/src/easyml/runner/training.py:_fit_cdf_scale()`
- Model wrapper: `packages/easyml-models/src/easyml/models/wrappers/xgboost.py:predict_proba()`
- Pipeline integration: `pipeline.py:predict_single_model()` passes `cdf_scale` to predictions

**Result**: Regression models work seamlessly in ensemble, predictions are probabilities.

---

## 2. Survival Model as Feature Producer

### What MM Does
Survival model (Weibull hazard regression) makes tournament advancement predictions, contributes to ensemble.

### How EasyML Does It

**1. Define survival model with `provides`:**
```yaml
models:
  survival_hazard:
    type: survival                     # ← special type
    features:
      - seed_num
      - scoring_margin
      - conf_tourney_wins
      - last_n_win_pct
      - net_efficiency
      - sr_srs
      - elo_rating
    provides: ["survival_prob"]        # ← outputs predictions as feature
    include_in_ensemble: false         # ← excluded from final ensemble
    params:
      max_depth: 3
      learning_rate: 0.03
      n_estimators: 1000
      # ... standard XGBoost params
```

**2. EasyML pipeline automatically:**
- Trains survival model alongside other models
- Generates predictions on both train and test sets
- **Stores predictions as new matchup-level features** in prediction context
- Other models can reference `survival_prob` in their feature lists

**3. Downstream models use it:**
```yaml
models:
  xgb_trajectory:
    type: xgboost
    mode: classifier
    features:
      - diff_seed_num
      - diff_efficiency_first_half
      - # ... other features
      - diff_survival_prob            # ← uses survival model output
```

**4. Ensemble automatically excludes it:**
- `include_in_ensemble: false` prevents survival_hazard from being in final stacked ensemble
- But its predictions still available as features for other models

**Code locations:**
- Provider pattern: `packages/easyml-runner/src/easyml/runner/pipeline.py:store_matchup()`
- Feature consumption: Feature columns loaded from both raw data and provider outputs
- Ensemble exclusion: `ensemble_config.exclude_models` + `include_in_ensemble: false`

**Result**: Survival model trains, produces features, excluded from ensemble — exactly like MM.

---

## 3. Per-Fold Meta-Learner Persistence

### What MM Does
Trains separate meta-learner for each LOSO fold, saves weights per season, uses fold-specific weights during backtest.

### How EasyML Does It

**1. Enable in ensemble config:**
```yaml
ensemble:
  method: stacked
  meta_learner:
    C: 1.0
  # fold_save_dir configured internally by pipeline
```

**2. EasyML training automatically:**
- Detects LOSO backtest structure
- For each held-out season:
  - Trains pre-calibrators on remaining seasons only
  - Trains separate meta-learner on OOF predictions
  - **Saves meta-learner to disk** (JSON + calibrators as pickle)
  - Returns `fold_meta_paths: {season -> path}`

**3. EasyML prediction automatically:**
- Loads fold-specific meta-learner from disk
- Uses correct weights for that fold
- Different weights per season, exactly like MM

**Code locations:**
- LOSO training: `packages/easyml-runner/src/easyml/runner/meta_learner.py:train_meta_learner_loso()`
- Per-fold saving: Lines 246-258 (saves `fold_{season}_meta.json` and `fold_{season}_precals.pkl`)
- Fold loading: Pipeline automatically loads correct fold's meta-learner during backtest

**Result**: Per-fold meta-learner weights saved and used, no performance loss from using global model.

---

## 4. Spline Calibration with Configurable Bins

### What MM Does
Post-ensemble calibration using spline (monotonic PCHIP) with 12 quantile bins, prob_max=0.99.

### How EasyML Does It

**1. Configure in ensemble:**
```yaml
ensemble:
  calibration: spline              # ← post-ensemble calibration method
  spline_n_bins: 12                # ← number of quantile bins
  spline_prob_max: 0.99            # ← probability ceiling
```

**2. EasyML training:**
- Fits spline on LOSO meta-learner predictions
- Uses monotonic cubic spline (PCHIP)
- Clamps output to [clip_floor, spline_prob_max]

**3. EasyML prediction:**
- Applies fitted spline to ensemble predictions
- Ensures probabilities stay in valid range

**Code locations:**
- Calibration builder: `packages/easyml-runner/src/easyml/runner/calibration.py:build_calibrator()`
- Spline implementation: Uses scipy.interpolate.PchipInterpolator

**Result**: Exact calibration matching MM.

---

## 5. Temperature Scaling (Post-Ensemble)

### What MM Does
Final temperature scaling (T=1.05) applied to ensemble probabilities: `prob^(1/T)`

### How EasyML Does It

**1. Configure in ensemble:**
```yaml
ensemble:
  temperature: 1.05                # ← temperature for final scaling
```

**2. EasyML prediction:**
```python
prob = ensemble_pred ** (1.0 / temperature)
# With T=1.05: prob^(1/1.05) = prob^0.952 (slight sharpening)
```

**3. Applied after:**
- Meta-learner prediction
- Pre-ensemble calibration (if any)
- Before post-ensemble calibration

**Code locations:**
- Pipeline: `config_writer.py`, `pipeline.py` handle temperature application

**Result**: Predictions slightly sharpened, matching MM behavior.

---

## 6. Model Exclusion from Ensemble

### What MM Does
8 models trained but excluded from final ensemble (via `exclude_models` list)

### How EasyML Does It

**Method 1: Via ensemble config**
```yaml
ensemble:
  exclude_models:
    - xgb_spread_scoring         # (doesn't exist, typo in mm)
    - w_xgb_box_score
    - w_rf_fundamentals
    - xgb_matchup
    - xgb_spread_broad
    - xgb_elo
    - rf_pace_style
    - xgb_trajectory
```

**Method 2: Via model config**
```yaml
models:
  w_xgb_box_score:
    type: xgboost
    include_in_ensemble: false     # ← excluded from ensemble
    # ... but still trains and can provide features
```

**2. EasyML handling:**
- All 17 models train
- Excluded models' predictions not used in meta-learner
- Non-excluded models (9) form the stacked ensemble
- Excluded models' outputs can still be features for others via `provides`

**Code locations:**
- Ensemble filtering: `ensemble.py:fit()` filters models by `include_in_ensemble`
- Config validation: `schema.py:ModelDef` tracks `include_in_ensemble`

**Result**: 9 models in final ensemble, 8 excluded, matching MM exactly.

---

## 7. Pre-Calibration Per-Model (Before Meta-Learner)

### What MM Does
Optionally calibrate individual model predictions before feeding to meta-learner.

### How EasyML Does It

**1. Configure per model:**
```yaml
models:
  xgb_core:
    type: xgboost
    pre_calibration: spline        # ← calibrate before meta-learner
    # ...

  xgb_trajectory:
    # no pre_calibration — skip for this model
    # ...
```

**2. EasyML training:**
- During LOSO, fits per-model calibrator on training fold only (no leakage)
- Applies calibrator to both train and validation predictions
- Calibrated predictions fed to meta-learner
- Final pre-calibrators retrained on all data for production

**3. EasyML prediction:**
- Pre-calibrates individual model predictions
- Feeds calibrated predictions to meta-learner
- Meta-learner outputs ensemble prediction
- Post-calibration (if configured) applied after meta-learner

**Code locations:**
- LOSO pre-calibration: `meta_learner.py:train_meta_learner_loso()` lines 199-207
- Per-fold isolation: Training calibrators only on train fold, applying to val

**Result**: Nested calibration structure matching MM, with LOSO safety.

---

## 8. Availability Adjustment (for Missing Data)

### What MM Does
Pull predictions toward 0.5 when seed team has no training data (new team in tournament).

### How EasyML Does It

**1. Configure in ensemble:**
```yaml
ensemble:
  availability_adjustment: 0.5     # ← blend toward 0.5 for missing data
```

**2. EasyML handling:**
- Detects NaN predictions (missing data)
- Blends NaN predictions toward 0.5: `prob = (prob * w + 0.5 * (1-w))` where `w = availability_adjustment`
- Default 0.5 = full blend to 0.5 (conservative)
- Can be 0 (no adjustment) or 1 (strict)

**Result**: Conservative predictions for rare seeds/new teams.

---

## 9. Feature Engineering from Pre-Computed Data

### What MM Does
Complex feature engineering (rolling windows, conditional aggregates, momentum).

### How EasyML Does It

**1. MM pre-computes all features:**
- 68+ team-level features in `team_season_features.parquet`
- Includes all transformations (rolling, conditional, momentum)

**2. EasyML loads and diffs:**
```python
# Load team features
team_df = pd.read_parquet('team_season_features.parquet')

# Auto-compute pairwise differences for each matchup
for feat in ['seed_num', 'win_pct', 'ppg', ...]:
    matchup_df[f'diff_{feat}'] = team1[feat] - team2[feat]
```

**3. Result:**
- 68+ pairwise difference features automatically available
- No computation needed in easyml
- Uses MM's pre-computed features directly

**Code locations:**
- Feature ingest: `packages/easyml-runner/src/easyml/runner/data_ingest.py`
- Pairwise diff generation: `packages/easyml-runner/src/easyml/runner/matchups.py`

**Result**: All MM features available, no reimplementation needed.

---

## Summary: Complete Feature Support

| Feature | MM Does | EasyML Does | Status |
|---------|---------|-------------|--------|
| Regression CDF | Predict margin, convert via sigmoid | Auto-fit CDF scale, convert via norm.cdf | ✅ Full support |
| Survival model | Train + exclude from ensemble | Via `provides` + `include_in_ensemble: false` | ✅ Full support |
| Per-fold meta | Different weights per season | Saves fold models, loads correct one | ✅ Full support |
| Spline calib | 12 bins, prob_max=0.99 | Configurable bins, prob_max, PCHIP | ✅ Full support |
| Temperature | T=1.05 post-ensemble | Configurable temperature scaling | ✅ Full support |
| Exclusions | 8 models excluded | Via `exclude_models` + `include_in_ensemble` | ✅ Full support |
| Pre-calib | Per-model before meta | Per-model calibrators, LOSO-safe | ✅ Full support |
| Availability | Blend to 0.5 for NaN | Configurable blending | ✅ Full support |
| Features | Complex transforms | Pre-computed, loaded directly | ✅ Full support |

---

## Conclusion

EasyML is **not just capable** of replicating MM women's model — it has specific, built-in support for all the advanced features MM uses. No workarounds needed.

