# EasyML Gaps vs MM: Functional Incompatibilities

**Summary**: 4 gaps identified where easyml cannot directly replicate MM's women's model configuration. Actual impact unknown until tested.

---

## Gap 1: Regression Models with CDF Conversion

### What MM Does
- **3 XGBoost regression models** train to predict **scoring margin** (continuous value):
  - `xgb_spread_broad` (13 features, margin prediction)
  - `xgb_spread_new` (12 features, margin + advanced efficiency)
  - `xgb_spread_deep` (9 features, margin + basic stats)
  - `xgb_luck` (3 features, luck index)
- Converts margin predictions to **probability** via **sigmoid/CDF**:
  ```
  probability = sigmoid(margin / scale_factor)
  ```
- These are configured with `objective: reg:squarederror` in MM's models/production_w.yaml

### What EasyML Does
- All models must be **binary classifiers** (`objective: binary:logistic`)
- No regression mode for XGBoost
- Cannot convert regression output to probability via CDF

### Impact
- **Unknown** - can only test by running backtest
- These 3 models might contribute to MM's 0.1348 Brier, or they might not
- Excluding them would be a workaround, but reduces ensemble diversity

### Fix Options

**Option A: Convert to Classifiers** (Zero effort)
- Change `type: xgboost_regression` to `type: xgboost` with `objective: binary:logistic`
- Simpler, works immediately
- May or may not impact metrics

**Option B: Implement Regression Mode** (Non-trivial effort)
- Add `type: xgboost_regression` model type to easyml
- Implement CDF sigmoid conversion in ensemble postprocessing
- Requires changes to model registry and ensemble pipeline

**Recommendation**: Start with Option A (convert to classifiers), see if it matters

---

## Gap 2: Survival Model (Weibull Hazard)

### What MM Does
- **Survival model** predicts **tournament hazard** (elimination probability per round)
- Uses **7 features** (seed, margin, efficiency, rating, tourney wins, SRS, Elo)
- Trains on **historical tournament progression** (which round each team reached)
- Output: Cumulative survival probability (advancing to each round)
- Configured with `type: survival` in MM's models/production_w.yaml

### What EasyML Does
- No survival model support
- Only binary classification models

### Impact
- **Unknown** - this model may or may not be critical to MM's 0.1348 Brier
- Only way to know: test with and without it

### Fix Options

**Option A: Train Separately, Import Predictions** (Practical)
- Train survival model in MM's codebase
- Export predictions as CSV
- Include as pre-computed feature in easyml
- Simplest approach, minimal code changes

**Option B: Implement Survival Wrapper** (High effort)
- Create Weibull hazard model wrapper in easyml
- Requires specialized survival analysis library
- Non-standard output format

**Option C: Replace with Classifier** (Zero effort)
- Train XGBoost binary classifier with same 7 features
- Simplest alternative
- May not capture tournament progression signal

**Recommendation**: Option A (pragmatic) or Option C (simplest)

---

## Gap 3: Per-Fold Meta-Learner Persistence

### What MM Does
- **LOSO backtest**: Leave one season out, train on others
- For each fold:
  1. Train all models on (N-1) seasons
  2. Generate OOF predictions on held-out season
  3. **Train separate meta-learner on OOF from that fold**
  4. **Save meta-learner model to disk**
5. During backtest prediction: Load fold-specific meta-learner
6. Result: **Different ensemble weights per season**

### What EasyML Does
- **LOSO backtest structure exists** ✓
- **Single meta-learner trained on all OOF predictions** (not per-fold)
- Same meta-learner weights applied to all folds
- Result: **Identical ensemble weights across all seasons**

### Impact
- **Unknown** - different weights might matter for MM's 0.1348 Brier, or they might not
- Only way to know: implement both and compare

### Example Difference
```
MM (per-fold):
  2015 fold: xgb_core weight=0.25, xgb_resume weight=0.18, ...
  2016 fold: xgb_core weight=0.22, xgb_resume weight=0.21, ...
  (weights vary by season)

EasyML (single):
  All folds: xgb_core weight=0.23, xgb_resume weight=0.19, ...
  (weights same for all seasons)
```

### Fix
**Implementation**: Save per-fold meta-learner models (non-trivial effort)
- Modify LOSO training to save separate meta-learner per fold
- Load fold-specific meta-learner during backtest prediction
- Requires changes to meta-learner persistence layer

**Recommendation**: Don't implement unless testing shows per-fold matters

---

## Gap 4: Feature Engineering Transformations

### What MM Does
- **Advanced feature transformations**:
  - Rolling windows: `last_n_win_pct` (win pct over last 10 games)
  - Conditional aggregates: `close_game_win_pct` (win pct when margin ≤ 3 pts)
  - Temporal splits: `efficiency_first_half` (efficiency in games before mid-season)
  - Momentum: `elo_momentum` (Elo rating change over recent games)
  - Multi-source joins: Coach data + tournament history + Elo ratings

### What EasyML Does
- ✓ Load team-level features from parquet
- ✓ Auto-compute pairwise differences (diff_*)
- ✗ Rolling windows (no native support)
- ✗ Conditional filters (avg when condition)
- ✗ Temporal splits (split season by date)
- ✗ Momentum calculations (exponential decay)

### Impact
- **Zero** - MM's features are pre-computed in team_season_features.parquet
- EasyML just loads and diffs them
- No gap if using MM's pre-computed data

**Status**: ✓ Not a blocker. Use MM's pre-computed team_season_features.parquet.

---

## Summary Table: All Gaps

| Gap | Type | Status | Impact | Fix Effort | Recommendation |
|-----|------|--------|--------|-----------|-----------------|
| Regression CDF | Functional | Confirmed | Unknown | Medium | Start with Option A (classifiers) |
| Survival model | Functional | Confirmed | Unknown | High | Option A (train separate) or C (skip) |
| Per-fold meta-learner | Functional | Confirmed | Unknown | Medium | Skip unless testing shows it matters |
| Feature transforms | Data | Not a gap | None | N/A | Use MM's pre-computed data |

---

## What to Do Now

1. **Create configs** using mm-women-config-mapping.md
2. **For Gap 1**: Convert regression models to classifiers (simplest start)
3. **For Gap 2**: Skip survival model or train separately (can add later)
4. **For Gap 3**: Use single meta-learner (don't implement per-fold)
5. **Run backtest** and see what you get
6. **Compare to MM baseline**: Brier 0.1348, Accuracy 81.43%
7. **Decide based on results**: If metrics are close, gaps don't matter much. If metrics are bad, investigate which gaps are causing it.

---

## Notes

- This document identifies what easyml *cannot do*, not what *will fail*.
- All gaps are addressable with code changes.
- Most gaps have pragmatic workarounds.
- Only testing will reveal which gaps actually impact performance.

