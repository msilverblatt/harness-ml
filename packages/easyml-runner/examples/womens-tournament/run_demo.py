"""Women's Tournament Prediction Demo — easyml-runner.

Demonstrates the programmatic Project API for building an ML pipeline
without writing any YAML configuration files.

mm baseline to compare against:
  Brier: 0.1361  |  Accuracy: 81.59%  |  ECE: 0.0339
"""
from pathlib import Path

from easyml.runner import Project

PROJECT_DIR = Path(__file__).parent

# -----------------------------------------------------------------------
# Step 0: Create project + explore data
# -----------------------------------------------------------------------

project = Project(PROJECT_DIR)
project.set_data(features_dir="data/features", gender="W")

# What features do we have?
diff_features = project.available_features(prefix="diff_")
print(f"Available diff_ features: {len(diff_features)}")

# What does the data look like?
profile = project.profile()
print(profile.format_summary())
print()

# -----------------------------------------------------------------------
# Step 1: Declare data sources for leakage tracking
# -----------------------------------------------------------------------

project.add_source(
    "kaggle_seeds",
    temporal_safety="pre_tournament",
    outputs=["seed_num"],
    leakage_notes="Seeds are public before tournament begins",
)
project.add_source(
    "sports_reference",
    temporal_safety="pre_tournament",
    outputs=["sr_srs", "sr_sos", "sr_pace", "sr_ortg", "sr_drtg",
             "sr_efg_pct", "sr_tov_pct", "sr_orb_pct", "sr_ft_per_fga",
             "sr_opp_efg_pct", "sr_opp_tov_pct", "sr_opp_orb_pct", "sr_opp_ftr"],
    leakage_notes="Regular-season stats, scraped before tournament",
)
project.add_source(
    "composites",
    temporal_safety="pre_tournament",
    outputs=["composite_net", "composite_offense", "composite_defense",
             "composite_rank", "scoring_margin", "win_pct", "ppg",
             "opp_ppg", "scoring_margin_std"],
    leakage_notes="Derived from pre-tournament component ratings",
)
project.add_source(
    "prestige",
    temporal_safety="pre_tournament",
    outputs=["prestige_f4_last5", "prestige_wins_last5",
             "prestige_e8_last5", "prestige_tourney_last5"],
    leakage_notes="Historical tournament results from prior years only",
)
project.add_source(
    "momentum",
    temporal_safety="pre_tournament",
    outputs=["last_n_win_pct", "last_n_margin", "luck_index",
             "close_game_win_pct", "blowout_rate"],
    leakage_notes="Regular-season momentum indicators",
)

# -----------------------------------------------------------------------
# Step 2: Single-model baseline (logreg_seed)
# -----------------------------------------------------------------------

project.add_model(
    "logreg_seed",
    type="logistic_regression",
    features=["diff_seed_num"],
    params={"C": 1.0},
)

# Configure for women's backtest seasons
project.configure_backtest(
    seasons=[2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
)

# -----------------------------------------------------------------------
# Step 3: Multi-model
# -----------------------------------------------------------------------

project.add_model(
    "xgb_core",
    type="xgboost",
    features=[
        "diff_seed_num", "diff_composite_net", "diff_sr_srs",
        "diff_win_pct", "diff_scoring_margin",
    ],
    params={"max_depth": 2, "learning_rate": 0.01, "n_estimators": 300, "reg_lambda": 3.0},
)

project.add_model(
    "xgb_resume",
    type="xgboost",
    features=[
        "diff_seed_num", "diff_composite_net", "diff_sr_srs",
        "diff_sr_sos", "diff_win_pct", "diff_scoring_margin",
        "diff_last_n_win_pct", "diff_luck_index",
        "diff_close_game_win_pct", "diff_blowout_rate",
    ],
    params={"max_depth": 3, "learning_rate": 0.01, "n_estimators": 300, "reg_lambda": 5.0},
)

# -----------------------------------------------------------------------
# Step 4: Regressor model (margin prediction via CDF conversion)
# -----------------------------------------------------------------------

project.add_model(
    "xgb_spread",
    type="xgboost_regression",
    features=[
        "diff_seed_num", "diff_scoring_margin", "diff_win_pct",
        "diff_ppg", "diff_opp_ppg", "diff_close_game_win_pct",
        "diff_blowout_rate", "diff_last_n_win_pct",
        "diff_scoring_margin_std",
    ],
    mode="regressor",
    params={"max_depth": 1, "learning_rate": 0.01, "n_estimators": 200, "reg_lambda": 3.0},
)

# -----------------------------------------------------------------------
# Step 5: Ensemble + calibration
# -----------------------------------------------------------------------

project.configure_ensemble(
    method="stacked",
    C=1.0,
    calibration="spline",
    spline_n_bins=12,
    temperature=1.0,
    availability_adjustment=0.1,
)

# -----------------------------------------------------------------------
# Run backtest
# -----------------------------------------------------------------------

print("=" * 60)
print("Running backtest...")
print(f"Models: {list(project._models.keys())}")
print(f"Seasons: {project._backtest.seasons}")
print("=" * 60)

results = project.backtest()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
metrics = results["metrics"]
for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, dict):
        # Per-model metrics
        ensemble_val = metric_value.get("ensemble", "N/A")
        print(f"  {metric_name}: {ensemble_val}")
    else:
        print(f"  {metric_name}: {metric_value}")

print("\nPer-fold results:")
for fold_id, fold_metrics in sorted(results.get("per_fold", {}).items()):
    if isinstance(fold_metrics, dict):
        brier = fold_metrics.get("brier", {})
        if isinstance(brier, dict):
            brier = brier.get("ensemble", "N/A")
        print(f"  Season {fold_id}: Brier={brier}")

print(f"\nmm baseline: Brier=0.1361, Accuracy=81.59%, ECE=0.0339")

# Save config for reproducibility
project.save_to_yaml()
print(f"\nConfig saved to {PROJECT_DIR / 'config'}")
