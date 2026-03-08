---
layout: plain
title: For Agents
permalink: /for-agents
---

[← Back]({{ site.baseurl }}/)

## What This Is

Harness ML is an MCP server that gives you structured access to a full ML pipeline — data ingestion, feature engineering, model training, backtesting, calibration, ensembling, and experiment tracking. You never write training loops or ML code. You declare intent through tool calls, and Harness handles execution, validation, and logging.

7 tools. 80+ actions. 12 guardrails. 45 metrics across 6 task types. 8 model wrappers. 22 view transform steps. 4 calibration methods. 5 CV strategies.

Everything below is your complete protocol reference.

---

## Constraints

- Every experiment REQUIRES a hypothesis before creation
- Every experiment REQUIRES a conclusion after completion
- 3 non-overridable guardrails: feature leakage, temporal ordering, critical path — cannot be bypassed
- 9 overridable guardrails: sanity check, naming convention, do-not-retry, single variable, config protection, rate limit, experiment logged, feature staleness, feature diversity
- Workflow phases: EDA → Feature Discovery → Model Diversity → Feature Engineering → Tuning → Ensemble. Optional hard gates with `workflow.enforce_phases: true`
- All responses are markdown strings. Errors start with `**Error**:`
- If you pass an invalid enum value, fuzzy matching will suggest the closest valid option
- All tools accept optional `project_dir` (defaults to cwd)

---

## Bootstrap Sequence

```
configure(action="init", project_name="my_project", task="regression", target_column="price")
data(action="add", data_path="data/raw/train.csv")
data(action="inspect")
features(action="discover")
models(action="add", name="xgb_main", model_type="xgboost")
models(action="add", name="lgb_main", model_type="lightgbm")
configure(action="backtest", cv_strategy="kfold", fold_values=[1,2,3,4,5], metrics=["rmse","mae","r2"])
configure(action="ensemble", method="stacked", calibration="spline")
pipeline(action="run_backtest")
pipeline(action="diagnostics")
```

---

## Tool: configure

```
init             → project_name?, task?, target_column?, key_columns?, time_column?
update_data      → target_column?, key_columns?, time_column?
ensemble         → method?(stacked|average|weighted), temperature?, exclude_models?, calibration?(spline|isotonic|platt|none), pre_calibration?(JSON), spline_n_bins?
backtest         → cv_strategy?, fold_values?(list), metrics?(list), min_train_folds?, fold_column?
show             → detail?(summary|full), section?(models|ensemble|backtest)
check_guardrails →
exclude_columns  → add_columns?(list), remove_columns?(list)
set_denylist     → add_columns?, remove_columns?
add_target       → name, target_column, task?(binary), metrics?
list_targets     →
set_target       → name
studio           →
```

## Tool: data

```
add              → data_path, join_on?(list), prefix?, auto_clean?
validate         → data_path
fill_nulls       → column, strategy?(median|mean|mode|zero|value), value?
fill_nulls_batch → columns(JSON array of {column, strategy?, value?})
drop_duplicates  → columns?(subset list)
drop_rows        → column?, condition?("null" or pandas query)
rename           → mapping(JSON {old: new})
derive_column    → name, expression, group_by?, dtype?
inspect          → column?
profile          → category?
list_features    → prefix?
status           →
add_source       → name, data_path, format?(auto|csv|parquet|excel)
add_sources_batch → sources(JSON array)
add_view         → name, source, steps?(JSON array), description?
add_views_batch  → views(JSON array)
update_view      → name, source?, steps?, description?
remove_view      → name
list_views       →
preview_view     → name, n_rows?(5)
set_features_view → name
view_dag         →
list_sources     →
check_freshness  →
refresh          → name
refresh_all      →
validate_source  → name
sample           → fraction(0.0-1.0), stratify_column?, seed?(42)
restore          →
fetch_url        → data_path(URL), name?
upload_drive     → files(list), folder_id?, folder_name?, name?
upload_kaggle    → files, dataset_slug, title?
```

View step formats (for `steps` JSON array):

```
{op: "filter", expr: "col > 0"}
{op: "select", columns: ["a", "b"]}
{op: "derive", columns: {new_col: "a - b"}}
{op: "group_by", keys: ["k"], aggs: {col: "mean"}}
{op: "join", other: "view_name", on: {left_col: right_col}}
{op: "union", other: "view_name"}
{op: "rolling", keys: ["grp"], order_by: "col", window: 5, aggs: {new: "src:mean"}}
{op: "sort", by: ["col"], ascending: true}
{op: "head", keys: ["grp"], n: 5, order_by: "col"}
{op: "cast", columns: {col: "int"}}
{op: "distinct", columns: ["col"]}
{op: "rank", columns: {rank_col: "value_col"}, keys: ["grp"], ascending: false}
{op: "isin", column: "col", values: ["a", "b"]}
{op: "cond_agg", keys: ["k"], aggs: {new: "src:sum:condition_expr"}}
{op: "lag", keys: ["k"], order_by: "col", columns: {new: "src"}, periods: 1}
{op: "ewm", keys: ["k"], order_by: "col", columns: {new: "src"}, span: 10}
{op: "diff", keys: ["k"], order_by: "col", columns: {new: "src"}}
{op: "trend", keys: ["k"], order_by: "col", columns: {new: "src"}, window: 5}
{op: "encode", columns: ["cat_col"], method: "ordinal"}
{op: "bin", column: "col", bins: 5, labels: [...]}
{op: "datetime", column: "dt", extract: ["year", "month", "dow"]}
{op: "null_indicator", columns: ["col"]}
```

## Tool: features

```
add                  → name, formula?|source?|condition?|type?, pairwise_mode?(diff|ratio|both|none), category?, description?
add_batch            → features(JSON array of {name, formula?, type?, source?, condition?, pairwise_mode?, category?, description?})
test_transformations → features(list of column names), test_interactions?(true)
discover             → top_n?(20), method?(xgboost|mutual_info)
diversity            →
auto_search          → features?(list), search_types?(interactions|lags|rolling), top_n?
```

Feature types: entity (auto-generates pairwise), pairwise (instance formula), instance (context column), regime (boolean flag).

## Tool: models

```
add          → name, model_type?|preset?, features?, params?(JSON), active?, include_in_ensemble?, mode?(classifier|regressor), prediction_type?, cdf_scale?, zero_fill_features?, class_weight?
update       → name, features?, append_features?, remove_features?, params?, active?, include_in_ensemble?, mode?, prediction_type?, cdf_scale?, replace_params?(false)
remove       → name, purge?(false)
list         →
show         → name
presets      →
add_batch    → items(JSON array)
update_batch → items(JSON array)
remove_batch → items(JSON array of {name, purge?})
clone        → name(source), items(JSON {new_name, ...overrides})
```

Model types: xgboost, lightgbm, catboost, random_forest, logistic_regression, elastic_net, mlp, tabnet.

class_weight: "balanced" or JSON dict {"0": 1.0, "1": 2.5}. Regressors need cdf_scale for probability conversion. params default-merges unless replace_params=true.

## Tool: experiments

```
create        → description, hypothesis
write_overlay → experiment_id, overlay(JSON — supports dot-notation keys)
run           → experiment_id, primary_metric?, variant?
promote       → experiment_id, primary_metric?
quick_run     → description, overlay(JSON), hypothesis, primary_metric?
explore       → search_space(JSON {axes, budget, primary_metric}), detail?(summary|full)
promote_trial → experiment_id(exploration ID), trial?(best), primary_metric?, hypothesis?
compare       → experiment_ids(list of exactly 2)
journal       → last_n?(20)
log_result    → experiment_id, description?, hypothesis?, conclusion?, verdict?(keep|revert|partial)
```

Search space format:
```json
{"axes": {"learning_rate": {"type": "float", "low": 0.01, "high": 0.3}}, "budget": 10, "primary_metric": "rmse"}
```

## Tool: pipeline

```
run_backtest       → experiment_id?, variant?
predict            → fold_value, run_id?, variant?
diagnostics        → run_id?, detail?(summary|full)
list_runs          →
show_run           → run_id?, detail?
compare_runs       → run_ids(list of exactly 2)
compare_latest     →
compare_targets    →
explain            → name?(model), run_id?, top_n?(10)
inspect_predictions → run_id?, mode?(worst|best|uncertain), top_n?(10)
export_notebook    → destination(colab|kaggle|local), output_path?
progress           →
```

## Tool: competitions

```
create            → config(JSON), name?
list_formats      →
simulate          → name?, n_sims?(10000), seed?(42)
standings         → name?, top_n?(20)
round_probs       → name?, top_n?
generate_brackets → pool_size, name?, n_brackets?(3), n_sims?(10000), seed?
score_bracket     → picks(JSON), actuals(JSON), name?
adjust            → adjustments(JSON), name?
explain           → name?
profiles          → name?, top_n?
confidence        → name?
export            → output_dir, name?, format_type?(json|markdown|csv)
list_strategies   →
```

---

## Common Errors

```
No config/ directory       → Run configure(action="init") first
Empty feature store        → Run data(action="add", data_path="...") first
Unknown model type         → Lists valid types + fuzzy suggestion
Invalid action             → Lists valid actions + "Did you mean?"
Missing required param     → "**Error**: {param} is required"
Invalid JSON               → "Invalid JSON input: {details}"
Guardrail violation        → {blocked: bool, rule: str, message: str, overridable: bool}
Workflow gate              → "**Blocked**: Complete {phase} before {action}"
```

---

## Skills

Skills are structured prompts loaded into your context to enforce methodology.

- **harness-run-experiment**: Enforces hypothesis → action → diagnosis → conclusion. You must exhaust a strategy (3-5 configs) before abandoning.
- **harness-explore-space**: Phased workflow with gates. You cannot tune until you have explored model diversity.
- **harness-domain-research**: Domain research via web search before experimentation. Best ML gains come from domain-driven hypotheses.

---

## Reference

Metrics (45 across 6 task types): brier, accuracy, log_loss, ece, auroc, f1, precision, recall | macro/micro/weighted variants | rmse, mae, r2, mape | ndcg, mrr, map | concordance_index, brier_survival | crps, calibration, sharpness

Calibration: spline (PCHIP), isotonic, platt, beta

CV strategies: symmetric_loso, expanding_window, sliding_window, kfold, stratified

Models: xgboost, lightgbm, catboost, random_forest, logistic_regression, elastic_net, mlp, tabnet

View steps (22): filter, select, derive, group_by, join, union, unpivot, sort, head, rolling, cast, distinct, rank, isin, cond_agg, lag, ewm, diff, trend, encode, bin, datetime, null_indicator
