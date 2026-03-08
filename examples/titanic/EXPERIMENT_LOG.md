# Experiment Log

| ID | Hypothesis | Changes | Verdict | Notes |
|----|-----------|---------|---------|-------|
| exp-001 | 2026-03-07 | Establish a baseline with the simplest usable feature set — class, age, fare, and sex. These are the most obvious survival predictors from the Titanic literature. Expect accuracy around 0.78-0.80 based on published benchmarks for minimal feature sets. |  | 0.8215 (+0.0000) | - | - | 0.4205 (+0.0000) | neutral |  |  |
| exp-002 | 2026-03-07 | SibSp (siblings/spouses) and Parch (parents/children) capture family size on board. Having family might affect survival — both positively (motivation to survive together) and negatively (large families harder to evacuate). Adding them gives the model access to all available numeric features. Expect modest improvement since their raw correlations are weak (SibSp -0.04, Parch +0.08). | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch]}}}
 | 0.8249 (+0.0034) | - | - | 0.4194 (-0.0010) | improved |  |  |
| exp-003 | 2026-03-07 | Family size has a known U-shaped relationship with survival: solo travelers and very large families died more often, while families of 2-4 had better survival rates. The linear correlation is near zero (0.02) but XGBoost can capture the non-linear U-shape through splits. Replacing separate SibSp/Parch with a single family_size gives the model a more direct signal. | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch, family_size]}}}
 | 0.8339 (+0.0090) | - | - | 0.4158 (-0.0036) | improved |  |  |
| exp-004 | 2026-03-07 | Children under 12 had evacuation priority per "women and children first" protocol. While XGBoost can learn an age threshold, an explicit binary flag makes it easier to learn the sharp survival discontinuity at the child boundary. Expect Brier improvement, especially for young passengers in the tails. | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch, family_size,
        is_child]}}}
 | 0.8339 (+0.0000) | - | - | 0.4158 (+0.0000) | neutral |  |  |
| exp-005 | 2026-03-07 | Solo travelers (60% of passengers) had no family to coordinate with during evacuation. This binary flag makes the alone/not-alone distinction explicit. It's redundant with family_size for XGBoost (which can split at 1), but it may help by giving a direct feature the model can combine with other signals like class and gender. | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch, family_size,
        is_alone]}}}
 | 0.8339 (+0.0000) | - | - | 0.4158 (+0.0000) | neutral |  |  |
| exp-006 | 2026-03-07 | Raw Fare often covers a whole ticket group, inflating the apparent wealth of passengers traveling with family. Fare per person is a cleaner wealth signal. Correlation with survival is +0.22 (vs raw Fare +0.26), but it should provide a less noisy signal that's more independent from family_size. XGBoost may learn the Fare/family_size ratio through interaction splits, but providing it directly should help at depth 3. | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch, family_size,
        fare_per_person]}}}
 | 0.8395 (+0.0056) | - | - | 0.4104 (-0.0054) | improved |  |  |
| exp-007 | 2026-03-07 | Age*Pclass captures the compounding disadvantage of being old AND in a lower class. Correlation with survival is -0.32 (strong). At depth 3, XGBoost might struggle to capture this multiplicative interaction — it would need one split for Age and one for Pclass, leaving only one split for other features. Providing the product directly frees up tree depth for other patterns. | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch, family_size,
        fare_per_person, age_class_interact]}}}
 | 0.8305 (-0.0090) | - | - | 0.4128 (+0.0024) | regressed |  |  |
| exp-008 | 2026-03-07 | Only 23% of passengers have cabin info recorded. Cabin recording correlates with class and wealth — upper-class passengers were more likely to have their cabin documented. This provides an independent wealth/status signal beyond Pclass and Fare. Since 77% are missing, the missingness itself is informative. | {models: {xgb_base: {features: [Pclass, Age, Fare, is_female, SibSp, Parch, family_size,
        fare_per_person, has_cabin]}}}
 | 0.8260 (-0.0135) | - | - | 0.4101 (-0.0003) | regressed |  |  |
| exp-009 | 2026-03-07 | From the wine quality project, we learned that XGBoost + RandomForest produces genuine algorithmic diversity (boosting vs bagging). The stacked meta-learner can weight them optimally. RF uses the same 8 features. Expect Brier improvement from the ensemble averaging out individual model errors. | {models: {rf_base: {active: true, features: [Pclass, Age, Fare, is_female, SibSp,
        Parch, family_size, fare_per_person], include_in_ensemble: true, params: {
        max_depth: 8, min_samples_leaf: 5, n_estimators: 200}, type: random_forest}}}
 | 0.8361 (+0.0000) | - | - | 0.4076 (-0.0156) | regressed |  |  |
| exp-010 | 2026-03-07 | From the wine quality project, we learned that giving RF engineered ratio features while XGBoost handles raw features produces more diverse predictions. RF can't learn ratios natively, so fare_per_person and age_class_interact should help RF more than XGBoost. Giving RF features that XGBoost doesn't use (has_cabin, is_alone, age_class_interact) creates prediction diversity for the meta-learner. | {models: {rf_base: {active: true, features: [Pclass, Age, Fare, is_female, family_size,
        fare_per_person, age_class_interact, has_cabin, is_child], include_in_ensemble: true,
      params: {max_depth: 8, min_samples_leaf: 5, n_estimators: 200}, type: random_forest}}}
 | 0.8361 (+0.0000) | - | - | 0.4067 (-0.0165) | improved |  |  |
| exp-011 | 2026-03-07 | The title "Master" identifies young boys specifically — the one male group that had evacuation priority. is_female captures adult women, but boys under ~14 were also prioritized. This is more precise than is_child because it only flags male children. Adding to RF only since threshold features don't help XGBoost. Expect improved prediction for the ~40 "Master" passengers. | {models: {rf_base: {features: [Pclass, Age, Fare, is_female, family_size, fare_per_person,
        age_class_interact, has_cabin, is_child, is_master]}}}
 | 0.8328 (-0.0011) | - | - | 0.4189 (+0.0125) | improved |  |  |
| exp-012 | 2026-03-07 | With 8 features and 891 rows, depth 3 may underfit some interaction patterns. Depth 4 allows one more level of feature interaction per tree. Wine quality saw a big jump from depth 5→6. However, Titanic has fewer rows (891 vs 1599), so overfitting risk is higher. Expect modest Brier improvement if the depth 3 model was indeed underfitting. | {models: {xgb_base: {params: {learning_rate: 0.1, max_depth: 4, n_estimators: 100}}}}
 | 0.8361 (+0.0034) | - | - | 0.4114 (+0.0078) | regressed |  |  |
| exp-013 | 2026-03-07 | Instead of more depth, add stochastic regularization. subsample=0.8 and colsample_bytree=0.8 force each tree to see a different subset of rows and features, reducing overfitting. This is standard XGBoost practice and helped in the wine quality project. Should stabilize the depth-3 model's fold variance. | {models: {xgb_base: {params: {colsample_bytree: 0.8, learning_rate: 0.1, max_depth: 3,
        n_estimators: 100, subsample: 0.8}}}}
 | 0.8283 (-0.0067) | - | - | 0.4099 (+0.0051) | regressed |  |  |
| exp-014 | 2026-03-07 | Lower learning rate with more trees produces a smoother, more stable model. Each tree contributes less, reducing the impact of noisy splits. The total "learning budget" (n_estimators * lr) stays roughly the same (100*0.1 vs 200*0.05), but the smaller step size should produce better calibration. | {models: {xgb_base: {params: {learning_rate: 0.05, max_depth: 3, n_estimators: 200}}}}
 | 0.8339 (+0.0000) | - | - | 0.4193 (+0.0155) | regressed |  |  |
