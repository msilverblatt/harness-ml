# Experiment Log

| ID | Hypothesis | Changes | Verdict | Notes |
|----|-----------|---------|---------|-------|
| exp-001 | 2026-03-07 | Adding class, family, and embarkation features will significantly improve predictions since pclass has -0.34 correlation with survival | {models: {lgb_main: {features: [age, fare, sibsp, parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q]}, lr_baseline: {
      features: [age, fare, sibsp, parch, sex_male, pclass, family_size, is_alone,
        fare_per_person, is_child, embarked_c, embarked_q]}, xgb_main: {features: [
        age, fare, sibsp, parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q]}}}
 | 0.7643 (-0.0090) | - | 0.1885 (-0.0494) | 0.6802 (-0.1069) | improved |  |
| exp-002 | 2026-03-07 | Interaction features capture non-linear relationships (male+low class = worst survival) that tree models may not fully exploit with raw features | {models: {lgb_main: {append_features: [pclass_x_sex, pclass_div_famsize, famsize_x_sex]},
    lr_baseline: {append_features: [pclass_x_sex, pclass_div_famsize, famsize_x_sex]},
    xgb_main: {append_features: [pclass_x_sex, pclass_div_famsize, famsize_x_sex]}}}
 | 0.7643 (+0.0000) | - | 0.1885 (+0.0000) | 0.6802 (+0.0000) | neutral |  |
| exp-003 | 2026-03-07 | More diverse model types should improve ensemble performance through complementary error patterns | {models: {catboost_main: {active: true, features: [age, fare, sibsp, parch, sex_male,
        pclass, family_size, is_alone, fare_per_person, is_child, embarked_c, embarked_q],
      include_in_ensemble: true, mode: classifier, params: {depth: 4, iterations: 500,
        learning_rate: 0.03}, type: catboost}, rf_main: {active: true, features: [
        age, fare, sibsp, parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q], include_in_ensemble: true, mode: classifier,
      params: {max_depth: 6, min_samples_leaf: 5, n_estimators: 500}, type: random_forest}}}
 | 0.7654 (+0.0011) | - | 0.2037 (+0.0152) | 0.6998 (+0.0195) | regressed |  |
| exp-004 | 2026-03-07 | CatBoost alone adds diversity without noise from RF; spline calibration should improve ECE | {ensemble: {calibration: spline}, models: {catboost_main: {active: true, features: [
        age, fare, sibsp, parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q], include_in_ensemble: true, mode: classifier,
      params: {depth: 4, iterations: 500, learning_rate: 0.03}, type: catboost}}}
 | 0.7666 (+0.0022) | - | 0.2034 (+0.0149) | 0.6882 (+0.0080) | regressed |  |
| exp-005 | 2026-03-07 | XGBoost is redundant with LightGBM and adds noise; removing it should improve or maintain brier | {models: {xgb_main: {include_in_ensemble: false}}}
 | 0.7542 (-0.0101) | - | 0.1814 (-0.0072) | 0.6871 (+0.0069) | regressed |  |
| exp-006 | 2026-03-07 | Spline calibration should improve ECE which is currently 0.19 | {ensemble: {calibration: spline}}
 | 0.7643 (+0.0000) | - | 0.1885 (+0.0000) | 0.6802 (+0.0000) | neutral |  |
| exp-007 | 2026-03-07 | Different feature views per model creates genuine diversity rather than all models seeing the same data | {models: {lr_baseline: {features: [sex_male, pclass, fare_per_person, embarked_c,
        embarked_q, is_child, is_alone]}, xgb_main: {features: [age, sex_male, pclass,
        is_child, is_alone, family_size, embarked_c]}}}
 | 0.7789 (+0.0146) | - | 0.1655 (-0.0230) | 0.6792 (-0.0010) | regressed |  |
| exp-008 | 2026-03-07 | male_adult (-0.56), name_len (+0.33), cabin_deck_abc (+0.21) are strong new signals that should improve brier | {models: {lgb_main: {append_features: [ticket_freq, cabin_deck_abc, title_rare, large_family,
        female_low_class, male_adult, name_len]}, lr_baseline: {append_features: [
        ticket_freq, cabin_deck_abc, title_rare, large_family, female_low_class, male_adult,
        name_len]}, xgb_main: {append_features: [ticket_freq, cabin_deck_abc, title_rare,
        large_family, female_low_class, male_adult, name_len]}}}
 | 0.7643 (+0.0000) | - | 0.1514 (+0.0000) | 0.6781 (+0.0000) | neutral |  |
| exp-009 | 2026-03-07 | male_adult (-0.56), name_len (+0.33), cabin_deck_abc (+0.21) are strong new signals | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, ticket_freq, cabin_deck_abc,
        title_rare, large_family, female_low_class, male_adult, name_len]}, lr_baseline: {
      features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone,
        fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr,
        title_mrs, title_miss, title_master, ticket_freq, cabin_deck_abc, title_rare,
        large_family, female_low_class, male_adult, name_len]}, xgb_main: {features: [
        Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr, title_mrs,
        title_miss, title_master, ticket_freq, cabin_deck_abc, title_rare, large_family,
        female_low_class, male_adult, name_len]}}}
 | 0.7733 (+0.0090) | - | 0.2036 (+0.0523) | 0.7266 (+0.0485) | regressed |  |
| exp-010 | 2026-03-07 | Just two high-signal features to avoid overfitting: cabin_deck_abc (+0.21) and name_len (+0.33) | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, cabin_deck_abc, name_len]},
    lr_baseline: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, cabin_deck_abc, name_len]},
    xgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, cabin_deck_abc, name_len]}}}
 | 0.7733 (+0.0090) | - | 0.1856 (+0.0342) | 0.6940 (+0.0159) | regressed |  |
| exp-011 | 2026-03-07 | male_adult (-0.56) is the single strongest feature; large_family (-0.13) captures evacuation difficulty | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, male_adult, large_family]},
    lr_baseline: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, male_adult, large_family]},
    xgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, male_adult, large_family]}}}
 | 0.7699 (+0.0056) | - | 0.1873 (+0.0360) | 0.6776 (-0.0005) | regressed |  |
| exp-012 | 2026-03-07 | Reducing multicollinearity with 891 rows and 3 folds should help generalization and improve brier | {models: {lgb_main: {features: [Age, fare_per_person, pclass, family_size, is_child,
        embarked_c, has_cabin, age_known, title_mr, title_mrs, title_miss, title_master]},
    lr_baseline: {features: [Age, fare_per_person, pclass, family_size, is_child,
        embarked_c, has_cabin, age_known, title_mr, title_mrs, title_miss, title_master]},
    xgb_main: {features: [Age, fare_per_person, pclass, family_size, is_child, embarked_c,
        has_cabin, age_known, title_mr, title_mrs, title_miss, title_master]}}}
 | 0.7239 (-0.0404) | - | 0.1315 (-0.0199) | 0.6516 (-0.0266) | regressed |  |
| exp-013 | 2026-03-07 | log_fare handles fare skew (max 512 vs median 14); ticket_freq captures actual travel group vs family declarations | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, log_fare, ticket_freq]}, lr_baseline: {
      features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone,
        fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr,
        title_mrs, title_miss, title_master, log_fare, ticket_freq]}, xgb_main: {
      features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone,
        fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr,
        title_mrs, title_miss, title_master, log_fare, ticket_freq]}}}
 | 0.7508 (-0.0135) | - | 0.1544 (+0.0030) | 0.6474 (-0.0307) | regressed |  |
| exp-014 | 2026-03-07 | age_pclass captures that old 3rd class passengers had worst survival; young 1st class had best | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, age_pclass]}, lr_baseline: {
      features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone,
        fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr,
        title_mrs, title_miss, title_master, age_pclass]}, xgb_main: {features: [
        Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr, title_mrs,
        title_miss, title_master, age_pclass]}}}
 | 0.7744 (+0.0101) | - | 0.1830 (+0.0316) | 0.6689 (-0.0092) | improved |  |
| exp-015 | 2026-03-07 | Disjoint feature sets force genuine model diversity. XGB=gender/age policy, LGB=wealth/status, LR=family/social. Meta-learner combines orthogonal signals. | {models: {lgb_main: {features: [pclass, Fare, fare_per_person, has_cabin, embarked_c,
        embarked_q, age_pclass]}, lr_baseline: {features: [family_size, is_alone,
        SibSp, Parch, ticket_freq, sex_male, pclass]}, xgb_main: {features: [sex_male,
        Age, is_child, age_known, title_mr, title_mrs, title_miss, title_master]}}}
 | 0.7093 (-0.0651) | - | 0.1828 (-0.0001) | 0.7030 (+0.0341) | regressed |  |
| exp-016 | 2026-03-07 | Shared anchors prevent collapse while focused features create genuine diversity for the meta-learner | {models: {lgb_main: {features: [sex_male, pclass, Fare, fare_per_person, has_cabin,
        embarked_c, embarked_q, age_pclass]}, lr_baseline: {features: [sex_male, pclass,
        family_size, is_alone, SibSp, Parch, ticket_freq, is_child, age_pclass]},
    xgb_main: {features: [sex_male, pclass, Age, is_child, age_known, title_mr, title_mrs,
        title_miss, title_master, age_pclass]}}}
 | 0.7834 (+0.0090) | - | 0.2176 (+0.0346) | 0.6789 (+0.0100) | regressed |  |
| exp-017 | 2026-03-07 | Features that regressed under 3-fold may improve now that models have 713 training rows per fold | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, age_pclass, cabin_deck_abc,
        name_len, male_adult, large_family, ticket_freq]}, lr_baseline: {features: [
        Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr, title_mrs,
        title_miss, title_master, age_pclass, cabin_deck_abc, name_len, male_adult,
        large_family, ticket_freq]}, xgb_main: {features: [Age, Fare, SibSp, Parch,
        sex_male, pclass, family_size, is_alone, fare_per_person, is_child, embarked_c,
        embarked_q, has_cabin, age_known, title_mr, title_mrs, title_miss, title_master,
        age_pclass, cabin_deck_abc, name_len, male_adult, large_family, ticket_freq]}}}
 | 0.8305 (+0.0022) | - | 0.0243 (+0.0036) | 0.3990 (-0.0049) | improved |  |
| exp-018 | 2026-03-07 | With 5 folds the models have enough data to learn focused hypotheses; shared anchors prevent collapse | {models: {lgb_main: {features: [sex_male, pclass, Fare, fare_per_person, has_cabin,
        cabin_deck_abc, embarked_c, embarked_q, age_pclass, name_len]}, lr_baseline: {
      features: [sex_male, pclass, family_size, is_alone, SibSp, Parch, ticket_freq,
        large_family, is_child, age_pclass]}, xgb_main: {features: [sex_male, pclass,
        Age, is_child, age_known, title_mr, title_mrs, title_miss, title_master, age_pclass,
        male_adult]}}}
 | 0.8215 (-0.0090) | - | 0.0229 (-0.0014) | 0.4020 (+0.0031) | regressed |  |
| exp-019 | 2026-03-07 | woman_upper_class (+0.56) directly encodes highest survival group; small_family (+0.28) captures optimal family size for evacuation | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, age_pclass, cabin_deck_abc,
        name_len, male_adult, large_family, ticket_freq, cabin_count, small_family,
        first_class, woman_upper_class, fare_bin, age_bin]}, lr_baseline: {features: [
        Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone, fare_per_person,
        is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr, title_mrs,
        title_miss, title_master, age_pclass, cabin_deck_abc, name_len, male_adult,
        large_family, ticket_freq, cabin_count, small_family, first_class, woman_upper_class,
        fare_bin, age_bin]}, xgb_main: {features: [Age, Fare, SibSp, Parch, sex_male,
        pclass, family_size, is_alone, fare_per_person, is_child, embarked_c, embarked_q,
        has_cabin, age_known, title_mr, title_mrs, title_miss, title_master, age_pclass,
        cabin_deck_abc, name_len, male_adult, large_family, ticket_freq, cabin_count,
        small_family, first_class, woman_upper_class, fare_bin, age_bin]}}}
 | 0.8350 (+0.0045) | - | 0.0261 (+0.0019) | 0.3917 (-0.0073) | improved |  |
| exp-020 | 2026-03-07 | third_class_male (-0.41) encodes worst survival group; cabin_deck_ord (+0.29) adds granularity over binary has_cabin; is_mother (+0.17) captures lifeboat priority | {models: {lgb_main: {features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size,
        is_alone, fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known,
        title_mr, title_mrs, title_miss, title_master, age_pclass, cabin_deck_abc,
        name_len, male_adult, large_family, ticket_freq, cabin_count, small_family,
        first_class, woman_upper_class, fare_bin, age_bin, is_mother, fare_per_ticket,
        third_class_male, cabin_deck_ord]}, lr_baseline: {features: [Age, Fare, SibSp,
        Parch, sex_male, pclass, family_size, is_alone, fare_per_person, is_child,
        embarked_c, embarked_q, has_cabin, age_known, title_mr, title_mrs, title_miss,
        title_master, age_pclass, cabin_deck_abc, name_len, male_adult, large_family,
        ticket_freq, cabin_count, small_family, first_class, woman_upper_class, fare_bin,
        age_bin, is_mother, fare_per_ticket, third_class_male, cabin_deck_ord]}, xgb_main: {
      features: [Age, Fare, SibSp, Parch, sex_male, pclass, family_size, is_alone,
        fare_per_person, is_child, embarked_c, embarked_q, has_cabin, age_known, title_mr,
        title_mrs, title_miss, title_master, age_pclass, cabin_deck_abc, name_len,
        male_adult, large_family, ticket_freq, cabin_count, small_family, first_class,
        woman_upper_class, fare_bin, age_bin, is_mother, fare_per_ticket, third_class_male,
        cabin_deck_ord]}}}
 | 0.8305 (-0.0045) | - | 0.0205 (-0.0056) | 0.3903 (-0.0014) | improved |  |
| exp-021 | 2026-03-07 | Applying overlay from expl-002 best trial. | {models: {lgb_main: {params: {colsample_bytree: 0.888362206456006, learning_rate: 0.005349319115885771,
        max_depth: 5, min_child_samples: 28, n_estimators: 1035, num_leaves: 51, subsample: 0.7621685341038414}}}}
 | 0.8373 (+0.0067) | - | 0.0162 (-0.0043) | 0.3875 (-0.0028) | improved |  |
| exp-022 | 2026-03-07 | Applying overlay from expl-003 best trial. | {models: {xgb_main: {params: {colsample_bytree: 0.665080544056281, learning_rate: 0.09732311588377787,
        max_depth: 3, min_child_weight: 12, n_estimators: 702, reg_alpha: 0.05350822875106499,
        reg_lambda: 0.38779589884435683, subsample: 0.5032837044352211}}}}
 | 0.8350 (+0.0045) | - | 0.0141 (-0.0065) | 0.3829 (-0.0074) | improved |  |
| exp-023 | 2026-03-07 | Applying overlay from expl-004 best trial. | {models: {lr_baseline: {params: {C: 8.431013932082463, penalty: l2, solver: liblinear}}}}
 | 0.8361 (-0.0011) | - | 0.0161 (-0.0001) | 0.3884 (+0.0009) | regressed |  |
