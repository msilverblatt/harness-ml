/**
 * Glossary of ML / statistics terms shown on hover throughout the Studio UI.
 *
 * Keys are the canonical lowercase identifiers used in configs and API
 * responses.  Values are short, plain-English explanations.
 */

export const GLOSSARY: Record<string, string> = {
    // ── Metrics: binary classification ──────────────────────────
    accuracy:
        'Fraction of predictions that are correct. Simple but misleading on imbalanced datasets — a model predicting the majority class always can score high.',
    brier:
        'Brier score — mean squared error between predicted probabilities and outcomes (0 or 1). Lower is better; penalizes confident wrong predictions heavily. Range: 0 (perfect) to 1.',
    ece:
        'Expected Calibration Error — measures how well predicted probabilities match observed frequencies. Lower is better. A model predicting 70% should win ~70% of the time.',
    log_loss:
        'Logarithmic loss — measures the quality of probabilistic predictions. Heavily penalizes confident wrong predictions. Lower is better; 0 is perfect.',
    auc:
        'Area Under the ROC Curve — measures how well the model separates classes across all thresholds. 1.0 is perfect separation, 0.5 is random guessing.',
    auc_roc:
        'Area under receiver operating characteristic curve. Higher is better.',
    auc_pr:
        'Area under precision-recall curve. Better for imbalanced datasets. Higher is better.',
    f1:
        'Harmonic mean of precision and recall. Higher is better.',
    precision:
        'Fraction of positive predictions that are correct. Higher is better.',
    recall:
        'Fraction of actual positives correctly identified. Higher is better.',
    brier_score:
        'Mean squared error of probability estimates. Lower is better.',
    f1_macro:
        'F1 averaged across classes (unweighted). Higher is better.',
    f1_weighted:
        'F1 averaged across classes (weighted by support). Higher is better.',
    positive_rate:
        'Fraction of samples in the positive class.',
    class_balance:
        'Ratio of minority to majority class size.',

    // ── Metrics: regression ─────────────────────────────────────
    rmse:
        'Root Mean Squared Error — square root of the average squared difference between predictions and actuals. Same units as the target. Penalizes large errors more than MAE.',
    mae:
        'Mean Absolute Error — average of the absolute differences between predictions and actuals. Same units as the target. More robust to outliers than RMSE.',
    r_squared:
        'R² (coefficient of determination) — proportion of variance in the target explained by the model. 1.0 is perfect; 0 means no better than predicting the mean.',
    r2:
        'R² (coefficient of determination) — proportion of variance in the target explained by the model. 1.0 is perfect; 0 means no better than predicting the mean.',
    mape:
        'Mean Absolute Percentage Error — average percentage difference between predictions and actuals. Scale-independent but undefined when actuals are zero.',
    mse:
        'Mean Squared Error — average of squared differences between predictions and actuals. Penalizes large errors quadratically. RMSE is its square root.',

    // ── Metrics: ranking / other ────────────────────────────────
    ndcg:
        'Normalized Discounted Cumulative Gain — measures ranking quality, weighting items higher in the list more. 1.0 is a perfect ranking.',
    map:
        'Mean Average Precision — average of precision scores at each relevant result. Commonly used for ranking and information retrieval.',

    // ── Model types ─────────────────────────────────────────────
    xgboost:
        'Extreme Gradient Boosting — fast, regularized gradient boosting. Builds trees sequentially, each correcting the previous errors. Strong baseline for tabular data.',
    lightgbm:
        'Light Gradient Boosting Machine — gradient boosting that grows trees leaf-wise instead of level-wise. Faster training and often lower memory usage than XGBoost.',
    catboost:
        'CatBoost — gradient boosting with native categorical feature handling and ordered boosting to reduce overfitting. Requires less hyperparameter tuning.',
    random_forest:
        'Random Forest — ensemble of decision trees trained on random subsets of data and features. Predictions are averaged (regression) or voted (classification). Resistant to overfitting.',
    logistic:
        'Logistic Regression — linear model that outputs probabilities via the sigmoid function. Fast, interpretable, and a strong baseline for binary classification.',
    elastic_net:
        'Elastic Net — linear regression combining L1 (Lasso) and L2 (Ridge) regularization. Handles correlated features better than Lasso alone.',
    mlp:
        'Multi-Layer Perceptron — feedforward neural network with one or more hidden layers. Can learn nonlinear relationships but needs more tuning than tree-based models.',
    tabnet:
        'TabNet — attention-based deep learning model designed for tabular data. Learns which features to focus on at each step. Competitive with gradient boosting on some tasks.',

    // ── Hyperparameters ─────────────────────────────────────────
    max_depth:
        'Maximum depth of each decision tree. Deeper trees capture more complex patterns but are more likely to overfit. Common range: 3–10.',
    learning_rate:
        'Step size for each boosting iteration. Smaller values need more trees but generalize better. Common range: 0.01–0.3.',
    n_estimators:
        'Number of boosting rounds (trees) to build. More trees reduce bias but increase training time and risk overfitting without early stopping.',
    num_leaves:
        'Maximum number of leaves per tree (LightGBM). Controls model complexity — more leaves means more capacity but higher overfitting risk.',
    C:
        'Inverse regularization strength (Logistic Regression). Smaller values = stronger regularization. Controls the bias-variance tradeoff.',
    max_iter:
        'Maximum number of optimization iterations. Higher values give the solver more time to converge but increase training time.',
    dropout:
        'Fraction of neurons randomly zeroed during training (neural networks). Prevents co-adaptation of neurons and reduces overfitting. Common range: 0.1–0.5.',
    hidden_layers:
        'Number and size of hidden layers in a neural network. More/larger layers increase capacity but need more data and training time.',
    normalize:
        'Whether to standardize features to zero mean and unit variance before training. Important for distance-based and gradient-based models.',
    batch_norm:
        'Batch Normalization — normalizes activations within each mini-batch during training. Stabilizes learning and allows higher learning rates.',
    early_stopping:
        'Stop training when validation performance stops improving. Prevents overfitting by finding the optimal number of iterations automatically.',
    weight_decay:
        'L2 regularization penalty on model weights (neural networks). Discourages large weights and reduces overfitting.',
    seed_stride:
        'Offset added to the random seed for each fold, ensuring different random initializations across cross-validation folds.',
    val_fraction:
        'Fraction of training data held out for validation during model training. Used with early stopping to monitor generalization.',
    eval_set:
        'Whether to use a validation set for early stopping during training. Enables the model to stop before overfitting.',

    // ── Ensemble concepts ───────────────────────────────────────
    stacking:
        'Stacking — ensemble method where base model predictions become features for a meta-learner. Learns optimal model combination weights from cross-validated predictions.',
    average:
        'Simple average ensemble — takes the unweighted mean of all base model predictions. No training required; works well when models are diverse.',
    weighted:
        'Weighted average ensemble — combines predictions using learned weights. Gives more influence to better-performing models.',
    meta_learner:
        'The second-level model in a stacking ensemble that learns how to optimally combine base model predictions. Typically logistic regression or ridge regression.',
    meta_coefficients:
        'Weights assigned to each base model by the meta-learner. Higher absolute value means more influence in the ensemble prediction.',
    cv_strategy:
        'Cross-validation strategy — how data is split into training and validation folds. Common strategies: k-fold, leave-one-season-out (LOSO), stratified.',
    fold_column:
        'Column in the dataset used to define cross-validation fold assignments. Ensures temporal or group-based splits rather than random.',
    temperature:
        'Scaling factor applied to logits before the sigmoid/softmax. Values > 1 flatten probabilities (less confident); < 1 sharpen them.',
    clip_floor:
        'Minimum probability value — predictions below this are clipped up. Prevents extreme probabilities near 0 that can blow up log-loss.',

    // ── Calibration ─────────────────────────────────────────────
    calibration:
        'Post-hoc adjustment of predicted probabilities so they match observed frequencies. A well-calibrated model predicting 80% should be correct ~80% of the time.',
    spline:
        'Spline calibration (PCHIP) — fits a smooth monotonic curve through calibration points. Flexible and preserves probability ordering.',
    isotonic:
        'Isotonic regression calibration — fits a non-decreasing step function. Very flexible but can overfit with few calibration samples.',
    platt:
        'Platt scaling — fits a logistic regression on the model outputs. Simple two-parameter calibration; works well for sigmoid-shaped distortions.',
    beta:
        'Beta calibration — fits a Beta distribution to model outputs. More flexible than Platt scaling; handles asymmetric distortions.',
    pre_calibration:
        'Per-model calibration applied before the meta-learner combines predictions. Ensures each base model feeds well-calibrated probabilities into the ensemble.',

    // ── Correlation / analysis ───────────────────────────────────
    correlation:
        'Prediction correlation — Pearson correlation between two models\' predicted probabilities. High correlation (>0.8) means models are redundant; diverse models improve ensembles.',
    eta_squared:
        'Eta-squared — proportion of variance in predictions explained by a categorical variable. Used in multiclass analysis to measure feature/class associations.',

    // ── Residual analysis ───────────────────────────────────────
    residual:
        'Difference between actual and predicted values (actual − predicted). Analyzing residual patterns reveals systematic model errors.',
    mean_residual:
        'Average residual across all predictions. Should be near zero; a non-zero mean indicates systematic over- or under-prediction.',
    std_residual:
        'Standard deviation of residuals. Measures prediction consistency — lower means more reliable predictions.',
    median_residual:
        'Middle value of residuals when sorted. More robust to outliers than the mean; should also be near zero.',

    // ── Task types ──────────────────────────────────────────────
    binary:
        'Binary classification — predicting one of two outcomes (yes/no, win/lose). Models output a probability between 0 and 1.',
    multiclass:
        'Multiclass classification — predicting one of three or more categories. Models output a probability distribution across all classes.',
    regression:
        'Regression — predicting a continuous numeric value (price, score, temperature). Models output a real number.',
    ranking:
        'Ranking — ordering items by relevance or preference. Models score items so that higher scores correspond to more relevant results.',
    survival:
        'Survival analysis — predicting time to an event (failure, churn). Models estimate survival probabilities over time.',
    probabilistic:
        'Probabilistic forecasting — predicting a full probability distribution over outcomes rather than a single point estimate.',

    // ── Experiment verdicts ──────────────────────────────────────
    keep:
        'Experiment verdict: the change improved performance and should be kept in the production configuration.',
    improved:
        'Experiment verdict: the change improved the primary metric. Alias for "keep".',
    partial:
        'Experiment verdict: mixed results — some metrics improved, others did not. Needs further investigation.',
    neutral:
        'Experiment verdict: no meaningful change in performance. The hypothesis was not supported.',
    revert:
        'Experiment verdict: performance degraded. The change should be rolled back.',
    regressed:
        'Experiment verdict: the primary metric worsened. Alias for "revert".',
};

/**
 * Format a metric key for display — handles the R² superscript case
 * and general prettifying.
 */
export function formatMetricLabel(key: string): string | { text: string; jsx: true } {
    const lower = key.toLowerCase();
    if (lower === 'r_squared' || lower === 'r2') {
        return { text: 'R²', jsx: true };
    }
    return key;
}

/**
 * Look up a glossary definition for a term. Tries the raw key first,
 * then lowercased, then with underscores and hyphens stripped.
 */
export function lookupGlossary(term: string): string | undefined {
    if (GLOSSARY[term]) return GLOSSARY[term];
    const lower = term.toLowerCase();
    if (GLOSSARY[lower]) return GLOSSARY[lower];
    const normalized = lower.replace(/[-\s]/g, '_');
    if (GLOSSARY[normalized]) return GLOSSARY[normalized];
    return undefined;
}
