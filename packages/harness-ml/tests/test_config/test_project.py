import pytest
from harness.ml.config.project import CVConfig, ProjectConfig


class TestCVConfig:
    def test_defaults(self):
        cfg = CVConfig()
        assert cfg.strategy == "kfold"
        assert cfg.n_folds == 5
        assert cfg.fold_column is None
        assert cfg.fold_values is None
        assert cfg.min_train_folds == 2

    def test_custom_values(self):
        cfg = CVConfig(
            strategy="stratified_kfold",
            n_folds=10,
            fold_column="week",
            fold_values=[1, 2, 3],
            min_train_folds=4,
        )
        assert cfg.strategy == "stratified_kfold"
        assert cfg.n_folds == 10
        assert cfg.fold_column == "week"
        assert cfg.fold_values == [1, 2, 3]
        assert cfg.min_train_folds == 4

    def test_all_fields_accessible(self):
        cfg = CVConfig()
        _ = cfg.strategy
        _ = cfg.n_folds
        _ = cfg.fold_column
        _ = cfg.fold_values
        _ = cfg.min_train_folds


class TestProjectConfig:
    def test_defaults(self):
        cfg = ProjectConfig()
        assert cfg.task_type == "binary"
        assert cfg.target_column == "target"
        assert isinstance(cfg.cv, CVConfig)
        assert cfg.metrics == ["brier", "accuracy"]
        assert cfg.eval_filter is None

    def test_task_specific_metric_defaults(self):
        assert ProjectConfig(task_type="regression").metrics == ["rmse", "mae", "r2"]
        assert ProjectConfig(task_type="multiclass").metrics == [
            "log_loss",
            "accuracy",
        ]

    def test_custom_values(self):
        cfg = ProjectConfig(
            task_type="multiclass",
            target_column="label",
            cv=CVConfig(strategy="stratified_kfold", n_folds=3),
            metrics=["accuracy", "f1"],
            eval_filter="split == 'val'",
        )
        assert cfg.task_type == "multiclass"
        assert cfg.target_column == "label"
        assert cfg.cv.strategy == "stratified_kfold"
        assert cfg.cv.n_folds == 3
        assert cfg.metrics == ["accuracy", "f1"]
        assert cfg.eval_filter == "split == 'val'"

    def test_cv_is_independent_per_instance(self):
        cfg1 = ProjectConfig()
        cfg2 = ProjectConfig()
        cfg1.cv.n_folds = 10
        assert cfg2.cv.n_folds == 5

    def test_metrics_independent_per_instance(self):
        cfg1 = ProjectConfig()
        cfg2 = ProjectConfig()
        cfg1.metrics.append("roc_auc")
        assert "roc_auc" not in cfg2.metrics

    def test_all_fields_accessible(self):
        cfg = ProjectConfig()
        _ = cfg.task_type
        _ = cfg.target_column
        _ = cfg.cv
        _ = cfg.metrics
        _ = cfg.eval_filter
