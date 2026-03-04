from easyml.core.config.resolver import resolve_config


def test_resolve_config_single_file(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("backtest_folds: [2020, 2021, 2022]")
    config = resolve_config(
        config_dir=tmp_path,
        file_map={"pipeline": "pipeline.yaml"},
    )
    assert config["backtest_folds"] == [2020, 2021, 2022]


def test_resolve_config_multiple_files(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("backtest_folds: [2020, 2021, 2022]")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "production.yaml").write_text(
        "models:\n  xgb_core:\n    type: xgboost\n    features: [feat_a]\n    params:\n      max_depth: 3"
    )
    (tmp_path / "ensemble.yaml").write_text(
        "ensemble:\n  method: stacked\n  meta_learner_params:\n    C: 2.5"
    )

    config = resolve_config(
        config_dir=tmp_path,
        file_map={
            "pipeline": "pipeline.yaml",
            "models": ["models/production.yaml"],
            "ensemble": "ensemble.yaml",
        },
    )
    assert config["models"]["xgb_core"]["type"] == "xgboost"
    assert config["ensemble"]["method"] == "stacked"
    assert config["backtest_folds"] == [2020, 2021, 2022]


def test_resolve_config_with_overlay(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "production.yaml").write_text(
        "models:\n  xgb_core:\n    type: xgboost\n    params:\n      max_depth: 3"
    )

    config = resolve_config(
        config_dir=tmp_path,
        file_map={"models": ["models/production.yaml"]},
        overlay={"models": {"xgb_core": {"params": {"max_depth": 5}}}},
    )
    assert config["models"]["xgb_core"]["params"]["max_depth"] == 5


def test_resolve_config_with_variant(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("data_dir: data/m/")
    (tmp_path / "pipeline_w.yaml").write_text("data_dir: data/w/")
    config = resolve_config(
        config_dir=tmp_path,
        file_map={"pipeline": "pipeline.yaml"},
        variant="w",
    )
    assert config["data_dir"] == "data/w/"


def test_resolve_config_list_of_files_merged(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "production.yaml").write_text("models:\n  xgb:\n    type: xgboost")
    (models_dir / "experimental.yaml").write_text(
        "models:\n  cat:\n    type: catboost"
    )

    config = resolve_config(
        config_dir=tmp_path,
        file_map={
            "models": ["models/production.yaml", "models/experimental.yaml"]
        },
    )
    assert "xgb" in config["models"]
    assert "cat" in config["models"]
