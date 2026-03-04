from easyml.core.config.loader import load_config_file


def test_load_basic(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("data_dir: data/m/\nbacktest_folds: [2020, 2021]")
    result = load_config_file(tmp_path, "pipeline.yaml")
    assert result["data_dir"] == "data/m/"
    assert result["backtest_folds"] == [2020, 2021]


def test_variant_resolution(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("data_dir: data/m/")
    (tmp_path / "pipeline_w.yaml").write_text("data_dir: data/w/")
    result = load_config_file(tmp_path, "pipeline.yaml", variant="w")
    assert result["data_dir"] == "data/w/"


def test_variant_fallback(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("data_dir: data/m/")
    result = load_config_file(tmp_path, "pipeline.yaml", variant="w")
    assert result["data_dir"] == "data/m/"


def test_load_from_subdirectory(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "production.yaml").write_text("models:\n  xgb:\n    type: xgboost")
    result = load_config_file(tmp_path, "models/production.yaml")
    assert result["models"]["xgb"]["type"] == "xgboost"
