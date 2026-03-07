"""Tests for notebook generation."""
from __future__ import annotations

import pytest


@pytest.fixture
def mini_project(tmp_path):
    """Create a minimal harnessml project structure."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    config = {
        "project": {"name": "test-project"},
        "data": {
            "target_column": "target",
            "key_columns": ["id"],
            "features_path": "data/features.parquet",
        },
        "models": {
            "xgb_test": {
                "type": "xgboost",
                "active": True,
                "include_in_ensemble": True,
                "features": ["feat_a", "feat_b"],
                "params": {"n_estimators": 100, "max_depth": 3},
            }
        },
        "ensemble": {"method": "stacked"},
        "backtest": {
            "cv_strategy": "loso",
            "fold_column": "fold",
            "metrics": ["brier", "accuracy"],
        },
    }

    import yaml
    (config_dir / "config.yaml").write_text(yaml.dump(config))
    return tmp_path


class TestNotebookGeneration:
    def test_generate_local_notebook(self, mini_project):
        from harnessml.core.runner.notebook import generate_notebook
        nb_path = generate_notebook(mini_project, destination="local")
        assert nb_path.exists()
        assert nb_path.suffix == ".ipynb"
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        assert len(nb.cells) >= 5
        for cell in nb.cells:
            assert cell.cell_type in ("code", "markdown")

    def test_generate_colab_notebook_has_drive_mount(self, mini_project):
        from harnessml.core.runner.notebook import generate_notebook
        nb_path = generate_notebook(mini_project, destination="colab")
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        all_source = "\n".join(c.source for c in nb.cells)
        assert "drive.mount" in all_source

    def test_generate_kaggle_notebook_has_kaggle_input(self, mini_project):
        from harnessml.core.runner.notebook import generate_notebook
        nb_path = generate_notebook(mini_project, destination="kaggle")
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        all_source = "\n".join(c.source for c in nb.cells)
        assert "/kaggle/input" in all_source

    def test_generate_notebook_custom_output_path(self, mini_project, tmp_path):
        from harnessml.core.runner.notebook import generate_notebook
        out = tmp_path / "custom_dir" / "my_notebook.ipynb"
        nb_path = generate_notebook(mini_project, destination="local", output_path=out)
        assert nb_path == out
        assert nb_path.exists()

    def test_generate_notebook_contains_config(self, mini_project):
        from harnessml.core.runner.notebook import generate_notebook
        nb_path = generate_notebook(mini_project, destination="local")
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        all_source = "\n".join(c.source for c in nb.cells)
        assert "test-project" in all_source

    def test_generate_notebook_invalid_destination(self, mini_project):
        from harnessml.core.runner.notebook import generate_notebook
        with pytest.raises(ValueError, match="destination"):
            generate_notebook(mini_project, destination="invalid")

    def test_generate_notebook_installs_harnessml(self, mini_project):
        from harnessml.core.runner.notebook import generate_notebook
        nb_path = generate_notebook(mini_project, destination="colab")
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        all_source = "\n".join(c.source for c in nb.cells)
        assert "pip install" in all_source
        assert "harness-core" in all_source
