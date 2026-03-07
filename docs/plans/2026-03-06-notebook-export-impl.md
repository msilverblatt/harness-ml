# Notebook Export & Cloud Upload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add notebook generation from project config, plus Google Drive and Kaggle upload adapters, exposed via MCP tools.

**Architecture:** `core/runner/notebook.py` generates platform-aware `.ipynb` files from ProjectConfig. `core/runner/drives/drive.py` and `core/runner/drives/kaggle.py` handle uploads via optional deps. MCP exposes `pipeline(action="export_notebook")`, `data(action="upload_drive")`, and `data(action="upload_kaggle")`.

**Tech Stack:** `nbformat` (notebook generation), `google-api-python-client` + `google-auth-oauthlib` (Drive), `kaggle` (Kaggle API). All optional deps.

---

### Task 1: Add optional dependencies

**Files:**
- Modify: `packages/harness-core/pyproject.toml`

**Step 1: Add notebook, drive, and kaggle optional dep groups**

In `packages/harness-core/pyproject.toml`, add after the `quality` line in `[project.optional-dependencies]`:

```toml
notebook = ["nbformat>=5.9"]
drive = ["google-api-python-client>=2.100", "google-auth-oauthlib>=1.0"]
kaggle = ["kaggle>=1.6"]
```

Update the `all` extra to include the new groups:

```toml
all = [
    "harness-core[xgboost,catboost,lightgbm,neural,explore,shap,viz,quality,notebook,drive,kaggle]",
]
```

**Step 2: Sync the lockfile**

Run: `uv lock`
Expected: lockfile updates without errors.

**Step 3: Install the new deps for development**

Run: `uv sync --extra notebook --extra drive --extra kaggle`
Expected: packages install successfully.

**Step 4: Commit**

```bash
git add packages/harness-core/pyproject.toml uv.lock
git commit -m "feat: add optional deps for notebook, drive, kaggle"
```

---

### Task 2: Create the notebook builder core

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/notebook.py`
- Create: `packages/harness-core/tests/runner/test_notebook.py`

**Step 1: Write the failing test**

Create `packages/harness-core/tests/runner/test_notebook.py`:

```python
"""Tests for notebook generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def mini_project(tmp_path):
    """Create a minimal harnessml project structure."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Minimal config.yaml
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

        # Should have cells: install/setup, imports, config, data, features, train, evaluate, results
        assert len(nb.cells) >= 5

        # All cells should be code or markdown
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/harness-core/tests/runner/test_notebook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harnessml.core.runner.notebook'`

**Step 3: Write the notebook builder**

Create `packages/harness-core/src/harnessml/core/runner/notebook.py`:

```python
"""Generate Jupyter notebooks from harnessml project config."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

VALID_DESTINATIONS = ("colab", "kaggle", "local")
Destination = Literal["colab", "kaggle", "local"]


def _read_config(project_dir: Path) -> dict:
    """Read the project's config.yaml."""
    config_path = project_dir / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml at {config_path}")
    return yaml.safe_load(config_path.read_text())


def _make_markdown_cell(source: str):
    """Create a markdown cell dict."""
    import nbformat
    return nbformat.v4.new_markdown_cell(source)


def _make_code_cell(source: str):
    """Create a code cell dict."""
    import nbformat
    return nbformat.v4.new_code_cell(source)


def _build_install_cell(destination: Destination) -> str:
    """Build the pip install cell based on destination."""
    if destination == "local":
        return "# harness-core assumed to be installed locally"
    quiet = "-q " if destination == "kaggle" else ""
    return f"!pip install {quiet}harness-core"


def _build_setup_cell(destination: Destination, config: dict) -> str:
    """Build platform-specific setup cell."""
    project_name = config.get("project", {}).get("name", "project")
    if destination == "colab":
        return (
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "\n"
            f"PROJECT_DIR = '/content/drive/MyDrive/harnessml/{project_name}'\n"
            "DATA_DIR = f'{PROJECT_DIR}/data'\n"
            "OUTPUT_DIR = f'{PROJECT_DIR}/output'"
        )
    elif destination == "kaggle":
        return (
            f"PROJECT_DIR = '/kaggle/input/{project_name}'\n"
            "DATA_DIR = f'{PROJECT_DIR}/data'\n"
            "OUTPUT_DIR = '/kaggle/working'"
        )
    else:
        return (
            "from pathlib import Path\n"
            "\n"
            "PROJECT_DIR = str(Path.cwd())\n"
            "DATA_DIR = f'{PROJECT_DIR}/data'\n"
            "OUTPUT_DIR = f'{PROJECT_DIR}/output'"
        )


def _build_config_cell(config: dict) -> str:
    """Embed the project config as inline YAML."""
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    return (
        "import yaml\n"
        "from pathlib import Path\n"
        "\n"
        "CONFIG_YAML = \"\"\"\n"
        f"{config_yaml}"
        "\"\"\"\n"
        "\n"
        "config = yaml.safe_load(CONFIG_YAML)"
    )


def _build_data_cell(destination: Destination, config: dict) -> str:
    """Build data loading cell."""
    features_path = config.get("data", {}).get("features_path", "data/features.parquet")
    return (
        "import pandas as pd\n"
        "\n"
        f"# Load feature store\n"
        f"features_path = f'{{DATA_DIR}}/{Path(features_path).name}'\n"
        "df = pd.read_parquet(features_path)\n"
        "print(f'Loaded {{df.shape[0]}} rows, {{df.shape[1]}} columns')\n"
        "df.head()"
    )


def _build_train_cell(config: dict) -> str:
    """Build the training cell."""
    return (
        "from harnessml.core.runner.pipeline import PipelineRunner\n"
        "\n"
        "# Write config to temp file for pipeline\n"
        "import tempfile, os\n"
        "config_dir = os.path.join(OUTPUT_DIR, 'config')\n"
        "os.makedirs(config_dir, exist_ok=True)\n"
        "config_path = os.path.join(config_dir, 'config.yaml')\n"
        "with open(config_path, 'w') as f:\n"
        "    yaml.dump(config, f)\n"
        "\n"
        "# Copy data to output dir so pipeline can find it\n"
        "import shutil\n"
        "out_data = os.path.join(OUTPUT_DIR, 'data')\n"
        "os.makedirs(out_data, exist_ok=True)\n"
        "for f_name in os.listdir(DATA_DIR):\n"
        "    src = os.path.join(DATA_DIR, f_name)\n"
        "    if os.path.isfile(src):\n"
        "        shutil.copy2(src, os.path.join(out_data, f_name))\n"
        "\n"
        "runner = PipelineRunner(OUTPUT_DIR)\n"
        "result = runner.run()"
    )


def _build_evaluate_cell() -> str:
    """Build the evaluation cell."""
    return (
        "# Display backtest results\n"
        "if hasattr(result, 'metrics'):\n"
        "    print('=== Backtest Metrics ===')\n"
        "    for k, v in result.metrics.items():\n"
        "        print(f'  {k}: {v:.4f}')\n"
        "elif isinstance(result, dict):\n"
        "    for k, v in result.items():\n"
        "        print(f'  {k}: {v}')\n"
        "else:\n"
        "    print(result)"
    )


def generate_notebook(
    project_dir: Path | str,
    *,
    destination: Destination = "local",
    output_path: Path | str | None = None,
) -> Path:
    """Generate a Jupyter notebook from a project config.

    Parameters
    ----------
    project_dir : Path
        Root of the harnessml project (must contain config/config.yaml).
    destination : str
        Target platform: "colab", "kaggle", or "local".
    output_path : Path, optional
        Where to write the .ipynb. Defaults to <project_dir>/<name>_<dest>.ipynb.

    Returns
    -------
    Path
        Path to the generated notebook file.
    """
    if destination not in VALID_DESTINATIONS:
        raise ValueError(
            f"destination must be one of {VALID_DESTINATIONS}, got '{destination}'"
        )

    import nbformat

    project_dir = Path(project_dir)
    config = _read_config(project_dir)
    project_name = config.get("project", {}).get("name", "harnessml_project")

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # 1. Title
    cells.append(_make_markdown_cell(
        f"# {project_name}\n\n"
        f"Auto-generated by harnessml for **{destination}**."
    ))

    # 2. Install dependencies
    cells.append(_make_code_cell(_build_install_cell(destination)))

    # 3. Platform setup (drive mount, paths)
    cells.append(_make_markdown_cell("## Setup"))
    cells.append(_make_code_cell(_build_setup_cell(destination, config)))

    # 4. Inline config
    cells.append(_make_markdown_cell("## Configuration"))
    cells.append(_make_code_cell(_build_config_cell(config)))

    # 5. Load data
    cells.append(_make_markdown_cell("## Load Data"))
    cells.append(_make_code_cell(_build_data_cell(destination, config)))

    # 6. Train
    cells.append(_make_markdown_cell("## Train & Evaluate"))
    cells.append(_make_code_cell(_build_train_cell(config)))

    # 7. Results
    cells.append(_make_markdown_cell("## Results"))
    cells.append(_make_code_cell(_build_evaluate_cell()))

    nb.cells = cells

    # Determine output path
    if output_path is None:
        output_path = project_dir / f"{project_name}_{destination}.ipynb"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nbformat.write(nb, str(output_path))
    logger.info("Notebook written to %s", output_path)
    return output_path
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-core/tests/runner/test_notebook.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/runner/notebook.py packages/harness-core/tests/runner/test_notebook.py
git commit -m "feat: add notebook generator with colab/kaggle/local destinations"
```

---

### Task 3: Create Google Drive adapter

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/drives/__init__.py`
- Create: `packages/harness-core/src/harnessml/core/runner/drives/drive.py`
- Create: `packages/harness-core/tests/runner/test_drive_adapter.py`

**Step 1: Write the failing tests**

Create `packages/harness-core/tests/runner/test_drive_adapter.py`:

```python
"""Tests for Google Drive adapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDriveImportGuard:
    def test_import_error_without_deps(self):
        """Verify graceful error when google deps not installed."""
        from harnessml.core.runner.drives.drive import _check_deps

        # _check_deps should not raise when deps ARE installed
        # (they are in our test env). Test the function exists.
        _check_deps()


class TestDriveAuth:
    @patch("harnessml.core.runner.drives.drive._build_service")
    def test_get_service_caches_token(self, mock_build, tmp_path):
        from harnessml.core.runner.drives.drive import get_service

        mock_service = MagicMock()
        mock_build.return_value = mock_service

        svc = get_service(credentials_dir=tmp_path)
        assert svc is mock_service


class TestDriveUpload:
    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_upload_file(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import upload_file

        # Create a test file
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")

        # Mock the Drive service
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {
            "id": "file123",
            "name": "test.csv",
        }

        result = upload_file(test_file, credentials_dir=tmp_path)

        assert result["id"] == "file123"
        assert result["name"] == "test.csv"

    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_upload_file_with_folder(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import upload_file

        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {
            "id": "file456",
            "name": "test.csv",
        }

        result = upload_file(test_file, folder_id="folder789", credentials_dir=tmp_path)

        assert result["id"] == "file456"

    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_upload_returns_colab_url_for_ipynb(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import upload_file

        test_file = tmp_path / "notebook.ipynb"
        test_file.write_text("{}")

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {
            "id": "nb123",
            "name": "notebook.ipynb",
        }

        result = upload_file(test_file, credentials_dir=tmp_path)

        assert "colab_url" in result
        assert "nb123" in result["colab_url"]


class TestDriveCreateFolder:
    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_create_folder(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import create_folder

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {
            "id": "folder123",
            "name": "my_folder",
        }

        result = create_folder("my_folder", credentials_dir=tmp_path)
        assert result["id"] == "folder123"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/harness-core/tests/runner/test_drive_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harnessml.core.runner.drives'`

**Step 3: Write the Drive adapter**

Create `packages/harness-core/src/harnessml/core/runner/drives/__init__.py`:

```python
```

Create `packages/harness-core/src/harnessml/core/runner/drives/drive.py`:

```python
"""Google Drive adapter for harnessml.

Requires optional deps: google-api-python-client, google-auth-oauthlib.
Install with: pip install harness-core[drive]
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = "drive_token.json"
COLAB_URL_TEMPLATE = "https://colab.research.google.com/drive/{file_id}"


def _check_deps():
    """Verify Google API deps are installed."""
    try:
        import google.auth  # noqa: F401
        from googleapiclient import discovery  # noqa: F401
        from google_auth_oauthlib import flow  # noqa: F401
    except ImportError:
        raise ImportError(
            "Google Drive support requires google-api-python-client and "
            "google-auth-oauthlib. Install with: pip install harness-core[drive]"
        )


def _build_service(credentials_dir: Path):
    """Build an authenticated Drive v3 service.

    On first run, opens a browser for OAuth consent. Token is cached
    in credentials_dir/drive_token.json for subsequent calls.
    """
    _check_deps()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    token_path = credentials_dir / TOKEN_FILE
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret = credentials_dir / "client_secret.json"
            if not client_secret.exists():
                raise FileNotFoundError(
                    f"No client_secret.json found at {client_secret}. "
                    "Download OAuth client credentials from Google Cloud Console "
                    "and save them there."
                )
            app_flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secret), SCOPES
            )
            creds = app_flow.run_local_server(port=0)

        token_path.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_service(*, credentials_dir: Path | str | None = None):
    """Get an authenticated Drive service.

    Parameters
    ----------
    credentials_dir : Path, optional
        Directory containing client_secret.json and where drive_token.json
        is cached. Defaults to ~/.harnessml/
    """
    if credentials_dir is None:
        credentials_dir = Path.home() / ".harnessml"
    credentials_dir = Path(credentials_dir)
    credentials_dir.mkdir(parents=True, exist_ok=True)
    return _build_service(credentials_dir)


def upload_file(
    file_path: Path | str,
    *,
    folder_id: str | None = None,
    name: str | None = None,
    credentials_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Upload a file to Google Drive.

    Parameters
    ----------
    file_path : Path
        Local file to upload.
    folder_id : str, optional
        Drive folder ID to upload into.
    name : str, optional
        Filename on Drive. Defaults to local filename.
    credentials_dir : Path, optional
        Directory for OAuth credentials.

    Returns
    -------
    dict
        File metadata from Drive API, plus colab_url for .ipynb files.
    """
    from googleapiclient.http import MediaFileUpload

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    service = get_service(credentials_dir=credentials_dir)

    file_name = name or file_path.name
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    file_metadata: dict[str, Any] = {"name": file_name}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)
    result = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
        .execute()
    )

    logger.info("Uploaded %s -> Drive file ID: %s", file_path, result.get("id"))

    # Add Colab URL for notebooks
    if file_path.suffix == ".ipynb":
        result["colab_url"] = COLAB_URL_TEMPLATE.format(file_id=result["id"])

    return result


def create_folder(
    name: str,
    *,
    parent_id: str | None = None,
    credentials_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Create a folder on Google Drive.

    Parameters
    ----------
    name : str
        Folder name.
    parent_id : str, optional
        Parent folder ID.
    credentials_dir : Path, optional
        Directory for OAuth credentials.

    Returns
    -------
    dict
        Folder metadata from Drive API.
    """
    service = get_service(credentials_dir=credentials_dir)

    file_metadata: dict[str, Any] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        file_metadata["parents"] = [parent_id]

    result = (
        service.files()
        .create(body=file_metadata, fields="id, name")
        .execute()
    )

    logger.info("Created folder '%s' -> Drive folder ID: %s", name, result.get("id"))
    return result


def list_files(
    *,
    folder_id: str | None = None,
    credentials_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List files in a Drive folder.

    Parameters
    ----------
    folder_id : str, optional
        Folder ID to list. Lists root if omitted.
    credentials_dir : Path, optional
        Directory for OAuth credentials.

    Returns
    -------
    list[dict]
        List of file metadata dicts.
    """
    service = get_service(credentials_dir=credentials_dir)

    query = f"'{folder_id}' in parents" if folder_id else None
    results = (
        service.files()
        .list(q=query, fields="files(id, name, mimeType, modifiedTime)")
        .execute()
    )
    return results.get("files", [])
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-core/tests/runner/test_drive_adapter.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/runner/drives/ packages/harness-core/tests/runner/test_drive_adapter.py
git commit -m "feat: add Google Drive adapter with upload, folder, and list"
```

---

### Task 4: Create Kaggle adapter

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/drives/kaggle.py`
- Create: `packages/harness-core/tests/runner/test_kaggle_adapter.py`

**Step 1: Write the failing tests**

Create `packages/harness-core/tests/runner/test_kaggle_adapter.py`:

```python
"""Tests for Kaggle adapter."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestKaggleImportGuard:
    def test_check_deps(self):
        from harnessml.core.runner.drives.kaggle import _check_deps
        _check_deps()


class TestKaggleDatasetUpload:
    @patch("harnessml.core.runner.drives.kaggle._get_api")
    def test_upload_dataset_creates_new(self, mock_get_api, tmp_path):
        from harnessml.core.runner.drives.kaggle import upload_dataset

        # Create test files
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api

        result = upload_dataset(
            files=[data_file],
            dataset_slug="testuser/test-dataset",
            title="Test Dataset",
        )

        assert result["status"] == "ok"
        assert "testuser/test-dataset" in result["slug"]

    @patch("harnessml.core.runner.drives.kaggle._get_api")
    def test_upload_dataset_multiple_files(self, mock_get_api, tmp_path):
        from harnessml.core.runner.drives.kaggle import upload_dataset

        f1 = tmp_path / "train.csv"
        f2 = tmp_path / "test.csv"
        f1.write_text("a,b\n1,2")
        f2.write_text("a,b\n3,4")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api

        result = upload_dataset(
            files=[f1, f2],
            dataset_slug="testuser/multi",
            title="Multi File Dataset",
        )

        assert result["status"] == "ok"


class TestKaggleNotebookUpload:
    @patch("harnessml.core.runner.drives.kaggle._get_api")
    def test_upload_notebook(self, mock_get_api, tmp_path):
        from harnessml.core.runner.drives.kaggle import upload_notebook

        nb_file = tmp_path / "notebook.ipynb"
        nb_file.write_text(json.dumps({
            "cells": [],
            "metadata": {"kernelspec": {"language": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }))

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api

        result = upload_notebook(
            notebook_path=nb_file,
            kernel_slug="testuser/test-kernel",
            title="Test Kernel",
        )

        assert result["status"] == "ok"
        assert "testuser/test-kernel" in result["slug"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/harness-core/tests/runner/test_kaggle_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harnessml.core.runner.drives.kaggle'`

**Step 3: Write the Kaggle adapter**

Create `packages/harness-core/src/harnessml/core/runner/drives/kaggle.py`:

```python
"""Kaggle adapter for harnessml.

Requires optional dep: kaggle.
Install with: pip install harness-core[kaggle]
Auth: ~/.kaggle/kaggle.json (standard Kaggle API key).
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _check_deps():
    """Verify kaggle package is installed."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "Kaggle support requires the kaggle package. "
            "Install with: pip install harness-core[kaggle]"
        )
    except OSError:
        # kaggle raises OSError if ~/.kaggle/kaggle.json not found on import
        pass


def _get_api():
    """Get authenticated Kaggle API instance."""
    _check_deps()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def upload_dataset(
    files: list[Path | str],
    *,
    dataset_slug: str,
    title: str,
    update: bool = False,
) -> dict[str, Any]:
    """Upload files as a Kaggle dataset.

    Parameters
    ----------
    files : list[Path]
        Local files to include in the dataset.
    dataset_slug : str
        Kaggle dataset identifier (e.g. "username/dataset-name").
    title : str
        Display title for the dataset.
    update : bool
        If True, update existing dataset. If False, create new.

    Returns
    -------
    dict
        Status dict with slug and status.
    """
    api = _get_api()

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)

        # Copy files to staging
        for f in files:
            f = Path(f)
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")
            shutil.copy2(f, staging_path / f.name)

        # Write dataset-metadata.json
        owner, ds_name = dataset_slug.split("/")
        metadata = {
            "title": title,
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging_path / "dataset-metadata.json").write_text(json.dumps(metadata))

        if update:
            api.dataset_create_version(
                str(staging_path),
                version_notes="Updated by harnessml",
            )
        else:
            api.dataset_create_new(str(staging_path), public=False)

    logger.info("Uploaded dataset %s (%d files)", dataset_slug, len(files))
    return {"status": "ok", "slug": dataset_slug, "files": len(files)}


def upload_notebook(
    notebook_path: Path | str,
    *,
    kernel_slug: str,
    title: str,
    dataset_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Upload a notebook to Kaggle as a kernel.

    Parameters
    ----------
    notebook_path : Path
        Local .ipynb file.
    kernel_slug : str
        Kaggle kernel identifier (e.g. "username/kernel-name").
    title : str
        Display title for the kernel.
    dataset_sources : list[str], optional
        Dataset slugs to attach (e.g. ["username/dataset-name"]).

    Returns
    -------
    dict
        Status dict with slug and status.
    """
    api = _get_api()

    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)

        # Copy notebook
        shutil.copy2(notebook_path, staging_path / notebook_path.name)

        # Write kernel-metadata.json
        owner, kernel_name = kernel_slug.split("/")
        metadata = {
            "id": kernel_slug,
            "title": title,
            "code_file": notebook_path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": dataset_sources or [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (staging_path / "kernel-metadata.json").write_text(json.dumps(metadata))

        api.kernels_push(str(staging_path))

    logger.info("Uploaded notebook %s", kernel_slug)
    return {"status": "ok", "slug": kernel_slug}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-core/tests/runner/test_kaggle_adapter.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/runner/drives/kaggle.py packages/harness-core/tests/runner/test_kaggle_adapter.py
git commit -m "feat: add Kaggle adapter with dataset and notebook upload"
```

---

### Task 5: Add MCP handler for export_notebook

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py`
- Modify: `packages/harness-plugin/src/harnessml/plugin/mcp_server.py`

**Step 1: Add the handler function to pipeline.py**

Add to `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py`, before the `ACTIONS` dict:

```python
def _handle_export_notebook(*, destination, output_path, project_dir, **_kwargs):
    err = validate_required(destination, "destination")
    if err:
        return err
    err = validate_enum(destination, {"colab", "kaggle", "local"}, "destination")
    if err:
        return err
    from harnessml.core.runner.notebook import generate_notebook

    pdir = resolve_project_dir(project_dir)
    out = None
    if output_path:
        from pathlib import Path
        out = Path(output_path)

    nb_path = generate_notebook(pdir, destination=destination, output_path=out)
    return f"Notebook generated: `{nb_path}`"
```

Add `"export_notebook": _handle_export_notebook` to the `ACTIONS` dict.

**Step 2: Add destination and output_path params to the pipeline tool in mcp_server.py**

In `packages/harness-plugin/src/harnessml/plugin/mcp_server.py`, update the `pipeline` tool signature to add:

```python
    destination: str | None = None,
    output_path: str | None = None,
```

Add to the docstring under Actions:

```
      - "export_notebook": Generate a Jupyter notebook from the project config.
        Requires destination ("colab", "kaggle", or "local"). Optional: output_path.
```

Pass `destination=destination, output_path=output_path` to the handler dispatch call.

**Step 3: Verify existing tests still pass**

Run: `uv run pytest packages/harness-core/tests/runner/test_notebook.py -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py packages/harness-plugin/src/harnessml/plugin/mcp_server.py
git commit -m "feat: add pipeline export_notebook MCP action"
```

---

### Task 6: Add MCP handlers for upload_drive and upload_kaggle

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/data.py`
- Modify: `packages/harness-plugin/src/harnessml/plugin/mcp_server.py`

**Step 1: Add upload_drive handler to data.py**

Add to `packages/harness-plugin/src/harnessml/plugin/handlers/data.py`, before the `ACTIONS` dict:

```python
def _handle_upload_drive(*, files, folder_id, folder_name, name, project_dir, **_kwargs):
    if not files:
        return "**Error**: `files` (list of file paths) is required for upload_drive."
    parsed = parse_json_param(files) if isinstance(files, str) else files
    from harnessml.core.runner.drives.drive import upload_file, create_folder

    pdir = resolve_project_dir(project_dir)
    credentials_dir = pdir / ".harnessml"

    # Create folder if folder_name given and no folder_id
    target_folder_id = folder_id
    if folder_name and not folder_id:
        folder_result = create_folder(folder_name, credentials_dir=credentials_dir)
        target_folder_id = folder_result["id"]

    results = []
    for f in parsed:
        from pathlib import Path
        fp = Path(f)
        if not fp.is_absolute():
            fp = pdir / fp
        r = upload_file(fp, folder_id=target_folder_id, credentials_dir=credentials_dir)
        results.append(r)

    lines = [f"Uploaded {len(results)} file(s) to Google Drive:"]
    for r in results:
        line = f"- **{r['name']}** (ID: `{r['id']}`)"
        if "colab_url" in r:
            line += f"\n  Colab: {r['colab_url']}"
        lines.append(line)
    return "\n".join(lines)
```

**Step 2: Add upload_kaggle handler to data.py**

Add to `packages/harness-plugin/src/harnessml/plugin/handlers/data.py`, before the `ACTIONS` dict:

```python
def _handle_upload_kaggle(*, files, dataset_slug, title, name, project_dir, **_kwargs):
    if not dataset_slug:
        return "**Error**: `dataset_slug` (e.g. 'username/dataset-name') is required."
    if not files:
        return "**Error**: `files` (list of file paths) is required for upload_kaggle."
    parsed = parse_json_param(files) if isinstance(files, str) else files
    from harnessml.core.runner.drives.kaggle import upload_dataset

    pdir = resolve_project_dir(project_dir)
    resolved_files = []
    for f in parsed:
        from pathlib import Path
        fp = Path(f)
        if not fp.is_absolute():
            fp = pdir / fp
        resolved_files.append(fp)

    result = upload_dataset(
        files=resolved_files,
        dataset_slug=dataset_slug,
        title=title or dataset_slug.split("/")[-1],
    )
    return f"Uploaded {result['files']} file(s) to Kaggle dataset `{result['slug']}`."
```

**Step 3: Add both to the ACTIONS dict**

```python
    "upload_drive": _handle_upload_drive,
    "upload_kaggle": _handle_upload_kaggle,
```

**Step 4: Update data tool signature in mcp_server.py**

Add these params to the `data` tool in `mcp_server.py`:

```python
    files: list[str] | str | None = None,
    folder_id: str | None = None,
    folder_name: str | None = None,
    dataset_slug: str | None = None,
    title: str | None = None,
```

Add to the docstring:

```
      - "upload_drive": Upload file(s) to Google Drive. Requires files (list of
        paths). Optional: folder_id, folder_name (creates new folder), name.
        Returns file IDs and Colab URLs for notebooks.
      - "upload_kaggle": Upload file(s) as a Kaggle dataset. Requires files
        (list of paths), dataset_slug (e.g. "username/dataset-name").
        Optional: title.
```

Pass `files=files, folder_id=folder_id, folder_name=folder_name, dataset_slug=dataset_slug, title=title` to the handler dispatch.

**Step 5: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/handlers/data.py packages/harness-plugin/src/harnessml/plugin/mcp_server.py
git commit -m "feat: add upload_drive and upload_kaggle MCP actions"
```

---

### Task 7: Run full test suite

**Step 1: Run all tests**

Run: `uv run pytest packages/harness-core/tests/runner/test_notebook.py packages/harness-core/tests/runner/test_drive_adapter.py packages/harness-core/tests/runner/test_kaggle_adapter.py -v`
Expected: All tests PASS.

**Step 2: Run the broader test suite to check for regressions**

Run: `uv run pytest packages/harness-core/tests/ -x -q`
Expected: All existing tests still pass.

**Step 3: Final commit if any fixups needed**

Only commit if fixes were required.
