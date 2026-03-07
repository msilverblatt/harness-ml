"""Kaggle drive adapter — upload datasets and notebooks."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional


def _check_deps() -> None:
    """Verify that the kaggle package is installed and configured."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required. Install it with: pip install kaggle"
        )
    except OSError:
        # kaggle raises OSError if ~/.kaggle/kaggle.json is not found on import
        pass


def _get_api():
    """Import, authenticate, and return a KaggleApi instance."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def upload_dataset(
    files: List[Path],
    *,
    dataset_slug: str,
    title: str,
    update: bool = False,
) -> dict:
    """Upload files as a Kaggle dataset.

    Parameters
    ----------
    files : list of Path
        Files to include in the dataset.
    dataset_slug : str
        Kaggle dataset identifier, e.g. ``"username/dataset-name"``.
    title : str
        Human-readable title for the dataset.
    update : bool
        If True, create a new version of an existing dataset instead of
        creating a brand-new one.

    Returns
    -------
    dict
        ``{"status": "ok", "slug": dataset_slug, "files": <count>}``
    """
    api = _get_api()

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)

        # Copy all files into the staging directory
        for f in files:
            shutil.copy2(f, staging_path / Path(f).name)

        # Write dataset-metadata.json
        metadata = {
            "title": title,
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging_path / "dataset-metadata.json").write_text(json.dumps(metadata))

        if update:
            api.dataset_create_version(
                staging_path,
                version_notes="Updated via harnessml",
            )
        else:
            api.dataset_create_new(staging_path)

    return {"status": "ok", "slug": dataset_slug, "files": len(files)}


def upload_notebook(
    notebook_path: Path,
    *,
    kernel_slug: str,
    title: str,
    dataset_sources: Optional[List[str]] = None,
) -> dict:
    """Upload a Jupyter notebook to Kaggle Kernels.

    Parameters
    ----------
    notebook_path : Path
        Path to the ``.ipynb`` file.
    kernel_slug : str
        Kaggle kernel identifier, e.g. ``"username/kernel-name"``.
    title : str
        Human-readable title for the kernel.
    dataset_sources : list of str, optional
        Dataset slugs to attach as data sources.

    Returns
    -------
    dict
        ``{"status": "ok", "slug": kernel_slug}``
    """
    api = _get_api()

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)

        # Copy notebook into staging
        shutil.copy2(notebook_path, staging_path / Path(notebook_path).name)

        # Write kernel-metadata.json
        metadata = {
            "id": kernel_slug,
            "title": title,
            "code_file": Path(notebook_path).name,
            "language": "python",
            "kernel_type": "notebook",
            "dataset_sources": dataset_sources or [],
        }
        (staging_path / "kernel-metadata.json").write_text(json.dumps(metadata))

        api.kernels_push(staging_path)

    return {"status": "ok", "slug": kernel_slug}
