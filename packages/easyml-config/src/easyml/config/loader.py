"""Load YAML config files with variant suffix resolution."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config_file(
    config_dir: Path,
    filename: str,
    variant: str | None = None,
) -> dict:
    """Load a single YAML config file, with optional variant resolution.

    Parameters
    ----------
    config_dir:
        Root directory containing config files.
    filename:
        Relative path to the config file (e.g. ``"pipeline.yaml"``
        or ``"models/production.yaml"``).
    variant:
        If set, try ``stem_{variant}.suffix`` first (e.g.
        ``pipeline_w.yaml``).  Falls back to *filename* when the
        variant file does not exist.

    Returns
    -------
    dict
        Parsed YAML contents as a plain dict.
    """
    filepath = Path(config_dir) / filename

    if variant is not None:
        variant_name = f"{filepath.stem}_{variant}{filepath.suffix}"
        variant_path = filepath.parent / variant_name
        if variant_path.exists():
            filepath = variant_path

    with open(filepath) as fh:
        return yaml.safe_load(fh) or {}
