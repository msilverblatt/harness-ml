"""Fingerprint caching for model training and meta-learner artifacts.

Provides functions to compute, save, and check fingerprints for
both individual models and the meta-learner ensemble.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_fingerprint(
    model_config: dict,
    data_mtime: float | None = None,
) -> str:
    """Compute SHA-256 hash of model config + data modification time.

    Parameters
    ----------
    model_config : dict
        Model configuration dict (serialized deterministically).
    data_mtime : float | None
        Modification time of the training data file.

    Returns
    -------
    str
        16-character hex fingerprint.
    """
    payload = {
        "config": model_config,
        "data_mtime": data_mtime,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def is_cached(models_dir: Path, model_name: str, fingerprint: str) -> bool:
    """Check if a model's .fingerprint file matches.

    Parameters
    ----------
    models_dir : Path
        Directory containing model artifacts.
    model_name : str
        Name of the model.
    fingerprint : str
        Expected fingerprint to match.

    Returns
    -------
    bool
        True if the fingerprint file exists and matches.
    """
    fp_path = Path(models_dir) / f"{model_name}.fingerprint"
    if not fp_path.exists():
        return False
    try:
        stored = fp_path.read_text().strip()
        return stored == fingerprint
    except OSError:
        return False


def save_fingerprint(
    models_dir: Path,
    model_name: str,
    fingerprint: str,
) -> None:
    """Write {model_name}.fingerprint file.

    Parameters
    ----------
    models_dir : Path
        Directory to write the fingerprint file.
    model_name : str
        Name of the model.
    fingerprint : str
        Fingerprint string to save.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    fp_path = models_dir / f"{model_name}.fingerprint"
    fp_path.write_text(fingerprint)


def compute_meta_fingerprint(
    predictions_dir: Path,
    ensemble_config: dict,
) -> str:
    """Hash backtest prediction files (mtime+size) + ensemble config.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing backtest prediction parquet files.
    ensemble_config : dict
        Ensemble configuration dict.

    Returns
    -------
    str
        16-character hex fingerprint.
    """
    predictions_dir = Path(predictions_dir)
    file_info = []

    if predictions_dir.exists():
        for f in sorted(predictions_dir.glob("*")):
            if f.is_file():
                stat = f.stat()
                file_info.append({
                    "name": f.name,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                })

    payload = {
        "files": file_info,
        "ensemble_config": ensemble_config,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_meta_cache(
    cache_dir: Path,
    meta: Any,
    calibrator: Any,
    pre_calibrators: dict | None,
    fingerprint: str,
) -> None:
    """Save meta-learner artifacts to cache_dir using joblib.

    Parameters
    ----------
    cache_dir : Path
        Directory to save the cached artifacts.
    meta : StackedEnsemble
        The meta-learner to cache.
    calibrator : calibrator or None
        The post-calibrator to cache.
    pre_calibrators : dict or None
        Dict of model_name -> pre-calibrator.
    fingerprint : str
        The fingerprint for cache validation.
    """
    import joblib

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(meta, cache_dir / "meta.joblib")
    joblib.dump(calibrator, cache_dir / "calibrator.joblib")
    joblib.dump(pre_calibrators, cache_dir / "pre_calibrators.joblib")

    fp_path = cache_dir / "meta.fingerprint"
    fp_path.write_text(fingerprint)


def load_meta_cache(
    cache_dir: Path,
    fingerprint: str,
) -> tuple | None:
    """Load cached meta-learner artifacts if fingerprint matches.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached artifacts.
    fingerprint : str
        Expected fingerprint to match.

    Returns
    -------
    tuple of (meta, calibrator, pre_calibrators) or None on cache miss.
    """
    import joblib

    cache_dir = Path(cache_dir)
    fp_path = cache_dir / "meta.fingerprint"

    if not fp_path.exists():
        return None

    try:
        stored = fp_path.read_text().strip()
        if stored != fingerprint:
            return None

        meta = joblib.load(cache_dir / "meta.joblib")
        calibrator = joblib.load(cache_dir / "calibrator.joblib")
        pre_calibrators = joblib.load(cache_dir / "pre_calibrators.joblib")
        return (meta, calibrator, pre_calibrators)
    except (OSError, Exception) as e:
        logger.warning("Failed to load meta cache: %s", e)
        return None
