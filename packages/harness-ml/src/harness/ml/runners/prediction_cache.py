import hashlib
import json
from pathlib import Path

import numpy as np


CACHE_SCHEMA_VERSION = 2


class PredictionCache:
    def __init__(self, cache_dir: Path | None = None):
        self._dir = Path(cache_dir) if cache_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def compute_fingerprint(
        self,
        model_config: dict,
        feature_schema: str,
        upstream_fingerprints: dict[str, str] | None = None,
        *,
        data_fingerprint: str | None = None,
        target_fingerprint: str | None = None,
        fold_fingerprint: str | None = None,
        task_type: str | None = None,
    ) -> str:
        """Fingerprint every input that can change a fold's predictions."""
        payload = {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "config": model_config,
            "feature_schema": feature_schema,
            "upstream": upstream_fingerprints or {},
            "data": data_fingerprint,
            "target": target_fingerprint,
            "fold": fold_fingerprint,
            "task_type": task_type,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def get(
        self,
        model_name: str,
        fold_id: str,
        fingerprint: str,
        expected_length: int | None = None,
    ) -> np.ndarray | None:
        if self._dir is None:
            return None
        path = self._cache_path(model_name, fold_id, fingerprint)
        if not path.exists():
            return None
        try:
            predictions = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if expected_length is not None and len(predictions) != expected_length:
            return None
        return predictions

    def put(self, model_name: str, fold_id: str, fingerprint: str, predictions: np.ndarray):
        if self._dir is None:
            return
        path = self._cache_path(model_name, fold_id, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, predictions)

    def has(self, model_name: str, fold_id: str, fingerprint: str) -> bool:
        if self._dir is None:
            return False
        return self._cache_path(model_name, fold_id, fingerprint).exists()

    def _cache_path(self, model_name: str, fold_id: str, fingerprint: str) -> Path:
        return self._dir / model_name / f"{fold_id}_{fingerprint}.npy"
