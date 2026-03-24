import hashlib
import json
from pathlib import Path

import numpy as np


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
    ) -> str:
        payload = {
            "config": model_config,
            "feature_schema": feature_schema,
            "upstream": upstream_fingerprints or {},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def get(self, model_name: str, fold_id: str, fingerprint: str) -> np.ndarray | None:
        if self._dir is None:
            return None
        path = self._cache_path(model_name, fold_id, fingerprint)
        if path.exists():
            return np.load(path)
        return None

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
