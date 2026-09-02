import numpy as np
import pandas as pd


class ProviderContext:
    """Per-fold transient storage for provider model outputs."""

    def __init__(self):
        self._instance: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # model → (train_preds, test_preds)
        self._entity: dict[str, pd.DataFrame] = {}  # model → entity DataFrame

    def store_instance(self, model_name: str, train_preds: np.ndarray, test_preds: np.ndarray):
        self._instance[model_name] = (train_preds, test_preds)

    def store_entity(self, model_name: str, entity_df: pd.DataFrame):
        self._entity[model_name] = entity_df

    def get_instance(self, model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        return self._instance.get(model_name)

    def get_entity(self, model_name: str) -> pd.DataFrame | None:
        return self._entity.get(model_name)

    def inject_features(self, df: pd.DataFrame, split: str, model_deps: list[str]) -> pd.DataFrame:
        """Inject provider outputs as feature columns.
        split: 'train' or 'test' — selects which predictions to use.
        """
        result = df.copy()
        for dep in model_deps:
            instance = self._instance.get(dep)
            if instance is not None:
                idx = 0 if split == "train" else 1
                col_name = f"pred_{dep}"
                preds = instance[idx]
                if len(preds) == len(result):
                    result[col_name] = preds
            # Entity-level injection would go here (lookup + diff)
        return result

    def available_providers(self) -> list[str]:
        return list(set(list(self._instance.keys()) + list(self._entity.keys())))
