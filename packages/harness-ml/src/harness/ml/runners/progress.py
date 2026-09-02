from typing import Protocol


class BacktestProgress(Protocol):
    def on_fold_start(self, fold_id: str, fold_num: int, total_folds: int) -> None: ...
    def on_model_trained(self, model_name: str, fold_id: str, duration_s: float) -> None: ...
    def on_wave_complete(self, wave_num: int, total_waves: int) -> None: ...
    def on_backtest_complete(self, metrics: dict[str, float]) -> None: ...


class NoOpProgress:
    """Default no-op progress callback."""

    def on_fold_start(self, fold_id, fold_num, total_folds):
        pass

    def on_model_trained(self, model_name, fold_id, duration_s):
        pass

    def on_wave_complete(self, wave_num, total_waves):
        pass

    def on_backtest_complete(self, metrics):
        pass
