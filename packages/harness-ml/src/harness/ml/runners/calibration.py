import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR


class Calibrator:
    """Calibration methods: isotonic, platt."""

    @staticmethod
    def fit(y_true: np.ndarray, y_pred: np.ndarray, method: str = "isotonic"):
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(y_pred, y_true)
            return cal
        elif method == "platt":
            X = y_pred.reshape(-1, 1)
            cal = LR(C=1e10, max_iter=1000)
            cal.fit(X, y_true)
            return cal
        else:
            return None

    @staticmethod
    def transform(y_pred: np.ndarray, calibrator) -> np.ndarray:
        if calibrator is None:
            return y_pred
        if isinstance(calibrator, IsotonicRegression):
            return calibrator.predict(y_pred)
        if isinstance(calibrator, LR):
            return calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return y_pred
