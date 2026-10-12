"""
iforest.py — Isolation Forest baseline for anomaly detection.

Operates on windowed data: either flattened (N, W*D) or summary features
(mean, std, min, max per metric → (N, 4*D)).
"""

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def _extract_features(windows: np.ndarray, mode: str = "summary") -> np.ndarray:
    """
    Convert (N, W, D) windows to 2-D feature matrix.

    Parameters
    ----------
    mode : "summary" or "flatten"
        summary → (N, 4*D): per-metric mean, std, min, max.
        flatten → (N, W*D): raw flattened window.
    """
    if mode == "flatten":
        return windows.reshape(windows.shape[0], -1)

    # Summary statistics along the time axis
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    wmin = windows.min(axis=1)
    wmax = windows.max(axis=1)
    return np.concatenate([mean, std, wmin, wmax], axis=1)


class IForestDetector:
    """Thin wrapper around sklearn IsolationForest for our pipeline."""

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        feature_mode: str = "summary",
        random_state: int = 42,
    ):
        self.feature_mode = feature_mode
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, train_windows: np.ndarray) -> "IForestDetector":
        """Fit on normal training windows."""
        X = _extract_features(train_windows, self.feature_mode)
        logger.info(f"Fitting Isolation Forest on {X.shape} features …")
        self.model.fit(X)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        X = _extract_features(windows, self.feature_mode)
        # sklearn score_samples: higher → more normal; negate for anomaly score
        return -self.model.score_samples(X)
