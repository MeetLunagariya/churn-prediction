"""Probability calibration.

Isotonic and Platt (sigmoid) calibration of a fitted classifier on a
held-out validation set, plus a vectorized Expected Calibration Error
implementation. Calibration is gated by *probabilities*, not by ranking,
so calibrated models can shift threshold decisions noticeably even when
ROC-AUC is unchanged.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

CalibrationMethod = Literal["isotonic", "sigmoid"]


def calibrate(
    fitted_classifier: ClassifierMixin,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: CalibrationMethod = "isotonic",
) -> CalibratedClassifierCV:
    """Wrap a *fitted* classifier with post-hoc probability calibration.

    The classifier must already be fit on the *training* set; ``X_val/y_val``
    is a separate validation slice used to fit the calibrator. Using the
    same data for both fitting and calibrating is a leakage anti-pattern.

    sklearn 1.7+ removed ``cv='prefit'``; we wrap the fitted estimator in
    ``FrozenEstimator`` to express the same intent.
    """
    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_classifier), method=method)
    calibrator.fit(X_val, y_val)
    return calibrator


def expected_calibration_error(y_true: ArrayLike, y_proba: ArrayLike, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples whose predicted probability falls in
    bin b, ``acc`` is the empirical positive rate, and ``conf`` is the
    mean predicted probability in the bin. Lower is better; 0 is perfect
    calibration.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_proba_arr, bin_edges) - 1, 0, n_bins - 1)
    n = len(y_true_arr)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_acc = float(y_true_arr[mask].mean())
        bin_conf = float(y_proba_arr[mask].mean())
        ece += (int(mask.sum()) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    y_true: ArrayLike, y_proba: ArrayLike, n_bins: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return (bin_conf, bin_acc, bin_counts) for plotting reliability diagrams."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_proba_arr, bin_edges) - 1, 0, n_bins - 1)
    confs = np.zeros(n_bins, dtype=np.float64)
    accs = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            confs[b] = float(y_proba_arr[mask].mean())
            accs[b] = float(y_true_arr[mask].mean())
    return confs, accs, counts
