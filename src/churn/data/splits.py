"""Deterministic stratified train/val/test splits.

A single seed is threaded through every split call so the same input
DataFrame always produces the same partitions — a precondition for
comparing experiments fairly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def stratified_split(
    df: pd.DataFrame,
    target: str = TARGET,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Two-stage stratified split.

    Test is held out first against the *full* distribution, then val is
    carved from the remainder so the test set's positive rate matches the
    population exactly. Returns a frozen container so downstream code can't
    accidentally re-split.
    """
    y = (df[target] == "Yes").astype(int)
    X = df.drop(columns=[target])

    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem, test_size=val_fraction, stratify=y_rem, random_state=seed
    )
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)
