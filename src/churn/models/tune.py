"""Hyperparameter tuning via Optuna.

The search space targets scikit-learn's ``HistGradientBoostingClassifier``.
The objective is mean validation PR-AUC over a stratified 5-fold CV on
the training partition only — the test set is never visible inside the
tuner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from churn.models.train import build_pipeline


def _hgb_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 95),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 1.0, log=True),
    }


def tune_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    use_engineered: bool = False,
    n_trials: int = 30,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], optuna.Study]:
    """Run Optuna search and return (best_params, study).

    Objective: mean PR-AUC across stratified k-fold CV. We use PR-AUC
    rather than ROC-AUC because the positive class is the minority and
    PR-AUC is more responsive to gains in the regime that matters for
    retention decisions.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = _hgb_search_space(trial)
        fold_scores: list[float] = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipeline = build_pipeline(
                "hgb",
                seed=seed,
                use_engineered=use_engineered,
                estimator_params=params,
            )
            pipeline.fit(X_tr, y_tr)
            proba = pipeline.predict_proba(X_vl)[:, 1]
            fold_scores.append(float(average_precision_score(y_vl, proba)))
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study
