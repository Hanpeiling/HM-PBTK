"""Hyperparameter search utilities for the Random Forest model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from config import (
    CV_FOLDS,
    N_JOBS,
    PARAM_GRID_MAX_DEPTH,
    PARAM_GRID_MAX_FEATURES,
    PARAM_GRID_N_ESTIMATORS,
    RANDOM_STATE,
)


@dataclass
class TuningStepResult:
    searched_parameter: str
    best_params: Dict[str, Any]
    best_score: float



def _run_grid_search(X_train, y_train, estimator, param_grid, searched_parameter: str) -> TuningStepResult:
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="r2",
        cv=CV_FOLDS,
        n_jobs=N_JOBS,
    )
    grid_search.fit(X_train, y_train)
    return TuningStepResult(
        searched_parameter=searched_parameter,
        best_params=grid_search.best_params_,
        best_score=float(grid_search.best_score_),
    )



def tune_random_forest(X_train, y_train):
    results = []

    step_1 = _run_grid_search(
        X_train=X_train,
        y_train=y_train,
        estimator=RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID_N_ESTIMATORS,
        searched_parameter="n_estimators",
    )
    results.append(step_1)
    best_n_estimators = step_1.best_params["n_estimators"]

    step_2 = _run_grid_search(
        X_train=X_train,
        y_train=y_train,
        estimator=RandomForestRegressor(
            n_estimators=best_n_estimators,
            random_state=RANDOM_STATE,
        ),
        param_grid=PARAM_GRID_MAX_FEATURES,
        searched_parameter="max_features",
    )
    results.append(step_2)
    best_max_features = step_2.best_params["max_features"]

    step_3 = _run_grid_search(
        X_train=X_train,
        y_train=y_train,
        estimator=RandomForestRegressor(
            n_estimators=best_n_estimators,
            max_features=best_max_features,
            random_state=RANDOM_STATE,
        ),
        param_grid=PARAM_GRID_MAX_DEPTH,
        searched_parameter="max_depth",
    )
    results.append(step_3)
    best_max_depth = step_3.best_params["max_depth"]

    final_params = {
        "n_estimators": best_n_estimators,
        "max_features": best_max_features,
        "max_depth": best_max_depth,
        "random_state": RANDOM_STATE,
    }

    return final_params, [asdict(item) for item in results]
