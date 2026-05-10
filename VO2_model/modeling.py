"""Model training and evaluation helpers."""

from __future__ import annotations

from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config import CV_FOLDS, N_JOBS



def build_random_forest_model(params: Dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(**params)



def train_random_forest(X_train, y_train, params: Dict[str, Any]) -> RandomForestRegressor:
    model = build_random_forest_model(params)
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    cv_r2 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=N_JOBS,
    ).mean()
    cv_mse = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=N_JOBS,
    ).mean()

    return {
        "test_r2": float(r2_score(y_test, test_predictions)),
        "test_mse": float(mean_squared_error(y_test, test_predictions)),
        "train_r2": float(r2_score(y_train, train_predictions)),
        "train_mse": float(mean_squared_error(y_train, train_predictions)),
        "cv_r2_mean": float(cv_r2),
        "cv_mse_mean": float(cv_mse),
    }
