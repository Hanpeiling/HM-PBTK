from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config import CROSS_VALIDATION_FOLDS


def make_bagging_regressor(base_model=None, **kwargs) -> BaggingRegressor:
    parameters = inspect.signature(BaggingRegressor).parameters
    if base_model is not None:
        if "estimator" in parameters:
            kwargs["estimator"] = base_model
        else:
            kwargs["base_estimator"] = base_model
    return BaggingRegressor(**kwargs)


def evaluate_regressor(model, x_train, y_train, x_test, y_test) -> dict[str, float]:
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)

    cv_r2 = cross_val_score(
        model,
        x_train,
        y_train,
        cv=CROSS_VALIDATION_FOLDS,
        scoring="r2",
    ).mean()
    cv_mse_scores = -cross_val_score(
        model,
        x_train,
        y_train,
        cv=CROSS_VALIDATION_FOLDS,
        scoring="neg_mean_squared_error",
    )

    return {
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_mse": float(mean_squared_error(y_test, test_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "train_mse": float(mean_squared_error(y_train, train_pred)),
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_mse_mean": float(np.mean(cv_mse_scores)),
    }


def print_metrics(model_name: str, metrics_dict: dict[str, float]) -> None:
    print(f"===== {model_name} =====")
    print("Test Set R2 Score:", metrics_dict["test_r2"])
    print("Test Set MSE:", metrics_dict["test_mse"])
    print("Train Set R2 Score:", metrics_dict["train_r2"])
    print("Train Set MSE:", metrics_dict["train_mse"])
    print("Cross-Validation R2 Scores:", metrics_dict["cv_r2_mean"])
    print("Average Cross-Validation MSE Score:", metrics_dict["cv_mse_mean"])
    print()


def save_results_json(results: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
