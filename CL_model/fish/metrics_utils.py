from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _to_builtin(value: Any):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    return value


def calculate_multiclass_specificity(y_true, y_pred) -> float:
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    specificity_scores = []
    for i in range(conf_matrix.shape[0]):
        true_negative = conf_matrix.sum() - conf_matrix[i, :].sum() - conf_matrix[:, i].sum() + conf_matrix[i, i]
        false_positive = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_scores.append(true_negative / (true_negative + false_positive))
    return float(sum(specificity_scores) / len(specificity_scores))


def calculate_per_class_sensitivity_and_specificity(y_true, y_pred):
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    sensitivity_scores = []
    specificity_scores = []
    for i in range(conf_matrix.shape[0]):
        true_positive = conf_matrix[i, i]
        false_negative = conf_matrix[i, :].sum() - true_positive
        sensitivity_scores.append(true_positive / (true_positive + false_negative))
        true_negative = conf_matrix.sum() - conf_matrix[i, :].sum() - conf_matrix[:, i].sum() + conf_matrix[i, i]
        false_positive = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_scores.append(true_negative / (true_negative + false_positive))
    return [_to_builtin(x) for x in sensitivity_scores], [_to_builtin(x) for x in specificity_scores]


def evaluate_classifier(model, x_train, y_train, x_test, y_test):
    predictions = model.predict(x_test)
    sensitivity_scores, specificity_scores = calculate_per_class_sensitivity_and_specificity(y_test, predictions)
    cv = StratifiedKFold(n_splits=10)
    specificity_scorer = make_scorer(calculate_multiclass_specificity)
    results = {
        "test_accuracy": metrics.accuracy_score(y_test, predictions),
        "test_sensitivity": sensitivity_scores,
        "test_specificity": specificity_scores,
        "test_f1_macro": metrics.f1_score(y_test, predictions, average="macro"),
        "cross_validation_specificity": cross_val_score(model, x_train, y_train, cv=cv, scoring=specificity_scorer, n_jobs=-1).mean(),
        "cross_validation_accuracy": cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean(),
        "cross_validation_recall_macro": cross_val_score(model, x_train, y_train, cv=cv, scoring="recall_macro", n_jobs=-1).mean(),
        "cross_validation_f1_macro": cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean(),
    }
    return _to_builtin(results)


def evaluate_regressor(model, x_train, y_train, x_test, y_test):
    test_predictions = model.predict(x_test)
    train_predictions = model.predict(x_train)
    results = {
        "test_r2": metrics.r2_score(y_test, test_predictions),
        "test_mse": metrics.mean_squared_error(y_test, test_predictions),
        "train_r2": metrics.r2_score(y_train, train_predictions),
        "train_mse": metrics.mean_squared_error(y_train, train_predictions),
        "cross_validation_r2": cross_val_score(model, x_train, y_train, cv=10, scoring="r2", n_jobs=-1).mean(),
        "cross_validation_mse": (-cross_val_score(model, x_train, y_train, cv=10, scoring="neg_mean_squared_error", n_jobs=-1)).mean(),
    }
    return _to_builtin(results)


def save_results_as_json(results: dict[str, Any], output_path: str | Path):
    output_path = Path(output_path)
    output_path.write_text(json.dumps(_to_builtin(results), indent=2), encoding="utf-8")
