from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics, preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer


FINAL_CLASSIFIER_PARAMS = {
    "max_features": 26,
    "random_state": 0,
    "n_estimators": 391,
}


def _calculate_average_specificity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    specificity_values = []
    for i in range(conf_matrix.shape[0]):
        tn = (
            conf_matrix.sum()
            - conf_matrix[i, :].sum()
            - conf_matrix[:, i].sum()
            + conf_matrix[i, i]
        )
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_values.append(tn / (tn + fp))
    return float(sum(specificity_values) / len(specificity_values))



def prepare_smote_classification_data(data_path: str) -> dict[str, Any]:
    df = pd.read_excel(data_path)

    x = df.iloc[:, 2:]
    y = df["Bin3.Adj"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    smote = SMOTE(random_state=0)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    x_test = x_test.copy()
    y_test = y_test.copy()
    x_train_res = x_train_res.copy()
    y_train_res = y_train_res.copy()

    x_train_res.index = range(x_train_res.shape[0])
    x_test.index = range(x_test.shape[0])
    y_train_res.index = range(y_train_res.shape[0])
    y_test.index = range(y_test.shape[0])

    scaler = preprocessing.StandardScaler().fit(x_train_res)
    x_train_std = pd.DataFrame(
        scaler.transform(x_train_res),
        columns=x_train_res.columns,
    )
    x_test_std = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
    )

    return {
        "raw_data": df,
        "x_train": x_train_res,
        "x_test": x_test,
        "y_train": y_train_res,
        "y_test": y_test,
        "x_train_model": x_train_std,
        "x_test_model": x_test_std,
        "scaler": scaler,
    }



def run_smote_classification_pipeline(data_path: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)

    prepared = prepare_smote_classification_data(data_path)
    x_train_model = prepared["x_train_model"]
    x_test_model = prepared["x_test_model"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    gb_classifier = GradientBoostingClassifier(**FINAL_CLASSIFIER_PARAMS)
    gb_classifier.fit(x_train_model, y_train)
    y_pred_test = gb_classifier.predict(x_test_model)

    test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
    test_f1 = metrics.f1_score(y_test, y_pred_test, average="macro")

    conf_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    sensitivity_values = []
    specificity_values = []
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        sensitivity_values.append(tp / (tp + fn))

        tn = (
            conf_matrix.sum()
            - conf_matrix[i, :].sum()
            - conf_matrix[:, i].sum()
            + conf_matrix[i, i]
        )
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_values.append(tn / (tn + fp))

    cv = StratifiedKFold(n_splits=10)
    specificity_scorer = make_scorer(_calculate_average_specificity)
    cv_specificity = cross_val_score(
        gb_classifier, x_train_model, y_train, cv=cv, scoring=specificity_scorer
    ).mean()
    cv_accuracy = cross_val_score(
        gb_classifier, x_train_model, y_train, cv=cv, scoring="accuracy"
    ).mean()
    cv_recall_macro = cross_val_score(
        gb_classifier, x_train_model, y_train, cv=cv, scoring="recall_macro"
    ).mean()
    cv_f1_macro = cross_val_score(
        gb_classifier, x_train_model, y_train, cv=cv, scoring="f1_macro"
    ).mean()

    return {
        "model_name": "GradientBoostingClassifier",
        "model_params": dict(FINAL_CLASSIFIER_PARAMS),
        "test_accuracy": float(test_accuracy),
        "test_sensitivity": [float(x) for x in sensitivity_values],
        "test_specificity": [float(x) for x in specificity_values],
        "test_f1_macro": float(test_f1),
        "cv_specificity": float(cv_specificity),
        "cv_accuracy": float(cv_accuracy),
        "cv_recall_macro": float(cv_recall_macro),
        "cv_f1_macro": float(cv_f1_macro),
        "prepared_data": prepared,
    }
