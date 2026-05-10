from __future__ import annotations

from typing import Any
import warnings

import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import cross_val_score, train_test_split


GCN_LATE_GBDT_PARAMS = {
    "max_features": 2,
    "random_state": 0,
    "n_estimators": 41,
}
GCN_LATE_ET_PARAMS = {
    "max_depth": 8,
    "max_features": 11,
    "n_estimators": 231,
}
GCN_LATE_RF_PARAMS = {
    "max_features": 1,
    "max_depth": 6,
    "random_state": 86,
    "n_estimators": 151,
}



def _evaluate_regressor(model, x_train, y_train, x_test, y_test) -> dict[str, float]:
    fitted = model.fit(x_train, y_train)
    y_pred_test = fitted.predict(x_test)
    y_pred_train = fitted.predict(x_train)

    test_r2 = metrics.r2_score(y_test, y_pred_test)
    test_mse = metrics.mean_squared_error(y_test, y_pred_test)
    train_r2 = metrics.r2_score(y_train, y_pred_train)
    train_mse = metrics.mean_squared_error(y_train, y_pred_train)
    cv_r2 = cross_val_score(fitted, x_train, y_train, cv=10, scoring="r2").mean()
    cv_mse = -cross_val_score(
        fitted, x_train, y_train, cv=10, scoring="neg_mean_squared_error"
    ).mean()

    return {
        "test_r2": float(test_r2),
        "test_mse": float(test_mse),
        "train_r2": float(train_r2),
        "train_mse": float(train_mse),
        "cv_r2": float(cv_r2),
        "cv_mse": float(cv_mse),
    }



def _prepare_ml_numerical_data(data_path: str) -> dict[str, Any]:
    data = pd.read_excel(data_path)

    y = data.iloc[:, 0]
    x = data.iloc[:, 2:]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=86
    )

    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    x_train.index = range(x_train.shape[0])
    x_test.index = range(x_test.shape[0])
    y_train.index = range(y_train.shape[0])
    y_test.index = range(y_test.shape[0])

    scaler_input = x_train.iloc[:, 1:]
    column_names = scaler_input.columns
    scaler = preprocessing.StandardScaler().fit(scaler_input)
    x_train_std = pd.DataFrame(scaler.transform(scaler_input), columns=column_names)

    preserved_train_column = x_train.iloc[:, 0]
    x_train_model = pd.concat([x_train_std, preserved_train_column], axis=1)

    x_test_std_input = pd.DataFrame(x_test.iloc[:, 1:])
    preserved_test_column = x_test.iloc[:, 0]
    x_test_std = pd.DataFrame(
        scaler.transform(x_test_std_input), columns=x_test_std_input.columns
    )
    x_test_model = pd.concat([x_test_std, preserved_test_column], axis=1)

    return {
        "data": data,
        "x_train_model": x_train_model,
        "x_test_model": x_test_model,
        "y_train": y_train,
        "y_test": y_test,
    }



def _prepare_gcn_late_data(data_path: str) -> dict[str, Any]:
    data = pd.read_excel(data_path)

    y = data.iloc[:, 1]
    x = data.iloc[:, 2:]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=86
    )

    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    x_train.index = range(x_train.shape[0])
    x_test.index = range(x_test.shape[0])
    y_train.index = range(y_train.shape[0])
    y_test.index = range(y_test.shape[0])

    column_names = x_train.columns
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_model = pd.DataFrame(scaler.transform(x_train), columns=column_names)

    x_test_std_input = pd.DataFrame(x_test)
    x_test_model = pd.DataFrame(
        scaler.transform(x_test_std_input), columns=x_test_std_input.columns
    )

    return {
        "data": data,
        "x_train_model": x_train_model,
        "x_test_model": x_test_model,
        "y_train": y_train,
        "y_test": y_test,
    }



def run_ml_numerical_pipeline(data_path: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)
    prepared = _prepare_ml_numerical_data(data_path)
    x_train_model = prepared["x_train_model"]
    x_test_model = prepared["x_test_model"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    gb_reg = GradientBoostingRegressor(max_features=3, random_state=0, n_estimators=51)
    fixed_metrics = _evaluate_regressor(
        gb_reg, x_train_model, y_train, x_test_model, y_test
    )

    return {
        "model_name": "GradientBoostingRegressor",
        "model_params": {"max_features": 3, "random_state": 0, "n_estimators": 51},
        "metrics": fixed_metrics,
        "prepared_data": prepared,
        "fixed_model_object": gb_reg,
    }



def run_gcn_late_pipeline(data_path: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)
    prepared = _prepare_gcn_late_data(data_path)
    x_train_model = prepared["x_train_model"]
    x_test_model = prepared["x_test_model"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    gbdt_model = GradientBoostingRegressor(**GCN_LATE_GBDT_PARAMS)
    et_model = ExtraTreesRegressor(**GCN_LATE_ET_PARAMS)
    rf_model = RandomForestRegressor(**GCN_LATE_RF_PARAMS)
    voting_model = VotingRegressor(
        estimators=[
            ("et", ExtraTreesRegressor(**GCN_LATE_ET_PARAMS)),
            ("gbdt", GradientBoostingRegressor(**GCN_LATE_GBDT_PARAMS)),
            ("rf", RandomForestRegressor(**GCN_LATE_RF_PARAMS)),
        ]
    )

    return {
        "gbdt": {
            "model_name": "GradientBoostingRegressor",
            "model_params": dict(GCN_LATE_GBDT_PARAMS),
            "metrics": _evaluate_regressor(
                gbdt_model, x_train_model, y_train, x_test_model, y_test
            ),
        },
        "extra_trees": {
            "model_name": "ExtraTreesRegressor",
            "model_params": dict(GCN_LATE_ET_PARAMS),
            "metrics": _evaluate_regressor(
                et_model, x_train_model, y_train, x_test_model, y_test
            ),
        },
        "random_forest": {
            "model_name": "RandomForestRegressor",
            "model_params": dict(GCN_LATE_RF_PARAMS),
            "metrics": _evaluate_regressor(
                rf_model, x_train_model, y_train, x_test_model, y_test
            ),
        },
        "voting": {
            "model_name": "VotingRegressor",
            "model_params": {
                "base_models": {
                    "ET": dict(GCN_LATE_ET_PARAMS),
                    "GBDT": dict(GCN_LATE_GBDT_PARAMS),
                    "RF": dict(GCN_LATE_RF_PARAMS),
                }
            },
            "metrics": _evaluate_regressor(
                voting_model, x_train_model, y_train, x_test_model, y_test
            ),
        },
        "prepared_data": prepared,
    }
