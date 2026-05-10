from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

from config import BAGGING_FINAL_PARAMS, ET_FINAL_PARAMS, GBDT_FINAL_PARAMS, RF_FINAL_PARAMS
from model_utils import evaluate_regressor, make_bagging_regressor


def fit_random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(**RF_FINAL_PARAMS)
    fitted_model = model.fit(x_train, y_train)
    metrics_dict = evaluate_regressor(fitted_model, x_train, y_train, x_test, y_test)
    return fitted_model, metrics_dict


def fit_gradient_boosting(x_train, y_train, x_test, y_test):
    model = GradientBoostingRegressor(**GBDT_FINAL_PARAMS)
    fitted_model = model.fit(x_train, y_train)
    metrics_dict = evaluate_regressor(fitted_model, x_train, y_train, x_test, y_test)
    return fitted_model, metrics_dict


def fit_extra_trees(x_train, y_train, x_test, y_test):
    model = ExtraTreesRegressor(**ET_FINAL_PARAMS)
    fitted_model = model.fit(x_train, y_train)
    metrics_dict = evaluate_regressor(fitted_model, x_train, y_train, x_test, y_test)
    return fitted_model, metrics_dict


def fit_bagging(x_train, y_train, x_test, y_test):
    model = make_bagging_regressor(**BAGGING_FINAL_PARAMS)
    fitted_model = model.fit(x_train, y_train)
    metrics_dict = evaluate_regressor(fitted_model, x_train, y_train, x_test, y_test)
    return fitted_model, metrics_dict


def fit_voting_regressor(x_train, y_train, x_test, y_test, et_model, rf_model, bagging_model):
    model = VotingRegressor(
        estimators=[("et", et_model), ("rf", rf_model), ("bb", bagging_model)]
    )
    fitted_model = model.fit(x_train, y_train)
    metrics_dict = evaluate_regressor(fitted_model, x_train, y_train, x_test, y_test)
    return fitted_model, metrics_dict
