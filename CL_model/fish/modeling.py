from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression


def train_logistic_regression_baseline(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model


def build_gradient_boosting_classifier(best_params: dict):
    return GradientBoostingClassifier(**best_params)


def build_gradient_boosting_regressor(best_params: dict):
    return GradientBoostingRegressor(**best_params)


def build_extra_trees_regressor(best_params: dict):
    params = {**best_params}
    params.setdefault("n_jobs", -1)
    return ExtraTreesRegressor(**params)


def build_random_forest_regressor(best_params: dict):
    params = {**best_params}
    params.setdefault("n_jobs", -1)
    return RandomForestRegressor(**params)


def build_voting_regressor(rf_model, gb_model, et_model):
    return VotingRegressor(estimators=[("RF_reg", rf_model), ("GB_reg", gb_model), ("ET_reg", et_model)])
