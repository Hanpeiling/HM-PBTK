from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from config import (
    CLASSIFICATION_MODEL_RANDOM_STATE,
    CLASSIFICATION_TUNING_GRID,
    ET_TUNING_GRID,
    GB_REG_RANDOM_STATE,
    GB_REG_TUNING_GRID,
    NOTEBOOK_BEST_PARAMS,
    RF_RANDOM_STATE,
    RF_TUNING_GRID,
)


def tune_gradient_boosting_classifier(x_train, y_train):
    tuning_results = {}
    search_n_estimators = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=CLASSIFICATION_MODEL_RANDOM_STATE),
        param_grid={"n_estimators": CLASSIFICATION_TUNING_GRID["n_estimators"]},
        cv=10,
        scoring="accuracy",
        n_jobs=-1,
    )
    search_n_estimators.fit(x_train, y_train)
    tuning_results["n_estimators"] = search_n_estimators.best_params_["n_estimators"]

    search_max_features = GridSearchCV(
        estimator=GradientBoostingClassifier(
            n_estimators=tuning_results["n_estimators"],
            random_state=CLASSIFICATION_MODEL_RANDOM_STATE,
        ),
        param_grid={"max_features": CLASSIFICATION_TUNING_GRID["max_features"]},
        cv=10,
        scoring="accuracy",
        n_jobs=-1,
    )
    search_max_features.fit(x_train, y_train)
    tuning_results["max_features"] = search_max_features.best_params_["max_features"]
    tuning_results["random_state"] = CLASSIFICATION_MODEL_RANDOM_STATE
    return tuning_results


def tune_gradient_boosting_regressor(x_train, y_train):
    tuning_results = {}
    search_n_estimators = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=GB_REG_RANDOM_STATE),
        param_grid={"n_estimators": GB_REG_TUNING_GRID["n_estimators"]},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    search_n_estimators.fit(x_train, y_train)
    tuning_results["n_estimators"] = search_n_estimators.best_params_["n_estimators"]

    search_max_features = GridSearchCV(
        estimator=GradientBoostingRegressor(
            n_estimators=tuning_results["n_estimators"],
            random_state=GB_REG_RANDOM_STATE,
        ),
        param_grid={"max_features": GB_REG_TUNING_GRID["max_features"]},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    search_max_features.fit(x_train, y_train)
    tuning_results["max_features"] = search_max_features.best_params_["max_features"]
    tuning_results["random_state"] = GB_REG_RANDOM_STATE
    return tuning_results


def tune_extra_trees_regressor(x_train, y_train):
    search = GridSearchCV(
        estimator=ExtraTreesRegressor(random_state=NOTEBOOK_BEST_PARAMS["regression_et"]["random_state"]),
        param_grid=ET_TUNING_GRID,
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return {**search.best_params_, "random_state": NOTEBOOK_BEST_PARAMS["regression_et"]["random_state"]}


def tune_random_forest_regressor(x_train, y_train):
    tuning_results = {}
    search_n_estimators = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RF_RANDOM_STATE),
        param_grid={"n_estimators": RF_TUNING_GRID["n_estimators"]},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    search_n_estimators.fit(x_train, y_train)
    tuning_results["n_estimators"] = search_n_estimators.best_params_["n_estimators"]

    search_max_features = GridSearchCV(
        estimator=RandomForestRegressor(
            n_estimators=tuning_results["n_estimators"],
            random_state=RF_RANDOM_STATE,
        ),
        param_grid={"max_features": RF_TUNING_GRID["max_features"]},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    search_max_features.fit(x_train, y_train)
    tuning_results["max_features"] = search_max_features.best_params_["max_features"]

    search_max_depth = GridSearchCV(
        estimator=RandomForestRegressor(
            n_estimators=tuning_results["n_estimators"],
            max_features=tuning_results["max_features"],
            random_state=RF_RANDOM_STATE,
        ),
        param_grid={"max_depth": RF_TUNING_GRID["max_depth"]},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    search_max_depth.fit(x_train, y_train)
    tuning_results["max_depth"] = search_max_depth.best_params_["max_depth"]
    tuning_results["random_state"] = RF_RANDOM_STATE
    return tuning_results
