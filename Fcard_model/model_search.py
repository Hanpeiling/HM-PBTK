from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from config import (
    BAGGING_PARAM_GRID,
    CROSS_VALIDATION_FOLDS,
    ET_PARAM_GRID,
    GBDT_PARAM_GRID_MAX_FEATURES,
    GBDT_PARAM_GRID_N_ESTIMATORS,
    RF_PARAM_GRID_MAX_DEPTH,
    RF_PARAM_GRID_MAX_FEATURES,
    RF_PARAM_GRID_N_ESTIMATORS,
)
from model_utils import make_bagging_regressor


def tune_random_forest(x_train, y_train) -> dict[str, dict]:
    grid_search_n_estimators = GridSearchCV(
        estimator=RandomForestRegressor(random_state=86),
        param_grid=RF_PARAM_GRID_N_ESTIMATORS,
        scoring="r2",
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search_n_estimators.fit(x_train, y_train)

    grid_search_max_features = GridSearchCV(
        estimator=RandomForestRegressor(n_estimators=161, random_state=86),
        param_grid=RF_PARAM_GRID_MAX_FEATURES,
        scoring="r2",
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search_max_features.fit(x_train, y_train)

    grid_search_max_depth = GridSearchCV(
        estimator=RandomForestRegressor(
            max_features=18,
            n_estimators=161,
            random_state=86,
        ),
        param_grid=RF_PARAM_GRID_MAX_DEPTH,
        scoring="r2",
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search_max_depth.fit(x_train, y_train)

    return {
        "n_estimators": {
            "best_params": grid_search_n_estimators.best_params_,
            "best_score": float(grid_search_n_estimators.best_score_),
        },
        "max_features": {
            "best_params": grid_search_max_features.best_params_,
            "best_score": float(grid_search_max_features.best_score_),
        },
        "max_depth": {
            "best_params": grid_search_max_depth.best_params_,
            "best_score": float(grid_search_max_depth.best_score_),
        },
    }


def tune_gradient_boosting(x_train, y_train) -> dict[str, dict]:
    grid_search_n_estimators = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=0),
        param_grid=GBDT_PARAM_GRID_N_ESTIMATORS,
        scoring="r2",
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search_n_estimators.fit(x_train, y_train)

    grid_search_max_features = GridSearchCV(
        estimator=GradientBoostingRegressor(n_estimators=181, random_state=0),
        param_grid=GBDT_PARAM_GRID_MAX_FEATURES,
        scoring="r2",
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search_max_features.fit(x_train, y_train)

    return {
        "n_estimators": {
            "best_params": grid_search_n_estimators.best_params_,
            "best_score": float(grid_search_n_estimators.best_score_),
        },
        "max_features": {
            "best_params": grid_search_max_features.best_params_,
            "best_score": float(grid_search_max_features.best_score_),
        },
    }


def tune_extra_trees(x_train, y_train) -> dict[str, object]:
    grid_search = GridSearchCV(
        ExtraTreesRegressor(),
        ET_PARAM_GRID,
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    return {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
    }


def tune_bagging(x_train, y_train) -> dict[str, object]:
    base_model = DecisionTreeRegressor()
    bagging_model = make_bagging_regressor(base_model=base_model)

    grid_search = GridSearchCV(
        bagging_model,
        BAGGING_PARAM_GRID,
        cv=CROSS_VALIDATION_FOLDS,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    return {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
    }


def print_search_results(section_name: str, search_results: dict) -> None:
    print(f"===== {section_name} parameters =====")
    for key, value in search_results.items():
        if isinstance(value, dict) and "best_params" in value:
            print(f"{key} -> best_params: {value['best_params']}")
            print(f"{key} -> best_score: {value['best_score']}")
        else:
            print(f"{key}: {value}")
    print()
