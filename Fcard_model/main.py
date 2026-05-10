from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_DATA_PATH, OUTPUT_DIR
from data_pipeline import prepare_data
from model_search import (
    tune_bagging,
    tune_extra_trees,
    tune_gradient_boosting,
    tune_random_forest,
)
from model_utils import print_metrics, save_results_json
from models import (
    fit_bagging,
    fit_extra_trees,
    fit_gradient_boosting,
    fit_random_forest,
    fit_voting_regressor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the code to develop the fcard model"
          
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the Excel data file. Default: cardiac_output.xlsx in the current directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    prepared = prepare_data(data_path)

    # Train each base model first because VotingRegressor depends on them.
    rf_model, rf_metrics = fit_random_forest(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
    )
    rf_search_results = tune_random_forest(prepared.x_train, prepared.y_train)

    gbdt_model, gbdt_metrics = fit_gradient_boosting(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
    )
    gbdt_search_results = tune_gradient_boosting(prepared.x_train, prepared.y_train)

    et_search_results = tune_extra_trees(prepared.x_train, prepared.y_train)
    et_model, et_metrics = fit_extra_trees(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
    )

    bagging_search_results = tune_bagging(prepared.x_train, prepared.y_train)
    bagging_model, bagging_metrics = fit_bagging(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
    )

    _, voting_metrics = fit_voting_regressor(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
        et_model=et_model,
        rf_model=rf_model,
        bagging_model=bagging_model,
    )


    print_metrics("Voting", voting_metrics)


if __name__ == "__main__":
    main()
