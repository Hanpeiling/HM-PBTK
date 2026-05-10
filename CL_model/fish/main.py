from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from config import DEFAULT_DATA_PATH, DEFAULT_RESULTS_PATH, NOTEBOOK_BEST_PARAMS
from data_preprocessing import load_dataset, prepare_classification_data, prepare_regression_data
from metrics_utils import evaluate_classifier, evaluate_regressor, save_results_as_json
from modeling import (
    build_extra_trees_regressor,
    build_gradient_boosting_classifier,
    build_gradient_boosting_regressor,
    build_random_forest_regressor,
    build_voting_regressor,
    train_logistic_regression_baseline,
)
from reporting import print_classification_results, print_regression_results, print_section_title
from tuning import (
    tune_extra_trees_regressor,
    tune_gradient_boosting_classifier,
    tune_gradient_boosting_regressor,
    tune_random_forest_regressor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the code to develop the fish cl model"
            "The classification workflow runs first, and the regression workflow runs second."
        )
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to the Excel dataset file.")
    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to the JSON file used to save all metrics and best parameters.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    results_path = Path(args.results)
    df = load_dataset(data_path)

    classification_data = prepare_classification_data(df)
    baseline_model = train_logistic_regression_baseline(
        classification_data.raw_train_features,
        classification_data.train_target,
    )
    baseline_predictions = baseline_model.predict(classification_data.raw_test_features)

    if args.skip_tuning:
        classification_best_params = NOTEBOOK_BEST_PARAMS["classification_gb"]
    else:
        classification_best_params = tune_gradient_boosting_classifier(
            classification_data.train_features,
            classification_data.train_target,
        )

    final_classifier = build_gradient_boosting_classifier(classification_best_params)
    final_classifier.fit(classification_data.train_features, classification_data.train_target)
    classifier_results = evaluate_classifier(
        final_classifier,
        classification_data.train_features,
        classification_data.train_target,
        classification_data.test_features,
        classification_data.test_target,
    )

    regression_data = prepare_regression_data(df)
    if args.skip_tuning:
        regression_gb_best_params = NOTEBOOK_BEST_PARAMS["regression_gb"]
        regression_et_best_params = NOTEBOOK_BEST_PARAMS["regression_et"]
        regression_rf_best_params = NOTEBOOK_BEST_PARAMS["regression_rf"]
    else:
        regression_gb_best_params = tune_gradient_boosting_regressor(
            regression_data.train_features,
            regression_data.train_target,
        )
        regression_et_best_params = tune_extra_trees_regressor(
            regression_data.train_features,
            regression_data.train_target,
        )
        regression_rf_best_params = tune_random_forest_regressor(
            regression_data.train_features,
            regression_data.train_target,
        )

    gb_regressor = build_gradient_boosting_regressor(regression_gb_best_params)
    gb_regressor.fit(regression_data.train_features, regression_data.train_target)
    gb_results = evaluate_regressor(
        gb_regressor,
        regression_data.train_features,
        regression_data.train_target,
        regression_data.test_features,
        regression_data.test_target,
    )

    et_regressor = build_extra_trees_regressor(regression_et_best_params)
    et_regressor.fit(regression_data.train_features, regression_data.train_target)
    et_results = evaluate_regressor(
        et_regressor,
        regression_data.train_features,
        regression_data.train_target,
        regression_data.test_features,
        regression_data.test_target,
    )

    rf_regressor = build_random_forest_regressor(regression_rf_best_params)
    rf_regressor.fit(regression_data.train_features, regression_data.train_target)
    rf_results = evaluate_regressor(
        rf_regressor,
        regression_data.train_features,
        regression_data.train_target,
        regression_data.test_features,
        regression_data.test_target,
    )

    voting_regressor = build_voting_regressor(rf_regressor, gb_regressor, et_regressor)
    voting_regressor.fit(regression_data.train_features, regression_data.train_target)
    voting_results = evaluate_regressor(
        voting_regressor,
        regression_data.train_features,
        regression_data.train_target,
        regression_data.test_features,
        regression_data.test_target,
    )

    print_section_title("Classification model performance")
    print_classification_results(classifier_results)

    print_section_title("Quantitative model performance")
    print_regression_results(voting_results)


if __name__ == "__main__":
    main()
