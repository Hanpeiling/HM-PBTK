from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_DATA_PATH, NOTEBOOK_FINAL_PARAMS
from data_preprocessing import prepare_data
from modeling import train_random_forest, evaluate_model
from reporting import print_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the code to develop the vo2 model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the Excel data file.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    prepared = prepare_data(data_path)
    model = train_random_forest(prepared.X_train, prepared.y_train, NOTEBOOK_FINAL_PARAMS)
    metrics = evaluate_model(
        model,
        prepared.X_train,
        prepared.y_train,
        prepared.X_test,
        prepared.y_test,
    )

    print_metrics(metrics)


if __name__ == "__main__":
    main()
