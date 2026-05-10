from __future__ import annotations

import argparse

from classification_smote import run_smote_classification_pipeline
from common import print_section, resolve_data_dir
from traditional_regression import run_gcn_late_pipeline



def _print_metric_block(title: str, metrics: dict) -> None:
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value}")



def main():
    parser = argparse.ArgumentParser(
        description="Run the code for the human cl classification model and the regression models."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder that contains the original Excel/CSV files. Default: parent folder of this refactor package.",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)

    print_section("Stage 1/2 - Classification model")
    classification_output = run_smote_classification_pipeline(
        str(data_dir / "human bin.xlsx")
    )
    print("Model:", classification_output["model_name"])
    print("Parameters:", classification_output["model_params"])
    _print_metric_block(
        "Classification performance:",
        {
            "test_accuracy": classification_output["test_accuracy"],
            "test_sensitivity": classification_output["test_sensitivity"],
            "test_specificity": classification_output["test_specificity"],
            "test_f1_macro": classification_output["test_f1_macro"],
            "cv_specificity": classification_output["cv_specificity"],
            "cv_accuracy": classification_output["cv_accuracy"],
            "cv_recall_macro": classification_output["cv_recall_macro"],
            "cv_f1_macro": classification_output["cv_f1_macro"],
        },
    )

    
    print_section("Stage 2/2 - GCN_Late regression model")
    gcn_late_output = run_gcn_late_pipeline(str(data_dir / "GCN_Late.xlsx"))

    voting_output = gcn_late_output["voting"]
    print(f"Model: {voting_output['model_name']}")
    _print_metric_block("Performance:", voting_output["metrics"])
    print()


if __name__ == "__main__":
    main()
