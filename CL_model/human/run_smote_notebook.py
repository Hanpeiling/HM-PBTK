from __future__ import annotations

import argparse

from classification_smote import run_smote_classification_pipeline
from common import print_section, resolve_data_dir, save_json


def main():
    parser = argparse.ArgumentParser(
        description="SMOTE"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder that contains human bin.xlsx. Default: parent folder of this refactor package.",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print_section("Running SMOTE.ipynb as modular Python files")
    output = run_smote_classification_pipeline(str(data_dir / "human bin.xlsx"))

    summary = {k: v for k, v in output.items() if k != "prepared_data"}
    save_json(summary, data_dir / "smote_summary.json")

    print("Saved summary file:", data_dir / "smote_summary.json")
    print("Run complete.")


if __name__ == "__main__":
    main()
