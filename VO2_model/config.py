"""Project configuration for the VO2 notebook refactor."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILENAME = "oxygen_consumption.xlsx"
DEFAULT_DATA_PATH = PROJECT_ROOT / DEFAULT_DATA_FILENAME
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN_INDEX = 1
FEATURE_START_INDEX = 2
TEST_SIZE = 0.2
RANDOM_STATE = 86
CV_FOLDS = 10
N_JOBS = 1

NOTEBOOK_FINAL_PARAMS = {
    "n_estimators": 111,
    "max_features": 20,
    "max_depth": 16,
    "random_state": 86,
}

PARAM_GRID_N_ESTIMATORS = {"n_estimators": range(1, 300, 10)}
PARAM_GRID_MAX_FEATURES = {"max_features": range(1, 50, 1)}
PARAM_GRID_MAX_DEPTH = {"max_depth": range(1, 20, 1)}
