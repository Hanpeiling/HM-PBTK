from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "fish_cl.xlsx"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results.json"

STANDARDIZE_FIRST_N_COLUMNS = 35
CLASSIFICATION_FEATURE_START_INDEX = 5
CLASSIFICATION_TARGET_COLUMN = "Bin3"
REGRESSION_TARGET_COLUMN_INDEX = 0
REGRESSION_FEATURE_START_INDEX = 5

CLASSIFICATION_TEST_SIZE = 0.2
CLASSIFICATION_SPLIT_RANDOM_STATE = 0
SMOTE_RANDOM_STATE = 0
CLASSIFICATION_MODEL_RANDOM_STATE = 0

REGRESSION_TEST_SIZE = 0.2
REGRESSION_SPLIT_RANDOM_STATE = 64
GB_REG_RANDOM_STATE = 0
RF_RANDOM_STATE = 86
ET_RANDOM_STATE = 0

CLASSIFICATION_TUNING_GRID = {
    "n_estimators": range(1, 400, 10),
    "max_features": range(1, 31, 1),
}

GB_REG_TUNING_GRID = {
    "n_estimators": range(1, 100, 10),
    "max_features": range(1, 50, 1),
}

RF_TUNING_GRID = {
    "n_estimators": range(1, 300, 10),
    "max_features": range(1, 67, 1),
    "max_depth": range(1, 30, 1),
}

ET_TUNING_GRID = {
    "n_estimators": range(1, 300, 10),
    "max_features": range(1, 50, 10),
    "max_depth": range(1, 20, 1),
}

NOTEBOOK_BEST_PARAMS = {
    "classification_gb": {
        "n_estimators": 111,
        "max_features": 12,
        "random_state": CLASSIFICATION_MODEL_RANDOM_STATE,
    },
    "regression_gb": {
        "n_estimators": 21,
        "max_features": 32,
        "random_state": GB_REG_RANDOM_STATE,
    },
    "regression_et": {
        "n_estimators": 71,
        "max_features": 7,
        "max_depth": 28,
        "random_state": ET_RANDOM_STATE,
    },
    "regression_rf": {
        "n_estimators": 251,
        "max_features": 31,
        "max_depth": 8,
        "random_state": RF_RANDOM_STATE,
    },
}
