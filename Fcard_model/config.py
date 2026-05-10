from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "cardiac_output.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN_INDEX = 0
FEATURE_START_COLUMN_INDEX = 2
CATEGORICAL_COLUMN_SLICE = slice(0, 2)
NUMERICAL_COLUMN_SLICE = slice(2, 5)
TEST_SIZE = 0.2
TRAIN_TEST_SPLIT_RANDOM_STATE = 220

ONE_HOT_COLUMN_NAMES = [
    "Family _Acipenseridae",
    "Family _Anguillidae",
    "Family _Catostomidae",
    "Family _Centrarchidae",
    "Family _Coryphaenidae",
    "Family _Cyprinidae",
    "Family _Danionidae",
    "Family _Eleginopidae",
    "Family _Ictaluridae",
    "Family _Leuciscidae",
    "Family _Moronidae",
    "Family _Salmonidae",
    "Environment_Freshwater",
    "Environment_Marine",
    "Environment_Marine; freshwater",
]

RF_FINAL_PARAMS = {
    "max_features": 18,
    "max_depth": 19,
    "random_state": 86,
    "n_estimators": 161,
}

GBDT_FINAL_PARAMS = {
    "max_features": 11,
    "random_state": 0,
    "n_estimators": 181,
}

ET_FINAL_PARAMS = {
    "max_depth": 18,
    "max_features": 21,
    "n_estimators": 71,
}

BAGGING_FINAL_PARAMS = {
    "n_estimators": 91,
    "max_samples": 0.9,
    "max_features": 16,
}

RF_PARAM_GRID_N_ESTIMATORS = {"n_estimators": range(1, 300, 10)}
RF_PARAM_GRID_MAX_FEATURES = {"max_features": range(1, 20, 1)}
RF_PARAM_GRID_MAX_DEPTH = {"max_depth": range(1, 20, 1)}

GBDT_PARAM_GRID_N_ESTIMATORS = {"n_estimators": range(1, 300, 10)}
GBDT_PARAM_GRID_MAX_FEATURES = {"max_features": range(1, 31, 1)}

ET_PARAM_GRID = {
    "n_estimators": range(1, 300, 10),
    "max_features": range(1, 50, 10),
    "max_depth": range(1, 20, 1),
}

BAGGING_PARAM_GRID = {
    "n_estimators": range(1, 300, 10),
    "max_features": range(1, 17, 1),
    "max_samples": [0.5, 0.7, 0.9],
}

CROSS_VALIDATION_FOLDS = 10
