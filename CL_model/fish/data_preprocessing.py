from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from config import (
    CLASSIFICATION_FEATURE_START_INDEX,
    CLASSIFICATION_SPLIT_RANDOM_STATE,
    CLASSIFICATION_TARGET_COLUMN,
    CLASSIFICATION_TEST_SIZE,
    DEFAULT_DATA_PATH,
    REGRESSION_FEATURE_START_INDEX,
    REGRESSION_SPLIT_RANDOM_STATE,
    REGRESSION_TARGET_COLUMN_INDEX,
    REGRESSION_TEST_SIZE,
    SMOTE_RANDOM_STATE,
    STANDARDIZE_FIRST_N_COLUMNS,
)


@dataclass
class ClassificationDataBundle:
    raw_train_features: pd.DataFrame
    raw_test_features: pd.DataFrame
    train_target: pd.Series
    test_target: pd.Series
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    scaler: preprocessing.StandardScaler


@dataclass
class RegressionDataBundle:
    raw_train_features: pd.DataFrame
    raw_test_features: pd.DataFrame
    train_target: pd.Series
    test_target: pd.Series
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    scaler: preprocessing.StandardScaler


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    resolved_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    return pd.read_excel(resolved_path)


def _reset_indices(*items):
    for item in items:
        item.index = range(item.shape[0])


def _standardize_first_n_columns(train_frame: pd.DataFrame, test_frame: pd.DataFrame, num_columns_to_standardize: int = STANDARDIZE_FIRST_N_COLUMNS):
    numeric_train = train_frame.iloc[:, 0:num_columns_to_standardize]
    numeric_test = test_frame.iloc[:, 0:num_columns_to_standardize]
    scaler = preprocessing.StandardScaler().fit(numeric_train)
    train_scaled = pd.DataFrame(scaler.transform(numeric_train), columns=numeric_train.columns)
    test_scaled = pd.DataFrame(scaler.transform(numeric_test), columns=numeric_test.columns)
    train_remaining = train_frame.iloc[:, num_columns_to_standardize:]
    test_remaining = test_frame.iloc[:, num_columns_to_standardize:]
    train_processed = pd.concat([train_scaled, train_remaining], axis=1)
    test_processed = pd.concat([test_scaled, test_remaining], axis=1)
    return train_processed, test_processed, scaler


def prepare_classification_data(df: pd.DataFrame) -> ClassificationDataBundle:
    features = df.iloc[:, CLASSIFICATION_FEATURE_START_INDEX:-1]
    target = df[CLASSIFICATION_TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=CLASSIFICATION_TEST_SIZE, random_state=CLASSIFICATION_SPLIT_RANDOM_STATE)
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    _reset_indices(x_train_resampled, x_test, y_train_resampled, y_test)
    x_train_processed, x_test_processed, scaler = _standardize_first_n_columns(x_train_resampled, x_test)
    return ClassificationDataBundle(x_train_resampled, x_test, y_train_resampled, y_test, x_train_processed, x_test_processed, scaler)


def prepare_regression_data(df: pd.DataFrame) -> RegressionDataBundle:
    target = df.iloc[:, REGRESSION_TARGET_COLUMN_INDEX]
    features = df.iloc[:, REGRESSION_FEATURE_START_INDEX:]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=REGRESSION_TEST_SIZE, random_state=REGRESSION_SPLIT_RANDOM_STATE)
    _reset_indices(x_train, x_test, y_train, y_test)
    x_train_processed, x_test_processed, scaler = _standardize_first_n_columns(x_train, x_test)
    return RegressionDataBundle(x_train, x_test, y_train, y_test, x_train_processed, x_test_processed, scaler)
