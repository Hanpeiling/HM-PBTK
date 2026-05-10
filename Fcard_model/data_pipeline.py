from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_COLUMN_SLICE,
    FEATURE_START_COLUMN_INDEX,
    NUMERICAL_COLUMN_SLICE,
    ONE_HOT_COLUMN_NAMES,
    TARGET_COLUMN_INDEX,
    TEST_SIZE,
    TRAIN_TEST_SPLIT_RANDOM_STATE,
)


@dataclass
class PreparedData:
    raw_data: pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    encoder: OneHotEncoder
    scaler: StandardScaler


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    return pd.read_excel(data_path)


def split_raw_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = data.iloc[:, TARGET_COLUMN_INDEX]
    x = data.iloc[:, FEATURE_START_COLUMN_INDEX:]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=TRAIN_TEST_SPLIT_RANDOM_STATE,
    )

    for frame in (x_train, x_test):
        frame.index = range(frame.shape[0])
    for series in (y_train, y_test):
        series.index = range(series.shape[0])

    return x_train, x_test, y_train, y_test


def _build_encoder(x_train: pd.DataFrame) -> OneHotEncoder:
    categorical_train = pd.DataFrame(x_train.iloc[:, CATEGORICAL_COLUMN_SLICE])
    encoder = OneHotEncoder(categories="auto")
    encoder.fit(categorical_train)
    return encoder


def _transform_categorical(encoder: OneHotEncoder, x_part: pd.DataFrame) -> pd.DataFrame:
    categorical_part = pd.DataFrame(x_part.iloc[:, CATEGORICAL_COLUMN_SLICE])
    encoded = encoder.transform(categorical_part).toarray()
    return pd.DataFrame(encoded, columns=ONE_HOT_COLUMN_NAMES)


def _build_scaler(x_train: pd.DataFrame) -> tuple[StandardScaler, list[str]]:
    numerical_train = x_train.iloc[:, NUMERICAL_COLUMN_SLICE]
    scaler = StandardScaler().fit(numerical_train)
    return scaler, list(numerical_train.columns)


def _transform_numerical(
    scaler: StandardScaler,
    x_part: pd.DataFrame,
    numeric_column_names: list[str],
) -> pd.DataFrame:
    numerical_part = pd.DataFrame(x_part.iloc[:, NUMERICAL_COLUMN_SLICE])
    scaled = scaler.transform(numerical_part)
    return pd.DataFrame(scaled, columns=numeric_column_names)


def prepare_data(data_path: str | Path) -> PreparedData:
    raw_data = load_raw_data(data_path)
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = split_raw_data(raw_data)

    encoder = _build_encoder(x_train_raw)
    scaler, numeric_column_names = _build_scaler(x_train_raw)

    x_train_processed = pd.concat(
        [
            _transform_numerical(scaler, x_train_raw, numeric_column_names),
            _transform_categorical(encoder, x_train_raw),
        ],
        axis=1,
    )
    x_test_processed = pd.concat(
        [
            _transform_numerical(scaler, x_test_raw, numeric_column_names),
            _transform_categorical(encoder, x_test_raw),
        ],
        axis=1,
    )

    y_train = np.ravel(pd.DataFrame(y_train_raw))
    y_test = np.ravel(pd.DataFrame(y_test_raw))

    return PreparedData(
        raw_data=raw_data,
        x_train=x_train_processed,
        x_test=x_test_processed,
        y_train=y_train,
        y_test=y_test,
        encoder=encoder,
        scaler=scaler,
    )
