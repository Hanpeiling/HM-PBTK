"""Data loading and preprocessing utilities.

This module reproduces the notebook workflow explicitly:
1. Read the Excel file.
2. Split features and target.
3. Reset indices after train/test split.
4. One-hot encode the last two categorical columns.
5. Standardize the remaining numerical columns.
6. Concatenate the processed numerical and categorical features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FEATURE_START_INDEX, RANDOM_STATE, TARGET_COLUMN_INDEX, TEST_SIZE


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    raw_train_features: pd.DataFrame
    raw_test_features: pd.DataFrame
    numerical_columns: List[str]
    categorical_columns: List[str]
    encoded_columns: List[str]



def load_excel_data(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_excel(data_path)



def split_features_and_target(data: pd.DataFrame):
    y = data.iloc[:, TARGET_COLUMN_INDEX]
    x = data.iloc[:, FEATURE_START_INDEX:]
    return x, y



def _reset_indices(*objects):
    for obj in objects:
        obj.index = range(obj.shape[0])



def build_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(categories="auto", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")



def _rename_encoded_columns(column_names: List[str]) -> List[str]:
    renamed = []
    for name in column_names:
        if name.startswith("Family_"):
            renamed.append(name.replace("Family_", "family _", 1))
        else:
            renamed.append(name)
    return renamed



def prepare_data(data_path: str | Path) -> PreparedData:
    data = load_excel_data(data_path)
    x, y = split_features_and_target(data)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    _reset_indices(Xtrain, Xtest, Ytrain, Ytest)

    categorical_columns = Xtrain.columns[-2:].tolist()
    numerical_columns = Xtrain.columns[:-2].tolist()

    x_categorical_train = pd.DataFrame(Xtrain.iloc[:, -2:])
    encoder = build_encoder().fit(x_categorical_train)
    encoded_column_names = _rename_encoded_columns(
        list(encoder.get_feature_names_out(categorical_columns))
    )
    encoded_train = pd.DataFrame(
        encoder.transform(x_categorical_train),
        columns=encoded_column_names,
    )

    x_numerical_train = Xtrain.iloc[:, 0:-2].copy()
    scaler = StandardScaler().fit(x_numerical_train)
    standardized_train = pd.DataFrame(
        scaler.transform(x_numerical_train),
        columns=numerical_columns,
    )

    x_train_processed = pd.concat([standardized_train, encoded_train], axis=1)

    x_categorical_test = pd.DataFrame(Xtest.iloc[:, -2:])
    encoded_test = pd.DataFrame(
        encoder.transform(x_categorical_test),
        columns=encoded_column_names,
    )

    x_numerical_test = pd.DataFrame(Xtest.iloc[:, 0:-2])
    standardized_test = pd.DataFrame(
        scaler.transform(x_numerical_test),
        columns=numerical_columns,
    )

    x_test_processed = pd.concat([standardized_test, encoded_test], axis=1)

    return PreparedData(
        X_train=x_train_processed,
        X_test=x_test_processed,
        y_train=np.ravel(pd.DataFrame(Ytrain)),
        y_test=np.ravel(pd.DataFrame(Ytest)),
        raw_train_features=Xtrain,
        raw_test_features=Xtest,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        encoded_columns=encoded_column_names,
    )
