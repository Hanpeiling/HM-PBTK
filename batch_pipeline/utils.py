from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def ensure_required_files(base_dir: Path, relative_paths: Sequence[str]) -> None:
    missing = [str(base_dir / rel) for rel in relative_paths if not (base_dir / rel).exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "The following files were not found. Please place them in the same directory structure as main.py.：\n"
            f"{joined}"
        )


def merge_columns_by_smiles(target_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    merged = target_df.copy()
    for index, row in merged.iterrows():
        smiles = row['SMILES']
        matching_rows = reference_df[reference_df['SMILES'] == smiles]
        if matching_rows.empty:
            continue
        for col in reference_df.columns:
            if col == 'SMILES':
                continue
            merged.at[index, col] = ', '.join(matching_rows[col].astype(str).values)
    return merged


def clean_numeric_string(value):
   
    if isinstance(value, str):
        if ',' in value:
            cleaned = value.split(',')[0].strip()
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        try:
            return float(value.strip())
        except ValueError:
            return np.nan
    return value


def clean_dataframe(df: pd.DataFrame, skip_columns: Iterable[str] | None = None) -> pd.DataFrame:
    skip_columns = set(skip_columns or [])
    cleaned = df.copy()
    for col in cleaned.columns:
        if col in skip_columns:
            continue
        if cleaned[col].dtype == 'object':
            cleaned[col] = cleaned[col].apply(clean_numeric_string)
    return cleaned


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')
