from __future__ import annotations

from typing import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when required data contracts are not met."""



def require_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise DataValidationError(
            f"{dataset_name} is missing required columns: {missing}"
        )



def validate_non_empty(df: pd.DataFrame, dataset_name: str) -> None:
    if df.empty:
        raise DataValidationError(f"{dataset_name} is empty.")



def validate_unique_key(df: pd.DataFrame, key_columns: list[str], dataset_name: str) -> None:
    duplicate_count = int(df.duplicated(subset=key_columns).sum())
    if duplicate_count > 0:
        raise DataValidationError(
            f"{dataset_name} has {duplicate_count} duplicate rows for key {key_columns}."
        )



def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "null_count": [int(df[c].isna().sum()) for c in df.columns],
        "null_pct": [float(df[c].isna().mean()) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return summary.sort_values(["null_pct", "null_count"], ascending=False).reset_index(drop=True)
