from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

DAY_PATTERN = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\b",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:-\s*\d{1,2}(?::\d{2})?\s*)?(?:am|pm))\b",
    re.IGNORECASE,
)
OUTWARD_POSTCODE_PATTERN = re.compile(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)")
DAY_MAP = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}


def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def coerce_datetimes(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def normalize_blank_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    replacements = {"": np.nan, "nan": np.nan, "None": np.nan}
    for column in columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .replace(replacements)
                .replace(r"^\s*$", np.nan, regex=True)
            )
    return df


def drop_empty_and_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            if not ((series.isna()) | (series == 0)).all():
                keep_columns.append(column)
        elif not series.isna().all():
            keep_columns.append(column)
    return df[keep_columns].copy()


def standardize_customer_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Customer Email" in df.columns:
        df["Customer Email"] = (
            df["Customer Email"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )
    if "Customer Name" in df.columns:
        df["Customer Name"] = (
            df["Customer Name"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )
    return df


def build_client_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    email_series = df.get("Customer Email", pd.Series(index=df.index, dtype="object"))
    name_series = df.get("Customer Name", pd.Series(index=df.index, dtype="object"))
    df["client_id"] = email_series.fillna(name_series)
    return df


def extract_day(text: object) -> str | None:
    if pd.isna(text):
        return None
    match = DAY_PATTERN.search(str(text))
    if not match:
        return None
    return DAY_MAP.get(match.group(1).lower())


def extract_time(text: object) -> str | None:
    if pd.isna(text):
        return None
    match = TIME_PATTERN.search(str(text))
    if not match:
        return None
    value = match.group(1).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s*(am|pm)$", r" \1", value)


def extract_class_type(text: object) -> str | None:
    if pd.isna(text):
        return None
    return "ASC" if re.search(r"\bASC\b", str(text), flags=re.IGNORECASE) else "ACD"


def extract_outward_postcode(series: pd.Series, valid_postcodes: Iterable[str]) -> pd.Series:
    cleaned = series.astype(str).str.upper().str.replace(" ", "", regex=False)
    candidate = cleaned.str.extract(OUTWARD_POSTCODE_PATTERN, expand=False)
    return candidate.where(candidate.isin(set(valid_postcodes)))


def propagate_first_non_null_by_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    non_null_lookup = (
        df[[group_col, target_col]]
        .dropna(subset=[group_col, target_col])
        .drop_duplicates(subset=[group_col], keep="first")
        .set_index(group_col)[target_col]
    )
    missing_mask = df[target_col].isna() & df[group_col].notna()
    df.loc[missing_mask, target_col] = df.loc[missing_mask, group_col].map(non_null_lookup)
    return df


def enrich_class_columns(df: pd.DataFrame, source_col: str, class_type_col: str = "class_type") -> pd.DataFrame:
    df = df.copy()
    df["class_day"] = df[source_col].map(extract_day)
    df["class_time"] = df[source_col].map(extract_time)
    df[class_type_col] = df[source_col].map(extract_class_type)
    return df
