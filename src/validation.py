"""Validation helpers shared across modules."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.config import DATE_COLUMN, RAW_DATE_COLUMN


def ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
    """Raise a helpful error when expected columns are missing."""
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for {context}: {missing_columns}")


def normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a canonical Date column in datetime format."""
    normalized = frame.copy()
    if DATE_COLUMN not in normalized.columns and RAW_DATE_COLUMN in normalized.columns:
        normalized = normalized.rename(columns={RAW_DATE_COLUMN: DATE_COLUMN})

    ensure_columns(normalized, [DATE_COLUMN], "date normalization")
    normalized[DATE_COLUMN] = pd.to_datetime(normalized[DATE_COLUMN], errors="coerce")
    normalized = normalized.dropna(subset=[DATE_COLUMN])
    return normalized.sort_values(DATE_COLUMN).reset_index(drop=True)