"""Preprocessing utilities for feature generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DATE_COLUMN,
    FEATURE_LAG_DAYS,
    FEATURE_ROLLING_WINDOW,
    GPR_COLUMN,
    OUTLIER_SIGMA_THRESHOLD,
    PRICE_GOLD_COLUMN,
    PRICE_SILVER_COLUMN,
)
from src.data.loader import load_raw_data
from src.validation import ensure_columns, normalize_date_column


FEATURE_OUTPUT_PATH = Path(__file__).resolve().parent / "processed" / "features.csv"


def clean_data(df: pd.DataFrame, sigma_threshold: float = OUTLIER_SIGMA_THRESHOLD) -> pd.DataFrame:
    """Fill missing values and remove extreme outliers."""
    frame = normalize_date_column(df)
    required_columns = [PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN, GPR_COLUMN]
    ensure_columns(frame, required_columns, "cleaning")

    frame = frame.copy()
    frame[required_columns] = frame[required_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.ffill()

    for column in required_columns:
        std_dev = frame[column].std()
        if pd.isna(std_dev) or std_dev == 0:
            continue
        mean_value = frame[column].mean()
        lower_bound = mean_value - sigma_threshold * std_dev
        upper_bound = mean_value + sigma_threshold * std_dev
        frame = frame[frame[column].between(lower_bound, upper_bound)]

    return frame.reset_index(drop=True)


def feature_engineering(
    df: pd.DataFrame,
    rolling_window: int = FEATURE_ROLLING_WINDOW,
    lag_days: tuple[int, ...] = FEATURE_LAG_DAYS,
) -> pd.DataFrame:
    """Create lag features and rolling statistics for time series analysis."""
    frame = normalize_date_column(df)
    required_columns = [PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN, GPR_COLUMN]
    ensure_columns(frame, required_columns, "feature engineering")

    frame = frame.copy()
    for lag_day in lag_days:
        frame[f"gold_lag{lag_day}"] = frame[PRICE_GOLD_COLUMN].shift(lag_day)
        frame[f"silver_lag{lag_day}"] = frame[PRICE_SILVER_COLUMN].shift(lag_day)

    frame[f"{GPR_COLUMN}_lag7"] = frame[GPR_COLUMN].shift(7)
    frame[f"gold_roll_mean{rolling_window}"] = frame[PRICE_GOLD_COLUMN].rolling(rolling_window).mean()
    frame[f"gold_volatility{rolling_window}"] = frame[PRICE_GOLD_COLUMN].rolling(rolling_window).std()
    frame[f"silver_roll_mean{rolling_window}"] = frame[PRICE_SILVER_COLUMN].rolling(rolling_window).mean()
    frame[f"silver_volatility{rolling_window}"] = frame[PRICE_SILVER_COLUMN].rolling(rolling_window).std()

    engineered = frame.dropna().reset_index(drop=True)
    return engineered.sort_values(DATE_COLUMN).reset_index(drop=True)


def save_features(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist a processed dataframe to CSV and return the output path."""
    output_path = Path(path) if path is not None else FEATURE_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_preprocessing_pipeline(
    raw_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw data, clean it, engineer features, and save the result."""
    raw_data = load_raw_data(raw_dir=raw_dir)
    cleaned = clean_data(raw_data)
    features = feature_engineering(cleaned)
    save_features(features, path=output_path)
    return features