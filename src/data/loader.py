"""Raw dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATE_COLUMN
from src.validation import normalize_date_column


DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "raw"


def load_raw_data(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load raw CSV files and merge them into a single dataframe."""
    data_directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    required_files = [
        "Gold_Spot_Price_Daily.csv",
        "Silver_Spot_Price_Daily.csv",
        "Geopolitical_Risk_Index_Daily.csv",
    ]

    missing_files = [file_name for file_name in required_files if not (data_directory / file_name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing raw data file(s): "
            f"{', '.join(missing_files)}. Expected in directory: {data_directory}"
        )

    gold = pd.read_csv(
        data_directory / "Gold_Spot_Price_Daily.csv",
        parse_dates=["DATE"],
        dayfirst=True,
    )
    silver = pd.read_csv(
        data_directory / "Silver_Spot_Price_Daily.csv",
        parse_dates=["DATE"],
        dayfirst=True,
    )
    gpr = pd.read_csv(
        data_directory / "Geopolitical_Risk_Index_Daily.csv",
        parse_dates=["DATE"],
        dayfirst=True,
    )

    gold = gold[["DATE", "GOLD_PRICE"]].rename(columns={"GOLD_PRICE": "Price_gold"})
    silver = silver[["DATE", "SILVER_PRICE"]].rename(columns={"SILVER_PRICE": "Price_silver"})
    gpr = gpr[["DATE", "GPRD", "EVENT"]].rename(columns={"GPRD": "GPR"})

    gold["Price_gold"] = pd.to_numeric(gold["Price_gold"], errors="coerce")
    silver["Price_silver"] = pd.to_numeric(silver["Price_silver"], errors="coerce")
    gpr["GPR"] = pd.to_numeric(gpr["GPR"], errors="coerce")
    gpr["EVENT"] = gpr["EVENT"].fillna("").astype(str)

    merged = gold.merge(silver, on="DATE", how="inner")
    merged = merged.merge(gpr, on="DATE", how="inner")
    merged = normalize_date_column(merged)
    return merged.sort_values(DATE_COLUMN).reset_index(drop=True)