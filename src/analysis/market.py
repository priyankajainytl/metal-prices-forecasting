"""Reusable market analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import DATE_COLUMN, EVENT_COLUMN, PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN
from src.validation import ensure_columns, normalize_date_column


@dataclass(frozen=True)
class EventImpactSummary:
    """Container for event-day versus non-event-day analysis."""

    summary_table: pd.DataFrame
    metrics: dict[str, float | int]


def compute_yearly_average(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Compute yearly average price for the given metal."""
    frame = normalize_date_column(df)
    ensure_columns(frame, [DATE_COLUMN, price_column], f"yearly average for {price_column}")

    yearly_frame = frame[[DATE_COLUMN, price_column]].copy()
    yearly_frame["Year"] = yearly_frame[DATE_COLUMN].dt.year
    return yearly_frame.groupby("Year", as_index=False)[price_column].mean()


def compute_rolling_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling volatility for gold and silver returns."""
    frame = normalize_date_column(df)
    ensure_columns(frame, [DATE_COLUMN, PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN], "rolling volatility")

    volatility_frame = frame[[DATE_COLUMN, PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN]].dropna().copy()
    volatility_frame["Gold_Return_%"] = volatility_frame[PRICE_GOLD_COLUMN].pct_change() * 100
    volatility_frame["Silver_Return_%"] = volatility_frame[PRICE_SILVER_COLUMN].pct_change() * 100
    volatility_frame["Gold_Volatility_%"] = volatility_frame["Gold_Return_%"].rolling(window).std()
    volatility_frame["Silver_Volatility_%"] = volatility_frame["Silver_Return_%"].rolling(window).std()
    return volatility_frame


def summarize_market_metrics(df: pd.DataFrame, volatility_df: pd.DataFrame) -> dict[str, float]:
    """Return current headline price and volatility metrics."""
    frame = normalize_date_column(df)
    ensure_columns(frame, [PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN], "market metrics")

    latest_gold = frame.dropna(subset=[PRICE_GOLD_COLUMN]).iloc[-1]
    latest_silver = frame.dropna(subset=[PRICE_SILVER_COLUMN]).iloc[-1]

    metrics = {
        "gold_latest": float(latest_gold[PRICE_GOLD_COLUMN]),
        "gold_max": float(frame[PRICE_GOLD_COLUMN].max()),
        "silver_latest": float(latest_silver[PRICE_SILVER_COLUMN]),
        "silver_max": float(frame[PRICE_SILVER_COLUMN].max()),
    }

    latest_volatility = volatility_df.dropna(subset=["Gold_Volatility_%", "Silver_Volatility_%"])
    if not latest_volatility.empty:
        last_row = latest_volatility.iloc[-1]
        metrics["gold_volatility"] = float(last_row["Gold_Volatility_%"])
        metrics["silver_volatility"] = float(last_row["Silver_Volatility_%"])

    return metrics


def compute_event_impact_summary(df: pd.DataFrame) -> EventImpactSummary:
    """Compare average prices and returns on event versus non-event days."""
    frame = normalize_date_column(df)
    ensure_columns(
        frame,
        [DATE_COLUMN, PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN, EVENT_COLUMN],
        "event impact summary",
    )

    impact_frame = frame[[DATE_COLUMN, PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN, EVENT_COLUMN]].copy()
    impact_frame["is_event_day"] = impact_frame[EVENT_COLUMN].fillna("").astype(str).str.strip().ne("")
    impact_frame["Gold_Return_%"] = impact_frame[PRICE_GOLD_COLUMN].pct_change() * 100
    impact_frame["Silver_Return_%"] = impact_frame[PRICE_SILVER_COLUMN].pct_change() * 100

    grouped = impact_frame.groupby("is_event_day").agg(
        avg_gold_price=(PRICE_GOLD_COLUMN, "mean"),
        avg_silver_price=(PRICE_SILVER_COLUMN, "mean"),
        avg_gold_return=("Gold_Return_%", "mean"),
        avg_silver_return=("Silver_Return_%", "mean"),
        days=(DATE_COLUMN, "count"),
    )

    if True not in grouped.index or False not in grouped.index:
        return EventImpactSummary(summary_table=pd.DataFrame(), metrics={})

    event_stats = grouped.loc[True]
    non_event_stats = grouped.loc[False]
    summary_table = pd.DataFrame(
        [
            {
                "Group": "Event Days",
                "Avg Gold Price": event_stats["avg_gold_price"],
                "Avg Silver Price": event_stats["avg_silver_price"],
                "Avg Gold Daily Return %": event_stats["avg_gold_return"],
                "Avg Silver Daily Return %": event_stats["avg_silver_return"],
            },
            {
                "Group": "Non-Event Days",
                "Avg Gold Price": non_event_stats["avg_gold_price"],
                "Avg Silver Price": non_event_stats["avg_silver_price"],
                "Avg Gold Daily Return %": non_event_stats["avg_gold_return"],
                "Avg Silver Daily Return %": non_event_stats["avg_silver_return"],
            },
        ]
    )

    metrics = {
        "event_days": int(event_stats["days"]),
        "non_event_days": int(non_event_stats["days"]),
        "gold_return_impact": float(event_stats["avg_gold_return"] - non_event_stats["avg_gold_return"]),
        "silver_return_impact": float(event_stats["avg_silver_return"] - non_event_stats["avg_silver_return"]),
    }
    return EventImpactSummary(summary_table=summary_table, metrics=metrics)