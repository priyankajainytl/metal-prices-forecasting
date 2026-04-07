"""Matplotlib chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_yearly_average(yearly_frame: pd.DataFrame, price_column: str, title: str, color: str, figsize: tuple[int, int]):
    """Plot yearly average prices with point annotations."""
    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(yearly_frame["Year"], yearly_frame[price_column], marker="o", color=color)
    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel("Average Price")
    axis.grid(alpha=0.25)
    axis.tick_params(axis="x", labelrotation=45)

    for index, (_, row) in enumerate(yearly_frame.iterrows()):
        offset_y = 10 if index % 2 == 0 else -14
        axis.annotate(
            f"{row[price_column]:,.1f}",
            (row["Year"], row[price_column]),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
        )
    return figure


def plot_price_trends(df: pd.DataFrame, selection: str):
    """Plot price trends for gold, silver, or both."""
    figure, gold_axis = plt.subplots(figsize=(12, 6))

    if selection == "Both":
        silver_axis = gold_axis.twinx()
        gold_line = gold_axis.plot(df["Date"], df["Price_gold"], label="Gold", color="tab:orange")
        silver_line = silver_axis.plot(df["Date"], df["Price_silver"], label="Silver", color="tab:blue")
        gold_axis.set_xlabel("Date")
        gold_axis.set_ylabel("Gold Price", color="tab:orange")
        silver_axis.set_ylabel("Silver Price", color="tab:blue")
        lines = gold_line + silver_line
        labels = [line.get_label() for line in lines]
        gold_axis.legend(lines, labels, loc="upper left")

        latest = df.dropna(subset=["Price_gold", "Price_silver"]).iloc[-1]
        gold_axis.annotate(
            f"Latest: {latest['Price_gold']:,.2f}",
            (latest["Date"], latest["Price_gold"]),
            textcoords="offset points",
            xytext=(8, -12),
            color="tab:orange",
            fontsize=9,
            fontweight="bold",
        )
        silver_axis.annotate(
            f"Latest: {latest['Price_silver']:,.2f}",
            (latest["Date"], latest["Price_silver"]),
            textcoords="offset points",
            xytext=(8, 8),
            color="tab:blue",
            fontsize=9,
            fontweight="bold",
        )
    elif selection == "Gold":
        gold_axis.plot(df["Date"], df["Price_gold"], label="Gold", color="tab:orange", linewidth=2)
        gold_axis.set_xlabel("Date")
        gold_axis.set_ylabel("Gold Price", color="tab:orange")
        gold_axis.legend(loc="upper left")
        latest = df.dropna(subset=["Price_gold"]).iloc[-1]
        gold_axis.annotate(
            f"Latest: {latest['Price_gold']:,.2f}",
            (latest["Date"], latest["Price_gold"]),
            textcoords="offset points",
            xytext=(8, -12),
            color="tab:orange",
            fontsize=9,
            fontweight="bold",
        )
    else:
        gold_axis.plot(df["Date"], df["Price_silver"], label="Silver", color="tab:blue", linewidth=2)
        gold_axis.set_xlabel("Date")
        gold_axis.set_ylabel("Silver Price", color="tab:blue")
        gold_axis.legend(loc="upper left")
        latest = df.dropna(subset=["Price_silver"]).iloc[-1]
        gold_axis.annotate(
            f"Latest: {latest['Price_silver']:,.2f}",
            (latest["Date"], latest["Price_silver"]),
            textcoords="offset points",
            xytext=(8, -12),
            color="tab:blue",
            fontsize=9,
            fontweight="bold",
        )

    figure.tight_layout()
    return figure


def plot_rolling_volatility(volatility_frame: pd.DataFrame, window: int):
    """Plot rolling volatility for gold and silver."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        volatility_frame["Date"],
        volatility_frame["Gold_Volatility_%"],
        color="tab:orange",
        label=f"Gold {window}D Rolling Volatility",
    )
    axis.plot(
        volatility_frame["Date"],
        volatility_frame["Silver_Volatility_%"],
        color="tab:blue",
        label=f"Silver {window}D Rolling Volatility",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Volatility (% Std. Dev. of Daily Returns)")
    axis.set_title("Gold vs Silver Volatility Over Time")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_forecast(result, metal_name: str, history_display: int | str):
    """Plot actual history, backtest forecast, and future forecast."""
    figure, axis = plt.subplots(figsize=(12, 6))
    history_window = len(result.target_series) if history_display == "Full" else min(int(history_display), len(result.target_series))
    axis.plot(
        result.target_series.index[-history_window:],
        result.target_series.iloc[-history_window:],
        label=f"{metal_name} Actual",
        color="tab:blue",
    )
    if len(result.backtest_forecast) > 0:
        axis.plot(
            result.test_slice.index,
            result.backtest_forecast,
            label="Backtest Forecast",
            linestyle="--",
            color="tab:orange",
        )
    axis.plot(
        result.future_dates,
        result.future_forecast.values,
        label="Future Forecast",
        linestyle="--",
        color="tab:green",
    )
    axis.set_title(f"{metal_name} {result.model_label} Forecast")
    axis.set_xlabel("Date")
    axis.set_ylabel("Price")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_event_study(
    gold_event_study: pd.DataFrame,
    silver_event_study: pd.DataFrame,
    avg_gold_event_study: pd.DataFrame,
    avg_silver_event_study: pd.DataFrame,
):
    """Plot event-study trajectories for gold and silver."""
    figure, (gold_axis, silver_axis) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    for _, event_slice in gold_event_study.groupby("event_name"):
        gold_axis.plot(event_slice["relative_day"], event_slice["return_vs_event_%"], alpha=0.25, linewidth=1)
    gold_axis.plot(
        avg_gold_event_study["relative_day"],
        avg_gold_event_study["return_vs_event_%"],
        color="tab:orange",
        linewidth=3,
        label="Average",
    )
    gold_axis.axvline(0, color="black", linestyle="--", linewidth=1)
    gold_axis.axhline(0, color="grey", linestyle=":", linewidth=1)
    gold_axis.set_title("Gold Around Events")
    gold_axis.set_xlabel("Relative Trading Day")
    gold_axis.set_ylabel("Return vs Event Day (%)")
    gold_axis.legend(loc="upper left")

    for _, event_slice in silver_event_study.groupby("event_name"):
        silver_axis.plot(event_slice["relative_day"], event_slice["return_vs_event_%"], alpha=0.25, linewidth=1)
    silver_axis.plot(
        avg_silver_event_study["relative_day"],
        avg_silver_event_study["return_vs_event_%"],
        color="tab:blue",
        linewidth=3,
        label="Average",
    )
    silver_axis.axvline(0, color="black", linestyle="--", linewidth=1)
    silver_axis.axhline(0, color="grey", linestyle=":", linewidth=1)
    silver_axis.set_title("Silver Around Events")
    silver_axis.set_xlabel("Relative Trading Day")
    silver_axis.set_ylabel("Return vs Event Day (%)")
    silver_axis.legend(loc="upper left")
    figure.tight_layout()
    return figure