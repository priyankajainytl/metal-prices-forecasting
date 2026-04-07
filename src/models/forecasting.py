"""ARIMA and SARIMAX forecasting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import DATE_COLUMN, EVENT_COLUMN, MIN_FORECAST_POINTS, MIN_TRAIN_POINTS
from src.validation import ensure_columns, normalize_date_column


@dataclass(frozen=True)
class ForecastResult:
    """Artifacts needed by the Streamlit view for forecast display."""

    model_label: str
    target_series: pd.Series
    backtest_forecast: pd.Series
    test_slice: pd.Series
    future_dates: pd.DatetimeIndex
    future_forecast: pd.Series
    backtest_mae: float
    event_coef: float
    event_p_value: float


def prepare_target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Build a daily time series ready for ARIMA-style modeling."""
    frame = normalize_date_column(df)
    ensure_columns(frame, [DATE_COLUMN, target_column], f"forecast target preparation for {target_column}")

    return (
        frame[[DATE_COLUMN, target_column]]
        .dropna()
        .drop_duplicates(subset=[DATE_COLUMN])
        .set_index(DATE_COLUMN)[target_column]
        .asfreq("D")
        .interpolate(method="time")
        .dropna()
    )


def _prepare_event_series(df: pd.DataFrame, target_index: pd.Index) -> pd.Series | None:
    """Convert event text into a daily binary exogenous feature."""
    if EVENT_COLUMN not in df.columns:
        return None

    frame = normalize_date_column(df)
    event_series = (
        frame[[DATE_COLUMN, EVENT_COLUMN]]
        .drop_duplicates(subset=[DATE_COLUMN])
        .assign(event_flag=lambda current: current[EVENT_COLUMN].fillna("").astype(str).str.strip().ne("").astype(int))
        .set_index(DATE_COLUMN)["event_flag"]
        .asfreq("D")
        .fillna(0)
        .reindex(target_index, fill_value=0)
    )
    return event_series


def run_forecast(
    df: pd.DataFrame,
    target_column: str,
    order: tuple[int, int, int],
    forecast_horizon: int,
    train_fraction: float,
    use_event_exog: bool,
) -> ForecastResult:
    """Fit a forecast model and return all display artifacts."""
    target_series = prepare_target_series(df, target_column)
    if len(target_series) < MIN_FORECAST_POINTS:
        raise ValueError(
            f"Not enough data points to fit ARIMA reliably. Need at least {MIN_FORECAST_POINTS} daily observations."
        )

    split_index = int(len(target_series) * train_fraction)
    split_index = max(MIN_TRAIN_POINTS, min(split_index, len(target_series) - 1))
    train_series = target_series.iloc[:split_index]
    test_series = target_series.iloc[split_index:]
    backtest_steps = min(len(test_series), forecast_horizon)
    event_series = _prepare_event_series(df, target_series.index)

    model_label = f"ARIMA({order[0]},{order[1]},{order[2]})"
    event_coef = float("nan")
    event_p_value = float("nan")

    if use_event_exog and event_series is not None:
        train_exog = event_series.iloc[:split_index].to_frame(name="event_flag")
        test_exog = event_series.iloc[split_index:].to_frame(name="event_flag")
        backtest_model = SARIMAX(
            train_series,
            exog=train_exog,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        if backtest_steps > 0:
            backtest_forecast = backtest_model.forecast(steps=backtest_steps, exog=test_exog.iloc[:backtest_steps])
            test_slice = test_series.iloc[:backtest_steps]
            backtest_mae = float((test_slice - backtest_forecast).abs().mean())
        else:
            backtest_forecast = pd.Series(dtype="float64")
            test_slice = pd.Series(dtype="float64")
            backtest_mae = float("nan")

        full_exog = event_series.to_frame(name="event_flag")
        full_model = SARIMAX(
            target_series,
            exog=full_exog,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        model_label = f"ARIMAX({order[0]},{order[1]},{order[2]})"
        event_coef = float(full_model.params.get("event_flag", float("nan")))
        event_p_value = float(full_model.pvalues.get("event_flag", float("nan")))
    else:
        backtest_model = ARIMA(train_series, order=order).fit()
        if backtest_steps > 0:
            backtest_forecast = backtest_model.forecast(steps=backtest_steps)
            test_slice = test_series.iloc[:backtest_steps]
            backtest_mae = float((test_slice - backtest_forecast).abs().mean())
        else:
            backtest_forecast = pd.Series(dtype="float64")
            test_slice = pd.Series(dtype="float64")
            backtest_mae = float("nan")
        full_model = ARIMA(target_series, order=order).fit()

    future_dates = pd.date_range(start=target_series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    if use_event_exog and event_series is not None:
        future_exog = pd.DataFrame({"event_flag": [0] * forecast_horizon}, index=future_dates)
        future_forecast = full_model.forecast(steps=forecast_horizon, exog=future_exog)
    else:
        future_forecast = full_model.forecast(steps=forecast_horizon)

    return ForecastResult(
        model_label=model_label,
        target_series=target_series,
        backtest_forecast=backtest_forecast,
        test_slice=test_slice,
        future_dates=future_dates,
        future_forecast=future_forecast,
        backtest_mae=backtest_mae,
        event_coef=event_coef,
        event_p_value=event_p_value,
    )