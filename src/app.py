from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.analysis.events import (
    build_event_impact_table,
    build_event_reference,
    build_event_study_frame,
    filter_event_studies,
    summarize_event_prices,
    summarize_event_returns,
)
from src.analysis.market import (
    compute_event_impact_summary,
    compute_rolling_volatility,
    compute_yearly_average,
    summarize_market_metrics,
)
from src.charts import (
    plot_event_study,
    plot_forecast,
    plot_price_trends,
    plot_rolling_volatility,
    plot_yearly_average,
)
from src.config import (
    DEFAULT_ARIMA_ORDER,
    DEFAULT_EVENT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VOLATILITY_WINDOW,
    EVENT_COLUMN,
    PRICE_GOLD_COLUMN,
    PRICE_SILVER_COLUMN,
)
from src.data.loader import load_raw_data
from src.models.forecasting import run_forecast


st.set_page_config(page_title="Precious Metal Analysis", layout="wide")


@st.cache_data(show_spinner=False)
def get_dashboard_data() -> pd.DataFrame:
    """Load dashboard data once per Streamlit session."""
    return load_raw_data()


def render_yearly_analysis(df: pd.DataFrame) -> None:
    """Render yearly average charts for gold and silver."""
    st.header("1. Yearly Average Analysis")
    st.subheader("Yearly Average Price Analysis")

    silver_average = compute_yearly_average(df, PRICE_SILVER_COLUMN)
    gold_average = compute_yearly_average(df, PRICE_GOLD_COLUMN)

    year_col1, year_col2 = st.columns([1, 1.7])
    with year_col1:
        st.pyplot(
            plot_yearly_average(
                silver_average,
                PRICE_SILVER_COLUMN,
                title="Average Silver Price per Year",
                color="tab:blue",
                figsize=(10, 5),
            )
        )
    with year_col2:
        st.pyplot(
            plot_yearly_average(
                gold_average,
                PRICE_GOLD_COLUMN,
                title="Average Gold Price per Year",
                color="tab:orange",
                figsize=(14, 6),
            )
        )


def render_market_overview(df: pd.DataFrame) -> None:
    """Render current price trends and rolling volatility."""
    st.header("2. Market Overview Dashboard")
    st.subheader("Gold & Silver Price Trends")

    price_trend_selection = st.selectbox("Display", ["Both", "Gold", "Silver"], index=0, key="price_trend_select")
    st.pyplot(plot_price_trends(df, price_trend_selection), width="stretch")

    st.subheader("Rolling Volatility of Daily Returns")
    vol_window = st.slider(
        "Volatility window (days)",
        min_value=7,
        max_value=120,
        value=DEFAULT_VOLATILITY_WINDOW,
        step=1,
    )
    volatility_df = compute_rolling_volatility(df, vol_window)
    st.pyplot(plot_rolling_volatility(volatility_df, vol_window), width="stretch")

    metrics = summarize_market_metrics(df, volatility_df)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Gold Latest", f"{metrics['gold_latest']:,.2f}")
    metric_col2.metric("Gold Max", f"{metrics['gold_max']:,.2f}")
    metric_col3.metric("Silver Latest", f"{metrics['silver_latest']:,.2f}")
    metric_col4.metric("Silver Max", f"{metrics['silver_max']:,.2f}")

    if "gold_volatility" in metrics and "silver_volatility" in metrics:
        volatility_col1, volatility_col2 = st.columns(2)
        volatility_col1.metric("Gold Latest Volatility", f"{metrics['gold_volatility']:.3f}%")
        volatility_col2.metric("Silver Latest Volatility", f"{metrics['silver_volatility']:.3f}%")


def render_forecasting(df: pd.DataFrame) -> None:
    """Render ARIMA or ARIMAX forecasts."""
    st.header("3. ARIMA Forecasting")
    st.subheader("Forecast Future Prices Using ARIMA")

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    selected_metal = control_col1.selectbox(
        "Metal",
        ["Gold", "Silver"],
        index=0,
        help="Select the target series to forecast. Gold usually behaves as a safer asset, while silver is often more cyclical.",
    )
    arima_p = control_col2.number_input(
        "p",
        min_value=0,
        max_value=5,
        value=DEFAULT_ARIMA_ORDER[0],
        step=1,
        help="Autoregressive order: number of past values used by ARIMA.",
    )
    arima_d = control_col3.number_input(
        "d",
        min_value=0,
        max_value=2,
        value=DEFAULT_ARIMA_ORDER[1],
        step=1,
        help="Differencing order: how many times the series is differenced to stabilize trend/non-stationarity.",
    )
    arima_q = control_col4.number_input(
        "q",
        min_value=0,
        max_value=5,
        value=DEFAULT_ARIMA_ORDER[2],
        step=1,
        help="Moving-average order: number of lagged forecast errors used by ARIMA.",
    )

    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        min_value=7,
        max_value=180,
        value=DEFAULT_FORECAST_HORIZON,
        step=1,
        help="How many future days to predict. Shorter horizons are usually more reliable than longer ones.",
    )
    train_fraction = st.slider(
        "Train split",
        min_value=0.70,
        max_value=0.95,
        value=DEFAULT_TRAIN_FRACTION,
        step=0.05,
        help="Fraction of history used for training. The remaining data is used for backtest error (MAE).",
    )
    use_event_exog = st.checkbox(
        "Include EVENT as exogenous feature (ARIMAX)",
        value=True,
        help="ARIMAX adds an external event signal (event day=1, non-event=0) on top of price history.",
    )
    history_display = st.select_slider(
        "History shown on forecast chart",
        options=[90, 180, 365, 730, "Full"],
        value=365,
        help="Controls how much historical context is visible on the chart without changing model training.",
    )

    st.caption(
        "Quick tuning guide: set d first, then try small p and q combinations, compare backtest MAE, and prefer the simplest setting when MAE is similar."
    )

    target_column = PRICE_GOLD_COLUMN if selected_metal == "Gold" else PRICE_SILVER_COLUMN
    try:
        forecast_result = run_forecast(
            df=df,
            target_column=target_column,
            order=(arima_p, arima_d, arima_q),
            forecast_horizon=forecast_horizon,
            train_fraction=train_fraction,
            use_event_exog=use_event_exog,
        )
    except ValueError as error:
        st.warning(str(error))
        return
    except Exception as error:
        st.error(f"ARIMA model failed with order ({arima_p},{arima_d},{arima_q}): {error}")
        return

    st.pyplot(plot_forecast(forecast_result, selected_metal, history_display), width="stretch")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        "Backtest MAE",
        f"{forecast_result.backtest_mae:,.2f}" if pd.notna(forecast_result.backtest_mae) else "N/A",
    )
    metric_col2.metric("Forecast Horizon", f"{forecast_horizon} days")
    if use_event_exog and pd.notna(forecast_result.event_coef):
        metric_col3.metric("Event Coefficient", f"{forecast_result.event_coef:,.4f}")
        if pd.notna(forecast_result.event_p_value):
            st.caption(
                f"Event coefficient p-value: {forecast_result.event_p_value:.4f} (lower means stronger statistical evidence)."
            )
        st.caption("Future forecast assumes no new events (event_flag=0) for upcoming days.")

    forecast_table = pd.DataFrame(
        {
            "Date": forecast_result.future_dates,
            f"Forecast_{selected_metal}": forecast_result.future_forecast.values,
        }
    )
    st.subheader("Forecast Values")
    st.dataframe(forecast_table, width="stretch")


def render_event_sections(df: pd.DataFrame) -> None:
    """Render event impact and event-study sections when event data is available."""
    if EVENT_COLUMN not in df.columns:
        return

    st.header("4. Event Impact Summary")
    st.subheader("How Event Days Differ from Non-Event Days")
    impact_summary = compute_event_impact_summary(df)
    if impact_summary.summary_table.empty:
        st.info("Need both event and non-event records to compute event impact summary.")
    else:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Event Days", f"{impact_summary.metrics['event_days']}")
        metric_col2.metric("Non-Event Days", f"{impact_summary.metrics['non_event_days']}")
        metric_col3.metric("Gold Return Impact", f"{impact_summary.metrics['gold_return_impact']:+.3f}%")
        metric_col4.metric("Silver Return Impact", f"{impact_summary.metrics['silver_return_impact']:+.3f}%")
        st.dataframe(impact_summary.summary_table, width="stretch")

    st.header("5. Event Study")
    st.subheader("Returns Around Each Event")
    event_window = st.slider(
        "Event window (trading days before/after)",
        min_value=3,
        max_value=20,
        value=DEFAULT_EVENT_WINDOW,
        step=1,
    )

    gold_event_study = build_event_study_frame(df, PRICE_GOLD_COLUMN, window_size=event_window)
    silver_event_study = build_event_study_frame(df, PRICE_SILVER_COLUMN, window_size=event_window)
    if gold_event_study.empty or silver_event_study.empty:
        st.info("Not enough surrounding observations to build the event-study chart for the selected window.")
        return

    event_reference = build_event_reference(gold_event_study)
    min_year = int(event_reference["Event Year"].min())
    max_year = int(event_reference["Event Year"].max())
    filter_col1, filter_col2 = st.columns(2)
    selected_years = filter_col1.slider(
        "Event year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    event_types = ["All"] + sorted(event_reference["Event Type"].unique().tolist())
    selected_event_type = filter_col2.selectbox("Event type", event_types, index=0)

    filtered_reference, filtered_gold, filtered_silver = filter_event_studies(
        gold_event_study,
        silver_event_study,
        event_reference,
        year_range=selected_years,
        event_type=selected_event_type,
    )
    if filtered_gold.empty or filtered_silver.empty:
        st.info("No events match the current filter selection.")
        return

    avg_gold_event_study = filtered_gold.groupby("relative_day", as_index=False)["return_vs_event_%"].mean()
    avg_silver_event_study = filtered_silver.groupby("relative_day", as_index=False)["return_vs_event_%"].mean()
    st.pyplot(
        plot_event_study(filtered_gold, filtered_silver, avg_gold_event_study, avg_silver_event_study),
        width="stretch",
    )

    pre_post_return_table = summarize_event_returns(filtered_gold, filtered_silver)
    st.subheader("Pre- and Post-Event Average Returns")
    st.dataframe(pre_post_return_table, width="stretch")
    st.caption(
        "Pre-event and post-event averages are calculated over the selected window using returns relative to the event-day price."
    )

    pre_post_price_table = summarize_event_prices(filtered_gold, filtered_silver)
    st.subheader("Pre- and Post-Event Average Prices")
    st.dataframe(pre_post_price_table, width="stretch")
    st.caption("Price averages are calculated over the selected event window using the filtered event set.")

    event_impact_table = build_event_impact_table(filtered_reference, filtered_gold, filtered_silver)
    event_impact_display = event_impact_table[
        [
            "Event Date",
            "Event",
            "Event Type",
            "Gold Pre-Event Price",
            "Gold Event-Day Price",
            "Gold Post-Event Price",
            "Silver Pre-Event Price",
            "Silver Event-Day Price",
            "Silver Post-Event Price",
            "Gold % Change Pre->Post",
            "Silver % Change Pre->Post",
        ]
    ]
    st.subheader("Event-Wise Metal Price Impact")
    st.dataframe(event_impact_display, width="stretch")
    st.caption("Per-event impacts use average prices before the event, on the event day, and after the event within the selected window.")
    st.caption(
        f"Thin lines show individual event paths. The bold line is the average across {len(filtered_reference)} events. Relative day 0 is the event date."
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    st.title("📊 Precious Metal Price Analysis Dashboard")
    st.info("📊 Dashboard showing all analysis steps below.")

    try:
        df = get_dashboard_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    render_yearly_analysis(df)
    render_market_overview(df)
    render_forecasting(df)
    render_event_sections(df)


if __name__ == "__main__":
    main()