import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    from data_loader import load_raw_data


def classify_event_type(event_name):
    event_text = str(event_name).strip().lower()
    if any(keyword in event_text for keyword in ['war', 'invades', 'bombing', 'desert storm']):
        return 'War / Military Conflict'
    if any(keyword in event_text for keyword in ['terror', 'bombings', 'threat']):
        return 'Terrorism / Security Threat'
    if any(keyword in event_text for keyword in ['tension', 'approval', 'escalate']):
        return 'Geopolitical Tension'
    return 'Other'


def build_event_study_frame(price_df, price_col, event_col='EVENT', window_size=5):
    frame = (
        price_df[['Date', price_col, event_col]]
        .copy()
        .sort_values('Date')
        .reset_index(drop=True)
    )
    frame[event_col] = frame[event_col].fillna('').astype(str)
    event_positions = frame.index[frame[event_col].str.strip().ne('')]

    study_rows = []
    for event_pos in event_positions:
        start = event_pos - window_size
        end = event_pos + window_size
        if start < 0 or end >= len(frame):
            continue

        event_name = frame.at[event_pos, event_col].strip()
        event_price = frame.at[event_pos, price_col]
        if pd.isna(event_price) or event_price == 0:
            continue

        window = frame.iloc[start:end + 1].copy()
        window['relative_day'] = range(-window_size, window_size + 1)
        window['event_name'] = event_name
        window['event_date'] = frame.at[event_pos, 'Date']
        window['event_year'] = frame.at[event_pos, 'Date'].year
        window['event_type'] = classify_event_type(event_name)
        window['price_value'] = window[price_col]
        window['return_vs_event_%'] = ((window[price_col] / event_price) - 1) * 100
        study_rows.append(
            window[
                ['Date', 'relative_day', 'event_name', 'event_date', 'event_year', 'event_type', 'price_value', 'return_vs_event_%']
            ]
        )

    if not study_rows:
        return pd.DataFrame(
            columns=['Date', 'relative_day', 'event_name', 'event_date', 'event_year', 'event_type', 'price_value', 'return_vs_event_%']
        )

    return pd.concat(study_rows, ignore_index=True)

st.set_page_config(page_title="Precious Metal Analysis", layout="wide")

st.title("📊 Precious Metal Price Analysis Dashboard")

# Load data
df = load_raw_data()

if 'Date' not in df.columns and 'DATE' in df.columns:
    df = df.rename(columns={'DATE': 'Date'})

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar Navigation removed - displaying all steps
st.info("📊 Dashboard showing all analysis steps below.")

# ========== DASHBOARD LAYOUT ==========

# Step 1: Yearly Average Analysis
st.header("1. Yearly Average Analysis")
st.subheader("Yearly Average Price Analysis")

# Extract year
df['Year'] = df['Date'].dt.year

# Silver yearly average
silver_avg = df.groupby('Year')['Price_silver'].mean().reset_index()

fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(silver_avg['Year'], silver_avg['Price_silver'], marker='o')
ax1.set_title("Average Silver Price per Year")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Price")
for _, row in silver_avg.iterrows():
    ax1.annotate(
        f"{row['Price_silver']:,.1f}",
        (row['Year'], row['Price_silver']),
        textcoords='offset points', xytext=(0, 8),
        ha='center', fontsize=8
    )

year_col1, year_col2 = st.columns([1, 1.7])

# Gold yearly average (extra insight)
gold_avg = df.groupby('Year')['Price_gold'].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(gold_avg['Year'], gold_avg['Price_gold'], marker='o', color='orange')
ax2.set_title("Average Gold Price per Year")
ax2.set_xlabel("Year")
ax2.set_ylabel("Average Price")
ax2.grid(alpha=0.25)
ax2.tick_params(axis='x', labelrotation=45)
for idx, (_, row) in enumerate(gold_avg.iterrows()):
    offset_y = 10 if idx % 2 == 0 else -14
    ax2.annotate(
        f"{row['Price_gold']:,.1f}",
        (row['Year'], row['Price_gold']),
        textcoords='offset points', xytext=(0, offset_y),
        ha='center', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none')
    )

with year_col1:
    st.pyplot(fig1)
with year_col2:
    st.pyplot(fig2)

# Step 2: Market Overview Dashboard
st.header("2. Market Overview Dashboard")

st.subheader("Gold & Silver Price Trends")
price_trend_selection = st.selectbox("Display", ["Both", "Gold", "Silver"], index=0, key="price_trend_select")

fig, ax_gold = plt.subplots(figsize=(12, 6))

if price_trend_selection == "Both":
    ax_silver = ax_gold.twinx()
    
    gold_line = ax_gold.plot(df['Date'], df['Price_gold'], label='Gold', color='tab:orange')
    silver_line = ax_silver.plot(df['Date'], df['Price_silver'], label='Silver', color='tab:blue')
    
    ax_gold.set_xlabel("Date")
    ax_gold.set_ylabel("Gold Price", color='tab:orange')
    ax_silver.set_ylabel("Silver Price", color='tab:blue')
    
    lines = gold_line + silver_line
    labels = [line.get_label() for line in lines]
    ax_gold.legend(lines, labels, loc='upper left')
    
    latest = df.dropna(subset=['Price_gold', 'Price_silver']).iloc[-1]
    ax_gold.annotate(
        f"Latest: {latest['Price_gold']:,.2f}",
        (latest['Date'], latest['Price_gold']),
        textcoords='offset points', xytext=(8, -12),
        color='tab:orange', fontsize=9, fontweight='bold'
    )
    ax_silver.annotate(
        f"Latest: {latest['Price_silver']:,.2f}",
        (latest['Date'], latest['Price_silver']),
        textcoords='offset points', xytext=(8, 8),
        color='tab:blue', fontsize=9, fontweight='bold'
    )
elif price_trend_selection == "Gold":
    ax_gold.plot(df['Date'], df['Price_gold'], label='Gold', color='tab:orange', linewidth=2)
    ax_gold.set_xlabel("Date")
    ax_gold.set_ylabel("Gold Price", color='tab:orange')
    ax_gold.legend(loc='upper left')
    
    latest = df.dropna(subset=['Price_gold']).iloc[-1]
    ax_gold.annotate(
        f"Latest: {latest['Price_gold']:,.2f}",
        (latest['Date'], latest['Price_gold']),
        textcoords='offset points', xytext=(8, -12),
        color='tab:orange', fontsize=9, fontweight='bold'
    )
else:  # Silver
    ax_gold.plot(df['Date'], df['Price_silver'], label='Silver', color='tab:blue', linewidth=2)
    ax_gold.set_xlabel("Date")
    ax_gold.set_ylabel("Silver Price", color='tab:blue')
    ax_gold.legend(loc='upper left')
    
    latest = df.dropna(subset=['Price_silver']).iloc[-1]
    ax_gold.annotate(
        f"Latest: {latest['Price_silver']:,.2f}",
        (latest['Date'], latest['Price_silver']),
        textcoords='offset points', xytext=(8, -12),
        color='tab:blue', fontsize=9, fontweight='bold'
    )

fig.tight_layout()

st.pyplot(fig, use_container_width=True)

st.subheader("Rolling Volatility of Daily Returns")

vol_window = st.slider("Volatility window (days)", min_value=7, max_value=120, value=30, step=1)

volatility_df = (
    df[['Date', 'Price_gold', 'Price_silver']]
    .dropna()
    .sort_values('Date')
    .copy()
)
volatility_df['Gold_Return_%'] = volatility_df['Price_gold'].pct_change() * 100
volatility_df['Silver_Return_%'] = volatility_df['Price_silver'].pct_change() * 100
volatility_df['Gold_Volatility_%'] = volatility_df['Gold_Return_%'].rolling(vol_window).std()
volatility_df['Silver_Volatility_%'] = volatility_df['Silver_Return_%'].rolling(vol_window).std()

fig_vol, ax_vol = plt.subplots(figsize=(12, 6))
ax_vol.plot(
    volatility_df['Date'],
    volatility_df['Gold_Volatility_%'],
    color='tab:orange',
    label=f'Gold {vol_window}D Rolling Volatility'
)
ax_vol.plot(
    volatility_df['Date'],
    volatility_df['Silver_Volatility_%'],
    color='tab:blue',
    label=f'Silver {vol_window}D Rolling Volatility'
)
ax_vol.set_xlabel('Date')
ax_vol.set_ylabel('Volatility (% Std. Dev. of Daily Returns)')
ax_vol.set_title('Gold vs Silver Volatility Over Time')
ax_vol.legend(loc='upper left')

fig_vol.tight_layout()

st.pyplot(fig_vol, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gold Latest", f"{df['Price_gold'].iloc[-1]:,.2f}")
c2.metric("Gold Max", f"{df['Price_gold'].max():,.2f}")
c3.metric("Silver Latest", f"{df['Price_silver'].iloc[-1]:,.2f}")
c4.metric("Silver Max", f"{df['Price_silver'].max():,.2f}")

latest_vol = volatility_df.dropna(subset=['Gold_Volatility_%', 'Silver_Volatility_%']).iloc[-1]
v1, v2 = st.columns(2)
v1.metric("Gold Latest Volatility", f"{latest_vol['Gold_Volatility_%']:.3f}%")
v2.metric("Silver Latest Volatility", f"{latest_vol['Silver_Volatility_%']:.3f}%")

# Step 3: ARIMA Forecasting
st.header("3. ARIMA Forecasting")
st.subheader("Forecast Future Prices Using ARIMA")

col_a, col_b, col_c, col_d = st.columns(4)
selected_metal = col_a.selectbox("Metal", ["Gold", "Silver"], index=0)
arima_p = col_b.number_input("p", min_value=0, max_value=5, value=2, step=1)
arima_d = col_c.number_input("d", min_value=0, max_value=2, value=1, step=1)
arima_q = col_d.number_input("q", min_value=0, max_value=5, value=2, step=1)

forecast_horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=180, value=30, step=1)
train_fraction = st.slider("Train split", min_value=0.70, max_value=0.95, value=0.90, step=0.05)
use_event_exog = st.checkbox("Include EVENT as exogenous feature (ARIMAX)", value=True)
history_display = st.select_slider(
    "History shown on forecast chart",
    options=[90, 180, 365, 730, "Full"],
    value=365
)

target_column = 'Price_gold' if selected_metal == 'Gold' else 'Price_silver'
target_series = (
    df[['Date', target_column]]
    .dropna()
    .drop_duplicates(subset=['Date'])
    .set_index('Date')[target_column]
    .asfreq('D')
    .interpolate(method='time')
    .dropna()
)

event_series = None
if 'EVENT' in df.columns:
    event_series = (
        df[['Date', 'EVENT']]
        .drop_duplicates(subset=['Date'])
        .assign(event_flag=lambda frame: frame['EVENT'].fillna('').astype(str).str.strip().ne('').astype(int))
        .set_index('Date')['event_flag']
        .asfreq('D')
        .fillna(0)
        .reindex(target_series.index, fill_value=0)
    )

if len(target_series) < 60:
    st.warning("Not enough data points to fit ARIMA reliably. Need at least 60 daily observations.")
else:
    split_index = int(len(target_series) * train_fraction)
    split_index = max(30, min(split_index, len(target_series) - 1))
    train_series = target_series.iloc[:split_index]
    test_series = target_series.iloc[split_index:]

    try:
        backtest_steps = min(len(test_series), forecast_horizon)
        model_label = f"ARIMA({arima_p},{arima_d},{arima_q})"
        event_coef = float('nan')
        event_p_value = float('nan')

        if use_event_exog and event_series is not None:
            train_exog = event_series.iloc[:split_index].to_frame(name='event_flag')
            test_exog = event_series.iloc[split_index:].to_frame(name='event_flag')
            backtest_model = SARIMAX(
                train_series,
                exog=train_exog,
                order=(arima_p, arima_d, arima_q),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            if backtest_steps > 0:
                backtest_forecast = backtest_model.forecast(
                    steps=backtest_steps,
                    exog=test_exog.iloc[:backtest_steps]
                )
                test_slice = test_series.iloc[:backtest_steps]
                backtest_mae = (test_slice - backtest_forecast).abs().mean()
            else:
                backtest_forecast = pd.Series(dtype='float64')
                test_slice = pd.Series(dtype='float64')
                backtest_mae = float('nan')

            full_exog = event_series.to_frame(name='event_flag')
            full_model = SARIMAX(
                target_series,
                exog=full_exog,
                order=(arima_p, arima_d, arima_q),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            model_label = f"ARIMAX({arima_p},{arima_d},{arima_q})"
            event_coef = full_model.params.get('event_flag', float('nan'))
            event_p_value = full_model.pvalues.get('event_flag', float('nan'))
        else:
            backtest_model = ARIMA(train_series, order=(arima_p, arima_d, arima_q)).fit()
            if backtest_steps > 0:
                backtest_forecast = backtest_model.forecast(steps=backtest_steps)
                test_slice = test_series.iloc[:backtest_steps]
                backtest_mae = (test_slice - backtest_forecast).abs().mean()
            else:
                backtest_forecast = pd.Series(dtype='float64')
                test_slice = pd.Series(dtype='float64')
                backtest_mae = float('nan')
            full_model = ARIMA(target_series, order=(arima_p, arima_d, arima_q)).fit()

        future_dates = pd.date_range(
            start=target_series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )

        if use_event_exog and event_series is not None:
            future_exog = pd.DataFrame({'event_flag': [0] * forecast_horizon}, index=future_dates)
            future_forecast = full_model.forecast(steps=forecast_horizon, exog=future_exog)
        else:
            future_forecast = full_model.forecast(steps=forecast_horizon)

        fig_forecast, forecast_axis = plt.subplots(figsize=(12, 6))
        history_window = len(target_series) if history_display == "Full" else min(history_display, len(target_series))
        forecast_axis.plot(
            target_series.index[-history_window:],
            target_series.iloc[-history_window:],
            label=f'{selected_metal} Actual',
            color='tab:blue'
        )
        if len(backtest_forecast) > 0:
            forecast_axis.plot(
                test_slice.index,
                backtest_forecast,
                label='Backtest Forecast',
                linestyle='--',
                color='tab:orange'
            )
        forecast_axis.plot(
            future_dates,
            future_forecast.values,
            label='Future Forecast',
            linestyle='--',
            color='tab:green'
        )
        forecast_axis.set_title(f"{selected_metal} {model_label} Forecast")
        forecast_axis.set_xlabel("Date")
        forecast_axis.set_ylabel("Price")
        forecast_axis.legend(loc='upper left')
        fig_forecast.tight_layout()
        st.pyplot(fig_forecast, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Backtest MAE", f"{backtest_mae:,.2f}" if pd.notna(backtest_mae) else "N/A")
        m2.metric("Forecast Horizon", f"{forecast_horizon} days")
        if use_event_exog and event_series is not None:
            m3.metric("Event Coefficient", f"{event_coef:,.4f}" if pd.notna(event_coef) else "N/A")
            if pd.notna(event_p_value):
                st.caption(f"Event coefficient p-value: {event_p_value:.4f} (lower means stronger statistical evidence).")
            st.caption("Future forecast assumes no new events (event_flag=0) for upcoming days.")

        forecast_table = pd.DataFrame({
            'Date': future_dates,
            f'Forecast_{selected_metal}': future_forecast.values
        })
        st.subheader("Forecast Values")
        st.dataframe(forecast_table, use_container_width=True)

    except Exception as error:
        st.error(f"ARIMA model failed with order ({arima_p},{arima_d},{arima_q}): {error}")

if 'EVENT' in df.columns:
    st.header("4. Event Impact Summary")
    st.subheader("How Event Days Differ from Non-Event Days")

    impact_df = df[['Date', 'Price_gold', 'Price_silver', 'EVENT']].copy()
    impact_df['is_event_day'] = impact_df['EVENT'].fillna('').astype(str).str.strip().ne('')
    impact_df = impact_df.sort_values('Date')
    impact_df['Gold_Return_%'] = impact_df['Price_gold'].pct_change() * 100
    impact_df['Silver_Return_%'] = impact_df['Price_silver'].pct_change() * 100

    grouped = impact_df.groupby('is_event_day').agg(
        avg_gold_price=('Price_gold', 'mean'),
        avg_silver_price=('Price_silver', 'mean'),
        avg_gold_return=('Gold_Return_%', 'mean'),
        avg_silver_return=('Silver_Return_%', 'mean'),
        days=('Date', 'count')
    )

    if True in grouped.index and False in grouped.index:
        event_stats = grouped.loc[True]
        non_event_stats = grouped.loc[False]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Event Days", f"{int(event_stats['days'])}")
        k2.metric("Non-Event Days", f"{int(non_event_stats['days'])}")
        k3.metric("Gold Return Impact", f"{event_stats['avg_gold_return'] - non_event_stats['avg_gold_return']:+.3f}%")
        k4.metric("Silver Return Impact", f"{event_stats['avg_silver_return'] - non_event_stats['avg_silver_return']:+.3f}%")

        summary_table = pd.DataFrame([
            {
                'Group': 'Event Days',
                'Avg Gold Price': event_stats['avg_gold_price'],
                'Avg Silver Price': event_stats['avg_silver_price'],
                'Avg Gold Daily Return %': event_stats['avg_gold_return'],
                'Avg Silver Daily Return %': event_stats['avg_silver_return'],
            },
            {
                'Group': 'Non-Event Days',
                'Avg Gold Price': non_event_stats['avg_gold_price'],
                'Avg Silver Price': non_event_stats['avg_silver_price'],
                'Avg Gold Daily Return %': non_event_stats['avg_gold_return'],
                'Avg Silver Daily Return %': non_event_stats['avg_silver_return'],
            },
        ])
        st.dataframe(summary_table, use_container_width=True)
    else:
        st.info("Need both event and non-event records to compute event impact summary.")

    st.header("5. Event Study")
    st.subheader("Returns Around Each Event")

    event_window = st.slider("Event window (trading days before/after)", min_value=3, max_value=20, value=5, step=1)

    gold_event_study = build_event_study_frame(df, 'Price_gold', window_size=event_window)
    silver_event_study = build_event_study_frame(df, 'Price_silver', window_size=event_window)

    if gold_event_study.empty or silver_event_study.empty:
        st.info("Not enough surrounding observations to build the event-study chart for the selected window.")
    else:
        event_reference = (
            gold_event_study[['event_date', 'event_name', 'event_year', 'event_type']]
            .drop_duplicates()
            .sort_values('event_date')
            .rename(
                columns={
                    'event_date': 'Event Date',
                    'event_name': 'Event',
                    'event_year': 'Event Year',
                    'event_type': 'Event Type'
                }
            )
        )

        min_year = int(event_reference['Event Year'].min())
        max_year = int(event_reference['Event Year'].max())
        filter_col1, filter_col2 = st.columns(2)
        selected_years = filter_col1.slider(
            "Event year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        event_types = ['All'] + sorted(event_reference['Event Type'].unique().tolist())
        selected_event_type = filter_col2.selectbox("Event type", event_types, index=0)

        filtered_event_reference = event_reference[
            event_reference['Event Year'].between(selected_years[0], selected_years[1])
        ].copy()
        if selected_event_type != 'All':
            filtered_event_reference = filtered_event_reference[
                filtered_event_reference['Event Type'] == selected_event_type
            ]

        selected_event_names = filtered_event_reference['Event'].tolist()
        gold_event_study = gold_event_study[gold_event_study['event_name'].isin(selected_event_names)].copy()
        silver_event_study = silver_event_study[silver_event_study['event_name'].isin(selected_event_names)].copy()

        if gold_event_study.empty or silver_event_study.empty:
            st.info("No events match the current filter selection.")
        else:
            avg_gold_event_study = gold_event_study.groupby('relative_day', as_index=False)['return_vs_event_%'].mean()
            avg_silver_event_study = silver_event_study.groupby('relative_day', as_index=False)['return_vs_event_%'].mean()

            fig_event_study, (ax_gold_event, ax_silver_event) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            for event_name, event_slice in gold_event_study.groupby('event_name'):
                ax_gold_event.plot(
                    event_slice['relative_day'],
                    event_slice['return_vs_event_%'],
                    alpha=0.25,
                    linewidth=1
                )
            ax_gold_event.plot(
                avg_gold_event_study['relative_day'],
                avg_gold_event_study['return_vs_event_%'],
                color='tab:orange',
                linewidth=3,
                label='Average'
            )
            ax_gold_event.axvline(0, color='black', linestyle='--', linewidth=1)
            ax_gold_event.axhline(0, color='grey', linestyle=':', linewidth=1)
            ax_gold_event.set_title('Gold Around Events')
            ax_gold_event.set_xlabel('Relative Trading Day')
            ax_gold_event.set_ylabel('Return vs Event Day (%)')
            ax_gold_event.legend(loc='upper left')

            for event_name, event_slice in silver_event_study.groupby('event_name'):
                ax_silver_event.plot(
                    event_slice['relative_day'],
                    event_slice['return_vs_event_%'],
                    alpha=0.25,
                    linewidth=1
                )
            ax_silver_event.plot(
                avg_silver_event_study['relative_day'],
                avg_silver_event_study['return_vs_event_%'],
                color='tab:blue',
                linewidth=3,
                label='Average'
            )
            ax_silver_event.axvline(0, color='black', linestyle='--', linewidth=1)
            ax_silver_event.axhline(0, color='grey', linestyle=':', linewidth=1)
            ax_silver_event.set_title('Silver Around Events')
            ax_silver_event.set_xlabel('Relative Trading Day')
            ax_silver_event.set_ylabel('Return vs Event Day (%)')
            ax_silver_event.legend(loc='upper left')

            st.pyplot(fig_event_study)

            pre_post_table = pd.DataFrame([
                {
                    'Metal': 'Gold',
                    'Average Pre-Event Return %': gold_event_study.loc[gold_event_study['relative_day'] < 0, 'return_vs_event_%'].mean(),
                    'Average Post-Event Return %': gold_event_study.loc[gold_event_study['relative_day'] > 0, 'return_vs_event_%'].mean(),
                    'Average Event-Day Return %': gold_event_study.loc[gold_event_study['relative_day'] == 0, 'return_vs_event_%'].mean(),
                    'Events Included': gold_event_study['event_name'].nunique()
                },
                {
                    'Metal': 'Silver',
                    'Average Pre-Event Return %': silver_event_study.loc[silver_event_study['relative_day'] < 0, 'return_vs_event_%'].mean(),
                    'Average Post-Event Return %': silver_event_study.loc[silver_event_study['relative_day'] > 0, 'return_vs_event_%'].mean(),
                    'Average Event-Day Return %': silver_event_study.loc[silver_event_study['relative_day'] == 0, 'return_vs_event_%'].mean(),
                    'Events Included': silver_event_study['event_name'].nunique()
                }
            ])

            st.subheader("Pre- and Post-Event Average Returns")
            st.dataframe(pre_post_table, use_container_width=True)
            st.caption(
                "Pre-event and post-event averages are calculated over the selected window using returns relative to the event-day price."
            )

            pre_post_price_table = pd.DataFrame([
                {
                    'Metal': 'Gold',
                    'Average Pre-Event Price': gold_event_study.loc[gold_event_study['relative_day'] < 0, 'price_value'].mean(),
                    'Average Event-Day Price': gold_event_study.loc[gold_event_study['relative_day'] == 0, 'price_value'].mean(),
                    'Average Post-Event Price': gold_event_study.loc[gold_event_study['relative_day'] > 0, 'price_value'].mean(),
                    'Events Included': gold_event_study['event_name'].nunique()
                },
                {
                    'Metal': 'Silver',
                    'Average Pre-Event Price': silver_event_study.loc[silver_event_study['relative_day'] < 0, 'price_value'].mean(),
                    'Average Event-Day Price': silver_event_study.loc[silver_event_study['relative_day'] == 0, 'price_value'].mean(),
                    'Average Post-Event Price': silver_event_study.loc[silver_event_study['relative_day'] > 0, 'price_value'].mean(),
                    'Events Included': silver_event_study['event_name'].nunique()
                }
            ])

            st.subheader("Pre- and Post-Event Average Prices")
            st.dataframe(pre_post_price_table, use_container_width=True)
            st.caption(
                "Price averages are calculated over the selected event window using the filtered event set."
            )

            gold_event_impact = (
                gold_event_study.groupby(['event_name', 'event_date'], as_index=False)
                .apply(
                    lambda frame: pd.Series({
                        'Gold Pre-Event Price': frame.loc[frame['relative_day'] < 0, 'price_value'].mean(),
                        'Gold Event-Day Price': frame.loc[frame['relative_day'] == 0, 'price_value'].mean(),
                        'Gold Post-Event Price': frame.loc[frame['relative_day'] > 0, 'price_value'].mean(),
                    })
                )
            )
            silver_event_impact = (
                silver_event_study.groupby(['event_name', 'event_date'], as_index=False)
                .apply(
                    lambda frame: pd.Series({
                        'Silver Pre-Event Price': frame.loc[frame['relative_day'] < 0, 'price_value'].mean(),
                        'Silver Event-Day Price': frame.loc[frame['relative_day'] == 0, 'price_value'].mean(),
                        'Silver Post-Event Price': frame.loc[frame['relative_day'] > 0, 'price_value'].mean(),
                    })
                )
            )

            event_impact_table = (
                filtered_event_reference.rename(columns={'Event': 'event_name', 'Event Date': 'event_date'})
                .merge(gold_event_impact, on=['event_name', 'event_date'], how='left')
                .merge(silver_event_impact, on=['event_name', 'event_date'], how='left')
                .rename(columns={'event_name': 'Event', 'event_date': 'Event Date'})
            )

            event_impact_table['Gold % Change Pre->Event'] = (
                (event_impact_table['Gold Event-Day Price'] / event_impact_table['Gold Pre-Event Price'] - 1) * 100
            )
            event_impact_table['Gold % Change Event->Post'] = (
                (event_impact_table['Gold Post-Event Price'] / event_impact_table['Gold Event-Day Price'] - 1) * 100
            )
            event_impact_table['Gold % Change Pre->Post'] = (
                (event_impact_table['Gold Post-Event Price'] / event_impact_table['Gold Pre-Event Price'] - 1) * 100
            )

            event_impact_table['Silver % Change Pre->Event'] = (
                (event_impact_table['Silver Event-Day Price'] / event_impact_table['Silver Pre-Event Price'] - 1) * 100
            )
            event_impact_table['Silver % Change Event->Post'] = (
                (event_impact_table['Silver Post-Event Price'] / event_impact_table['Silver Event-Day Price'] - 1) * 100
            )
            event_impact_table['Silver % Change Pre->Post'] = (
                (event_impact_table['Silver Post-Event Price'] / event_impact_table['Silver Pre-Event Price'] - 1) * 100
            )

            event_impact_display = event_impact_table[[
                'Event Date',
                'Event',
                'Event Type',
                'Gold Pre-Event Price',
                'Gold Event-Day Price',
                'Gold Post-Event Price',
                'Silver Pre-Event Price',
                'Silver Event-Day Price',
                'Silver Post-Event Price',
                'Gold % Change Pre->Post',
                'Silver % Change Pre->Post'
            ]]

            st.subheader("Event-Wise Metal Price Impact")
            st.dataframe(event_impact_display, use_container_width=True)
            st.caption(
                "Per-event impacts use average prices before the event, on the event day, and after the event within the selected window."
            )

            st.caption(
                f"Thin lines show individual event paths. The bold line is the average across {len(filtered_event_reference)} events. "
                "Relative day 0 is the event date."
            )