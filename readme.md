# Metal Prices Forecasting

This project analyzes gold and silver spot prices alongside geopolitical risk data and event annotations. It provides an interactive Streamlit dashboard for market exploration, event impact analysis, and ARIMA or ARIMAX forecasting.

## What This Project Covers

- Yearly trend analysis for gold and silver.
- Daily market trend and rolling-volatility views.
- Forecasting with ARIMA and optional event-aware ARIMAX.
- Event impact summary comparing event days vs non-event days.
- Event-study windows showing pre and post event behavior.
- Feature engineering pipeline for downstream modeling workflows.

## Quick Start

1. From the repository root, install dependencies:
	pip install -r requirements.txt
2. Start the dashboard:
	streamlit run src/app.py
3. Open the local URL shown by Streamlit in your terminal.

## Repository Layout

- [src/app.py](src/app.py): Streamlit entrypoint and page flow.
- [src/charts.py](src/charts.py): Matplotlib plotting helpers used by the dashboard.
- [src/analysis/events.py](src/analysis/events.py): Event classification, event-study generation, filtering, and impact tables.
- [src/analysis/market.py](src/analysis/market.py): Market aggregates, volatility, metrics, and event-day summary logic.
- [src/models/forecasting.py](src/models/forecasting.py): ARIMA and SARIMAX forecasting logic and forecast artifacts.
- [src/data/loader.py](src/data/loader.py): Raw CSV validation, loading, merge, and date normalization.
- [src/data/preprocessor.py](src/data/preprocessor.py): Cleaning, feature engineering, and pipeline output writer.
- [src/config.py](src/config.py): Shared constants and default modeling or analysis settings.
- [src/validation.py](src/validation.py): Shared validation and date normalization helpers.
- [tests/test_preprocessor.py](tests/test_preprocessor.py): Unit tests for cleaning and engineered features.
- [tests/test_events.py](tests/test_events.py): Unit tests for event classification and event-study logic.

## Data Inputs

Expected raw files in [src/data/raw](src/data/raw):

- Gold_Spot_Price_Daily.csv
- Silver_Spot_Price_Daily.csv
- Geopolitical_Risk_Index_Daily.csv

The loader merges these files on date and standardizes key columns, including Date, Price_gold, Price_silver, GPR, and EVENT.

## Preprocessing Pipeline

The preprocessing flow is implemented as Python functions in [src/data/preprocessor.py](src/data/preprocessor.py):

- clean_data: numeric coercion, forward-fill, and outlier trimming.
- feature_engineering: lag features and rolling statistics.
- run_preprocessing_pipeline: end-to-end pipeline and CSV persistence.

Default output path:

- [src/data/processed/features.csv](src/data/processed/features.csv)

There is also a root-level processed artifact at [data/processed/features.csv](data/processed/features.csv). If you need a single canonical output location for your workflow, align your scripts to one path.

## Dashboard Walkthrough

Dashboard sections are orchestrated in [src/app.py](src/app.py):

1. Yearly Average Analysis.
2. Market Overview Dashboard.
3. ARIMA Forecasting (with optional ARIMAX via event flags).
4. Event Impact Summary.
5. Event Study.

The app uses caching for data loading and delegates heavy lifting to modular analysis, charting, and forecasting helpers.

## Run Tests

Run the test suite from the repository root:

python -m unittest discover -s tests

Current tests validate core preprocessing and event-analysis logic.

## Dependencies

Dependencies are defined in [requirements.txt](requirements.txt).

Note: Python version is not pinned in the repository yet. Use a modern Python 3.x environment compatible with Streamlit, pandas, and statsmodels.
