## Metal Prices Forecasting

This project analyzes gold and silver prices alongside geopolitical risk events, and exposes the results through a Streamlit dashboard.

## Structure

- `src/app.py`: Streamlit entrypoint and UI orchestration.
- `src/data/`: raw-data loading and preprocessing helpers.
- `src/analysis/`: reusable market and event-study analysis helpers.
- `src/models/`: forecasting helpers for ARIMA and ARIMAX workflows.
- `src/charts.py`: plotting helpers used by the dashboard.
- `tests/`: focused unit tests for preprocessing and event analysis.

## Run

```bash
streamlit run src/app.py
```

## Test

```bash
python -m unittest discover -s tests
```
