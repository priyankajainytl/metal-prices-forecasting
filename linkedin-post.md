I built an end-to-end metal price forecasting project to answer a practical question: how differently do gold and silver react to geopolitical shocks, and what does that mean for forecasting reliability?

Using daily spot prices and event-tagged timelines, I combined trend analysis, volatility diagnostics, and time-series modeling with backtesting.

Key findings:
- Gold showed much stronger long-term structural appreciation, with average annual growth around 71.17 versus 0.71 for silver.
- Silver was significantly more volatile, with daily standard deviation near 1.74% versus 0.98% for gold (about 1.8x higher).
- Around major geopolitical events, gold's event-day return averaged -0.62% (statistically significant, p=0.028), while silver's averaged -0.05% (p=0.83, not significant).
- During extreme stress (2008), volatility surged for both, but silver expanded much more (about 6.14% vs 3.47% for gold), implying wider uncertainty in silver forecasts.

Modeling approach:
- [ARIMA (AutoRegressive Integrated Moving Average)](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average): forecasts a series using its own history, differencing, and past error structure.
- [ARIMAX (ARIMA with exogenous inputs)](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model#Exogenous_inputs): extends ARIMA by adding external drivers, such as a geopolitical event flag.

How I configured and tuned model settings:
- p controls how many lagged past values are used.
- d controls how much differencing is applied to stabilize trend/non-stationarity.
- q controls how many lagged error terms are used.
- Forecast horizon is how many future days are predicted.
- Train split determines how much history is used for training versus backtesting.
- Tuning process: set d first for stationarity, run a small grid over p and q, compare backtest MAE, and prefer the simplest model with comparable error to reduce overfitting.

I wrapped the full workflow in a Streamlit app with modular preprocessing, event-feature engineering, ARIMA/ARIMAX paths, MAE-based backtesting, and clear visual outputs. This project strengthened my ability to turn statistical analysis into an interpretable, decision-ready forecasting product.