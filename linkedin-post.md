I built an end-to-end metal price forecasting project to answer a practical question: how differently do gold and silver react to geopolitical shocks, and what does that mean for forecasting reliability?

Using daily spot prices and event-tagged timelines, I combined trend analysis, volatility diagnostics, event study analysis, and time-series modeling with backtesting.

## Result 1: Gold's long-term structural strength is clearly visible

Gold showed much stronger long-term structural appreciation, with average annual growth around 71.17 versus 0.71 for silver.

![Yearly Average Analysis](src/screenshots/1.Yearly%20Average%20Analysis.png)

This yearly view shows two different long-term shapes: gold trends upward with stronger momentum, while silver remains more cyclical and slower-growing. The chart supports the idea that these two metals should not be modeled as identical risk assets.

## Result 2: Silver carries a much higher volatility profile

Silver was significantly more volatile, with daily standard deviation near 1.74% versus 0.98% for gold (about 1.8x higher).

![Market Overview Dashboard](src/screenshots/2.1.Market%20Overview%20Dashboard.png)

The dual-axis trend chart highlights how silver's path exhibits sharper swings even when both metals rise over time. It also shows that short-term dislocations are typically larger and less stable for silver than for gold.

![Rolling Volatility of Daily Returns](src/screenshots/2.2.Rolling%20Volatility%20of%20Daily%20Returns.png)

The rolling-volatility panel confirms this difference quantitatively across regimes, with silver repeatedly spiking above gold in stress periods. This directly explains why silver forecasts usually require wider uncertainty expectations.

## Result 3: Event-day response is asymmetric between metals

Around major geopolitical events, gold's event-day return averaged -0.62% (statistically significant, p=0.028), while silver's averaged -0.05% (p=0.83, not significant).

![Event Impact Summary](src/screenshots/4.Event%20Impact%20Summary.png)

The event-impact summary shows measurable divergence in how gold and silver react on event versus non-event days. Gold presents a clearer geopolitical signal, while silver appears noisier and more influenced by broader macro and industrial dynamics.

## Result 4: Event-window behavior adds context beyond single-day moves

During extreme stress (2008), volatility surged for both metals, but silver expanded much more (about 6.14% vs 3.47% for gold), implying wider uncertainty in silver forecasts.

![Event Study Charts](src/screenshots/5.1.Event%20Study.png)

The event-window curves show that pre-event and post-event behavior is path-dependent, especially for silver. Looking beyond event-day returns helps distinguish temporary shocks from follow-through regimes.

![Event Study Tables](src/screenshots/5.2.Event%20Study.png)

The event-level summary tables make the same point with traceable records by event date and event type. This structure improves interpretability because each aggregate claim can be tied back to observed event windows.

## Result 5: Forecasting pipeline translates analysis into decision-ready outputs

Modeling approach:
- [ARIMA (AutoRegressive Integrated Moving Average)](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average): forecasts a series using its own history, differencing, and past error structure.
- [ARIMAX (ARIMA with exogenous inputs)](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model#Exogenous_inputs): extends ARIMA by adding external drivers, such as a geopolitical event flag.

![ARIMA Forecasting Controls and Chart](src/screenshots/3.1.ARIMA%20Forecasting.png)

This view shows the interactive controls for selecting metal, ARIMA order, forecast horizon, and event exogenous input. It is designed to make model assumptions explicit before generating any output.

![ARIMA Forecasting Metrics and Output](src/screenshots/3.2.ARIMA%20Forecasting.png)

The forecast panel reports backtest MAE and event coefficient context alongside the projected path, so accuracy and interpretation are presented together. This makes the output more practical for communication and decision support.

How I configured and tuned model settings:
- p controls how many lagged past values are used.
- d controls how much differencing is applied to stabilize trend/non-stationarity.
- q controls how many lagged error terms are used.
- Forecast horizon is how many future days are predicted.
- Train split determines how much history is used for training versus backtesting.
- Tuning process: set d first for stationarity, run a small grid over p and q, compare backtest MAE, and prefer the simplest model with comparable error to reduce overfitting.

I wrapped the full workflow in a Streamlit app with modular preprocessing, event-feature engineering, ARIMA/ARIMAX paths, MAE-based backtesting, and clear visual outputs. This project strengthened my ability to turn statistical analysis into an interpretable, decision-ready forecasting product.