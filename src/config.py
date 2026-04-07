"""Shared configuration and constants for the project."""

DATE_COLUMN = "Date"
RAW_DATE_COLUMN = "DATE"
EVENT_COLUMN = "EVENT"
GPR_COLUMN = "GPR"
PRICE_GOLD_COLUMN = "Price_gold"
PRICE_SILVER_COLUMN = "Price_silver"
PRICE_COLUMNS = (PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN)
REQUIRED_MODEL_COLUMNS = (DATE_COLUMN, PRICE_GOLD_COLUMN, PRICE_SILVER_COLUMN, GPR_COLUMN)

RAW_FILE_NAMES = {
    PRICE_GOLD_COLUMN: "Gold_Spot_Price_Daily.csv",
    PRICE_SILVER_COLUMN: "Silver_Spot_Price_Daily.csv",
    GPR_COLUMN: "Geopolitical_Risk_Index_Daily.csv",
}

EVENT_TYPE_KEYWORDS = {
    "War / Military Conflict": ("war", "invades", "bombing", "desert storm"),
    "Terrorism / Security Threat": ("terror", "bombings", "threat"),
    "Geopolitical Tension": ("tension", "approval", "escalate"),
}

OUTLIER_SIGMA_THRESHOLD = 3.0
FEATURE_LAG_DAYS = (1, 7)
FEATURE_ROLLING_WINDOW = 30
DEFAULT_VOLATILITY_WINDOW = 30
DEFAULT_EVENT_WINDOW = 5
DEFAULT_FORECAST_HORIZON = 30
DEFAULT_TRAIN_FRACTION = 0.90
DEFAULT_ARIMA_ORDER = (2, 1, 2)
MIN_FORECAST_POINTS = 60
MIN_TRAIN_POINTS = 30