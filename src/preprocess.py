# src/preprocess.py
import pandas as pd
import os

# -----------------------------
# Step 1: Clean Data
# -----------------------------
def clean_data(df):
    """
    Fill missing values and remove extreme outliers.
    """
    required_cols = {'Price_gold', 'Price_silver', 'GPR'}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for cleaning: {sorted(missing_cols)}")

    df = df.copy()

    # Forward fill missing values
    df.ffill(inplace=True)

    # Remove extreme outliers (beyond 3 standard deviations)
    for col in ['Price_gold', 'Price_silver', 'GPR']:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]

    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
def feature_engineering(df):
    """
    Create lag features and rolling statistics for time series.
    """
    required_cols = {'Price_gold', 'Price_silver', 'GPR'}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing_cols)}")

    df = df.copy()

    df['gold_lag1'] = df['Price_gold'].shift(1)
    df['gold_lag7'] = df['Price_gold'].shift(7)
    df['silver_lag1'] = df['Price_silver'].shift(1)
    df['silver_lag7'] = df['Price_silver'].shift(7)
    df['GPR_lag7'] = df['GPR'].shift(7)

    # Rolling mean and volatility (30-day)
    df['gold_roll_mean30'] = df['Price_gold'].rolling(30).mean()
    df['gold_volatility30'] = df['Price_gold'].rolling(30).std()
    df['silver_roll_mean30'] = df['Price_silver'].rolling(30).mean()
    df['silver_volatility30'] = df['Price_silver'].rolling(30).std()

    # Drop rows with NaNs created by shift/rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------
# Step 3: Save Features
# -----------------------------
def save_features(df, path=None):
    """
    Save the cleaned and feature-engineered DataFrame to CSV.
    Creates the folder if it doesn't exist.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'features.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Features saved to {path}")