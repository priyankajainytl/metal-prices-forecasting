import pandas as pd

def load_raw_data() -> pd.DataFrame:
    """Load raw CSV files and merge into a single dataframe."""
    gold = pd.read_csv("/Users/akshayjain/Documents/projects/metal-prices-forecasting/src/data/raw/Gold_Spot_Price_Daily.csv", parse_dates=['DATE'])
    silver = pd.read_csv("/Users/akshayjain/Documents/projects/metal-prices-forecasting/src/data/raw/Silver_Spot_Price_Daily.csv", parse_dates=['DATE'])
    gpr = pd.read_csv("/Users/akshayjain/Documents/projects/metal-prices-forecasting/src/data/raw/Geopolitical_Risk_Index_Daily.csv", parse_dates=['DATE'])
    
    # Merge datasets on Date
    merged = gold.merge(silver, on='DATE', suffixes=('_gold', '_silver'))
    merged = merged.merge(gpr, on='DATE')
    
    return merged

loaded_data = load_raw_data()
print(loaded_data.head())