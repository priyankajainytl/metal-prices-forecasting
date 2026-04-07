import pandas as pd
import os

def load_raw_data() -> pd.DataFrame:
    """Load raw CSV files and merge into a single dataframe."""
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    required_files = [
        'Gold_Spot_Price_Daily.csv',
        'Silver_Spot_Price_Daily.csv',
        'Geopolitical_Risk_Index_Daily.csv',
    ]
    missing_files = [name for name in required_files if not os.path.exists(os.path.join(raw_dir, name))]
    if missing_files:
        missing_str = ', '.join(missing_files)
        raise FileNotFoundError(
            f"Missing raw data file(s): {missing_str}. Expected in directory: {raw_dir}"
        )

    gold = pd.read_csv(
        os.path.join(raw_dir, 'Gold_Spot_Price_Daily.csv'),
        parse_dates=['DATE'],
        dayfirst=True,
    )
    silver = pd.read_csv(
        os.path.join(raw_dir, 'Silver_Spot_Price_Daily.csv'),
        parse_dates=['DATE'],
        dayfirst=True,
    )
    gpr = pd.read_csv(
        os.path.join(raw_dir, 'Geopolitical_Risk_Index_Daily.csv'),
        parse_dates=['DATE'],
        dayfirst=True,
    )

    gold = gold[['DATE', 'GOLD_PRICE']].rename(columns={'GOLD_PRICE': 'Price_gold'})
    silver = silver[['DATE', 'SILVER_PRICE']].rename(columns={'SILVER_PRICE': 'Price_silver'})
    gpr = gpr[['DATE', 'GPRD', 'EVENT']].rename(columns={'GPRD': 'GPR'})

    gold['Price_gold'] = pd.to_numeric(gold['Price_gold'], errors='coerce')
    silver['Price_silver'] = pd.to_numeric(silver['Price_silver'], errors='coerce')
    gpr['GPR'] = pd.to_numeric(gpr['GPR'], errors='coerce')
    gpr['EVENT'] = gpr['EVENT'].fillna('').astype(str)
    
    # Merge datasets on Date
    merged = gold.merge(silver, on='DATE', how='inner')
    merged = merged.merge(gpr, on='DATE', how='inner')
    merged.sort_values('DATE', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged