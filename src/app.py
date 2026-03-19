from preprocess import clean_data, feature_engineering, save_features
from data_loader import load_raw_data
import pandas as pd

def main():
    # Step 1: Load raw data
    raw_data: pd.DataFrame = load_raw_data()
    
    # Step 2: Clean data
    cleaned_data: pd.DataFrame = clean_data(raw_data)
    
    # Step 3: Feature engineering
    features: pd.DataFrame = feature_engineering(cleaned_data)
    
    # Step 4: Save features
    save_features(features)

if __name__ == "__main__":
    main()