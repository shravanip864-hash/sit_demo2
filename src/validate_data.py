from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["area_sqft", "bedrooms", "age_years", "distance_km", "price_lakh"]

def validate_data(path="data/housing.csv"):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if (df[["area_sqft", "bedrooms", "age_years", "distance_km"]] <= 0).any().any():
        raise ValueError("Input features must be greater than zero")
    if (df["price_lakh"] <= 0).any():
        raise ValueError("Target must be greater than zero")
    return df

if __name__ == "__main__":
    print(f"Data validation passed: {len(validate_data())} rows")
