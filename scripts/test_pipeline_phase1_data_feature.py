import os
import pandas as pd

# Make sure Python can find your src package
import sys
sys.path.append(os.path.abspath("src"))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
# === CONFIG ===
DATA_PATH = r"E:\CHURN\data\raw\Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

def main():
    print("=== Testing Phase 1 : Load → preprocess → Build Features ===")

    # 1. Load Data
    print("\n[1] Loading Data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print("n\[2] Preprocessing data...")
    df_clean  = preprocess_data(df, target_col = TARGET_COL)
    print("Data after Preprocessing . Shape: {df_clean.shape}")

    # 3. Build Features
    print(f"\n[3] Building Features...")
    df_features = build_features(df_clean, target_col = TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n Phase 1 piipeline completed succesfully !")

if __name__ == "__main__":
    main()