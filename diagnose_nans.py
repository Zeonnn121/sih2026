"""
diagnose_nans.py

Quick diagnostic: shows exactly which depth columns are NaN, how often,
and prints a few full rows so we can see the pattern.
"""

import pandas as pd

INPUT_CSV = "training_data_full.csv"
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100]
TEMP_COLS = [f"T_{d}" for d in DEPTHS]

df = pd.read_csv(INPUT_CSV, parse_dates=["date"])

print(f"Total rows: {len(df)}\n")

print("NaN count per depth column:")
for col in TEMP_COLS:
    if col not in df.columns:
        print(f"  {col}: COLUMN MISSING ENTIRELY")
        continue
    n_nan = df[col].isna().sum()
    print(f"  {col}: {n_nan} / {len(df)} NaN")

print("\nRows where ALL temperature columns are NaN:")
all_nan_mask = df[TEMP_COLS].isna().all(axis=1)
print(f"  {all_nan_mask.sum()} / {len(df)} rows")

print("\nRows where SOME (but not all) temperature columns are NaN:")
some_nan_mask = df[TEMP_COLS].isna().any(axis=1) & ~all_nan_mask
print(f"  {some_nan_mask.sum()} / {len(df)} rows")

print("\nSample of 5 full rows (to see the actual pattern):")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df[["date", "lat", "lon"] + TEMP_COLS].head(5))
