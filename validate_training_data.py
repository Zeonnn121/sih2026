"""
validate_training_data.py

Run this on your machine, in the same folder as training_data_full.csv.
Checks:
  1. Missing values (NaNs) per column
  2. Basic range sanity checks for each variable
  3. Depth profile plots (temperature vs depth) for a few sample points
  4. Saves a plot + a text summary you can eyeball before training

Usage:
    python validate_training_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "training_data_full.csv"
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100]
TEMP_COLS = [f"T_{d}" for d in DEPTHS]

df = pd.read_csv(INPUT_CSV, parse_dates=["date"])

print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}\n")

# ---- 1. Missing values ----
print("=" * 50)
print("MISSING VALUES PER COLUMN")
print("=" * 50)
nan_counts = df.isna().sum()
nan_pct = (nan_counts / len(df) * 100).round(1)
missing_report = pd.DataFrame({"nan_count": nan_counts, "nan_pct": nan_pct})
print(missing_report[missing_report["nan_count"] > 0])
if missing_report["nan_count"].sum() == 0:
    print("No missing values found.")

rows_with_nan = df[df.isna().any(axis=1)]
if len(rows_with_nan) > 0:
    print(f"\n{len(rows_with_nan)} rows have at least one NaN.")
    print("Sample of affected rows:")
    print(rows_with_nan[["date", "lat", "lon"]].head(10))

# ---- 2. Range sanity checks ----
print("\n" + "=" * 50)
print("RANGE SANITY CHECKS")
print("=" * 50)

expected_ranges = {
    "SST": (-2, 35),        # deg C
    "SSS": (30, 40),        # psu, rough North Indian Ocean range
    "SSH": (-1, 1),         # m, typical SLA range
    "u_curr": (-2, 2),      # m/s
    "v_curr": (-2, 2),      # m/s
    "u_wind": (-30, 30),    # m/s
    "v_wind": (-30, 30),    # m/s
}
for depth in DEPTHS:
    # Temperature should generally decrease or stay similar with depth,
    # and stay within plausible ocean ranges
    expected_ranges[f"T_{depth}"] = (-2, 35)

for col, (low, high) in expected_ranges.items():
    if col not in df.columns:
        print(f"  [MISSING COLUMN] {col} not found in CSV")
        continue
    actual_min = df[col].min()
    actual_max = df[col].max()
    flag = "OK"
    if actual_min < low or actual_max > high:
        flag = "OUT OF EXPECTED RANGE"
    print(f"  {col:10s} min={actual_min:8.3f}  max={actual_max:8.3f}  expected=({low},{high})  [{flag}]")

# ---- 3. Depth monotonicity check ----
print("\n" + "=" * 50)
print("DEPTH MONOTONICITY CHECK (temperature should generally decrease with depth)")
print("=" * 50)
non_monotonic_count = 0
for idx, row in df.iterrows():
    temps = row[TEMP_COLS].values.astype(float)
    valid = ~np.isnan(temps)
    if valid.sum() < 2:
        continue
    valid_temps = temps[valid]
    # Allow small non-monotonicity (thermocline effects) but flag big violations
    diffs = np.diff(valid_temps)
    if np.any(diffs > 2.0):  # temp jumps UP by more than 2C going deeper = suspicious
        non_monotonic_count += 1

print(f"{non_monotonic_count} / {len(df)} rows have a suspicious temperature increase with depth (>2C jump).")
print("A few small jumps can be normal ocean physics; many suggests a merge/interpolation issue.")

# ---- 4. Plot depth profiles for a handful of sample points ----
print("\n" + "=" * 50)
print("PLOTTING SAMPLE DEPTH PROFILES")
print("=" * 50)

sample_df = df.dropna(subset=TEMP_COLS).sample(min(6, len(df)), random_state=42)

plt.figure(figsize=(7, 6))
for _, row in sample_df.iterrows():
    temps = row[TEMP_COLS].values.astype(float)
    label = f"{row['date'].date()} ({row['lat']:.1f}, {row['lon']:.1f})"
    plt.plot(temps, DEPTHS, marker="o", label=label)

plt.gca().invert_yaxis()  # depth increases downward
plt.xlabel("Temperature (°C)")
plt.ylabel("Depth (m)")
plt.title("Sample Temperature-Depth Profiles")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("depth_profiles_sample.png", dpi=150)
print("Saved plot to depth_profiles_sample.png")

# ---- 5. Final summary ----
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Total rows: {len(df)}")
print(f"Rows with any NaN: {len(rows_with_nan)}")
print(f"Rows with suspicious depth non-monotonicity: {non_monotonic_count}")
print("Review the printed ranges above and depth_profiles_sample.png before training.")
