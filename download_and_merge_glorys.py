"""
download_and_merge_glorys.py

Run this on YOUR machine (the one where you ran `copernicusmarine login`),
NOT in a sandboxed environment without internet access to Copernicus Marine.

What it does:
1. Reads your existing surface_data_march.csv (date, lat, lon, SST, SSS, SSH,
   u_curr, v_curr, u_wind, v_wind).
2. Downloads GLORYS12V1 daily potential temperature (thetao) from Copernicus
   Marine for exactly the bounding box + dates you need.
3. Interpolates GLORYS onto your exact (date, lat, lon) points.
4. Extracts temperature at 8 depths: 0, 5, 10, 20, 30, 50, 75, 100 m.
5. Merges everything into one final training CSV with 18 columns:
   date, lat, lon, SST, SSS, SSH, u_curr, v_curr, u_wind, v_wind,
   T_0, T_5, T_10, T_20, T_30, T_50, T_75, T_100

Prereqs (already done based on your last message):
    pip install copernicusmarine xarray netCDF4
    copernicusmarine login
"""

import pandas as pd
import xarray as xr
import copernicusmarine
import numpy as np

# ---- CONFIG ----
SURFACE_CSV = "surface_data_march.csv"      # your existing input file
OUTPUT_CSV = "training_data_full.csv"       # final merged output
GLORYS_NC = "glorys_thetao_subset.nc"       # intermediate downloaded file

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLE = "thetao"
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100]    # meters

# ---- STEP 1: Read your surface data to figure out date/bbox needed ----
surface_df = pd.read_csv(SURFACE_CSV, parse_dates=["date"])

min_lat, max_lat = surface_df["lat"].min(), surface_df["lat"].max()
min_lon, max_lon = surface_df["lon"].min(), surface_df["lon"].max()
min_date, max_date = surface_df["date"].min(), surface_df["date"].max()

# Pad the bounding box slightly so interpolation has neighbors on all sides
pad = 0.5
min_lat, max_lat = min_lat - pad, max_lat + pad
min_lon, max_lon = min_lon - pad, max_lon + pad

print(f"Requesting GLORYS thetao for:")
print(f"  lat: {min_lat:.2f} to {max_lat:.2f}")
print(f"  lon: {min_lon:.2f} to {max_lon:.2f}")
print(f"  dates: {min_date.date()} to {max_date.date()}")
print(f"  depths: {DEPTHS} m")

# ---- STEP 2: Download the subset from Copernicus Marine ----
copernicusmarine.subset(
    dataset_id=DATASET_ID,
    variables=[VARIABLE],
    minimum_longitude=float(min_lon),
    maximum_longitude=float(max_lon),
    minimum_latitude=float(min_lat),
    maximum_latitude=float(max_lat),
    start_datetime=min_date.strftime("%Y-%m-%dT00:00:00"),
    end_datetime=max_date.strftime("%Y-%m-%dT00:00:00"),
    minimum_depth=0,
    maximum_depth=150,   # GLORYS jumps from ~92m to ~109.73m, so 100m needs
                         # that 109.73m level included to interpolate correctly.
                         # 150m gives comfortable headroom.
    output_filename=GLORYS_NC,
    overwrite=True,   # v2 renamed force_download/overwrite_output_data to just "overwrite"
)

print(f"Downloaded GLORYS subset to {GLORYS_NC}")

# ---- STEP 3: Open the downloaded file ----
ds = xr.open_dataset(GLORYS_NC)

# GLORYS depth coordinate is usually named "depth"; confirm and rename if needed
depth_dim = "depth" if "depth" in ds.coords else list(ds.coords)[-1]

# ---- STEP 4: Interpolate onto exact points, for each of your 7 dates ----
records = []

for _, row in surface_df.iterrows():
    target_time = row["date"]
    target_lat = row["lat"]
    target_lon = row["lon"]

    # Interpolate thetao at this exact date/lat/lon, for all requested depths
    point = ds[VARIABLE].interp(
        time=target_time,
        latitude=target_lat,
        longitude=target_lon,
        **{depth_dim: DEPTHS},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
        # NOTE: this also allows lat/lon/time to extrapolate if a point falls
        # slightly outside the downloaded subset. Since we padded the bbox by
        # 0.5 deg and used your CSV's own min/max dates, this should only ever
        # kick in for the depth dimension in practice — but if you see any
        # oddly extreme temperature values after this fix, check whether a
        # point is silently extrapolating lat/lon instead of depth.
    )

    temps = point.values  # array of length len(DEPTHS)

    record = row.to_dict()
    for depth, temp in zip(DEPTHS, temps):
        record[f"T_{depth}"] = float(temp) if not np.isnan(temp) else None

    records.append(record)

# ---- STEP 5: Save merged CSV ----
merged_df = pd.DataFrame(records)
merged_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved merged training data to {OUTPUT_CSV}")
print(merged_df.head())
