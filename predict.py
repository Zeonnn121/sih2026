"""
predict.py
Loads a trained model.pt (from train_model_torch.py) and predicts subsurface
temperature profiles (T_5..T_75) for new rows of surface/satellite data.

Usage:
    # Predict on a CSV of new rows (must have the same feature columns:
    # lat, lon, SST, SSS, SSH, u_curr, v_curr, u_wind, v_wind, date)
    python predict.py --model model.pt --data new_points.csv --out predictions.csv

    # Or sanity-check against your original training data
    # (compares predictions vs. actual observed values where available)
    python predict.py --model model.pt --data training_data_full.csv --compare
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class ProfileNet(nn.Module):
    def __init__(self, n_features: int, n_targets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_targets),
        )

    def forward(self, x):
        return self.net(x)


def load_model(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    target_cols = checkpoint["target_cols"]
    model = ProfileNet(len(feature_cols), len(target_cols))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict(model, checkpoint, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = checkpoint["feature_cols"]
    target_cols = checkpoint["target_cols"]
    x_mean, x_std = checkpoint["x_mean"], checkpoint["x_std"]
    y_mean, y_std = checkpoint["y_mean"], checkpoint["y_std"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    X = df[feature_cols].values.astype(np.float64)
    if np.isnan(X).any():
        bad_rows = df[np.isnan(X).any(axis=1)]
        raise ValueError(
            f"{len(bad_rows)} row(s) have missing feature values -- "
            f"fill or drop them before predicting.\nAffected rows:\n{bad_rows}"
        )

    X_scaled = (X - x_mean) / x_std
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32))
    pred = pred_scaled.numpy() * y_std + y_mean

    pred_df = pd.DataFrame(pred, columns=[f"pred_{c}" for c in target_cols], index=df.index)
    return pred_df, target_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt", help="Path to trained model.pt")
    parser.add_argument("--data", required=True, help="CSV of rows to predict on")
    parser.add_argument("--out", default="predictions.csv", help="Where to save predictions")
    parser.add_argument(
        "--compare", action="store_true",
        help="If the input CSV also has true T_5..T_75 columns, show predicted vs actual"
    )
    args = parser.parse_args()

    model, checkpoint = load_model(args.model)
    df = pd.read_csv(args.data, parse_dates=["date"]) if "date" in pd.read_csv(args.data, nrows=1).columns else pd.read_csv(args.data)
    if "date" in df.columns:
        df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear

    pred_df, target_cols = predict(model, checkpoint, df)
    result = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    if args.compare:
        print("Predicted vs. actual (only rows/depths with real observed data shown):\n")
        for t in target_cols:
            if t in df.columns:
                actual = df[t].values
                predicted = pred_df[f"pred_{t}"].values
                valid = ~np.isnan(actual)
                if valid.sum() == 0:
                    continue
                err = np.abs(actual[valid] - predicted[valid])
                print(f"{t}: n={valid.sum()}, MAE={err.mean():.3f} °C, max_err={err.max():.3f} °C")
        print()

    result.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")
    print(f"\nPreview:\n{result.head()}")


if __name__ == "__main__":
    main()
