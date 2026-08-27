"""
train_model.py (PyTorch)
Trains a single multi-output network to predict subsurface ocean temperature
profiles (T_5..T_75) from surface/satellite variables.

Assumptions (confirmed with user):
- T_0 and T_100 are 100% missing -> dropped entirely, not used as features or targets.
- Remaining NaNs in T_5..T_75 correspond to land grid points (no ocean at that
  depth/location) -> legitimate absences. Handled via a MASKED loss: each
  row contributes to the loss only for the depths it actually has data for,
  instead of dropping rows or imputing.

Given the very small dataset (~98 rows), the model is intentionally small
and regularized (dropout + weight decay), and we use K-fold CV to get a
realistic performance estimate before the final fit.

Usage:
    python train_model.py --data training_data.csv --epochs 300
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

FEATURE_COLS = [
    "lat", "lon", "SST", "SSS", "SSH",
    "u_curr", "v_curr", "u_wind", "v_wind",
]
TARGET_COLS = ["T_5", "T_10", "T_20", "T_30", "T_50", "T_75"]
DROP_COLS = ["T_0", "T_100"]  # confirmed 100% missing


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["doy"] = df["date"].dt.dayofyear
    return df


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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    MSE that ignores entries where mask == 0 (i.e. missing/land-point targets).
    """
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


def standardize(arr: np.ndarray):
    """Nan-aware standardization. Returns scaled array, mean, std (per column)."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    scaled = (arr - mean) / std
    return scaled, mean, std


def train_one_fold(X_train, Y_train, M_train, X_val, Y_val, M_val, epochs, lr, device):
    model = ProfileNet(X_train.shape[1], Y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_train, dtype=torch.float32, device=device)
    Mt = torch.tensor(M_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_val, dtype=torch.float32, device=device)
    Mv = torch.tensor(M_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    patience, patience_ctr = 30, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(Xt)
        loss = masked_mse(pred, Yt, Mt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = masked_mse(val_pred, Yv, Mv).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_prepare(args.data)
    print(f"Loaded {len(df)} rows after dropping T_0/T_100 columns.")

    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    Y_raw = df[TARGET_COLS].values.astype(np.float64)

    # Drop rows missing any FEATURE (features should be complete; targets may not be)
    feat_ok = ~np.isnan(X_raw).any(axis=1)
    X_raw, Y_raw = X_raw[feat_ok], Y_raw[feat_ok]
    print(f"{len(X_raw)} rows have complete features (usable for training).\n")

    mask = ~np.isnan(Y_raw)  # 1 where target is real data, 0 where land/missing
    Y_filled = np.nan_to_num(Y_raw, nan=0.0)  # placeholder value, ignored via mask

    X_scaled, x_mean, x_std = standardize(X_raw)
    # standardize targets using only valid (masked) values, per-column
    y_mean = np.array([np.nanmean(Y_raw[:, i]) for i in range(Y_raw.shape[1])])
    y_std = np.array([np.nanstd(Y_raw[:, i]) or 1.0 for i in range(Y_raw.shape[1])])
    Y_scaled = (Y_filled - y_mean) / y_std

    # --- K-fold CV to estimate real-world performance ---
    n_splits = min(args.folds, len(X_scaled))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []

    print(f"Running {n_splits}-fold cross-validation...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
        model, _ = train_one_fold(
            X_scaled[tr_idx], Y_scaled[tr_idx], mask[tr_idx].astype(np.float32),
            X_scaled[val_idx], Y_scaled[val_idx], mask[val_idx].astype(np.float32),
            args.epochs, args.lr, device,
        )
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_scaled[val_idx], dtype=torch.float32, device=device))
            pred = pred_scaled.cpu().numpy() * y_std + y_mean
        # Use Y_filled (NaNs already replaced with 0), NOT Y_raw, because
        # NaN * 0 is still NaN in numpy -- masking only works cleanly on
        # arrays with no NaNs left in them.
        true = Y_filled[val_idx]
        m = mask[val_idx]
        abs_err = np.abs(pred - true) * m
        mae = abs_err.sum() / max(m.sum(), 1)
        fold_maes.append(mae)
        print(f"  fold {fold+1}: MAE={mae:.3f} °C (valid targets: {int(m.sum())})")

    print(f"\nMean CV MAE across folds: {np.mean(fold_maes):.3f} °C\n")

    # --- Final fit on all data ---
    print("Training final model on full dataset...")
    final_model, _ = train_one_fold(
        X_scaled, Y_scaled, mask.astype(np.float32),
        X_scaled, Y_scaled, mask.astype(np.float32),  # no held-out set left; same data as "val" just for early stop bookkeeping
        args.epochs, args.lr, device,
    )

    torch.save({
        "model_state": final_model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
    }, args.out)
    print(f"Saved model + scalers to {args.out}")


if __name__ == "__main__":
    main()