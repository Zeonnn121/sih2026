# Ocean Temperature Profile Model

This project trains a small PyTorch neural network to predict subsurface ocean temperature profiles at 5, 10, 20, 30, 50, and 75 meters. The model uses surface and satellite measurements from a CSV file and saves the trained model with its feature and target scalers.

## Prerequisites

- Git
- Python 3.10 or newer
- Internet access for installing Python packages

Python 3.13 is used by the development environment, but a fresh virtual environment should be created after cloning. Do not copy or commit the local `Scripts/`, `Lib/`, or `Include/` folders.

## 1. Clone the repository

Replace `<repository-url>` with the URL of this GitHub repository:

```powershell
git clone <repository-url>
cd ocean-env
```

If the repository is cloned into a different folder, use that folder name in the `cd` command.

## 2. Create a virtual environment

On Windows PowerShell:

```powershell
py -3 -m venv .venv
```

On macOS or Linux:

```bash
python3 -m venv .venv
```

## 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

macOS or Linux:

```bash
source .venv/bin/activate
```

When activation succeeds, the terminal prompt normally starts with `(.venv)`.

## 4. Install the dependencies

Upgrade `pip`, then install the packages used by `train_model_torch.py`:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn torch
```

Verify that PyTorch is available:

```powershell
python -c "import torch; print(torch.__version__)"
```

## 5. Check the training data

The default dataset included in this repository is `training_data_full.csv`. It must contain these feature columns:

```text
date, lat, lon, SST, SSS, SSH, u_curr, v_curr, u_wind, v_wind
```

It must also contain these target columns:

```text
T_5, T_10, T_20, T_30, T_50, T_75
```

The optional `T_0` and `T_100` columns are removed automatically. Missing target values are supported and are ignored by the masked loss. Rows with missing feature values are excluded from training.

## 6. Run the training script

From the repository root, run:

```powershell
python train_model_torch.py --data training_data_full.csv
```

The script will:

1. Load and prepare the CSV data.
2. Standardize the input features and temperature targets.
3. Run 5-fold cross-validation.
4. Print the mean validation MAE in degrees Celsius.
5. Train a final model using all usable rows.
6. Save the model and scalers to `model.pt`.

## 7. Customize training

Use `--help` to view all available options:

```powershell
python train_model_torch.py --help
```

Example with 500 training epochs, a different learning rate, three folds, and a custom output path:

```powershell
python train_model_torch.py `
  --data training_data_full.csv `
  --epochs 500 `
  --lr 0.0005 `
  --folds 3 `
  --out model_custom.pt
```

The output file contains the trained weights, feature names, target names, and the statistics needed to standardize future predictions.

## 8. Use the trained model for predictions

After training, the saved `model.pt` can be used by `predict.py`. Check its command-line options first:

```powershell
python predict.py --help
```

## Deactivate the environment

When finished:

```powershell
deactivate
```

## Optional: regenerate the training dataset

`download_and_merge_glorys.py` can download GLORYS ocean-temperature data and merge it with `surface_data_march.csv`. This requires a Copernicus Marine account and login. Install its additional dependencies before using it:

```powershell
python -m pip install copernicusmarine xarray netCDF4
copernicusmarine login
python download_and_merge_glorys.py
```

The script produces `glorys_thetao_subset.nc` and `training_data_full.csv`. The existing CSV files in the repository are sufficient to run the training script, so this step is optional.

## Troubleshooting

### `FileNotFoundError` for the CSV

Make sure the command is run from the `ocean-env` directory, or provide an absolute or relative path:

```powershell
python train_model_torch.py --data .\training_data_full.csv
```

### `ModuleNotFoundError`

Confirm that the virtual environment is active and reinstall the dependencies:

```powershell
python -m pip install numpy pandas scikit-learn torch
```

### CUDA or GPU issues

The script automatically uses CUDA when PyTorch detects a compatible GPU. Otherwise, it trains on the CPU. No code or command-line change is required.
