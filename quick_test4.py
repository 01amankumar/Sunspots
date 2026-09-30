import numpy as np
import torch

from data import load_resolution, chronological_split, make_windows, DEFAULT_LOOKBACK
from models import RNNForecaster
from train_all_datasets import metrics, train_torch_model, predict_torch, TRAIN_CFG

resolution = "monthly_mean"
lookback = DEFAULT_LOOKBACK[resolution]
cfg = TRAIN_CFG[resolution]
series = load_resolution(resolution)
train_s, val_s, test_s = chronological_split(series)
train_arr, val_arr, test_arr = train_s.values, val_s.values, test_s.values
X_train, y_train = make_windows(train_arr, lookback, horizon=1)
val_ctx = np.concatenate([train_arr[-lookback:], val_arr])
X_val, y_val = make_windows(val_ctx, lookback, horizon=1)
test_ctx = np.concatenate([val_arr[-lookback:], test_arr])
X_test, y_test = make_windows(test_ctx, lookback, horizon=1)

preds = []
for seed in (42, 123, 2024):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RNNForecaster(cell="gru", hidden=64, horizon=1)
    model, _ = train_torch_model(model, X_train, y_train, X_val, y_val, **cfg)
    pred = predict_torch(model, X_test).reshape(-1)
    preds.append(pred)
    print(f"  GRU seed={seed}: RMSE={metrics(y_test, pred)['RMSE']:.4f}")

ens = np.mean(preds, axis=0)
m = metrics(y_test, ens)
print(f"\nGRU 3-seed ensemble: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")
