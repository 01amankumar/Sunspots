import sys
import numpy as np
import torch

from data import load_resolution, chronological_split, make_windows, DEFAULT_LOOKBACK
from models import ResidualLSTM, LSTMForecaster
from train_all_datasets import metrics, train_torch_model, predict_torch, TRAIN_CFG

def test(resolution, model_cls, name, seeds=(42, 123, 2024)):
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
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = model_cls(lookback=lookback, horizon=1) if model_cls is ResidualLSTM else model_cls(hidden=64, horizon=1)
        model, _ = train_torch_model(model, X_train, y_train, X_val, y_val, **cfg)
        pred = predict_torch(model, X_test).reshape(-1)
        preds.append(pred)
        m = metrics(y_test, pred)
        print(f"  [{resolution}] {name} seed={seed}: RMSE={m['RMSE']:.4f}")
    ens = np.mean(preds, axis=0)
    m = metrics(y_test, ens)
    print(f"[{resolution}] {name} 3-seed ensemble: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")
    return m["RMSE"]

if __name__ == "__main__":
    for res in ["daily"]:
        r1 = test(res, ResidualLSTM, "ResidualLSTM")
        r2 = test(res, LSTMForecaster, "LSTM")
        print(f"==> {res}: ResidualLSTM={r1:.4f}  LSTM={r2:.4f}  {'WINS' if r1<r2 else 'loses'}\n")
