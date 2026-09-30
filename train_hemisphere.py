"""
Trains the full validated model suite (5 traditional DL baselines, PatchTST,
and the proposed ResidualLSTM as a 3-seed ensemble) separately on the North
and South hemispheric 13-month-smoothed sunspot number series, using the
151-year (1874-2025) Veronig-et-al.-2021 + SILSO spliced record (see
hem_data.py) -- versus the ~33 years available from SILSO's own hemispheric
product alone.

Same leak-free, RevIN-normalized, chronological-70/15/15-split methodology
as the total-SN study (train_all_datasets.py / train_final_proposed.py).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hem_data import build_hemisphere_series
from data import chronological_split, make_windows
from models import LSTMForecaster, RNNForecaster, CNN1DForecaster, PatchTST, ResidualLSTM, NaivePersistence
from train_all_datasets import metrics, train_torch_model, predict_torch, OUT_ROOT

LOOKBACK = 14
PATCH_CFG = dict(patch_len=7, stride=1)
TRAIN_CFG = dict(epochs=300, patience=25, batch_size=64, lr=1e-3)
SEEDS = [42, 123, 2024]


def build_models():
    return {
        "LSTM":    lambda: LSTMForecaster(hidden=64, horizon=1),
        "GRU":     lambda: RNNForecaster(cell="gru", hidden=64, horizon=1),
        "RNN":     lambda: RNNForecaster(cell="rnn", hidden=64, horizon=1),
        "BiLSTM":  lambda: RNNForecaster(cell="lstm", hidden=64, horizon=1, bidirectional=True),
        "CNN1D":   lambda: CNN1DForecaster(lookback=LOOKBACK, horizon=1),
        "PatchTST": lambda: PatchTST(lookback=LOOKBACK, patch_len=PATCH_CFG["patch_len"],
                                      stride=PATCH_CFG["stride"], d_model=64, nhead=4,
                                      num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1),
    }


def run_hemisphere(hemisphere: str):
    print(f"\n{'='*70}\n{hemisphere.upper()} (13-month smoothed, 1874-2025 spliced record)\n{'='*70}")
    out_dir = OUT_ROOT / f"hemisphere_{hemisphere}"
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    series = build_hemisphere_series(hemisphere, smoothed=True)
    train_s, val_s, test_s = chronological_split(series)
    print(f"n={len(series)}  train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    print(f"train: {train_s.index.min().date()} - {train_s.index.max().date()}")
    print(f"val:   {val_s.index.min().date()} - {val_s.index.max().date()}")
    print(f"test:  {test_s.index.min().date()} - {test_s.index.max().date()}")

    train_arr, val_arr, test_arr = train_s.values, val_s.values, test_s.values
    X_train, y_train = make_windows(train_arr, LOOKBACK, horizon=1)
    val_ctx = np.concatenate([train_arr[-LOOKBACK:], val_arr])
    X_val, y_val = make_windows(val_ctx, LOOKBACK, horizon=1)
    test_ctx = np.concatenate([val_arr[-LOOKBACK:], test_arr])
    X_test, y_test = make_windows(test_ctx, LOOKBACK, horizon=1)
    test_dates = test_s.index[:len(y_test)]

    results = []
    predictions = {"date": test_dates, "actual": y_test.reshape(-1)}

    naive_pred = NaivePersistence().predict(X_test).reshape(-1)
    predictions["Naive Persistence"] = naive_pred
    results.append({"Model": "Naive Persistence", **metrics(y_test, naive_pred)})
    print(f"Naive Persistence: RMSE={results[-1]['RMSE']:.4f}")

    for name, factory in build_models().items():
        t0 = time.time()
        torch.manual_seed(42)
        np.random.seed(42)
        model = factory()
        model, val_loss = train_torch_model(model, X_train, y_train, X_val, y_val, **TRAIN_CFG)
        pred = predict_torch(model, X_test).reshape(-1)
        predictions[name] = pred
        m = metrics(y_test, pred)
        m["train_seconds"] = round(time.time() - t0, 1)
        results.append({"Model": name, **m})
        torch.save(model.state_dict(), out_dir / "models" / f"{name}.pt")
        print(f"{name}: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f} ({m['train_seconds']}s)")

    seed_preds = []
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ResidualLSTM(lookback=LOOKBACK, horizon=1)
        model, val_loss = train_torch_model(model, X_train, y_train, X_val, y_val, **TRAIN_CFG)
        pred = predict_torch(model, X_test).reshape(-1)
        seed_preds.append(pred)
        sm = metrics(y_test, pred)
        print(f"  ResidualLSTM seed={seed}: RMSE={sm['RMSE']:.4f} ({time.time()-t0:.1f}s)")
        torch.save(model.state_dict(), out_dir / "models" / f"ResidualLSTM_seed{seed}.pt")
    ens_pred = np.mean(seed_preds, axis=0)
    predictions["ResidualLSTM (Proposed)"] = ens_pred
    m = metrics(y_test, ens_pred)
    results.append({"Model": "ResidualLSTM (Proposed)", **m})
    print(f"ResidualLSTM (Proposed) [3-seed ensemble]: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "split_info.json", "w") as f:
        json.dump({
            "n_total": len(series), "lookback": LOOKBACK,
            "train_range": [str(train_s.index.min()), str(train_s.index.max())],
            "val_range": [str(val_s.index.min()), str(val_s.index.max())],
            "test_range": [str(test_s.index.min()), str(test_s.index.max())],
        }, f, indent=2)

    print(f"\n{hemisphere} results:\n{results_df.to_string(index=False)}")
    return results_df


if __name__ == "__main__":
    for hemi in ["north", "south"]:
        run_hemisphere(hemi)
    print("\nBoth hemispheres done.")
