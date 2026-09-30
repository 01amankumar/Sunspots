"""
Trains the full model suite (5 traditional DL baselines + PatchTST transformer)
on all 4 SILSO resolutions (daily / monthly mean / monthly smoothed / yearly),
with identical leak-free, RevIN-normalized methodology across the board so
comparisons isolate architecture rather than preprocessing.

Saves, per resolution, under outputs/<resolution>/:
  - metrics.csv           RMSE/MAE/MSE/R2/MAPE per model on the held-out test split
  - predictions.csv       date, actual, <model predictions...> for every test point
  - models/*.pt           trained weights for each model
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_resolution, chronological_split, make_windows, RESOLUTIONS, DEFAULT_LOOKBACK
from models import LSTMForecaster, RNNForecaster, CNN1DForecaster, PatchTST, NaivePersistence

SEED = 42
OUT_ROOT = Path(r"C:\Users\amank\Downloads\sunspot_transformer\outputs")
DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# per-resolution training budget (daily has ~100x the rows of monthly, so fewer
# epochs / earlier stopping is both necessary and sufficient)
TRAIN_CFG = {
    "daily":             dict(epochs=60,  patience=8,  batch_size=256, lr=1e-3),
    "monthly_mean":      dict(epochs=300, patience=25, batch_size=64,  lr=1e-3),
    "monthly_smoothed":  dict(epochs=300, patience=25, batch_size=64,  lr=1e-3),
    "yearly":            dict(epochs=400, patience=35, batch_size=32,  lr=1e-3),
}

PATCH_CFG = {
    "daily":            dict(patch_len=10, stride=5),
    "monthly_mean":     dict(patch_len=7,  stride=1),
    "monthly_smoothed": dict(patch_len=7,  stride=1),
    "yearly":           dict(patch_len=3,  stride=1),
}


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "R2": r2, "MAPE": mape}


def train_torch_model(model, X_train, y_train, X_val, y_val, epochs, patience, batch_size, lr):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, device=DEVICE)
    yt = torch.tensor(y_train, device=DEVICE)
    Xv = torch.tensor(X_val, device=DEVICE)
    yv = torch.tensor(y_val, device=DEVICE)

    n = Xt.shape[0]
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, yv).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_torch(model, X, batch_size=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], device=DEVICE)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def build_models(lookback, patch_cfg):
    return {
        "LSTM":    lambda: LSTMForecaster(hidden=64, horizon=1),
        "GRU":     lambda: RNNForecaster(cell="gru", hidden=64, horizon=1),
        "RNN":     lambda: RNNForecaster(cell="rnn", hidden=64, horizon=1),
        "BiLSTM":  lambda: RNNForecaster(cell="lstm", hidden=64, horizon=1, bidirectional=True),
        "CNN1D":   lambda: CNN1DForecaster(lookback=lookback, horizon=1),
        "PatchTST": lambda: PatchTST(lookback=lookback, patch_len=patch_cfg["patch_len"],
                                      stride=patch_cfg["stride"], d_model=64, nhead=4,
                                      num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1),
    }


def run_resolution(resolution):
    print(f"\n{'='*70}\n{resolution}\n{'='*70}")
    out_dir = OUT_ROOT / resolution
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    series = load_resolution(resolution)
    train_s, val_s, test_s = chronological_split(series)
    lookback = DEFAULT_LOOKBACK[resolution]
    cfg = TRAIN_CFG[resolution]
    patch_cfg = PATCH_CFG[resolution]

    print(f"n={len(series)}  train={len(train_s)} val={len(val_s)} test={len(test_s)}  lookback={lookback}")
    print(f"train: {train_s.index.min()} - {train_s.index.max()}")
    print(f"test:  {test_s.index.min()} - {test_s.index.max()}")

    train_arr, val_arr, test_arr = train_s.values, val_s.values, test_s.values
    X_train, y_train = make_windows(train_arr, lookback, horizon=1)
    val_ctx = np.concatenate([train_arr[-lookback:], val_arr])
    X_val, y_val = make_windows(val_ctx, lookback, horizon=1)
    test_ctx = np.concatenate([val_arr[-lookback:], test_arr])
    X_test, y_test = make_windows(test_ctx, lookback, horizon=1)
    test_dates = test_s.index[:len(y_test)]

    results = []
    predictions = {"date": test_dates, "actual": y_test.reshape(-1)}

    naive_pred = NaivePersistence().predict(X_test).reshape(-1)
    predictions["Naive Persistence"] = naive_pred
    results.append({"Model": "Naive Persistence", **metrics(y_test, naive_pred)})
    print(f"Naive Persistence: RMSE={results[-1]['RMSE']:.4f}")

    model_factories = build_models(lookback, patch_cfg)
    trained = {}
    val_preds = {}
    for name, factory in model_factories.items():
        t0 = time.time()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = factory()
        model, val_loss = train_torch_model(model, X_train, y_train, X_val, y_val, **cfg)
        pred = predict_torch(model, X_test).reshape(-1)
        predictions[name] = pred
        val_preds[name] = predict_torch(model, X_val).reshape(-1)
        m = metrics(y_test, pred)
        m["train_seconds"] = round(time.time() - t0, 1)
        results.append({"Model": name, **m})
        trained[name] = model
        torch.save(model.state_dict(), out_dir / "models" / f"{name}.pt")
        print(f"{name}: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f} ({m['train_seconds']}s)")

    # Proposed model: pick, ON VALIDATION ONLY (never test), whichever of
    # {RevIN-LSTM alone, PatchTST alone, an unweighted average of the two}
    # generalizes best. This lets the transformer's contribution show up where
    # it earns its keep (larger, higher-resolution series) without forcing a
    # blend that would drag the result down on data too small to support it
    # (e.g. the 226-point yearly train split, see README/report discussion).
    y_val_flat = y_val.reshape(-1)
    candidates = {
        "LSTM": val_preds["LSTM"],
        "PatchTST": val_preds["PatchTST"],
        "Average": (val_preds["LSTM"] + val_preds["PatchTST"]) / 2,
    }
    val_mse = {k: float(np.mean((y_val_flat - v) ** 2)) for k, v in candidates.items()}
    chosen = min(val_mse, key=val_mse.get)
    ens_pred = {
        "LSTM": predictions["LSTM"],
        "PatchTST": predictions["PatchTST"],
        "Average": (predictions["LSTM"] + predictions["PatchTST"]) / 2,
    }[chosen]
    predictions["Proposed (LSTM+PatchTST)"] = ens_pred
    results.append({"Model": "Proposed (LSTM+PatchTST)", "selected_on_val": chosen, **metrics(y_test, ens_pred)})
    print(f"Proposed (LSTM+PatchTST) [val-selected: {chosen}, val_mse={val_mse}]: RMSE={results[-1]['RMSE']:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "split_info.json", "w") as f:
        json.dump({
            "n_total": len(series), "lookback": lookback,
            "train_range": [str(train_s.index.min()), str(train_s.index.max())],
            "val_range": [str(val_s.index.min()), str(val_s.index.max())],
            "test_range": [str(test_s.index.min()), str(test_s.index.max())],
        }, f, indent=2)

    print(f"\n{resolution} results:\n{results_df.to_string(index=False)}")
    return results_df


if __name__ == "__main__":
    for res in RESOLUTIONS:
        run_resolution(res)
    print("\nAll resolutions done.")
