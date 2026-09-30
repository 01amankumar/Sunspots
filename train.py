import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_series, chronological_split, make_windows
from models import LSTMForecaster, PatchTST, NaivePersistence

SEED = 42
CSV_PATH = r"C:\Users\amank\Downloads\SN_ms_tot_V2.0 (2).csv"
OUT_DIR = Path(r"C:\Users\amank\Downloads\sunspot_transformer\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1.0, None)  # avoid div-by-near-0 blowups
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "R2": r2, "MAPE": mape}


def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=300, patience=25,
                       batch_size=64, lr=1e-3, log_prefix=""):
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
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

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


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, device=DEVICE))
    return pred.cpu().numpy()


def run():
    series = load_series(CSV_PATH)
    train_s, val_s, test_s = chronological_split(series)
    print(f"train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    print(f"train: {train_s.index.min().date()} - {train_s.index.max().date()}")
    print(f"val:   {val_s.index.min().date()} - {val_s.index.max().date()}")
    print(f"test:  {test_s.index.min().date()} - {test_s.index.max().date()}")

    train_arr = train_s.values
    val_arr = val_s.values
    test_arr = test_s.values

    results = []
    predictions = {}
    best_by_model = {}

    lookbacks = [14, 24, 36]

    for lookback in lookbacks:
        # windows must be built with continuity across split boundaries so the
        # first predictions of val/test aren't thrown away
        val_ctx = np.concatenate([train_arr[-lookback:], val_arr])
        test_ctx = np.concatenate([val_arr[-lookback:], test_arr])

        X_train, y_train = make_windows(train_arr, lookback, horizon=1)
        X_val, y_val = make_windows(val_ctx, lookback, horizon=1)
        X_test, y_test = make_windows(test_ctx, lookback, horizon=1)

        # ---- Naive persistence (no training needed) ----
        naive = NaivePersistence()
        test_pred = naive.predict(X_test)
        m = metrics(y_test, test_pred)
        results.append({"Model": "Naive Persistence", "Lookback": lookback, "Split": "test", **m})

        # ---- LSTM baseline (RevIN-normalized, leak-free split) ----
        t0 = time.time()
        lstm = LSTMForecaster(hidden=64, horizon=1)
        lstm, val_loss = train_torch_model(lstm, X_train, y_train, X_val, y_val,
                                            log_prefix=f"LSTM L{lookback}")
        test_pred = predict_torch(lstm, X_test)
        m = metrics(y_test, test_pred)
        m["val_mse"] = val_loss
        m["train_seconds"] = round(time.time() - t0, 1)
        results.append({"Model": "LSTM (fixed)", "Lookback": lookback, "Split": "test", **m})
        predictions[("LSTM (fixed)", lookback)] = test_pred.reshape(-1)
        if "LSTM (fixed)" not in best_by_model or val_loss < best_by_model["LSTM (fixed)"][0]:
            best_by_model["LSTM (fixed)"] = (val_loss, lookback, lstm, m)

        # ---- PatchTST transformer ----
        patch_len = 6 if lookback >= 24 else 7
        stride = 2 if lookback >= 24 else 1
        t0 = time.time()
        ptst = PatchTST(lookback=lookback, patch_len=patch_len, stride=stride,
                         d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                         dropout=0.1, horizon=1)
        ptst, val_loss = train_torch_model(ptst, X_train, y_train, X_val, y_val,
                                            log_prefix=f"PatchTST L{lookback}")
        test_pred = predict_torch(ptst, X_test)
        m = metrics(y_test, test_pred)
        m["val_mse"] = val_loss
        m["train_seconds"] = round(time.time() - t0, 1)
        results.append({"Model": "PatchTST (proposed)", "Lookback": lookback, "Split": "test", **m})
        predictions[("PatchTST (proposed)", lookback)] = test_pred.reshape(-1)
        if "PatchTST (proposed)" not in best_by_model or val_loss < best_by_model["PatchTST (proposed)"][0]:
            best_by_model["PatchTST (proposed)"] = (val_loss, lookback, ptst, m)

        print(f"[lookback={lookback}] done")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "all_results.csv", index=False)
    print("\n=== All configs (test set) ===")
    print(results_df.to_string(index=False))

    print("\n=== Best config per model (selected by val loss) ===")
    for name, (val_loss, lookback, model, m) in best_by_model.items():
        print(f"{name}: lookback={lookback} test_RMSE={m['RMSE']:.4f} test_MAE={m['MAE']:.4f} test_R2={m['R2']:.5f}")
        torch.save(model.state_dict(), OUT_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_best.pt")

    # Save test-set actuals + best predictions for plotting
    lookback = best_by_model["PatchTST (proposed)"][1]
    test_ctx = np.concatenate([val_arr[-lookback:], test_arr])
    _, y_test = make_windows(test_ctx, lookback, horizon=1)
    test_dates = test_s.index[:len(y_test)]

    export = pd.DataFrame({"date": test_dates, "actual": y_test.reshape(-1)})
    for name, (val_loss, lb, model, m) in best_by_model.items():
        ctx = np.concatenate([val_arr[-lb:], test_arr])
        X_t, _ = make_windows(ctx, lb, horizon=1)
        pred = predict_torch(model, X_t).reshape(-1)
        export[name] = pred[:len(export)]
    naive_pred = NaivePersistence().predict(make_windows(test_ctx, lookback, horizon=1)[0])
    export["Naive Persistence"] = naive_pred[:len(export)]
    export.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    with open(OUT_DIR / "best_config.json", "w") as f:
        json.dump({name: {"lookback": lb, "metrics": m} for name, (vl, lb, mdl, m) in best_by_model.items()},
                   f, indent=2)

    print("\nSaved outputs to", OUT_DIR)


if __name__ == "__main__":
    run()
