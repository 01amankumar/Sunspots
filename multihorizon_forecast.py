"""
Direct multi-horizon Cycle 26 forecast (chosen over recursive rollout, which
failed validation -- see forecast_cycle26.py). ResidualLSTM's linear and
recurrent heads already project to an arbitrary `horizon`, so the same
validated architecture is reused with horizon=157 (Dec 2024 -> Jan 2038,
matching Ruiz et al. 2024's forecast window) instead of horizon=1, predicting
the entire future trajectory in one forward pass -- no recursive error
compounding.

Lookback is widened from 14 to 132 months (~one average 11-year solar cycle)
for this task specifically: predicting 13 years ahead from only 14 months of
recent history has essentially no information about cycle phase; giving the
model a full recent cycle of context is the physically motivated choice.

Two stages, same as the (failed) recursive attempt, so the comparison is
apples to apples:
  1. Backtest: train on train+val only, issue ONE 157-month-ahead forecast
     from the end of val, compare against the real held-out test months.
  2. Production: retrain on the full record, forecast to January 2038,
     extract Cycle 26's peak value/month per hemisphere.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hem_data import build_hemisphere_series
from data import chronological_split, make_windows
from models import ResidualLSTM
from train_all_datasets import metrics, predict_torch, OUT_ROOT

LOOKBACK = 132
HORIZON = 157
TRAIN_CFG = dict(epochs=250, patience=20, batch_size=64, lr=1e-3)
SEEDS = [42, 123, 2024]
FORECAST_END = "2038-01-01"


def train_multihorizon(X_train, y_train, X_val, y_val, epochs, patience, batch_size, lr):
    model = ResidualLSTM(lookback=LOOKBACK, horizon=HORIZON)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
    Xv, yv = torch.tensor(X_val), torch.tensor(y_val)
    n = Xt.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_ensemble(X_train, y_train, X_val, y_val):
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        models.append(train_multihorizon(X_train, y_train, X_val, y_val, **TRAIN_CFG))
    return models


def ensemble_predict_one(models, window):
    x = torch.tensor(window.reshape(1, LOOKBACK, 1), dtype=torch.float32)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(x).numpy().reshape(-1))
    return np.mean(preds, axis=0)


def backtest(hemisphere):
    series = build_hemisphere_series(hemisphere, smoothed=True)
    train_s, val_s, test_s = chronological_split(series)
    train_arr, val_arr, test_arr = train_s.values, val_s.values, test_s.values

    X_train, y_train = make_windows(train_arr, LOOKBACK, horizon=HORIZON)
    val_ctx = np.concatenate([train_arr[-LOOKBACK:], val_arr])
    X_val, y_val = make_windows(val_ctx, LOOKBACK, horizon=HORIZON)
    print(f"[{hemisphere}] backtest: {len(X_train)} train windows, {len(X_val)} val windows "
          f"(lookback={LOOKBACK}, horizon={HORIZON})")

    models = train_ensemble(X_train, y_train, X_val, y_val)

    last_window = val_arr[-LOOKBACK:]
    forecast = ensemble_predict_one(models, last_window)
    n_compare = min(HORIZON, len(test_arr))
    actual = test_arr[:n_compare]
    forecast = forecast[:n_compare]

    m_all = metrics(actual, forecast)
    print(f"[{hemisphere}] direct {n_compare}-month-ahead backtest: RMSE={m_all['RMSE']:.3f} "
          f"MAE={m_all['MAE']:.3f} R2={m_all['R2']:.4f}")
    for h in [1, 12, 36, 60, 120, n_compare]:
        if h > n_compare:
            continue
        seg_actual = actual[max(0, h - 12):h] if h > 12 else actual[:h]
        seg_forecast = forecast[max(0, h - 12):h] if h > 12 else forecast[:h]
        seg_m = metrics(seg_actual, seg_forecast)
        print(f"    step ~{h}: RMSE={seg_m['RMSE']:.3f}")

    out = pd.DataFrame({"date": test_s.index[:n_compare], "actual": actual, "direct_forecast": forecast})
    out.to_csv(OUT_ROOT / f"hemisphere_{hemisphere}" / "multihorizon_backtest.csv", index=False)
    return m_all


def production_forecast(hemisphere):
    series = build_hemisphere_series(hemisphere, smoothed=True)
    n = len(series)
    n_train = int(n * 0.85)
    train_arr = series.values[:n_train]
    val_arr = series.values[n_train:]

    X_train, y_train = make_windows(train_arr, LOOKBACK, horizon=HORIZON)
    val_ctx = np.concatenate([train_arr[-LOOKBACK:], val_arr])
    X_val, y_val = make_windows(val_ctx, LOOKBACK, horizon=HORIZON)
    print(f"\n[{hemisphere}] production: {len(X_train)} train windows, {len(X_val)} val windows")

    models = train_ensemble(X_train, y_train, X_val, y_val)

    last_date = series.index[-1]
    last_window = series.values[-LOOKBACK:]
    n_steps = int(round((pd.Timestamp(FORECAST_END) - last_date).days / 30.44))
    forecast = ensemble_predict_one(models, last_window)[:n_steps]
    forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_steps, freq="MS")

    out = pd.DataFrame({"date": forecast_dates, "forecast": forecast})
    out.to_csv(OUT_ROOT / f"hemisphere_{hemisphere}" / "cycle26_multihorizon_forecast.csv", index=False)

    search = out[out.date >= "2028-01-01"]
    peak_row = search.loc[search["forecast"].idxmax()]
    print(f"[{hemisphere}] Cycle 26 peak (direct multi-horizon): {peak_row['forecast']:.1f} "
          f"in {peak_row['date'].strftime('%b %Y')}")
    print(f"[{hemisphere}] forecast trajectory range: min={out['forecast'].min():.1f} max={out['forecast'].max():.1f}")
    return peak_row, out


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: direct multi-horizon backtest validation")
    print("=" * 70)
    for hemi in ["north", "south"]:
        backtest(hemi)

    print("\n" + "=" * 70)
    print("STEP 2: production Cycle 26 forecast to", FORECAST_END)
    print("=" * 70)
    peaks = {}
    for hemi in ["north", "south"]:
        peaks[hemi], _ = production_forecast(hemi)

    print("\n\nSummary vs. Ruiz et al. (2024, Solar Physics):")
    print("  Ruiz et al.: North 51.5 in Nov 2033 (RMSE 6.1); South 70.1 in Nov 2034 (RMSE 6.8)")
    for hemi, row in peaks.items():
        print(f"  This work:   {hemi.capitalize():5s} {row['forecast']:.1f} in {row['date'].strftime('%b %Y')}")
