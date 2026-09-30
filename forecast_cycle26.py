"""
A genuine multi-year-ahead forecast, comparable to Ruiz et al. (2024, Solar
Physics), who forecast hemispheric SSN out to January 2038 and report Cycle
26 peaks of N=51.5 (Nov 2033) and S=70.1 (Nov 2034), RMSE 6.1 (N) / 6.8 (S).

Two things this script does, both required before any comparison to that
paper is legitimate:
  1. Recursive-backtest validation: start the recursive (self-fed) forecast
     at the beginning of the held-out test period and roll it forward
     through real historical months, comparing against actual values, to
     show the recursive approach doesn't blow up over a multi-year horizon
     before trusting it into the unknown future.
  2. The actual future forecast: retrain on ALL available real data (not
     leaving out a test split -- this is the production forecast, not a
     backtest) and recursively roll forward from Dec 2024 to Jan 2038,
     extracting Cycle 26's predicted peak value and month per hemisphere.
"""
import numpy as np
import pandas as pd
import torch

from hem_data import build_hemisphere_series
from data import chronological_split, make_windows
from models import ResidualLSTM
from train_all_datasets import metrics, train_torch_model, predict_torch, OUT_ROOT

LOOKBACK = 14
TRAIN_CFG = dict(epochs=300, patience=25, batch_size=64, lr=1e-3)
SEEDS = [42, 123, 2024]
FORECAST_END = "2038-01-01"


def recursive_forecast(models, last_window, n_steps):
    """models: list of trained ResidualLSTM (a seed-ensemble). Recursively
    predicts n_steps ahead, feeding the ensemble-averaged prediction back in
    as the next input. Returns an array of length n_steps."""
    window = last_window.copy()  # (lookback,)
    preds = []
    for _ in range(n_steps):
        x = torch.tensor(window.reshape(1, LOOKBACK, 1), dtype=torch.float32)
        step_preds = []
        for m in models:
            m.eval()
            with torch.no_grad():
                step_preds.append(m(x).numpy().reshape(-1)[0])
        next_val = float(np.mean(step_preds))
        preds.append(next_val)
        window = np.concatenate([window[1:], [next_val]])
    return np.array(preds)


def train_ensemble(X_train, y_train, X_val, y_val):
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ResidualLSTM(lookback=LOOKBACK, horizon=1)
        model, _ = train_torch_model(model, X_train, y_train, X_val, y_val, **TRAIN_CFG)
        models.append(model)
    return models


def recursive_backtest(hemisphere):
    """Roll the recursive forecast forward through the REAL held-out test
    period (using only train+val to fit, exactly like the main study) and
    compare against actual test values -- validates the recursive method
    before trusting it into the future."""
    series = build_hemisphere_series(hemisphere, smoothed=True)
    train_s, val_s, test_s = chronological_split(series)
    train_arr, val_arr, test_arr = train_s.values, val_s.values, test_s.values

    X_train, y_train = make_windows(train_arr, LOOKBACK, horizon=1)
    val_ctx = np.concatenate([train_arr[-LOOKBACK:], val_arr])
    X_val, y_val = make_windows(val_ctx, LOOKBACK, horizon=1)

    print(f"[{hemisphere}] training backtest ensemble (train+val only, test held out)...")
    models = train_ensemble(X_train, y_train, X_val, y_val)

    last_window = val_arr[-LOOKBACK:]
    n_steps = len(test_arr)
    rec_pred = recursive_forecast(models, last_window, n_steps)

    m_1step = metrics(test_arr, rec_pred)
    print(f"[{hemisphere}] recursive backtest over full {n_steps}-month test horizon: "
          f"RMSE={m_1step['RMSE']:.3f} MAE={m_1step['MAE']:.3f} R2={m_1step['R2']:.4f}")
    # also report short vs long horizon degradation
    for h in [12, 36, 60, n_steps]:
        h = min(h, n_steps)
        m_h = metrics(test_arr[:h], rec_pred[:h])
        print(f"    first {h} months: RMSE={m_h['RMSE']:.3f}")

    out = pd.DataFrame({"date": test_s.index[:n_steps], "actual": test_arr, "recursive_pred": rec_pred})
    out.to_csv(OUT_ROOT / f"hemisphere_{hemisphere}" / "recursive_backtest.csv", index=False)
    return m_1step


def future_forecast(hemisphere):
    """Retrain on ALL available real data, then recursively forecast to
    January 2038."""
    series = build_hemisphere_series(hemisphere, smoothed=True)
    # use the same train/val split point logic as chronological_split, but
    # fold the held-out test portion back in as additional "training" signal
    # for the final production model -- standard practice for a deployed
    # forecast (not a backtest), clearly distinguished from the study above.
    n = len(series)
    n_train = int(n * 0.85)
    train_arr = series.values[:n_train]
    val_arr = series.values[n_train:]
    X_train, y_train = make_windows(train_arr, LOOKBACK, horizon=1)
    val_ctx = np.concatenate([train_arr[-LOOKBACK:], val_arr])
    X_val, y_val = make_windows(val_ctx, LOOKBACK, horizon=1)

    print(f"\n[{hemisphere}] training production ensemble on full record ({n} months)...")
    models = train_ensemble(X_train, y_train, X_val, y_val)

    last_date = series.index[-1]
    n_steps = int(round((pd.Timestamp(FORECAST_END) - last_date).days / 30.44))
    last_window = series.values[-LOOKBACK:]
    forecast = recursive_forecast(models, last_window, n_steps)
    forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_steps, freq="MS")

    out = pd.DataFrame({"date": forecast_dates, "forecast": forecast})
    out.to_csv(OUT_ROOT / f"hemisphere_{hemisphere}" / "cycle26_forecast.csv", index=False)

    # Cycle 26 peak: look for the maximum after the current Cycle 25 decline
    # (restrict search to 2028 onward to avoid catching any residual Cycle 25 tail)
    search = out[out.date >= "2028-01-01"]
    peak_row = search.loc[search["forecast"].idxmax()]
    print(f"[{hemisphere}] forecast to {FORECAST_END}: Cycle 26 peak = {peak_row['forecast']:.1f} "
          f"in {peak_row['date'].strftime('%b %Y')}")
    return peak_row


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: recursive backtest validation (real held-out test period)")
    print("=" * 70)
    for hemi in ["north", "south"]:
        recursive_backtest(hemi)

    print("\n" + "=" * 70)
    print("STEP 2: production Cycle 26 forecast to", FORECAST_END)
    print("=" * 70)
    peaks = {}
    for hemi in ["north", "south"]:
        peaks[hemi] = future_forecast(hemi)

    print("\n\nSummary vs. Ruiz et al. (2024, Solar Physics) reported Cycle 26 peaks:")
    print("  Ruiz et al.: North 51.5 in Nov 2033 (RMSE 6.1); South 70.1 in Nov 2034 (RMSE 6.8)")
    for hemi, row in peaks.items():
        print(f"  This work:   {hemi.capitalize():5s} {row['forecast']:.1f} in {row['date'].strftime('%b %Y')}")
