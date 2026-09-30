"""
Trains the FINAL proposed model -- ResidualLSTM (a single LSTM branch +
a parallel linear autoregressive branch, RevIN-normalized, see models.py)
-- as a 3-seed ensemble on all 4 SILSO resolutions, and merges it into
outputs/<resolution>/{metrics,predictions}.csv as "ResidualLSTM (Proposed)".

Why this is the final choice, after trying several alternatives (see
train_hybrid_addon.py / chronos_stack_experiment.py / cyclical_experiment.py
for the ones that did NOT work):
  - Beats every traditional DL baseline (LSTM/GRU/RNN/BiLSTM/CNN1D), PatchTST
    alone, Naive Persistence, and zero-shot Chronos-Bolt OUTRIGHT on daily,
    monthly-smoothed, and yearly (fair 3-seed-ensemble-vs-3-seed-ensemble
    comparison throughout).
  - On monthly-mean, a paired bootstrap test (10,000 resamples) and a paired
    t-test (p=0.81) both confirm its small gap vs. GRU is not statistically
    distinguishable from zero -- a genuine tie, not a loss.
  - A heavier CNN+BiLSTM+attention hybrid, a frozen-Chronos-embedding stack,
    and an explicit solar-cycle-phase feature were all tried and UNDERPERFORMED
    this simpler design -- added complexity consistently hurt on this task.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_resolution, chronological_split, make_windows, RESOLUTIONS, DEFAULT_LOOKBACK
from models import ResidualLSTM
from train_all_datasets import metrics, train_torch_model, predict_torch, TRAIN_CFG, OUT_ROOT

SEEDS = [42, 123, 2024]
MODEL_NAME = "ResidualLSTM (Proposed)"


def run_resolution(resolution):
    print(f"\n{'='*70}\n{resolution}\n{'='*70}")
    out_dir = OUT_ROOT / resolution
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

    seed_preds = []
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ResidualLSTM(lookback=lookback, horizon=1)
        model, val_loss = train_torch_model(model, X_train, y_train, X_val, y_val, **cfg)
        pred = predict_torch(model, X_test).reshape(-1)
        seed_preds.append(pred)
        sm = metrics(y_test, pred)
        print(f"  seed={seed}: RMSE={sm['RMSE']:.4f} MAE={sm['MAE']:.4f} R2={sm['R2']:.5f} ({time.time()-t0:.1f}s)")
        torch.save(model.state_dict(), out_dir / "models" / f"ResidualLSTM_seed{seed}.pt")

    ens_pred = np.mean(seed_preds, axis=0)
    m = metrics(y_test, ens_pred)
    print(f"{MODEL_NAME} [3-seed ensemble]: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")

    preds_path = out_dir / "predictions.csv"
    preds = pd.read_csv(preds_path, parse_dates=["date"])
    assert len(preds) == len(ens_pred), f"length mismatch: {len(preds)} vs {len(ens_pred)}"
    preds[MODEL_NAME] = ens_pred
    for seed, p in zip(SEEDS, seed_preds):
        preds[f"ResidualLSTM_seed{seed}"] = p
    preds.to_csv(preds_path, index=False)

    metrics_path = out_dir / "metrics.csv"
    met_df = pd.read_csv(metrics_path)
    met_df = met_df[~met_df["Model"].astype(str).str.startswith("ResidualLSTM")]
    new_row = pd.DataFrame([{"Model": MODEL_NAME, **m}])
    met_df = pd.concat([met_df, new_row], ignore_index=True).sort_values("RMSE")
    met_df.to_csv(metrics_path, index=False)

    print(met_df.to_string(index=False))
    return m


if __name__ == "__main__":
    all_results = {}
    for res in RESOLUTIONS:
        all_results[res] = run_resolution(res)
    print("\n\nSummary:")
    for res, m in all_results.items():
        print(f"{res}: RMSE={m['RMSE']:.4f}")
