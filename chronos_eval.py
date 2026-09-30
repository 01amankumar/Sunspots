"""
Zero-shot evaluation of Amazon's Chronos-Bolt (a real pretrained time-series
foundation model, via the official `chronos-forecasting` package) on the same
1-step-ahead test windows used for the other models, for a genuine
"time-series LLM" comparison point (no fine-tuning, no training on this data).
"""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_series, chronological_split, make_windows

CSV_PATH = r"C:\Users\amank\Downloads\SN_ms_tot_V2.0 (2).csv"
CONTEXT = 64
MODEL_ID = "amazon/chronos-bolt-small"
BATCH = 64


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


def run():
    from chronos import BaseChronosPipeline

    series = load_series(CSV_PATH)
    train_s, val_s, test_s = chronological_split(series)
    val_arr = val_s.values
    test_arr = test_s.values

    test_ctx_full = np.concatenate([val_arr[-CONTEXT:], test_arr])
    X_test, y_test = make_windows(test_ctx_full, CONTEXT, horizon=1)
    print(f"Evaluating Chronos zero-shot on {len(X_test)} test points, context={CONTEXT}")

    print(f"Loading {MODEL_ID} ...")
    t0 = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)
    print(f"Loaded in {time.time()-t0:.1f}s")

    preds = []
    t0 = time.time()
    for i in range(0, len(X_test), BATCH):
        batch = X_test[i:i + BATCH, :, 0]
        contexts = [torch.tensor(row, dtype=torch.float32) for row in batch]
        quantiles, mean = pipeline.predict_quantiles(
            inputs=contexts, prediction_length=1, quantile_levels=[0.5]
        )
        median = quantiles[:, 0, 0].numpy()
        preds.append(median)
        if i % (BATCH * 4) == 0:
            print(f"  {i}/{len(X_test)}  elapsed={time.time()-t0:.1f}s")
    preds = np.concatenate(preds)

    m = metrics(y_test, preds)
    print("\n=== Chronos-Bolt zero-shot (test set) ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    out = pd.DataFrame({
        "date": test_s.index[:len(y_test)],
        "actual": y_test.reshape(-1),
        "chronos_pred": preds,
    })
    out.to_csv(r"C:\Users\amank\Downloads\sunspot_transformer\outputs\chronos_predictions.csv", index=False)

    pd.DataFrame([{"Model": "Chronos-Bolt-Small (zero-shot)", "Lookback": CONTEXT, "Split": "test", **m}]).to_csv(
        r"C:\Users\amank\Downloads\sunspot_transformer\outputs\chronos_metrics.csv", index=False)
    print("Saved outputs.")


if __name__ == "__main__":
    run()
