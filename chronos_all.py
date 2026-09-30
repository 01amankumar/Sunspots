"""
Zero-shot Chronos-Bolt on the same test windows used by train_all_datasets.py,
for all 4 SILSO resolutions. No training on this data at all.
"""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_resolution, chronological_split, make_windows, RESOLUTIONS

OUT_ROOT = r"C:\Users\amank\Downloads\sunspot_transformer\outputs"
MODEL_ID = "amazon/chronos-bolt-small"
BATCH = 128

CHRONOS_CONTEXT = {
    "daily": 256,
    "monthly_mean": 64,
    "monthly_smoothed": 64,
    "yearly": 24,
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


def run():
    from chronos import BaseChronosPipeline
    print(f"Loading {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)

    for resolution in RESOLUTIONS:
        context = CHRONOS_CONTEXT[resolution]
        series = load_resolution(resolution)
        train_s, val_s, test_s = chronological_split(series)
        val_arr, test_arr = val_s.values, test_s.values
        test_ctx_full = np.concatenate([val_arr[-context:], test_arr])
        X_test, y_test = make_windows(test_ctx_full, context, horizon=1)
        print(f"\n[{resolution}] {len(X_test)} test points, context={context}")

        preds = []
        t0 = time.time()
        for i in range(0, len(X_test), BATCH):
            batch = X_test[i:i + BATCH, :, 0]
            contexts = [torch.tensor(row, dtype=torch.float32) for row in batch]
            quantiles, _ = pipeline.predict_quantiles(inputs=contexts, prediction_length=1, quantile_levels=[0.5])
            preds.append(quantiles[:, 0, 0].numpy())
        preds = np.concatenate(preds)
        print(f"  done in {time.time()-t0:.1f}s")

        m = metrics(y_test, preds)
        print(f"  RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")

        test_dates = test_s.index[:len(y_test)]
        out = pd.DataFrame({"date": test_dates, "actual": y_test.reshape(-1), "chronos_pred": preds})
        out.to_csv(f"{OUT_ROOT}\\{resolution}\\chronos_predictions.csv", index=False)
        pd.DataFrame([{"Model": "Chronos-Bolt (zero-shot)", **m}]).to_csv(
            f"{OUT_ROOT}\\{resolution}\\chronos_metrics.csv", index=False)

    print("\nAll resolutions done.")


if __name__ == "__main__":
    run()
