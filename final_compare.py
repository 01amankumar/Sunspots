import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = r"C:\Users\amank\Downloads\sunspot_transformer\outputs"


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


main = pd.read_csv(f"{OUT}/test_predictions.csv", parse_dates=["date"])
chronos = pd.read_csv(f"{OUT}/chronos_predictions.csv", parse_dates=["date"])

merged = main.merge(chronos[["date", "chronos_pred"]], on="date", how="inner")
merged["Ensemble (LSTM+PatchTST)"] = (merged["LSTM (fixed)"] + merged["PatchTST (proposed)"]) / 2
merged["Ensemble (LSTM+PatchTST+Chronos)"] = (
    merged["LSTM (fixed)"] + merged["PatchTST (proposed)"] + merged["chronos_pred"]
) / 3

print(f"merged rows: {len(merged)}  ({merged.date.min().date()} to {merged.date.max().date()})\n")

rows = []
for col, label in [
    ("Naive Persistence", "Naive Persistence"),
    ("LSTM (fixed)", "LSTM (fixed, RevIN, no leakage)"),
    ("PatchTST (proposed)", "PatchTST transformer (proposed)"),
    ("chronos_pred", "Chronos-Bolt (zero-shot, no training)"),
    ("Ensemble (LSTM+PatchTST)", "Ensemble: LSTM + PatchTST"),
    ("Ensemble (LSTM+PatchTST+Chronos)", "Ensemble: LSTM + PatchTST + Chronos"),
]:
    m = metrics(merged["actual"], merged[col])
    rows.append({"Model": label, **m})

df = pd.DataFrame(rows).sort_values("RMSE")
print(df.to_string(index=False))
df.to_csv(f"{OUT}/final_comparison.csv", index=False)
merged.to_csv(f"{OUT}/final_merged_predictions.csv", index=False)
print(f"\nsaved to {OUT}/final_comparison.csv and final_merged_predictions.csv")
