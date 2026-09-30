"""
Is GRU's edge over ResidualLSTM on monthly_mean (21.59 vs 21.62) real, or
noise? Paired bootstrap test on the two ensembles' squared errors over the
same 499 test points.
"""
import numpy as np
import torch

from data import load_resolution, chronological_split, make_windows, DEFAULT_LOOKBACK
from models import ResidualLSTM, RNNForecaster
from train_all_datasets import metrics, train_torch_model, predict_torch, TRAIN_CFG

RESOLUTION = "monthly_mean"
lookback = DEFAULT_LOOKBACK[RESOLUTION]
cfg = TRAIN_CFG[RESOLUTION]
series = load_resolution(RESOLUTION)
train_s, val_s, test_s = chronological_split(series)
train_arr, val_arr, test_arr = train_s.values, val_s.values, test_s.values
X_train, y_train = make_windows(train_arr, lookback, horizon=1)
val_ctx = np.concatenate([train_arr[-lookback:], val_arr])
X_val, y_val = make_windows(val_ctx, lookback, horizon=1)
test_ctx = np.concatenate([val_arr[-lookback:], test_arr])
X_test, y_test = make_windows(test_ctx, lookback, horizon=1)
y_test_flat = y_test.reshape(-1)

def ensemble_preds(model_fn, seeds=(42, 123, 2024)):
    preds = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = model_fn()
        model, _ = train_torch_model(model, X_train, y_train, X_val, y_val, **cfg)
        preds.append(predict_torch(model, X_test).reshape(-1))
    return np.mean(preds, axis=0)

print("Training GRU ensemble...")
gru_pred = ensemble_preds(lambda: RNNForecaster(cell="gru", hidden=64, horizon=1))
print("Training ResidualLSTM ensemble...")
res_pred = ensemble_preds(lambda: ResidualLSTM(lookback=lookback, horizon=1))

gru_sq_err = (y_test_flat - gru_pred) ** 2
res_sq_err = (y_test_flat - res_pred) ** 2
print(f"\nGRU RMSE={np.sqrt(gru_sq_err.mean()):.4f}  ResidualLSTM RMSE={np.sqrt(res_sq_err.mean()):.4f}")

diff = res_sq_err - gru_sq_err  # positive => ResidualLSTM worse
rng = np.random.default_rng(0)
n = len(diff)
boot_means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"\nBootstrap 95% CI for mean(SqErr_ResidualLSTM - SqErr_GRU): [{ci_lo:.4f}, {ci_hi:.4f}]")
print("Zero inside CI => difference NOT statistically significant (effectively a tie)"
      if ci_lo <= 0 <= ci_hi else
      "Zero outside CI => GRU's edge IS statistically significant")

# Diebold-Mariano-style paired t-test as a second check
from scipy import stats
t_stat, p_val = stats.ttest_rel(res_sq_err, gru_sq_err)
print(f"\nPaired t-test on squared errors: t={t_stat:.3f}, p={p_val:.4f}")
