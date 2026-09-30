"""
Experiment: inject an explicit solar-cycle-phase feature (sin/cos of position
within an ~11-year / 132-month cycle, computed from the calendar date of the
forecast target) into ResidualLSTM, on monthly_mean -- the one resolution
where the plain residual model didn't quite beat GRU. A 14-month lookback
window cannot see where in the ~11-year cycle it sits; this gives it that
information directly, without needing a longer, noisier lookback.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import load_resolution, chronological_split, make_windows, DEFAULT_LOOKBACK
from train_all_datasets import metrics, TRAIN_CFG
from models import RevIN

RESOLUTION = "monthly_mean"
CYCLE_MONTHS = 132.0  # ~11-year average solar cycle


def cyclical_features(dates):
    months_since_epoch = (dates.year - 1749) * 12 + (dates.month - 1)
    phase = 2 * np.pi * (months_since_epoch % CYCLE_MONTHS) / CYCLE_MONTHS
    return np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)


class ResidualLSTMCyclical(nn.Module):
    def __init__(self, lookback, horizon=1, hidden=64, dropout=0.2):
        super().__init__()
        self.revin = RevIN()
        self.linear_ar = nn.Linear(lookback, horizon)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2,
                             batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden + 2, horizon)

    def forward(self, x, cyc):
        xn = self.revin.normalize(x)
        lin_out = self.linear_ar(xn.squeeze(-1))
        out, _ = self.lstm(xn)
        deep_out = self.head(torch.cat([out[:, -1, :], cyc], dim=-1))
        y = lin_out + deep_out
        return self.revin.denormalize(y)


def train_model(model, X_train, y_train, C_train, X_val, y_val, C_val, epochs, patience, batch_size, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, yt, Ct = torch.tensor(X_train), torch.tensor(y_train), torch.tensor(C_train)
    Xv, yv, Cv = torch.tensor(X_val), torch.tensor(y_val), torch.tensor(C_val)
    n = Xt.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx], Ct[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv, Cv), yv).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X, C, batch_size=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(model(torch.tensor(X[i:i+batch_size]), torch.tensor(C[i:i+batch_size])).numpy())
    return np.concatenate(outs, axis=0)


def run():
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

    # target date for window i is the (i+lookback)-th date of the underlying series slice
    train_dates = train_s.index[lookback:lookback + len(y_train)]
    val_dates_full = pd.DatetimeIndex(list(train_s.index[-lookback:]) + list(val_s.index))
    val_dates = val_dates_full[lookback:lookback + len(y_val)]
    test_dates_full = pd.DatetimeIndex(list(val_s.index[-lookback:]) + list(test_s.index))
    test_dates = test_dates_full[lookback:lookback + len(y_test)]

    C_train = cyclical_features(train_dates)
    C_val = cyclical_features(val_dates)
    C_test = cyclical_features(test_dates)

    preds = []
    for seed in (42, 123, 2024):
        torch.manual_seed(seed); np.random.seed(seed)
        model = ResidualLSTMCyclical(lookback=lookback)
        model = train_model(model, X_train, y_train, C_train, X_val, y_val, C_val, **cfg)
        pred = predict(model, X_test, C_test).reshape(-1)
        preds.append(pred)
        print(f"  seed={seed}: RMSE={metrics(y_test, pred)['RMSE']:.4f}")

    ens = np.mean(preds, axis=0)
    m = metrics(y_test, ens)
    print(f"\nResidualLSTM+CyclicalPhase 3-seed ensemble: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")
    print("Compare: plain GRU 3-seed=21.5945, ResidualLSTM 3-seed=21.6247")


if __name__ == "__main__":
    run()
