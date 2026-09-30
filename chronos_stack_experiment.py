"""
Experiment: a genuine LLM+DL stacked hybrid. Frozen Chronos-Bolt encoder
embeddings (a real pretrained time-series foundation model's internal
representation, no gradient) are concatenated with a small trainable
GRU + linear-residual head, on monthly_mean -- the one resolution where the
pure-DL ResidualLSTM/GRU didn't quite beat plain GRU.
"""
import time
import numpy as np
import torch
import torch.nn as nn

from data import load_resolution, chronological_split, make_windows, DEFAULT_LOOKBACK
from train_all_datasets import metrics, TRAIN_CFG
from models import RevIN

RESOLUTION = "monthly_mean"
DEVICE = "cpu"


def get_chronos_embeddings(pipeline, X, batch_size=128):
    """X: (n, lookback, 1) -> (n, d_model) mean-pooled frozen embeddings."""
    feats = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size, :, 0]
        contexts = [torch.tensor(row, dtype=torch.float32) for row in batch]
        emb, _ = pipeline.embed(contexts)  # (batch, num_patches+1, d_model)
        feats.append(emb.mean(dim=1).numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


class ChronosStackForecaster(nn.Module):
    def __init__(self, lookback, chronos_dim=512, proj_dim=32, hidden=64, horizon=1, dropout=0.2):
        super().__init__()
        self.revin = RevIN()
        self.linear_ar = nn.Linear(lookback, horizon)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=2,
                           batch_first=True, dropout=dropout)
        self.chronos_proj = nn.Sequential(
            nn.Linear(chronos_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden + proj_dim, horizon)

    def forward(self, x, chronos_feat):
        xn = self.revin.normalize(x)
        lin_out = self.linear_ar(xn.squeeze(-1))
        out, _ = self.gru(xn)
        gru_hidden = out[:, -1, :]
        cf = self.chronos_proj(chronos_feat)
        deep_out = self.head(torch.cat([gru_hidden, cf], dim=-1))
        y = lin_out + deep_out
        return self.revin.denormalize(y)


def train_stack_model(model, X_train, y_train, C_train, X_val, y_val, C_val,
                       epochs, patience, batch_size, lr):
    model.to(DEVICE)
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
            pred = model(Xt[idx], Ct[idx])
            loss = loss_fn(pred, yt[idx])
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
    return model, best_val


def predict_stack(model, X, C, batch_size=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size])
            cb = torch.tensor(C[i:i + batch_size])
            outs.append(model(xb, cb).numpy())
    return np.concatenate(outs, axis=0)


def run():
    from chronos import BaseChronosPipeline
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

    print("Loading Chronos-Bolt (frozen feature extractor)...")
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)

    print("Computing frozen Chronos embeddings for train/val/test windows...")
    t0 = time.time()
    C_train = get_chronos_embeddings(pipeline, X_train)
    C_val = get_chronos_embeddings(pipeline, X_val)
    C_test = get_chronos_embeddings(pipeline, X_test)
    print(f"  done in {time.time()-t0:.1f}s  shapes: {C_train.shape} {C_val.shape} {C_test.shape}")

    preds = []
    for seed in (42, 123, 2024):
        torch.manual_seed(seed); np.random.seed(seed)
        model = ChronosStackForecaster(lookback=lookback, chronos_dim=C_train.shape[1])
        model, val_loss = train_stack_model(model, X_train, y_train, C_train, X_val, y_val, C_val, **cfg)
        pred = predict_stack(model, X_test, C_test).reshape(-1)
        preds.append(pred)
        print(f"  seed={seed}: RMSE={metrics(y_test, pred)['RMSE']:.4f}")

    ens = np.mean(preds, axis=0)
    m = metrics(y_test, ens)
    print(f"\nChronos-Stack (GRU+frozen-Chronos-embed+linear) 3-seed ensemble: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.5f}")
    print("Compare: plain GRU 3-seed ensemble was RMSE=21.5945")


if __name__ == "__main__":
    run()
