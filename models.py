import math
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Normalizes each input window by its own mean/std before the model sees it,
    then de-normalizes the model's output with the same stats. This matters a
    lot here because sunspot cycle amplitude varies ~3x between weak and strong
    cycles (e.g. cycle 24 vs cycle 19) -- a single global scaler under- or
    over-scales large chunks of the series.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.std = None

    def normalize(self, x):
        # x: (batch, lookback, 1)
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + self.eps
        return (x - self.mean) / self.std

    def denormalize(self, y):
        # y: (batch, horizon)
        mean = self.mean.squeeze(-1)
        std = self.std.squeeze(-1)
        return y * std + mean


class NaivePersistence:
    """Predicts next value = last observed value in the lookback window."""

    def predict(self, X):
        # X: (n, lookback, 1) unscaled
        return X[:, -1, 0]


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64, horizon=1, dropout=0.2):
        super().__init__()
        self.revin = RevIN()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2,
                             batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        x = self.revin.normalize(x)
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :])
        return self.revin.denormalize(y)


class ResidualLSTM(nn.Module):
    """Recurrent cell (LSTM or GRU) + a parallel linear autoregressive branch
    (DLinear-style), combined additively under one RevIN normalization.
    Isolates whether a lightweight residual anchor helps on top of a plain
    recurrent model, without the extra CNN/attention machinery of
    HybridForecaster. Despite the name, `cell="gru"` swaps in a GRU.
    """

    def __init__(self, lookback, horizon=1, hidden=64, dropout=0.2, cell="lstm"):
        super().__init__()
        self.revin = RevIN()
        self.linear_ar = nn.Linear(lookback, horizon)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        xn = self.revin.normalize(x)
        lin_out = self.linear_ar(xn.squeeze(-1))
        out, _ = self.rnn(xn)
        deep_out = self.head(out[:, -1, :])
        y = lin_out + deep_out
        return self.revin.denormalize(y)


class RNNForecaster(nn.Module):
    """Traditional recurrent baseline: GRU, plain (Elman) RNN, or BiLSTM,
    selected by `cell`. Same RevIN + leak-free-scaling treatment as the
    proposed models, so comparisons isolate architecture, not preprocessing.
    """

    def __init__(self, cell="gru", hidden=64, horizon=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.revin = RevIN()
        cell = cell.lower()
        rnn_cls = {"gru": nn.GRU, "rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        kwargs = dict(input_size=1, hidden_size=hidden, num_layers=2,
                      batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if cell == "rnn":
            kwargs["nonlinearity"] = "relu"
        self.rnn = rnn_cls(**kwargs)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, horizon)

    def forward(self, x):
        x = self.revin.normalize(x)
        out, _ = self.rnn(x)
        y = self.head(out[:, -1, :])
        return self.revin.denormalize(y)


class CNN1DForecaster(nn.Module):
    """Traditional 1D-CNN baseline: stacked causal conv blocks over the
    RevIN-normalized lookback window, flattened to the forecast horizon.
    """

    def __init__(self, lookback, horizon=1, channels=(32, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        self.revin = RevIN()
        layers = []
        in_ch = 1
        length = lookback
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
            length = length // 2
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * length, horizon),
        )

    def forward(self, x):
        x = self.revin.normalize(x)
        x = x.transpose(1, 2)  # (batch, 1, lookback)
        x = self.conv(x)
        y = self.head(x)
        return self.revin.denormalize(y)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HybridForecaster(nn.Module):
    """SSN-Hybrid: a residual multi-scale CNN + BiLSTM + self-attention
    network for sunspot number forecasting.

    Design rationale (why this should generalize across very different
    sample sizes, from 226 yearly training points to 50k+ daily points,
    where a pure patch-transformer does not):

      1. Linear residual branch (RevIN-normalized window -> horizon, a la
         DLinear/NLinear). This is a strong, low-variance anchor: on small
         data (yearly) it alone captures most of the persistence/trend
         signal, so the deep branch only has to learn a *correction* rather
         than the whole mapping -- far less prone to overfitting a 226-point
         training set than an attention model with no such anchor.
      2. Multi-scale 1D-conv front end (parallel kernels at short/medium/long
         scale, sized relative to the lookback window) extracts local shape
         features cheaply, with a strong translation-invariant inductive
         bias that needs little data to learn.
      3. A BiLSTM over the conv features captures ordered, bidirectional
         context across the window.
      4. A single lightweight self-attention block over the BiLSTM sequence
         adds the capacity for longer-range / cyclical dependencies (e.g.
         solar-cycle phase) without the token count and data appetite of a
         full patch-transformer.

    All four pieces sit under one shared RevIN normalization, and the final
    forecast is linear_branch + deep_branch (residual combination).
    """

    def __init__(self, lookback, horizon=1, cnn_channels=24, kernel_sizes=(3, 5, 7),
                 lstm_hidden=48, n_heads=2, dropout=0.2):
        super().__init__()
        self.revin = RevIN()
        self.linear_ar = nn.Linear(lookback, horizon)

        self.convs = nn.ModuleList([
            nn.Conv1d(1, cnn_channels, k, padding=k // 2) for k in kernel_sizes
        ])
        conv_out_dim = cnn_channels * len(kernel_sizes)

        self.bilstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden,
                               num_layers=1, batch_first=True, bidirectional=True)
        attn_dim = lstm_hidden * 2
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(attn_dim, horizon)

    def forward(self, x):
        # x: (batch, lookback, 1)
        xn = self.revin.normalize(x)

        lin_out = self.linear_ar(xn.squeeze(-1))  # (batch, horizon)

        xt = xn.transpose(1, 2)  # (batch, 1, lookback)
        feats = [torch.relu(conv(xt)) for conv in self.convs]
        cat = torch.cat(feats, dim=1).transpose(1, 2)  # (batch, lookback, conv_out_dim)

        lstm_out, _ = self.bilstm(cat)  # (batch, lookback, attn_dim)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, need_weights=False)
        attn_out = self.norm(attn_out + lstm_out)
        pooled = self.dropout(attn_out.mean(dim=1))
        deep_out = self.head(pooled)

        y = lin_out + deep_out
        return self.revin.denormalize(y)


class PatchTST(nn.Module):
    """A lightweight PatchTST-style transformer (Nie et al., 2023) for
    univariate forecasting: split the RevIN-normalized lookback window into
    overlapping patches, embed each patch as a token, run a transformer
    encoder over the token sequence, then linearly project the flattened
    representation to the forecast horizon.
    """

    def __init__(self, lookback, patch_len=6, stride=3, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1):
        super().__init__()
        self.revin = RevIN()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (lookback - patch_len) // stride + 1
        self.n_patches = n_patches

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_patches + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon),
        )

    def _make_patches(self, x):
        # x: (batch, lookback, 1) -> (batch, n_patches, patch_len)
        x = x.squeeze(-1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches

    def forward(self, x):
        x = self.revin.normalize(x)
        patches = self._make_patches(x)
        tokens = self.patch_embed(patches)
        tokens = self.pos_enc(tokens)
        encoded = self.encoder(tokens)
        y = self.head(encoded)
        return self.revin.denormalize(y)
