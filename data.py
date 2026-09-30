"""
Data loading / preparation for SILSO total sunspot number series, at any of the
four native resolutions (daily / monthly mean / monthly smoothed / yearly).

Fixes vs. the original notebook:
  - Rows with definitive_flag == 0 (provisional, not-yet-final smoothing near the
    series tail) are dropped instead of being kept/blindly filled.
  - Any sentinel -1 values in the target column are dropped rather than replaced
    with 1 (replacing with 1 fabricates a fake near-zero sunspot month).
  - Scalers are fit on the TRAIN split only, then applied to val/test. The
    original notebook fit MinMaxScaler on the full series (train+test) before
    splitting, which leaks test-set min/max information into training.

File formats (as provided in this project, verified against cross-checks):
  - SN_m_tot_V2.0.csv        monthly mean,     7 cols: year,month,frac,sn,sd,nobs,flag
  - SN_ms_tot_V2.0 (2).csv   monthly smoothed, 7 cols: year,month,frac,sn,sd,nobs,flag
  - SN_y_tot_V2.0.csv        yearly mean,      1 col:  sn  (row i -> year 1700+i)
  - SN_d_tot_V2.0.csv        daily total,      1 col:  sn  (row i -> date 1818-01-01+i)
"""
import numpy as np
import pandas as pd

COLUMN_NAMES_7COL = ["year", "month", "frac_year", "sn", "sd", "n_obs", "definitive"]

RESOLUTIONS = ("daily", "monthly_smoothed", "yearly")

DATASET_PATHS = {
    "daily": r"C:\Users\amank\Downloads\SN_d_tot_V2.0.csv",
    "monthly_smoothed": r"C:\Users\amank\Downloads\SN_ms_tot_V2.0 (2).csv",
    "yearly": r"C:\Users\amank\Downloads\SN_y_tot_V2.0.csv",
}

# lookback (in native steps of that resolution) and Chronos context length,
# chosen to represent a physically comparable span across resolutions
DEFAULT_LOOKBACK = {
    "daily": 60,             # ~2 solar rotations
    "monthly_smoothed": 14,  # matches the original paper
    "yearly": 11,            # one solar cycle
}

# The daily series has 4.29% missing observations (SILSO sentinel -1),
# entirely confined to 1818-01-01..1848-12-22 (sparser 19th-century observing
# network); 1849 onward is fully gap-free. Rather than silently dropping rows
# (which would splice non-adjacent calendar days together across gaps as if
# they were consecutive -- corrupting any lookback window that spans a gap),
# the series is truncated to start after the last known gap.
DAILY_CLEAN_START = "1849-01-01"


def load_series_7col(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, names=COLUMN_NAMES_7COL)
    df = df[df["definitive"] == 1]
    df = df[df["sn"] != -1]
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("date").set_index("date")
    return df["sn"].astype(np.float32)


def load_series_yearly(csv_path: str) -> pd.Series:
    vals = pd.read_csv(csv_path, header=None, names=["sn"])["sn"].astype(np.float32)
    years = pd.date_range("1700-01-01", periods=len(vals), freq="YS")
    s = pd.Series(vals.values, index=years)
    return s[s != -1]


def load_series_daily(csv_path: str) -> pd.Series:
    vals = pd.read_csv(csv_path, header=None, names=["sn"])["sn"].astype(np.float32)
    days = pd.date_range("1818-01-01", periods=len(vals), freq="D")
    s = pd.Series(vals.values, index=days)
    s = s[s.index >= DAILY_CLEAN_START]
    assert (s == -1).sum() == 0, (
        f"unexpected -1 sentinel values after {DAILY_CLEAN_START}; "
        "the no-gap assumption for this truncation no longer holds -- check the source file"
    )
    gaps = s.index.to_series().diff().dt.days
    assert (gaps.dropna() == 1).all(), "daily series is not calendar-contiguous after truncation"
    return s


def load_series(csv_path: str) -> pd.Series:
    """Back-compat: 7-column monthly loader (used by the original single-dataset scripts)."""
    return load_series_7col(csv_path)


def load_resolution(resolution: str) -> pd.Series:
    assert resolution in RESOLUTIONS, resolution
    path = DATASET_PATHS[resolution]
    if resolution in ("monthly_mean", "monthly_smoothed"):
        return load_series_7col(path)
    if resolution == "yearly":
        return load_series_yearly(path)
    if resolution == "daily":
        return load_series_daily(path)


def chronological_split(series: pd.Series, train_frac=0.70, val_frac=0.15):
    n = len(series)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = series.iloc[:n_train]
    val = series.iloc[n_train:n_train + n_val]
    test = series.iloc[n_train + n_val:]
    return train, val, test


def make_windows(arr: np.ndarray, lookback: int, horizon: int = 1):
    """arr: 1D scaled array. Returns X (n, lookback, 1), y (n, horizon)."""
    X, y = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback:i + lookback + horizon])
    X = np.asarray(X, dtype=np.float32)[..., None]
    y = np.asarray(y, dtype=np.float32)
    return X, y
