"""
Long-record (1874-2025) hemispheric sunspot number series, built by splicing:
  - Veronig et al. (2021, A&A 652, A56) reconstruction, 1874-05 to 2020-10
    (Catalogue_B.txt, monthly: columns 2-5 = the paper's recommended merged
    N/S monthly + N/S 13-month-smoothed series, blending sunspot-area
    reconstruction 1874-1944, Temmer et al. 2006 recalibration 1945-2004,
    and official SILSO hemispheric numbers 1992-2020).
  - SILSO's own official hemispheric files, 2020-11 onward (SN_m_hem_V2.0.csv,
    SN_ms_hem_V2.0.csv), to bring the record up to the present.

This gives a continuous ~151-year hemispheric record, versus the ~33 years
(1992-2025) available from SILSO's official hemispheric product alone.
"""
import numpy as np
import pandas as pd

CATALOGUE_B = r"C:\Users\amank\Downloads\sunspot_transformer\hem_data\Catalogue_B.txt"
SILSO_M_HEM = r"C:\Users\amank\Downloads\SN_m_hem_V2.0.csv"
SILSO_MS_HEM = r"C:\Users\amank\Downloads\SN_ms_hem_V2.0.csv"

SPLICE_DATE = "2020-11-01"  # first month taken from SILSO's own official file


def _nan_from_sentinel(df, cols):
    for c in cols:
        df[c] = df[c].replace(-1.0, np.nan)
    return df


def load_veronig_monthly():
    df = pd.read_csv(CATALOGUE_B, sep=r"\s+", header=None,
                      names=["date", "n_m", "s_m", "n_ms", "s_ms"] + [f"extra{i}" for i in range(12)])
    df = df[["date", "n_m", "s_m", "n_ms", "s_ms"]]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df = _nan_from_sentinel(df, ["n_m", "s_m", "n_ms", "s_ms"])
    return df.set_index("date")


def load_silso_hem_monthly(path, ncols):
    # SN_m_hem / SN_ms_hem columns (per SILSO docs): year, month, frac,
    # total_sn, n_sn, s_sn, sd_total, sd_n, sd_s, nobs_total, nobs_n, nobs_s, flag
    names = ["year", "month", "frac", "total_sn", "n_sn", "s_sn",
              "sd_total", "sd_n", "sd_s", "nobs_total", "nobs_n", "nobs_s", "flag"]
    df = pd.read_csv(path, names=names)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df = df.set_index("date")
    df = _nan_from_sentinel(df, ["n_sn", "s_sn"])
    df.loc[df["flag"] != 1, ["n_sn", "s_sn"]] = np.nan  # drop provisional
    return df[["n_sn", "s_sn"]]


def validate_overlap():
    """Sanity check: Veronig's own SILSO-derived columns should already match
    SILSO's official file for 1992-2020 -- confirms the parsing is correct."""
    ver = load_veronig_monthly()
    silso_m = load_silso_hem_monthly(SILSO_M_HEM, 13)
    overlap = ver.join(silso_m, how="inner", rsuffix="_silso")
    overlap = overlap.dropna()
    corr_n = overlap["n_m"].corr(overlap["n_sn"])
    corr_s = overlap["s_m"].corr(overlap["s_sn"])
    return corr_n, corr_s, len(overlap)


def build_hemisphere_series(hemisphere: str, smoothed: bool):
    """hemisphere: 'north' or 'south'. Returns a spliced pd.Series, 1874-05 to
    2025 (whatever SILSO's file currently extends to)."""
    assert hemisphere in ("north", "south")
    col_ver = ("n_ms" if smoothed else "n_m") if hemisphere == "north" else ("s_ms" if smoothed else "s_m")

    ver = load_veronig_monthly()
    ver_part = ver[col_ver]
    ver_part = ver_part[ver_part.index < SPLICE_DATE]

    silso_path = SILSO_MS_HEM if smoothed else SILSO_M_HEM
    silso = load_silso_hem_monthly(silso_path, 13)
    silso_col = "n_sn" if hemisphere == "north" else "s_sn"
    silso_part = silso[silso_col]
    silso_part = silso_part[silso_part.index >= SPLICE_DATE]

    combined = pd.concat([ver_part.rename("sn"), silso_part.rename("sn")])
    combined = combined.dropna().sort_index()
    return combined.astype(np.float32)


if __name__ == "__main__":
    corr_n, corr_s, n_overlap = validate_overlap()
    print(f"Overlap validation (Veronig vs SILSO official, {n_overlap} months, 1992-2020):")
    print(f"  North correlation: {corr_n:.5f}")
    print(f"  South correlation: {corr_s:.5f}")

    for hemi in ["north", "south"]:
        for smoothed in [False, True]:
            s = build_hemisphere_series(hemi, smoothed)
            label = f"{hemi} {'smoothed' if smoothed else 'monthly'}"
            print(f"\n{label}: n={len(s)}  {s.index.min().date()} to {s.index.max().date()}")
            print(s.tail(3))
