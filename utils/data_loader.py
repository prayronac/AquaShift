"""
Data loading, sliding-window extraction, and normalization utilities.

The "sliding window" approach:
  Given a bloom peak date, extract N days before and M days after.
  For prediction, only pre-bloom data is visible to the model.
  Post-bloom data is kept for validation (did the model call it right?).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Maps variant column names (after lowercase/strip normalization) to the
# canonical names expected by the model.
_COLUMN_ALIASES = {
    "chlorophyll_rfu": "chlorophyll",
    "chlorophyll_relative_fluorescence_(fchl)": "chlorophyll",
    "phycocyanin_rfu": "phycocyanin",
    "phycocyanin_relative_fluorescence_(fpc)": "phycocyanin",
    "dissolved_oxygen_mg_l": "dissolved_oxygen",
    "turbidity_fnu": "turbidity",
}


def load_bloom_timeline(filepath: str) -> pd.DataFrame:
    """
    Load an Excel file from bloom_timelines/.
    
    Expected columns: date, chlorophyll, phycocyanin, turbidity,
                      dissolved_oxygen, ph, temperature, bloom_peak_date
    
    Returns a DataFrame sorted by date with parsed datetime columns.
    """
    filepath = Path(filepath)
    if filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")

    # Normalize column names: lowercase, strip whitespace, replace spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Apply column aliases (e.g. chlorophyll_rfu → chlorophyll)
    df.rename(columns=_COLUMN_ALIASES, inplace=True)

    # Parse dates; strip timezone so comparisons are always tz-naive
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["bloom_peak_date"] = pd.to_datetime(df["bloom_peak_date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def extract_window(df: pd.DataFrame, days_before: int = 21, days_after: int = 21):
    """
    Extract a time window around the bloom peak.
    
    Returns:
        pre_bloom  — DataFrame of rows before (and including) bloom peak
        post_bloom — DataFrame of rows after bloom peak
        peak_date  — the bloom peak date
    """
    peak_date = df["bloom_peak_date"].dropna().iloc[0]
    window_start = peak_date - timedelta(days=days_before)
    window_end = peak_date + timedelta(days=days_after)

    windowed = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()
    pre_bloom = windowed[windowed["date"] <= peak_date].copy()
    post_bloom = windowed[windowed["date"] > peak_date].copy()

    return pre_bloom, post_bloom, peak_date


def sliding_prediction_windows(df: pd.DataFrame, window_size: int = 7, step: int = 1):
    """
    Generate overlapping windows for running predictions over time.
    
    Yields (window_df, window_end_date) tuples.
    Each window contains `window_size` consecutive days of data.
    The model sees only this window and must decide: bloom coming or not?
    
    This simulates real-time monitoring where you only see the last N days.
    """
    dates = df["date"].unique()
    dates.sort()

    for i in range(window_size, len(dates) + 1, step):
        window_dates = dates[i - window_size : i]
        window_df = df[df["date"].isin(window_dates)].copy()
        yield window_df, window_dates[-1]


def normalize_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Min-max normalize numeric columns to [0, 1] range.
    Useful for comparing parameters on the same scale.
    
    Returns a copy with '_norm' suffix columns added.
    """
    if columns is None:
        columns = ["chlorophyll", "phycocyanin", "turbidity",
                    "dissolved_oxygen", "ph", "temperature"]

    df = df.copy()
    for col in columns:
        if col in df.columns:
            cmin, cmax = df[col].min(), df[col].max()
            if cmax > cmin:
                df[f"{col}_norm"] = (df[col] - cmin) / (cmax - cmin)
            else:
                df[f"{col}_norm"] = 0.0
    return df


def list_bloom_files(data_dir: str = "data/bloom_timelines") -> list:
    """List all Excel/CSV files in the bloom timelines directory."""
    data_path = Path(data_dir)
    files = list(data_path.glob("*.xlsx")) + list(data_path.glob("*.csv"))
    # Exclude Excel lock files (created when a file is open in Excel)
    files = [f for f in files if not f.name.startswith("~$")]
    return sorted(files)
