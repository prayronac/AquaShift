"""
Baseline models for comparison against the ESS approach.

These implement simpler/standard methods so you can show ESS outperforms them.
"""

import numpy as np
import pandas as pd


def threshold_model(df: pd.DataFrame, chl_threshold: float = 10.0) -> dict:
    """
    Naive threshold model: predict bloom when chlorophyll exceeds a fixed value.
    
    This is the simplest possible approach — just a hard cutoff on one parameter.
    Your ESS model should beat this easily, but it's a useful floor.
    """
    df = df.copy()
    bloom_detected = df[df["chlorophyll"] >= chl_threshold]

    if len(bloom_detected) > 0:
        transition_date = bloom_detected.iloc[0]["date"]
        peak_date = df.loc[df["chlorophyll"].idxmax(), "date"]
    else:
        transition_date = None
        peak_date = None

    return {
        "model": "threshold",
        "transition_date": transition_date,
        "predicted_peak_date": peak_date,
    }


def rate_of_change_model(df: pd.DataFrame, roc_threshold: float = 0.3) -> dict:
    """
    Rate-of-change model: predict bloom when chlorophyll rate of change
    exceeds a threshold (day-over-day percentage increase).
    
    Slightly smarter than threshold — looks at acceleration, not just level.
    """
    df = df.copy().sort_values("date")
    df["chl_roc"] = df["chlorophyll"].pct_change()

    bloom_detected = df[df["chl_roc"] >= roc_threshold]

    if len(bloom_detected) > 0:
        transition_date = bloom_detected.iloc[0]["date"]
        peak_date = df.loc[df["chlorophyll"].idxmax(), "date"]
    else:
        transition_date = None
        peak_date = None

    return {
        "model": "rate_of_change",
        "transition_date": transition_date,
        "predicted_peak_date": peak_date,
    }


def multi_param_threshold_model(df: pd.DataFrame) -> dict:
    """
    Multi-parameter threshold model: requires several parameters to
    simultaneously exceed favorable ranges.
    
    This is closer to what monitoring agencies actually use — but still
    lacks the stoichiometric integration that ESS provides.
    """
    df = df.copy()

    conditions = (
        (df["ph"] >= 7.5) &
        (df["temperature"] >= 20.0) &
        (df["chlorophyll"] >= 5.0) &
        (df["dissolved_oxygen"] >= 6.0)
    )

    bloom_detected = df[conditions]

    if len(bloom_detected) > 0:
        transition_date = bloom_detected.iloc[0]["date"]
        peak_date = df.loc[df["chlorophyll"].idxmax(), "date"]
    else:
        transition_date = None
        peak_date = None

    return {
        "model": "multi_param_threshold",
        "transition_date": transition_date,
        "predicted_peak_date": peak_date,
    }
