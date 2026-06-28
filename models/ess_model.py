"""
ESS Model — Equilibrium-State Stoichiometry for bloom prediction.

Core idea:
  The Redfield equation describes photosynthetic biomass production:
    106 CO2 + 16 HNO3 + H3PO4 + 122 H2O  ⇌  biomass + 138 O2

  K_eq = equilibrium constant (calculated from Anabaena literature parameters)
  Q    = reaction quotient (calculated from daily measurements)

  When Q > K for sustained periods → conditions favor biomass production
  → phase shift (transition from linear to exponential growth) is imminent.

The model tracks Q/K ratios over time and detects the inflection point.
"""

import numpy as np
import pandas as pd
from utils.constants import REDFIELD, ANABAENA_FAVORABLE


# ── K Calculation ─────────────────────────────────────────────────────────

def calculate_K(temperature_c: float = 25.0) -> float:
    """
    Calculate the equilibrium constant K for the Redfield biomass equation.
    
    K represents the threshold Q value above which conditions favor rapid
    biomass production. It's calibrated to Anabaena-favorable conditions
    with van 't Hoff temperature correction.
    
    Returns: K value (dimensionless, normalized favorability score)
    """
    # K is the Q value that would be computed under "just barely favorable"
    # conditions — the boundary between bloom and non-bloom.
    # We compute it from the lower bounds of Anabaena's favorable range.
    fav = ANABAENA_FAVORABLE

    # Score each parameter at the favorable threshold
    ph_score = _ph_favorability(fav["ph_min"])          # just enters favorable
    temp_score = _temp_favorability(fav["temperature_min_c"])
    chl_score = _biomass_score(fav["chlorophyll_bloom_ug_l"], fav["chlorophyll_bloom_ug_l"])
    do_score = _do_favorability(fav["do_min_mg_l"])

    K_ref = ph_score * temp_score * chl_score * do_score

    # Van 't Hoff temperature adjustment
    E_a = 50_000  # J/mol (cyanobacteria photosynthesis activation energy)
    R = 8.314     # J/(mol·K)
    T_ref = 25.0 + 273.15
    T_actual = temperature_c + 273.15
    temp_factor = np.exp((E_a / R) * (1 / T_ref - 1 / T_actual))

    return K_ref * temp_factor


def _ph_favorability(ph: float) -> float:
    """Score pH on 0-1 scale. Peak at 8.5, drops off outside 7.5-9.5."""
    optimal = 8.5
    return max(0, 1.0 - ((ph - optimal) / 2.0) ** 2)


def _temp_favorability(temp_c: float) -> float:
    """Score temperature on 0-1 scale. Peak at 25°C."""
    optimal = 25.0
    return max(0, 1.0 - ((temp_c - optimal) / 10.0) ** 2)


def _do_favorability(do_mg_l: float) -> float:
    """Score dissolved oxygen. Higher DO suggests active photosynthesis."""
    return min(do_mg_l / 10.0, 1.0)


def _biomass_score(chl: float, threshold: float = 10.0) -> float:
    """Score biomass level relative to bloom threshold."""
    return min(chl / threshold, 2.0)  # can exceed 1 during bloom


# ── Q Calculation ─────────────────────────────────────────────────────────

def calculate_Q(row: pd.Series) -> float:
    """
    Calculate the reaction quotient Q from a single day's measurements.
    
    Q is a composite favorability score derived from Redfield-relevant proxies.
    Each measured parameter is scored on a normalized scale reflecting how
    favorable it is for the forward (biomass production) direction.
    
    When Q > K, all parameters are simultaneously favorable → bloom likely.
    
    Returns: Q value (dimensionless favorability score)
    """
    ph = row.get("ph", 7.0)
    chl = max(row.get("chlorophyll", 0.1), 0.01)
    pc = max(row.get("phycocyanin", 0.1), 0.01)
    do = max(row.get("dissolved_oxygen", 0.1), 0.01)
    turb = max(row.get("turbidity", 0.1), 0.01)
    temp = row.get("temperature", 20.0)

    # Score each parameter
    ph_score = _ph_favorability(ph)
    temp_score = _temp_favorability(temp)
    do_score = _do_favorability(do)

    # Biomass indicators (products side of Redfield equation)
    # Chlorophyll: general biomass. Phycocyanin: cyano-specific.
    # Weight phycocyanin more heavily (cyano-specific signal)
    chl_ref = ANABAENA_FAVORABLE["chlorophyll_bloom_ug_l"]
    biomass_score = _biomass_score(chl, chl_ref) * 0.4 + min(pc / 5.0, 2.0) * 0.6

    # Q = product of all favorability scores
    Q = ph_score * temp_score * do_score * biomass_score

    return Q


# ── Daily Q/K Series ──────────────────────────────────────────────────────

def compute_qk_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Q, K, and Q/K ratio for each day in the dataframe.
    
    Returns: DataFrame with added columns: Q, K, QK_ratio, Q_exceeds_K
    """
    df = df.copy()

    df["K"] = df["temperature"].apply(calculate_K)
    df["Q"] = df.apply(calculate_Q, axis=1)
    df["QK_ratio"] = df["Q"] / df["K"]
    df["Q_exceeds_K"] = df["QK_ratio"] > 1.0

    return df


# ── Phase Shift Detection ────────────────────────────────────────────────

def predict_transition_day(df: pd.DataFrame, sustained_days: int = 3) -> dict:
    """
    Predict the day of phase shift (transition to exponential growth).
    
    Logic:
      1. Compute Q/K ratio for each day
      2. Find where Q/K > 1 for `sustained_days` consecutive days
      3. The first day of that sustained run is the predicted transition
    
    Also detects the predicted peak as the day with maximum Q/K ratio.
    
    Args:
        df: DataFrame with measurement columns
        sustained_days: number of consecutive Q>K days required to call it
    
    Returns: dict with:
        - transition_date: predicted phase shift date (or None)
        - peak_date: predicted bloom peak date
        - qk_series: the full Q/K DataFrame for analysis
        - days_before_peak: how many days before peak the transition was detected
    """
    if len(df) == 0:
        return {
            "transition_date": None,
            "predicted_peak_date": None,
            "days_before_peak": None,
            "qk_series": df,
        }

    qk = compute_qk_series(df)

    # Find sustained Q > K runs
    transition_date = None
    exceeds = qk["Q_exceeds_K"].values
    for i in range(len(exceeds) - sustained_days + 1):
        if all(exceeds[i : i + sustained_days]):
            transition_date = qk.iloc[i]["date"]
            break

    # Peak = max Q/K ratio day
    valid_qk = qk["QK_ratio"].dropna()
    if len(valid_qk) == 0:
        return {
            "transition_date": transition_date,
            "predicted_peak_date": None,
            "days_before_peak": None,
            "qk_series": qk,
        }
    peak_idx = valid_qk.idxmax()
    predicted_peak = qk.loc[peak_idx, "date"]

    # Days between transition and peak
    days_before_peak = None
    if transition_date is not None:
        days_before_peak = (predicted_peak - transition_date).days

    return {
        "transition_date": transition_date,
        "predicted_peak_date": predicted_peak,
        "days_before_peak": days_before_peak,
        "qk_series": qk,
    }


# ── Bloom Rate Classification ────────────────────────────────────────────

def predict_bloom_rate(df: pd.DataFrame) -> dict:
    """
    Classify the bloom as fast / slow / steady exponential growth.
    
    Fits an exponential model to the chlorophyll or Q/K trajectory
    in the pre-transition → peak window.
    
    Returns: dict with rate_class, doubling_time, r_squared, exp_params
    """
    from scipy.optimize import curve_fit

    qk = compute_qk_series(df)

    # Use Q/K ratio as the signal to fit
    qk = qk.dropna(subset=["QK_ratio"])
    if len(qk) < 4:
        return {"rate_class": "insufficient_data", "doubling_time": None,
                "r_squared": None, "exp_params": None}

    # Days since first measurement
    t = (qk["date"] - qk["date"].iloc[0]).dt.total_seconds() / 86400
    t = t.values.astype(float)
    y = qk["QK_ratio"].values.astype(float)

    # Exponential model: y = a * exp(r * t)
    def exp_model(t, a, r):
        return a * np.exp(r * t)

    try:
        popt, _ = curve_fit(exp_model, t, y, p0=[y[0], 0.1], maxfev=5000)
        a, r = popt

        # R-squared
        y_pred = exp_model(t, a, r)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Doubling time
        doubling_time = np.log(2) / r if r > 0 else None

        # Classify
        if doubling_time is None or doubling_time > 14:
            rate_class = "slow"
        elif doubling_time > 5:
            rate_class = "steady"
        else:
            rate_class = "fast"

    except (RuntimeError, ValueError):
        a, r = None, None
        r_squared = None
        doubling_time = None
        rate_class = "fit_failed"

    return {
        "rate_class": rate_class,
        "doubling_time": doubling_time,
        "r_squared": r_squared,
        "exp_params": {"a": a, "r": r},
    }
