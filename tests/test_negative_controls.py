"""
Test: Negative controls — the model should NOT predict a bloom under
conditions where one would not occur.

Controls:
  1. Low pH (acidic water, pH < 6.5) — unfavorable for cyanobacteria
  2. Low nutrients (oligotrophic conditions) — no fuel for growth
  3. Low chlorophyll (no existing biomass to seed exponential growth)

For each, we generate synthetic "no-bloom" data and verify ESS does NOT
predict a transition. A good model rejects false positives.
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from models.ess_model import predict_transition_day


def _make_synthetic_data(days: int = 42, **overrides) -> pd.DataFrame:
    """
    Generate synthetic daily measurements.
    Default values represent normal, non-bloom conditions.
    Override specific parameters to create test scenarios.
    """
    dates = [datetime(2024, 6, 1) + timedelta(days=i) for i in range(days)]

    # Default: boring, non-bloom water
    defaults = {
        "chlorophyll": np.random.uniform(1.0, 3.0, days),
        "phycocyanin": np.random.uniform(0.5, 2.0, days),
        "turbidity": np.random.uniform(2.0, 5.0, days),
        "dissolved_oxygen": np.random.uniform(6.0, 8.0, days),
        "ph": np.random.uniform(7.0, 7.5, days),
        "temperature": np.random.uniform(18.0, 22.0, days),
    }
    defaults.update(overrides)

    df = pd.DataFrame({
        "date": dates,
        **defaults,
        "bloom_peak_date": datetime(2024, 6, 22),  # arbitrary
    })
    return df


def test_low_ph_no_bloom():
    """
    Low pH (< 6.5): Cyanobacteria strongly prefer alkaline conditions.
    The model should NOT predict a bloom at acidic pH.
    """
    df = _make_synthetic_data(
        ph=np.random.uniform(5.5, 6.3, 42),
        temperature=np.random.uniform(22.0, 26.0, 42),  # warm enough otherwise
    )
    result = predict_transition_day(df)
    bloom_predicted = result["transition_date"] is not None

    status = "PASS" if not bloom_predicted else "FAIL"
    print(f"  [Low pH control]       {status} — bloom predicted: {bloom_predicted}")
    return {"test": "low_ph", "bloom_predicted": bloom_predicted, "pass": not bloom_predicted}


def test_low_nutrients_no_bloom():
    """
    Low nutrient conditions: very low chlorophyll and phycocyanin
    throughout the timeline (oligotrophic water).
    """
    df = _make_synthetic_data(
        chlorophyll=np.random.uniform(0.1, 0.5, 42),
        phycocyanin=np.random.uniform(0.05, 0.3, 42),
        turbidity=np.random.uniform(0.5, 1.5, 42),   # clear water
        dissolved_oxygen=np.random.uniform(7.0, 9.0, 42),
    )
    result = predict_transition_day(df)
    bloom_predicted = result["transition_date"] is not None

    status = "PASS" if not bloom_predicted else "FAIL"
    print(f"  [Low nutrients ctrl]   {status} — bloom predicted: {bloom_predicted}")
    return {"test": "low_nutrients", "bloom_predicted": bloom_predicted, "pass": not bloom_predicted}


def test_low_chlorophyll_no_bloom():
    """
    Low chlorophyll before bloom window: if there's no seed population,
    even favorable conditions shouldn't trigger an exponential transition.
    """
    df = _make_synthetic_data(
        chlorophyll=np.random.uniform(0.2, 1.0, 42),
        ph=np.random.uniform(8.0, 8.5, 42),           # favorable pH
        temperature=np.random.uniform(23.0, 27.0, 42), # favorable temp
        dissolved_oxygen=np.random.uniform(5.0, 7.0, 42),
    )
    result = predict_transition_day(df)
    bloom_predicted = result["transition_date"] is not None

    status = "PASS" if not bloom_predicted else "FAIL"
    print(f"  [Low chlorophyll ctrl] {status} — bloom predicted: {bloom_predicted}")
    return {"test": "low_chlorophyll", "bloom_predicted": bloom_predicted, "pass": not bloom_predicted}


def test_stable_conditions_no_bloom():
    """
    Bonus control: completely flat, unchanging measurements.
    No trend should mean no predicted transition.
    """
    df = _make_synthetic_data(
        chlorophyll=np.full(42, 2.0),
        phycocyanin=np.full(42, 1.0),
        turbidity=np.full(42, 3.0),
        dissolved_oxygen=np.full(42, 7.5),
        ph=np.full(42, 7.2),
        temperature=np.full(42, 20.0),
    )
    result = predict_transition_day(df)
    bloom_predicted = result["transition_date"] is not None

    status = "PASS" if not bloom_predicted else "FAIL"
    print(f"  [Stable flat ctrl]     {status} — bloom predicted: {bloom_predicted}")
    return {"test": "stable_flat", "bloom_predicted": bloom_predicted, "pass": not bloom_predicted}


def run_all_negative_controls() -> pd.DataFrame:
    """Run all negative control tests and return summary."""
    results = [
        test_low_ph_no_bloom(),
        test_low_nutrients_no_bloom(),
        test_low_chlorophyll_no_bloom(),
        test_stable_conditions_no_bloom(),
    ]
    summary = pd.DataFrame(results)
    passed = summary["pass"].sum()
    total = len(summary)
    print(f"\n  Negative controls: {passed}/{total} passed")
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Negative Controls")
    print("=" * 60)
    summary = run_all_negative_controls()
    output_path = "outputs/predictions/negative_control_results.csv"
    summary.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
