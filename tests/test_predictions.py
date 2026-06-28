"""
Test: Prediction accuracy — how close is ESS to the actual bloom peak?

Metrics:
  - Days error: predicted peak vs actual peak
  - Transition lead time: how many days before peak did we detect it?
  - Hit rate: across all datasets, % where transition was detected at all
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path

from utils.data_loader import load_bloom_timeline, extract_window, list_bloom_files
from models.ess_model import predict_transition_day


def test_peak_prediction_error(filepath: str) -> dict:
    """
    Run ESS on one bloom timeline and measure error from actual peak.
    """
    df = load_bloom_timeline(filepath)
    pre_bloom, post_bloom, actual_peak = extract_window(df)

    # Run prediction on pre-bloom data only (fair test)
    result = predict_transition_day(pre_bloom)

    # Also run on full data to get predicted peak
    full_result = predict_transition_day(df)

    error_days = None
    if full_result["predicted_peak_date"] is not None:
        error_days = (full_result["predicted_peak_date"] - actual_peak).days

    return {
        "file": Path(filepath).name,
        "actual_peak": actual_peak,
        "predicted_peak": full_result["predicted_peak_date"],
        "peak_error_days": error_days,
        "transition_detected": result["transition_date"] is not None,
        "transition_date": result["transition_date"],
        "lead_time_days": result["days_before_peak"],
    }


def run_all_prediction_tests(data_dir: str = "data/bloom_timelines") -> pd.DataFrame:
    """
    Run prediction accuracy tests on all bloom timeline files.
    Returns a summary DataFrame.
    """
    files = list_bloom_files(data_dir)
    if not files:
        print(f"No data files found in {data_dir}/")
        return pd.DataFrame()

    results = []
    for f in files:
        print(f"  Testing: {f.name}")
        try:
            result = test_peak_prediction_error(str(f))
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"file": f.name, "error": str(e)})

    summary = pd.DataFrame(results)

    # Compute aggregate metrics
    if "peak_error_days" in summary.columns:
        valid = summary["peak_error_days"].dropna()
        if len(valid) > 0:
            print(f"\n  Mean absolute error: {valid.abs().mean():.1f} days")
            print(f"  Max error: {valid.abs().max():.0f} days")
            print(f"  Detection rate: {summary['transition_detected'].mean()*100:.0f}%")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Prediction Accuracy")
    print("=" * 60)
    summary = run_all_prediction_tests()
    if len(summary) > 0:
        output_path = "outputs/predictions/accuracy_results.csv"
        summary.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("No datasets to test. Add Excel files to data/bloom_timelines/")
