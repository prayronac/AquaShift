"""
Test: Model comparisons — ESS vs baseline methods on the same datasets.

Compares:
  1. ESS (your model)
  2. Threshold model (chlorophyll > X)
  3. Rate-of-change model (chlorophyll acceleration)
  4. Multi-parameter threshold model

Metrics per model:
  - Mean absolute error (days from actual peak)
  - Detection rate (did it find a transition at all?)
  - Lead time (how early was the warning?)
  - False positive rate on negative controls
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path

from utils.data_loader import load_bloom_timeline, extract_window, list_bloom_files
from models.ess_model import predict_transition_day
from models.baseline_models import (
    threshold_model,
    rate_of_change_model,
    multi_param_threshold_model,
)


def compare_models_on_file(filepath: str) -> list:
    """Run all models on one bloom timeline and return comparison rows."""
    df = load_bloom_timeline(filepath)
    pre_bloom, post_bloom, actual_peak = extract_window(df)
    fname = Path(filepath).name

    results = []

    # ── ESS Model ──
    ess = predict_transition_day(df)
    ess_full = predict_transition_day(df)
    ess_error = None
    if ess_full["predicted_peak_date"] is not None:
        ess_error = (ess_full["predicted_peak_date"] - actual_peak).days

    ess_pre = predict_transition_day(pre_bloom)
    results.append({
        "file": fname,
        "model": "ESS",
        "predicted_peak": ess_full["predicted_peak_date"],
        "actual_peak": actual_peak,
        "peak_error_days": ess_error,
        "transition_detected": ess_pre["transition_date"] is not None,
        "transition_date": ess_pre["transition_date"],
    })

    # ── Threshold Model ──
    thr = threshold_model(df)
    thr_error = None
    if thr["predicted_peak_date"] is not None:
        thr_error = (thr["predicted_peak_date"] - actual_peak).days
    thr_pre = threshold_model(pre_bloom)
    results.append({
        "file": fname,
        "model": "Threshold",
        "predicted_peak": thr["predicted_peak_date"],
        "actual_peak": actual_peak,
        "peak_error_days": thr_error,
        "transition_detected": thr_pre["transition_date"] is not None,
        "transition_date": thr_pre["transition_date"],
    })

    # ── Rate of Change Model ──
    roc = rate_of_change_model(df)
    roc_error = None
    if roc["predicted_peak_date"] is not None:
        roc_error = (roc["predicted_peak_date"] - actual_peak).days
    roc_pre = rate_of_change_model(pre_bloom)
    results.append({
        "file": fname,
        "model": "Rate_of_Change",
        "predicted_peak": roc["predicted_peak_date"],
        "actual_peak": actual_peak,
        "peak_error_days": roc_error,
        "transition_detected": roc_pre["transition_date"] is not None,
        "transition_date": roc_pre["transition_date"],
    })

    # ── Multi-Param Threshold ──
    mpt = multi_param_threshold_model(df)
    mpt_error = None
    if mpt["predicted_peak_date"] is not None:
        mpt_error = (mpt["predicted_peak_date"] - actual_peak).days
    mpt_pre = multi_param_threshold_model(pre_bloom)
    results.append({
        "file": fname,
        "model": "Multi_Param_Threshold",
        "predicted_peak": mpt["predicted_peak_date"],
        "actual_peak": actual_peak,
        "peak_error_days": mpt_error,
        "transition_detected": mpt_pre["transition_date"] is not None,
        "transition_date": mpt_pre["transition_date"],
    })

    return results


def run_all_comparisons(data_dir: str = "data/bloom_timelines") -> pd.DataFrame:
    """Run model comparisons on all bloom timeline files."""
    files = list_bloom_files(data_dir)
    if not files:
        print(f"No data files found in {data_dir}/")
        return pd.DataFrame()

    all_results = []
    for f in files:
        print(f"  Comparing models on: {f.name}")
        try:
            results = compare_models_on_file(str(f))
            all_results.extend(results)
        except Exception as e:
            print(f"    ERROR: {e}")

    comparison = pd.DataFrame(all_results)

    # Print summary per model
    if len(comparison) > 0:
        print("\n  Model Comparison Summary:")
        print("  " + "-" * 55)
        for model in comparison["model"].unique():
            m = comparison[comparison["model"] == model]
            valid_errors = m["peak_error_days"].dropna()
            mae = valid_errors.abs().mean() if len(valid_errors) > 0 else float("nan")
            det_rate = m["transition_detected"].mean() * 100
            print(f"  {model:25s}  MAE: {mae:5.1f} days  Detection: {det_rate:.0f}%")

    return comparison


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Model Comparisons")
    print("=" * 60)
    comparison = run_all_comparisons()
    if len(comparison) > 0:
        output_path = "outputs/comparisons/model_comparison.csv"
        comparison.to_csv(output_path, index=False)
        print(f"\n  Results saved to {output_path}")
    else:
        print("No datasets to compare. Add Excel files to data/bloom_timelines/")
