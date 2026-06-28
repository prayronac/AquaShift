"""
ESS Threshold Optimizer
========================
Sweeps phycocyanin bloom thresholds and evaluates ESS + baseline model
performance at each value. Finds the threshold that maximizes a combined
objective across precision, recall, and confusion matrix quality.

The chemistry layer (Q/K ratios, convergence features) is computed ONCE.
Only the bloom labeling and downstream windowing/evaluation are re-run
per threshold, since those are the only steps affected.

USAGE:
  Place in same directory as ess_v3.py, then:
    python threshold_optimizer.py

OUTPUTS:
  - threshold_sweep.png          -- metrics vs threshold plot
  - threshold_results.csv        -- full numeric results
  - Console summary of optimal threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, auc,
    matthews_corrcoef, confusion_matrix,
    balanced_accuracy_score
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. LOAD AND PREPARE DATA (once)
# =============================================================================

def prepare_daily_data():
    """Run Lake Erie pipeline through daily aggregation.
    Returns daily DataFrame with all chemistry computed but NO bloom labels yet."""
    import ess_v3 as erie

    file_pairs = [
        ('data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary_phosphate.csv'),
    ]

    print("Loading and preparing Lake Erie data (one-time)...")
    df = erie.load_and_combine(file_pairs)

    col_map = {
        'water_temperature': 'temp_c',
        'organic_dissolved_oxygen': 'do_mgl',
        'pH': 'ph',
        'phosphate': 'srp_ugl',
        'phycocyanin': 'pc_rfu',
        'chlorophylla': 'chla_rfu',
        'specific_conductivity': 'cond_uscm',
    }
    df = df.rename(columns=col_map)
    for col in ['temp_c', 'do_mgl', 'ph', 'srp_ugl', 'pc_rfu', 'chla_rfu', 'cond_uscm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'cond_uscm' in df.columns:
        df['cond_uscm'] = df['cond_uscm'] * 1000

    if 'srp_ugl' in df.columns:
        df['srp_ugl'] = df['srp_ugl'].fillna(df['srp_ugl'].median())
    else:
        df['srp_ugl'] = 5.0

    required = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'chla_rfu', 'cond_uscm']
    df = df.dropna(subset=required).reset_index(drop=True)

    # Chemistry (threshold-independent)
    df = erie.compute_derived_variables(df)
    df = erie.compute_qk_ratios(df)
    df = erie.compute_convergence_features(df)
    df = erie.compute_ecological_ratios(df)

    # Daily aggregation (aggregate pc_rfu as mean for threshold application)
    df['date'] = df['timestamp'].dt.date
    exclude_cols = ['timestamp', 'date', 'source_file']
    agg_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    agg_dict = {col: 'mean' for col in agg_cols}
    # Keep pc_rfu as mean for threshold comparison
    daily = df.groupby('date').agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    print(f"  Daily data: {len(daily)} rows")
    print(f"  PC RFU range: {daily['pc_rfu'].min():.3f} to {daily['pc_rfu'].max():.3f}")
    print(f"  PC RFU median: {daily['pc_rfu'].median():.3f}")
    print(f"  PC RFU mean:   {daily['pc_rfu'].mean():.3f}")

    return daily


# =============================================================================
# 2. EVALUATE AT A SINGLE THRESHOLD
# =============================================================================

def evaluate_at_threshold(daily, threshold, erie_module):
    """Label blooms at given threshold, window, and evaluate.
    Returns dict of metrics or None if insufficient positive samples."""

    # Re-label
    daily_copy = daily.copy()
    daily_copy['bloom_label'] = (daily_copy['pc_rfu'] >= threshold).astype(int)

    n_pos = int(daily_copy['bloom_label'].sum())
    n_neg = len(daily_copy) - n_pos

    if n_pos < 10 or n_neg < 10:
        return None  # Not enough of either class

    # Segment and window
    segments = erie_module.identify_continuous_segments(daily_copy)
    X_baseline, X_ess, y = erie_module.build_sliding_windows_multi(segments)

    n_pos_w = int(y.sum())
    n_neg_w = len(y) - n_pos_w

    if n_pos_w < 5 or n_neg_w < 5:
        return None

    # Cross-validate ESS model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_clean = np.nan_to_num(X_ess.values, nan=0.0, posinf=10, neginf=-10)

    try:
        pipe_ess = make_pipeline()
        y_prob_ess = cross_val_predict(pipe_ess, X_clean, y, cv=skf, method="predict_proba")[:, 1]
        y_pred_ess = cross_val_predict(pipe_ess, X_clean, y, cv=skf)

        pipe_base = make_pipeline()
        X_base_clean = np.nan_to_num(X_baseline.values, nan=0.0, posinf=10, neginf=-10)
        y_prob_base = cross_val_predict(pipe_base, X_base_clean, y, cv=skf, method="predict_proba")[:, 1]
        y_pred_base = cross_val_predict(pipe_base, X_base_clean, y, cv=skf)
    except Exception:
        return None

    cm_ess = confusion_matrix(y, y_pred_ess)
    cm_base = confusion_matrix(y, y_pred_base)

    prec_curve, rec_curve, _ = precision_recall_curve(y, y_prob_ess)
    prauc_ess = auc(rec_curve, prec_curve)

    prec_curve_b, rec_curve_b, _ = precision_recall_curve(y, y_prob_base)
    prauc_base = auc(rec_curve_b, prec_curve_b)

    # --- Per-class metrics (the fix for the low-threshold trap) ---
    # Bloom class (positive)
    bloom_f1 = f1_score(y, y_pred_ess, pos_label=1)
    bloom_prec = precision_score(y, y_pred_ess, pos_label=1, zero_division=0)
    bloom_rec = recall_score(y, y_pred_ess, pos_label=1)  # = sensitivity = TPR

    # No-bloom class (negative) -- THIS is what was missing
    no_bloom_f1 = f1_score(y, y_pred_ess, pos_label=0)
    no_bloom_prec = precision_score(y, y_pred_ess, pos_label=0, zero_division=0)  # = NPV
    specificity = recall_score(y, y_pred_ess, pos_label=0)  # = TNR

    # Macro F1: average of bloom F1 and no-bloom F1
    # Only high when BOTH classes are well-predicted
    macro_f1 = f1_score(y, y_pred_ess, average='macro')

    # Same for baseline
    base_bloom_f1 = f1_score(y, y_pred_base, pos_label=1)
    base_no_bloom_f1 = f1_score(y, y_pred_base, pos_label=0)
    base_macro_f1 = f1_score(y, y_pred_base, average='macro')
    base_specificity = recall_score(y, y_pred_base, pos_label=0)

    return {
        'threshold': threshold,
        'n_bloom_days': n_pos,
        'n_no_bloom_days': n_neg,
        'bloom_pct': 100 * n_pos / len(daily_copy),
        'n_windows': len(y),
        'n_pos_windows': n_pos_w,
        'n_neg_windows': n_neg_w,
        'pos_window_pct': 100 * n_pos_w / len(y),
        # ESS: bloom class
        'ess_bloom_f1': bloom_f1,
        'ess_bloom_prec': bloom_prec,
        'ess_bloom_rec': bloom_rec,       # sensitivity / TPR
        # ESS: no-bloom class
        'ess_no_bloom_f1': no_bloom_f1,
        'ess_no_bloom_prec': no_bloom_prec,  # NPV
        'ess_specificity': specificity,       # TNR
        # ESS: aggregate
        'ess_macro_f1': macro_f1,
        'ess_prauc': prauc_ess,
        'ess_mcc': matthews_corrcoef(y, y_pred_ess),
        'ess_balanced_acc': balanced_accuracy_score(y, y_pred_ess),
        'ess_tn': cm_ess[0, 0],
        'ess_fp': cm_ess[0, 1],
        'ess_fn': cm_ess[1, 0],
        'ess_tp': cm_ess[1, 1],
        # Baseline
        'base_bloom_f1': base_bloom_f1,
        'base_no_bloom_f1': base_no_bloom_f1,
        'base_macro_f1': base_macro_f1,
        'base_specificity': base_specificity,
        'base_prauc': prauc_base,
        'base_mcc': matthews_corrcoef(y, y_pred_base),
        'base_balanced_acc': balanced_accuracy_score(y, y_pred_base),
        'base_tn': cm_base[0, 0],
        'base_fp': cm_base[0, 1],
        'base_fn': cm_base[1, 0],
        'base_tp': cm_base[1, 1],
        # ESS advantage (using macro F1 -- not bloom-only)
        'delta_macro_f1': macro_f1 - base_macro_f1,
        'delta_prauc': prauc_ess - prauc_base,
        'delta_mcc': matthews_corrcoef(y, y_pred_ess) - matthews_corrcoef(y, y_pred_base),
    }


def make_pipeline():
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])


# =============================================================================
# 3. SWEEP THRESHOLDS
# =============================================================================

def sweep_thresholds(daily, erie_module, thresholds=None):
    """Evaluate model performance across a range of bloom thresholds."""

    if thresholds is None:
        # Sweep from 10th percentile to 90th percentile of PC RFU
        pc_values = daily['pc_rfu'].dropna()
        p10 = pc_values.quantile(0.10)
        p90 = pc_values.quantile(0.90)
        thresholds = np.linspace(max(p10, 0.1), p90, 40)
        # Also include some commonly-used values
        common = [0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]
        common = [c for c in common if p10 <= c <= p90 * 1.2]
        thresholds = np.unique(np.sort(np.concatenate([thresholds, common])))

    print(f"\nSweeping {len(thresholds)} thresholds from {thresholds[0]:.3f} to {thresholds[-1]:.3f} RFU...")
    print(f"{'Threshold':>10} {'Bloom%':>7} {'Windows':>8} {'MacroF1':>8} {'BloomF1':>8} "
          f"{'NoBlmF1':>8} {'Specif':>7} {'MCC':>7}")
    print("-" * 80)

    results = []
    for i, thr in enumerate(thresholds):
        result = evaluate_at_threshold(daily, thr, erie_module)
        if result is not None:
            results.append(result)
            r = result
            print(f"  {thr:>8.3f}  {r['bloom_pct']:>5.1f}%  {r['n_windows']:>7d}  "
                  f"{r['ess_macro_f1']:>7.4f}  {r['ess_bloom_f1']:>7.4f}  "
                  f"{r['ess_no_bloom_f1']:>7.4f}  {r['ess_specificity']:>6.4f}  "
                  f"{r['ess_mcc']:>6.4f}")
        else:
            print(f"  {thr:>8.3f}  -- skipped (insufficient class balance)")

    return pd.DataFrame(results)


# =============================================================================
# 4. FIND OPTIMAL THRESHOLD
# =============================================================================

def find_optimal(df):
    """
    Find optimal threshold using criteria that require BOTH classes to perform well.
    
    Hard floor: any threshold where no-bloom F1 < 0.40 OR bloom F1 < 0.40 OR
    specificity < 0.30 is disqualified. This prevents the low-threshold trap
    where the model achieves high bloom F1 by predicting bloom almost always.
    
    Criteria:
    1. Max Macro F1 -- average of bloom F1 and no-bloom F1 (requires both high)
    2. Max MCC -- accounts for all four CM quadrants simultaneously
    3. Max Balanced Accuracy -- average of sensitivity (TPR) and specificity (TNR)
    4. Max Composite -- macro_f1 * MCC * min(bloom_f1, no_bloom_f1) * (1 + delta)
       The min() term is critical: it forces the optimizer to care about
       whichever class is WORSE, preventing the one-sided exploit.
    """
    df = df.copy()

    # Hard floor: reject thresholds where either class collapses
    df['passes_floor'] = (
        (df['ess_bloom_f1'] >= 0.40) &
        (df['ess_no_bloom_f1'] >= 0.40) &
        (df['ess_specificity'] >= 0.30)
    )
    valid = df[df['passes_floor']].copy()

    if len(valid) == 0:
        print("  WARNING: No thresholds pass the hard floor. Using all thresholds.")
        valid = df.copy()

    n_rejected = len(df) - len(valid)
    if n_rejected > 0:
        print(f"  {n_rejected} thresholds rejected by hard floor "
              f"(bloom F1 < 0.40 or no-bloom F1 < 0.40 or specificity < 0.30)")

    # Composite score:
    # - Uses MACRO F1 (not bloom-only F1)
    # - MCC shifted to [0,1]
    # - min(bloom_f1, no_bloom_f1) as a "weakest link" penalty
    # - Bonus for ESS advantage over baseline
    mcc_shifted = (valid['ess_mcc'] + 1) / 2
    weakest_class = np.minimum(valid['ess_bloom_f1'], valid['ess_no_bloom_f1'])
    valid['composite'] = (
        valid['ess_macro_f1']
        * mcc_shifted
        * weakest_class
        * (1 + np.clip(valid['delta_macro_f1'], 0, None))
    )

    criteria = {
        'Max Macro F1':          valid.loc[valid['ess_macro_f1'].idxmax()],
        'Max MCC':               valid.loc[valid['ess_mcc'].idxmax()],
        'Max Balanced Accuracy': valid.loc[valid['ess_balanced_acc'].idxmax()],
        'Max Composite':         valid.loc[valid['composite'].idxmax()],
    }

    # Copy composite back to full df for plotting
    df['composite'] = 0.0
    df.loc[valid.index, 'composite'] = valid['composite']

    return criteria, df


# =============================================================================
# 5. PLOT RESULTS
# =============================================================================

def plot_threshold_sweep(df, criteria, filename="threshold_sweep.png"):
    """Three-panel plot: per-class F1, aggregate metrics, and class balance."""

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11),
                                          height_ratios=[2.5, 2.5, 1],
                                          sharex=True, gridspec_kw={'hspace': 0.08})

    thr = df['threshold']

    # --- Panel 1: Per-class F1 scores (the key diagnostic) ---
    ax1.plot(thr, df['ess_bloom_f1'], color="#d4442a", lw=2, label="Bloom F1 (positive class)")
    ax1.plot(thr, df['ess_no_bloom_f1'], color="#2277aa", lw=2, label="No-Bloom F1 (negative class)")
    ax1.plot(thr, df['ess_macro_f1'], color="#1b7340", lw=2.5, label="Macro F1 (average of both)",
             zorder=5)

    # Shade the danger zone where no-bloom F1 drops below floor
    ax1.axhline(0.40, color="#999999", linestyle=":", lw=1, alpha=0.5)
    ax1.text(thr.iloc[0], 0.42, "Hard floor (0.40)", fontsize=8, color="#999999")

    # Mark optimal
    markers = {
        'Max Macro F1': ('o', '#1b7340'),
        'Max MCC': ('s', '#8b5cf6'),
        'Max Balanced Accuracy': ('^', '#e6a817'),
        'Max Composite': ('*', '#d4442a'),
    }
    for label, row in criteria.items():
        marker, color = markers[label]
        ax1.axvline(row['threshold'], color=color, alpha=0.25, lw=1, linestyle='--')
        ax1.scatter(row['threshold'], row['ess_macro_f1'], marker=marker, s=120,
                    color=color, zorder=10, edgecolors='black', linewidths=0.5,
                    label=f"{label}: {row['threshold']:.2f} RFU")

    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("Per-Class F1 vs. Bloom Threshold (both classes must be high)",
                   fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])

    # --- Panel 2: Aggregate metrics ---
    ax2.plot(thr, df['ess_mcc'], color="#8b5cf6", lw=2, label="MCC")
    ax2.plot(thr, df['ess_balanced_acc'], color="#e6a817", lw=1.8, linestyle="--",
             label="Balanced Accuracy")
    ax2.plot(thr, df['ess_prauc'], color="#1b7340", lw=1.5, linestyle="-.",
             label="PR-AUC")
    ax2.plot(thr, df['ess_specificity'], color="#2277aa", lw=1.5, alpha=0.7,
             label="Specificity (TNR)")
    ax2.plot(thr, df['ess_bloom_rec'], color="#d4442a", lw=1.5, alpha=0.7,
             label="Sensitivity (TPR)")

    # Baseline MCC for reference
    ax2.plot(thr, df['base_mcc'], color="#bbbbbb", lw=1.5, linestyle=":", label="Baseline MCC")

    for label, row in criteria.items():
        marker, color = markers[label]
        ax2.axvline(row['threshold'], color=color, alpha=0.25, lw=1, linestyle='--')

    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Aggregate Metrics vs. Bloom Threshold",
                   fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.15, 1.05])

    # --- Panel 3: Class balance ---
    ax3.fill_between(thr, df['pos_window_pct'], alpha=0.3, color="#d4442a",
                     label="Bloom windows %")
    ax3.plot(thr, df['pos_window_pct'], color="#d4442a", lw=1.5)
    ax3.set_xlabel("Phycocyanin Bloom Threshold (RFU)", fontsize=12)
    ax3.set_ylabel("Bloom %", fontsize=11)
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc="upper right")

    ax3.axvline(1.3, color="black", lw=1.2, linestyle="--", alpha=0.5)
    ax3.annotate("Current (1.3)", xy=(1.3, 85), fontsize=8, ha="center",
                 color="black", alpha=0.7)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {filename}")
    plt.close(fig)


# =============================================================================
# 6. CONFUSION MATRIX DETAIL AT OPTIMAL
# =============================================================================

def print_optimal_detail(criteria):
    """Print detailed breakdown for each optimal threshold."""
    print(f"\n{'='*70}")
    print(f"  OPTIMAL THRESHOLD ANALYSIS")
    print(f"{'='*70}")

    for label, row in criteria.items():
        print(f"\n  --- {label} ---")
        print(f"  Threshold:        {row['threshold']:.3f} RFU")
        print(f"  Bloom days:       {row['n_bloom_days']:.0f} ({row['bloom_pct']:.1f}%)")
        print(f"  Windows:          {row['n_windows']:.0f} "
              f"({row['n_pos_windows']:.0f}+ / {row['n_neg_windows']:.0f}-)")
        print(f"")
        print(f"  ESS Model -- Bloom Class:")
        print(f"    Bloom F1:       {row['ess_bloom_f1']:.4f}")
        print(f"    Bloom Prec:     {row['ess_bloom_prec']:.4f}")
        print(f"    Sensitivity:    {row['ess_bloom_rec']:.4f}  (TPR: catches real blooms)")
        print(f"")
        print(f"  ESS Model -- No-Bloom Class:")
        print(f"    No-Bloom F1:    {row['ess_no_bloom_f1']:.4f}")
        print(f"    No-Bloom Prec:  {row['ess_no_bloom_prec']:.4f}  (NPV: trust a 'safe' call)")
        print(f"    Specificity:    {row['ess_specificity']:.4f}  (TNR: catches real non-blooms)")
        print(f"")
        print(f"  ESS Model -- Aggregate:")
        print(f"    Macro F1:       {row['ess_macro_f1']:.4f}  (avg of bloom + no-bloom F1)")
        print(f"    MCC:            {row['ess_mcc']:.4f}")
        print(f"    Balanced Acc:   {row['ess_balanced_acc']:.4f}")
        print(f"    PR-AUC:         {row['ess_prauc']:.4f}")
        print(f"    Confusion:  TN={row['ess_tn']:.0f}  FP={row['ess_fp']:.0f}")
        print(f"                FN={row['ess_fn']:.0f}  TP={row['ess_tp']:.0f}")
        print(f"")
        print(f"  Baseline Model:")
        print(f"    Macro F1:       {row['base_macro_f1']:.4f}")
        print(f"    MCC:            {row['base_mcc']:.4f}")
        print(f"    Specificity:    {row['base_specificity']:.4f}")
        print(f"    Confusion:  TN={row['base_tn']:.0f}  FP={row['base_fp']:.0f}")
        print(f"                FN={row['base_fn']:.0f}  TP={row['base_tp']:.0f}")
        print(f"")
        print(f"  ESS Advantage:")
        print(f"    Delta Macro F1: {row['delta_macro_f1']:+.4f}")
        print(f"    Delta PR-AUC:   {row['delta_prauc']:+.4f}")
        print(f"    Delta MCC:      {row['delta_mcc']:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("="*70)
    print("  ESS Threshold Optimizer")
    print("  Sweeping phycocyanin bloom thresholds for Lake Erie")
    print("="*70)

    # Suppress segment/window print noise during sweep
    import ess_v3 as erie

    # Step 1: Prepare data (one-time)
    daily = prepare_daily_data()

    # Step 2: Describe PC distribution
    pc = daily['pc_rfu'].dropna()
    print(f"\n  Phycocyanin RFU distribution:")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"    P{int(q*100):02d}: {pc.quantile(q):.3f}")

    # Step 3: Sweep
    # Redirect stdout during sweep to suppress per-segment prints
    import io
    import contextlib

    # Run sweep with suppressed inner prints
    original_identify = erie.identify_continuous_segments
    original_build = erie.build_sliding_windows_multi

    def quiet_identify(daily_df, max_gap_days=5):
        with contextlib.redirect_stdout(io.StringIO()):
            return original_identify(daily_df, max_gap_days)

    def quiet_build(segments, input_days=20, target_days=5, step=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return original_build(segments, input_days, target_days, step)

    erie.identify_continuous_segments = quiet_identify
    erie.build_sliding_windows_multi = quiet_build

    results_df = sweep_thresholds(daily, erie)

    # Restore
    erie.identify_continuous_segments = original_identify
    erie.build_sliding_windows_multi = original_build

    if len(results_df) == 0:
        print("\nERROR: No valid thresholds found. Check data.")
        sys.exit(1)

    # Step 4: Find optimal
    criteria, results_df = find_optimal(results_df)

    # Step 5: Print detail
    print_optimal_detail(criteria)

    # Step 6: Plot
    print("\nGenerating threshold sweep plot...")
    plot_threshold_sweep(results_df, criteria)

    # Step 7: Save CSV
    results_df.to_csv("threshold_results.csv", index=False, float_format="%.4f")
    print(f"  Saved: threshold_results.csv")

    # Step 8: Final recommendation
    composite_best = criteria['Max Composite']
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Composite-optimal threshold: {composite_best['threshold']:.3f} RFU")
    print(f"  This threshold maximizes the product of macro F1, MCC, and the")
    print(f"  weakest per-class F1 (preventing one class from collapsing),")
    print(f"  with a bonus when ESS outperforms baseline.")
    print(f"")
    print(f"  Current threshold (1.3 RFU) vs recommended ({composite_best['threshold']:.3f} RFU):")

    # Find current 1.3 result if it exists
    current = results_df[results_df['threshold'].between(1.25, 1.35)]
    if len(current) > 0:
        cur = current.iloc[0]
        print(f"    {'Metric':<20} {'Current (1.3)':>14} {'Recommended':>14} {'Delta':>10}")
        print(f"    {'-'*20} {'-'*14} {'-'*14} {'-'*10}")
        for metric, label in [('ess_macro_f1', 'Macro F1'),
                               ('ess_bloom_f1', 'Bloom F1'),
                               ('ess_no_bloom_f1', 'No-Bloom F1'),
                               ('ess_bloom_rec', 'Sensitivity (TPR)'),
                               ('ess_specificity', 'Specificity (TNR)'),
                               ('ess_mcc', 'MCC'),
                               ('ess_prauc', 'PR-AUC')]:
            c_val = cur[metric]
            r_val = composite_best[metric]
            delta = r_val - c_val
            print(f"    {label:<20} {c_val:>14.4f} {r_val:>14.4f} {delta:>+10.4f}")
    else:
        print(f"    (1.3 RFU was not in the evaluated range)")

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}")