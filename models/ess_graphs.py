"""
ESS Visualization Suite
========================
Generates four ISEF figures for both Lake Erie and Columbia River pipelines.

USAGE:
  1. Place this file in the same directory as ess_v3.py and ess_v4_columbia.py
  2. Run:  python ess_visualizations.py
  3. Figures saved to current directory as PNG files

The script re-runs both pipelines internally to capture intermediate data
needed for plotting (daily dataframes, segments, windowed features, labels).
You don't need to input anything -- it reads the same data files your
pipelines already use.

If you want to run just one site, edit the __main__ block at the bottom.

Figures produced:
  fig1_precision_recall.png     -- PR curves for ESS vs baseline (Lake Erie)
  fig2_feature_importance.png   -- Top logistic regression coefficients (Lake Erie)
  fig3_time_series.png          -- Bloom probability vs phycocyanin over time (Lake Erie)
  fig4_baseline_comparison.png  -- ESS vs baseline metrics, Lake Erie + Columbia side by side
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FIGURE 1: Precision-Recall Curve
# =============================================================================

def plot_precision_recall(X_ess, X_baseline, y, site_label="Lake Erie",
                          filename="fig1_precision_recall.png"):
    """PR curve comparing ESS vs baseline models."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ess_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    # Clean inputs
    X_ess_clean = np.nan_to_num(X_ess.values if hasattr(X_ess, 'values') else X_ess,
                                 nan=0.0, posinf=10, neginf=-10)
    X_base_clean = np.nan_to_num(X_baseline.values if hasattr(X_baseline, 'values') else X_baseline,
                                  nan=0.0, posinf=10, neginf=-10)

    y_prob_ess = cross_val_predict(ess_pipe, X_ess_clean, y, cv=skf, method="predict_proba")[:, 1]
    y_prob_base = cross_val_predict(base_pipe, X_base_clean, y, cv=skf, method="predict_proba")[:, 1]

    prec_ess, rec_ess, _ = precision_recall_curve(y, y_prob_ess)
    ap_ess = average_precision_score(y, y_prob_ess)

    prec_base, rec_base, _ = precision_recall_curve(y, y_prob_base)
    ap_base = average_precision_score(y, y_prob_base)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(rec_ess, prec_ess, color="#1b7340", lw=2.2,
            label=f"ESS model (AP = {ap_ess:.3f})")
    ax.plot(rec_base, prec_base, color="#888888", lw=1.8, linestyle="--",
            label=f"Baseline (AP = {ap_base:.3f})")

    prevalence = y.mean()
    ax.axhline(prevalence, color="#cccccc", linestyle=":", lw=1,
               label=f"No skill ({prevalence:.2f})")

    ax.set_xlabel("Recall", fontsize=15)
    ax.set_ylabel("Precision", fontsize=15)
    ax.set_title(f"Precision-Recall Curve -- {site_label}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1.01])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)

    return y_prob_ess, y_prob_base


# =============================================================================
# FIGURE 2: Feature Importance
# =============================================================================

def plot_feature_importance(X_ess, y, feature_names, top_n=15,
                             site_label="Lake Erie",
                             filename="fig2_feature_importance.png"):
    """Bar chart of top logistic regression coefficients."""
    scaler = StandardScaler()
    X_clean = np.nan_to_num(
        scaler.fit_transform(X_ess.values if hasattr(X_ess, 'values') else X_ess),
        nan=0.0, posinf=10, neginf=-10
    )

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_clean, y)
    coefs = model.coef_[0]

    top_n = min(top_n, len(feature_names))
    idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    sorted_names = [feature_names[i] for i in idx]
    sorted_coefs = coefs[idx]

    colors = ["#1b7340" if c > 0 else "#555555" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(8, 0.45 * top_n + 1.5))
    ax.barh(range(top_n), sorted_coefs[::-1], color=colors[::-1],
            edgecolor="white", height=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel("Logistic Regression Coefficient (standardized)", fontsize=11)
    ax.set_title(f"Top {top_n} ESS Feature Importances -- {site_label}",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


# =============================================================================
# FIGURE 3: Time Series Overlay
# =============================================================================

def plot_time_series(segments, X_ess, y, site_label="Lake Erie",
                     input_days=20, target_days=5,
                     filename="fig3_time_series.png"):
    """
    Bloom probability vs observed phycocyanin over time.
    Aligns windowed predictions back to calendar dates using segment structure.
    """
    # Get cross-val bloom probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    X_clean = np.nan_to_num(X_ess.values if hasattr(X_ess, 'values') else X_ess,
                             nan=0.0, posinf=10, neginf=-10)
    bloom_probs = cross_val_predict(pipe, X_clean, y, cv=skf, method="predict_proba")[:, 1]

    # Reconstruct date + phycocyanin alignment from segments
    # Each window's "prediction date" = last day of the input window
    dates_all = []
    pc_all = []

    for seg in segments:
        total_len = len(seg)
        if total_len < input_days + target_days:
            continue
        seg_dates = pd.to_datetime(seg['date']).values
        seg_pc = seg['pc_rfu'].values

        for i in range(0, total_len - input_days - target_days + 1, 1):
            prediction_date = seg_dates[i + input_days - 1]
            target_pc = seg_pc[i + input_days:i + input_days + target_days].mean()
            dates_all.append(prediction_date)
            pc_all.append(target_pc)

    dates_all = np.array(dates_all)
    pc_all = np.array(pc_all)

    if len(dates_all) != len(bloom_probs):
        print(f"  WARNING: date alignment mismatch ({len(dates_all)} dates vs "
              f"{len(bloom_probs)} predictions)")
        min_len = min(len(dates_all), len(bloom_probs))
        dates_all = dates_all[:min_len]
        pc_all = pc_all[:min_len]
        bloom_probs = bloom_probs[:min_len]

    sort_idx = np.argsort(dates_all)
    dates_all = dates_all[sort_idx]
    pc_all = pc_all[sort_idx]
    bloom_probs = bloom_probs[sort_idx]

    dates_pd = pd.to_datetime(dates_all)
    years = sorted(dates_pd.year.unique())

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Phycocyanin on left axis
    ax1.plot(dates_pd, pc_all, color="#2277aa", lw=1.2, alpha=0.7,
             label="Phycocyanin (RFU, 5-day target mean)")
    ax1.set_ylabel("Phycocyanin (RFU)", color="#2277aa", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#2277aa")
    ax1.axhline(1.3, color="#2277aa", linestyle=":", lw=1, alpha=0.4,
                label="Bloom threshold (1.3 RFU)")

    # Bloom probability on right axis
    ax2 = ax1.twinx()
    ax2.plot(dates_pd, bloom_probs, color="#d4442a", lw=1.5, alpha=0.8,
             label="ESS Bloom Probability")
    ax2.set_ylabel("Bloom Probability", color="#d4442a", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d4442a")
    ax2.set_ylim([-0.05, 1.05])
    ax2.axhline(0.5, color="#d4442a", linestyle=":", lw=1, alpha=0.4)

    # Year separation lines
    for yr in years[1:]:
        yr_start = pd.Timestamp(f"{yr}-01-01")
        ax1.axvline(yr_start, color="#999999", linestyle="--", lw=0.8, alpha=0.5)

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title(f"ESS Bloom Probability vs. Observed Phycocyanin -- {site_label}",
                   fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


# =============================================================================
# FIGURE 4: Cross-Site Baseline vs ESS Comparison
# =============================================================================

def compute_metrics_cv(X, y, n_splits=5):
    """Cross-val metrics. Falls back to full-data fit if too few positives."""
    n_pos = int(y.sum())
    actual_splits = min(n_splits, n_pos, len(y) - n_pos)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    X_clean = np.nan_to_num(X.values if hasattr(X, 'values') else X,
                             nan=0.0, posinf=10, neginf=-10)

    if actual_splits >= 2:
        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
        y_prob = cross_val_predict(pipe, X_clean, y, cv=skf, method="predict_proba")[:, 1]
        y_pred = cross_val_predict(pipe, X_clean, y, cv=skf)
        cv_label = f"{actual_splits}-fold CV"
    else:
        # Too few samples for CV -- fit on full data (label accordingly)
        pipe.fit(X_clean, y)
        y_pred = pipe.predict(X_clean)
        y_prob = pipe.predict_proba(X_clean)[:, 1]
        cv_label = "full-data fit (no CV)"

    metrics = {
        'F1': f1_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'PR-AUC': average_precision_score(y, y_prob),
        'MCC': matthews_corrcoef(y, y_pred),
    }
    return metrics, cv_label


def plot_baseline_comparison(erie_ess, erie_base, erie_y,
                              columbia_ess=None, columbia_base=None, columbia_y=None,
                              filename="fig4_baseline_comparison.png"):
    """Side-by-side ESS vs baseline. Columbia is optional."""

    erie_ess_m, erie_cv = compute_metrics_cv(erie_ess, erie_y)
    erie_base_m, _ = compute_metrics_cv(erie_base, erie_y)
    print(f"  Lake Erie: {erie_cv}")

    has_columbia = (columbia_ess is not None and columbia_y is not None
                    and int(columbia_y.sum()) >= 2)

    if has_columbia:
        col_ess_m, col_cv = compute_metrics_cv(columbia_ess, columbia_y)
        col_base_m, _ = compute_metrics_cv(columbia_base, columbia_y)
        print(f"  Columbia River: {col_cv}")

    metric_names = ['F1', 'Precision', 'Recall', 'PR-AUC', 'MCC']
    x = np.arange(len(metric_names))

    if has_columbia:
        width = 0.18
        fig, ax = plt.subplots(figsize=(11, 5.5))

        erie_ess_vals = [erie_ess_m[m] for m in metric_names]
        erie_base_vals = [erie_base_m[m] for m in metric_names]
        col_ess_vals = [col_ess_m[m] for m in metric_names]
        col_base_vals = [col_base_m[m] for m in metric_names]

        bars1 = ax.bar(x - 1.5*width, erie_ess_vals, width,
                       label="Lake Erie ESS (5 rxn)", color="#1b7340", edgecolor="white")
        bars2 = ax.bar(x - 0.5*width, erie_base_vals, width,
                       label="Lake Erie Baseline", color="#8fbc8f", edgecolor="white")
        bars3 = ax.bar(x + 0.5*width, col_ess_vals, width,
                       label="Columbia River ESS (3 rxn)", color="#2277aa", edgecolor="white")
        bars4 = ax.bar(x + 1.5*width, col_base_vals, width,
                       label="Columbia River Baseline", color="#a8c8e8", edgecolor="white")

        all_bars = [(bars1, "#1b7340"), (bars2, "#3a6e3a"),
                    (bars3, "#2277aa"), (bars4, "#4a7a9a")]
    else:
        width = 0.32
        fig, ax = plt.subplots(figsize=(8, 5.5))

        erie_ess_vals = [erie_ess_m[m] for m in metric_names]
        erie_base_vals = [erie_base_m[m] for m in metric_names]

        bars1 = ax.bar(x - width/2, erie_ess_vals, width,
                       label="ESS Model", color="#1b7340", edgecolor="white")
        bars2 = ax.bar(x + width/2, erie_base_vals, width,
                       label="Baseline", color="#aaaaaa", edgecolor="white")

        all_bars = [(bars1, "#1b7340"), (bars2, "#555555")]

    for bars, color in all_bars:
        for bar in bars:
            val = bar.get_height()
            if val >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    title = ("ESS vs Baseline: Lake Erie (5 rxn) and Columbia River (3 rxn)"
             if has_columbia else "ESS Model vs Baseline (No Q/K Features)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0, 1.18])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


# =============================================================================
# PIPELINE RUNNERS (re-run pipelines to capture intermediate data)
# =============================================================================

def run_erie_pipeline():
    """Run Lake Erie pipeline, return (X_baseline, X_ess, y, segments)."""
    import ess_v3 as erie

    file_pairs = [
        ('data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary_phosphate.csv'),
    ]

    print("\n" + "="*60)
    print("  Loading Lake Erie data for visualization...")
    print("="*60)

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

    df = erie.compute_derived_variables(df)
    df = erie.compute_qk_ratios(df)
    df = erie.compute_convergence_features(df)
    df = erie.compute_ecological_ratios(df)
    df = erie.label_blooms(df)

    daily = erie.aggregate_to_daily(df)
    segments = erie.identify_continuous_segments(daily)
    X_baseline, X_ess, y = erie.build_sliding_windows_multi(segments)

    print(f"  Erie: {len(y)} windows, {int(y.sum())} positive")
    return X_baseline, X_ess, y, segments


def run_columbia_pipeline():
    """Run Columbia River pipeline, return (X_baseline, X_ess, y, segments)."""
    import ess_v4_columbia as columbia

    # 2023 only (primary bloom dataset) — pulled from the canonical DATASETS dict
    xlsx_paths = columbia.DATASETS['2023']

    print("\n" + "="*60)
    print("  Loading Columbia River data for visualization...")
    print("="*60)

    df = columbia.load_columbia_data(xlsx_paths)
    required = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'cond_uscm']
    df = df.dropna(subset=required).reset_index(drop=True)

    df = columbia.compute_derived_variables(df)
    df = columbia.compute_qk_ratios(df)
    df = columbia.compute_convergence_features(df)
    df = columbia.label_blooms(df)

    daily = columbia.aggregate_to_daily(df)
    segments = columbia.identify_continuous_segments(daily)
    X_baseline, X_ess, y = columbia.build_sliding_windows(segments)

    print(f"  Columbia: {len(y)} windows, {int(y.sum())} positive")
    return X_baseline, X_ess, y, segments


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Usage:
    #   python ess_graphs.py          → both sites (all 9 figures)
    #   python ess_graphs.py erie     → Lake Erie only
    #   python ess_graphs.py columbia → Columbia River only

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'
    run_erie = mode in ('erie', 'both')
    run_columbia = mode in ('columbia', 'both')

    print("="*60)
    print("  ESS Visualization Suite")
    print("="*60)

    # ---- Run pipelines ----
    erie_base = erie_ess = erie_y = erie_segs = None
    col_base = col_ess = col_y = col_segs = None

    if run_erie:
        erie_base, erie_ess, erie_y, erie_segs = run_erie_pipeline()

    if run_columbia:
        try:
            col_base, col_ess, col_y, col_segs = run_columbia_pipeline()
        except Exception as e:
            print(f"\n  Columbia River pipeline failed: {e}")
            run_columbia = False

    # ---- Generate figures ----
    print("\n" + "="*60)
    print("  Generating figures...")
    print("="*60)

    fig_n = 1

    if run_erie:
        erie_feat_names = list(erie_ess.columns) if hasattr(erie_ess, 'columns') else []

        print(f"\n[{fig_n}] Precision-Recall Curve (Lake Erie)...")
        plot_precision_recall(erie_ess, erie_base, erie_y,
                              site_label="Lake Erie (5 reactions, 2016-2018)",
                              filename="fig1_erie_precision_recall.png")
        fig_n += 1

        print(f"\n[{fig_n}] Feature Importance (Lake Erie)...")
        plot_feature_importance(erie_ess, erie_y, erie_feat_names,
                                 top_n=15, site_label="Lake Erie",
                                 filename="fig2_erie_feature_importance.png")
        fig_n += 1

        print(f"\n[{fig_n}] Time Series Overlay (Lake Erie)...")
        plot_time_series(erie_segs, erie_ess, erie_y,
                         site_label="Lake Erie (2016-2018)",
                         filename="fig3_erie_time_series.png")
        fig_n += 1

    if run_columbia:
        col_feat_names = list(col_ess.columns) if hasattr(col_ess, 'columns') else []

        print(f"\n[{fig_n}] Precision-Recall Curve (Columbia River)...")
        plot_precision_recall(col_ess, col_base, col_y,
                              site_label="Columbia River (3 reactions, 2023 Johnson Island)",
                              filename="fig1_columbia_precision_recall.png")
        fig_n += 1

        print(f"\n[{fig_n}] Feature Importance (Columbia River)...")
        plot_feature_importance(col_ess, col_y, col_feat_names,
                                 top_n=len(col_feat_names), site_label="Columbia River",
                                 filename="fig2_columbia_feature_importance.png")
        fig_n += 1

        print(f"\n[{fig_n}] Time Series Overlay (Columbia River)...")
        plot_time_series(col_segs, col_ess, col_y,
                         site_label="Columbia River (2023 Johnson Island)",
                         filename="fig3_columbia_time_series.png")
        fig_n += 1

    if run_erie and run_columbia:
        print(f"\n[{fig_n}] Baseline vs ESS Comparison (both sites)...")
        plot_baseline_comparison(
            erie_ess, erie_base, erie_y,
            columbia_ess=col_ess, columbia_base=col_base, columbia_y=col_y,
            filename="fig4_baseline_comparison.png"
        )
    elif run_erie:
        print(f"\n[{fig_n}] Baseline vs ESS Comparison (Lake Erie only)...")
        plot_baseline_comparison(erie_ess, erie_base, erie_y,
                                  filename="fig4_baseline_comparison.png")
    elif run_columbia:
        print(f"\n[{fig_n}] Baseline vs ESS Comparison (Columbia River only)...")
        plot_baseline_comparison(col_ess, col_base, col_y,
                                  filename="fig4_baseline_comparison.png")

    print("\n" + "="*60)
    print("  All figures saved. Done.")
    print("="*60)