"""
ESS Threshold Sensitivity Analysis
Sweeps phycocyanin bloom threshold from 1.0 to 8.0 RFU in 0.1 increments.
Plots F1, MCC, and PR-AUC for both Baseline and ESS at each threshold.

This is a SENSITIVITY ANALYSIS, not hyperparameter optimization.
It tests how the bloom definition affects model performance, answering:
"Does ESS consistently outperform the baseline regardless of threshold?"

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
"""

import numpy as np
import pandas as pd
import sys
import os

# Import everything from your v3 file
# We'll reuse all chemistry functions and just override the pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    matthews_corrcoef, confusion_matrix,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Copy all chemistry/thermodynamic functions from ess_v3.py
# (These don't change with threshold, so they're identical)
# =============================================================================

def carbonate_K1(T_K):
    log_K1 = (-356.3094 - 0.06091964 * T_K + 21834.37 / T_K
              + 126.8339 * np.log10(T_K) - 1684915.0 / T_K**2)
    return 10**log_K1

def carbonate_K2(T_K):
    log_K2 = (-107.8871 - 0.03252849 * T_K + 5151.79 / T_K
              + 38.92561 * np.log10(T_K) - 563713.9 / T_K**2)
    return 10**log_K2

def henry_KH(T_K):
    ln_KH = (-60.2409 + 93.4517 * (100.0 / T_K)
             + 23.3585 * np.log(T_K / 100.0))
    return np.exp(ln_KH)

def Kw(T_K):
    log_Kw = -4470.99 / T_K + 6.0846 - 0.01706 * T_K
    return 10**log_Kw

def phosphate_Ka1(T_K):
    return 10**(-2.148 + 0.0 * (T_K - 298.15))

def phosphate_Ka2(T_K):
    return 10**(-7.198 + 0.0 * (T_K - 298.15))

def phosphate_Ka3(T_K):
    return 10**(-12.375 + 0.0 * (T_K - 298.15))

def O2_saturation(T_K):
    T_K = np.clip(T_K, 273.16, None)
    ln_DO = (-139.34411 + (1.575701e5 / T_K) - (6.642308e7 / T_K**2)
             + (1.243800e10 / T_K**3) - (8.621949e11 / T_K**4))
    return np.exp(ln_DO)

def compute_derived_variables(df):
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])
    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)
    alk_mol = (5.0 * df['cond_uscm']) * 1e-6
    alpha1 = K1 * H / (H**2 + K1 * H + K1 * K2)
    alpha2 = K1 * K2 / (H**2 + K1 * H + K1 * K2)
    OH = Kw(T_K) / H
    denom = alpha1 + 2 * alpha2
    denom = np.where(denom > 0, denom, 1e-12)
    DIC = (alk_mol - OH + H) / denom
    DIC = np.clip(DIC, 1e-9, None)
    df['hco3_mol'] = DIC * alpha1
    df['co3_mol'] = DIC * alpha2
    df['co2_aq_mol'] = DIC - df['hco3_mol'] - df['co3_mol']
    df['co2_aq_mol'] = np.clip(df['co2_aq_mol'], 1e-12, None)
    Ka1 = phosphate_Ka1(T_K)
    Ka2p = phosphate_Ka2(T_K)
    Ka3 = phosphate_Ka3(T_K)
    P_total = (df['srp_ugl'] * 1e-6) / 30.97
    P_total = np.clip(P_total, 1e-12, None)
    denom_p = (H**3 + Ka1 * H**2 + Ka1 * Ka2p * H + Ka1 * Ka2p * Ka3)
    df['h2po4_mol'] = P_total * (Ka1 * H**2) / denom_p
    df['hpo4_mol'] = P_total * (Ka1 * Ka2p * H) / denom_p
    df['o2_sat_mgl'] = O2_saturation(T_K)
    df['do_mol'] = (df['do_mgl'] / 32.0) * 1e-3
    df['o2_sat_mol'] = (df['o2_sat_mgl'] / 32.0) * 1e-3
    return df

def compute_qk_ratios(df):
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])
    co2 = np.clip(df['co2_aq_mol'], 1e-12, None)
    do = np.clip(df['do_mol'], 1e-12, None)
    Q_01 = (do**6) / (co2**6)
    df['qk_rxn01'] = np.log10(1.0 / Q_01)
    K1 = carbonate_K1(T_K)
    Q_02 = (df['hco3_mol'] * H) / co2
    df['qk_rxn02'] = Q_02 / K1
    Ka2 = phosphate_Ka2(T_K)
    h2po4 = np.clip(df['h2po4_mol'], 1e-15, None)
    Q_03 = (df['hpo4_mol'] * H) / h2po4
    df['qk_rxn03'] = Q_03 / Ka2
    df['qk_rxn05'] = df['srp_ugl'] / 5.0
    hpo4 = np.clip(df['hpo4_mol'], 1e-15, None)
    df['qk_rxn06'] = np.log10(co2 * hpo4) * -1
    return df

def compute_convergence_features(df):
    rxn_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']
    n_rxns = len(rxn_cols)
    p33_01 = df['qk_rxn01'].quantile(0.33)
    p66_01 = df['qk_rxn01'].quantile(0.66)
    df['state_rxn01'] = np.where(df['qk_rxn01'] > p66_01, 1,
                         np.where(df['qk_rxn01'] < p33_01, -1, 0))
    df['state_rxn02'] = np.where(df['qk_rxn02'] > 1.2, 1,
                         np.where(df['qk_rxn02'] < 0.8, -1, 0))
    df['state_rxn03'] = np.where(df['qk_rxn03'] > 1.2, 1,
                         np.where(df['qk_rxn03'] < 0.8, -1, 0))
    df['state_rxn05'] = np.where(df['qk_rxn05'] > 1.0, 1,
                         np.where(df['qk_rxn05'] < 0.5, -1, 0))
    p33_06 = df['qk_rxn06'].quantile(0.33)
    p66_06 = df['qk_rxn06'].quantile(0.66)
    df['state_rxn06'] = np.where(df['qk_rxn06'] > p66_06, 1,
                         np.where(df['qk_rxn06'] < p33_06, -1, 0))
    state_cols = ['state_rxn01', 'state_rxn02', 'state_rxn03',
                  'state_rxn05', 'state_rxn06']
    df['n_favorable'] = (df[state_cols] == 1).sum(axis=1)
    df['frac_favorable'] = df['n_favorable'] / n_rxns
    df['all_favorable'] = (df['n_favorable'] == n_rxns).astype(int)
    df['n_unfavorable'] = (df[state_cols] == -1).sum(axis=1)
    df['convergence_score'] = df[state_cols].sum(axis=1)
    for col in rxn_cols:
        p50 = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        k = 4.0 / max(iqr, 1e-6)
        df[f'_fav_{col}'] = 1.0 / (1.0 + np.exp(-k * (df[col] - p50)))
    fav_cols = [f'_fav_{col}' for col in rxn_cols]
    df['favorability_product'] = df[fav_cols].prod(axis=1)
    df.drop(columns=fav_cols, inplace=True)
    return df

def compute_ecological_ratios(df):
    if 'nitrate_umol' in df.columns:
        N_umol = df['nitrate_umol']
    elif 'nitrate_mgl' in df.columns:
        N_umol = (df['nitrate_mgl'] / 14.007) * 1000
    else:
        N_umol = (df['cond_uscm'] * 0.002) * 1e-3 / 14.007 * 1e6
    P_umol = df['srp_ugl'] / 30.97
    P_umol = np.clip(P_umol, 0.001, None)
    df['np_ratio'] = N_umol / P_umol
    return df

def load_and_combine(file_pairs):
    all_dfs = []
    for data_path, phosphate_path in file_pairs:
        df = pd.read_csv(data_path, skiprows=[1, 2], low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if phosphate_path:
            phos = pd.read_csv(phosphate_path, skiprows=[1, 2])
            phos['timestamp'] = pd.to_datetime(phos['timestamp'], errors='coerce')
            phos = phos.dropna(subset=['timestamp'])
            phos['phosphate'] = pd.to_numeric(phos['phosphate'], errors='coerce')
            phos = phos[(phos['phosphate'] >= 0) & (phos['phosphate'] <= 500)]
            phos['date'] = phos['timestamp'].dt.date
            phos_daily = phos.groupby('date')['phosphate'].mean().reset_index()
            df['date'] = df['timestamp'].dt.date
            df = df.merge(phos_daily, on='date', how='left')
            df['phosphate'] = df['phosphate'].ffill()
            df.drop(columns=['date'], inplace=True)
        df['source_file'] = data_path
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined

def identify_continuous_segments(daily_df, max_gap_days=5):
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(daily_df['date'])
    gaps = dates.diff().dt.days
    break_points = gaps[gaps > max_gap_days].index.tolist()
    segments = []
    start = 0
    for bp in break_points:
        seg = daily_df.iloc[start:bp].copy()
        if len(seg) > 0:
            segments.append(seg)
        start = bp
    seg = daily_df.iloc[start:].copy()
    if len(seg) > 0:
        segments.append(seg)
    return segments

def build_sliding_windows_multi(segments, input_days=20, target_days=5, step=1):
    feature_cols_baseline = ['temp_c', 'do_mgl', 'ph', 'srp_ugl', 'cond_uscm']
    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score',
                 'favorability_product']
    feature_cols_ess = qk_cols + ['np_ratio'] + conv_cols
    X_baseline, X_ess, y = [], [], []
    for seg in segments:
        total_len = len(seg)
        if total_len < input_days + target_days:
            continue
        for i in range(0, total_len - input_days - target_days + 1, step):
            input_window = seg.iloc[i:i + input_days]
            target_window = seg.iloc[i + input_days:i + input_days + target_days]
            base_feats = []
            for col in feature_cols_baseline:
                base_feats.append(input_window[col].mean())
                base_feats.append(input_window[col].std())
            ess_feats = []
            for col in feature_cols_ess:
                ess_feats.append(input_window[col].mean())
                ess_feats.append(input_window[col].std())
            ess_feats.append(input_window['n_favorable'].max())
            ess_feats.append(input_window['all_favorable'].sum())
            af = input_window['all_favorable'].values
            max_consec = 0
            current_run = 0
            for v in af:
                if v == 1:
                    current_run += 1
                    max_consec = max(max_consec, current_run)
                else:
                    current_run = 0
            ess_feats.append(max_consec)
            n_fav = input_window['n_favorable'].values
            if len(n_fav) > 1:
                slope = np.polyfit(np.arange(len(n_fav)), n_fav, 1)[0]
            else:
                slope = 0.0
            ess_feats.append(slope)
            ess_feats.append(input_window['favorability_product'].max())
            label = int(target_window['bloom_label'].max())
            X_baseline.append(base_feats)
            X_ess.append(ess_feats)
            y.append(label)
    base_names = []
    for col in feature_cols_baseline:
        base_names.extend([f'{col}_mean', f'{col}_std'])
    ess_names = []
    for col in feature_cols_ess:
        ess_names.extend([f'{col}_mean', f'{col}_std'])
    ess_names.extend(['max_n_favorable', 'days_all_favorable',
                      'max_consec_all_favorable', 'convergence_trend',
                      'peak_favorability_product'])
    X_baseline = pd.DataFrame(X_baseline, columns=base_names)
    X_ess = pd.DataFrame(X_ess, columns=ess_names)
    y = np.array(y)
    return X_baseline, X_ess, y


# =============================================================================
# QUIET EVALUATION (no printing, just returns metrics)
# =============================================================================

def evaluate_quiet(X, y, n_splits=5):
    """Run stratified k-fold CV silently, return metrics dict."""
    if y.sum() < n_splits or (len(y) - y.sum()) < n_splits:
        return {'f1': 0.0, 'prauc': 0.0, 'mcc': 0.0,
                'roc_auc': 0.0,
                'bloom_precision': 0.0, 'bloom_recall': 0.0,
                'nobloom_precision': 0.0, 'nobloom_recall': 0.0}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    f1_scores, prauc_scores, mcc_scores, rocauc_scores = [], [], [], []
    bloom_prec_scores, bloom_rec_scores = [], []
    nobloom_prec_scores, nobloom_rec_scores = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=10, neginf=-10)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=10, neginf=-10)

        model = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='lbfgs', random_state=42
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))
        bloom_prec_scores.append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        bloom_rec_scores.append(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        nobloom_prec_scores.append(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
        nobloom_rec_scores.append(recall_score(y_test, y_pred, pos_label=0, zero_division=0))

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        prauc_scores.append(auc(recall, precision))
        rocauc_scores.append(roc_auc_score(y_test, y_prob))

    return {
        'f1': np.mean(f1_scores),
        'prauc': np.mean(prauc_scores),
        'mcc': np.mean(mcc_scores),
        'roc_auc': np.mean(rocauc_scores),
        'bloom_precision': np.mean(bloom_prec_scores),
        'bloom_recall': np.mean(bloom_rec_scores),
        'nobloom_precision': np.mean(nobloom_prec_scores),
        'nobloom_recall': np.mean(nobloom_rec_scores),
    }


def evaluate_with_roc_curve(X, y, n_splits=5):
    """Run stratified k-fold CV, return mean ROC curve (TPR vs FPR) across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=10, neginf=-10)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=10, neginf=-10)

        model = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='lbfgs', random_state=42
        )
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, np.std(tprs, axis=0), np.mean(aucs)


# =============================================================================
# MAIN: THRESHOLD SWEEP
# =============================================================================

def run_threshold_sweep(file_pairs, threshold_min=0.0, threshold_max=6.0, step=0.1):
    """
    1. Load data and compute all chemistry ONCE
    2. For each threshold, relabel blooms, re-aggregate, re-window, evaluate
    3. Collect results and save to CSV
    """

    # =========================================================
    # STEP 1: Load and compute chemistry (threshold-independent)
    # =========================================================
    print("Loading and combining datasets...")
    df = load_and_combine(file_pairs)

    col_map = {
        'water_temperature': 'temp_c',
        'organic_dissolved_oxygen': 'do_mgl',
        'pH': 'ph',
        'phosphate': 'srp_ugl',
        'phycocyanin': 'pc_rfu',
        'chlorophylla': 'chla_rfu',
        'specific_conductivity': 'cond_uscm',
        'NO3M_suna': 'nitrate_umol',
    }
    df = df.rename(columns=col_map)

    numeric_cols = ['temp_c', 'do_mgl', 'ph', 'srp_ugl',
                    'pc_rfu', 'chla_rfu', 'cond_uscm', 'nitrate_umol']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'cond_uscm' in df.columns:
        df['cond_uscm'] = df['cond_uscm'] * 1000

    if 'srp_ugl' in df.columns:
        median_srp = df['srp_ugl'].median()
        if pd.isna(median_srp):
            median_srp = 5.0
        df['srp_ugl'] = df['srp_ugl'].fillna(median_srp)
    else:
        df['srp_ugl'] = 5.0

    required = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'chla_rfu', 'cond_uscm']
    df = df.dropna(subset=required).reset_index(drop=True)

    print(f"  Total rows after cleaning: {len(df)}")

    print("Computing chemistry (one time)...")
    df = compute_derived_variables(df)
    df = compute_qk_ratios(df)
    df = compute_convergence_features(df)
    df = compute_ecological_ratios(df)

    print("Chemistry complete. Starting threshold sweep...\n")

    # =========================================================
    # STEP 2: Sweep thresholds
    # =========================================================
    thresholds = np.arange(threshold_min, threshold_max + step/2, step)
    thresholds = np.round(thresholds, 1)

    results = []

    for i, thresh in enumerate(thresholds):
        # Relabel blooms at this threshold
        df_copy = df.copy()
        df_copy['bloom_label'] = (df_copy['pc_rfu'] >= thresh).astype(int)

        # Aggregate to daily
        df_copy['date'] = df_copy['timestamp'].dt.date
        exclude_cols = ['timestamp', 'date', 'source_file']
        agg_cols = [c for c in df_copy.columns
                    if c not in exclude_cols
                    and df_copy[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        agg_dict = {col: 'mean' for col in agg_cols}
        agg_dict['bloom_label'] = 'max'
        daily = df_copy.groupby('date').agg(agg_dict).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date').reset_index(drop=True)

        # Count blooms
        n_bloom_days = int(daily['bloom_label'].sum())
        n_total_days = len(daily)

        # Build windows
        segments = identify_continuous_segments(daily)
        X_base, X_ess, y = build_sliding_windows_multi(segments)

        n_pos = int(y.sum())
        n_total = len(y)

        # Evaluate (skip if too few positives)
        if n_pos < 5 or (n_total - n_pos) < 5:
            row = {
                'threshold': thresh,
                'bloom_days': n_bloom_days,
                'pos_windows': n_pos,
                'total_windows': n_total,
                'baseline_f1': 0.0, 'baseline_prauc': 0.0, 'baseline_mcc': 0.0,
                'baseline_roc_auc': 0.0,
                'baseline_bloom_precision': 0.0, 'baseline_bloom_recall': 0.0,
                'baseline_nobloom_precision': 0.0, 'baseline_nobloom_recall': 0.0,
                'ess_f1': 0.0, 'ess_prauc': 0.0, 'ess_mcc': 0.0,
                'ess_roc_auc': 0.0,
                'ess_bloom_precision': 0.0, 'ess_bloom_recall': 0.0,
                'ess_nobloom_precision': 0.0, 'ess_nobloom_recall': 0.0,
                'delta_f1': 0.0,
            }
            status = "SKIPPED (too few positives)"
        else:
            res_base = evaluate_quiet(X_base, y)
            res_ess = evaluate_quiet(X_ess, y)

            row = {
                'threshold': thresh,
                'bloom_days': n_bloom_days,
                'pos_windows': n_pos,
                'total_windows': n_total,
                'baseline_f1': res_base['f1'],
                'baseline_prauc': res_base['prauc'],
                'baseline_mcc': res_base['mcc'],
                'baseline_roc_auc': res_base['roc_auc'],
                'baseline_bloom_precision': res_base['bloom_precision'],
                'baseline_bloom_recall': res_base['bloom_recall'],
                'baseline_nobloom_precision': res_base['nobloom_precision'],
                'baseline_nobloom_recall': res_base['nobloom_recall'],
                'ess_f1': res_ess['f1'],
                'ess_prauc': res_ess['prauc'],
                'ess_mcc': res_ess['mcc'],
                'ess_roc_auc': res_ess['roc_auc'],
                'ess_bloom_precision': res_ess['bloom_precision'],
                'ess_bloom_recall': res_ess['bloom_recall'],
                'ess_nobloom_precision': res_ess['nobloom_precision'],
                'ess_nobloom_recall': res_ess['nobloom_recall'],
                'delta_f1': res_ess['f1'] - res_base['f1'],
            }
            status = f"Base F1={res_base['f1']:.3f}  ESS F1={res_ess['f1']:.3f}  Delta={row['delta_f1']:+.3f}"

        results.append(row)

        pct = 100 * (i + 1) / len(thresholds)
        print(f"  [{pct:5.1f}%] Threshold={thresh:.1f} RFU | "
              f"Blooms={n_bloom_days:3d} days, {n_pos:3d} windows | {status}")

    # =========================================================
    # STEP 3: Save results
    # =========================================================
    results_df = pd.DataFrame(results)
    output_csv = 'threshold_sensitivity_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # =========================================================
    # STEP 4: Print summary
    # =========================================================
    valid = results_df[results_df['ess_f1'] > 0]
    if len(valid) > 0:
        best_ess = valid.loc[valid['ess_f1'].idxmax()]
        best_delta = valid.loc[valid['delta_f1'].idxmax()]
        ess_wins = (valid['delta_f1'] > 0).sum()

        print(f"\n{'='*60}")
        print(f"  THRESHOLD SENSITIVITY SUMMARY")
        print(f"{'='*60}")
        print(f"  Thresholds tested: {len(thresholds)}")
        print(f"  Valid (enough positives): {len(valid)}")
        print(f"  ESS wins on F1: {ess_wins}/{len(valid)} ({100*ess_wins/len(valid):.0f}%)")
        print(f"\n  Best ESS F1: {best_ess['ess_f1']:.4f} at {best_ess['threshold']:.1f} RFU")
        print(f"  Best ESS advantage: +{best_delta['delta_f1']:.4f} at {best_delta['threshold']:.1f} RFU")
        print(f"\n  WHO Alert Level 1 (1.8 RFU):")
        row_18 = results_df[np.isclose(results_df['threshold'], 1.8)]
        if len(row_18) > 0:
            r = row_18.iloc[0]
            print(f"    Baseline F1={r['baseline_f1']:.4f}  ESS F1={r['ess_f1']:.4f}  Delta={r['delta_f1']:+.4f}")

    # =========================================================
    # STEP 5: Compute ROC curves at best ESS threshold
    # =========================================================
    roc_data = {}
    valid_rows = results_df[results_df['ess_f1'] > 0]
    if len(valid_rows) > 0:
        roc_thresh = valid_rows.loc[valid_rows['ess_f1'].idxmax(), 'threshold']
        print(f"\nComputing ROC curves at best threshold ({roc_thresh:.1f} RFU)...")
        df_roc = df.copy()
        df_roc['bloom_label'] = (df_roc['pc_rfu'] >= roc_thresh).astype(int)
        df_roc['date'] = df_roc['timestamp'].dt.date
        exclude_cols = ['timestamp', 'date', 'source_file']
        agg_cols = [c for c in df_roc.columns
                    if c not in exclude_cols
                    and df_roc[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        agg_dict = {col: 'mean' for col in agg_cols}
        agg_dict['bloom_label'] = 'max'
        daily_roc = df_roc.groupby('date').agg(agg_dict).reset_index()
        daily_roc['date'] = pd.to_datetime(daily_roc['date'])
        daily_roc = daily_roc.sort_values('date').reset_index(drop=True)
        segs_roc = identify_continuous_segments(daily_roc)
        X_base_roc, X_ess_roc, y_roc = build_sliding_windows_multi(segs_roc)
        if int(y_roc.sum()) >= 5 and (len(y_roc) - int(y_roc.sum())) >= 5:
            fpr_b, tpr_b, std_b, auc_b = evaluate_with_roc_curve(X_base_roc, y_roc)
            fpr_e, tpr_e, std_e, auc_e = evaluate_with_roc_curve(X_ess_roc, y_roc)
            roc_data = {
                'threshold': roc_thresh,
                'fpr_base': fpr_b, 'tpr_base': tpr_b, 'std_base': std_b, 'auc_base': auc_b,
                'fpr_ess': fpr_e, 'tpr_ess': tpr_e, 'std_ess': std_e, 'auc_ess': auc_e,
            }

    return results_df, roc_data


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Default file pairs -- update paths to match your setup
    file_pairs = [
        ('data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary_phosphate.csv'),
    ]

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        file_pairs = []
        for i in range(0, len(args), 2):
            main_file = args[i]
            phos_file = args[i+1] if i+1 < len(args) else None
            file_pairs.append((main_file, phos_file))

    print("="*60)
    print("  ESS Threshold Sensitivity Analysis")
    print("  Sweeping phycocyanin cutoff: 1.0 to 8.0 RFU")
    print("  71 thresholds x 2 models = 142 evaluations")
    print("="*60)

    results_df, roc_data = run_threshold_sweep(file_pairs)

    # Print the CSV so you can copy it if needed
    print("\n\nFull results table:")
    print(results_df.to_string(index=False))

    # =========================================================
    # Generate plot
    # =========================================================
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt

        valid = results_df[results_df['ess_f1'] > 0].copy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        BASE_COLOR = '#534AB7'
        ESS_COLOR  = '#1D9E75'
        WHO1_COLOR = '#A32D2D'
        WHO2_COLOR = '#791F1F'
        MSIZE, LWIDTH = 3, 1.5

        def _who_lines(ax):
            ax.axvline(x=1.8, color=WHO1_COLOR, linestyle='--', linewidth=1, alpha=0.7,
                       label='WHO Level 1 (1.8 RFU)')
            ax.axvline(x=5.0, color=WHO2_COLOR, linestyle=':', linewidth=1, alpha=0.5,
                       label='WHO Level 2 (5.0 RFU)')

        # --- Plot 1: F1 Score ---
        ax1.plot(valid['threshold'], valid['baseline_f1'], 'o-',
                 color=BASE_COLOR, label='Baseline', markersize=MSIZE, linewidth=LWIDTH)
        ax1.plot(valid['threshold'], valid['ess_f1'], 's-',
                 color=ESS_COLOR, label='ESS', markersize=MSIZE, linewidth=LWIDTH)
        _who_lines(ax1)
        ax1.set_title('F1 Score', fontsize=13)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # --- Plot 2: Bloom Precision & Recall ---
        ax2.plot(valid['threshold'], valid['baseline_bloom_precision'], 'o--',
                 color=BASE_COLOR, label='Baseline Precision', markersize=MSIZE, linewidth=LWIDTH)
        ax2.plot(valid['threshold'], valid['baseline_bloom_recall'], 'o-',
                 color=BASE_COLOR, label='Baseline Recall', markersize=MSIZE, linewidth=LWIDTH, alpha=0.5)
        ax2.plot(valid['threshold'], valid['ess_bloom_precision'], 's--',
                 color=ESS_COLOR, label='ESS Precision', markersize=MSIZE, linewidth=LWIDTH)
        ax2.plot(valid['threshold'], valid['ess_bloom_recall'], 's-',
                 color=ESS_COLOR, label='ESS Recall', markersize=MSIZE, linewidth=LWIDTH, alpha=0.5)
        _who_lines(ax2)
        ax2.set_title('Bloom (Class 1) — Precision & Recall', fontsize=13)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # --- Plot 3: No-Bloom Precision & Recall ---
        ax3.plot(valid['threshold'], valid['baseline_nobloom_precision'], 'o--',
                 color=BASE_COLOR, label='Baseline Precision', markersize=MSIZE, linewidth=LWIDTH)
        ax3.plot(valid['threshold'], valid['baseline_nobloom_recall'], 'o-',
                 color=BASE_COLOR, label='Baseline Recall', markersize=MSIZE, linewidth=LWIDTH, alpha=0.5)
        ax3.plot(valid['threshold'], valid['ess_nobloom_precision'], 's--',
                 color=ESS_COLOR, label='ESS Precision', markersize=MSIZE, linewidth=LWIDTH)
        ax3.plot(valid['threshold'], valid['ess_nobloom_recall'], 's-',
                 color=ESS_COLOR, label='ESS Recall', markersize=MSIZE, linewidth=LWIDTH, alpha=0.5)
        _who_lines(ax3)
        ax3.set_title('No-Bloom (Class 0) — Precision & Recall', fontsize=13)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_xlabel('Phycocyanin Bloom Threshold (RFU)', fontsize=11)
        ax3.legend(fontsize=8, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # --- Plot 4: Delta F1 (ESS advantage) ---
        colors = [ESS_COLOR if d > 0 else WHO1_COLOR for d in valid['delta_f1']]
        ax4.bar(valid['threshold'], valid['delta_f1'], width=0.08, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linewidth=0.8)
        _who_lines(ax4)
        ax4.set_title('ESS Advantage (Delta F1)', fontsize=13)
        ax4.set_ylabel('ESS F1 − Baseline F1', fontsize=11)
        ax4.set_xlabel('Phycocyanin Bloom Threshold (RFU)', fontsize=11)
        ax4.grid(True, alpha=0.3)

        fig.suptitle('ESS vs Baseline — Threshold Sensitivity Analysis', fontsize=15, y=1.01)
        plt.tight_layout()
        plot_path = 'threshold_sensitivity_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
        plt.close()

        # =========================================================
        # Separate AUC plot (ROC-AUC and PR-AUC)
        # =========================================================
        fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        # --- ROC-AUC ---
        axA.plot(valid['threshold'], valid['baseline_roc_auc'], 'o-',
                 color=BASE_COLOR, label='Baseline ROC-AUC', markersize=MSIZE, linewidth=LWIDTH)
        axA.plot(valid['threshold'], valid['ess_roc_auc'], 's-',
                 color=ESS_COLOR, label='ESS ROC-AUC', markersize=MSIZE, linewidth=LWIDTH)
        axA.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Random (0.5)')
        _who_lines(axA)
        axA.set_title('ROC-AUC Across Thresholds', fontsize=13)
        axA.set_ylabel('ROC-AUC', fontsize=11)
        axA.set_xlabel('Phycocyanin Bloom Threshold (RFU)', fontsize=11)
        axA.legend(fontsize=8, loc='lower right')
        axA.grid(True, alpha=0.3)
        axA.set_ylim(0, 1)

        # --- PR-AUC ---
        axB.plot(valid['threshold'], valid['baseline_prauc'], 'o-',
                 color=BASE_COLOR, label='Baseline PR-AUC', markersize=MSIZE, linewidth=LWIDTH)
        axB.plot(valid['threshold'], valid['ess_prauc'], 's-',
                 color=ESS_COLOR, label='ESS PR-AUC', markersize=MSIZE, linewidth=LWIDTH)
        baseline_val = valid['pos_windows'] / valid['total_windows']
        axB.plot(valid['threshold'], baseline_val, '--',
                 color='gray', linewidth=1, alpha=0.6, label='Random (class freq)')
        _who_lines(axB)
        axB.set_title('PR-AUC Across Thresholds', fontsize=13)
        axB.set_ylabel('PR-AUC', fontsize=11)
        axB.set_xlabel('Phycocyanin Bloom Threshold (RFU)', fontsize=11)
        axB.legend(fontsize=8, loc='upper right')
        axB.grid(True, alpha=0.3)
        axB.set_ylim(0, 1)

        fig2.suptitle('ESS vs Baseline — AUC Metrics', fontsize=14, y=1.02)
        plt.tight_layout()
        auc_plot_path = 'threshold_sensitivity_auc_plot.png'
        plt.savefig(auc_plot_path, dpi=150, bbox_inches='tight')
        print(f"AUC plot saved to {auc_plot_path}")
        plt.close()

        # =========================================================
        # ROC Curve plot (TPR vs FPR) at best threshold
        # =========================================================
        if roc_data:
            fig3, ax_roc = plt.subplots(figsize=(7, 6))

            ax_roc.plot(roc_data['fpr_base'], roc_data['tpr_base'],
                        color=BASE_COLOR, linewidth=2,
                        label=f"Baseline (AUC = {roc_data['auc_base']:.3f})")
            ax_roc.fill_between(roc_data['fpr_base'],
                                roc_data['tpr_base'] - roc_data['std_base'],
                                roc_data['tpr_base'] + roc_data['std_base'],
                                alpha=0.15, color=BASE_COLOR)

            ax_roc.plot(roc_data['fpr_ess'], roc_data['tpr_ess'],
                        color=ESS_COLOR, linewidth=2,
                        label=f"ESS (AUC = {roc_data['auc_ess']:.3f})")
            ax_roc.fill_between(roc_data['fpr_ess'],
                                roc_data['tpr_ess'] - roc_data['std_ess'],
                                roc_data['tpr_ess'] + roc_data['std_ess'],
                                alpha=0.15, color=ESS_COLOR)

            ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
            ax_roc.set_xlabel('False Positive Rate', fontsize=12)
            ax_roc.set_ylabel('True Positive Rate', fontsize=12)
            ax_roc.set_title(
                f'ROC Curve — Threshold {roc_data["threshold"]:.1f} RFU\n'
                f'(Mean ± std over 5-fold CV)',
                fontsize=13
            )
            ax_roc.legend(fontsize=10)
            ax_roc.grid(True, alpha=0.3)
            ax_roc.set_xlim(0, 1)
            ax_roc.set_ylim(0, 1)

            plt.tight_layout()
            roc_plot_path = 'roc_curve_plot.png'
            plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve plot saved to {roc_plot_path}")
            plt.close()

    except ImportError:
        print("\nMatplotlib not available. Copy threshold_sensitivity_results.csv")
        print("and paste the results back to Claude to generate a visualization.")