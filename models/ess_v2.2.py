"""
ESS Temporal Validation - Train on 2017 (severe), Test on 2016 & 2018 (mild)

This tests whether ESS patterns learned from a severe bloom year (SI=8)
can predict bloom events in mild bloom years (SI=3.2 and 3.6).
No cross-validation -- pure temporal train/test split.

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    matthews_corrcoef, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. THERMODYNAMIC CONSTANTS (identical to v2)
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


# =============================================================================
# 2. DERIVED CHEMICAL VARIABLES
# =============================================================================

def compute_derived_variables(df):
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])

    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)

    # FIXED: 5.0 multiplier (not 0.5)
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


# =============================================================================
# 3. Q/K RATIOS (5 reactions, no RXN 04)
# =============================================================================

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

    Ks_srp = 5.0
    df['qk_rxn05'] = df['srp_ugl'] / Ks_srp

    hpo4 = np.clip(df['hpo4_mol'], 1e-15, None)
    df['qk_rxn06'] = np.log10(co2 * hpo4) * -1

    return df


# =============================================================================
# 4. CONVERGENCE FEATURES
# =============================================================================

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


# =============================================================================
# 5. ECOLOGICAL RATIOS
# =============================================================================

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


# =============================================================================
# 6. BLOOM LABELING
# =============================================================================

def label_blooms(df):
    df['bloom_label'] = (df['pc_rfu'] >= 1.8).astype(int)
    return df


# =============================================================================
# 7. LOAD AND PROCESS ONE YEAR
# =============================================================================

def load_and_process_year(data_path, phosphate_path=None):
    """Load, merge phosphate, rename, compute chemistry, aggregate to daily."""

    df = pd.read_csv(data_path, skiprows=[1, 2], low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Merge phosphate
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

    # Rename columns
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

    # Impute missing SRP with median
    if 'srp_ugl' in df.columns:
        srp_missing = df['srp_ugl'].isna().sum()
        if srp_missing > 0:
            median_srp = df['srp_ugl'].median()
            if pd.isna(median_srp):
                median_srp = 5.0
            df['srp_ugl'] = df['srp_ugl'].fillna(median_srp)
    else:
        df['srp_ugl'] = 5.0

    required = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'chla_rfu', 'cond_uscm']
    df = df.dropna(subset=required).reset_index(drop=True)

    # Chemistry pipeline
    df = compute_derived_variables(df)
    df = compute_qk_ratios(df)
    df = compute_convergence_features(df)
    df = compute_ecological_ratios(df)
    df = label_blooms(df)

    # Aggregate to daily
    df['date'] = df['timestamp'].dt.date
    exclude_cols = ['timestamp', 'date']
    agg_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    agg_dict = {col: 'mean' for col in agg_cols}
    if 'bloom_label' in agg_dict:
        agg_dict['bloom_label'] = 'max'

    daily = df.groupby('date').agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    return daily


# =============================================================================
# 8. BUILD WINDOWS FROM ONE DATASET
# =============================================================================

def build_windows(df, input_days=20, target_days=5, step=1):
    """Build sliding windows from a single continuous daily dataset."""

    feature_cols_baseline = [
        'temp_c', 'do_mgl', 'ph', 'srp_ugl', 'cond_uscm'
    ]

    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score',
                 'favorability_product']
    feature_cols_ess = qk_cols + ['np_ratio'] + conv_cols

    X_baseline, X_ess, y = [], [], []
    total_len = len(df)

    for i in range(0, total_len - input_days - target_days + 1, step):
        input_window = df.iloc[i:i + input_days]
        target_window = df.iloc[i + input_days:i + input_days + target_days]

        # Check for gaps > 5 days within this window
        dates = pd.to_datetime(input_window['date'])
        max_gap = dates.diff().dt.days.max()
        if max_gap > 5:
            continue  # skip windows that span gaps

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
    ess_names.extend([
        'max_n_favorable', 'days_all_favorable',
        'max_consec_all_favorable', 'convergence_trend',
        'peak_favorability_product',
    ])

    X_baseline = pd.DataFrame(X_baseline, columns=base_names)
    X_ess = pd.DataFrame(X_ess, columns=ess_names)
    y = np.array(y)

    return X_baseline, X_ess, y


# =============================================================================
# 9. EVALUATE ON TEST SET (no cross-validation)
# =============================================================================

def evaluate_train_test(X_train, y_train, X_test, y_test, model_name):
    """Train on one set, test on another. No CV."""
    scaler = StandardScaler()

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

    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    prauc = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  PR-AUC:       {prauc:.4f}")
    print(f"  MCC:          {mcc:.4f}")
    print(f"\n  Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    else:
        print(f"    {cm}")
        print(f"    (Only one class in test set)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['No bloom', 'Bloom'],
                                zero_division=0))

    return {'f1': f1, 'prauc': prauc, 'mcc': mcc, 'confusion_matrix': cm}


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def run_pipeline(train_pair, test_pairs):
    """
    Train on one year, test on each other year separately.

    train_pair: (main_csv, phos_csv) for training year
    test_pairs: list of (name, main_csv, phos_csv) for test years
    """

    # --- Load and process training data ---
    # train_pair may be a single (path, phos) tuple or a list of such tuples
    if isinstance(train_pair[0], str):
        train_pairs_list = [train_pair]
    else:
        train_pairs_list = train_pair
    train_frames = []
    for train_path, train_phos in train_pairs_list:
        print(f"\n{'='*60}")
        print(f"  LOADING TRAINING DATA: {train_path}")
        print(f"{'='*60}")
        train_frames.append(load_and_process_year(train_path, train_phos))
    train_daily = pd.concat(train_frames, ignore_index=True)
    train_label = ' + '.join(os.path.splitext(os.path.basename(p))[0] for p, _ in train_pairs_list)

    n_pos = int(train_daily['bloom_label'].sum())
    n_neg = len(train_daily) - n_pos
    print(f"  Training days: {len(train_daily)}")
    print(f"  Bloom days: {n_pos} ({100*n_pos/len(train_daily):.1f}%)")

    # Build training windows
    X_train_base, X_train_ess, y_train = build_windows(train_daily)
    n_pos_w = int(y_train.sum())
    print(f"  Training windows: {len(y_train)} ({n_pos_w} positive)")

    # --- Test on each year ---
    for test_name, test_path, test_phos in test_pairs:
        print(f"\n\n{'#'*60}")
        print(f"  TESTING ON: {test_name}")
        print(f"{'#'*60}")

        test_daily = load_and_process_year(test_path, test_phos)

        n_pos_test = int(test_daily['bloom_label'].sum())
        n_neg_test = len(test_daily) - n_pos_test
        print(f"  Test days: {len(test_daily)}")
        print(f"  Bloom days: {n_pos_test} ({100*n_pos_test/len(test_daily):.1f}%)")

        X_test_base, X_test_ess, y_test = build_windows(test_daily)
        n_pos_tw = int(y_test.sum())
        print(f"  Test windows: {len(y_test)} ({n_pos_tw} positive)")

        if n_pos_tw == 0:
            print("\n  WARNING: No positive windows in test set. Cannot compute F1.")
            print("  Skipping this test year.")
            continue

        # Evaluate baseline
        res_base = evaluate_train_test(
            X_train_base, y_train, X_test_base, y_test,
            f"BASELINE on {test_name} (trained on {train_label})"
        )

        # Evaluate ESS
        res_ess = evaluate_train_test(
            X_train_ess, y_train, X_test_ess, y_test,
            f"ESS on {test_name} (trained on {train_label})"
        )

        # Summary for this test year
        print(f"\n  --- {test_name} Summary ---")
        print(f"  {'Metric':<12} {'Baseline':>10} {'ESS':>10} {'Delta':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for metric, label in [('f1', 'F1'), ('prauc', 'PR-AUC'), ('mcc', 'MCC')]:
            b = res_base[metric]
            e = res_ess[metric]
            delta = e - b
            sign = '+' if delta > 0 else ''
            print(f"  {label:<12} {b:>10.4f} {e:>10.4f} {sign}{delta:>9.4f}")

        winner = "ESS" if res_ess['f1'] > res_base['f1'] else "Baseline"
        gap = abs(res_ess['f1'] - res_base['f1'])
        print(f"\n  >> {winner} wins on {test_name} by {gap:.4f} F1")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    # Default file paths -- update to match your directory structure
    train_pair = [
        ('data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary_phosphate.csv'),
    ]

    test_pairs = [
        ('data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary_phosphate.csv'),
    ]
    test_pairs = [(os.path.splitext(os.path.basename(p))[0], p, ph) for p, ph in test_pairs]

    # Override with command line args if provided
    if len(sys.argv) > 1:
        # Usage: python ess_temporal.py train.csv train_phos.csv test1.csv test1_phos.csv ...
        args = sys.argv[1:]
        train_pair = (args[0], args[1] if len(args) > 1 else None)
        test_pairs = []
        for i in range(2, len(args), 2):
            name = f"Test {(i-2)//2 + 1}"
            main = args[i]
            phos = args[i+1] if i+1 < len(args) else None
            test_pairs.append((name, main, phos))

    print("="*60)
    print("  ESS Temporal Validation")
    print("  Train: 2017 (severe bloom, SI=8)")
    print("  Test:  2016 (mild, SI=3.2) & 2018 (mild, SI=3.6)")
    print("  1.8 RFU threshold (WHO Alert Level 1)")
    print("="*60)

    run_pipeline(train_pair, test_pairs)