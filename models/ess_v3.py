"""
ESS Model Comparison Pipeline - v2 Multi-Year
Baseline (raw features) vs ESS (Q/K-derived features) using logistic regression.
Same model, same data, different input representations.

v2 FIXES:
  - Alkalinity multiplier corrected from 0.5 to 5.0 (was 10x too low)
  - O2 saturation upgraded to full Benson & Krause equation
  - RXN 04 removed (was constant 1.9 every row, no N2 sensor)
  - PC and Chl-a removed from baseline features (circularity with target)
  - PC:Chl ratio removed from ESS features (same reason)
  - Daily aggregation BEFORE windowing (was 129k windows, should be ~500)
  - Convergence features added (synchronized multi-reaction shifts)
  - Phosphate merge fixed (was inflating rows 7.7x)
  - Multi-year support with segment-aware windowing
  - 3.0 RFU bloom threshold (WHO Alert Level 1, McQuaid et al. / Zamyadi et al. 2023)

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
Data: NOAA GLERL/CIGLR WE02/WE08 buoys, 2016-2018
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. THERMODYNAMIC CONSTANTS
# =============================================================================

def carbonate_K1(T_K):
    """First dissociation of CO2(aq). Plummer & Busenberg 1982."""
    log_K1 = (-356.3094 - 0.06091964 * T_K + 21834.37 / T_K
              + 126.8339 * np.log10(T_K) - 1684915.0 / T_K**2)
    return 10**log_K1

def carbonate_K2(T_K):
    """Second dissociation of bicarbonate. Plummer & Busenberg 1982."""
    log_K2 = (-107.8871 - 0.03252849 * T_K + 5151.79 / T_K
              + 38.92561 * np.log10(T_K) - 563713.9 / T_K**2)
    return 10**log_K2

def henry_KH(T_K):
    """Henry's law constant for CO2. Weiss 1974. Returns mol/(L*atm)."""
    ln_KH = (-60.2409 + 93.4517 * (100.0 / T_K)
             + 23.3585 * np.log(T_K / 100.0))
    return np.exp(ln_KH)

def Kw(T_K):
    """Ion product of water. Harned & Hamer 1933."""
    log_Kw = -4470.99 / T_K + 6.0846 - 0.01706 * T_K
    return 10**log_Kw

def phosphate_Ka1(T_K):
    """H3PO4 -> H2PO4- + H+. Stumm & Morgan 1996 / NIST SRD 46."""
    return 10**(-2.148 + 0.0 * (T_K - 298.15))

def phosphate_Ka2(T_K):
    """H2PO4- -> HPO4^2- + H+. Stumm & Morgan 1996."""
    return 10**(-7.198 + 0.0 * (T_K - 298.15))

def phosphate_Ka3(T_K):
    """HPO4^2- -> PO4^3- + H+. Stumm & Morgan 1996."""
    return 10**(-12.375 + 0.0 * (T_K - 298.15))

def N2_saturation(T_K):
    """Dissolved N2 saturation concentration (mol/L). Sander 2015."""
    KH_N2 = 6.1e-4 * np.exp(1300 * (1/T_K - 1/298.15))
    return KH_N2 * 0.78

# !!!! PROBLEM FIXED AREA !!!!
# v1 used a simple polynomial that ran 5-8% high compared to published values.
# v2 uses the full Benson & Krause 1984 equation with natural log terms.
def O2_saturation(T_K):
    """Dissolved O2 saturation (mg/L). Benson & Krause 1984."""
    T_K = np.clip(T_K, 273.16, None)
    ln_DO = (-139.34411 + (1.575701e5 / T_K) - (6.642308e7 / T_K**2)
             + (1.243800e10 / T_K**3) - (8.621949e11 / T_K**4))
    return np.exp(ln_DO)
# !!!! PROBLEM FIXED AREA !!!!


# =============================================================================
# 2. DERIVED CHEMICAL VARIABLES
# =============================================================================

def compute_derived_variables(df):
    """Compute intermediate chemical variables from raw sensor data."""
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])

    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)

    # !!!! PROBLEM FIXED AREA !!!!
    # v1 used 0.5 * conductivity -- 10x too low for Lake Erie.
    # v2 uses 5.0 * conductivity to match published alkalinity (~1600-2200 ueq/L).
    alk_mol = (5.0 * df['cond_uscm']) * 1e-6
    # !!!! PROBLEM FIXED AREA !!!!

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
    df['pco2_uatm'] = (df['co2_aq_mol'] / KH) * 1e6

    Ka1 = phosphate_Ka1(T_K)
    Ka2p = phosphate_Ka2(T_K)
    Ka3 = phosphate_Ka3(T_K)

    P_total = (df['srp_ugl'] * 1e-6) / 30.97
    P_total = np.clip(P_total, 1e-12, None)

    denom_p = (H**3 + Ka1 * H**2 + Ka1 * Ka2p * H + Ka1 * Ka2p * Ka3)
    df['h2po4_mol'] = P_total * (Ka1 * H**2) / denom_p
    df['hpo4_mol'] = P_total * (Ka1 * Ka2p * H) / denom_p

    df['n2_sat_mol'] = N2_saturation(T_K)

    pKa_nh4 = 0.09018 + 2729.92 / T_K
    Ka_nh4 = 10**(-pKa_nh4)
    df['nh3_fraction'] = Ka_nh4 / (Ka_nh4 + H)

    df['o2_sat_mgl'] = O2_saturation(T_K)
    df['do_mol'] = (df['do_mgl'] / 32.0) * 1e-3
    df['o2_sat_mol'] = (df['o2_sat_mgl'] / 32.0) * 1e-3

    df['Kw_val'] = Kw(T_K)
    df['KH_val'] = KH

    return df


# =============================================================================
# 3. REACTION Q/K RATIOS (5 reactions, RXN 04 removed)
# =============================================================================

def compute_qk_ratios(df):
    """Compute Q/K for 5 reactions. RXN 04 removed (no N2 sensor)."""
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])
    co2 = np.clip(df['co2_aq_mol'], 1e-12, None)
    do = np.clip(df['do_mol'], 1e-12, None)

    # RXN 01: Photosynthesis
    Q_01 = (do**6) / (co2**6)
    df['qk_rxn01'] = np.log10(1.0 / Q_01)

    # RXN 02: Carbonate equilibrium
    K1 = carbonate_K1(T_K)
    Q_02 = (df['hco3_mol'] * H) / co2
    df['qk_rxn02'] = Q_02 / K1

    # RXN 03: Phosphate speciation
    Ka2 = phosphate_Ka2(T_K)
    h2po4 = np.clip(df['h2po4_mol'], 1e-15, None)
    Q_03 = (df['hpo4_mol'] * H) / h2po4
    df['qk_rxn03'] = Q_03 / Ka2

    # !!!! PROBLEM FIXED AREA !!!!
    # RXN 04 REMOVED. Was constant 1.9 for every row (no N2 sensor).
    # !!!! PROBLEM FIXED AREA !!!!

    # RXN 05: Phosphate uptake (Monod)
    Ks_srp = 5.0
    df['qk_rxn05'] = df['srp_ugl'] / Ks_srp

    # RXN 06: Biomass synthesis (Redfield)
    hpo4 = np.clip(df['hpo4_mol'], 1e-15, None)
    df['qk_rxn06'] = np.log10(co2 * hpo4) * -1

    return df


# =============================================================================
# 4. CONVERGENCE FEATURES
# =============================================================================

def compute_convergence_features(df):
    """Classify reactions into Favorable/Neutral/Unfavorable states,
    then compute synchronized convergence features."""
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
# 5. ECOLOGICAL RATIOS (PC:Chl removed)
# =============================================================================

# !!!! PROBLEM FIXED AREA !!!!
# PC:Chl removed -- circularity with bloom label defined from pc_rfu.
# !!!! PROBLEM FIXED AREA !!!!

def compute_ecological_ratios(df):
    """N:P ratio only."""
    if 'nitrate_mgl' in df.columns:
        N_mol = (df['nitrate_mgl'] / 62.004) * 1e-3
    else:
        N_mol = (df['cond_uscm'] * 0.002) * 1e-3 / 14.007

    P_mol = (df['srp_ugl'] * 1e-6) / 30.97
    P_mol = np.clip(P_mol, 1e-12, None)
    df['np_ratio'] = N_mol / P_mol
    return df


# =============================================================================
# 6. BLOOM LABELING (3.0 RFU - WHO Alert Level 1)
# =============================================================================

# !!!! PROBLEM FIXED AREA !!!!
# v1 used sigmoid with midpoint at 1.0 RFU, which labeled everything above
# 1.0 RFU as a bloom (29% of data). That threshold captured minor algal
# background activity, not actual harmful bloom events.
# v2 uses 3.0 RFU, which corresponds to WHO Alert Level 1 for the YSI EXO2
# phycocyanin probe (McQuaid et al.; Zamyadi et al. 2023, Env Sci: Water
# Research & Technology). This is the internationally recognized threshold
# at which increased monitoring is recommended.
# !!!! PROBLEM FIXED AREA !!!!

def label_blooms(df):
    """Label bloom/no-bloom using phycocyanin threshold of 3.0 RFU.
    Based on WHO Alert Level 1 mapped to YSI EXO2 probe readings."""
    df['bloom_label'] = (df['pc_rfu'] >= 1.3).astype(int)
    return df


# =============================================================================
# 7. DAILY AGGREGATION
# =============================================================================

# !!!! PROBLEM FIXED AREA !!!!
# v1 ran windows on 15-min data (129k windows). v2 aggregates to daily first.
# !!!! PROBLEM FIXED AREA !!!!

def aggregate_to_daily(df):
    """Aggregate sub-daily data to daily means before windowing."""
    df['date'] = df['timestamp'].dt.date

    exclude_cols = ['timestamp', 'date', 'source_file']
    agg_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    agg_dict = {col: 'mean' for col in agg_cols}
    if 'bloom_label' in agg_dict:
        agg_dict['bloom_label'] = 'max'

    daily = df.groupby('date').agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    print(f"  Aggregated {len(df)} sub-daily rows -> {len(daily)} daily rows")
    return daily


# =============================================================================
# 8. MULTI-YEAR DATA LOADING
# =============================================================================

def load_and_combine(file_pairs):
    """Load multiple (main_csv, phosphate_csv) pairs and combine."""
    all_dfs = []

    for data_path, phosphate_path in file_pairs:
        print(f"\n  Loading {data_path}...")
        df = pd.read_csv(data_path, skiprows=[1, 2], low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        if phosphate_path:
            print(f"    Merging phosphate from {phosphate_path}...")
            phos = pd.read_csv(phosphate_path, skiprows=[1, 2])
            phos['timestamp'] = pd.to_datetime(phos['timestamp'], errors='coerce')
            phos = phos.dropna(subset=['timestamp'])
            phos['phosphate'] = pd.to_numeric(phos['phosphate'], errors='coerce')

            # !!!! PROBLEM FIXED AREA !!!!
            # v1 merged raw phosphate causing 7.7x row inflation.
            # v2 filters bad values and aggregates to daily BEFORE merging.
            phos = phos[(phos['phosphate'] >= 0) & (phos['phosphate'] <= 500)]
            phos['date'] = phos['timestamp'].dt.date
            phos_daily = phos.groupby('date')['phosphate'].mean().reset_index()

            df['date'] = df['timestamp'].dt.date
            df = df.merge(phos_daily, on='date', how='left')
            df['phosphate'] = df['phosphate'].ffill()
            df.drop(columns=['date'], inplace=True)
            # !!!! PROBLEM FIXED AREA !!!!

            n_with = df['phosphate'].notna().sum()
            print(f"    Phosphate coverage: {n_with}/{len(df)} rows ({100*n_with/len(df):.1f}%)")

        df['source_file'] = data_path
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    print(f"\n  Combined: {len(combined)} total rows")
    print(f"  Date range: {combined['timestamp'].min().date()} to {combined['timestamp'].max().date()}")

    return combined


# =============================================================================
# 9. CONTINUOUS SEGMENT DETECTION
# =============================================================================

def identify_continuous_segments(daily_df, max_gap_days=5):
    """Split daily data into segments where no gap exceeds max_gap_days.
    Prevents windows from spanning across winter gaps between seasons."""
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(daily_df['date'])
    gaps = dates.diff().dt.days

    break_points = gaps[gaps > max_gap_days].index.tolist()

    segments = []
    start = 0
    for bp in break_points:
        segment = daily_df.iloc[start:bp].copy()
        if len(segment) > 0:
            segments.append(segment)
        start = bp
    segment = daily_df.iloc[start:].copy()
    if len(segment) > 0:
        segments.append(segment)

    print(f"  Found {len(segments)} continuous segments:")
    for i, seg in enumerate(segments):
        seg_dates = pd.to_datetime(seg['date'])
        print(f"    Segment {i+1}: {seg_dates.min().date()} to {seg_dates.max().date()} ({len(seg)} days)")

    return segments


# =============================================================================
# 10. SLIDING WINDOWS (segment-aware)
# =============================================================================

def build_sliding_windows_multi(segments, input_days=20, target_days=5, step=1):
    """Build sliding windows from multiple segments. Never crosses gaps."""

    # !!!! PROBLEM FIXED AREA !!!!
    # v1 baseline included pc_rfu and chla_rfu (circularity with target).
    # v2 uses only abiotic measurements.
    feature_cols_baseline = [
        'temp_c', 'do_mgl', 'ph', 'srp_ugl', 'cond_uscm'
    ]
    # !!!! PROBLEM FIXED AREA !!!!

    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score',
                 'favorability_product']
    feature_cols_ess = qk_cols + ['np_ratio'] + conv_cols

    X_baseline, X_ess, y, window_dates = [], [], [], []

    for seg_idx, seg in enumerate(segments):
        total_len = len(seg)
        if total_len < input_days + target_days:
            print(f"    Segment {seg_idx+1}: too short ({total_len} days), skipping")
            continue

        n_windows = 0
        for i in range(0, total_len - input_days - target_days + 1, step):
            input_window = seg.iloc[i:i + input_days]
            target_window = seg.iloc[i + input_days:i + input_days + target_days]
            forecast_date = seg.iloc[i + input_days - 1]['date']

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
                x_time = np.arange(len(n_fav))
                slope = np.polyfit(x_time, n_fav, 1)[0]
            else:
                slope = 0.0
            ess_feats.append(slope)

            ess_feats.append(input_window['favorability_product'].max())

            label = int(target_window['bloom_label'].max())

            X_baseline.append(base_feats)
            X_ess.append(ess_feats)
            y.append(label)
            window_dates.append(forecast_date)
            n_windows += 1

        print(f"    Segment {seg_idx+1}: {n_windows} windows")

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

    return X_baseline, X_ess, y, window_dates


# =============================================================================
# 11. MODEL EVALUATION
# =============================================================================

def evaluate_model(X, y, model_name, n_splits=5):
    """Stratified k-fold CV with logistic regression."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    f1_scores, prauc_scores, mcc_scores = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    oof_probs = np.full(len(y), np.nan)

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

        f1_scores.append(f1_score(y_test, y_pred))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        prauc_scores.append(auc(recall, precision))

        oof_probs[test_idx] = y_prob

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  F1 Score:     {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print(f"  PR-AUC:       {np.mean(prauc_scores):.4f} +/- {np.std(prauc_scores):.4f}")
    print(f"  MCC:          {np.mean(mcc_scores):.4f} +/- {np.std(mcc_scores):.4f}")
    print(f"\n  Confusion Matrix (aggregated across folds):")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=['No bloom', 'Bloom']))

    return {
        'f1': np.mean(f1_scores), 'f1_std': np.std(f1_scores),
        'prauc': np.mean(prauc_scores), 'prauc_std': np.std(prauc_scores),
        'mcc': np.mean(mcc_scores), 'mcc_std': np.std(mcc_scores),
        'confusion_matrix': cm,
        'oof_probs': oof_probs,
    }


def print_feature_importance(X, y, feature_names, model_name):
    """Fit on full data and print top coefficient magnitudes."""
    scaler = StandardScaler()
    X_s = np.nan_to_num(scaler.fit_transform(X), nan=0.0, posinf=10, neginf=-10)

    model = LogisticRegression(
        class_weight='balanced', max_iter=1000,
        solver='lbfgs', random_state=42
    )
    model.fit(X_s, y)

    coefs = pd.Series(model.coef_[0], index=feature_names)
    coefs_sorted = coefs.abs().sort_values(ascending=False)

    print(f"\n  Feature Importance ({model_name}):")
    print(f"  {'Feature':<35} {'Coefficient':>12}")
    print(f"  {'-'*35} {'-'*12}")
    for feat in coefs_sorted.index[:15]:
        print(f"  {feat:<35} {coefs[feat]:>12.4f}")


# =============================================================================
# 12. EXCEL EXPORT
# =============================================================================

def export_daily_to_excel(daily_df, output_path=None):
    """Export daily Baseline and ESS values with timestamps to Excel."""
    if output_path is None:
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'ess_daily_results_{run_ts}.xlsx'

    baseline_cols = [
        'date', 'temp_c', 'do_mgl', 'ph', 'srp_ugl', 'cond_uscm',
        'baseline_pred', 'ess_pred', 'bloom_label',
    ]
    ess_cols = [
        'date',
        'qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06',
        'np_ratio',
        'n_favorable', 'frac_favorable', 'all_favorable', 'n_unfavorable',
        'convergence_score', 'favorability_product',
        'baseline_pred', 'ess_pred', 'bloom_label',
    ]

    baseline_cols = [c for c in baseline_cols if c in daily_df.columns]
    ess_cols = [c for c in ess_cols if c in daily_df.columns]

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        daily_df[baseline_cols].to_excel(writer, sheet_name='Baseline Daily', index=False)
        daily_df[ess_cols].to_excel(writer, sheet_name='ESS Daily', index=False)

    print(f"\n  Daily values exported to: {output_path}")
    return output_path


# =============================================================================
# 13. MAIN PIPELINE - v2 multi-year
# =============================================================================

def run_pipeline(file_pairs):
    """Full v2 multi-year pipeline."""

    # --- Load and combine ---
    print("Loading and combining datasets...")
    df = load_and_combine(file_pairs)

    # --- Rename columns ---
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

    numeric_cols = ['temp_c', 'do_mgl', 'ph', 'srp_ugl',
                    'pc_rfu', 'chla_rfu', 'cond_uscm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'cond_uscm' in df.columns:
        df['cond_uscm'] = df['cond_uscm'] * 1000

    required_no_phos = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'chla_rfu', 'cond_uscm']
    missing = [c for c in required_no_phos if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        return

    # Impute missing SRP with median rather than dropping rows
    if 'srp_ugl' in df.columns:
        srp_missing = df['srp_ugl'].isna().sum()
        if srp_missing > 0:
            median_srp = df['srp_ugl'].median()
            df['srp_ugl'] = df['srp_ugl'].fillna(median_srp)
            print(f"  Imputed {srp_missing} missing SRP values with median ({median_srp:.1f} ug/L)")
    else:
        print("  WARNING: No phosphate data. Using placeholder 5.0 ug/L")
        df['srp_ugl'] = 5.0

    initial_len = len(df)
    df = df.dropna(subset=required_no_phos).reset_index(drop=True)
    print(f"  Rows: {initial_len} -> {len(df)} after dropping NaN")

    # --- Chemistry pipeline ---
    print("\nComputing derived chemical variables (FIXED alkalinity)...")
    df = compute_derived_variables(df)

    print("Computing Q/K ratios for 5 reactions...")
    df = compute_qk_ratios(df)

    print("Computing convergence features...")
    df = compute_convergence_features(df)

    print("Computing N:P ratio...")
    df = compute_ecological_ratios(df)

    print("Labeling bloom events (3.0 RFU, WHO Alert Level 1)...")
    df = label_blooms(df)

    # --- Daily aggregation ---
    print("\nAggregating to daily resolution...")
    daily = aggregate_to_daily(df)

    n_pos = int(daily['bloom_label'].sum())
    n_neg = len(daily) - n_pos
    print(f"\n  Daily class distribution:")
    print(f"    Bloom:    {n_pos:4d} ({100*n_pos/len(daily):.1f}%)")
    print(f"    No bloom: {n_neg:4d} ({100*n_neg/len(daily):.1f}%)")

    # --- Segment and window ---
    print("\nIdentifying continuous segments (max 5-day gap)...")
    segments = identify_continuous_segments(daily)

    print("\nBuilding sliding windows (20-day input / 5-day target)...")
    X_baseline, X_ess, y, window_dates = build_sliding_windows_multi(segments)

    n_pos_w = int(y.sum())
    n_neg_w = len(y) - n_pos_w
    print(f"\n  Total windows:    {len(y)}")
    print(f"  Positive windows: {n_pos_w} ({100*n_pos_w/len(y):.1f}%)")
    print(f"  Negative windows: {n_neg_w} ({100*n_neg_w/len(y):.1f}%)")
    print(f"  Baseline features: {X_baseline.shape[1]}")
    print(f"  ESS features:      {X_ess.shape[1]}")

    if n_pos_w < 10:
        print("\n  WARNING: Very few positive samples. Results may be unreliable.")

    # --- Evaluate ---
    print("\n" + "="*60)
    print("  RUNNING MODEL COMPARISON - v2 multi-year")
    print("="*60)

    results_baseline = evaluate_model(
        X_baseline, y, "BASELINE: Logistic Regression on Raw Features"
    )
    results_ess = evaluate_model(
        X_ess, y, "ESS: Logistic Regression on Q/K + Convergence Features"
    )

    print_feature_importance(X_baseline, y, X_baseline.columns, "Baseline")
    print_feature_importance(X_ess, y, X_ess.columns, "ESS")

    # --- Map predictions to daily dates and export ---
    pred_df = pd.DataFrame({
        'date': pd.to_datetime(window_dates),
        'baseline_pred': results_baseline['oof_probs'],
        'ess_pred': results_ess['oof_probs'],
    })
    # Multiple windows can land on the same date — take the mean probability
    pred_df = pred_df.groupby('date')[['baseline_pred', 'ess_pred']].mean().reset_index()
    daily = daily.merge(pred_df, on='date', how='left')

    print("\nExporting daily ESS and Baseline values to Excel...")
    export_daily_to_excel(daily)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON - v2 multi-year")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Baseline':>18} {'ESS':>18} {'Delta':>10}")
    print(f"  {'-'*12} {'-'*18} {'-'*18} {'-'*10}")

    for metric, label in [('f1', 'F1'), ('prauc', 'PR-AUC'), ('mcc', 'MCC')]:
        b = results_baseline[metric]
        e = results_ess[metric]
        delta = e - b
        sign = '+' if delta > 0 else ''
        print(f"  {label:<12} {b:>12.4f} +/- {results_baseline[metric+'_std']:.3f}"
              f"  {e:>7.4f} +/- {results_ess[metric+'_std']:.3f}"
              f"  {sign}{delta:.4f}")

    winner = "ESS" if results_ess['f1'] > results_baseline['f1'] else "Baseline"
    gap = abs(results_ess['f1'] - results_baseline['f1'])
    print(f"\n  >> {winner} wins on F1 by {gap:.4f}")

    return results_baseline, results_ess


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    # Default: all three datasets
    file_pairs = [
        ('data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2016_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2017_annual_summary_phosphate.csv'),
        ('data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary.csv',
         'data/2016-2018 Lake Erie Water Data/WE02_2018_annual_summary_phosphate.csv'),
    ]

    # Override with command line: python ess_v2.py main1.csv phos1.csv main2.csv phos2.csv
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        file_pairs = []
        for i in range(0, len(args), 2):
            main_file = args[i]
            phos_file = args[i+1] if i+1 < len(args) else None
            file_pairs.append((main_file, phos_file))

    print("="*60)
    print("  ESS vs Baseline - v2 multi-year")
    print("  3 years | 3.0 RFU threshold | convergence features")
    print("  alkalinity fixed | segment-aware windowing")
    print("="*60)

    run_pipeline(file_pairs)