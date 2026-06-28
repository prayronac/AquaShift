"""
ESS Model Comparison Pipeline - v3 Seneca Lake Adapter
Baseline (raw features) vs ESS (Q/K-derived features) using logistic regression.
Same ESS framework as v3, adapted for USGS NWIS Seneca Lake data format.

ADAPTATIONS FROM LAKE ERIE v3:
  - Loads from single Excel file (USGS NWIS wide-format) instead of CSV pairs
  - pH ESTIMATED from DO saturation ratio (no pH sensor in download)
  - SRP placeholder at 5.0 ug/L (no phosphate sensor in download)
  - Conductivity already in uS/cm (no *1000 conversion)
  - Alkalinity multiplier 3.5 for Seneca Lake (~2200 ueq/L typical)
  - Bloom threshold calibrated for ug/L units (not RFU)
  - Nitrate available from s::can nitrolyser (parameter 99133)

IMPORTANT CAVEATS:
  - pH estimation is empirical, not measured. This introduces uncertainty
    into RXN 01, 02, 03, and 06. The DO-pH coupling preserves the
    photosynthesis/respiration signal but absolute Q/K values are approximate.
  - SRP placeholder makes RXN 03 and RXN 05 uninformative for this site.
    These reactions will show signal only if real SRP data is added later.

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
Data: USGS NWIS Seneca Lake platform (425027076564401), 2018-2019
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
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. THERMODYNAMIC CONSTANTS (identical to v3)
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

def N2_saturation(T_K):
    KH_N2 = 6.1e-4 * np.exp(1300 * (1/T_K - 1/298.15))
    return KH_N2 * 0.78

def O2_saturation(T_K):
    T_K = np.clip(T_K, 273.16, None)
    ln_DO = (-139.34411 + (1.575701e5 / T_K) - (6.642308e7 / T_K**2)
             + (1.243800e10 / T_K**3) - (8.621949e11 / T_K**4))
    return np.exp(ln_DO)


# =============================================================================
# 2. pH ESTIMATION (new for Seneca adapter)
# =============================================================================

def estimate_ph_from_do(df):
    """Estimate pH from DO saturation ratio.

    In productive lakes, photosynthesis and respiration drive both DO and pH:
      - Photosynthesis: CO2 consumed -> pH rises, O2 produced -> DO rises
      - Respiration: CO2 produced -> pH drops, O2 consumed -> DO drops

    This empirical model uses the DO/DO_sat ratio to estimate pH departure
    from a lake-specific baseline. For Seneca Lake (well-buffered, slightly
    alkaline, mesotrophic), baseline pH ~ 7.9 with buffering range 7.3-8.6.

    Reference baseline: Halfman et al. (2023), Seneca Lake CTD profiles.
    """
    T_K = df['temp_c'] + 273.15
    do_sat = O2_saturation(T_K)
    do_ratio = df['do_mgl'] / do_sat

    # Baseline pH for Seneca Lake + scaled departure from DO equilibrium
    # Coefficient 1.2 calibrated to match typical Finger Lakes pH range
    pH_BASELINE = 7.9
    pH_SENSITIVITY = 1.2
    df['ph'] = pH_BASELINE + pH_SENSITIVITY * (do_ratio - 1.0)
    df['ph'] = df['ph'].clip(6.5, 9.5)
    df['ph_is_estimated'] = True

    print(f"  pH estimated from DO saturation ratio")
    print(f"    Range: {df['ph'].min():.2f} - {df['ph'].max():.2f}")
    print(f"    Mean:  {df['ph'].mean():.2f}")

    return df


# =============================================================================
# 3. DERIVED CHEMICAL VARIABLES (adapted for Seneca)
# =============================================================================

def compute_derived_variables(df):
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])

    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)

    # Seneca Lake alkalinity: ~100-130 mg/L CaCO3 = ~2000-2600 ueq/L
    # Multiplier 3.5 * ~650 uS/cm * 1e-6 = ~0.0023 mol/L = 2275 ueq/L
    ALK_MULTIPLIER = 3.5
    alk_mol = (ALK_MULTIPLIER * df['cond_uscm']) * 1e-6

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
# 4. REACTION Q/K RATIOS (identical logic to v3)
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
# 5. CONVERGENCE FEATURES (identical to v3)
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
# 6. ECOLOGICAL RATIOS
# =============================================================================

def compute_ecological_ratios(df):
    if 'nitrate_mgl' in df.columns:
        N_mol = (df['nitrate_mgl'] / 62.004) * 1e-3
    else:
        N_mol = (df['cond_uscm'] * 0.002) * 1e-3 / 14.007

    P_mol = (df['srp_ugl'] * 1e-6) / 30.97
    P_mol = np.clip(P_mol, 1e-12, None)
    df['np_ratio'] = N_mol / P_mol
    return df


# =============================================================================
# 7. BLOOM LABELING (calibrated for ug/L units)
# =============================================================================

def label_blooms(df):
    """Label bloom/no-bloom using phycocyanin threshold.
    For EXO2 BGA-PC sensor in ug/L mode:
      3.0 ug/L ~ WHO Alert Level 1 equivalent
      Captures ~19% of Seneca observations (comparable to Lake Erie bloom rate)
    """
    BLOOM_THRESHOLD_UGL = 3.0
    df['bloom_label'] = (df['pc_ugl'] >= BLOOM_THRESHOLD_UGL).astype(int)
    return df


# =============================================================================
# 8. DAILY AGGREGATION (identical to v3)
# =============================================================================

def aggregate_to_daily(df):
    df['date'] = df['timestamp'].dt.date

    exclude_cols = ['timestamp', 'date', 'source_file', 'ph_is_estimated']
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
# 9. DATA LOADING (rewritten for USGS NWIS Excel format)
# =============================================================================

def load_seneca_data(filepath):
    """Load Seneca Lake combined Excel file (USGS NWIS wide format)."""
    print(f"\n  Loading {filepath}...")
    df = pd.read_excel(filepath)

    # Column mapping: USGS NWIS names -> ESS internal names
    col_map = {
        'datetime_utc': 'timestamp',
        'Temperature_degC': 'temp_c',
        'SpCond_uScm': 'cond_uscm',
        'DO_mgL': 'do_mgl',
        'Turbidity_FNU': 'turbidity_fnu',
        'Chlorophyll-a': 'chla_ugl',
        'Phycocyanin_ugL': 'pc_ugl',
        'pH': 'ph',
        'fDOM_QSE': 'fdom_qse',
        'Nitrate_mgL': 'nitrate_mgl',
        'NO3NO2_mgL': 'no3no2_mgl',
    }
    df = df.rename(columns=col_map)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    for col in ['temp_c', 'do_mgl', 'cond_uscm', 'pc_ugl', 'chla_ugl',
                'turbidity_fnu', 'fdom_qse', 'nitrate_mgl']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # NOTE: conductivity already in uS/cm from USGS NWIS -- no conversion
    print(f"  Rows loaded: {len(df)}")
    print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Conductivity range: {df['cond_uscm'].min():.0f} - {df['cond_uscm'].max():.0f} uS/cm (no conversion needed)")

    return df


# =============================================================================
# 10. CONTINUOUS SEGMENT DETECTION (identical to v3)
# =============================================================================

def identify_continuous_segments(daily_df, max_gap_days=5):
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
# 11. SLIDING WINDOWS (identical to v3)
# =============================================================================

def build_sliding_windows_multi(segments, input_days=20, target_days=5, step=1):
    feature_cols_baseline = [
        'temp_c', 'do_mgl', 'ph', 'cond_uscm'
    ]
    # Add nitrate to baseline if available (Seneca has it, Erie didn't)
    if 'nitrate_mgl' in segments[0].columns:
        feature_cols_baseline.append('nitrate_mgl')

    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score',
                 'favorability_product']
    feature_cols_ess = qk_cols + ['np_ratio'] + conv_cols

    X_baseline, X_ess, y = [], [], []

    for seg_idx, seg in enumerate(segments):
        total_len = len(seg)
        if total_len < input_days + target_days:
            print(f"    Segment {seg_idx+1}: too short ({total_len} days), skipping")
            continue

        n_windows = 0
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

    return X_baseline, X_ess, y


# =============================================================================
# 12. MODEL EVALUATION (identical to v3)
# =============================================================================

def evaluate_model(X, y, model_name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    f1_scores, prauc_scores, mcc_scores = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

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
        'confusion_matrix': cm
    }


def print_feature_importance(X, y, feature_names, model_name):
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
# 13. MAIN PIPELINE (adapted for Seneca Lake)
# =============================================================================

def run_pipeline(filepath):
    """Full pipeline for Seneca Lake USGS NWIS data."""

    # --- Load ---
    df = load_seneca_data(filepath)

    # --- Required columns check ---
    required = ['temp_c', 'do_mgl', 'cond_uscm', 'pc_ugl']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        return

    # --- Drop rows missing core parameters ---
    initial_len = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  Rows: {initial_len} -> {len(df)} after dropping NaN in core columns")

    # --- pH: use measured values if available, otherwise estimate ---
    if 'ph' in df.columns and df['ph'].notna().sum() > 0:
        print("\n  pH: using measured values from sensor")
    else:
        print("\nEstimating pH from DO saturation ratio (no sensor data)...")
        df = estimate_ph_from_do(df)

    # --- Handle missing SRP ---
    if 'srp_ugl' not in df.columns:
        print("\n  WARNING: No SRP data. Using placeholder 5.0 ug/L")
        print("    -> RXN 03 and RXN 05 will be uninformative constants")
        df['srp_ugl'] = 5.0

    # --- Chemistry pipeline ---
    print("\nComputing derived chemical variables...")
    df = compute_derived_variables(df)

    print("Computing Q/K ratios for 5 reactions...")
    df = compute_qk_ratios(df)

    print("Computing convergence features...")
    df = compute_convergence_features(df)

    print("Computing N:P ratio...")
    df = compute_ecological_ratios(df)

    print("Labeling bloom events (3.0 ug/L threshold)...")
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
    X_baseline, X_ess, y = build_sliding_windows_multi(segments)

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
    print("  RUNNING MODEL COMPARISON - Seneca Lake validation")
    print("="*60)

    results_baseline = evaluate_model(
        X_baseline, y, "BASELINE: Logistic Regression on Raw Features"
    )
    results_ess = evaluate_model(
        X_ess, y, "ESS: Logistic Regression on Q/K + Convergence Features"
    )

    print_feature_importance(X_baseline, y, X_baseline.columns, "Baseline")
    print_feature_importance(X_ess, y, X_ess.columns, "ESS")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON - Seneca Lake validation")
    print(f"{'='*60}")
    print(f"  Site: USGS 425027076564401 (Seneca Lake Platform)")
    print(f"  pH: ESTIMATED from DO saturation (not measured)")
    print(f"  SRP: PLACEHOLDER 5.0 ug/L (not measured)")
    print(f"  Bloom threshold: 3.0 ug/L phycocyanin")
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

    # Default path
    filepath = 'data/Seneca_Lake Data.xlsx'

    # Override with command line: python ess_v3_seneca.py path/to/file.xlsx
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    print("="*60)
    print("  ESS vs Baseline - Seneca Lake validation")
    print("  USGS NWIS data | pH estimated | SRP placeholder")
    print("  3.0 ug/L threshold | convergence features")
    print("="*60)

    run_pipeline(filepath)