"""
ESS Model Comparison Pipeline - Columbia River Edition
Baseline (raw features) vs ESS (Q/K-derived features) using logistic regression.
Same model, same data, different input representations.

COLUMBIA RIVER ADAPTATIONS:
  - 3 observable reactions (RXN01 Photosynthesis, RXN02 Carbonate, RXN03_CR DO Supersaturation)
  - Phosphate-dependent nodes removed (RXN03, RXN05 from Lake Erie version)
  - Alkalinity multiplier recalibrated: 9.0 (Columbia River ~68 mg/L CaCO3 at 150 uS/cm)
  - No conductance unit conversion (Columbia data already in uS/cm)
  - Data loader reads xlsx directly (USGS Water Data for the Nation format)
  - Bloom threshold: 1.3 RFU (same as Lake Erie v3)
  - RXN03_CR: DO supersaturation ratio (DO/DO_sat) as net community production proxy
  - Supports 2023 Johnson Island (bloom site) + 2025 Horn Rapids (negative control)

WHY 3 REACTIONS IS SUFFICIENT:
  The Columbia River is a fast-flowing, clear-water, low-nutrient system where
  cyanobacterial blooms should not occur. If ESS detects transient multi-reaction
  convergence windows preceding bloom events with only 3 observable nodes, this
  demonstrates that synchronized equilibrium shifts -- not single-variable thresholds
  -- are the mechanism enabling blooms in otherwise hostile environments.

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
Data: USGS stations 1247351910 (Johnson Island) and 1247351985 (Horn Rapids)
"""

import numpy as np
import pandas as pd
from datetime import datetime
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
# 1. THERMODYNAMIC CONSTANTS (unchanged from Lake Erie version)
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

def O2_saturation(T_K):
    """Dissolved O2 saturation (mg/L). Benson & Krause 1984."""
    T_K = np.clip(T_K, 273.16, None)
    ln_DO = (-139.34411 + (1.575701e5 / T_K) - (6.642308e7 / T_K**2)
             + (1.243800e10 / T_K**3) - (8.621949e11 / T_K**4))
    return np.exp(ln_DO)


# =============================================================================
# 2. DERIVED CHEMICAL VARIABLES (Columbia River calibration)
# =============================================================================

def compute_derived_variables(df):
    """Compute intermediate chemical variables from raw sensor data.
    
    Columbia River calibration:
      - Alkalinity multiplier: 9.0 (vs 5.0 for Lake Erie)
        Columbia River at Priest Rapids: ~55-80 mg/L CaCO3
        At median conductance 151 uS/cm: 9.0 * 151 = 1359 ueq/L = 68 mg/L CaCO3
      - No phosphate speciation (no sensor data)
      - DO supersaturation computed for RXN03_CR
    """
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])

    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)

    # Columbia River alkalinity: 9.0 * conductivity (vs 5.0 for Lake Erie)
    # Yields ~68 mg/L CaCO3 at median conductance, matching published values
    alk_mol = (9.0 * df['cond_uscm']) * 1e-6

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

    # DO saturation for RXN03_CR
    df['o2_sat_mgl'] = O2_saturation(T_K)
    df['do_mol'] = (df['do_mgl'] / 32.0) * 1e-3
    df['o2_sat_mol'] = (df['o2_sat_mgl'] / 32.0) * 1e-3
    df['do_supersaturation'] = df['do_mgl'] / df['o2_sat_mgl']

    df['Kw_val'] = Kw(T_K)
    df['KH_val'] = KH

    return df


# =============================================================================
# 3. REACTION Q/K RATIOS (3 reactions for Columbia River)
# =============================================================================

def compute_qk_ratios(df):
    """Compute Q/K for 3 observable reactions.
    
    RXN01: Photosynthesis (DO production relative to CO2 consumption)
    RXN02: Carbonate equilibrium (CO2 <-> HCO3- speciation)
    RXN03_CR: DO supersaturation (net community production indicator)
    
    Phosphate-dependent reactions removed:
      - Lake Erie RXN03 (phosphate speciation) -- no sensor
      - Lake Erie RXN05 (phosphate Monod uptake) -- no sensor
      - Lake Erie RXN06 (Redfield CO2*HPO4) -- no sensor
    """
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])
    co2 = np.clip(df['co2_aq_mol'], 1e-12, None)
    do = np.clip(df['do_mol'], 1e-12, None)

    # RXN 01: Photosynthesis
    # High Q/K = DO accumulation outpacing CO2, net autotrophy
    Q_01 = (do**6) / (co2**6)
    df['qk_rxn01'] = np.log10(1.0 / Q_01)

    # RXN 02: Carbonate equilibrium
    # Q/K > 1 = system shifted toward HCO3- (CO2 being consumed)
    K1 = carbonate_K1(T_K)
    Q_02 = (df['hco3_mol'] * H) / co2
    df['qk_rxn02'] = Q_02 / K1

    # RXN 03_CR: DO supersaturation (Columbia River specific)
    # DO/DO_sat > 1 = supersaturated = net photosynthetic production
    # DO/DO_sat < 1 = undersaturated = net respiration
    # This replaces phosphate-dependent nodes as a direct measure of
    # biological activity that is thermodynamically grounded (Henry's Law
    # equilibrium for O2 gas exchange)
    df['qk_rxn03_cr'] = df['do_supersaturation']

    return df


# =============================================================================
# 4. CONVERGENCE FEATURES (3-reaction version)
# =============================================================================

def compute_convergence_features(df):
    """Classify 3 reactions into Favorable/Neutral/Unfavorable states,
    then compute synchronized convergence features.
    
    State thresholds:
      RXN01: Quantile-based (same as Lake Erie -- scale-dependent)
      RXN02: Physical threshold (Q/K > 1.2 = CO2 depletion, favorable for blooms)
      RXN03_CR: Physical threshold (DO/DOsat > 1.05 = supersaturated = active production)
    """
    rxn_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03_cr']
    n_rxns = len(rxn_cols)

    # RXN01: Quantile-based (log-scale values)
    p33_01 = df['qk_rxn01'].quantile(0.33)
    p66_01 = df['qk_rxn01'].quantile(0.66)
    df['state_rxn01'] = np.where(df['qk_rxn01'] > p66_01, 1,
                         np.where(df['qk_rxn01'] < p33_01, -1, 0))

    # RXN02: Physical thresholds (same as Lake Erie)
    df['state_rxn02'] = np.where(df['qk_rxn02'] > 1.2, 1,
                         np.where(df['qk_rxn02'] < 0.8, -1, 0))

    # RXN03_CR: DO supersaturation thresholds
    # > 1.05 = supersaturated (active photosynthesis, favorable)
    # < 0.95 = undersaturated (net respiration, unfavorable)
    df['state_rxn03_cr'] = np.where(df['qk_rxn03_cr'] > 1.05, 1,
                            np.where(df['qk_rxn03_cr'] < 0.95, -1, 0))

    state_cols = ['state_rxn01', 'state_rxn02', 'state_rxn03_cr']

    df['n_favorable'] = (df[state_cols] == 1).sum(axis=1)
    df['frac_favorable'] = df['n_favorable'] / n_rxns
    df['all_favorable'] = (df['n_favorable'] == n_rxns).astype(int)
    df['n_unfavorable'] = (df[state_cols] == -1).sum(axis=1)
    df['convergence_score'] = df[state_cols].sum(axis=1)

    # Continuous favorability product (sigmoid transform)
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
# 5. BLOOM LABELING (1.3 RFU threshold -- same as v3)
# =============================================================================

def label_blooms(df):
    """Label bloom/no-bloom using phycocyanin threshold of 1.3 RFU."""
    df['bloom_label'] = (df['pc_rfu'] >= 1.3).astype(int)
    return df


# =============================================================================
# 6. DAILY AGGREGATION
# =============================================================================

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
# 7. DATA LOADING (Columbia River xlsx format)
# =============================================================================

def load_columbia_data(xlsx_paths):
    """Load Columbia River xlsx files and standardize columns.
    
    Accepts either:
      - Columbia_River_FIXED_Blooms.xlsx (clean format, 2023 Johnson Island)
      - Columbia_Datapoints.xlsx (USGS raw format, 2025 Horn Rapids)
    """
    all_dfs = []

    for path in xlsx_paths:
        print(f"\n  Loading {path}...")
        df = pd.read_excel(path)
        print(f"    Raw shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")

        # Detect format and standardize column names
        if 'temperature' in df.columns:
            # Columbia_River_FIXED_Blooms.xlsx format
            col_map = {
                'date': 'timestamp',
                'temperature': 'temp_c',
                'pH': 'ph',
                'dissolved_oxygen_mg_L': 'do_mgl',
                'chlorophyll_RFU': 'chla_rfu',
                'phycocyanin_RFU': 'pc_rfu',
                'conductance_uS_cm': 'cond_uscm',
                'turbidity_FNU': 'turbidity',
            }
            df = df.rename(columns=col_map)
            print(f"    Format: FIXED_Blooms (2023 Johnson Island)")

        elif 'Temperature' in df.columns:
            # Columbia_Datapoints.xlsx format (USGS raw)
            col_map = {
                'date': 'timestamp',
                'Temperature': 'temp_c',
                'pH': 'ph',
                'Dissolved Oxygen': 'do_mgl',
                'Chlorophyll relative fluorescence (fChl)': 'chla_rfu',
                'Phycocyanin relative fluorescence (fPC)': 'pc_rfu',
                'Specific conductance, water, unfiltered': 'cond_uscm',
                'Turbidity': 'turbidity',
            }
            df = df.rename(columns=col_map)
            print(f"    Format: USGS raw datapoints")
        else:
            print(f"    WARNING: Unrecognized format. Columns: {list(df.columns)}")
            continue

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # Ensure numeric types
        numeric_cols = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'chla_rfu',
                        'cond_uscm', 'turbidity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Strip timezone if present
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df['source_file'] = path
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    print(f"\n  Combined: {len(combined)} total rows")
    print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    return combined


# =============================================================================
# 8. CONTINUOUS SEGMENT DETECTION
# =============================================================================

def identify_continuous_segments(daily_df, max_gap_days=5):
    """Split daily data into segments where no gap exceeds max_gap_days."""
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
        n_bloom = int(seg['bloom_label'].sum())
        print(f"    Segment {i+1}: {seg_dates.min().date()} to {seg_dates.max().date()} "
              f"({len(seg)} days, {n_bloom} bloom days)")

    return segments


# =============================================================================
# 9. SLIDING WINDOWS (segment-aware, 3-reaction version)
# =============================================================================

def build_sliding_windows(segments, input_days=20, target_days=5, step=1):
    """Build sliding windows from multiple segments. Never crosses gaps.
    
    Baseline features: 4 abiotic sensor readings (no phosphate, no biology)
    ESS features: 3 Q/K ratios + convergence metrics
    """

    # Baseline: raw abiotic measurements only
    feature_cols_baseline = ['temp_c', 'do_mgl', 'ph', 'cond_uscm']

    # ESS: Q/K ratios + convergence
    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03_cr']
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score',
                 'favorability_product']
    feature_cols_ess = qk_cols + conv_cols

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

            # Baseline: mean + std of raw features
            base_feats = []
            for col in feature_cols_baseline:
                base_feats.append(input_window[col].mean())
                base_feats.append(input_window[col].std())

            # ESS: mean + std of Q/K and convergence features
            ess_feats = []
            for col in feature_cols_ess:
                ess_feats.append(input_window[col].mean())
                ess_feats.append(input_window[col].std())

            # Extra convergence summary features
            ess_feats.append(input_window['n_favorable'].max())
            ess_feats.append(input_window['all_favorable'].sum())

            # Max consecutive days with all reactions favorable
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

            # Convergence trend (slope of n_favorable over window)
            n_fav = input_window['n_favorable'].values
            if len(n_fav) > 1:
                x_time = np.arange(len(n_fav))
                slope = np.polyfit(x_time, n_fav, 1)[0]
            else:
                slope = 0.0
            ess_feats.append(slope)

            # Peak favorability product
            ess_feats.append(input_window['favorability_product'].max())

            # Target label: any bloom in next 5 days
            label = int(target_window['bloom_label'].max())

            X_baseline.append(base_feats)
            X_ess.append(ess_feats)
            y.append(label)
            n_windows += 1

        print(f"    Segment {seg_idx+1}: {n_windows} windows")

    # Build column names
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
# 10. MODEL EVALUATION
# =============================================================================

def evaluate_model(X, y, model_name, n_splits=5):
    """Stratified k-fold CV with logistic regression."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    # Adjust folds if too few positive samples
    actual_splits = min(n_splits, n_pos, n_neg)
    if actual_splits < 2:
        print(f"\n  WARNING: Only {n_pos} positive / {n_neg} negative samples.")
        print(f"  Cannot run cross-validation. Fitting on full data instead.")

        scaler = StandardScaler()
        X_s = np.nan_to_num(scaler.fit_transform(X), nan=0.0, posinf=10, neginf=-10)
        model = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='lbfgs', random_state=42
        )
        model.fit(X_s, y)
        y_pred = model.predict(X_s)
        y_prob = model.predict_proba(X_s)[:, 1]

        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        precision, recall, _ = precision_recall_curve(y, y_prob)
        prauc = auc(recall, precision)
        cm = confusion_matrix(y, y_pred)

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        print(f"  F1 Score:     {f1:.4f} (full-data fit, no CV)")
        print(f"  PR-AUC:       {prauc:.4f}")
        print(f"  MCC:          {mcc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
        print(f"\n  Classification Report:")
        print(classification_report(y, y_pred, target_names=['No bloom', 'Bloom']))

        return {
            'f1': f1, 'f1_std': 0.0,
            'prauc': prauc, 'prauc_std': 0.0,
            'mcc': mcc, 'mcc_std': 0.0,
            'confusion_matrix': cm
        }

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
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
# 11. DAILY PREDICTIONS
# =============================================================================

def add_daily_predictions(daily, X_baseline, X_ess, y, input_days=20):
    """Fit final models on all window data and assign rolling per-day predictions."""
    feature_cols_baseline = ['temp_c', 'do_mgl', 'ph', 'cond_uscm']
    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03_cr']
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score', 'favorability_product']
    feature_cols_ess = qk_cols + conv_cols

    scaler_b = StandardScaler()
    X_b_s = np.nan_to_num(scaler_b.fit_transform(X_baseline), nan=0.0, posinf=10, neginf=-10)
    model_b = LogisticRegression(class_weight='balanced', max_iter=1000,
                                  solver='lbfgs', random_state=42)
    model_b.fit(X_b_s, y)

    scaler_e = StandardScaler()
    X_e_s = np.nan_to_num(scaler_e.fit_transform(X_ess), nan=0.0, posinf=10, neginf=-10)
    model_e = LogisticRegression(class_weight='balanced', max_iter=1000,
                                  solver='lbfgs', random_state=42)
    model_e.fit(X_e_s, y)

    daily = daily.copy()
    daily['baseline_pred'] = np.nan
    daily['ess_pred'] = np.nan

    for i in range(input_days, len(daily)):
        window = daily.iloc[i - input_days:i]

        base_feats = []
        for col in feature_cols_baseline:
            base_feats.append(window[col].mean())
            base_feats.append(window[col].std())

        ess_feats = []
        for col in feature_cols_ess:
            ess_feats.append(window[col].mean())
            ess_feats.append(window[col].std())

        ess_feats.append(window['n_favorable'].max())
        ess_feats.append(window['all_favorable'].sum())

        af = window['all_favorable'].values
        max_consec = current_run = 0
        for v in af:
            if v == 1:
                current_run += 1
                max_consec = max(max_consec, current_run)
            else:
                current_run = 0
        ess_feats.append(max_consec)

        n_fav = window['n_favorable'].values
        slope = np.polyfit(np.arange(len(n_fav)), n_fav, 1)[0] if len(n_fav) > 1 else 0.0
        ess_feats.append(slope)
        ess_feats.append(window['favorability_product'].max())

        base_x = np.nan_to_num(scaler_b.transform([base_feats]), nan=0.0, posinf=10, neginf=-10)
        ess_x = np.nan_to_num(scaler_e.transform([ess_feats]), nan=0.0, posinf=10, neginf=-10)

        daily.at[daily.index[i], 'baseline_pred'] = model_b.predict_proba(base_x)[0, 1]
        daily.at[daily.index[i], 'ess_pred'] = model_e.predict_proba(ess_x)[0, 1]

    return daily


# =============================================================================
# 12. EXPORT
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
# 12. MAIN PIPELINE - Columbia River
# =============================================================================

def run_pipeline(xlsx_paths):
    """Full Columbia River pipeline."""

    # --- Load ---
    print("Loading Columbia River datasets...")
    df = load_columbia_data(xlsx_paths)

    # --- Validate required columns ---
    required = ['temp_c', 'do_mgl', 'ph', 'pc_rfu', 'cond_uscm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        return

    # Drop rows missing required abiotic + target data
    initial_len = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  Rows: {initial_len} -> {len(df)} after dropping NaN in required columns")

    # --- Chemistry pipeline ---
    print("\nComputing derived chemical variables (Columbia River alkalinity)...")
    df = compute_derived_variables(df)

    print("Computing Q/K ratios for 3 reactions...")
    df = compute_qk_ratios(df)

    print("Computing convergence features (3-reaction)...")
    df = compute_convergence_features(df)

    print("Labeling bloom events (1.3 RFU threshold)...")
    df = label_blooms(df)

    # --- Daily aggregation ---
    print("\nAggregating to daily resolution...")
    daily = aggregate_to_daily(df)

    n_pos = int(daily['bloom_label'].sum())
    n_neg = len(daily) - n_pos
    print(f"\n  Daily class distribution:")
    print(f"    Bloom:    {n_pos:4d} ({100*n_pos/len(daily):.1f}%)")
    print(f"    No bloom: {n_neg:4d} ({100*n_neg/len(daily):.1f}%)")

    if n_pos == 0:
        print("\n  WARNING: No bloom days detected. Check threshold and data.")
        print("  If this is a negative control dataset, this is expected.")

    # --- Segment and window ---
    print("\nIdentifying continuous segments (max 5-day gap)...")
    segments = identify_continuous_segments(daily)

    print("\nBuilding sliding windows (20-day input / 5-day target)...")
    X_baseline, X_ess, y = build_sliding_windows(segments)

    n_pos_w = int(y.sum())
    n_neg_w = len(y) - n_pos_w
    print(f"\n  Total windows:    {len(y)}")
    print(f"  Positive windows: {n_pos_w} ({100*n_pos_w/len(y) if len(y) > 0 else 0:.1f}%)")
    print(f"  Negative windows: {n_neg_w} ({100*n_neg_w/len(y) if len(y) > 0 else 0:.1f}%)")
    print(f"  Baseline features: {X_baseline.shape[1]}")
    print(f"  ESS features:      {X_ess.shape[1]}")

    if n_pos_w < 2:
        print("\n  CANNOT EVALUATE: Need at least 2 positive windows for classification.")
        print("  This dataset may be a negative control (no blooms).")
        return None, None

    if n_pos_w < 10:
        print("\n  WARNING: Very few positive samples. Results may be unreliable.")

    # --- Evaluate ---
    print("\n" + "="*60)
    print("  RUNNING MODEL COMPARISON - Columbia River")
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
    print(f"  SUMMARY COMPARISON - Columbia River")
    print(f"{'='*60}")
    print(f"  3 reactions: Photosynthesis | Carbonate Eq | DO Supersaturation")
    print(f"  No phosphate data (RXN03, RXN05, RXN06 from Lake Erie removed)")
    print(f"  Alkalinity multiplier: 9.0 (vs 5.0 for Lake Erie)")
    print(f"  Bloom threshold: 1.3 RFU")
    print(f"")
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

    # --- Export daily values with predictions ---
    print("\nGenerating per-day predictions...")
    daily = add_daily_predictions(daily, X_baseline, X_ess, y)
    export_daily_to_excel(daily)

    return results_baseline, results_ess


# =============================================================================
# ENTRY POINT
# =============================================================================

DATASETS = {
    '2023': [
        'data/Columbia River Points/Columbia_River_FIXED_Blooms.xlsx',
        'data/Columbia River Points/Johnson_Datapoints.xlsx',
    ],
    '2025': [
        'data/Columbia River Points/Columbia Datapoints.xlsx',
    ],
    'all': [
        'data/Columbia River Points/Columbia_River_FIXED_Blooms.xlsx',
        'data/Columbia River Points/Johnson_Datapoints.xlsx',
        'data/Columbia River Points/Columbia Datapoints.xlsx',
    ],
}

if __name__ == '__main__':
    import sys

    # Usage:
    #   python ess_v4_columbia.py           → 2023 only (primary result)
    #   python ess_v4_columbia.py 2025      → 2025 only (negative control)
    #   python ess_v4_columbia.py all       → combined (confounded — for reference only)
    #   python ess_v4_columbia.py path.xlsx → custom file(s)

    mode = sys.argv[1] if len(sys.argv) > 1 else '2023'

    if mode in DATASETS:
        xlsx_paths = DATASETS[mode]
        label = {
            '2023': '2023 Johnson Island  [PRIMARY: within-site bloom prediction]',
            '2025': '2025 Horn Rapids     [NEGATIVE CONTROL: non-bloom validation]',
            'all':  'All stations combined [CONFOUNDED — site identity leaks into features]',
        }[mode]
    else:
        xlsx_paths = sys.argv[1:]
        label = 'Custom files'

    print("="*60)
    print("  ESS vs Baseline - Columbia River Edition")
    print("  3 reactions | 1.3 RFU threshold | convergence features")
    print("  alkalinity recalibrated | no phosphate dependency")
    print("="*60)
    print(f"  Dataset: {label}")
    print("="*60)

    run_pipeline(xlsx_paths)