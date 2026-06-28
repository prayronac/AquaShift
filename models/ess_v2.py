"""
ESS Model Comparison Pipeline - v2
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

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
Data: NOAA GLERL/CIGLR WE8 buoy, accession 0194301, 2014-2018
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
import os
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
# v2 uses the full Benson & Krause 1984 equation with natural log terms,
# which is the standard reference for dissolved oxygen saturation.
#
# v1 (INACCURATE):
#   return (14.62 - 0.3898 * T_c + 0.006969 * T_c**2 - 5.897e-5 * T_c**3)
#
# v2 (FIXED):
def O2_saturation(T_K):
    """Dissolved O2 saturation (mg/L). Benson & Krause 1984.
    Full equation using natural log terms for accurate T dependence."""
    T_K = np.clip(T_K, 273.16, None)  # guard against log issues at 0K
    ln_DO = (-139.34411 + (1.575701e5 / T_K) - (6.642308e7 / T_K**2)
             + (1.243800e10 / T_K**3) - (8.621949e11 / T_K**4))
    return np.exp(ln_DO)
# !!!! PROBLEM FIXED AREA !!!!


# =============================================================================
# 2. DERIVED CHEMICAL VARIABLES
# =============================================================================

def compute_derived_variables(df):
    """
    From raw sensor data, compute intermediate chemical variables.

    Required columns:
        temp_c, do_mgl, ph, srp_ugl, pc_rfu, chla_rfu, cond_uscm
    """
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])

    # --- Carbonate system ---
    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)

    # !!!! PROBLEM FIXED AREA !!!!
    # v1 used 0.5 * conductivity which gave alkalinity ~10x too low for
    # Lake Erie. Western Lake Erie alkalinity is typically 80-110 mg/L as
    # CaCO3, which is ~1600-2200 ueq/L. With conductivity ~300-500 uS/cm,
    # the correct empirical relationship for Great Lakes freshwater is
    # approximately 5.0 * conductivity (ueq/L), not 0.5.
    # This error propagated into ALL carbonate species (HCO3, CO3, CO2),
    # pCO2, and every Q/K ratio that uses them (RXN 01, 02, 03, 06).
    #
    # v1 (WRONG):  alk_mol = (0.5 * df['cond_uscm']) * 1e-6
    # v2 (FIXED):
    alk_mol = (5.0 * df['cond_uscm']) * 1e-6  # convert ueq/L to mol/L
    # !!!! PROBLEM FIXED AREA !!!!

    # From alkalinity and pH, solve for DIC components
    alpha1 = K1 * H / (H**2 + K1 * H + K1 * K2)  # fraction as HCO3-
    alpha2 = K1 * K2 / (H**2 + K1 * H + K1 * K2)  # fraction as CO3^2-

    # Alkalinity = [HCO3-] + 2[CO3^2-] + [OH-] - [H+]
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

    # --- Phosphate speciation ---
    Ka1 = phosphate_Ka1(T_K)
    Ka2p = phosphate_Ka2(T_K)
    Ka3 = phosphate_Ka3(T_K)

    P_total = (df['srp_ugl'] * 1e-6) / 30.97
    P_total = np.clip(P_total, 1e-12, None)

    denom_p = (H**3 + Ka1 * H**2 + Ka1 * Ka2p * H + Ka1 * Ka2p * Ka3)
    df['h2po4_mol'] = P_total * (Ka1 * H**2) / denom_p
    df['hpo4_mol'] = P_total * (Ka1 * Ka2p * H) / denom_p

    # --- Nitrogen ---
    df['n2_sat_mol'] = N2_saturation(T_K)

    # --- Ammonium speciation ---
    pKa_nh4 = 0.09018 + 2729.92 / T_K  # Emerson et al. 1975
    Ka_nh4 = 10**(-pKa_nh4)
    df['nh3_fraction'] = Ka_nh4 / (Ka_nh4 + H)

    # --- Oxygen ---
    df['o2_sat_mgl'] = O2_saturation(T_K)
    df['do_mol'] = (df['do_mgl'] / 32.0) * 1e-3
    df['o2_sat_mol'] = (df['o2_sat_mgl'] / 32.0) * 1e-3

    df['Kw_val'] = Kw(T_K)
    df['KH_val'] = KH

    return df


# =============================================================================
# 3. REACTION Q/K RATIOS - v2 (5 reactions)
# =============================================================================

def compute_qk_ratios(df):
    """
    Compute Q/K for 5 reactions.
    RXN 04 removed in v2 (see PROBLEM FIXED AREA below).
    """
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])
    co2 = np.clip(df['co2_aq_mol'], 1e-12, None)
    do = np.clip(df['do_mol'], 1e-12, None)

    # ------------------------------------------------------------------
    # RXN 01: Photosynthesis
    # 6CO2 + 6H2O -> C6H12O6 + 6O2
    # Q = [O2]^6 / [CO2]^6
    # Low Q = substrates available = favorable for bloom growth
    # ------------------------------------------------------------------
    Q_01 = (do**6) / (co2**6)
    df['qk_rxn01'] = np.log10(1.0 / Q_01)

    # ------------------------------------------------------------------
    # RXN 02: Carbonate equilibrium
    # CO2(aq) + H2O <-> HCO3- + H+
    # Q/K > 1 means shifted toward bicarbonate (CO2 consumed)
    # ------------------------------------------------------------------
    K1 = carbonate_K1(T_K)
    Q_02 = (df['hco3_mol'] * H) / co2
    df['qk_rxn02'] = Q_02 / K1

    # ------------------------------------------------------------------
    # RXN 03: Phosphate speciation
    # H2PO4- <-> HPO4^2- + H+
    # ------------------------------------------------------------------
    Ka2 = phosphate_Ka2(T_K)
    h2po4 = np.clip(df['h2po4_mol'], 1e-15, None)
    Q_03 = (df['hpo4_mol'] * H) / h2po4
    df['qk_rxn03'] = Q_03 / Ka2

    # !!!! PROBLEM FIXED AREA !!!!
    # RXN 04 REMOVED in v2.
    #
    # v1 had:
    #   Ks_n2 = 0.5 * df['n2_sat_mol']
    #   n2_actual = df['n2_sat_mol'] * 0.95
    #   df['qk_rxn04'] = n2_actual / Ks_n2
    #
    # This always produced 0.95 / 0.5 = 1.9 for every single row because
    # there is no dissolved N2 sensor on the WE8 buoy. A constant feature
    # contributes zero discriminative power to the model. It cannot help
    # distinguish bloom from non-bloom conditions.
    # !!!! PROBLEM FIXED AREA !!!!

    # ------------------------------------------------------------------
    # RXN 05: Phosphate uptake (Monod kinetics)
    # [SRP] / Ks -- > 1 = P-replete, < 1 = P-limited
    # ------------------------------------------------------------------
    Ks_srp = 5.0  # ug P/L
    df['qk_rxn05'] = df['srp_ugl'] / Ks_srp

    # ------------------------------------------------------------------
    # RXN 06: Biomass synthesis (Redfield)
    # Substrate availability indicator
    # ------------------------------------------------------------------
    hpo4 = np.clip(df['hpo4_mol'], 1e-15, None)
    df['qk_rxn06'] = np.log10(co2 * hpo4) * -1

    return df


# =============================================================================
# 4. CONVERGENCE FEATURES - NEW in v2
#    This is the core ESS hypothesis: synchronized multi-reaction shifts
#    predict blooms, not individual Q/K values alone.
# =============================================================================

def compute_convergence_features(df):
    """
    Classify each reaction into Favorable(1) / Neutral(0) / Unfavorable(-1),
    then compute features that capture how many reactions are simultaneously
    favorable. This tests the actual ESS hypothesis.
    """
    rxn_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']
    n_rxns = len(rxn_cols)

    # --- Classify each reaction ---

    # RXN 01 (photosynthesis): high = more CO2 substrate available
    p33_01 = df['qk_rxn01'].quantile(0.33)
    p66_01 = df['qk_rxn01'].quantile(0.66)
    df['state_rxn01'] = np.where(df['qk_rxn01'] > p66_01, 1,
                         np.where(df['qk_rxn01'] < p33_01, -1, 0))

    # RXN 02 (carbonate): Q/K > 1.2 = CO2 consumed by biology
    df['state_rxn02'] = np.where(df['qk_rxn02'] > 1.2, 1,
                         np.where(df['qk_rxn02'] < 0.8, -1, 0))

    # RXN 03 (phosphate speciation): shifted = bioavailable form
    df['state_rxn03'] = np.where(df['qk_rxn03'] > 1.2, 1,
                         np.where(df['qk_rxn03'] < 0.8, -1, 0))

    # RXN 05 (P uptake): [SRP]/Ks > 1 = P-replete
    df['state_rxn05'] = np.where(df['qk_rxn05'] > 1.0, 1,
                         np.where(df['qk_rxn05'] < 0.5, -1, 0))

    # RXN 06 (biomass synthesis): high = substrate available
    p33_06 = df['qk_rxn06'].quantile(0.33)
    p66_06 = df['qk_rxn06'].quantile(0.66)
    df['state_rxn06'] = np.where(df['qk_rxn06'] > p66_06, 1,
                         np.where(df['qk_rxn06'] < p33_06, -1, 0))

    # --- Convergence features ---
    state_cols = ['state_rxn01', 'state_rxn02', 'state_rxn03',
                  'state_rxn05', 'state_rxn06']

    # How many reactions are favorable right now?
    df['n_favorable'] = (df[state_cols] == 1).sum(axis=1)
    df['frac_favorable'] = df['n_favorable'] / n_rxns

    # Are ALL reactions favorable? (the full convergence signal)
    df['all_favorable'] = (df['n_favorable'] == n_rxns).astype(int)

    # How many are unfavorable?
    df['n_unfavorable'] = (df[state_cols] == -1).sum(axis=1)

    # Overall direction score (-5 to +5)
    df['convergence_score'] = df[state_cols].sum(axis=1)

    # Product of sigmoid-mapped favorabilities (only high when ALL align)
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
# 5. ECOLOGICAL RATIOS - v2
# =============================================================================

# !!!! PROBLEM FIXED AREA !!!!
# v1 included pc_chl_ratio = phycocyanin / chlorophyll-a as an ESS feature.
# This was removed because phycocyanin defines the bloom label (sigmoid of
# pc_rfu). Using it as an input feature creates circularity: the model
# predicts phycocyanin from phycocyanin. It is not data leakage in the
# strict ML sense (past data predicting future), but it bypasses the
# chemistry entirely and wins via autocorrelation.
#
# v1 (REMOVED):  df['pc_chl_ratio'] = df['pc_rfu'] / chla
# v2 (KEPT):     N:P ratio only
# !!!! PROBLEM FIXED AREA !!!!

def compute_ecological_ratios(df):
    """N:P ratio only. PC:Chl removed to avoid circularity with bloom label."""
    if 'nitrate_mgl' in df.columns:
        N_mol = (df['nitrate_mgl'] / 62.004) * 1e-3
    else:
        N_mol = (df['cond_uscm'] * 0.002) * 1e-3 / 14.007

    P_mol = (df['srp_ugl'] * 1e-6) / 30.97
    P_mol = np.clip(P_mol, 1e-12, None)
    df['np_ratio'] = N_mol / P_mol

    return df


# =============================================================================
# 6. BLOOM LABELING
# =============================================================================

def label_blooms(df):
    """Label bloom/no-bloom using phycocyanin threshold of 5 RFU.
    Based on environmental monitoring benchmark for water treatment action."""
    df['bloom_label'] = (df['pc_rfu'] >= 5.0).astype(int)
    return df


# =============================================================================
# 7. DAILY AGGREGATION - NEW in v2
# =============================================================================

# !!!! PROBLEM FIXED AREA !!!!
# v1 ran sliding windows directly on 15-minute data, producing ~129,000
# windows where neighboring windows overlapped by 19 days 23 hours 45 min.
# This meant train and test sets in cross-validation contained nearly
# identical data, making metrics artificially stable (tiny std like +/-0.004)
# and inflated. v2 aggregates to daily means FIRST, then builds windows,
# giving ~500 windows which is the correct sample size for this dataset.
# !!!! PROBLEM FIXED AREA !!!!

def aggregate_to_daily(df):
    """Aggregate sub-daily (15-min) data to daily means before windowing."""
    df['date'] = df['timestamp'].dt.date

    exclude_cols = ['timestamp', 'date']
    agg_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    agg_dict = {col: 'mean' for col in agg_cols}
    if 'bloom_label' in agg_dict:
        agg_dict['bloom_label'] = 'max'  # bloom if ANY reading that day

    daily = df.groupby('date').agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    print(f"  Aggregated {len(df)} sub-daily rows -> {len(daily)} daily rows")
    return daily


# =============================================================================
# 8. SLIDING WINDOWS - v2
# =============================================================================

def build_sliding_windows(df, input_days=20, target_days=5, step=1):
    """
    Build sliding windows on DAILY data.

    Baseline: 5 raw abiotic features (mean + std = 10 columns)
    ESS: 5 Q/K ratios + N:P + convergence features (mean + std + summaries)
    """
    # !!!! PROBLEM FIXED AREA !!!!
    # v1 baseline included 'pc_rfu' and 'chla_rfu' as raw features.
    # Since bloom_label is defined from pc_rfu via sigmoid, including
    # phycocyanin as a feature let the model detect ongoing blooms via
    # autocorrelation rather than predict future blooms from chemistry.
    # v2 uses only abiotic sensor measurements in both paths.
    #
    # v1: ['temp_c','do_mgl','ph','srp_ugl','pc_rfu','chla_rfu','cond_uscm']
    # v2: ['temp_c','do_mgl','ph','srp_ugl','cond_uscm']
    feature_cols_baseline = [
        'temp_c', 'do_mgl', 'ph', 'srp_ugl', 'cond_uscm'
    ]
    # !!!! PROBLEM FIXED AREA !!!!

    # Individual Q/K ratios (5 reactions, no RXN 04)
    qk_cols = ['qk_rxn01', 'qk_rxn02', 'qk_rxn03', 'qk_rxn05', 'qk_rxn06']

    # Convergence features computed at daily level
    conv_cols = ['n_favorable', 'frac_favorable', 'convergence_score',
                 'favorability_product']

    feature_cols_ess = qk_cols + ['np_ratio'] + conv_cols

    X_baseline, X_ess, y = [], [], []
    total_len = len(df)

    for i in range(0, total_len - input_days - target_days + 1, step):
        input_window = df.iloc[i:i + input_days]
        target_window = df.iloc[i + input_days:i + input_days + target_days]

        # --- Baseline features: mean and std ---
        base_feats = []
        for col in feature_cols_baseline:
            base_feats.append(input_window[col].mean())
            base_feats.append(input_window[col].std())

        # --- ESS features: mean and std ---
        ess_feats = []
        for col in feature_cols_ess:
            ess_feats.append(input_window[col].mean())
            ess_feats.append(input_window[col].std())

        # --- Window-level convergence summaries (ESS only) ---

        # Max simultaneous favorable reactions in the window
        ess_feats.append(input_window['n_favorable'].max())

        # Days where ALL reactions were favorable
        ess_feats.append(input_window['all_favorable'].sum())

        # Longest consecutive run of full convergence
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

        # Trend: is convergence increasing over the window?
        n_fav = input_window['n_favorable'].values
        if len(n_fav) > 1:
            x_time = np.arange(len(n_fav))
            slope = np.polyfit(x_time, n_fav, 1)[0]
        else:
            slope = 0.0
        ess_feats.append(slope)

        # Peak favorability product in window
        ess_feats.append(input_window['favorability_product'].max())

        # --- Target label ---
        label = int(target_window['bloom_label'].max())

        X_baseline.append(base_feats)
        X_ess.append(ess_feats)
        y.append(label)

    # --- Column names ---
    base_names = []
    for col in feature_cols_baseline:
        base_names.extend([f'{col}_mean', f'{col}_std'])

    ess_names = []
    for col in feature_cols_ess:
        ess_names.extend([f'{col}_mean', f'{col}_std'])
    ess_names.extend([
        'max_n_favorable',
        'days_all_favorable',
        'max_consec_all_favorable',
        'convergence_trend',
        'peak_favorability_product',
    ])

    X_baseline = pd.DataFrame(X_baseline, columns=base_names)
    X_ess = pd.DataFrame(X_ess, columns=ess_names)
    y = np.array(y)

    return X_baseline, X_ess, y


# =============================================================================
# 9. MODEL EVALUATION (unchanged from v1)
# =============================================================================

def evaluate_model(X, y, model_name, n_splits=5):
    """Stratified k-fold CV with logistic regression."""
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
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
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
        'f1': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'prauc': np.mean(prauc_scores),
        'prauc_std': np.std(prauc_scores),
        'mcc': np.mean(mcc_scores),
        'mcc_std': np.std(mcc_scores),
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

    return coefs


# =============================================================================
# 10. EXCEL EXPORT
# =============================================================================

def export_to_excel(
    results_baseline, results_ess,
    n_daily_pos, n_daily_neg,
    n_win_pos, n_win_neg,
    coefs_baseline, coefs_ess,
    output_path='results/ess_results_v2.xlsx'
):
    """
    Append one run's results to the Excel workbook.
    Creates the file and sheets on first run; appends rows on subsequent runs.

    Sheets:
      Run Log          - one row per run: all metrics + class counts
      Confusion Matrix - TN/FP/FN/TP + precision/recall for both models
      Feature Importance (Baseline) - coefficients per run
      Feature Importance (ESS)      - coefficients per run
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    run_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_daily_total = n_daily_pos + n_daily_neg
    n_win_total = n_win_pos + n_win_neg

    run_row = {
        'timestamp': run_ts,
        'daily_bloom': n_daily_pos,
        'daily_no_bloom': n_daily_neg,
        'daily_total': n_daily_total,
        'daily_bloom_pct': round(100 * n_daily_pos / n_daily_total, 1) if n_daily_total else 0,
        'win_positive': n_win_pos,
        'win_negative': n_win_neg,
        'win_total': n_win_total,
        'win_positive_pct': round(100 * n_win_pos / n_win_total, 1) if n_win_total else 0,
        'baseline_f1': round(results_baseline['f1'], 4),
        'baseline_f1_std': round(results_baseline['f1_std'], 4),
        'baseline_prauc': round(results_baseline['prauc'], 4),
        'baseline_prauc_std': round(results_baseline['prauc_std'], 4),
        'baseline_mcc': round(results_baseline['mcc'], 4),
        'baseline_mcc_std': round(results_baseline['mcc_std'], 4),
        'ess_f1': round(results_ess['f1'], 4),
        'ess_f1_std': round(results_ess['f1_std'], 4),
        'ess_prauc': round(results_ess['prauc'], 4),
        'ess_prauc_std': round(results_ess['prauc_std'], 4),
        'ess_mcc': round(results_ess['mcc'], 4),
        'ess_mcc_std': round(results_ess['mcc_std'], 4),
        'f1_delta': round(results_ess['f1'] - results_baseline['f1'], 4),
        'prauc_delta': round(results_ess['prauc'] - results_baseline['prauc'], 4),
        'mcc_delta': round(results_ess['mcc'] - results_baseline['mcc'], 4),
        'winner': 'ESS' if results_ess['f1'] > results_baseline['f1'] else 'Baseline',
    }

    def cm_row(name, cm):
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return {
            'timestamp': run_ts,
            'model': name,
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
        }

    cm_rows = [
        cm_row('Baseline', results_baseline['confusion_matrix']),
        cm_row('ESS',      results_ess['confusion_matrix']),
    ]

    def importance_rows(coefs, model_name):
        rows = []
        for feat, coef in coefs.items():
            rows.append({
                'timestamp': run_ts,
                'model': model_name,
                'feature': feat,
                'coefficient': round(float(coef), 4),
                'abs_coefficient': round(abs(float(coef)), 4),
            })
        return sorted(rows, key=lambda r: r['abs_coefficient'], reverse=True)

    fi_base_rows = importance_rows(coefs_baseline, 'Baseline')
    fi_ess_rows  = importance_rows(coefs_ess, 'ESS')

    sheet_data = {
        'Run Log':                    (pd.DataFrame([run_row]),       list(run_row.keys())),
        'Confusion Matrix':           (pd.DataFrame(cm_rows),         ['timestamp','model','TN','FP','FN','TP','precision','recall']),
        'Feature Importance Baseline':(pd.DataFrame(fi_base_rows),    ['timestamp','model','feature','coefficient','abs_coefficient']),
        'Feature Importance ESS':     (pd.DataFrame(fi_ess_rows),     ['timestamp','model','feature','coefficient','abs_coefficient']),
    }

    if os.path.exists(output_path):
        existing = pd.read_excel(output_path, sheet_name=None)
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            for sheet_name, (new_df, cols) in sheet_data.items():
                if sheet_name in existing:
                    combined = pd.concat([existing[sheet_name], new_df], ignore_index=True)
                else:
                    combined = new_df
                combined[cols].to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            for sheet_name, (new_df, cols) in sheet_data.items():
                new_df[cols].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n  Results appended to: {output_path}")


# =============================================================================
# 11. MAIN PIPELINE - v2
# =============================================================================

def run_pipeline(data_path, phosphate_path=None):
    """
    v2 pipeline:
    load -> derive variables -> Q/K ratios -> convergence features ->
    daily aggregation -> sliding windows -> train/evaluate -> compare
    """
    # --- Load ---
    print("Loading data...")
    df = pd.read_csv(data_path, skiprows=[1, 2])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Merge phosphate if provided ---
    if phosphate_path:
        phos = pd.read_csv(phosphate_path, skiprows=[1, 2])
        phos['timestamp'] = pd.to_datetime(phos['timestamp'])
        phos['date'] = phos['timestamp'].dt.date
        phos = phos[['date', 'phosphate']].dropna(subset=['phosphate'])
        df['date'] = df['timestamp'].dt.date
        df = df.merge(phos, on='date', how='left')
        df['phosphate'] = pd.to_numeric(df['phosphate'], errors='coerce')
        df['phosphate'] = df['phosphate'].ffill()
        df.drop(columns=['date'], inplace=True)

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

    required = ['temp_c', 'do_mgl', 'ph', 'srp_ugl',
                'pc_rfu', 'chla_rfu', 'cond_uscm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    initial_len = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  Rows: {initial_len} -> {len(df)} after dropping NaN")

    # --- Step 1: Derive chemical variables (FIXED alkalinity) ---
    print("Computing derived chemical variables (FIXED alkalinity 0.5->5.0)...")
    df = compute_derived_variables(df)

    # --- Step 2: Q/K ratios (5 reactions, RXN 04 removed) ---
    print("Computing Q/K ratios for 5 reactions (RXN 04 removed)...")
    df = compute_qk_ratios(df)

    # --- Step 3: Convergence features (NEW) ---
    print("Computing convergence features (synchronized reaction shifts)...")
    df = compute_convergence_features(df)

    # --- Step 4: Ecological ratios (PC:Chl removed) ---
    print("Computing N:P ratio (PC:Chl removed)...")
    df = compute_ecological_ratios(df)

    # --- Step 5: Label blooms ---
    print("Labeling bloom events (5 RFU benchmark)...")
    df = label_blooms(df)

    # --- Step 6: Daily aggregation (NEW - fixes 129k window problem) ---
    print("\nAggregating to daily resolution...")
    daily = aggregate_to_daily(df)

    n_pos = int(daily['bloom_label'].sum())
    n_neg = len(daily) - n_pos
    print(f"\n  Daily class distribution:")
    print(f"    Bloom:    {n_pos:4d} ({100*n_pos/len(daily):.1f}%)")
    print(f"    No bloom: {n_neg:4d} ({100*n_neg/len(daily):.1f}%)")

    # --- Step 7: Sliding windows on DAILY data ---
    print("\nBuilding sliding windows (20-day input / 5-day target)...")
    X_baseline, X_ess, y = build_sliding_windows(daily)

    n_pos_w = int(y.sum())
    n_neg_w = len(y) - n_pos_w
    print(f"  Total windows:    {len(y)}")
    print(f"  Positive windows: {n_pos_w} ({100*n_pos_w/len(y):.1f}%)")
    print(f"  Negative windows: {n_neg_w} ({100*n_neg_w/len(y):.1f}%)")
    print(f"\n  Baseline features: {X_baseline.shape[1]}")
    print(f"  ESS features:      {X_ess.shape[1]}")

    if n_pos_w < 10:
        print("\nWARNING: Very few positive samples. Results may be unreliable.")

    # --- Step 8: Evaluate ---
    print("\n" + "="*60)
    print("  RUNNING MODEL COMPARISON - v2")
    print("="*60)

    results_baseline = evaluate_model(
        X_baseline, y, "BASELINE: Logistic Regression on Raw Features"
    )
    results_ess = evaluate_model(
        X_ess, y, "ESS: Logistic Regression on Q/K + Convergence Features"
    )

    # --- Step 9: Feature importance ---
    coefs_baseline = print_feature_importance(X_baseline, y, X_baseline.columns, "Baseline")
    coefs_ess      = print_feature_importance(X_ess, y, X_ess.columns, "ESS")

    # --- Step 10: Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON - v2")
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

    # --- Step 11: Export to Excel ---
    export_to_excel(
        results_baseline, results_ess,
        n_pos, n_neg,
        n_pos_w, n_neg_w,
        coefs_baseline, coefs_ess,
    )

    return results_baseline, results_ess


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        phosphate_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        data_path = 'data/2016-2018 Lake Erie Water Data/2017 bloom dataset.csv'
        phosphate_path = 'data/2016-2018 Lake Erie Water Data/WE08_2017_annual_summary_phosphate.csv'

    print("="*60)
    print("  ESS vs Baseline Model Comparison - v2")
    print("  FIXES: alkalinity 10x, O2 sat, RXN04 removed,")
    print("  daily aggregation, convergence features,")
    print("  PC/Chl removed from features")
    print("="*60)

    run_pipeline(data_path, phosphate_path)