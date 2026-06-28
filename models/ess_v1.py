"""
ESS Model Comparison Pipeline
Baseline (raw features) vs ESS (Q/K-derived features) using logistic regression.
Same model, same data, different input representations.

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
    """Henry's law constant for CO2. Weiss 1974. Returns mol/(L·atm)."""
    ln_KH = (-60.2409 + 93.4517 * (100.0 / T_K)
             + 23.3585 * np.log(T_K / 100.0))
    return np.exp(ln_KH)

def Kw(T_K):
    """Ion product of water. Harned & Hamer 1933."""
    log_Kw = -4470.99 / T_K + 6.0846 - 0.01706 * T_K
    return 10**log_Kw

def phosphate_Ka1(T_K):
    """H3PO4 -> H2PO4- + H+. Stumm & Morgan 1996 / NIST SRD 46."""
    return 10**(-2.148 + 0.0 * (T_K - 298.15))  # ~7.1e-3, weak T dependence

def phosphate_Ka2(T_K):
    """H2PO4- -> HPO4^2- + H+. Stumm & Morgan 1996."""
    return 10**(-7.198 + 0.0 * (T_K - 298.15))  # ~6.3e-8

def phosphate_Ka3(T_K):
    """HPO4^2- -> PO4^3- + H+. Stumm & Morgan 1996."""
    return 10**(-12.375 + 0.0 * (T_K - 298.15))  # ~4.2e-13

def N2_saturation(T_K):
    """Dissolved N2 saturation concentration (mol/L). Sander 2015.
    Uses solubility at 1 atm, 0.78 atm N2 partial pressure."""
    # Bunsen solubility coefficient approximation
    KH_N2 = 6.1e-4 * np.exp(1300 * (1/T_K - 1/298.15))  # mol/(L·atm)
    return KH_N2 * 0.78  # partial pressure of N2 in atm

def O2_saturation(T_K):
    """Dissolved O2 saturation (mg/L). Benson & Krause 1984."""
    T_c = T_K - 273.15
    return (14.62 - 0.3898 * T_c + 0.006969 * T_c**2 - 5.897e-5 * T_c**3)


# =============================================================================
# 2. DERIVED CHEMICAL VARIABLES (10 variables from raw sensor data)
# =============================================================================

def compute_derived_variables(df):
    """
    From raw sensor data, compute 10 intermediate chemical variables.
    
    Required columns in df:
        temp_c    : water temperature (°C)
        do_mgl    : dissolved oxygen (mg/L)
        ph        : pH
        srp_ugl   : soluble reactive phosphorus (µg P/L, from CycleP)
        pc_rfu    : phycocyanin (RFU, YSI EXO2)
        chla_rfu  : chlorophyll-a (RFU, YSI EXO2)
        cond_uscm : specific conductance (µS/cm)
    
    Returns df with 10 new columns.
    """
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])  # [H+] in mol/L

    # --- Carbonate system ---
    K1 = carbonate_K1(T_K)
    K2 = carbonate_K2(T_K)
    KH = henry_KH(T_K)

    # Total DIC approximation from pH and alkalinity proxy (conductivity)
    # Alkalinity ~ 0.5 * conductivity (µeq/L) for freshwater (approximate)
    alk_mol = (0.5 * df['cond_uscm']) * 1e-6  # convert µeq/L to mol/L

    # From alkalinity and pH, solve for DIC components
    alpha1 = K1 * H / (H**2 + K1 * H + K1 * K2)  # fraction as HCO3-
    alpha2 = K1 * K2 / (H**2 + K1 * H + K1 * K2)  # fraction as CO3^2-

    # Alkalinity ≈ [HCO3-] + 2[CO3^2-] + [OH-] - [H+]
    OH = Kw(T_K) / H
    denom = alpha1 + 2 * alpha2
    denom = np.where(denom > 0, denom, 1e-12)
    DIC = (alk_mol - OH + H) / denom
    DIC = np.clip(DIC, 1e-9, None)

    df['hco3_mol'] = DIC * alpha1          # [HCO3-] mol/L
    df['co3_mol'] = DIC * alpha2            # [CO3^2-] mol/L
    df['co2_aq_mol'] = DIC - df['hco3_mol'] - df['co3_mol']  # [CO2(aq)]
    df['co2_aq_mol'] = np.clip(df['co2_aq_mol'], 1e-12, None)
    df['pco2_uatm'] = (df['co2_aq_mol'] / KH) * 1e6  # pCO2 in µatm

    # --- Phosphate speciation ---
    Ka1 = phosphate_Ka1(T_K)
    Ka2 = phosphate_Ka2(T_K)
    Ka3 = phosphate_Ka3(T_K)

    # Total P from SRP (convert µg P/L to mol/L: MW of P = 30.97)
    P_total = (df['srp_ugl'] * 1e-6) / 30.97  # mol/L
    P_total = np.clip(P_total, 1e-12, None)

    denom_p = (H**3 + Ka1 * H**2 + Ka1 * Ka2 * H + Ka1 * Ka2 * Ka3)
    df['h2po4_mol'] = P_total * (Ka1 * H**2) / denom_p   # [H2PO4-]
    df['hpo4_mol'] = P_total * (Ka1 * Ka2 * H) / denom_p  # [HPO4^2-]

    # --- Nitrogen ---
    df['n2_sat_mol'] = N2_saturation(T_K)  # [N2] saturation

    # --- Ammonium speciation ---
    # pKa for NH4+ -> NH3 + H+: ~9.25 at 25°C, T-dependent
    pKa_nh4 = 0.09018 + 2729.92 / T_K  # Emerson et al. 1975
    Ka_nh4 = 10**(-pKa_nh4)
    df['nh3_fraction'] = Ka_nh4 / (Ka_nh4 + H)  # fraction as NH3 (vs NH4+)

    # --- Oxygen ---
    df['o2_sat_mgl'] = O2_saturation(T_K)
    df['do_mol'] = (df['do_mgl'] / 32.0) * 1e-3  # convert mg/L to mol/L
    df['o2_sat_mol'] = (df['o2_sat_mgl'] / 32.0) * 1e-3

    # --- Kw and KH stored for reaction use ---
    df['Kw_val'] = Kw(T_K)
    df['KH_val'] = KH

    return df


# =============================================================================
# 3. REACTION Q/K RATIO COMPUTATION
# =============================================================================

def compute_qk_ratios(df):
    """
    Compute Q/K (or Q/K', or [S]/Ks) for each of the 6 reactions.
    
    RXN 01: Photosynthesis (Q/K' using apparent equilibrium)
    RXN 02: Carbonate equilibrium (true thermodynamic Q/K)
    RXN 03: Phosphate speciation (true thermodynamic Q/K)
    RXN 04: Nitrogen fixation ([N2]/[N2]sat as availability indicator)
    RXN 05: Phosphate uptake ([SRP]/Ks using half-saturation)
    RXN 06: Biomass synthesis (Redfield stoichiometry Q/K')
    """
    T_K = df['temp_c'] + 273.15
    H = 10**(-df['ph'])

    # ------------------------------------------------------------------
    # RXN 01: Photosynthesis
    # 6CO2 + 6H2O -> C6H12O6 + 6O2
    # Q = [O2]^6 / [CO2]^6  (water = solvent, glucose consumed)
    # K' is extremely small (reaction is thermodynamically unfavorable)
    # but light energy drives it. We use Q as a favorability indicator:
    # high Q (high O2, low CO2) = photosynthesis has been active
    # low Q (low O2, high CO2) = favorable conditions for MORE photosynthesis
    # ------------------------------------------------------------------
    co2 = np.clip(df['co2_aq_mol'], 1e-12, None)
    do = np.clip(df['do_mol'], 1e-12, None)

    Q_01 = (do**6) / (co2**6)
    # For photosynthesis, low Q means favorable (substrate available)
    # Invert so that high ratio = favorable for bloom
    # Use log scale to manage dynamic range
    df['qk_rxn01'] = np.log10(1.0 / Q_01)

    # ------------------------------------------------------------------
    # RXN 02: Carbonate equilibrium
    # CO2(aq) + H2O <-> HCO3- + H+
    # Q = [HCO3-][H+] / [CO2(aq)]
    # K = K1 (first dissociation constant)
    # Q/K < 1: CO2 accumulating (not yet equilibrated)
    # Q/K > 1: shifted toward bicarbonate
    # ------------------------------------------------------------------
    K1 = carbonate_K1(T_K)
    Q_02 = (df['hco3_mol'] * H) / co2
    df['qk_rxn02'] = Q_02 / K1

    # ------------------------------------------------------------------
    # RXN 03: Phosphate speciation
    # H2PO4- <-> HPO4^2- + H+
    # Q = [HPO4^2-][H+] / [H2PO4-]
    # K = Ka2
    # Q/K indicates which phosphate species dominates
    # ------------------------------------------------------------------
    Ka2 = phosphate_Ka2(T_K)
    h2po4 = np.clip(df['h2po4_mol'], 1e-15, None)
    Q_03 = (df['hpo4_mol'] * H) / h2po4
    df['qk_rxn03'] = Q_03 / Ka2

    # ------------------------------------------------------------------
    # RXN 04: Nitrogen fixation (cyanobacteria-specific)
    # N2 + 8H+ + 8e- -> 2NH3 + H2
    # Use [N2]actual / [N2]sat as substrate availability ratio
    # When [N2] is near saturation, plenty of substrate for N-fixers
    # Ks for N-fixation ~ 0.5 * saturation (literature estimate)
    # ------------------------------------------------------------------
    Ks_n2 = 0.5 * df['n2_sat_mol']
    # Assume actual N2 ≈ saturation (no direct sensor); deviation from
    # equilibrium driven by biological drawdown estimated from context
    n2_actual = df['n2_sat_mol'] * 0.95  # slight undersaturation assumed
    df['qk_rxn04'] = n2_actual / Ks_n2

    # ------------------------------------------------------------------
    # RXN 05: Phosphate uptake (Monod kinetics)
    # [SRP] / Ks where Ks ~ 3-10 µg P/L for cyanobacteria
    # (Paerl & Huisman 2009, Dolman et al. 2012)
    # < 1 = P-limited, > 1 = P-replete
    # ------------------------------------------------------------------
    Ks_srp = 5.0  # µg P/L, mid-range for bloom-forming cyanobacteria
    df['qk_rxn05'] = df['srp_ugl'] / Ks_srp

    # ------------------------------------------------------------------
    # RXN 06: Biomass synthesis (Redfield equation)
    # 106CO2 + 16NO3- + HPO4^2- + ... -> biomass
    # Q' = 1 / ([CO2]^106 * [HPO4^2-])
    # Simplified: use log-scale product of key substrates
    # High substrate availability = favorable for biomass production
    # ------------------------------------------------------------------
    hpo4 = np.clip(df['hpo4_mol'], 1e-15, None)
    # Favorability = substrate availability (log scale)
    df['qk_rxn06'] = np.log10(co2 * hpo4) * -1  # invert: high substrates = high score

    return df


# =============================================================================
# 4. ECOLOGICAL RATIOS
# =============================================================================

def compute_ecological_ratios(df):
    """PC:Chl dominance ratio and N:P ratio."""

    # PC:Chl ratio (cyanobacterial dominance indicator)
    chla = np.clip(df['chla_rfu'], 0.01, None)
    df['pc_chl_ratio'] = df['pc_rfu'] / chla

    # N:P ratio (molar)
    # Without direct nitrate at WE8, use conductivity as ionic strength proxy
    # and SRP for P. This is a simplified N:P using available data.
    # If nitrate data from WE2 proxy is available, substitute here.
    if 'nitrate_mgl' in df.columns:
        N_mol = (df['nitrate_mgl'] / 62.004) * 1e-3  # mg/L NO3 to mol/L
    else:
        # Approximate from conductivity regression (Lake Erie empirical)
        N_mol = (df['cond_uscm'] * 0.002) * 1e-3 / 14.007  # rough proxy
    
    P_mol = (df['srp_ugl'] * 1e-6) / 30.97
    P_mol = np.clip(P_mol, 1e-12, None)
    df['np_ratio'] = N_mol / P_mol

    return df


# =============================================================================
# 5. BLOOM LABELING
# =============================================================================

def label_blooms(df, pc_threshold=1.0, method='sigmoid'):
    """
    Label bloom/no-bloom using phycocyanin.
    
    Sigmoid normalization: midpoint m=1.0 RFU, steepness k=3.0
    anchored to YSI EXO2 specs and WHO Alert Level thresholds.
    Binary label at 0.5 probability threshold.
    """
    if method == 'sigmoid':
        m = 1.0   # midpoint in RFU
        k = 3.0   # steepness
        bloom_prob = 1.0 / (1.0 + np.exp(-k * (df['pc_rfu'] - m)))
        df['bloom_label'] = (bloom_prob >= 0.5).astype(int)
    else:
        df['bloom_label'] = (df['pc_rfu'] >= pc_threshold).astype(int)

    return df


# =============================================================================
# 6. SLIDING WINDOW
# =============================================================================

def build_sliding_windows(df, input_days=20, target_days=5, step=1):
    """
    Build sliding windows: input_days of features -> target_days bloom label.
    Target = 1 if ANY day in the target window is bloom-positive.
    
    Returns X (aggregated features per window), y (binary labels).
    """
    feature_cols_baseline = [
        'temp_c', 'do_mgl', 'ph', 'srp_ugl', 'cond_uscm'
    ]
    feature_cols_ess = [
        'qk_rxn01', 'qk_rxn02', 'qk_rxn03',
        'qk_rxn04', 'qk_rxn05', 'qk_rxn06',
        'np_ratio'
    ]

    X_baseline, X_ess, y = [], [], []
    total_len = len(df)

    for i in range(0, total_len - input_days - target_days + 1, step):
        input_window = df.iloc[i:i + input_days]
        target_window = df.iloc[i + input_days:i + input_days + target_days]

        # Aggregate input window: mean and std for each feature
        base_feats = []
        for col in feature_cols_baseline:
            base_feats.append(input_window[col].mean())
            base_feats.append(input_window[col].std())

        ess_feats = []
        for col in feature_cols_ess:
            ess_feats.append(input_window[col].mean())
            ess_feats.append(input_window[col].std())

        # Target: bloom if any day in target window is positive
        label = int(target_window['bloom_label'].max())

        X_baseline.append(base_feats)
        X_ess.append(ess_feats)
        y.append(label)

    # Column names
    base_names = []
    for col in feature_cols_baseline:
        base_names.extend([f'{col}_mean', f'{col}_std'])
    ess_names = []
    for col in feature_cols_ess:
        ess_names.extend([f'{col}_mean', f'{col}_std'])

    X_baseline = pd.DataFrame(X_baseline, columns=base_names)
    X_ess = pd.DataFrame(X_ess, columns=ess_names)
    y = np.array(y)

    return X_baseline, X_ess, y


# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================

def evaluate_model(X, y, model_name, n_splits=5):
    """
    Stratified k-fold cross-validation with logistic regression.
    Reports F1, PR-AUC, MCC, and confusion matrix.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    f1_scores, prauc_scores, mcc_scores = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Handle any NaN/inf from chemistry calculations
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=10, neginf=-10)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=10, neginf=-10)

        # Logistic regression with balanced class weights
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

    # Print results
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  F1 Score:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  PR-AUC:       {np.mean(prauc_scores):.4f} ± {np.std(prauc_scores):.4f}")
    print(f"  MCC:          {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")
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
    """Fit on full data and print coefficient magnitudes."""
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
    print(f"  {'Feature':<25} {'Coefficient':>12}")
    print(f"  {'-'*25} {'-'*12}")
    for feat in coefs_sorted.index[:10]:
        print(f"  {feat:<25} {coefs[feat]:>12.4f}")


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def run_pipeline(data_path, phosphate_path=None):
    """
    Full pipeline: load data -> derive variables -> compute Q/K ->
    build windows -> train/evaluate both models -> compare.
    """
    # Load main sensor data (rows 1-2 are units/instrument metadata)
    print("Loading data...")
    df = pd.read_csv(data_path, skiprows=[1, 2])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Load and merge phosphate data if provided
    if phosphate_path:
        phos = pd.read_csv(phosphate_path, skiprows=[1, 2])
        phos['timestamp'] = pd.to_datetime(phos['timestamp'])
        phos['date'] = phos['timestamp'].dt.date
        phos = phos[['date', 'phosphate']].dropna(subset=['phosphate'])
        df['date'] = df['timestamp'].dt.date
        df = df.merge(phos, on='date', how='left')
        df['phosphate'] = pd.to_numeric(df['phosphate'], errors='coerce')
        # Forward-fill daily phosphate values across 15-min intervals
        df['phosphate'] = df['phosphate'].ffill()
        df.drop(columns=['date'], inplace=True)

    # Standardize column names
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

    # Convert all required sensor columns to numeric (flag columns cause string inference)
    numeric_cols = ['temp_c', 'do_mgl', 'ph', 'srp_ugl', 'pc_rfu', 'chla_rfu', 'cond_uscm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert specific conductivity from mS/cm to µS/cm
    if 'cond_uscm' in df.columns:
        df['cond_uscm'] = df['cond_uscm'] * 1000

    # Verify required columns exist
    required = ['temp_c', 'do_mgl', 'ph', 'srp_ugl',
                'pc_rfu', 'chla_rfu', 'cond_uscm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Drop rows with missing sensor data
    initial_len = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  Rows: {initial_len} -> {len(df)} after dropping NaN")

    # Step 1: Derive chemical variables
    print("Computing derived chemical variables (10 variables)...")
    df = compute_derived_variables(df)

    # Step 2: Compute Q/K ratios
    print("Computing Q/K ratios for 6 reactions...")
    df = compute_qk_ratios(df)

    # Step 3: Ecological ratios
    print("Computing ecological ratios (PC:Chl, N:P)...")
    df = compute_ecological_ratios(df)

    # Step 4: Label blooms
    print("Labeling bloom events (sigmoid, m=1.0, k=3.0)...")
    df = label_blooms(df)

    # Report class distribution
    n_pos = df['bloom_label'].sum()
    n_neg = len(df) - n_pos
    print(f"\n  Daily class distribution:")
    print(f"    Bloom:    {n_pos:4d} ({100*n_pos/len(df):.1f}%)")
    print(f"    No bloom: {n_neg:4d} ({100*n_neg/len(df):.1f}%)")

    # Step 5: Build sliding windows
    print("\nBuilding sliding windows (20-day input / 5-day target)...")
    X_baseline, X_ess, y = build_sliding_windows(df)

    n_pos_w = y.sum()
    n_neg_w = len(y) - n_pos_w
    print(f"  Total windows:    {len(y)}")
    print(f"  Positive windows: {n_pos_w} ({100*n_pos_w/len(y):.1f}%)")
    print(f"  Negative windows: {n_neg_w} ({100*n_neg_w/len(y):.1f}%)")

    if n_pos_w < 10:
        print("\nWARNING: Very few positive samples. Results may be unreliable.")

    # Step 6: Evaluate both models
    print("\n" + "="*60)
    print("  RUNNING MODEL COMPARISON")
    print("="*60)

    results_baseline = evaluate_model(
        X_baseline, y, "BASELINE: Logistic Regression on Raw Features"
    )
    results_ess = evaluate_model(
        X_ess, y, "ESS: Logistic Regression on Q/K-Derived Features"
    )

    # Step 7: Feature importance
    print_feature_importance(
        X_baseline, y, X_baseline.columns, "Baseline"
    )
    print_feature_importance(
        X_ess, y, X_ess.columns, "ESS"
    )

    # Step 8: Summary comparison
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Baseline':>18} {'ESS':>18} {'Δ':>10}")
    print(f"  {'-'*12} {'-'*18} {'-'*18} {'-'*10}")

    for metric, label in [('f1', 'F1'), ('prauc', 'PR-AUC'), ('mcc', 'MCC')]:
        b = results_baseline[metric]
        e = results_ess[metric]
        delta = e - b
        sign = '+' if delta > 0 else ''
        print(f"  {label:<12} {b:>12.4f} ± {results_baseline[metric+'_std']:.3f}"
              f"  {e:>7.4f} ± {results_ess[metric+'_std']:.3f}"
              f"  {sign}{delta:.4f}")

    if results_ess['f1'] > results_baseline['f1']:
        print(f"\n  >> ESS outperforms baseline on F1 by "
              f"{results_ess['f1'] - results_baseline['f1']:.4f}")
    else:
        print(f"\n  >> Baseline outperforms ESS on F1 by "
              f"{results_baseline['f1'] - results_ess['f1']:.4f}")

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
        data_path = 'data/2017_Bloom_params/2017 bloom dataset.csv'
        phosphate_path = 'data/2017_Bloom_params/WE08_2017_annual_summary_phosphate.csv'

    print("="*60)
    print("  ESS vs Baseline Model Comparison")
    print("  Equilibrium-State Stoichiometry for HAB Prediction")
    print("="*60)

    run_pipeline(data_path, phosphate_path)