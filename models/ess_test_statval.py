"""
ESS Statistical Validation: Permutation Test + Feature Ablation

PERMUTATION TEST:
  Shuffles ESS feature values (breaking their relationship with blooms)
  and reruns the model 500 times. If real ESS F1 beats shuffled ESS F1
  in 95%+ of trials, the features carry statistically significant signal.
  Produces a p-value.

FEATURE ABLATION:
  Removes one feature group at a time and measures the F1 drop.
  Shows which specific groups (Q/K ratios, convergence, N:P) contribute.

Author: Prayrona
Project: Equilibrium-State Stoichiometry for HAB Prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CHEMISTRY FUNCTIONS (identical to v3)
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
    else:
        N_umol = (df['cond_uscm'] * 0.002) * 1e-3 / 14.007 * 1e6
    P_umol = df['srp_ugl'] / 30.97
    P_umol = np.clip(P_umol, 0.001, None)
    df['np_ratio'] = N_umol / P_umol
    return df


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

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

def build_windows(segments, input_days=20, target_days=5, step=1):
    """Build windows and return baseline features, ESS features, and labels."""
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


def evaluate_f1(X, y, n_splits=5):
    """Quiet evaluation, returns mean F1 only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    f1_scores = []

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
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(f1_scores)


# =============================================================================
# PREPARE DATA (run once)
# =============================================================================

def prepare_data(file_pairs, threshold=1.3):
    """Load all data, compute chemistry, build windows. Returns X_base, X_ess, y."""
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
    print(f"  Rows after cleaning: {len(df)}")

    print("Computing chemistry...")
    df = compute_derived_variables(df)
    df = compute_qk_ratios(df)
    df = compute_convergence_features(df)
    df = compute_ecological_ratios(df)

    print(f"Labeling blooms at {threshold} RFU...")
    df['bloom_label'] = (df['pc_rfu'] >= threshold).astype(int)

    # Aggregate to daily
    df['date'] = df['timestamp'].dt.date
    exclude_cols = ['timestamp', 'date', 'source_file']
    agg_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    agg_dict = {col: 'mean' for col in agg_cols}
    agg_dict['bloom_label'] = 'max'
    daily = df.groupby('date').agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    n_pos = int(daily['bloom_label'].sum())
    print(f"  Daily: {len(daily)} days, {n_pos} bloom days ({100*n_pos/len(daily):.1f}%)")

    segments = identify_continuous_segments(daily)
    X_baseline, X_ess, y = build_windows(segments)

    n_pos_w = int(y.sum())
    print(f"  Windows: {len(y)} total, {n_pos_w} positive ({100*n_pos_w/len(y):.1f}%)")

    return X_baseline, X_ess, y


# =============================================================================
# TEST 1: PERMUTATION TEST
# =============================================================================

def run_permutation_test(X_ess, y, n_permutations=500):
    """
    Shuffle ESS features to break feature-label relationship.
    Compare real F1 to distribution of shuffled F1 values.
    
    If real F1 > 95% of shuffled F1 values, p < 0.05 (significant).
    """
    print(f"\n{'='*60}")
    print(f"  PERMUTATION TEST ({n_permutations} permutations)")
    print(f"{'='*60}")

    # Real ESS F1
    print("  Computing real ESS F1...")
    real_f1 = evaluate_f1(X_ess, y)
    print(f"  Real ESS F1: {real_f1:.4f}")

    # Shuffled F1 values
    print(f"  Running {n_permutations} permutations (this takes a few minutes)...")
    shuffled_f1s = []
    start_time = time.time()

    for i in range(n_permutations):
        # Shuffle each column independently (breaks feature correlations with labels)
        X_shuffled = X_ess.copy()
        rng = np.random.RandomState(i)
        for col in X_shuffled.columns:
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)

        f1_shuf = evaluate_f1(X_shuffled, y)
        shuffled_f1s.append(f1_shuf)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_permutations - i - 1) / rate
            print(f"    {i+1}/{n_permutations} done "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining) "
                  f"| shuffled F1 so far: {np.mean(shuffled_f1s):.4f}")

    shuffled_f1s = np.array(shuffled_f1s)

    # Compute p-value: fraction of shuffled trials that beat real F1
    p_value = (np.sum(shuffled_f1s >= real_f1) + 1) / (n_permutations + 1)

    print(f"\n  {'='*50}")
    print(f"  PERMUTATION TEST RESULTS")
    print(f"  {'='*50}")
    print(f"  Real ESS F1:          {real_f1:.4f}")
    print(f"  Shuffled F1 mean:     {np.mean(shuffled_f1s):.4f}")
    print(f"  Shuffled F1 std:      {np.std(shuffled_f1s):.4f}")
    print(f"  Shuffled F1 max:      {np.max(shuffled_f1s):.4f}")
    print(f"  Shuffled F1 95th pct: {np.percentile(shuffled_f1s, 95):.4f}")
    print(f"  Shuffled F1 99th pct: {np.percentile(shuffled_f1s, 99):.4f}")
    print(f"")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.001:
        print(f"  >>> HIGHLY SIGNIFICANT (p < 0.001)")
        print(f"  >>> ESS features carry real predictive signal.")
    elif p_value < 0.01:
        print(f"  >>> VERY SIGNIFICANT (p < 0.01)")
        print(f"  >>> ESS features carry real predictive signal.")
    elif p_value < 0.05:
        print(f"  >>> SIGNIFICANT (p < 0.05)")
        print(f"  >>> ESS features carry real predictive signal.")
    else:
        print(f"  >>> NOT SIGNIFICANT (p >= 0.05)")
        print(f"  >>> Cannot reject null hypothesis.")
    print(f"")
    print(f"  Interpretation: The probability of achieving F1 = {real_f1:.4f}")
    print(f"  by random chance is {p_value:.4f} ({100*p_value:.2f}%).")
    print(f"  Out of {n_permutations} random shuffles, {np.sum(shuffled_f1s >= real_f1)}")
    print(f"  achieved F1 >= the real value.")

    return real_f1, shuffled_f1s, p_value


# =============================================================================
# TEST 2: FEATURE ABLATION
# =============================================================================

def run_feature_ablation(X_ess, y):
    """
    Remove one feature group at a time and measure F1 drop.
    Shows which groups contribute to ESS performance.
    """
    print(f"\n{'='*60}")
    print(f"  FEATURE ABLATION TEST")
    print(f"{'='*60}")

    # Define feature groups
    qk_mean_cols = [c for c in X_ess.columns if c.startswith('qk_rxn') and c.endswith('_mean')]
    qk_std_cols = [c for c in X_ess.columns if c.startswith('qk_rxn') and c.endswith('_std')]
    qk_all = qk_mean_cols + qk_std_cols

    conv_cols = [c for c in X_ess.columns if any(
        c.startswith(p) for p in ['n_favorable', 'frac_favorable', 'convergence_score',
                                   'favorability_product', 'max_n_favorable',
                                   'days_all_favorable', 'max_consec',
                                   'convergence_trend', 'peak_favorability']
    )]

    np_cols = [c for c in X_ess.columns if c.startswith('np_ratio')]

    # Window-level summary cols
    window_cols = ['max_n_favorable', 'days_all_favorable',
                   'max_consec_all_favorable', 'convergence_trend',
                   'peak_favorability_product']
    window_cols = [c for c in window_cols if c in X_ess.columns]

    # Daily convergence cols (mean/std)
    daily_conv_cols = [c for c in conv_cols if c not in window_cols]

    groups = {
        'Full ESS (all features)': [],  # remove nothing
        'Remove Q/K ratios': qk_all,
        'Remove convergence (daily)': daily_conv_cols,
        'Remove convergence (window summaries)': window_cols,
        'Remove ALL convergence': conv_cols,
        'Remove N:P ratio': np_cols,
        'Q/K ratios ONLY': [c for c in X_ess.columns if c not in qk_all],
        'Convergence ONLY': [c for c in X_ess.columns if c not in conv_cols],
    }

    # Full ESS baseline
    full_f1 = evaluate_f1(X_ess, y)
    print(f"\n  Full ESS F1: {full_f1:.4f}")
    print(f"\n  {'Group':<40} {'F1':>8} {'Drop':>8} {'Relative':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*10}")

    results = {}
    for group_name, cols_to_remove in groups.items():
        if len(cols_to_remove) == 0:
            f1 = full_f1
            drop = 0.0
        else:
            remaining_cols = [c for c in X_ess.columns if c not in cols_to_remove]
            if len(remaining_cols) == 0:
                f1 = 0.0
                drop = full_f1
            else:
                X_reduced = X_ess[remaining_cols]
                f1 = evaluate_f1(X_reduced, y)
                drop = full_f1 - f1

        rel_drop = 100 * drop / full_f1 if full_f1 > 0 else 0
        results[group_name] = {'f1': f1, 'drop': drop, 'rel_drop': rel_drop}

        sign = '-' if drop > 0 else '+'
        print(f"  {group_name:<40} {f1:>8.4f} {sign}{abs(drop):>7.4f} {sign}{abs(rel_drop):>8.1f}%")

    # Most important group
    drops = {k: v['drop'] for k, v in results.items() if v['drop'] > 0}
    if drops:
        most_important = max(drops, key=drops.get)
        print(f"\n  Most impactful group: {most_important}")
        print(f"  Removing it drops F1 by {drops[most_important]:.4f} "
              f"({results[most_important]['rel_drop']:.1f}% relative)")

    return results


# =============================================================================
# TEST 3: McNEMAR'S TEST (bonus)
# =============================================================================

def run_mcnemar_test(X_baseline, X_ess, y):
    """
    McNemar's test compares whether the two models disagree on predictions
    in a statistically significant way.
    """
    print(f"\n{'='*60}")
    print(f"  McNEMAR'S TEST (Baseline vs ESS)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler_b = StandardScaler()
    scaler_e = StandardScaler()

    all_pred_base = []
    all_pred_ess = []
    all_true = []

    for train_idx, test_idx in skf.split(X_baseline, y):
        # Baseline
        Xb_train = scaler_b.fit_transform(X_baseline.iloc[train_idx])
        Xb_test = scaler_b.transform(X_baseline.iloc[test_idx])
        Xb_train = np.nan_to_num(Xb_train, nan=0.0, posinf=10, neginf=-10)
        Xb_test = np.nan_to_num(Xb_test, nan=0.0, posinf=10, neginf=-10)

        model_b = LogisticRegression(class_weight='balanced', max_iter=1000,
                                      solver='lbfgs', random_state=42)
        model_b.fit(Xb_train, y[train_idx])
        pred_b = model_b.predict(Xb_test)

        # ESS
        Xe_train = scaler_e.fit_transform(X_ess.iloc[train_idx])
        Xe_test = scaler_e.transform(X_ess.iloc[test_idx])
        Xe_train = np.nan_to_num(Xe_train, nan=0.0, posinf=10, neginf=-10)
        Xe_test = np.nan_to_num(Xe_test, nan=0.0, posinf=10, neginf=-10)

        model_e = LogisticRegression(class_weight='balanced', max_iter=1000,
                                      solver='lbfgs', random_state=42)
        model_e.fit(Xe_train, y[train_idx])
        pred_e = model_e.predict(Xe_test)

        all_pred_base.extend(pred_b)
        all_pred_ess.extend(pred_e)
        all_true.extend(y[test_idx])

    all_pred_base = np.array(all_pred_base)
    all_pred_ess = np.array(all_pred_ess)
    all_true = np.array(all_true)

    # McNemar contingency table
    # b = ESS correct, Baseline wrong
    # c = Baseline correct, ESS wrong
    base_correct = (all_pred_base == all_true)
    ess_correct = (all_pred_ess == all_true)

    b = np.sum(ess_correct & ~base_correct)  # ESS right, baseline wrong
    c = np.sum(base_correct & ~ess_correct)  # baseline right, ESS wrong

    both_right = np.sum(ess_correct & base_correct)
    both_wrong = np.sum(~ess_correct & ~base_correct)

    print(f"\n  Disagreement table:")
    print(f"                     ESS correct  ESS wrong")
    print(f"  Baseline correct      {both_right:4d}         {c:4d}")
    print(f"  Baseline wrong        {b:4d}         {both_wrong:4d}")

    # McNemar chi-squared (with continuity correction)
    if b + c > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        # p-value from chi-squared distribution with 1 df
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0.0
        p_value = 1.0

    print(f"\n  ESS corrects {b} predictions that baseline gets wrong")
    print(f"  Baseline corrects {c} predictions that ESS gets wrong")
    print(f"  Net advantage for ESS: {b - c} predictions")
    print(f"\n  Chi-squared: {chi2:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  >>> SIGNIFICANT: The models make significantly different predictions.")
    else:
        print(f"  >>> Not significant at p<0.05 level.")

    return b, c, chi2, p_value


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

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

    threshold = 1.3

    print("="*60)
    print("  ESS Statistical Validation")
    print(f"  Permutation Test + Feature Ablation + McNemar's Test")
    print(f"  Bloom threshold: {threshold} RFU")
    print("="*60)

    # Prepare data once
    X_baseline, X_ess, y = prepare_data(file_pairs, threshold=threshold)

    # Run all three tests
    real_f1, shuffled_f1s, p_value = run_permutation_test(X_ess, y, n_permutations=500)

    ablation_results = run_feature_ablation(X_ess, y)

    b, c, chi2, mcnemar_p = run_mcnemar_test(X_baseline, X_ess, y)

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  FINAL STATISTICAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Permutation test p-value:  {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
    print(f"  McNemar's test p-value:    {mcnemar_p:.4f} {'***' if mcnemar_p < 0.001 else '**' if mcnemar_p < 0.01 else '*' if mcnemar_p < 0.05 else 'n.s.'}")
    print(f"  (* p<0.05, ** p<0.01, *** p<0.001)")
    print(f"\n  ESS features carry {'SIGNIFICANT' if p_value < 0.05 else 'NO SIGNIFICANT'} predictive signal.")
    print(f"  ESS and baseline make {'SIGNIFICANTLY' if mcnemar_p < 0.05 else 'NOT significantly'} different predictions.")