"""
train_model.py — Learn CPTs from bloom-labeled sensor data
===========================================================
Usage:
    python utils/train_model.py

Reads all files in TRAINING_FILES, creates BloomOccurrence labels from
bloom_peak_date (±7 days = bloom), discretizes sensor readings, then fits
the Bayesian network CPTs using BayesianEstimator (Dirichlet priors).
Saves the trained model to utils/trained_model.pkl.

To add a new dataset (Lake Erie, Mississippi, etc.):
    1. Put the file in data/bloom_timelines/ (or data/training/)
    2. Add its path to TRAINING_FILES below
    3. Add any new column names to COLUMN_MAP if needed
    4. Re-run this script
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(__file__)

TRAINING_FILES = [
    os.path.join(SCRIPT_DIR, "..", "data", "bloom_timelines", "Columbia_River_FIXED_Blooms.xlsx"),
    os.path.join(SCRIPT_DIR, "..", "data", "bloom_timelines", "Johnson_Datapoints.xlsx"),
    # Add new files here as they arrive:
    # os.path.join(SCRIPT_DIR, "..", "data", "bloom_timelines", "LakeErie_Datapoints.xlsx"),
    # os.path.join(SCRIPT_DIR, "..", "data", "bloom_timelines", "Mississippi_Datapoints.xlsx"),
]

MODEL_OUT = os.path.join(SCRIPT_DIR, "trained_model.pkl")

# How many days around a bloom_peak_date count as a bloom event
BLOOM_WINDOW_DAYS = 3

# ─────────────────────────────────────────────────────────────
# COLUMN NAME NORMALIZATION
# Maps any dataset's column names → internal sensor keys
# ─────────────────────────────────────────────────────────────

COLUMN_MAP = {
    # Johnson / Columbia River (raw USGS names)
    "Dissolved Oxygen":                       "DO",
    "pH":                                     "pH",
    "Specific conductance, water, unfiltered": "conductance",
    "Turbidity":                              "turbidity",
    "Phycocyanin relative fluorescence (fPC)": "phycocyanin",
    # Columbia River FIXED (cleaner names)
    "dissolved_oxygen_mg_L":                  "DO",
    "conductance_uS_cm":                      "conductance",
    "turbidity_FNU":                          "turbidity",
    "phycocyanin_RFU":                        "phycocyanin",
    # Common aliases for Lake Erie / Mississippi datasets (extend as needed)
    "Dissolved_Oxygen":                       "DO",
    "SpCond_uS_cm":                           "conductance",
    "Turbidity_FNU":                          "turbidity",
    "Phycocyanin_RFU":                        "phycocyanin",
}

# ─────────────────────────────────────────────────────────────
# NETWORK STRUCTURE  (same DAG as inference script)
# ─────────────────────────────────────────────────────────────

# Observable-only network: only nodes with actual sensor data.
# Latent nodes (FlowVelocity, ResidenceTime, NitrogenFixation,
# NitrateAssimilation, BiomassSynthesis) are excluded until discharge/NO3
# sensor data is available. Add them back then.
EDGES = [
    ("Photosynthesis",       "CarbonateEquilibrium"),
    ("Photosynthesis",       "BloomOccurrence"),
    ("CarbonateEquilibrium", "BloomOccurrence"),
    ("PhosphateUptake",      "BloomOccurrence"),
    ("BenthicDetachment",    "BloomOccurrence"),
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def normalize_columns(df):
    """Rename dataset columns to internal sensor keys using COLUMN_MAP."""
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


def make_bloom_labels(df, window_days=BLOOM_WINDOW_DAYS):
    """
    Create BloomOccurrence column (0 or 1).
    Rows within ±window_days of any bloom_peak_date → 1, else 0.
    """
    dates = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None) if pd.to_datetime(df["date"]).dt.tz is not None else pd.to_datetime(df["date"])
    bloom_peaks = pd.to_datetime(df["bloom_peak_date"].dropna().unique())

    bloom_flag = pd.Series(0, index=df.index)
    for peak in bloom_peaks:
        delta = (dates - peak).abs()
        bloom_flag[delta <= pd.Timedelta(days=window_days)] = 1

    return bloom_flag


def discretize(df):
    """
    Convert normalized sensor columns → discrete node states (0/1/2).
    NaN is returned for nodes with no sensor data in this dataset.
    """
    out = pd.DataFrame(index=df.index)

    # FlowVelocity — no discharge sensor available → NaN (latent)
    out["FlowVelocity"] = np.nan

    # CarbonateEquilibrium — pH 2nd derivative
    if "pH" in df.columns:
        ph = df["pH"].ffill()
        win = min(11, len(ph) if len(ph) % 2 == 1 else len(ph) - 1)
        if win >= 5:
            ph_accel = savgol_filter(ph, window_length=win, polyorder=2, deriv=2)
            out["CarbonateEquilibrium"] = pd.cut(ph_accel,
                bins=[-np.inf, -0.002, 0.002, np.inf],
                labels=[0, 1, 2]).astype("float")
        else:
            out["CarbonateEquilibrium"] = np.nan
    else:
        out["CarbonateEquilibrium"] = np.nan

    # Photosynthesis — dissolved oxygen proxy
    if "DO" in df.columns:
        out["Photosynthesis"] = pd.cut(df["DO"],
            bins=[-np.inf, 8, 11, np.inf],
            labels=[2, 1, 0]).astype("float")
    else:
        out["Photosynthesis"] = np.nan

    # PhosphateUptake — conductance proxy
    if "conductance" in df.columns:
        out["PhosphateUptake"] = pd.cut(df["conductance"],
            bins=[-np.inf, 150, 250, np.inf],
            labels=[2, 1, 0]).astype("float")
    else:
        out["PhosphateUptake"] = np.nan

    # NitrateAssimilation — no NO3 sensor → NaN (latent)
    out["NitrateAssimilation"] = np.nan

    # NitrogenFixation — latent (derived in network from PhosphateUptake)
    out["NitrogenFixation"] = np.nan

    # ResidenceTime — latent (derived from FlowVelocity)
    out["ResidenceTime"] = np.nan

    # BiomassSynthesis — latent
    out["BiomassSynthesis"] = np.nan

    # BenthicDetachment — turbidity drop + phycocyanin spike
    if "turbidity" in df.columns and "phycocyanin" in df.columns:
        turb_delta = df["turbidity"].diff(periods=4)
        pc_delta   = df["phycocyanin"].diff(periods=4)
        detachment = (turb_delta < -2) & (pc_delta > 1.5)
        out["BenthicDetachment"] = np.where(detachment, 0,
                                    np.where(pc_delta > 0.5, 1.0, 2.0))
    else:
        out["BenthicDetachment"] = np.nan

    return out


def load_file(path):
    """Load an Excel or CSV file, return raw DataFrame."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path, parse_dates=["date"])
    return pd.read_csv(path, parse_dates=["date"])


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

all_frames = []

for path in TRAINING_FILES:
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        continue

    print(f"Loading: {os.path.basename(path)}")
    df_raw = load_file(path)
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    df_norm = normalize_columns(df_raw)

    node_df = discretize(df_norm)

    if "bloom_peak_date" in df_raw.columns:
        node_df["BloomOccurrence"] = make_bloom_labels(df_norm)
        n_bloom = node_df["BloomOccurrence"].sum()
        print(f"  {len(df_raw)} rows | {int(n_bloom)} bloom rows ({n_bloom/len(df_raw)*100:.1f}%)")
    else:
        node_df["BloomOccurrence"] = np.nan
        print(f"  {len(df_raw)} rows | no bloom_peak_date column")

    all_frames.append(node_df)

if not all_frames:
    raise RuntimeError("No training files found. Add files to TRAINING_FILES.")

training_df = pd.concat(all_frames, ignore_index=True)
print(f"\nTotal training rows: {len(training_df)}")
print(f"Bloom rows: {int(training_df['BloomOccurrence'].sum())} "
      f"({training_df['BloomOccurrence'].mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# BUILD MODEL AND FIT
# ─────────────────────────────────────────────────────────────

model = DiscreteBayesianNetwork(EDGES)

state_names = {
    "Photosynthesis":       [0, 1, 2],
    "CarbonateEquilibrium": [0, 1, 2],
    "PhosphateUptake":      [0, 1, 2],
    "BenthicDetachment":    [0, 1, 2],
    "BloomOccurrence":      [0, 1],
}

# Keep only the observable node columns
obs_cols = list(state_names.keys())
training_df = training_df[obs_cols].dropna(subset=["BloomOccurrence"])
# Fill remaining NaNs with neutral state (1)
training_df = training_df.fillna(1).astype(int)

print(f"Training rows after NaN drop: {len(training_df)}")

print("\nFitting CPTs (BayesianEstimator with BDeu priors)...")
model.fit(
    training_df,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10,
    state_names=state_names,
)

assert model.check_model(), "Fitted model failed check_model()"
print("Model valid: True")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print(f"\nTrained model saved to: {MODEL_OUT}")
print("Run utils/Untitled-3.py to use it for inference.")
