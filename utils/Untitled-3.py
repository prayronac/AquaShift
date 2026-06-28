"""
ESS Bayesian Network v2 — Columbia River Anomaly Edition
=========================================================
New nodes added:
  - FlowVelocity         : USGS discharge-derived suppression modifier
  - ResidenceTime        : inverse-flow proxy for how long water sits in reach
  - BenthicDetachment    : turbidity-drop + phycocyanin-spike event flag

All nodes use 3 states:
  0 = Favorable   (Q/K < 0.5  OR condition supports bloom)
  1 = Neutral     (0.5 ≤ Q/K ≤ 2)
  2 = Unfavorable (Q/K > 2    OR condition suppresses bloom)

Bloom alert fires when posterior P(BloomOccurrence=1) > 0.6
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pgmpy.inference import VariableElimination


# ─────────────────────────────────────────────────────────────
# 1. NETWORK STRUCTURE
# ─────────────────────────────────────────────────────────────
#
#  FlowVelocity ──────────────────────────────────────────────┐
#       │                                                      │
#       ├──→ ResidenceTime                                     │
#       │         │                                            ▼
#       │         └──────────────────────────────→  BiomassSynthesis ──→ BloomOccurrence
#       │                                            ▲   ▲   ▲   ▲              ▲
#  Photosynthesis ──→ CarbonateEquilibrium ──────────┘   │   │   │              │
#                                                         │   │   │              │
#  PhosphateUptake ──→ NitrogenFixation ─────────────────┘   │   │              │
#                  └──────────────────────────────────────────┘   │              │
#  NitrateAssimilation ────────────────────────────────────────────┘              │
#                                                                                  │
#  BenthicDetachment ────────────────────────────────────────────────────────────┘
#     (turbidity drop + phycocyanin spike = cells going planktonic)
#

# ─────────────────────────────────────────────────────────────
# 2. LOAD TRAINED MODEL
# ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "No trained model found. Run utils/train_model.py first.\n"
        f"  Expected: {MODEL_PATH}"
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

assert model.check_model(), "Loaded model failed validation"
print("Model valid: True")
print(f"Nodes: {model.nodes()}")
print()


# ─────────────────────────────────────────────────────────────
# 7. INFERENCE SCENARIOS
# ─────────────────────────────────────────────────────────────

infer = VariableElimination(model)

def query_bloom(label, evidence):
    valid = set(model.nodes()) - {"BloomOccurrence"}
    evidence = {k: v for k, v in evidence.items() if k in valid}
    result = infer.query(["BloomOccurrence"], evidence=evidence, show_progress=False)
    p_bloom = result.values[1]
    alert = "*** BLOOM ALERT ***" if p_bloom > 0.6 else ("!   ELEVATED" if p_bloom > 0.35 else "    LOW RISK")
    print("-" * 55)
    print(f"Scenario: {label}")
    print(f"Evidence: {evidence}")
    print(f"P(Bloom)  = {p_bloom:.3f}   {alert}")
    print()


# Scenario A: Normal high-flow Columbia River baseline
# → should show low bloom probability (model sanity check)
query_bloom(
    "Baseline — high flow, typical conditions",
    evidence={
        "FlowVelocity":        2,   # High/unfavorable
        "BenthicDetachment":   2,   # No detachment
        "NitrateAssimilation": 1,   # Neutral N
        "PhosphateUptake":     1,   # Neutral P
    }
)

# Scenario B: Low-flow event (drought / dam throttle)
# → flow suppression lifts, residence time extends
query_bloom(
    "Low-flow event — dam throttled or drought conditions",
    evidence={
        "FlowVelocity":        0,   # Low/favorable
        "BenthicDetachment":   2,   # No detachment yet
        "CarbonateEquilibrium":0,   # pH accelerating (ESS signal)
        "NitrateAssimilation": 1,   # Neutral N
        "PhosphateUptake":     0,   # P available
    }
)

# Scenario C: Benthic mat detachment event
# → turbidity drops, phycocyanin spikes = cells going planktonic
query_bloom(
    "Benthic detachment event — mat lifting off substrate",
    evidence={
        "FlowVelocity":        1,   # Moderate flow
        "BenthicDetachment":   0,   # Detachment occurring
        "NitrateAssimilation": 1,   # Neutral
        "PhosphateUptake":     1,   # Neutral
    }
)

# Scenario D: The Columbia anomaly — everything aligns
# Low flow + ESS pH signal + detachment event
query_bloom(
    "Columbia anomaly — low flow + pH acceleration + benthic detachment",
    evidence={
        "FlowVelocity":        0,   # Low flow
        "BenthicDetachment":   0,   # Detachment event
        "CarbonateEquilibrium":0,   # pH accelerating
        "PhosphateUptake":     0,   # P favorable
        "NitrogenFixation":    0,   # N-fixation active (Anabaena advantage)
    }
)


# ─────────────────────────────────────────────────────────────
# 8. DISCRETIZE SENSOR DATA → NETWORK STATES
# ─────────────────────────────────────────────────────────────

def discretize(df):
    out = pd.DataFrame(index=df.index)

    # FlowVelocity — no discharge sensor in this dataset; default to neutral
    # Replace with: pd.cut(df["discharge_cfs"], bins=[-np.inf,40000,100000,np.inf], labels=[0,1,2])
    out["FlowVelocity"] = 1

    # CarbonateEquilibrium — pH 2nd derivative (acceleration = ESS signal)
    ph = df["pH"].ffill()
    win = min(11, len(ph) if len(ph) % 2 == 1 else len(ph) - 1)
    if win >= 5:
        ph_accel = savgol_filter(ph, window_length=win, polyorder=2, deriv=2)
        out["CarbonateEquilibrium"] = pd.cut(ph_accel,
            bins=[-np.inf, -0.002, 0.002, np.inf],
            labels=[0, 1, 2]).astype(int)
    else:
        out["CarbonateEquilibrium"] = 1

    # Photosynthesis — dissolved oxygen proxy (high DO = favorable)
    out["Photosynthesis"] = pd.cut(df["Dissolved Oxygen"],
        bins=[-np.inf, 8, 11, np.inf],
        labels=[2, 1, 0]).fillna(1).astype(int)

    # PhosphateUptake — specific conductance proxy
    out["PhosphateUptake"] = pd.cut(df["Specific conductance, water, unfiltered"],
        bins=[-np.inf, 150, 250, np.inf],
        labels=[2, 1, 0]).fillna(1).astype(int)

    # NitrateAssimilation — no NO3 sensor; default to neutral
    out["NitrateAssimilation"] = 1

    # BenthicDetachment — turbidity drop + phycocyanin spike (15-min intervals → periods=4 = 1 hr)
    turb_delta = df["Turbidity"].diff(periods=4)
    pc_delta   = df["Phycocyanin relative fluorescence (fPC)"].diff(periods=4)
    detachment = (turb_delta < -2) & (pc_delta > 1.5)
    out["BenthicDetachment"] = np.where(detachment, 0,
                                np.where(pc_delta > 0.5, 1, 2))

    return out


# ─────────────────────────────────────────────────────────────
# 9. LOAD DATA FILE AND RUN INFERENCE
# ─────────────────────────────────────────────────────────────

import os
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "bloom_timelines", "Johnson_Datapoints.xlsx")

df_raw = pd.read_excel(DATA_FILE, parse_dates=["date"])
df_raw = df_raw.sort_values("date").reset_index(drop=True)

discrete_df = discretize(df_raw)

print(f"Loaded {len(df_raw)} rows from {DATA_FILE}")
print()

# Group into 100-day intervals and take the modal state for each node
discrete_df["_date"] = df_raw["date"].values
discrete_df["_period"] = ((discrete_df["_date"] - discrete_df["_date"].min())
                          .dt.total_seconds() // (14 * 86400)).astype(int)

node_cols = [c for c in discrete_df.columns if not c.startswith("_")]
groups = list(discrete_df.groupby("_period"))

for i, (period, grp) in enumerate(groups):
    start = grp["_date"].min().strftime("%Y-%m-%d")
    end   = grp["_date"].max().strftime("%Y-%m-%d")
    print(f"{'='*55}")
    print(f"14-DAY PERIOD: {start} to {end}")
    print(f"{'='*55}")

    # Aggregate by day (modal state per day) and run inference for each day
    grp = grp.copy()
    grp["_day"] = grp["_date"].dt.normalize()
    daily_groups = grp.groupby("_day")

    for day, day_grp in daily_groups:
        evidence = day_grp[node_cols].mode().iloc[0].dropna().to_dict()
        evidence = {k: int(v) for k, v in evidence.items()}
        query_bloom(day.strftime("%Y-%m-%d"), evidence)

    if i < len(groups) - 1:
        ans = input("Analyze next 14-day period? (Y/N): ").strip().upper()
        if ans != "Y":
            print("Analysis stopped.")
            break