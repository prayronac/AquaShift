"""
Constants for the ESS bloom prediction system.

Redfield Equation (photosynthesis direction):
  106 CO2 + 16 HNO3 + H3PO4 + 122 H2O  ⇌  (CH2O)106(NH3)16·H3PO4 + 138 O2

The equilibrium constant K_eq represents the product/reactant ratio at which
this reaction is thermodynamically balanced. When the reaction quotient Q
exceeds K, conditions favor forward (biomass production) — i.e., bloom.

We track Q using measured proxies for each reactant/product term.
"""

# ── Redfield stoichiometric coefficients ──────────────────────────────────
REDFIELD = {
    "CO2":   106,
    "HNO3":  16,
    "H3PO4": 1,
    "H2O":   122,
    "O2":    138,
}

# C:N:P molar ratio
REDFIELD_CNP = (106, 16, 1)

# ── Anabaena bloom thresholds (literature values) ─────────────────────────
# These define "favorable" ranges. Used for K calculation and negative controls.
ANABAENA_FAVORABLE = {
    "ph_min":              7.5,
    "ph_max":              9.5,
    "temperature_min_c":   20.0,
    "temperature_max_c":   30.0,
    "do_min_mg_l":         4.0,    # dissolved oxygen floor for active photosynthesis
    "chlorophyll_bloom_ug_l": 10.0,  # chlorophyll level indicating bloom onset
    "n_p_ratio_low":       10.0,   # N:P molar ratio range favorable for cyanos
    "n_p_ratio_high":      20.0,
}

# ── Proxy mappings ────────────────────────────────────────────────────────
# How measured parameters map to Redfield equation terms:
#   pH          → proxy for CO2 (inverse: high pH = low dissolved CO2)
#   phycocyanin → proxy for biomass (CH2O)106(NH3)16
#   chlorophyll → proxy for biomass (secondary)
#   dissolved_oxygen → direct O2 measurement
#   turbidity   → proxy for total particulate (biomass + inorganic)
#   temperature → affects K_eq (van 't Hoff scaling)

# Column names expected in data files
DATA_COLUMNS = [
    "date",
    "chlorophyll",
    "phycocyanin",
    "turbidity",
    "dissolved_oxygen",
    "ph",
    "temperature",
    "bloom_peak_date",
]
