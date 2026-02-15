"""Central configuration for the methane-portfolio pipeline.

All tunable parameters live here.  The run manifest (outputs/run_manifest.json)
records the values actually used so every result is fully reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Paths  (override via env var  METHANE_DATA_DIR / METHANE_OUTPUT_DIR)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR: Path = Path(os.environ.get(
    "METHANE_DATA_DIR",
    str(_ROOT / "data"),
))

OUTPUT_DIR: Path = Path(os.environ.get(
    "METHANE_OUTPUT_DIR",
    str(_ROOT / "outputs"),
))

FIGURE_DIR: Path = OUTPUT_DIR / "figures"

# CSV file basenames (without full path prefix):
EMISSION_INTENSITY_FILE = "cercetare-485010.faostat_clean.milk_emission_intensity_2020_2023.csv"
SPECIES_STRUCTURE_FILE  = "cercetare-485010.faostat_clean.milk_species_structure.csv"
COUNTRY_INTENSITY_FILE  = "cercetare-485010.faostat_clean.milk_intensity_country_year.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED: int = 20230101
PYMC_SEED: int = 42
CHAINS: int = 2
DRAWS: int = 1000
TUNE: int = 1000
TARGET_ACCEPT: float = 0.90

# ---------------------------------------------------------------------------
# Validation tolerances
# ---------------------------------------------------------------------------
SHARE_SUM_TOL: float = 1e-6
MILK_MATCH_REL_TOL: float = 1e-6
IDENTITY_TOL: float = 1e-10

# ---------------------------------------------------------------------------
# Shapley
# ---------------------------------------------------------------------------
SHAPLEY_RECON_TOL: float = 1e-8
BASE_YEAR: int = 2020
END_YEAR: int = 2023

# ---------------------------------------------------------------------------
# Bayesian model
# ---------------------------------------------------------------------------
REGIME_SHIFT_YEAR: int = 2022  # 1[t >= 2022]

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    """Parameters for robust portfolio optimization."""
    lam: float = 0.5            # weight on mean vs CVaR
    alpha: float = 0.90         # CVaR confidence level
    delta: float = 0.10         # TV-distance budget
    allow_expansion: bool = False  # allow currently-zero species?
    solver_method: str = "SLSQP"
    solver_maxiter: int = 2000
    solver_ftol: float = 1e-12

# ---------------------------------------------------------------------------
# Sensitivity grid
# ---------------------------------------------------------------------------
DELTA_GRID: Sequence[float]  = (0.01, 0.05, 0.10, 0.20)
KAPPA_GRID: Sequence[float]  = (100.0, 300.0, 1000.0)
LAMBDA_GRID: Sequence[float] = (0.2, 0.5, 0.8)
ALPHA_GRID: Sequence[float]  = (0.80, 0.90, 0.95)

# ---------------------------------------------------------------------------
# Uncertainty (Dirichlet)
# ---------------------------------------------------------------------------
DIRICHLET_KAPPA: float = 300.0        # concentration parameter multiplier
N_DIRICHLET_DRAWS: int = 500

# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
FAIR_SHARE_CAP_PCT: float = 5.0       # max % of global gain per country

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 600
FIGURE_FORMAT: str = "png"  # also saved as PDF

# ---------------------------------------------------------------------------
# Causal guardrails
# ---------------------------------------------------------------------------
CAUSAL_FLAG: bool = False
CAUSAL_DISCLAIMER = (
    "Results are accounting counterfactuals / scenario-based analyses. "
    "They do NOT imply causal effects.  No covariates supporting causal "
    "identification are present in the data."
)
