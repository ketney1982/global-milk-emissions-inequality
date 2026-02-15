"""Uncertainty propagation: Dirichlet share perturbation + posterior draws.

Combines two sources of uncertainty:
1. **Posterior draws** of species intensities I_cts from the Bayesian model.
2. **Dirichlet perturbation** of observed shares w_cts around
   Dir(κ · w_obs) to capture share-measurement uncertainty.

Also provides a sensitivity grid over (delta, kappa, lambda, alpha).
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from methane_portfolio import config
from methane_portfolio.robust_optimize import run_all_countries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dirichlet share draws
# ---------------------------------------------------------------------------

def dirichlet_shares(
    w_obs: np.ndarray,
    kappa: float = config.DIRICHLET_KAPPA,
    n_draws: int = config.N_DIRICHLET_DRAWS,
    seed: int = config.RNG_SEED,
) -> np.ndarray:
    """Draw species shares from Dir(kappa * w_obs).

    Parameters
    ----------
    w_obs : (S,) observed shares (summing to 1)
    kappa : concentration multiplier
    n_draws : number of draws

    Returns
    -------
    W_draws : (n_draws, S)
    """
    rng = np.random.default_rng(seed)
    # Avoid zero alpha values
    alpha = kappa * np.maximum(w_obs, 1e-10)
    return rng.dirichlet(alpha, size=n_draws)


# ---------------------------------------------------------------------------
# Combined uncertainty
# ---------------------------------------------------------------------------

def propagate_uncertainty(
    long_df: pd.DataFrame,
    I_samples: np.ndarray | None = None,
    country_list: list[int] | None = None,
    species_list: list[str] | None = None,
    *,
    year: int = config.END_YEAR,
    kappa: float = config.DIRICHLET_KAPPA,
    n_dirichlet: int = config.N_DIRICHLET_DRAWS,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute uncertainty summary per country.

    For each country, draws Dirichlet share perturbations and combines
    with intensity scenarios (posterior or lognormal fallback) to compute
    distributions of portfolio intensity.

    Returns
    -------
    DataFrame with columns: country_m49, country,
        mean_intensity, std_intensity, q05, q50, q95
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    sub = long_df[long_df["year"] == year].copy()
    all_species = sorted(sub["milk_species"].unique())
    S = len(all_species)

    rng = np.random.default_rng(config.RNG_SEED)
    rows = []

    countries = sub.groupby(["country_m49", "country"]).size().reset_index(name="_n")
    for _, crow in countries.iterrows():
        m49 = crow["country_m49"]
        cname = crow["country"]
        csub = sub[sub["country_m49"] == m49]

        w_obs = np.zeros(S)
        i_obs = np.zeros(S)
        for _, r in csub.iterrows():
            idx = all_species.index(r["milk_species"])
            w_obs[idx] = r["species_share"]
            i_obs[idx] = r["kg_co2e_per_ton_milk"]

        if w_obs.sum() < 1e-12:
            continue

        # Dirichlet share draws
        w_draws = dirichlet_shares(w_obs, kappa=kappa, n_draws=n_dirichlet)

        # Intensity scenarios
        if I_samples is not None and country_list is not None and species_list is not None:
            if m49 in country_list:
                cidx = country_list.index(m49)
                I_scen = np.zeros((I_samples.shape[0], S))
                for si, sp in enumerate(all_species):
                    if sp in species_list:
                        I_scen[:, si] = I_samples[:, cidx, species_list.index(sp)]
                    else:
                        I_scen[:, si] = i_obs[si]
            else:
                I_scen = _make_fallback(i_obs, w_obs, rng)
        else:
            I_scen = _make_fallback(i_obs, w_obs, rng)

        # Combine: for each Dirichlet draw, sample an intensity draw
        n_combined = min(n_dirichlet, I_scen.shape[0])
        intensities = np.array([
            w_draws[i] @ I_scen[i % I_scen.shape[0]]
            for i in range(n_combined)
        ])

        rows.append({
            "country_m49": m49,
            "country": cname,
            "mean_intensity_kg_co2e_per_t": float(intensities.mean()),
            "std_intensity_kg_co2e_per_t": float(intensities.std()),
            "q05_kg_co2e_per_t": float(np.percentile(intensities, 5)),
            "q50_kg_co2e_per_t": float(np.percentile(intensities, 50)),
            "q95_kg_co2e_per_t": float(np.percentile(intensities, 95)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "uncertainty_summary.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Sensitivity grid
# ---------------------------------------------------------------------------

def run_sensitivity_grid(
    long_df: pd.DataFrame,
    I_samples: np.ndarray | None = None,
    country_list: list[int] | None = None,
    species_list: list[str] | None = None,
    *,
    deltas: tuple[float, ...] = config.DELTA_GRID,
    kappas: tuple[float, ...] = config.KAPPA_GRID,
    lambdas: tuple[float, ...] = config.LAMBDA_GRID,
    alphas: tuple[float, ...] = config.ALPHA_GRID,
    year: int = config.END_YEAR,
    n_countries_max: int = 20,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the optimisation over a grid of hyper-parameters.

    To keep runtime tractable, only the top ``n_countries_max`` producers
    are included.

    Returns a tidy DataFrame saved to ``outputs/sensitivity_grid.csv``.
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Select top-producing countries
    sub = long_df[long_df["year"] == year]
    prod = (
        sub.groupby("country_m49")["milk_tonnes"]
        .sum()
        .nlargest(n_countries_max)
        .index.tolist()
    )
    sub_df = long_df[long_df["country_m49"].isin(prod)]

    all_rows = []
    total = len(deltas) * len(lambdas) * len(alphas)
    logger.info("Sensitivity grid: %d parameter combinations", total)

    for d, l, a in itertools.product(deltas, lambdas, alphas):
        res = run_all_countries(
            sub_df,
            I_samples=I_samples,
            country_list=country_list,
            species_list=species_list,
            year=year,
            lam=l, alpha=a, delta=d,
            output_dir=None,  # don't save intermediate
        )
        res["delta"] = d
        res["lambda"] = l
        res["alpha"] = a
        # Keep only summary columns
        keep_cols = [
            "country_m49", "country",
            "baseline_intensity_kg_co2e_per_t",
            "optimized_mean_kg_co2e_per_t",
            "optimized_cvar_kg_co2e_per_t",
            "reduction_mean_pct", "reduction_cvar_pct",
            "delta", "lambda", "alpha",
        ]
        all_rows.append(res[[c for c in keep_cols if c in res.columns]])

    grid_df = pd.concat(all_rows, ignore_index=True)
    grid_df.to_csv(out / "sensitivity_grid.csv", index=False)
    return grid_df


def _make_fallback(
    i_obs: np.ndarray,
    w_ref: np.ndarray,
    rng: np.random.Generator,
    n_draws: int = 500,
    cv: float = 0.2,
) -> np.ndarray:
    """Lognormal fallback for intensity scenarios."""
    S = len(i_obs)
    I_scen = np.zeros((n_draws, S))
    for s in range(S):
        if i_obs[s] > 0 and w_ref[s] > 0:
            mu_log = np.log(i_obs[s]) - 0.5 * np.log(1 + cv**2)
            sigma_log = np.sqrt(np.log(1 + cv**2))
            I_scen[:, s] = rng.lognormal(mu_log, sigma_log, n_draws)
        else:
            I_scen[:, s] = i_obs[s]
    return I_scen
