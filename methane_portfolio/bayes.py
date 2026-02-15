"""Bayesian hierarchical model for species-level emission intensities.

Model specification
-------------------
Response: y = log(kg_co2e_per_ton_milk)  where I > 0 and share > 0.

Likelihood:
    y ~ StudentT(ν, μ, σ_s)

Linear predictor:
    μ_cts = α_s + u_c + β_s · (t − 2020) + γ_s · 1[t ≥ 2022]

Random effects:
    u_c ~ Normal(0, τ)    (country random effects → partial pooling)

Hyperpriors:
    ν      ~ Gamma(2, 0.1)    (degrees of freedom)
    α_s    ~ Normal(0, 5)     (species intercepts)
    β_s    ~ Normal(0, 1)     (species trends)
    γ_s    ~ Normal(0, 1)     (regime shift)
    σ_s    ~ HalfNormal(1)    (species-level scale)
    τ      ~ HalfNormal(1)    (country random effect scale)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from methane_portfolio import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_bayes_data(long_df: pd.DataFrame) -> dict:
    """Prepare arrays for PyMC model from the long dataframe.

    Filters to rows with kg_co2e_per_ton_milk > 0 and species_share > 0.
    """
    df = long_df[
        (long_df["kg_co2e_per_ton_milk"] > 0)
        & (long_df["species_share"] > 0)
    ].copy()

    # Encode species and country as integer indices
    species_list = sorted(df["milk_species"].unique())
    country_list = sorted(df["country_m49"].unique())
    species_idx = {s: i for i, s in enumerate(species_list)}
    country_idx = {c: i for i, c in enumerate(country_list)}

    df["species_id"] = df["milk_species"].map(species_idx).astype(int)
    df["country_id"] = df["country_m49"].map(country_idx).astype(int)
    df["t_centered"] = df["year"] - 2020
    df["regime"] = (df["year"] >= config.REGIME_SHIFT_YEAR).astype(float)
    df["log_intensity"] = np.log(df["kg_co2e_per_ton_milk"])

    return {
        "df": df,
        "species_list": species_list,
        "country_list": country_list,
        "n_species": len(species_list),
        "n_countries": len(country_list),
        "species_id": df["species_id"].values,
        "country_id": df["country_id"].values,
        "t_centered": df["t_centered"].values,
        "regime": df["regime"].values,
        "y": df["log_intensity"].values,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(data: dict) -> pm.Model:
    """Construct the PyMC hierarchical model."""
    with pm.Model() as model:
        # Data containers
        sp = pm.Data("species_id", data["species_id"], dims="obs")
        ct = pm.Data("country_id", data["country_id"], dims="obs")
        t_cen = pm.Data("t_centered", data["t_centered"], dims="obs")
        reg = pm.Data("regime", data["regime"], dims="obs")

        # Hyperpriors
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        tau = pm.HalfNormal("tau", sigma=1.0)

        # Species-level priors
        alpha_s = pm.Normal("alpha_s", mu=0, sigma=5, shape=data["n_species"])
        beta_s = pm.Normal("beta_s", mu=0, sigma=1, shape=data["n_species"])
        gamma_s = pm.Normal("gamma_s", mu=0, sigma=1, shape=data["n_species"])
        sigma_s = pm.HalfNormal("sigma_s", sigma=1.0, shape=data["n_species"])

        # Country random effects
        u_c = pm.Normal("u_c", mu=0, sigma=tau, shape=data["n_countries"])

        # Linear predictor
        mu = (
            alpha_s[sp]
            + u_c[ct]
            + beta_s[sp] * t_cen
            + gamma_s[sp] * reg
        )

        # Likelihood
        pm.StudentT(
            "y_obs",
            nu=nu,
            mu=mu,
            sigma=sigma_s[sp],
            observed=data["y"],
            dims="obs",
        )

    return model


# ---------------------------------------------------------------------------
# Sampling & diagnostics
# ---------------------------------------------------------------------------

def _detect_sampler() -> tuple[str | None, bool]:
    """Detect the best available sampler backend.

    Returns (nuts_sampler, use_nutpie) where nuts_sampler is the string
    to pass to pm.sample() and use_nutpie indicates if nutpie is available.
    """
    try:
        import nutpie  # noqa: F401
        logger.info("nutpie sampler detected — using Rust-based NUTS (fast)")
        return "nutpie", True
    except ImportError:
        logger.info(
            "nutpie not installed — falling back to PyMC default sampler. "
            "Install nutpie for 5-20× speedup: pip install nutpie"
        )
        return None, False


def fit_model(
    long_df: pd.DataFrame,
    *,
    chains: int = config.CHAINS,
    draws: int = config.DRAWS,
    tune: int = config.TUNE,
    target_accept: float = config.TARGET_ACCEPT,
    seed: int = config.PYMC_SEED,
    cores: int | None = None,
    output_dir: Path | None = None,
) -> tuple[az.InferenceData, dict]:
    """Fit the Bayesian model and export diagnostics.

    Parameters
    ----------
    cores : int, optional
        Number of CPU cores for parallel chain sampling.
        Defaults to min(chains, cpu_count).

    Returns
    -------
    idata : arviz InferenceData  (also saved as NetCDF)
    data  : dict from prepare_bayes_data (needed for prediction)
    """
    import multiprocessing
    import warnings

    # Suppress pytensor C-compiler warning on Windows
    warnings.filterwarnings(
        "ignore",
        message=".*g\\+\\+.*|.*cxx.*|.*Performance may be severely degraded.*",
        category=UserWarning,
    )

    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_bayes_data(long_df)
    model = build_model(data)

    # Auto-detect best sampler
    nuts_sampler, use_nutpie = _detect_sampler()
    if cores is None:
        cores = min(chains, multiprocessing.cpu_count())

    logger.info(
        "Sampling %d chains × %d draws (tune=%d, target_accept=%.2f, cores=%d, sampler=%s)",
        chains, draws, tune, target_accept, cores,
        nuts_sampler or "pymc",
    )

    # Suggest increasing chains if we have more cores available
    max_cores = multiprocessing.cpu_count()
    if chains < max_cores:
        logger.info(
            "TIP: You have %d CPUs but are only running %d chains. "
            "Increase --chains to %d to speed up sampling.",
            max_cores, chains, max_cores
        )

    sample_kwargs: dict = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
        cores=cores,
    )
    if nuts_sampler:
        sample_kwargs["nuts_sampler"] = nuts_sampler

    with model:
        idata = pm.sample(**sample_kwargs)
        # Posterior predictive
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=seed))

    # Save posterior NetCDF
    idata.to_netcdf(str(out / "bayes_posterior.nc"))

    # Diagnostics
    diag = _compute_diagnostics(idata)
    with open(out / "bayes_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, default=_json_default)

    # PPC summary
    ppc_summary = _ppc_summary(idata, data)
    ppc_summary.to_csv(out / "bayes_ppc_summary.csv", index=False)

    return idata, data


def _compute_diagnostics(idata: az.InferenceData) -> dict:
    """R-hat, ESS, divergences."""
    summary = az.summary(idata, var_names=["alpha_s", "beta_s", "gamma_s",
                                            "sigma_s", "tau", "nu"])
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum().item()),
        "summary_table": summary.to_dict(),
    }


def _ppc_summary(idata: az.InferenceData, data: dict) -> pd.DataFrame:
    """Quick posterior predictive check summary."""
    y_obs = data["y"]
    y_rep = idata.posterior_predictive["y_obs"].values  # (chain, draw, obs)
    y_rep_mean = y_rep.mean(axis=(0, 1))
    residuals = y_obs - y_rep_mean

    return pd.DataFrame({
        "obs_idx": np.arange(len(y_obs)),
        "y_obs": y_obs,
        "y_rep_mean": y_rep_mean,
        "residual": residuals,
        "within_90ci": (
            (y_obs >= np.percentile(y_rep, 5, axis=(0, 1)))
            & (y_obs <= np.percentile(y_rep, 95, axis=(0, 1)))
        ),
    })


# ---------------------------------------------------------------------------
# Posterior sample extraction
# ---------------------------------------------------------------------------

def posterior_intensity_samples(
    idata: az.InferenceData,
    data: dict,
    year: int = config.END_YEAR,
    n_samples: int | None = None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Draw posterior samples of I_cs for each country-species pair in a year.

    Returns
    -------
    I_samples : (n_samples, n_countries, n_species)  in original intensity scale
    country_list : list of country_m49 codes
    species_list : list of species names
    """
    posterior = idata.posterior
    alpha_s = posterior["alpha_s"].values  # (chain, draw, n_species)
    beta_s = posterior["beta_s"].values
    gamma_s = posterior["gamma_s"].values
    sigma_s = posterior["sigma_s"].values
    u_c = posterior["u_c"].values          # (chain, draw, n_countries)
    nu = posterior["nu"].values            # (chain, draw)

    # Flatten chains
    n_ch, n_dr = alpha_s.shape[:2]
    total = n_ch * n_dr
    alpha_s = alpha_s.reshape(total, -1)
    beta_s = beta_s.reshape(total, -1)
    gamma_s = gamma_s.reshape(total, -1)
    sigma_s = sigma_s.reshape(total, -1)
    u_c = u_c.reshape(total, -1)
    nu = nu.reshape(total)

    if n_samples and n_samples < total:
        rng = np.random.default_rng(config.RNG_SEED)
        idx = rng.choice(total, size=n_samples, replace=False)
        alpha_s, beta_s, gamma_s, sigma_s = (
            alpha_s[idx], beta_s[idx], gamma_s[idx], sigma_s[idx],
        )
        u_c = u_c[idx]
        nu = nu[idx]
        total = n_samples

    t_cen = year - 2020
    regime = float(year >= config.REGIME_SHIFT_YEAR)

    n_c = data["n_countries"]
    n_s = data["n_species"]

    # mu[sample, country, species]
    mu = (
        alpha_s[:, np.newaxis, :]       # (S, 1, n_species)
        + u_c[:, :, np.newaxis]         # (S, n_countries, 1)
        + beta_s[:, np.newaxis, :] * t_cen
        + gamma_s[:, np.newaxis, :] * regime
    )

    # Convert from log scale (mean prediction, no noise added)
    I_samples = np.exp(mu)

    return I_samples, data["country_list"], data["species_list"]


def _json_default(obj):
    """JSON serialiser fallback."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
