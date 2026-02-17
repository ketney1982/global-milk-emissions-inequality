# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Bayesian hierarchical model for species-level emission intensities.

Model specification
-------------------
Response: y = log(kg_co2e_per_ton_milk), filtered to I > 0 and share > 0.

Likelihood:
    y ~ StudentT(nu, mu, sigma_s)

Linear predictor:
    mu_cts = alpha_s + u_c + beta_s * (t - 2020) + gamma_s * 1[t >= 2022]

Random effects:
    u_c ~ ZeroSumNormal(0, tau)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

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

def _safe_std(values: np.ndarray, fallback: float) -> float:
    """Finite standard deviation helper with fallback."""
    if values.size <= 1:
        return fallback
    sd = float(np.std(values, ddof=1))
    if np.isfinite(sd) and sd > 0:
        return sd
    return fallback


def _country_intercept_scale(data: dict, y_sd: float) -> float:
    """Estimate a weakly informative prior scale for country random effects."""
    country_means = (
        data["df"]
        .groupby("country_id", observed=True)["log_intensity"]
        .mean()
        .to_numpy(dtype=float)
    )
    empirical_sd = _safe_std(country_means, fallback=max(y_sd, 0.25))
    return float(np.clip(empirical_sd, 0.15, 1.0))


def build_model(data: dict) -> pm.Model:
    """Construct the PyMC hierarchical model."""
    y_mean = float(np.mean(data["y"]))
    y_sd = _safe_std(data["y"], fallback=1.0)
    alpha_sigma = float(np.clip(1.5 * max(y_sd, 0.25), 0.5, 2.5))
    tau_sigma = _country_intercept_scale(data, y_sd=y_sd)

    with pm.Model() as model:
        # Data containers
        sp = pm.Data("species_id", data["species_id"], dims="obs")
        ct = pm.Data("country_id", data["country_id"], dims="obs")
        t_cen = pm.Data("t_centered", data["t_centered"], dims="obs")
        reg = pm.Data("regime", data["regime"], dims="obs")

        # Hyperpriors
        nu = pm.Gamma("nu", alpha=6.0, beta=1.0)
        tau = pm.HalfNormal("tau", sigma=tau_sigma)

        # Species-level priors
        alpha_s = pm.Normal(
            "alpha_s",
            mu=y_mean,
            sigma=alpha_sigma,
            shape=data["n_species"],
        )
        beta_s = pm.Normal("beta_s", mu=0, sigma=0.5, shape=data["n_species"])
        gamma_s = pm.Normal("gamma_s", mu=0, sigma=0.5, shape=data["n_species"])
        sigma_s = pm.HalfNormal("sigma_s", sigma=0.5, shape=data["n_species"])

        # Non-centred zero-sum country effects reduce funnel geometry between
        # tau and country intercepts, improving convergence stability.
        u_c_raw = pm.ZeroSumNormal("u_c_raw", sigma=1.0, shape=data["n_countries"])
        u_c = pm.Deterministic("u_c", tau * u_c_raw)

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
        logger.info("nutpie sampler detected - using Rust-based NUTS (fast)")
        return "nutpie", True
    except ImportError:
        logger.info(
            "nutpie not installed - falling back to PyMC default sampler. "
            "Install nutpie for 5-20x speedup: pip install nutpie"
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
    fail_on_weak_convergence: bool = True,
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
    warnings.filterwarnings(
        "ignore",
        message="ArviZ is undergoing a major refactor.*",
        category=FutureWarning,
    )

    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_bayes_data(long_df)
    model = build_model(data)

    # Auto-detect best sampler
    nuts_sampler, use_nutpie = _detect_sampler()
    if cores is None:
        cores = min(chains, multiprocessing.cpu_count())

    if draws < config.DRAWS or tune < config.TUNE or target_accept < config.TARGET_ACCEPT:
        logger.warning(
            "Sampling settings below recommended defaults (draws=%d, tune=%d, target_accept=%.2f). "
            "Low settings can produce weak ESS/R-hat diagnostics.",
            config.DRAWS,
            config.TUNE,
            config.TARGET_ACCEPT,
        )

    logger.info(
        "Sampling %d chains x %d draws (tune=%d, target_accept=%.2f, cores=%d, sampler=%s)",
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
        init="jitter+adapt_diag",
        compute_convergence_checks=True,
    )
    if nuts_sampler:
        sample_kwargs["nuts_sampler"] = nuts_sampler
    # nutpie currently ignores idata_kwargs and emits a warning; avoid noisy output.
    if use_nutpie:
        sample_kwargs.pop("idata_kwargs", None)
    else:
        sample_kwargs["nuts"] = {"max_treedepth": 15}

    with model:
        idata = pm.sample(**sample_kwargs)

    # Save posterior NetCDF
    idata.to_netcdf(str(out / "bayes_posterior.nc"))

    # Diagnostics
    diag = _compute_diagnostics(idata)
    with open(out / "bayes_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, default=_json_default)
    if not diag.get("converged", False):
        logger.warning(
            "Bayesian diagnostics indicate weak convergence: max_rhat=%.3f, min_ess_bulk=%.1f, min_ess_tail=%.1f, divergences=%d",
            diag["max_rhat"],
            diag["min_ess_bulk"],
            diag["min_ess_tail"],
            diag["divergences"],
        )
        if diag.get("rhat_fail_params"):
            logger.warning(
                "R-hat >= %.2f parameters (first %d): %s",
                config.BAYES_RHAT_THRESHOLD,
                config.BAYES_DIAG_MAX_REPORT_PARAMS,
                ", ".join(diag["rhat_fail_params"]),
            )
        if diag.get("ess_bulk_fail_params") or diag.get("ess_tail_fail_params"):
            logger.warning(
                "ESS < %d parameters (bulk/tail, first %d): %s / %s",
                config.BAYES_ESS_MIN,
                config.BAYES_DIAG_MAX_REPORT_PARAMS,
                ", ".join(diag.get("ess_bulk_fail_params", [])) or "none",
                ", ".join(diag.get("ess_tail_fail_params", [])) or "none",
            )
    if fail_on_weak_convergence and not diag.get("converged_relaxed", False):
        raise RuntimeError(
            "Bayesian posterior failed relaxed convergence checks "
            f"(max_rhat={diag['max_rhat']:.3f}, min_ess_bulk={diag['min_ess_bulk']:.1f}, "
            f"min_ess_tail={diag['min_ess_tail']:.1f}, divergences={diag['divergences']}). "
            "Refit with stronger settings or rerun with allow_weak_convergence=True if intentional."
        )

    # Posterior predictive diagnostics (draw-capped for tractable runtime)
    ppc_idata = _posterior_for_ppc(idata, max_draws=config.BAYES_PPC_MAX_DRAWS)
    with model:
        ppc_idata = pm.sample_posterior_predictive(ppc_idata, random_seed=seed)
    ppc_summary = _ppc_summary(ppc_idata, data)
    ppc_summary.to_csv(out / "bayes_ppc_summary.csv", index=False)
    ppc_outliers = _ppc_outliers(ppc_summary, data, top_n=25)
    ppc_outliers.to_csv(out / "bayes_ppc_outliers.csv", index=False)
    ppc_diag = _ppc_diagnostics(ppc_summary)
    with open(out / "bayes_ppc_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(ppc_diag, f, indent=2, default=_json_default)
    if abs(ppc_diag["residual_mean"]) > config.BAYES_PPC_MEAN_BIAS_WARN:
        logger.warning(
            "PPC residual mean is %.3f (>|%.3f|). Median=%.3f, 90%% coverage=%.2f%%. Inspect high-residual observations.",
            ppc_diag["residual_mean"],
            config.BAYES_PPC_MEAN_BIAS_WARN,
            ppc_diag["residual_median"],
            100.0 * ppc_diag["coverage_90ci"],
        )

    return idata, data


def _posterior_for_ppc(
    idata: az.InferenceData,
    *,
    max_draws: int,
) -> az.InferenceData:
    """Return posterior subset used for posterior-predictive diagnostics."""
    n_chain = int(idata.posterior.sizes.get("chain", 1))
    n_draw = int(idata.posterior.sizes.get("draw", 1))
    total_draws = int(n_chain * n_draw)
    max_draws = int(max(1, max_draws))
    if total_draws <= max_draws:
        return idata

    draws_per_chain = max(1, max_draws // n_chain)
    if draws_per_chain >= n_draw:
        return idata

    rng = np.random.default_rng(config.RNG_SEED)
    draw_idx = np.sort(rng.choice(n_draw, size=draws_per_chain, replace=False))
    logger.info(
        "Posterior predictive sampling using %d/%d posterior draws",
        int(draws_per_chain * n_chain),
        total_draws,
    )
    return idata.sel(draw=draw_idx)


def _compute_diagnostics(idata: az.InferenceData) -> dict:
    """R-hat, ESS, divergences."""
    summary = az.summary(idata, var_names=["alpha_s", "beta_s", "gamma_s",
                                            "sigma_s", "tau", "nu"])
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())
    divergences = int(idata.sample_stats["diverging"].sum().item())

    strict_rhat = config.BAYES_RHAT_THRESHOLD
    strict_ess = config.BAYES_ESS_MIN
    relaxed_rhat = config.BAYES_RHAT_THRESHOLD_RELAXED
    relaxed_ess = config.BAYES_ESS_MIN_RELAXED
    max_report = config.BAYES_DIAG_MAX_REPORT_PARAMS

    rhat_fail = summary.index[summary["r_hat"] >= strict_rhat].tolist()
    ess_bulk_fail = summary.index[summary["ess_bulk"] < strict_ess].tolist()
    ess_tail_fail = summary.index[summary["ess_tail"] < strict_ess].tolist()

    strict_ok = bool(
        max_rhat < strict_rhat
        and min_ess_bulk >= strict_ess
        and min_ess_tail >= strict_ess
        and divergences == 0
    )
    relaxed_ok = bool(
        max_rhat < relaxed_rhat
        and min_ess_bulk >= relaxed_ess
        and min_ess_tail >= relaxed_ess
        and divergences == 0
    )

    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
        "thresholds": {
            "rhat_strict": strict_rhat,
            "rhat_relaxed": relaxed_rhat,
            "ess_strict": strict_ess,
            "ess_relaxed": relaxed_ess,
        },
        "converged": strict_ok,
        "converged_relaxed": relaxed_ok,
        "n_rhat_fail": int(len(rhat_fail)),
        "n_ess_bulk_fail": int(len(ess_bulk_fail)),
        "n_ess_tail_fail": int(len(ess_tail_fail)),
        "rhat_fail_params": rhat_fail[:max_report],
        "ess_bulk_fail_params": ess_bulk_fail[:max_report],
        "ess_tail_fail_params": ess_tail_fail[:max_report],
        "worst_rhat_param": str(summary["r_hat"].idxmax()),
        "worst_ess_bulk_param": str(summary["ess_bulk"].idxmin()),
        "worst_ess_tail_param": str(summary["ess_tail"].idxmin()),
        "summary_table": summary.to_dict(),
    }


def _ppc_summary(idata: az.InferenceData, data: dict) -> pd.DataFrame:
    """Quick posterior predictive check summary."""
    y_obs = data["y"]
    y_rep = idata.posterior_predictive["y_obs"].values  # (chain, draw, obs)
    n_chain, n_draw, n_obs = y_rep.shape
    total_draws = int(n_chain * n_draw)
    max_draws = int(max(1, config.BAYES_PPC_MAX_DRAWS))

    # Large posterior-predictive arrays can dominate runtime and memory.
    # Use a reproducible subsample of draws for PPC diagnostics when needed.
    y_rep_flat = y_rep.reshape(total_draws, n_obs)
    if total_draws > max_draws:
        rng = np.random.default_rng(config.RNG_SEED)
        idx = rng.choice(total_draws, size=max_draws, replace=False)
        y_rep_eval = y_rep_flat[idx, :]
        logger.info(
            "PPC diagnostics using %d/%d posterior predictive draws",
            max_draws,
            total_draws,
        )
    else:
        y_rep_eval = y_rep_flat

    y_rep_mean = y_rep_eval.mean(axis=0)
    y_rep_median = np.median(y_rep_eval, axis=0)
    y_rep_p05 = np.percentile(y_rep_eval, 5, axis=0)
    y_rep_p95 = np.percentile(y_rep_eval, 95, axis=0)
    residuals = y_obs - y_rep_mean

    return pd.DataFrame({
        "obs_idx": np.arange(len(y_obs)),
        "y_obs": y_obs,
        "y_rep_mean": y_rep_mean,
        "y_rep_median": y_rep_median,
        "y_rep_p05": y_rep_p05,
        "y_rep_p95": y_rep_p95,
        "residual": residuals,
        "residual_median_pred": y_obs - y_rep_median,
        "within_90ci": (
            (y_obs >= y_rep_p05)
            & (y_obs <= y_rep_p95)
        ),
    })


def _ppc_outliers(
    ppc_summary: pd.DataFrame,
    data: dict,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    """Join PPC residuals with source rows and keep largest absolute errors."""
    df = data["df"].reset_index(drop=True).copy()
    ppc = ppc_summary.reset_index(drop=True).copy()
    ppc["obs_idx"] = np.arange(ppc.shape[0], dtype=int)
    merged = pd.concat([ppc, df], axis=1)
    merged["abs_residual"] = merged["residual"].abs()
    cols = [
        "obs_idx",
        "country_m49",
        "country",
        "year",
        "milk_species",
        "species_share",
        "kg_co2e_per_ton_milk",
        "y_obs",
        "y_rep_mean",
        "y_rep_median",
        "y_rep_p05",
        "y_rep_p95",
        "residual",
        "residual_median_pred",
        "abs_residual",
        "within_90ci",
    ]
    available_cols = [c for c in cols if c in merged.columns]
    out = merged.sort_values("abs_residual", ascending=False)
    return out.loc[:, available_cols].head(top_n).reset_index(drop=True)


def _trimmed_mean(values: np.ndarray, trim_frac: float = 0.10) -> float:
    """Return symmetric trimmed mean."""
    if values.size == 0:
        return float("nan")
    arr = np.sort(np.asarray(values, dtype=float))
    k = int(np.floor(trim_frac * arr.size))
    if 2 * k >= arr.size:
        return float(np.mean(arr))
    return float(np.mean(arr[k:arr.size - k]))


def _ppc_diagnostics(ppc_summary: pd.DataFrame) -> dict:
    """Aggregate diagnostics for posterior predictive residuals."""
    residual = ppc_summary["residual"].to_numpy(dtype=float)
    abs_residual = np.abs(residual)
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_trimmed_mean_10pct": _trimmed_mean(residual, trim_frac=0.10),
        "residual_max_abs": float(np.max(abs_residual)),
        "residual_q95_abs": float(np.percentile(abs_residual, 95)),
        "coverage_90ci": float(ppc_summary["within_90ci"].mean()),
        "n_abs_residual_gt_2": int((abs_residual > 2.0).sum()),
        "n_abs_residual_gt_3": int((abs_residual > 3.0).sum()),
        "n_obs": int(ppc_summary.shape[0]),
    }


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
