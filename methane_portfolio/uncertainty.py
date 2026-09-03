# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: ketney.otto@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Uncertainty propagation: Dirichlet share perturbation + posterior draws.

Combines two sources of uncertainty:
1. **Posterior draws** of species intensities I_cts from the Bayesian model.
2. **Dirichlet perturbation** of observed shares w_cts around
   Dir(Îş Â· w_obs) to capture share-measurement uncertainty.

Also provides a sensitivity grid over (delta, kappa, lambda, alpha).
"""

from __future__ import annotations

import multiprocessing
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from methane_portfolio import config
from methane_portfolio.robust_optimize import run_all_countries

logger = logging.getLogger(__name__)


def _pick_country_name(names: pd.Series) -> str:
    """Choose a stable country label for a country_m49 code."""
    clean = names.dropna().astype(str).str.strip()
    if clean.empty:
        return "Unknown"
    return str(clean.value_counts().idxmax())


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

    countries = (
        sub.groupby("country_m49", as_index=False)
        .agg(country=("country", _pick_country_name))
    )
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

_SENSITIVITY_KEEP_COLS = [
    "country_m49", "country",
    "production_tonnes", "observed_ch4_t",
    "baseline_intensity_observed",
    "baseline_intensity", "optimized_mean",
    "baseline_cvar", "optimized_cvar",
    "reduction_mean_pct", "reduction_cvar_pct",
    "abs_reduction_mt_ch4", "abs_reduction_mt_ch4_posterior",
    "tv_distance", "active_species",
    "no_harm_applied", "no_harm_action", "lp_certified",
    "delta", "lambda", "alpha",
]

# Summary columns for the per-configuration digest written alongside the grid.
_SENSITIVITY_SUMMARY_COLS = [
    "delta", "lambda", "alpha",
    "mean", "median", "q25", "q75", "cvar", "certified",
]


def _summarise_grid(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(delta, lambda, alpha) digest of a sensitivity grid.

    Reproduces Table 8 of the manuscript: mean, median and quartiles of the
    percentage reduction in posterior mean intensity, the mean CVaR reduction,
    and the number of certified solves.
    """
    if grid_df.empty:
        return pd.DataFrame(columns=_SENSITIVITY_SUMMARY_COLS)
    rows = []
    for (d, lam, a), g in grid_df.groupby(["delta", "lambda", "alpha"], sort=True):
        rows.append({
            "delta": d, "lambda": lam, "alpha": a,
            "mean": float(g["reduction_mean_pct"].mean()),
            "median": float(g["reduction_mean_pct"].median()),
            "q25": float(g["reduction_mean_pct"].quantile(0.25)),
            "q75": float(g["reduction_mean_pct"].quantile(0.75)),
            "cvar": float(g["reduction_cvar_pct"].mean()),
            "certified": int(g["lp_certified"].sum()) if "lp_certified" in g else int(len(g)),
        })
    return pd.DataFrame(rows, columns=_SENSITIVITY_SUMMARY_COLS)


def _run_sensitivity_combo(
    sub_df: pd.DataFrame,
    I_samples: np.ndarray | None,
    country_list: list[int] | None,
    species_list: list[str] | None,
    year: int,
    allow_expansion: bool,
    delta: float,
    lam: float,
    alpha: float,
) -> pd.DataFrame:
    """Execute one (delta, lambda, alpha) sensitivity run."""
    res = run_all_countries(
        sub_df,
        I_samples=I_samples,
        country_list=country_list,
        species_list=species_list,
        year=year,
        lam=lam,
        alpha=alpha,
        delta=delta,
        allow_expansion=allow_expansion,
        log_skips=False,
        save_csv=False,
    )
    res["delta"] = delta
    res["lambda"] = lam
    res["alpha"] = alpha
    return res[[c for c in _SENSITIVITY_KEEP_COLS if c in res.columns]]


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
    n_countries_max: int | None = None,
    allow_expansion: bool = False,
    workers: int | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the optimisation over a grid of hyper-parameters.

    Scope
    -----
    ``n_countries_max=None`` (the default since R3) evaluates the grid on the
    **whole analysed panel**, which is what the manuscript reports: 181 countries
    x 36 configurations = 6,516 country-configuration solves. R1 and R2 defaulted
    to ``n_countries_max=20`` for runtime reasons, so the public entry point did
    not regenerate the reported grid. That constraint no longer applies: under the
    exact LP formulation one full-panel solve takes ~1.5 s, so the entire grid runs
    in under a minute on one core.

    Passing an integer still restricts the grid to that many largest producers;
    the top-20 restriction is retained as a *derived summary*, not as the scope of
    the grid itself, for comparability with the R1 release.

    Outputs
    -------
    ``sensitivity_grid.csv``            tidy per-country, per-configuration results
    ``sensitivity_summary_all181.csv``  Table 8, full panel
    ``sensitivity_summary_top20.csv``   Table 8, restricted to the 20 largest producers
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    sub = long_df[long_df["year"] == year]
    prod_rank = sub.groupby("country_m49")["milk_tonnes"].sum().sort_values(ascending=False)
    if n_countries_max is None:
        sub_df = long_df
        logger.info(
            "Sensitivity grid scope: full panel (%d countries in %d)",
            int(prod_rank.size), year,
        )
    else:
        keep = prod_rank.nlargest(int(n_countries_max)).index.tolist()
        sub_df = long_df[long_df["country_m49"].isin(keep)]
        logger.info("Sensitivity grid scope: top %d producers", int(n_countries_max))
    top20 = prod_rank.nlargest(20).index.tolist()

    all_rows = []
    combos = list(itertools.product(deltas, lambdas, alphas))
    total = len(combos)

    if workers is None:
        workers = min(max(multiprocessing.cpu_count() - 1, 1), total)
    workers = max(int(workers), 1) if total > 0 else 1
    logger.info(
        "Sensitivity grid: %d parameter combinations (workers=%d)",
        total,
        workers,
    )

    if workers == 1:
        for d, l, a in combos:
            all_rows.append(
                _run_sensitivity_combo(
                    sub_df,
                    I_samples,
                    country_list,
                    species_list,
                    year,
                    allow_expansion,
                    d,
                    l,
                    a,
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_sensitivity_combo,
                    sub_df,
                    I_samples,
                    country_list,
                    species_list,
                    year,
                    allow_expansion,
                    d,
                    l,
                    a,
                )
                for d, l, a in combos
            ]
            for idx, fut in enumerate(as_completed(futures), start=1):
                all_rows.append(fut.result())
                if idx == total or idx % 6 == 0:
                    logger.info("Sensitivity grid progress: %d/%d combinations", idx, total)

    if all_rows:
        grid_df = pd.concat(all_rows, ignore_index=True)
    else:
        grid_df = pd.DataFrame(columns=_SENSITIVITY_KEEP_COLS)
    if not grid_df.empty:
        grid_df.sort_values(
            ["delta", "lambda", "alpha", "country_m49"],
            inplace=True,
            kind="mergesort",
        )
        grid_df.reset_index(drop=True, inplace=True)
    grid_df.to_csv(out / "sensitivity_grid.csv", index=False)

    summary_all = _summarise_grid(grid_df)
    summary_all.to_csv(out / "sensitivity_summary_all181.csv", index=False)
    if not grid_df.empty:
        summary_top = _summarise_grid(grid_df[grid_df["country_m49"].isin(top20)])
    else:
        summary_top = _summarise_grid(grid_df)
    summary_top.to_csv(out / "sensitivity_summary_top20.csv", index=False)

    n_countries = int(grid_df["country_m49"].nunique()) if not grid_df.empty else 0
    logger.info(
        "Sensitivity grid: %d rows = %d countries x %d configurations",
        len(grid_df), n_countries, len(combos),
    )
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
