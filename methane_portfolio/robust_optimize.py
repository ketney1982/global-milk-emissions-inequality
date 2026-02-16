# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Robust portfolio optimisation with CVaR risk measure.

For each country we solve:
    minimise  Î» Â· E[I'(w)]  +  (1â’Î») Â· CVaR_Î±(I'(w))
    s.t.      ÎŁ w_s = 1
              w_s â‰Ą 0
              0.5 Â· â€–w â’ w_refâ€–â‚  â‰¤  Î´
              w_s = 0   for species not in current mix (unless allow_expansion)

CVaR is handled via the Rockafellarâ€“Uryasev linear relaxation:
    CVaR_Î± â‰ min_t  { t  +  (1/(K(1â’Î±))) Â· ÎŁ_k max(0, I'_k â’ t) }
which is folded into the objective.

NOTE: all results are labelled as **accounting counterfactuals** per the
causal guardrails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from methane_portfolio import config
from methane_portfolio.optimize import (
    mean_intensity,
    tv_distance_constraint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RobustResult:
    country_m49: int
    country: str
    w_baseline: np.ndarray
    w_optimal: np.ndarray
    species: list[str]
    baseline_intensity: float
    optimized_mean: float
    optimized_cvar: float
    reduction_mean_pct: float
    reduction_cvar_pct: float
    delta: float
    lam: float
    alpha: float
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_robust(
    w_ref: np.ndarray,
    I_scenarios: np.ndarray,
    *,
    lam: float = 0.5,
    alpha: float = 0.90,
    delta: float = 0.10,
    allow_expansion: bool = False,
    baseline_ceiling: float | None = None,
    no_harm_tol: float = 1e-10,
    method: str = "SLSQP",
    maxiter: int = 2000,
    ftol: float = 1e-12,
) -> dict:
    """Solve the robust species-portfolio problem for one country.

    Parameters
    ----------
    w_ref : (S,)  baseline species shares
    I_scenarios : (K, S)  posterior draws of species intensities
    lam : weight on mean (1-lam on CVaR)
    alpha : CVaR confidence level
    delta : TV-distance budget
    allow_expansion : allow species with w_ref=0
    baseline_ceiling : optional upper bound for E[I'(w)] (do-no-harm)

    Returns
    -------
    dict with keys: w_opt, mean_opt, cvar_opt, success, message
    """
    S = len(w_ref)
    K = I_scenarios.shape[0]

    # Bounds: w_s >= 0; if no expansion, w_s = 0 where w_ref = 0
    bounds = []
    for s in range(S):
        if not allow_expansion and w_ref[s] == 0.0:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, 1.0))

    # Decision: x = [w_1..w_S, t]  where t is the CVaR auxiliary
    x0 = np.concatenate([w_ref.copy(), [0.0]])  # initial: baseline + t=0
    n_var = S + 1

    # Extended bounds
    ext_bounds = list(bounds) + [(-1e6, 1e6)]  # t is unconstrained

    def objective(x):
        w = x[:S]
        t = x[S]
        portfolio_vals = I_scenarios @ w  # (K,)
        mean_val = portfolio_vals.mean()
        # CVaR upper bound via Rockafellar-Uryasev
        shortfall = np.maximum(0.0, portfolio_vals - t)
        cvar_val = t + shortfall.sum() / (K * (1 - alpha))
        return lam * mean_val + (1 - lam) * cvar_val

    def objective_jac(x):
        """Analytical gradient for speed."""
        w = x[:S]
        t = x[S]
        portfolio_vals = I_scenarios @ w
        # d(mean)/dw
        d_mean_dw = I_scenarios.mean(axis=0)
        # d(CVaR)/dw and d(CVaR)/dt
        indicator = (portfolio_vals > t).astype(float)  # (K,)
        coeff = 1.0 / (K * (1 - alpha))
        d_cvar_dw = coeff * (indicator[:, None] * I_scenarios).sum(axis=0)
        d_cvar_dt = 1.0 - coeff * indicator.sum()
        grad_w = lam * d_mean_dw + (1 - lam) * d_cvar_dw
        grad_t = (1 - lam) * d_cvar_dt
        return np.concatenate([grad_w, [grad_t]])

    # Constraints
    # 1. Simplex: ÎŁ w_s = 1  (only first S variables)
    simplex = {
        "type": "eq",
        "fun": lambda x: x[:S].sum() - 1.0,
        "jac": lambda x: np.concatenate([np.ones(S), [0.0]]),
    }
    # 2. TV distance
    tv_base = tv_distance_constraint(w_ref, delta)
    tv = {
        "type": "ineq",
        "fun": lambda x: tv_base["fun"](x[:S]),
    }
    constraints: list[dict] = [simplex, tv]
    if baseline_ceiling is not None:
        species_mean = I_scenarios.mean(axis=0)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(
                    baseline_ceiling - (species_mean @ x[:S]) + no_harm_tol
                ),
                "jac": lambda x: np.concatenate([-species_mean, [0.0]]),
            },
        )

    result = minimize(
        objective,
        x0,
        jac=objective_jac,
        method=method,
        bounds=ext_bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    w_opt = result.x[:S]
    if not np.all(np.isfinite(w_opt)):
        w_opt = w_ref.copy()

    # Numerical guardrails: preserve feasible simplex/bounds after optimizer noise.
    w_opt = np.clip(w_opt, 0.0, None)
    if not allow_expansion:
        w_opt[w_ref == 0.0] = 0.0

    w_sum = float(w_opt.sum())
    if w_sum <= 0:
        w_opt = w_ref.copy()
    else:
        w_opt = w_opt / w_sum

    tv_dist = 0.5 * float(np.abs(w_opt - w_ref).sum())
    if tv_dist > delta and tv_dist > 0:
        # Project by shrinking the move toward baseline; keeps simplex + non-negativity.
        step = delta / tv_dist
        w_opt = w_ref + step * (w_opt - w_ref)

    # Evaluate at optimum
    port_opt = I_scenarios @ w_opt
    mean_opt = port_opt.mean()
    t_opt = result.x[S]
    cvar_opt = t_opt + np.maximum(0.0, port_opt - t_opt).sum() / (K * (1 - alpha))

    return {
        "w_opt": w_opt,
        "mean_opt": float(mean_opt),
        "cvar_opt": float(cvar_opt),
        "success": result.success,
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_countries(
    long_df: pd.DataFrame,
    I_samples: np.ndarray | None = None,
    country_list: list[int] | None = None,
    species_list: list[str] | None = None,
    *,
    year: int = config.END_YEAR,
    lam: float = 0.5,
    alpha: float = 0.90,
    delta: float = 0.10,
    allow_expansion: bool = False,
    do_no_harm: bool = config.OptimConfig.do_no_harm,
    no_harm_tol: float = config.OptimConfig.no_harm_tol,
    log_skips: bool = True,
    output_dir: Path | None = None,
    save_csv: bool = True,
    save_audit: bool = True,
) -> pd.DataFrame:
    """Run robust optimisation for all countries with data in ``year``.

    When ``I_samples`` is None (no Bayesian posterior), uses lognormal
    fallback from observed data.

    Parameters
    ----------
    long_df : species-level data
    I_samples : (n_draws, n_countries, n_species) posterior samples
    country_list : countries matching I_samples dim-1
    species_list : species matching I_samples dim-2
    year : reference year
    lam, alpha, delta : optimisation parameters
    allow_expansion : allow zero-share species
    do_no_harm : enforce optimized_mean <= baseline_intensity
    output_dir : where to save results
    save_csv : if False, skip writing robust_optimization_results.csv
    save_audit : if True, write robust_optimization_audit.json alongside CSV

    Returns
    -------
    DataFrame with per-country optimisation results
    """
    out: Path | None = None
    if save_csv:
        out = output_dir or config.OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)

    sub = long_df[long_df["year"] == year].copy()
    all_species = sorted(sub["milk_species"].unique())

    results: list[dict] = []
    skipped_nan_intensity: list[tuple[str, int]] = []
    fixed_low_species: list[tuple[str, int, int]] = []
    expansion_disabled_no_posterior: list[tuple[str, int]] = []
    reverted_no_harm: list[tuple[str, int, float, float, str]] = []

    # Get unique countries
    countries = sub.groupby(["country_m49", "country"]).size().reset_index(name="_n")

    for _, crow in countries.iterrows():
        m49 = crow["country_m49"]
        cname = crow["country"]

        csub = sub[sub["country_m49"] == m49]
        invalid_active = csub[
            (csub["species_share"] > 0)
            & (csub["kg_co2e_per_ton_milk"].isna())
        ]
        if not invalid_active.empty:
            skipped_nan_intensity.append((str(cname), int(m49)))
            continue

        # Build w_ref aligned to all_species
        w_ref = np.zeros(len(all_species))
        i_obs = np.zeros(len(all_species))
        for _, row in csub.iterrows():
            idx = all_species.index(row["milk_species"])
            w_ref[idx] = row["species_share"]
            i_obs[idx] = (
                0.0 if pd.isna(row["kg_co2e_per_ton_milk"])
                else row["kg_co2e_per_ton_milk"]
            )

        active_species = int(np.sum(w_ref > 0))

        # Get I_scenarios for this country
        has_country_posterior = False
        if I_samples is not None and country_list is not None and species_list is not None:
            if m49 in country_list:
                has_country_posterior = True
                cidx = country_list.index(m49)
                # Align species
                I_scen = np.zeros((I_samples.shape[0], len(all_species)))
                for si, sp in enumerate(all_species):
                    if sp in species_list:
                        I_scen[:, si] = I_samples[:, cidx, species_list.index(sp)]
                    elif i_obs[si] > 0:
                        # Fallback: point estimate
                        I_scen[:, si] = i_obs[si]
            else:
                I_scen = _lognormal_fallback(i_obs, w_ref)
        else:
            I_scen = _lognormal_fallback(i_obs, w_ref)

        effective_allow_expansion = allow_expansion and has_country_posterior
        if allow_expansion and not has_country_posterior:
            expansion_disabled_no_posterior.append((str(cname), int(m49)))

        baseline = float(w_ref @ i_obs) if i_obs.sum() > 0 else 0.0
        if baseline <= 0 or w_ref.sum() < 1e-12:
            continue

        # Baseline CVaR
        port_base = I_scen @ w_ref
        t_base = np.percentile(port_base, alpha * 100)
        cvar_base = t_base + np.maximum(0.0, port_base - t_base).sum() / (
            len(port_base) * (1 - alpha)
        )

        if active_species < 2 and not effective_allow_expansion:
            # No decision freedom: keep baseline mix and include country in outputs.
            fixed_low_species.append((str(cname), int(m49), active_species))
            sol_raw = {
                "w_opt": w_ref.copy(),
                "mean_opt": baseline,
                "cvar_opt": float(cvar_base),
                "success": True,
                "message": "Fixed baseline (insufficient active species or expansion unavailable)",
            }
            sol_final = sol_raw.copy()
            no_harm_applied = False
            no_harm_action = "not_applicable_fixed_baseline"
        else:
            sol_raw = solve_robust(
                w_ref, I_scen,
                lam=lam, alpha=alpha, delta=delta,
                allow_expansion=effective_allow_expansion,
                baseline_ceiling=None,
                no_harm_tol=no_harm_tol,
            )
            sol_final = sol_raw
            no_harm_applied = False
            no_harm_action = "not_needed"

            if do_no_harm and sol_raw["mean_opt"] > baseline + no_harm_tol:
                no_harm_applied = True
                sol_constrained = solve_robust(
                    w_ref, I_scen,
                    lam=lam, alpha=alpha, delta=delta,
                    allow_expansion=effective_allow_expansion,
                    baseline_ceiling=baseline,
                    no_harm_tol=no_harm_tol,
                )
                if sol_constrained["mean_opt"] <= baseline + no_harm_tol:
                    sol_final = sol_constrained
                    no_harm_action = "constrained_solution"
                else:
                    sol_final = {
                        "w_opt": w_ref.copy(),
                        "mean_opt": baseline,
                        "cvar_opt": float(cvar_base),
                        "success": True,
                        "message": (
                            "Reverted to baseline (do-no-harm guard: "
                            f"raw_mean_opt={sol_raw['mean_opt']:.6g} > baseline={baseline:.6g})"
                        ),
                    }
                    no_harm_action = "baseline_revert"
                reverted_no_harm.append(
                    (
                        str(cname),
                        int(m49),
                        float(sol_raw["mean_opt"]),
                        float(baseline),
                        no_harm_action,
                    ),
                )
            elif not do_no_harm:
                no_harm_action = "disabled"

        red_mean_raw = (1.0 - sol_raw["mean_opt"] / baseline) * 100
        red_cvar_raw = (1.0 - sol_raw["cvar_opt"] / cvar_base) * 100 if cvar_base > 0 else 0.0
        red_mean = (1.0 - sol_final["mean_opt"] / baseline) * 100
        red_cvar = (1.0 - sol_final["cvar_opt"] / cvar_base) * 100 if cvar_base > 0 else 0.0
        production_tonnes = float(csub["milk_tonnes"].sum())
        absolute_reduction_kt = (
            (baseline - sol_final["mean_opt"]) * production_tonnes / 1e6
        )
        absolute_reduction_kt_raw = (
            (baseline - sol_raw["mean_opt"]) * production_tonnes / 1e6
        )

        row_dict = {
            "country_m49": m49,
            "country": cname,
            "production_tonnes": production_tonnes,
            "baseline_intensity": baseline,
            "baseline_intensity_kg_co2e_per_t": baseline,
            "raw_optimized_mean": sol_raw["mean_opt"],
            "raw_optimized_mean_kg_co2e_per_t": sol_raw["mean_opt"],
            "raw_optimized_cvar": sol_raw["cvar_opt"],
            "raw_optimized_cvar_kg_co2e_per_t": sol_raw["cvar_opt"],
            "raw_reduction_mean_pct": red_mean_raw,
            "raw_reduction_cvar_pct": red_cvar_raw,
            "raw_absolute_reduction_kt": absolute_reduction_kt_raw,
            "optimized_mean": sol_final["mean_opt"],
            "optimized_mean_kg_co2e_per_t": sol_final["mean_opt"],
            "optimized_cvar": sol_final["cvar_opt"],
            "optimized_cvar_kg_co2e_per_t": sol_final["cvar_opt"],
            "reduction_pct": red_mean,
            "reduction_mean_pct": red_mean,
            "reduction_cvar_pct": red_cvar,
            "absolute_reduction_kt": absolute_reduction_kt,
            "no_harm_enabled": bool(do_no_harm),
            "no_harm_applied": bool(no_harm_applied),
            "no_harm_action": no_harm_action,
            "no_harm_excess_raw": float(max(0.0, sol_raw["mean_opt"] - baseline)),
            "delta": delta,
            "lambda": lam,
            "alpha": alpha,
            "solver_success_raw": sol_raw["success"],
            "solver_message_raw": str(sol_raw.get("message", "")),
            "solver_success": sol_final["success"],
            "solver_message": str(sol_final.get("message", "")),
        }
        # Add optimal weights
        for si, sp in enumerate(all_species):
            row_dict[f"w_opt_{sp}"] = sol_final["w_opt"][si]
            row_dict[f"w_base_{sp}"] = w_ref[si]

        results.append(row_dict)

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values("reduction_mean_pct", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    if save_csv and out is not None:
        df.to_csv(out / "robust_optimization_results.csv", index=False)
        if save_audit:
            trigger_df = df[df["no_harm_applied"] == True]  # noqa: E712
            audit = {
                "do_no_harm_enabled": bool(do_no_harm),
                "n_countries": int(len(df)),
                "n_no_harm_applied": int(len(trigger_df)),
                "n_negative_raw_reductions": int((df["raw_reduction_mean_pct"] < 0).sum()),
                "n_negative_final_reductions": int((df["reduction_mean_pct"] < 0).sum()),
                "total_raw_absolute_reduction_kt": float(df["raw_absolute_reduction_kt"].sum()),
                "total_final_absolute_reduction_kt": float(df["absolute_reduction_kt"].sum()),
                "no_harm_actions": (
                    trigger_df["no_harm_action"].value_counts(dropna=False).to_dict()
                    if not trigger_df.empty
                    else {}
                ),
                "countries_no_harm_applied": trigger_df[
                    ["country_m49", "country", "no_harm_action", "no_harm_excess_raw"]
                ].to_dict(orient="records"),
                "note": (
                    "Raw optimisation outputs are preserved in raw_* columns. "
                    "Final outputs may apply do-no-harm guard but never alter source input data."
                ),
            }
            (out / "robust_optimization_audit.json").write_text(
                json.dumps(audit, indent=2),
                encoding="utf-8",
            )

    if log_skips:
        if skipped_nan_intensity:
            preview = ", ".join([f"{c} ({m})" for c, m in skipped_nan_intensity[:8]])
            logger.warning(
                "Skipped %d countries because active species had NaN intensity. Sample: %s%s",
                len(skipped_nan_intensity),
                preview,
                " ..." if len(skipped_nan_intensity) > 8 else "",
            )
        if fixed_low_species:
            preview = ", ".join([f"{c} ({m})" for c, m, _ in fixed_low_species[:8]])
            logger.info(
                "Included %d countries with fewer than 2 active species; optimisation fixed to baseline because expansion was unavailable. Sample: %s%s",
                len(fixed_low_species),
                preview,
                " ..." if len(fixed_low_species) > 8 else "",
            )
        if allow_expansion and expansion_disabled_no_posterior:
            preview = ", ".join([f"{c} ({m})" for c, m in expansion_disabled_no_posterior[:8]])
            logger.warning(
                "allow_expansion requested, but posterior intensities were unavailable for %d countries; those countries used allow_expansion=False. Sample: %s%s",
                len(expansion_disabled_no_posterior),
                preview,
                " ..." if len(expansion_disabled_no_posterior) > 8 else "",
            )
        if reverted_no_harm:
            preview = ", ".join(
                [f"{c} ({m})" for c, m, _, _, _ in reverted_no_harm[:8]],
            )
            logger.warning(
                "Do-no-harm guard triggered for %d countries (raw optimised posterior mean exceeded observed baseline). Sample: %s%s",
                len(reverted_no_harm),
                preview,
                " ..." if len(reverted_no_harm) > 8 else "",
            )

    return df


def _lognormal_fallback(
    i_obs: np.ndarray,
    w_ref: np.ndarray,
    n_draws: int = 500,
    cv: float = 0.2,
) -> np.ndarray:
    """Generate lognormal scenarios when no Bayesian posterior is available.

    Parameters
    ----------
    i_obs : (S,) observed intensities
    w_ref : (S,) weights (used to determine which species are active)
    n_draws : number of scenario draws
    cv : coefficient of variation

    Returns
    -------
    I_scenarios : (n_draws, S)
    """
    rng = np.random.default_rng(config.RNG_SEED)
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
