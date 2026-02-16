# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Advanced analytics for global contribution concentration and structure.

This module intentionally exposes pure, testable functions only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def _component_column(component: Literal["struct", "within", "total"]) -> str:
    return {
        "struct": "delta_struct",
        "within": "delta_within",
        "total": "delta_total",
    }[component]


def weighted_lorenz_curve(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted Lorenz curve coordinates (population_share, value_share)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x >= 0)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cum_w = np.cumsum(w)
    total_w = cum_w[-1]
    pop_share = np.concatenate([[0.0], cum_w / total_w])

    wx = w * x
    total_wx = float(wx.sum())
    if total_wx <= 0:
        # Degenerate all-zero case -> perfect equality (Gini = 0)
        return pop_share, pop_share.copy()

    cum_wx = np.cumsum(wx)
    val_share = np.concatenate([[0.0], cum_wx / total_wx])
    return pop_share, val_share


def weighted_gini(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted Gini coefficient for non-negative values."""
    pop_share, val_share = weighted_lorenz_curve(x, w)
    area = np.trapz(val_share, pop_share)
    gini = float(1.0 - 2.0 * area)
    return float(np.clip(gini, 0.0, 1.0))


def inequality_decomposition(
    df: pd.DataFrame,
    component: Literal["struct", "within", "total"],
    weights_col: str = "weight_interval",
    region_col: str = "region",
) -> dict:
    """ANOVA-like weighted variance decomposition between/within regions.

    Uses absolute weighted contributions:
        x_c = |w_c * Î”_c|
    """
    comp_col = _component_column(component)
    required = {weights_col, region_col, comp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for inequality decomposition: {sorted(missing)}")

    work = df[[comp_col, weights_col, region_col]].copy()
    work[region_col] = work[region_col].fillna("Unknown").replace("", "Unknown")
    work = work[np.isfinite(work[comp_col]) & np.isfinite(work[weights_col])]
    work = work[work[weights_col] > 0].copy()
    if work.empty:
        raise ValueError("No valid rows available for inequality decomposition.")

    w = work[weights_col].to_numpy(dtype=float)
    x = np.abs(w * work[comp_col].to_numpy(dtype=float))
    W = float(w.sum())
    if W <= 0:
        raise ValueError("Weights must sum to a positive value.")

    mu = float(np.sum(w * x) / W)
    var_total = float(np.sum(w * (x - mu) ** 2) / W)

    between_num = 0.0
    within_num = 0.0
    for _, grp in work.assign(_x=x).groupby(region_col, observed=True):
        w_r = grp[weights_col].to_numpy(dtype=float)
        x_r = grp["_x"].to_numpy(dtype=float)
        W_r = float(w_r.sum())
        if W_r <= 0:
            continue
        mu_r = float(np.sum(w_r * x_r) / W_r)
        between_num += W_r * (mu_r - mu) ** 2
        within_num += float(np.sum(w_r * (x_r - mu_r) ** 2))

    var_between = float(between_num / W)
    var_within = float(within_num / W)
    var_total_safe = var_total if var_total > 0 else np.nan

    return {
        "component": component,
        "var_total": var_total,
        "var_between": var_between,
        "var_within": var_within,
        "share_between": float(var_between / var_total_safe) if np.isfinite(var_total_safe) else np.nan,
        "share_within": float(var_within / var_total_safe) if np.isfinite(var_total_safe) else np.nan,
        "gini": weighted_gini(x, w),
        "n": int(len(work)),
    }


def loo_influence(
    df: pd.DataFrame,
    component: Literal["struct", "within", "total"],
    weights_col: str = "weight_interval",
) -> pd.DataFrame:
    """Vectorized leave-one-out influence on global Î”."""
    comp_col = _component_column(component)
    required = {comp_col, weights_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for LOO influence: {sorted(missing)}")

    work = df.copy()
    if "country_m49" not in work.columns:
        work["country_m49"] = np.arange(len(work))
    if "country" not in work.columns:
        work["country"] = work["country_m49"].astype(str)

    mask = np.isfinite(work[comp_col]) & np.isfinite(work[weights_col]) & (work[weights_col] > 0)
    work = work.loc[mask, ["country_m49", "country", comp_col, weights_col]].copy()
    if work.empty:
        raise ValueError("No valid rows available for LOO influence.")

    w = work[weights_col].to_numpy(dtype=float)
    d = work[comp_col].to_numpy(dtype=float)
    S = float(np.sum(w * d))
    W = float(np.sum(w))
    if W <= 0:
        raise ValueError("Weights must sum to a positive value.")

    global_delta = S / W
    denom = W - w
    without = np.full_like(d, np.nan, dtype=float)
    valid = denom > 0
    without[valid] = (S - w[valid] * d[valid]) / denom[valid]

    influence_abs = without - global_delta
    eps = 1e-12
    if abs(global_delta) <= eps:
        influence_rel = np.full_like(influence_abs, np.nan, dtype=float)
    else:
        influence_rel = influence_abs / global_delta

    return pd.DataFrame({
        "country_m49": work["country_m49"].to_numpy(),
        "country": work["country"].to_numpy(),
        "influence_abs": influence_abs,
        "influence_rel": influence_rel,
        "weight_interval": w,
        "delta_component": d,
    })


def hill_pareto_exponent(
    x: np.ndarray,
    top_frac: float = 0.10,
    n_bootstrap: int = 200,
    seed: int = 20230101,
) -> dict:
    """Estimate Pareto tail exponent with Hill estimator on top fraction."""
    if not (0 < top_frac <= 1):
        raise ValueError("top_frac must be in (0, 1].")

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = int(x.size)
    if n < 3:
        raise ValueError("Need at least 3 positive observations for Hill estimator.")

    xs = np.sort(x)[::-1]
    k = max(2, int(np.ceil(top_frac * n)))
    k = min(k, n)
    x_k = float(xs[k - 1])
    inv_alpha = float(np.mean(np.log(xs[:k] / x_k)))
    alpha_hat = float(1.0 / inv_alpha) if inv_alpha > 0 else np.inf

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(xs, size=n, replace=True)
        sample.sort()
        sample = sample[::-1]
        x_k_b = sample[k - 1]
        if x_k_b <= 0:
            continue
        inv_alpha_b = np.mean(np.log(sample[:k] / x_k_b))
        if np.isfinite(inv_alpha_b) and inv_alpha_b > 0:
            boot.append(1.0 / inv_alpha_b)

    if boot:
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).astype(float)
    else:
        ci_low, ci_high = np.nan, np.nan

    return {
        "alpha_hat": alpha_hat,
        "k": int(k),
        "x_k": x_k,
        "top_frac": float(top_frac),
        "alpha_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "alpha_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "n": n,
    }


def hill_plot_curve(
    x: np.ndarray,
    k_min: int = 2,
    k_max: int | None = None,
) -> pd.DataFrame:
    """Return alpha-vs-k diagnostics for a Hill plot."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 3:
        return pd.DataFrame(columns=["k", "alpha_hat"])

    xs = np.sort(x)[::-1]
    n = xs.size
    if k_max is None:
        k_max = max(k_min + 1, int(np.floor(0.5 * n)))
    k_max = min(k_max, n)

    rows = []
    for k in range(k_min, k_max + 1):
        x_k = xs[k - 1]
        inv_alpha = np.mean(np.log(xs[:k] / x_k))
        alpha_hat = np.nan if inv_alpha <= 0 else 1.0 / inv_alpha
        rows.append({"k": int(k), "alpha_hat": float(alpha_hat)})
    return pd.DataFrame(rows)


def build_phase_space(
    df_shapley: pd.DataFrame,
    df_opt: pd.DataFrame,
    weights_col: str = "weight_interval",
) -> pd.DataFrame:
    """Build merged table for structural phase-diagram visualization."""
    required_shapley = {"country_m49", "delta_struct", "delta_within", "delta_total"}
    missing_shapley = required_shapley - set(df_shapley.columns)
    if missing_shapley:
        raise ValueError(f"Missing required shapley columns: {sorted(missing_shapley)}")

    opt = df_opt.copy()
    if "reduction_pct" not in opt.columns and "reduction_mean_pct" in opt.columns:
        opt["reduction_pct"] = opt["reduction_mean_pct"]
    if "baseline_intensity" not in opt.columns and "baseline_intensity_kg_co2e_per_t" in opt.columns:
        opt["baseline_intensity"] = opt["baseline_intensity_kg_co2e_per_t"]

    required_opt = {"country_m49", "reduction_pct", "absolute_reduction_kt", "baseline_intensity"}
    missing_opt = required_opt - set(opt.columns)
    if missing_opt:
        raise ValueError(f"Missing required optimization columns: {sorted(missing_opt)}")

    shapley_cols = ["country_m49", "delta_struct", "delta_within", "delta_total"]
    optional_shapley = [c for c in ["country", weights_col, "region"] if c in df_shapley.columns]
    opt_cols = ["country_m49", "reduction_pct", "absolute_reduction_kt", "baseline_intensity"]
    optional_opt = [c for c in ["country", "region"] if c in opt.columns]

    merged = df_shapley[shapley_cols + optional_shapley].merge(
        opt[opt_cols + optional_opt],
        on="country_m49",
        how="inner",
        suffixes=("", "_opt"),
    )

    if "country" not in merged.columns:
        if "country_opt" in merged.columns:
            merged["country"] = merged["country_opt"]
        else:
            merged["country"] = merged["country_m49"].astype(str)

    if weights_col not in merged.columns:
        merged[weights_col] = 1.0

    if "region" not in merged.columns:
        if "region_opt" in merged.columns:
            merged["region"] = merged["region_opt"]
        else:
            merged["region"] = "Unknown"

    merged["region"] = merged["region"].fillna("Unknown").replace("", "Unknown")

    keep = [
        "country_m49",
        "country",
        "delta_struct",
        "delta_within",
        "delta_total",
        "reduction_pct",
        "absolute_reduction_kt",
        "baseline_intensity",
        weights_col,
        "region",
    ]
    return merged[keep].copy()
