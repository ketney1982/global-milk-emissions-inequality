"""Exact two-factor Shapley decomposition of ΔI (2020→2023).

The two factors are:
  - **structure** (w): change in species portfolio shares
  - **within** (i): change in species-level emission intensities

For each country the national intensity is a mixture:
    I_ct = Σ_s  w_cts · I_cts

The change ΔI = I_{c,2023} − I_{c,2020} is decomposed into:
    ΔI = Δ_struct + Δ_within

using the exact two-factor Shapley formula:
    Δ_struct  = 0.5 * [Σ_s (w'_s − w_s) * I_s  +  Σ_s (w'_s − w_s) * I'_s]
    Δ_within  = 0.5 * [Σ_s w_s  * (I'_s − I_s)   +  Σ_s w'_s * (I'_s − I_s)]

where primed = 2023, unprimed = 2020.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from methane_portfolio import config

logger = logging.getLogger(__name__)


def build_wide_matrices(
    long_df: pd.DataFrame,
    year: int,
    species_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (countries_m49, W, I) for a given year.

    W[c, s] = species_share, I[c, s] = kg_co2e_per_ton_milk.
    Rows are countries (in consistent m49 order), columns are species.
    Species not present for a country get share=0, intensity=0.
    """
    sub = long_df[long_df["year"] == year].copy()
    pivot_w = sub.pivot_table(
        index="country_m49", columns="milk_species",
        values="species_share", fill_value=0.0, observed=True,
    )
    pivot_i = sub.pivot_table(
        index="country_m49", columns="milk_species",
        values="kg_co2e_per_ton_milk", fill_value=0.0, observed=True,
    )

    # Align to canonical species order
    for s in species_order:
        if s not in pivot_w.columns:
            pivot_w[s] = 0.0
            pivot_i[s] = 0.0
    pivot_w = pivot_w[species_order]
    pivot_i = pivot_i[species_order]

    # Align indices and keep increasing order for np.searchsorted usage.
    common_idx = pivot_w.index.intersection(pivot_i.index)
    pivot_w = pivot_w.loc[common_idx].sort_index()
    pivot_i = pivot_i.loc[common_idx].sort_index()

    return (
        pivot_w.index.values.copy(),
        pivot_w.values.copy(),
        pivot_i.values.copy(),
    )


def build_interval_weights(
    prod0: pd.Series,
    prod1: pd.Series,
    method: Literal["base", "end", "avg", "sum", "trapz"] = "avg",
) -> pd.Series:
    """Build interval production weights from endpoint production."""
    if not prod0.index.equals(prod1.index):
        raise ValueError("prod0 and prod1 must have identical index alignment.")

    method = method.lower()
    if method == "base":
        w = prod0.copy()
    elif method == "end":
        w = prod1.copy()
    elif method == "avg":
        w = 0.5 * (prod0 + prod1)
    elif method == "sum":
        w = prod0 + prod1
    elif method == "trapz":
        # For relative weighting, trapezoidal integration is proportional to avg.
        w = 0.5 * (prod0 + prod1)
    else:
        raise ValueError(
            "Unknown weight method. Use one of: base, end, avg, sum, trapz."
        )
    return w.astype(float)


def shapley_decompose(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    species_order: list[str],
    *,
    base_year: int = config.BASE_YEAR,
    end_year: int = config.END_YEAR,
    weight_method: Literal["base", "end", "avg", "sum", "trapz"] = config.DEFAULT_WEIGHT_METHOD,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run exact 2-factor Shapley decomposition country-by-country.

    Returns a DataFrame with columns:
        country_m49, country, delta_struct, delta_within,
        delta_total, delta_obs, reconstruction_error, weight_interval

    Also writes:
        outputs/shapley_country.csv
        outputs/shapley_global.json
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Exclude countries where active share is reported but intensity is missing.
    invalid_mask = (
        long_df["year"].isin([base_year, end_year])
        & (long_df["species_share"] > 0)
        & (long_df["kg_co2e_per_ton_milk"].isna())
    )
    invalid_countries = sorted(
        long_df.loc[invalid_mask, "country_m49"].dropna().unique().tolist()
    )
    if invalid_countries:
        logger.warning(
            "Excluding %d countries from Shapley due to active share with NaN intensity: %s",
            len(invalid_countries),
            invalid_countries[:10],
        )
        long_df = long_df[~long_df["country_m49"].isin(invalid_countries)].copy()
        agg_df = agg_df[~agg_df["country_m49"].isin(invalid_countries)].copy()

    m49_0, W0, I0 = build_wide_matrices(long_df, base_year, species_order)
    m49_1, W1, I1 = build_wide_matrices(long_df, end_year, species_order)

    # Common countries
    common = np.intersect1d(m49_0, m49_1)
    idx0 = np.searchsorted(m49_0, common)
    idx1 = np.searchsorted(m49_1, common)
    W0, I0 = W0[idx0], I0[idx0]
    W1, I1 = W1[idx1], I1[idx1]

    # Shapley formula (vectorised over countries)
    dW = W1 - W0  # (C, S)
    dI = I1 - I0  # (C, S)
    delta_struct = 0.5 * ((dW * I0).sum(axis=1) + (dW * I1).sum(axis=1))
    delta_within = 0.5 * ((W0 * dI).sum(axis=1) + (W1 * dI).sum(axis=1))
    delta_total = delta_struct + delta_within

    # Observed delta from agg_df
    agg_base = agg_df[agg_df["year"] == base_year].set_index("country_m49")
    agg_end = agg_df[agg_df["year"] == end_year].set_index("country_m49")
    common_agg = agg_base.index.intersection(agg_end.index).intersection(pd.Index(common))
    delta_obs = (
        agg_end.loc[common_agg, "kg_co2e_per_ton_milk"].values
        - agg_base.loc[common_agg, "kg_co2e_per_ton_milk"].values
    )

    # Align decomposition outputs to common_agg
    mask = np.isin(common, common_agg.values)
    delta_struct_aligned = delta_struct[mask]
    delta_within_aligned = delta_within[mask]
    delta_total_aligned = delta_total[mask]

    # Mandatory consistency audit before aggregation.
    consistency_gap = np.abs(
        delta_total_aligned - (delta_struct_aligned + delta_within_aligned)
    )
    max_consistency_gap = float(consistency_gap.max()) if consistency_gap.size else 0.0
    if max_consistency_gap > config.IDENTITY_TOL:
        logger.warning(
            "Consistency check failed: max |delta_total-(delta_struct+delta_within)| = %.3e",
            max_consistency_gap,
        )
        raise ValueError(
            "Shapley consistency check failed: "
            f"{max_consistency_gap:.3e} > {config.IDENTITY_TOL:.1e}"
        )

    recon_err = np.abs(delta_total_aligned - delta_obs)

    # Country names
    name_map = (
        long_df[["country_m49", "country"]]
        .drop_duplicates()
        .set_index("country_m49")["country"]
    )

    # Interval weights from base/end-year production
    prod_base = agg_base.loc[common_agg, "milk_total_tonnes"]
    prod_end = agg_end.loc[common_agg, "milk_total_tonnes"]
    interval_weight = build_interval_weights(prod_base, prod_end, weight_method)
    total_interval_weight = float(interval_weight.sum())
    if total_interval_weight <= 0:
        raise ValueError("Interval weights sum to zero; cannot compute global aggregate.")

    result = pd.DataFrame({
        "country_m49": common_agg.values,
        "country": [name_map.get(m, "") for m in common_agg.values],
        "delta_struct": delta_struct_aligned,
        "delta_within": delta_within_aligned,
        "delta_total": delta_total_aligned,
        "delta_obs": delta_obs,
        "reconstruction_error": recon_err,
        "weight_interval": interval_weight.values,
        "weight_interval_norm": interval_weight.values / total_interval_weight,
    })

    # Validate reconstruction
    max_err = float(recon_err.max()) if recon_err.size else 0.0
    if max_err > config.SHAPLEY_RECON_TOL:
        raise ValueError(
            f"Shapley reconstruction error {max_err:.2e} exceeds "
            f"tolerance {config.SHAPLEY_RECON_TOL:.2e}"
        )

    result.to_csv(out / "shapley_country.csv", index=False)

    # Global production-weighted aggregates
    w = interval_weight.values
    global_struct = float(np.sum(w * delta_struct_aligned) / total_interval_weight)
    global_within = float(np.sum(w * delta_within_aligned) / total_interval_weight)
    global_total = float(np.sum(w * delta_total_aligned) / total_interval_weight)

    global_summary = {
        "base_year": base_year,
        "end_year": end_year,
        "weight_method": weight_method,
        "n_countries": int(len(common_agg)),
        "global_struct": global_struct,
        "global_within": global_within,
        "global_total": global_total,
        "net": global_total,
        "total_interval_weight": total_interval_weight,
        "max_reconstruction_error": max_err,
        "max_consistency_gap": max_consistency_gap,
    }
    with open(out / "shapley_global.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    return result
