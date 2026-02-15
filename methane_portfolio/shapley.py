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
from pathlib import Path

import numpy as np
import pandas as pd

from methane_portfolio import config


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
        values="species_share", fill_value=0.0,
    )
    pivot_i = sub.pivot_table(
        index="country_m49", columns="milk_species",
        values="kg_co2e_per_ton_milk", fill_value=0.0,
    )
    # Align to canonical species order
    for s in species_order:
        if s not in pivot_w.columns:
            pivot_w[s] = 0.0
            pivot_i[s] = 0.0
    pivot_w = pivot_w[species_order]
    pivot_i = pivot_i[species_order]

    # Align indices
    common_idx = pivot_w.index.intersection(pivot_i.index)
    pivot_w = pivot_w.loc[common_idx]
    pivot_i = pivot_i.loc[common_idx]

    return (
        pivot_w.index.values.copy(),
        pivot_w.values.copy(),
        pivot_i.values.copy(),
    )


def shapley_decompose(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    species_order: list[str],
    *,
    base_year: int = config.BASE_YEAR,
    end_year: int = config.END_YEAR,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run exact 2-factor Shapley decomposition country-by-country.

    Returns a DataFrame with columns:
        country_m49, country, delta_struct, delta_within,
        delta_total, delta_obs, reconstruction_error

    Also writes:
        outputs/shapley_country.csv
        outputs/shapley_global.json
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    m49_0, W0, I0 = build_wide_matrices(long_df, base_year, species_order)
    m49_1, W1, I1 = build_wide_matrices(long_df, end_year, species_order)

    # Common countries
    common = np.intersect1d(m49_0, m49_1)
    idx0 = np.searchsorted(m49_0, common)
    idx1 = np.searchsorted(m49_1, common)
    W0, I0 = W0[idx0], I0[idx0]
    W1, I1 = W1[idx1], I1[idx1]

    # Shapley formula  (vectorised over countries)
    dW = W1 - W0  # (C, S)
    dI = I1 - I0  # (C, S)

    delta_struct = 0.5 * ((dW * I0).sum(axis=1) + (dW * I1).sum(axis=1))
    delta_within = 0.5 * ((W0 * dI).sum(axis=1) + (W1 * dI).sum(axis=1))
    delta_total = delta_struct + delta_within

    # Observed delta from agg_df
    agg_base = agg_df[agg_df["year"] == base_year].set_index("country_m49")
    agg_end = agg_df[agg_df["year"] == end_year].set_index("country_m49")
    common_agg = agg_base.index.intersection(agg_end.index).intersection(
        pd.Index(common)
    )
    delta_obs = (
        agg_end.loc[common_agg, "kg_co2e_per_ton_milk"].values
        - agg_base.loc[common_agg, "kg_co2e_per_ton_milk"].values
    )

    # Align to common_agg
    mask = np.isin(common, common_agg.values)
    delta_struct_aligned = delta_struct[mask]
    delta_within_aligned = delta_within[mask]
    delta_total_aligned = delta_total[mask]

    recon_err = np.abs(delta_total_aligned - delta_obs)

    # country names
    name_map = (
        long_df[["country_m49", "country"]]
        .drop_duplicates()
        .set_index("country_m49")["country"]
    )

    result = pd.DataFrame({
        "country_m49": common_agg.values,
        "country": [name_map.get(m, "") for m in common_agg.values],
        "delta_struct": delta_struct_aligned,
        "delta_within": delta_within_aligned,
        "delta_total": delta_total_aligned,
        "delta_obs": delta_obs,
        "reconstruction_error": recon_err,
    })

    # Validate reconstruction
    max_err = recon_err.max()
    if max_err > config.SHAPLEY_RECON_TOL:
        raise ValueError(
            f"Shapley reconstruction error {max_err:.2e} exceeds "
            f"tolerance {config.SHAPLEY_RECON_TOL:.2e}"
        )

    result.to_csv(out / "shapley_country.csv", index=False)

    # Global production-weighted aggregates
    prod_base = agg_base.loc[common_agg, "milk_total_tonnes"].values
    total_prod = prod_base.sum()
    weights = prod_base / total_prod

    global_summary = {
        "base_year": base_year,
        "end_year": end_year,
        "n_countries": int(len(common_agg)),
        "global_struct": float(np.dot(weights, delta_struct_aligned)),
        "global_within": float(np.dot(weights, delta_within_aligned)),
        "net": float(np.dot(weights, delta_total_aligned)),
        "max_reconstruction_error": float(max_err),
    }
    with open(out / "shapley_global.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    return result
