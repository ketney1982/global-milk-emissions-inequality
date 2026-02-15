"""Validation of accounting identities and data consistency.

Checks:
1. Species shares sum to 1 per (country_m49, year).
2. Species-level milk tonnes sum matches aggregate milk_total_tonnes.
3. Mixture identity: I_ct == Σ_s w_cts · I_cts   (within tolerance).

Results are written to ``outputs/validation_report.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from methane_portfolio import config


class ValidationError(Exception):
    """Raised when a critical accounting identity is violated."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_all(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    raise_on_fail: bool = True,
) -> pd.DataFrame:
    """Run every validation check and return a combined report DataFrame.

    Parameters
    ----------
    long_df : species-level DataFrame (from ``io.build_long_df``)
    agg_df  : aggregate DataFrame (from ``io.build_agg_df``)
    output_dir : where to write ``validation_report.csv``
    raise_on_fail : if True, raise ``ValidationError`` on first violation

    Returns
    -------
    report : DataFrame with columns [country_m49, country, year,
             check, status, detail]
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    reports: list[pd.DataFrame] = []

    reports.append(_check_shares_sum(long_df, raise_on_fail))
    reports.append(_check_milk_totals(long_df, agg_df, raise_on_fail))
    reports.append(_check_identity(long_df, agg_df, raise_on_fail))

    report = pd.concat(reports, ignore_index=True)
    report.to_csv(out / "validation_report.csv", index=False)
    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_shares_sum(
    long_df: pd.DataFrame,
    raise_on_fail: bool,
) -> pd.DataFrame:
    """Species shares must sum to 1 per (country_m49, year)."""
    grouped = (
        long_df
        .groupby(["country_m49", "country", "year"], observed=True)["species_share"]
        .sum()
        .reset_index(name="share_sum")
    )
    grouped["error"] = np.abs(grouped["share_sum"] - 1.0)
    grouped["status"] = np.where(
        grouped["error"] <= config.SHARE_SUM_TOL, "PASS", "FAIL"
    )
    grouped["check"] = "shares_sum_to_1"
    grouped["detail"] = grouped.apply(
        lambda r: f"sum={r['share_sum']:.10f}, error={r['error']:.2e}", axis=1,
    )

    fails = grouped[grouped["status"] == "FAIL"]
    if raise_on_fail and len(fails) > 0:
        sample = fails.head(5).to_string(index=False)
        raise ValidationError(
            f"Species shares do not sum to 1 for {len(fails)} "
            f"(country_m49, year) groups (tol={config.SHARE_SUM_TOL}):\n{sample}"
        )

    return grouped[["country_m49", "country", "year", "check", "status", "detail"]]


def _check_milk_totals(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    raise_on_fail: bool,
) -> pd.DataFrame:
    """Sum of species milk_tonnes should match aggregate milk_total_tonnes."""
    species_sum = (
        long_df
        .groupby(["country_m49", "country", "year"], observed=True)["milk_tonnes"]
        .sum()
        .reset_index(name="species_milk_sum")
    )
    merged = species_sum.merge(
        agg_df[["country_m49", "year", "milk_total_tonnes"]],
        on=["country_m49", "year"],
        how="inner",
    )
    merged["rel_error"] = np.abs(
        merged["species_milk_sum"] - merged["milk_total_tonnes"]
    ) / merged["milk_total_tonnes"].clip(lower=1e-30)
    merged["status"] = np.where(
        merged["rel_error"] <= config.MILK_MATCH_REL_TOL, "PASS", "FAIL"
    )
    merged["check"] = "milk_totals_match"
    merged["detail"] = merged.apply(
        lambda r: (
            f"species_sum={r['species_milk_sum']:.2f}, "
            f"agg={r['milk_total_tonnes']:.2f}, "
            f"rel_err={r['rel_error']:.2e}"
        ),
        axis=1,
    )

    fails = merged[merged["status"] == "FAIL"]
    if raise_on_fail and len(fails) > 0:
        sample = fails.head(5).to_string(index=False)
        raise ValidationError(
            f"Milk totals mismatch for {len(fails)} groups "
            f"(rel_tol={config.MILK_MATCH_REL_TOL}):\n{sample}"
        )

    return merged[["country_m49", "country", "year", "check", "status", "detail"]]


def _check_identity(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    raise_on_fail: bool,
) -> pd.DataFrame:
    """Identity:  I_ct == Σ_s  w_cts · I_cts.

    The mixture intensity from species shares and species intensities
    must match the reported aggregate intensity.
    """
    # Compute mixture
    tmp = long_df.copy()
    tmp["w_times_I"] = tmp["species_share"] * tmp["kg_co2e_per_ton_milk"]
    mixture = (
        tmp
        .groupby(["country_m49", "country", "year"], observed=True)["w_times_I"]
        .sum()
        .reset_index(name="mixture_intensity")
    )
    merged = mixture.merge(
        agg_df[["country_m49", "year", "kg_co2e_per_ton_milk"]],
        on=["country_m49", "year"],
        how="inner",
    )
    merged.rename(columns={"kg_co2e_per_ton_milk": "agg_intensity"}, inplace=True)
    merged["abs_error"] = np.abs(
        merged["mixture_intensity"] - merged["agg_intensity"]
    )
    merged["status"] = np.where(
        merged["abs_error"] <= config.IDENTITY_TOL, "PASS", "FAIL"
    )
    merged["check"] = "identity_I_eq_sum_wI"
    merged["detail"] = merged.apply(
        lambda r: (
            f"mixture={r['mixture_intensity']:.12f}, "
            f"agg={r['agg_intensity']:.12f}, "
            f"abs_err={r['abs_error']:.2e}"
        ),
        axis=1,
    )

    fails = merged[merged["status"] == "FAIL"]
    if raise_on_fail and len(fails) > 0:
        sample = fails.head(5).to_string(index=False)
        raise ValidationError(
            f"Mixture identity violated for {len(fails)} groups "
            f"(tol={config.IDENTITY_TOL}):\n{sample}"
        )

    return merged[["country_m49", "country", "year", "check", "status", "detail"]]
