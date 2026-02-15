"""Tests for data validation and accounting identities."""

import numpy as np
import pandas as pd
import pytest

from methane_portfolio.io import load_all
from methane_portfolio.validate import ValidationError, validate_all


@pytest.fixture(scope="module")
def data():
    """Load real data once per module."""
    long_df, agg_df, species = load_all()
    return long_df, agg_df, species


class TestValidateShareSum:
    """Species shares must sum to 1 per (country_m49, year)."""

    def test_all_shares_sum_to_one(self, data):
        long_df, _, _ = data
        sums = (
            long_df
            .groupby(["country_m49", "year"], observed=True)["species_share"]
            .sum()
        )
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)

    def test_no_negative_shares(self, data):
        long_df, _, _ = data
        assert (long_df["species_share"] >= 0).all()


class TestValidateMilkTotals:
    """Species milk tonnes should sum to aggregate."""

    def test_species_sum_matches_agg(self, data):
        long_df, agg_df, _ = data
        species_sum = (
            long_df
            .groupby(["country_m49", "year"], observed=True)["milk_tonnes"]
            .sum()
            .reset_index(name="sp_sum")
        )
        merged = species_sum.merge(
            agg_df[["country_m49", "year", "milk_total_tonnes"]],
            on=["country_m49", "year"],
        )
        rel_err = np.abs(
            merged["sp_sum"] - merged["milk_total_tonnes"]
        ) / merged["milk_total_tonnes"].clip(lower=1)
        assert (rel_err < 1e-4).all(), f"Max rel error: {rel_err.max()}"


class TestValidateIdentity:
    """Mixture identity: I_ct = Σ w_cts · I_cts."""

    def test_identity_holds(self, data):
        long_df, agg_df, _ = data
        tmp = long_df.copy()
        tmp["w_I"] = tmp["species_share"] * tmp["kg_co2e_per_ton_milk"]
        mixture = (
            tmp
            .groupby(["country_m49", "year"], observed=True)["w_I"]
            .sum()
            .reset_index(name="mixture")
        )
        merged = mixture.merge(
            agg_df[["country_m49", "year", "kg_co2e_per_ton_milk"]],
            on=["country_m49", "year"],
        )
        abs_err = np.abs(merged["mixture"] - merged["kg_co2e_per_ton_milk"])
        assert abs_err.max() < 1e-6, f"Max identity error: {abs_err.max()}"


class TestValidateAll:
    """Full validation pipeline."""

    def test_passes_without_error(self, data, tmp_path):
        long_df, agg_df, _ = data
        report = validate_all(
            long_df, agg_df,
            output_dir=tmp_path,
            raise_on_fail=False,
        )
        assert len(report) > 0
        assert "PASS" in report["status"].values

    def test_report_columns(self, data, tmp_path):
        long_df, agg_df, _ = data
        report = validate_all(
            long_df, agg_df,
            output_dir=tmp_path,
            raise_on_fail=False,
        )
        expected_cols = {"country_m49", "country", "year", "check", "status", "detail"}
        assert expected_cols.issubset(set(report.columns))
