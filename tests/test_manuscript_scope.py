# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: ketney.otto@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Regression tests pinning the public entry point to the reported analysis.

Two things went wrong between R2 and the manuscript, and both were invisible to
the test suite because nothing asserted the *scope* or the *definition* of what
`run-all` produces:

1. `run_sensitivity_grid` defaulted to the 20 largest producers, so `run-all`
   emitted a 720-row grid while the manuscript reported 6,516 solves over the
   full panel.
2. the optimiser exported a single `absolute_reduction_kt` column that held the
   posterior-scaled difference in megatonnes, while the manuscript reported the
   inventory-scaled reduction. Same name, wrong unit, wrong quantity.

These tests fail if either regression is reintroduced.
"""

import numpy as np
import pandas as pd
import pytest

from methane_portfolio import config
from methane_portfolio.robust_optimize import run_all_countries
from methane_portfolio.uncertainty import run_sensitivity_grid


def _panel(n_countries: int = 5) -> pd.DataFrame:
    """Small multi-species panel: every country can actually reallocate."""
    rows = []
    for i in range(n_countries):
        m49 = 100 + i
        for sp, share, intensity, tonnes in [
            ("cattle", 0.70, 0.05, 1_000_000.0 * (i + 1)),
            ("goats", 0.20, 0.20, 285_714.0 * (i + 1)),
            ("sheep", 0.10, 0.55, 142_857.0 * (i + 1)),
        ]:
            rows.append({
                "country_m49": m49,
                "country": f"C{i}",
                "year": 2023,
                "milk_species": sp,
                "species_share": share,
                "milk_tonnes": tonnes,
                "kg_co2e_per_ton_milk": intensity,
            })
    return pd.DataFrame(rows)


class TestSensitivityGridScope:
    """The grid must cover the whole analysed panel by default."""

    def test_default_scope_is_the_full_panel(self, tmp_path):
        long_df = _panel(5)
        grid = run_sensitivity_grid(
            long_df, year=2023, workers=1, output_dir=tmp_path,
        )
        n_conf = len(config.DELTA_GRID) * len(config.LAMBDA_GRID) * len(config.ALPHA_GRID)
        assert n_conf == 36, "the reported grid is 4 x 3 x 3 configurations"
        assert grid["country_m49"].nunique() == 5
        assert len(grid) == 5 * n_conf

    def test_explicit_subset_still_available(self, tmp_path):
        long_df = _panel(5)
        grid = run_sensitivity_grid(
            long_df, year=2023, n_countries_max=2, workers=1, output_dir=tmp_path,
        )
        assert grid["country_m49"].nunique() == 2

    def test_summaries_are_written_alongside_the_grid(self, tmp_path):
        run_sensitivity_grid(_panel(3), year=2023, workers=1, output_dir=tmp_path)
        for name in (
            "sensitivity_grid.csv",
            "sensitivity_summary_all181.csv",
            "sensitivity_summary_top20.csv",
        ):
            assert (tmp_path / name).exists(), name
        summary = pd.read_csv(tmp_path / "sensitivity_summary_all181.csv")
        assert len(summary) == 36
        assert {"delta", "lambda", "alpha", "mean", "median", "q25", "q75",
                "cvar", "certified"} <= set(summary.columns)


class TestAbsoluteReductionDefinition:
    """The two absolute-reduction quantities must stay distinct and named."""

    def test_both_quantities_are_exported(self, tmp_path):
        df = run_all_countries(
            _panel(3), year=2023, output_dir=tmp_path, save_csv=False,
        )
        for col in ("observed_ch4_t", "abs_reduction_mt_ch4",
                    "abs_reduction_mt_ch4_posterior",
                    "raw_abs_reduction_mt_ch4",
                    "raw_abs_reduction_mt_ch4_posterior"):
            assert col in df.columns, col

    def test_ambiguous_column_is_gone(self, tmp_path):
        df = run_all_countries(
            _panel(3), year=2023, output_dir=tmp_path, save_csv=False,
        )
        assert "absolute_reduction_kt" not in df.columns
        assert "raw_absolute_reduction_kt" not in df.columns

    def test_inventory_scaled_is_the_percentage_applied_to_the_inventory(self, tmp_path):
        df = run_all_countries(
            _panel(3), year=2023, output_dir=tmp_path, save_csv=False,
        )
        expected = df["observed_ch4_t"] * df["reduction_mean_pct"] / 100.0 / 1e6
        assert np.allclose(df["abs_reduction_mt_ch4"], expected, rtol=0, atol=1e-15)

    def test_posterior_scaled_is_the_intensity_difference_times_production(self, tmp_path):
        df = run_all_countries(
            _panel(3), year=2023, output_dir=tmp_path, save_csv=False,
        )
        expected = (
            (df["baseline_intensity"] - df["optimized_mean"])
            * df["production_tonnes"] / 1e6
        )
        assert np.allclose(df["abs_reduction_mt_ch4_posterior"], expected,
                           rtol=0, atol=1e-15)

    def test_observed_inventory_matches_intensity_times_production(self, tmp_path):
        """observed_ch4_t is a mass in tonnes, reconstructible from the inputs."""
        long_df = _panel(3)
        df = run_all_countries(
            long_df, year=2023, output_dir=tmp_path, save_csv=False,
        ).set_index("country_m49")
        sub = long_df[long_df["year"] == 2023]
        direct = (
            sub.assign(t=sub["milk_tonnes"] * sub["kg_co2e_per_ton_milk"])
            .groupby("country_m49")["t"].sum()
        )
        assert np.allclose(df["observed_ch4_t"], direct.loc[df.index], rtol=1e-12)

    def test_the_two_quantities_differ_under_a_posterior_reference(self, tmp_path):
        """With a posterior reference the two are not interchangeable.

        This is the whole point of separating them: the panel totals in the
        reported analysis are 8.45 and 10.73 Mt CH4.
        """
        long_df = _panel(3)
        rng = np.random.default_rng(0)
        species = sorted(long_df["milk_species"].unique())
        countries = sorted(long_df["country_m49"].unique())
        # posterior scenarios deliberately offset from the observed intensities,
        # which is exactly what hierarchical shrinkage does
        base = np.array([0.05, 0.20, 0.55])
        I = np.exp(
            np.log(base * 1.35)[None, None, :]
            + 0.05 * rng.standard_normal((200, len(countries), len(species)))
        )
        df = run_all_countries(
            long_df, I_samples=I, country_list=countries, species_list=species,
            year=2023, output_dir=tmp_path, save_csv=False,
        )
        assert (df["reduction_mean_pct"] > 0).any()
        assert not np.allclose(
            df["abs_reduction_mt_ch4"], df["abs_reduction_mt_ch4_posterior"],
        )
