# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Tests for robust portfolio optimisation."""

import logging
import numpy as np
import pandas as pd
import pytest

from methane_portfolio.cli import build_parser
from methane_portfolio.optimize import (
    mean_intensity,
    simplex_constraint,
    tv_distance_constraint,
)
from methane_portfolio.robust_optimize import run_all_countries, solve_robust


class TestOptimizeHelpers:
    """Helper function tests."""

    def test_simplex_constraint(self):
        c = simplex_constraint(5)
        assert c.lb == 1.0
        assert c.ub == 1.0

    def test_tv_distance_satisfied(self):
        w_ref = np.array([0.5, 0.3, 0.2])
        con = tv_distance_constraint(w_ref, 0.1)
        # w = w_ref â†’ TV = 0 â‰¤ 0.1 â†’ positive (satisfied)
        assert con["fun"](w_ref) >= 0

    def test_tv_distance_violated(self):
        w_ref = np.array([0.5, 0.3, 0.2])
        con = tv_distance_constraint(w_ref, 0.01)
        w_far = np.array([0.8, 0.1, 0.1])
        # TV = 0.5 * (0.3 + 0.2 + 0.1) = 0.3 >> 0.01
        assert con["fun"](w_far) < 0

    def test_mean_intensity(self):
        w = np.array([0.5, 0.5])
        I = np.array([[2.0, 4.0], [3.0, 5.0]])
        # mean = mean([0.5*2+0.5*4, 0.5*3+0.5*5]) = mean([3, 4]) = 3.5
        assert np.isclose(mean_intensity(w, I), 3.5)


class TestSolveRobust:
    """Optimizer sanity checks."""

    def test_simplex_preserved(self):
        """Optimal weights must sum to 1."""
        rng = np.random.default_rng(42)
        w_ref = np.array([0.6, 0.3, 0.1])
        # Species 0 has high intensity, species 2 has low â†’ optimizer
        # should shift weight away from 0
        I_scenarios = np.column_stack([
            rng.lognormal(np.log(10), 0.1, 200),
            rng.lognormal(np.log(5), 0.1, 200),
            rng.lognormal(np.log(2), 0.1, 200),
        ])
        sol = solve_robust(w_ref, I_scenarios, delta=0.2)
        np.testing.assert_allclose(sol["w_opt"].sum(), 1.0, atol=1e-8)

    def test_reduction_positive(self):
        """With room to shift (delta>0), optimised mean should be â‰¤ baseline."""
        rng = np.random.default_rng(42)
        w_ref = np.array([0.7, 0.2, 0.1])
        I_scenarios = np.column_stack([
            rng.lognormal(np.log(10), 0.1, 200),
            rng.lognormal(np.log(3), 0.1, 200),
            rng.lognormal(np.log(1), 0.1, 200),
        ])
        baseline = float(I_scenarios @ w_ref).mean() if False else float(np.mean(I_scenarios @ w_ref))
        sol = solve_robust(w_ref, I_scenarios, delta=0.25,  lam=1.0)
        assert sol["mean_opt"] <= baseline + 1e-6

    def test_no_expansion_zeros_stay_zero(self):
        """If a species has w_ref=0 and allow_expansion=False, w_opt should stay 0."""
        rng = np.random.default_rng(42)
        w_ref = np.array([0.6, 0.4, 0.0])
        I_scenarios = np.column_stack([
            rng.lognormal(np.log(5), 0.1, 200),
            rng.lognormal(np.log(3), 0.1, 200),
            rng.lognormal(np.log(1), 0.1, 200),  # low but zero share
        ])
        sol = solve_robust(w_ref, I_scenarios, allow_expansion=False, delta=0.2)
        assert sol["w_opt"][2] == 0.0

    def test_tv_constraint_respected_after_postprocess(self):
        """Post-processed solution should satisfy TV budget up to tiny tolerance."""
        rng = np.random.default_rng(7)
        w_ref = np.array([0.5, 0.3, 0.2])
        I_scenarios = np.column_stack([
            rng.lognormal(np.log(9), 0.2, 300),
            rng.lognormal(np.log(4), 0.2, 300),
            rng.lognormal(np.log(1.5), 0.2, 300),
        ])
        delta = 0.1
        sol = solve_robust(w_ref, I_scenarios, delta=delta, allow_expansion=False)
        tv = 0.5 * np.abs(sol["w_opt"] - w_ref).sum()
        assert tv <= delta + 1e-10


class TestRunAllCountriesLowSpecies:
    """Countries with one active species should be included in output."""

    def test_low_species_included_and_fixed(self, caplog, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "MonoLand",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 1.0,
                    "milk_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 10.0,
                },
                {
                    "country_m49": 2,
                    "country": "DualLand",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.7,
                    "milk_tonnes": 70.0,
                    "kg_co2e_per_ton_milk": 8.0,
                },
                {
                    "country_m49": 2,
                    "country": "DualLand",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.3,
                    "milk_tonnes": 30.0,
                    "kg_co2e_per_ton_milk": 5.0,
                },
            ],
        )

        with caplog.at_level(logging.INFO, logger="methane_portfolio.robust_optimize"):
            out = run_all_countries(long_df, year=2023, output_dir=tmp_path, log_skips=True)

        assert len(out) == 2
        mono = out[out["country"] == "MonoLand"].iloc[0]
        assert mono["reduction_mean_pct"] == 0.0
        assert mono["reduction_cvar_pct"] == 0.0
        records = [
            rec for rec in caplog.records
            if "optimisation fixed to baseline" in rec.getMessage()
        ]
        assert records
        assert all(rec.levelno == logging.INFO for rec in records)

    def test_allow_expansion_without_posterior_is_safely_disabled(self, caplog, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "MonoLand",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 1.0,
                    "milk_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 10.0,
                },
                {
                    "country_m49": 2,
                    "country": "DualLand",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.7,
                    "milk_tonnes": 70.0,
                    "kg_co2e_per_ton_milk": 8.0,
                },
                {
                    "country_m49": 2,
                    "country": "DualLand",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.3,
                    "milk_tonnes": 30.0,
                    "kg_co2e_per_ton_milk": 5.0,
                },
            ],
        )

        with caplog.at_level(logging.INFO, logger="methane_portfolio.robust_optimize"):
            out = run_all_countries(
                long_df,
                year=2023,
                output_dir=tmp_path,
                log_skips=True,
                allow_expansion=True,
            )

        mono = out[out["country"] == "MonoLand"].iloc[0]
        assert mono["reduction_mean_pct"] == 0.0
        warn_records = [
            rec for rec in caplog.records
            if "posterior intensities were unavailable" in rec.getMessage()
        ]
        assert warn_records


class TestCliAllowExpansion:
    """allow-expansion should be parsed for optimize and run-all."""

    def test_optimize_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["optimize"])
        assert args.allow_expansion is False

    def test_optimize_flag_true(self):
        parser = build_parser()
        args = parser.parse_args(["optimize", "--allow-expansion"])
        assert args.allow_expansion is True

    def test_run_all_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["run-all"])
        assert args.allow_expansion is False

    def test_run_all_flag_true(self):
        parser = build_parser()
        args = parser.parse_args(["run-all", "--allow-expansion"])
        assert args.allow_expansion is True
