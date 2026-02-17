# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Tests for robust portfolio optimisation."""

import json
import logging
import numpy as np
import pandas as pd
import pytest

import methane_portfolio.robust_optimize as robust_optimize
import methane_portfolio.cli as cli_module
from methane_portfolio.cli import build_parser
from methane_portfolio.optimize import (
    mean_intensity,
    simplex_constraint,
    tv_distance_constraint,
)
from methane_portfolio.robust_optimize import run_all_countries, solve_robust
from methane_portfolio.uncertainty import run_sensitivity_grid


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

    def test_iteration_limit_retries_with_higher_maxiter(self, monkeypatch):
        """Iteration-limit failures should trigger an SLSQP retry."""

        class _Res:
            def __init__(self, *, x, success, message):
                self.x = x
                self.success = success
                self.message = message

        calls: list[tuple[str, int]] = []

        def _fake_minimize(*args, **kwargs):
            method = kwargs["method"]
            maxiter = kwargs["options"]["maxiter"]
            calls.append((method, int(maxiter)))
            if len(calls) == 1:
                return _Res(
                    x=np.array([0.6, 0.4, 0.0]),
                    success=False,
                    message="Iteration limit reached",
                )
            return _Res(
                x=np.array([0.58, 0.42, 0.0]),
                success=True,
                message="Optimization terminated successfully",
            )

        monkeypatch.setattr(robust_optimize, "minimize", _fake_minimize)
        w_ref = np.array([0.6, 0.4])
        I_scenarios = np.array([[8.0, 3.0], [9.0, 2.8], [7.5, 3.1]])
        sol = solve_robust(w_ref, I_scenarios, maxiter=10)

        assert calls[0] == ("SLSQP", 10)
        assert calls[1] == ("SLSQP", 6000)
        assert len(calls) == 2
        assert sol["success"] is True
        assert sol["solver_method"] == "SLSQP"
        assert len(sol["solver_attempts"]) == 2

    def test_iteration_limit_can_fallback_to_trust_constr(self, monkeypatch):
        """Second iteration-limit failure should escalate to trust-constr."""

        class _Res:
            def __init__(self, *, x, success, message):
                self.x = x
                self.success = success
                self.message = message

        calls: list[tuple[str, int]] = []

        def _fake_minimize(*args, **kwargs):
            method = kwargs["method"]
            maxiter = kwargs["options"]["maxiter"]
            calls.append((method, int(maxiter)))
            if len(calls) < 3:
                return _Res(
                    x=np.array([0.6, 0.4, 0.0]),
                    success=False,
                    message="Iteration limit reached",
                )
            return _Res(
                x=np.array([0.57, 0.43, 0.0]),
                success=True,
                message="`gtol` termination condition is satisfied.",
            )

        monkeypatch.setattr(robust_optimize, "minimize", _fake_minimize)
        w_ref = np.array([0.6, 0.4])
        I_scenarios = np.array([[8.0, 3.0], [9.0, 2.8], [7.5, 3.1]])
        sol = solve_robust(w_ref, I_scenarios, maxiter=10)

        assert [m for m, _ in calls] == ["SLSQP", "SLSQP", "trust-constr"]
        assert sol["success"] is True
        assert sol["solver_method"] == "trust-constr"
        assert len(sol["solver_attempts"]) == 3


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

    def test_country_name_normalized_by_m49(self, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 156,
                    "country": "China",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.8,
                    "milk_tonnes": 80.0,
                    "kg_co2e_per_ton_milk": 9.0,
                },
                {
                    "country_m49": 156,
                    "country": "China, mainland",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.2,
                    "milk_tonnes": 20.0,
                    "kg_co2e_per_ton_milk": 5.0,
                },
            ],
        )

        out = run_all_countries(
            long_df,
            year=2023,
            output_dir=tmp_path,
            save_csv=False,
            do_no_harm=False,
        )
        assert len(out) == 1
        assert int(out.iloc[0]["country_m49"]) == 156


class TestRunAllCountriesGuards:
    """Regression guards for do-no-harm and output persistence."""

    def test_do_no_harm_reverts_harmful_solution(self, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "RiskLand",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.5,
                    "milk_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 1.0,
                },
                {
                    "country_m49": 1,
                    "country": "RiskLand",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.5,
                    "milk_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 1.0,
                },
            ],
        )
        # Posterior scenarios intentionally much higher than observed baseline.
        i_samples = np.array(
            [
                [[10.0, 20.0]],
                [[11.0, 19.0]],
                [[9.5, 18.5]],
            ],
        )

        out = run_all_countries(
            long_df,
            I_samples=i_samples,
            country_list=[1],
            species_list=["Raw milk of cattle", "Raw milk of goat"],
            year=2023,
            output_dir=tmp_path,
            save_csv=False,
            do_no_harm=True,
        )

        row = out.iloc[0]
        assert np.isclose(row["baseline_intensity"], 1.0)
        assert np.isclose(row["raw_optimized_mean"], row["baseline_intensity"])
        assert np.isclose(row["raw_reduction_mean_pct"], 0.0)
        assert np.isclose(row["optimized_mean"], row["baseline_intensity"])
        assert np.isclose(row["reduction_mean_pct"], 0.0)
        assert bool(row["no_harm_applied"]) is True
        assert row["no_harm_action"] in {"baseline_revert", "constrained_solution"}
        assert row["no_harm_excess_raw"] > 0
        assert "do-no-harm" in row["solver_message"]

    def test_sensitivity_grid_does_not_overwrite_main_optimization_csv(self, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "A",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.8,
                    "milk_tonnes": 80.0,
                    "kg_co2e_per_ton_milk": 10.0,
                },
                {
                    "country_m49": 1,
                    "country": "A",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.2,
                    "milk_tonnes": 20.0,
                    "kg_co2e_per_ton_milk": 4.0,
                },
                {
                    "country_m49": 2,
                    "country": "B",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.6,
                    "milk_tonnes": 60.0,
                    "kg_co2e_per_ton_milk": 8.0,
                },
                {
                    "country_m49": 2,
                    "country": "B",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.4,
                    "milk_tonnes": 40.0,
                    "kg_co2e_per_ton_milk": 5.0,
                },
            ],
        )

        run_all_countries(long_df, year=2023, output_dir=tmp_path, save_csv=True)
        main_csv = tmp_path / "robust_optimization_results.csv"
        before = main_csv.read_bytes()

        run_sensitivity_grid(
            long_df,
            year=2023,
            n_countries_max=1,
            workers=1,
            output_dir=tmp_path,
        )
        after = main_csv.read_bytes()

        assert before == after

    def test_audit_json_records_guard_application(self, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "RiskLand",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.5,
                    "milk_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 1.0,
                },
                {
                    "country_m49": 1,
                    "country": "RiskLand",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.5,
                    "milk_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 1.0,
                },
            ],
        )
        i_samples = np.array(
            [
                [[10.0, 20.0]],
                [[11.0, 19.0]],
                [[9.5, 18.5]],
            ],
        )

        run_all_countries(
            long_df,
            I_samples=i_samples,
            country_list=[1],
            species_list=["Raw milk of cattle", "Raw milk of goat"],
            year=2023,
            output_dir=tmp_path,
            save_csv=True,
            save_audit=True,
            do_no_harm=True,
        )

        audit_path = tmp_path / "robust_optimization_audit.json"
        assert audit_path.exists()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["do_no_harm_enabled"] is True
        assert audit["n_no_harm_applied"] == 1
        assert audit["n_negative_raw_reductions"] >= audit["n_negative_final_reductions"]


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


class TestCliRunAllSkipBayes:
    """run-all --skip-bayes should still produce full downstream outputs."""

    def test_skip_bayes_runs_sensitivity_grid(self, monkeypatch, tmp_path):
        long_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "A",
                    "year": 2023,
                    "milk_species": "Raw milk of cattle",
                    "species_share": 0.7,
                    "milk_tonnes": 70.0,
                    "kg_co2e_per_ton_milk": 10.0,
                },
                {
                    "country_m49": 1,
                    "country": "A",
                    "year": 2023,
                    "milk_species": "Raw milk of goat",
                    "species_share": 0.3,
                    "milk_tonnes": 30.0,
                    "kg_co2e_per_ton_milk": 5.0,
                },
            ],
        )
        agg_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "A",
                    "year": 2023,
                    "milk_total_tonnes": 100.0,
                    "kg_co2e_per_ton_milk": 8.5,
                },
            ],
        )
        species = sorted(long_df["milk_species"].unique())
        shapley_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "A",
                    "delta_struct": 0.0,
                    "delta_within": 0.0,
                    "delta_total": 0.0,
                    "delta_obs": 0.0,
                },
            ],
        )
        opt_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "A",
                    "reduction_mean_pct": 0.0,
                },
            ],
        )
        unc_df = pd.DataFrame([{"country_m49": 1, "country": "A"}])
        sens_df = pd.DataFrame(
            [
                {
                    "country_m49": 1,
                    "country": "A",
                    "baseline_intensity_kg_co2e_per_t": 8.5,
                    "optimized_mean_kg_co2e_per_t": 8.5,
                    "optimized_cvar_kg_co2e_per_t": 8.5,
                    "reduction_mean_pct": 0.0,
                    "reduction_cvar_pct": 0.0,
                    "delta": 0.1,
                    "lambda": 0.5,
                    "alpha": 0.9,
                },
            ],
        )
        calls = {"sensitivity": 0}

        monkeypatch.setattr(cli_module.config, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(cli_module.config, "FIGURE_DIR", tmp_path / "figures")
        monkeypatch.setattr(cli_module, "write_manifest", lambda *_args, **_kwargs: None)

        import methane_portfolio.figures as figures_module
        import methane_portfolio.io as io_module
        import methane_portfolio.report as report_module
        import methane_portfolio.shapley as shapley_module
        import methane_portfolio.tables as tables_module
        import methane_portfolio.uncertainty as uncertainty_module
        import methane_portfolio.validate as validate_module

        monkeypatch.setattr(
            io_module,
            "load_all",
            lambda _data_dir: (long_df.copy(), agg_df.copy(), species),
        )
        monkeypatch.setattr(
            validate_module,
            "validate_all",
            lambda *_args, **_kwargs: pd.DataFrame({"status": ["PASS"]}),
        )
        monkeypatch.setattr(
            shapley_module,
            "shapley_decompose",
            lambda *_args, **_kwargs: shapley_df.copy(),
        )
        monkeypatch.setattr(
            robust_optimize,
            "run_all_countries",
            lambda *_args, **_kwargs: opt_df.copy(),
        )
        monkeypatch.setattr(
            uncertainty_module,
            "propagate_uncertainty",
            lambda *_args, **_kwargs: unc_df.copy(),
        )

        def _fake_run_sensitivity_grid(*_args, **_kwargs):
            calls["sensitivity"] += 1
            return sens_df.copy()

        monkeypatch.setattr(
            uncertainty_module,
            "run_sensitivity_grid",
            _fake_run_sensitivity_grid,
        )
        monkeypatch.setattr(tables_module, "make_all_tables", lambda **_kwargs: None)
        monkeypatch.setattr(figures_module, "make_all_figures", lambda **_kwargs: None)
        monkeypatch.setattr(
            report_module,
            "generate_appendix",
            lambda **_kwargs: tmp_path / "methods_appendix.md",
        )

        parser = build_parser()
        args = parser.parse_args(["run-all", "--skip-bayes"])
        cli_module.cmd_run_all(args)

        assert calls["sensitivity"] == 1
