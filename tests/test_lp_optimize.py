# Autor: Ketney Otto
# Affiliation: Lucian Blaga University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: ketney.otto@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Tests for the exact LP form of the mean-CVaR species-portfolio problem."""
from __future__ import annotations

import numpy as np
import pytest

from methane_portfolio.lp_optimize import empirical_cvar, solve_lp
from methane_portfolio.robust_optimize import solve_robust


def _objective(w, I_scen, lam, alpha):
    port = I_scen @ w
    return lam * port.mean() + (1.0 - lam) * empirical_cvar(port, alpha)


@pytest.fixture()
def scenarios():
    rng = np.random.default_rng(7)
    # 3 species with clearly separated intensities, 400 lognormal scenarios
    mu = np.log(np.array([0.03, 0.30, 1.20]))
    sd = np.array([0.15, 0.45, 0.70])
    return np.exp(rng.normal(mu, sd, size=(400, 3)))


def test_empirical_cvar_matches_definition():
    x = np.arange(1.0, 101.0)
    alpha = 0.90
    t = np.percentile(x, 90.0)
    expected = t + np.maximum(0.0, x - t).mean() / (1.0 - alpha)
    assert empirical_cvar(x, alpha) == pytest.approx(expected)
    # CVaR must not fall below the quantile it is taken from
    assert empirical_cvar(x, alpha) >= t


def test_lp_respects_simplex_and_budget(scenarios):
    w_ref = np.array([0.70, 0.20, 0.10])
    for delta in (0.0, 0.01, 0.05, 0.10, 0.25):
        sol = solve_lp(w_ref, scenarios, lam=0.5, alpha=0.9, delta=delta)
        assert sol["success"]
        w = sol["w_opt"]
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()
        tv = 0.5 * np.abs(w - w_ref).sum()
        assert tv <= delta + 1e-7, f"budget violated at delta={delta}: {tv}"


def test_lp_no_expansion_keeps_absent_species_at_zero(scenarios):
    w_ref = np.array([0.80, 0.20, 0.00])          # third species absent
    sol = solve_lp(w_ref, scenarios, delta=0.20, allow_expansion=False)
    assert sol["w_opt"][2] == pytest.approx(0.0)
    sol_x = solve_lp(w_ref, scenarios, delta=0.20, allow_expansion=True)
    assert sol_x["success"]


def test_lp_moves_share_to_the_lowest_intensity_species(scenarios):
    w_ref = np.array([0.50, 0.30, 0.20])
    sol = solve_lp(w_ref, scenarios, lam=0.5, alpha=0.9, delta=0.10)
    # species 0 has the lowest intensity, so it must gain exactly the budget
    assert sol["w_opt"][0] > w_ref[0]
    assert sol["w_opt"][0] - w_ref[0] == pytest.approx(0.10, abs=1e-6)


def test_lp_is_never_worse_than_the_nonlinear_solver(scenarios):
    """The LP is exact, so its objective must be <= the nonlinear solver's."""
    rng = np.random.default_rng(11)
    lam, alpha, delta = 0.5, 0.9, 0.10
    for _ in range(12):
        w_ref = rng.dirichlet(np.array([6.0, 2.0, 1.0]))
        lp = solve_lp(w_ref, scenarios, lam=lam, alpha=alpha, delta=delta)
        nl = solve_robust(w_ref, scenarios, lam=lam, alpha=alpha, delta=delta)
        f_lp = _objective(lp["w_opt"], scenarios, lam, alpha)
        f_nl = _objective(nl["w_opt"], scenarios, lam, alpha)
        assert f_lp <= f_nl + 1e-9, f"LP worse than nonlinear: {f_lp} > {f_nl}"


def test_lp_matches_brute_force_on_a_coarse_simplex(scenarios):
    """Against an exhaustive grid search over the feasible set."""
    w_ref = np.array([0.60, 0.25, 0.15])
    lam, alpha, delta = 0.5, 0.9, 0.05
    sol = solve_lp(w_ref, scenarios, lam=lam, alpha=alpha, delta=delta)
    best = np.inf
    grid = np.linspace(0.0, 1.0, 101)
    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c < -1e-12:
                continue
            w = np.array([a, b, max(c, 0.0)])
            if 0.5 * np.abs(w - w_ref).sum() > delta + 1e-12:
                continue
            best = min(best, _objective(w, scenarios, lam, alpha))
    f_lp = _objective(sol["w_opt"], scenarios, lam, alpha)
    # the LP optimises over the continuum, so it can only beat a 0.01-spaced grid
    assert f_lp <= best + 1e-9


def test_ceiling_constraint_is_honoured(scenarios):
    w_ref = np.array([0.40, 0.35, 0.25])
    Ibar = scenarios.mean(axis=0)
    ref_mean = float(Ibar @ w_ref)
    sol = solve_lp(w_ref, scenarios, delta=0.20, ceiling=ref_mean)
    assert sol["success"]
    assert float(Ibar @ sol["w_opt"]) <= ref_mean + 1e-9


def test_zero_budget_returns_the_reference_mix(scenarios):
    w_ref = np.array([0.55, 0.30, 0.15])
    sol = solve_lp(w_ref, scenarios, delta=0.0)
    assert sol["w_opt"] == pytest.approx(w_ref, abs=1e-7)
    assert sol["tv"] == pytest.approx(0.0, abs=1e-7)


class TestPosteriorBaselineRemovesTheGuard:
    """With a posterior-consistent reference the do-no-harm guard cannot bind.

    The reference mix is always feasible under the budget constraint, so a certified
    optimum of a convex objective can never return a strictly worse mean. Under the R1
    configuration, where the reference came from the observed intensities while the
    optimum was evaluated on posterior scenarios, the guard fired whenever posterior
    shrinkage moved the two scales apart.
    """

    @staticmethod
    def _panel():
        import pandas as pd
        return pd.DataFrame(
            [
                {"country_m49": 1, "country": "RiskLand", "year": 2023,
                 "milk_species": "Raw milk of cattle", "species_share": 0.5,
                 "milk_tonnes": 100.0, "kg_co2e_per_ton_milk": 1.0},
                {"country_m49": 1, "country": "RiskLand", "year": 2023,
                 "milk_species": "Raw milk of goat", "species_share": 0.5,
                 "milk_tonnes": 100.0, "kg_co2e_per_ton_milk": 1.0},
            ],
        )

    # posterior scenarios an order of magnitude above the observed ratio
    _SAMPLES = np.array([[[10.0, 20.0]], [[11.0, 19.0]], [[9.5, 18.5]]])
    _SPECIES = ["Raw milk of cattle", "Raw milk of goat"]

    def test_guard_does_not_fire_with_posterior_reference(self, tmp_path):
        from methane_portfolio.robust_optimize import run_all_countries
        out = run_all_countries(
            self._panel(), I_samples=self._SAMPLES, country_list=[1],
            species_list=self._SPECIES, year=2023, output_dir=tmp_path,
            save_csv=False, do_no_harm=True,           # R2 defaults
        )
        row = out.iloc[0]
        assert bool(row["no_harm_applied"]) is False
        assert row["reduction_mean_pct"] >= -1e-9      # never negative
        assert bool(row["reference_is_posterior"]) is True
        assert row["solver"] == "highs-lp"
        assert bool(row["lp_certified"]) is True
        # the reference is the posterior mean of the observed mix, not the observed ratio
        assert row["baseline_intensity"] == pytest.approx(
            float((self._SAMPLES[:, 0, :] @ np.array([0.5, 0.5])).mean()))
        assert row["baseline_intensity_observed"] == pytest.approx(1.0)

    def test_guard_does_fire_under_the_r1_configuration(self, tmp_path):
        from methane_portfolio.robust_optimize import run_all_countries
        out = run_all_countries(
            self._panel(), I_samples=self._SAMPLES, country_list=[1],
            species_list=self._SPECIES, year=2023, output_dir=tmp_path,
            save_csv=False, do_no_harm=True, posterior_baseline=False,
        )
        row = out.iloc[0]
        assert bool(row["no_harm_applied"]) is True
        assert row["baseline_intensity"] == pytest.approx(1.0)

    def test_raw_columns_are_the_unguarded_solution(self, tmp_path):
        """R1 exported the guarded solution into raw_*; R2 must not."""
        from methane_portfolio.robust_optimize import run_all_countries
        out = run_all_countries(
            self._panel(), I_samples=self._SAMPLES, country_list=[1],
            species_list=self._SPECIES, year=2023, output_dir=tmp_path,
            save_csv=False, do_no_harm=True, posterior_baseline=False,
        )
        row = out.iloc[0]
        # the guard reverted the reported solution to the reference, so the unguarded
        # solver output must differ from it
        assert not np.isclose(row["raw_optimized_mean"], row["optimized_mean"])
        assert row["no_harm_excess_raw"] > 0
