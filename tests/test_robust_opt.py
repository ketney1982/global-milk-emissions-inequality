"""Tests for robust portfolio optimisation."""

import numpy as np
import pytest

from methane_portfolio.optimize import (
    mean_intensity,
    simplex_constraint,
    tv_distance_constraint,
)
from methane_portfolio.robust_optimize import solve_robust


class TestOptimizeHelpers:
    """Helper function tests."""

    def test_simplex_constraint(self):
        c = simplex_constraint(5)
        assert c.lb == 1.0
        assert c.ub == 1.0

    def test_tv_distance_satisfied(self):
        w_ref = np.array([0.5, 0.3, 0.2])
        con = tv_distance_constraint(w_ref, 0.1)
        # w = w_ref → TV = 0 ≤ 0.1 → positive (satisfied)
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
        # Species 0 has high intensity, species 2 has low → optimizer
        # should shift weight away from 0
        I_scenarios = np.column_stack([
            rng.lognormal(np.log(10), 0.1, 200),
            rng.lognormal(np.log(5), 0.1, 200),
            rng.lognormal(np.log(2), 0.1, 200),
        ])
        sol = solve_robust(w_ref, I_scenarios, delta=0.2)
        np.testing.assert_allclose(sol["w_opt"].sum(), 1.0, atol=1e-8)

    def test_reduction_positive(self):
        """With room to shift (delta>0), optimised mean should be ≤ baseline."""
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
