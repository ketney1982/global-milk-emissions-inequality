"""Tests for advanced analytics helpers."""

import numpy as np
import pandas as pd

from methane_portfolio.advanced_analytics import hill_pareto_exponent, loo_influence


class TestLooInfluence:
    """Leave-one-out influence should match brute-force computation."""

    def test_vectorized_matches_bruteforce(self):
        df = pd.DataFrame({
            "country_m49": [10, 20, 30],
            "country": ["A", "B", "C"],
            "delta_total": [1.0, -2.0, 0.5],
            "weight_interval": [2.0, 3.0, 5.0],
        })

        out = loo_influence(df, component="total", weights_col="weight_interval")
        out = out.sort_values("country_m49").reset_index(drop=True)

        w = df["weight_interval"].to_numpy(dtype=float)
        d = df["delta_total"].to_numpy(dtype=float)
        S = np.sum(w * d)
        W = np.sum(w)
        g = S / W

        brute_abs = []
        brute_rel = []
        for i in range(len(df)):
            g_loo = (S - w[i] * d[i]) / (W - w[i])
            inf_abs = g_loo - g
            brute_abs.append(inf_abs)
            brute_rel.append(inf_abs / g if abs(g) > 1e-12 else np.nan)

        np.testing.assert_allclose(out["influence_abs"].to_numpy(), np.array(brute_abs), atol=1e-12)
        np.testing.assert_allclose(out["influence_rel"].to_numpy(), np.array(brute_rel), atol=1e-12)


class TestHillParetoExponent:
    """Hill estimator should recover tail exponent on synthetic Pareto data."""

    def test_recovers_alpha_on_synthetic_pareto(self):
        rng = np.random.default_rng(123)
        alpha_true = 2.0
        # numpy pareto(a) returns Y with tail exponent a for (Y + 1).
        x = rng.pareto(alpha_true, size=50000) + 1.0

        out = hill_pareto_exponent(x, top_frac=0.10, n_bootstrap=50, seed=123)
        assert np.isfinite(out["alpha_hat"])
        assert abs(out["alpha_hat"] - alpha_true) < 0.7
