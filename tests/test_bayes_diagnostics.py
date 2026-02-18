# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Unit tests for Bayesian diagnostics helpers."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from methane_portfolio import config
from methane_portfolio.bayes import (
    _compute_diagnostics,
    _posterior_for_ppc,
    _ppc_diagnostics,
    _ppc_summary,
)
from methane_portfolio.config import BAYES_ESS_MIN, BAYES_RHAT_THRESHOLD


class TestComputeDiagnostics:
    """Convergence diagnostics should expose strict/relaxed status and failed params."""

    def test_strict_vs_relaxed_flags(self, monkeypatch):
        summary = pd.DataFrame(
            {
                "r_hat": [1.0, 1.04, 1.0],
                "ess_bulk": [1200.0, 350.0, 900.0],
                "ess_tail": [1100.0, 380.0, 850.0],
            },
            index=["alpha_s[0]", "tau", "nu"],
        )
        monkeypatch.setattr(az, "summary", lambda *args, **kwargs: summary)

        idata = az.InferenceData(
            sample_stats=xr.Dataset(
                {"diverging": (("chain", "draw"), np.zeros((2, 4), dtype=int))}
            )
        )
        diag = _compute_diagnostics(idata)

        assert diag["converged"] is False
        assert diag["converged_relaxed"] is True
        assert "tau" in diag["rhat_fail_params"]
        assert "tau" in diag["ess_bulk_fail_params"]
        assert "tau" in diag["ess_tail_fail_params"]
        assert diag["thresholds"]["rhat_strict"] == BAYES_RHAT_THRESHOLD
        assert diag["thresholds"]["ess_strict"] == BAYES_ESS_MIN

    def test_divergences_fail_relaxed_too(self, monkeypatch):
        summary = pd.DataFrame(
            {
                "r_hat": [1.0],
                "ess_bulk": [1000.0],
                "ess_tail": [1000.0],
            },
            index=["tau"],
        )
        monkeypatch.setattr(az, "summary", lambda *args, **kwargs: summary)

        idata = az.InferenceData(
            sample_stats=xr.Dataset(
                {"diverging": (("chain", "draw"), np.ones((1, 3), dtype=int))}
            )
        )
        diag = _compute_diagnostics(idata)

        assert diag["divergences"] == 3
        assert diag["converged"] is False
        assert diag["converged_relaxed"] is False


class TestPpcDiagnostics:
    """Posterior predictive diagnostics should summarize residual behavior."""

    def test_ppc_diagnostics_summary(self):
        ppc = pd.DataFrame(
            {
                "residual": [0.0, 0.1, -0.2, 3.5, -4.1],
                "within_90ci": [True, True, True, False, False],
            }
        )
        diag = _ppc_diagnostics(ppc)

        assert diag["n_obs"] == 5
        assert diag["coverage_90ci"] == pytest.approx(0.6)
        assert diag["n_abs_residual_gt_2"] == 2
        assert diag["n_abs_residual_gt_3"] == 2
        assert diag["residual_max_abs"] == pytest.approx(4.1)
        assert diag["residual_trimmed_mean_10pct"] == pytest.approx(np.mean(ppc["residual"]))


class TestPpcSummary:
    """Posterior predictive summary should cap draw count for speed."""

    def test_ppc_summary_subsamples_draws(self, monkeypatch):
        monkeypatch.setattr(config, "BAYES_PPC_MAX_DRAWS", 3)

        y_rep = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
        idata = az.InferenceData(
            posterior_predictive=xr.Dataset(
                {"y_obs": (("chain", "draw", "obs"), y_rep)}
            )
        )
        data = {"y": np.zeros(5, dtype=float)}

        out = _ppc_summary(idata, data)

        rng = np.random.default_rng(config.RNG_SEED)
        idx = rng.choice(8, size=3, replace=False)
        expected_mean = y_rep.reshape(8, 5)[idx].mean(axis=0)

        assert list(out.columns) == [
            "obs_idx",
            "y_obs",
            "y_rep_mean",
            "y_rep_median",
            "y_rep_p05",
            "y_rep_p95",
            "residual",
            "residual_median_pred",
            "within_90ci",
        ]
        assert len(out) == 5
        assert_allclose(out["y_rep_mean"].to_numpy(), expected_mean)


class TestPosteriorForPpc:
    """Posterior subset should cap draw volume for PPC generation."""

    def test_caps_draws_per_chain(self):
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "alpha_s": (
                        ("chain", "draw", "species"),
                        np.ones((4, 10, 1), dtype=float),
                    )
                }
            )
        )

        sub = _posterior_for_ppc(idata, max_draws=12)
        assert int(sub.posterior.sizes["chain"]) == 4
        assert int(sub.posterior.sizes["draw"]) == 3
