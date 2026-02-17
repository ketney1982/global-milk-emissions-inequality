# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Small smoke test for the Bayesian module (no full MCMC)."""

import numpy as np
import pytest

pytest.importorskip("pymc")

from methane_portfolio.io import load_all
from methane_portfolio.bayes import prepare_bayes_data, build_model, ConvergenceError


@pytest.fixture(scope="module")
def data():
    long_df, agg_df, species = load_all()
    return long_df, agg_df, species


class TestBayesDataPrep:
    """Data preparation for PyMC model."""

    def test_filters_zeros(self, data):
        long_df, _, _ = data
        bd = prepare_bayes_data(long_df)
        assert (bd["y"] > -np.inf).all()  # log should be finite
        assert len(bd["y"]) > 0

    def test_encoding(self, data):
        long_df, _, _ = data
        bd = prepare_bayes_data(long_df)
        assert bd["n_species"] > 0
        assert bd["n_countries"] > 0
        assert bd["species_id"].max() < bd["n_species"]
        assert bd["country_id"].max() < bd["n_countries"]

    def test_regime_indicator(self, data):
        long_df, _, _ = data
        bd = prepare_bayes_data(long_df)
        # 2020 and 2021 should have regime=0, 2022 and 2023 regime=1
        df = bd["df"]
        assert (df[df["year"] == 2020]["regime"] == 0).all()
        assert (df[df["year"] == 2023]["regime"] == 1).all()


class TestBayesModelBuild:
    """Model construction (no sampling)."""

    def test_model_compiles(self, data):
        long_df, _, _ = data
        bd = prepare_bayes_data(long_df)
        model = build_model(bd)
        assert model is not None
        # Check that free RVs include expected parameters
        rv_names = [rv.name for rv in model.free_RVs]
        named_vars = set(model.named_vars.keys())
        assert "alpha_s" in rv_names
        assert "beta_s" in rv_names
        assert "gamma_s" in rv_names
        assert "u_c_raw" in rv_names
        assert "u_c" in named_vars
        assert "tau" in rv_names
        assert "nu" in rv_names

    def test_non_centered_parameterization(self, data):
        """Verify u_c_raw uses Normal (not ZeroSumNormal) for non-centered param."""
        import pymc as pm
        long_df, _, _ = data
        bd = prepare_bayes_data(long_df)
        model = build_model(bd)
        # u_c_raw should be a free RV (Normal), u_c should be a Deterministic
        rv_names = [rv.name for rv in model.free_RVs]
        det_names = [d.name for d in model.deterministics]
        assert "u_c_raw" in rv_names
        assert "u_c" in det_names

    def test_china_not_double_counted(self, data):
        """Ensure 'China' aggregate (m49=159) is dropped from loaded data."""
        long_df, _, _ = data
        assert 159 not in long_df["country_m49"].values

    def test_convergence_error_is_runtime_error(self):
        """ConvergenceError should be catchable as RuntimeError."""
        assert issubclass(ConvergenceError, RuntimeError)
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("test")
