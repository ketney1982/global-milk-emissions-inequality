"""Small smoke test for the Bayesian module (no full MCMC)."""

import numpy as np
import pytest

from methane_portfolio.io import load_all
from methane_portfolio.bayes import prepare_bayes_data, build_model


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
        assert "alpha_s" in rv_names
        assert "beta_s" in rv_names
        assert "gamma_s" in rv_names
        assert "u_c" in rv_names
        assert "tau" in rv_names
        assert "nu" in rv_names
