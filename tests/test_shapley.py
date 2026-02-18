# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Tests for Shapley decomposition."""

import numpy as np
import pandas as pd
import pytest

from methane_portfolio.io import load_all
from methane_portfolio.shapley import (
    build_interval_weights,
    build_wide_matrices,
    shapley_decompose,
)


@pytest.fixture(scope="module")
def data():
    long_df, agg_df, species = load_all()
    return long_df, agg_df, species


class TestBuildWideMatrices:
    """Wide matrix construction."""

    def test_shape(self, data):
        long_df, _, species = data
        m49, W, I = build_wide_matrices(long_df, 2020, species)
        assert W.shape[1] == len(species)
        assert I.shape[1] == len(species)
        assert len(m49) == W.shape[0]

    def test_shares_sum(self, data):
        long_df, _, species = data
        m49, W, I = build_wide_matrices(long_df, 2020, species)
        # Shares should sum to ~1 for each country
        sums = W.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)


class TestBuildIntervalWeights:
    """Interval-weight builder behavior."""

    def test_avg_and_sum_alignment(self):
        idx = pd.Index([100, 200, 300], name="country_m49")
        prod0 = pd.Series([10.0, 20.0, 30.0], index=idx)
        prod1 = pd.Series([30.0, 40.0, 50.0], index=idx)

        w_avg = build_interval_weights(prod0, prod1, method="avg")
        w_sum = build_interval_weights(prod0, prod1, method="sum")

        assert w_avg.index.equals(idx)
        assert w_sum.index.equals(idx)
        np.testing.assert_allclose(w_avg.values, np.array([20.0, 30.0, 40.0]))
        np.testing.assert_allclose(w_sum.values, np.array([40.0, 60.0, 80.0]))


class TestShapleyDecomposition:
    """Full Shapley decomposition."""

    def test_reconstruction(self, data, tmp_path):
        long_df, agg_df, species = data
        result = shapley_decompose(long_df, agg_df, species, output_dir=tmp_path)
        # struct + within should equal total
        np.testing.assert_allclose(
            result["delta_struct"] + result["delta_within"],
            result["delta_total"],
            atol=1e-10,
        )

    def test_matches_observed(self, data, tmp_path):
        long_df, agg_df, species = data
        result = shapley_decompose(long_df, agg_df, species, output_dir=tmp_path)
        np.testing.assert_allclose(
            result["delta_total"].values,
            result["delta_obs"].values,
            atol=1e-6,
        )

    def test_output_files(self, data, tmp_path):
        long_df, agg_df, species = data
        shapley_decompose(long_df, agg_df, species, output_dir=tmp_path)
        assert (tmp_path / "shapley_country.csv").exists()
        assert (tmp_path / "shapley_global.json").exists()
