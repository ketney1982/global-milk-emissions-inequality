# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Tests for manuscript table generation helpers."""

from __future__ import annotations

import json

import pandas as pd

from methane_portfolio.tables import table3_bayes_summary, table4_optimization


class TestTable3BayesSummary:
    """Table 3 should preserve parameter names in CSV/LaTeX exports."""

    def test_parameter_column_is_explicit(self, tmp_path):
        diag = {
            "summary_table": {
                "mean": {"alpha_s[0]": 0.1, "beta_s[0]": -0.2},
                "sd": {"alpha_s[0]": 0.01, "beta_s[0]": 0.03},
                "hdi_3%": {"alpha_s[0]": 0.05, "beta_s[0]": -0.4},
                "hdi_97%": {"alpha_s[0]": 0.2, "beta_s[0]": 0.0},
                "ess_bulk": {"alpha_s[0]": 250.0, "beta_s[0]": 310.0},
                "ess_tail": {"alpha_s[0]": 240.0, "beta_s[0]": 280.0},
                "r_hat": {"alpha_s[0]": 1.0, "beta_s[0]": 1.01},
            },
        }
        diag_path = tmp_path / "bayes_diagnostics.json"
        diag_path.write_text(json.dumps(diag), encoding="utf-8")

        df = table3_bayes_summary(diag_path=diag_path, output_dir=tmp_path)
        assert "parameter" in df.columns
        assert set(df["parameter"]) == {"alpha_s[0]", "beta_s[0]"}

        csv_df = pd.read_csv(tmp_path / "Table3_bayes_summary.csv")
        assert csv_df.columns[0] == "parameter"

        tex = (tmp_path / "Table3_bayes_summary.tex").read_text(encoding="utf-8")
        assert "parameter" in tex


class TestTable4Optimization:
    """Table 4 should rank by absolute reduction when available."""

    def test_ranks_by_absolute_reduction(self, tmp_path):
        opt_df = pd.DataFrame(
            [
                {"country_m49": 1, "country": "A", "absolute_reduction_kt": 1.0, "reduction_mean_pct": 10.0},
                {"country_m49": 2, "country": "B", "absolute_reduction_kt": 5.0, "reduction_mean_pct": 4.0},
                {"country_m49": 3, "country": "C", "absolute_reduction_kt": 3.0, "reduction_mean_pct": 6.0},
            ]
        )

        out = table4_optimization(opt_df, n=3, output_dir=tmp_path)
        assert out["country"].tolist() == ["B", "C", "A"]
