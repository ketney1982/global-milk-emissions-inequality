# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Manuscript tables generation.

Produces 5 publication tables saved as CSV and LaTeX.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from methane_portfolio import config

logger = logging.getLogger(__name__)
_LATEX_FALLBACK_WARNED = False


def table1_descriptive(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Table 1: Descriptive statistics by species and year.

    Columns: species, year, n_countries, mean_share, mean_intensity,
             total_milk_Mt, total_ch4_MtCO2e
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (sp, yr), g in long_df.groupby(["milk_species", "year"], observed=True):
        rows.append({
            "species": sp,
            "year": yr,
            "n_countries": g["country_m49"].nunique(),
            "mean_share": g["species_share"].mean(),
            "median_share": g["species_share"].median(),
            "mean_intensity_kg_co2e_per_t": g["kg_co2e_per_ton_milk"].mean(),
            "median_intensity_kg_co2e_per_t": g["kg_co2e_per_ton_milk"].median(),
            "total_milk_Mt": g["milk_tonnes"].sum() / 1e6,
        })

    df = pd.DataFrame(rows)
    df.sort_values(["species", "year"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(out / "Table1_descriptive.csv", index=False)
    _to_latex(df, out / "Table1_descriptive.tex", "Descriptive Statistics by Species and Year")
    return df


def table2_shapley_top(
    shapley_df: pd.DataFrame,
    n: int = 20,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Table 2: Top-N countries by |Î”I| with Shapley components."""
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    df = shapley_df.copy()
    df["abs_delta"] = df["delta_total"].abs()
    df = df.nlargest(n, "abs_delta")
    df = df[["country_m49", "country", "delta_struct", "delta_within",
             "delta_total", "delta_obs"]].copy()
    df["struct_pct_of_total"] = (
        df["delta_struct"] / df["delta_total"].replace(0, np.nan) * 100
    )

    df.to_csv(out / "Table2_shapley_top.csv", index=False)
    _to_latex(df, out / "Table2_shapley_top.tex",
              f"Top-{n} Countries by Shapley Decomposition")
    return df


def table3_bayes_summary(
    diag_path: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Table 3: Bayesian model parameter summary (medians + 94% HDI)."""
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    dpath = diag_path or (config.OUTPUT_DIR / "bayes_diagnostics.json")
    if not dpath.exists():
        logger.warning("Bayesian diagnostics not found; skipping Table 3.")
        return pd.DataFrame()

    with open(dpath, "r", encoding="utf-8") as f:
        diag = json.load(f)

    summary = diag.get("summary_table", {})
    if not summary:
        return pd.DataFrame()

    df = pd.DataFrame(summary)
    df.to_csv(out / "Table3_bayes_summary.csv")
    _to_latex(df, out / "Table3_bayes_summary.tex",
              "Bayesian Model Parameter Estimates")
    return df


def table4_optimization(
    opt_df: pd.DataFrame,
    n: int = 20,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Table 4: Top-N countries by optimisation gain."""
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    keep = [c for c in opt_df.columns if not c.startswith("w_opt_") and not c.startswith("w_base_")]
    df = opt_df[keep].head(n).copy()

    df.to_csv(out / "Table4_optimization.csv", index=False)
    _to_latex(df, out / "Table4_optimization.tex",
              f"Top-{n} Countries by Robust Optimisation Gain")
    return df


def table5_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Table 5: Sensitivity grid summary (aggregated over countries)."""
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    agg = (
        sensitivity_df
        .groupby(["delta", "lambda", "alpha"])
        .agg(
            mean_reduction_pct=("reduction_mean_pct", "mean"),
            median_reduction_pct=("reduction_mean_pct", "median"),
            q25_reduction_pct=("reduction_mean_pct", lambda x: np.percentile(x, 25)),
            q75_reduction_pct=("reduction_mean_pct", lambda x: np.percentile(x, 75)),
            mean_cvar_reduction_pct=("reduction_cvar_pct", "mean"),
        )
        .reset_index()
    )

    agg.to_csv(out / "Table5_sensitivity.csv", index=False)
    _to_latex(agg, out / "Table5_sensitivity.tex",
              "Sensitivity Analysis Summary")
    return agg


def make_all_tables(
    long_df: pd.DataFrame | None = None,
    agg_df: pd.DataFrame | None = None,
    shapley_df: pd.DataFrame | None = None,
    opt_df: pd.DataFrame | None = None,
    sensitivity_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate all 5 tables."""
    out = output_dir or config.OUTPUT_DIR

    if long_df is not None and agg_df is not None:
        table1_descriptive(long_df, agg_df, output_dir=out)

    if shapley_df is not None:
        table2_shapley_top(shapley_df, output_dir=out)

    table3_bayes_summary(output_dir=out)

    if opt_df is not None:
        table4_optimization(opt_df, output_dir=out)

    if sensitivity_df is not None:
        table5_sensitivity(sensitivity_df, output_dir=out)


# ---------------------------------------------------------------------------
# LaTeX helper
# ---------------------------------------------------------------------------

def _to_latex(df: pd.DataFrame, path: Path, caption: str) -> None:
    """Write DataFrame as a simple LaTeX table."""
    global _LATEX_FALLBACK_WARNED
    try:
        latex = df.to_latex(
            index=False,
            float_format="%.4f",
            caption=caption,
            label=f"tab:{path.stem.lower()}",
        )
    except ImportError as exc:
        # pandas >=2 routes LaTeX export through Styler, which requires Jinja2.
        # Keep pipeline functional even if optional dependency is missing.
        if not _LATEX_FALLBACK_WARNED:
            logger.info(
                "LaTeX export fallback active (missing Jinja2). "
                "Using internal formatter; install jinja2 to restore native pandas LaTeX formatting. Example error: %s",
                exc,
            )
            _LATEX_FALLBACK_WARNED = True
        latex = _to_latex_fallback(df, caption=caption, label=f"tab:{path.stem.lower()}")
    path.write_text(latex, encoding="utf-8")


def _latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def _to_latex_fallback(df: pd.DataFrame, *, caption: str, label: str) -> str:
    cols = list(df.columns)
    colspec = "l" * len(cols) if cols else "l"
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(_latex_escape(c) for c in cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                if np.isfinite(value):
                    vals.append(f"{float(value):.4f}")
                else:
                    vals.append("NA")
            else:
                vals.append(_latex_escape(value))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines) + "\n"
