# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Advanced manuscript tables and figures.

Run:
    python -m methane_portfolio.figures_advanced --weight-method avg --mc-n 1000 --delta 0.10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methane_portfolio import config
from methane_portfolio.advanced_analytics import (
    build_phase_space,
    hill_pareto_exponent,
    inequality_decomposition,
    loo_influence,
    weighted_gini,
    weighted_lorenz_curve,
)
from methane_portfolio.io import load_all, load_region_mapping
from methane_portfolio.robust_optimize import run_all_countries
from methane_portfolio.shapley import shapley_decompose
from methane_portfolio.utils import ensure_dirs
from methane_portfolio.validate import validate_all

logger = logging.getLogger(__name__)


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def _attach_regions(df: pd.DataFrame, regions_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(regions_df, on="country_m49", how="left")
    missing_mask = out["region"].isna() | (out["region"].astype(str).str.strip() == "")
    if missing_mask.any():
        missing_codes = sorted(out.loc[missing_mask, "country_m49"].astype(int).unique().tolist())
        logger.warning(
            "Missing region mapping for %d countries; setting region='Unknown'. Sample: %s",
            missing_mask.sum(),
            missing_codes[:10],
        )
        out.loc[missing_mask, "region"] = "Unknown"
    return out


def _weighted_global(delta: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * delta) / np.sum(w))


def _mc_global_table(
    shapley_df: pd.DataFrame,
    weight_method: str,
    mc_n: int,
    seed: int = config.RNG_SEED,
) -> pd.DataFrame:
    if mc_n <= 0:
        raise ValueError("mc_n must be positive.")

    work = shapley_df.copy()
    work = work[np.isfinite(work["weight_interval"]) & (work["weight_interval"] > 0)]
    if work.empty:
        raise ValueError("No valid rows with positive weight_interval for MC summary.")

    w = work["weight_interval"].to_numpy(dtype=float)
    p = w / w.sum()
    n = len(work)
    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(n, size=(mc_n, n), replace=True, p=p)
    sampled_w = w[draw_idx]

    rows = []
    for comp, col in [("struct", "delta_struct"), ("within", "delta_within"), ("total", "delta_total")]:
        d = work[col].to_numpy(dtype=float)
        sampled_d = d[draw_idx]
        draws = np.sum(sampled_w * sampled_d, axis=1) / np.sum(sampled_w, axis=1)
        rows.append({
            "component": comp,
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws, ddof=1)),
            "p2.5": float(np.percentile(draws, 2.5)),
            "p50": float(np.percentile(draws, 50)),
            "p97.5": float(np.percentile(draws, 97.5)),
            "weight_method": weight_method,
            "N": int(mc_n),
        })
    return pd.DataFrame(rows)


def _plot_phase_diagram(df_phase: pd.DataFrame, path: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    work = df_phase[np.isfinite(df_phase["delta_struct"]) & np.isfinite(df_phase["reduction_pct"])].copy()
    if work.empty:
        raise ValueError("No valid rows available for phase diagram.")

    w = work["weight_interval"].to_numpy(dtype=float)
    w = np.clip(w, a_min=0.0, a_max=None)
    max_w = w.max() if w.size > 0 else 1.0
    sizes = 20 + 380 * np.sqrt(w / max_w) if max_w > 0 else np.full_like(w, 20.0)
    work["_size"] = sizes

    regions = sorted(work["region"].fillna("Unknown").unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {r: cmap(i % 10) for i, r in enumerate(regions)}

    for region in regions:
        sub = work[work["region"] == region]
        ax.scatter(
            sub["delta_struct"],
            sub["reduction_pct"],
            s=sub["_size"],
            c=[color_map[region]],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.4,
            label=region,
            zorder=3,
        )

    # Optional 2D KDE contour.
    try:
        from scipy.stats import gaussian_kde

        if len(work) >= 8:
            x = work["delta_struct"].to_numpy(dtype=float)
            y = work["reduction_pct"].to_numpy(dtype=float)
            if np.std(x) > 0 and np.std(y) > 0:
                kde = gaussian_kde(np.vstack([x, y]))
                xg = np.linspace(np.min(x), np.max(x), 120)
                yg = np.linspace(np.min(y), np.max(y), 120)
                xx, yy = np.meshgrid(xg, yg)
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors="#444444", linewidths=0.7, alpha=0.45, zorder=2)
    except Exception:
        logger.info("KDE contour not generated (scipy.stats.gaussian_kde unavailable or singular).")

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    y_med = float(np.median(work["reduction_pct"]))
    ax.axhline(y_med, color="black", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Î”_struct (2020â†’2023)")
    ax.set_ylabel("Reduction potential (%)")
    ax.set_title("Structural Phase Diagram")
    ax.grid(True, linewidth=0.35, alpha=0.35, color="#D9D9D9", zorder=1)
    ax.legend(frameon=False, ncol=2, loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_concentration_curve(shapley_df: pd.DataFrame, path: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    work = shapley_df.copy()
    w = work["weight_interval"].to_numpy(dtype=float)
    x_total = np.abs(w * work["delta_total"].to_numpy(dtype=float))
    x_struct = np.abs(w * work["delta_struct"].to_numpy(dtype=float))

    p_t, L_t = weighted_lorenz_curve(x_total, w)
    p_s, L_s = weighted_lorenz_curve(x_struct, w)
    g_t = weighted_gini(x_total, w)
    g_s = weighted_gini(x_struct, w)

    ax.plot(p_t, L_t, color="#1B7FA6", linewidth=2.0, label=f"|wÂ·Î”_total| (Gini={g_t:.3f})")
    ax.plot(p_s, L_s, color="#C0392B", linewidth=2.0, label=f"|wÂ·Î”_struct| (Gini={g_s:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--", alpha=0.8, label="Equality line")

    ax.set_xlabel("Cumulative weight share")
    ax.set_ylabel("Cumulative contribution share")
    ax.set_title("Concentration Curves")
    ax.grid(True, linewidth=0.35, alpha=0.35, color="#D9D9D9")
    ax.legend(frameon=False, loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_advanced_exports(
    *,
    weight_method: Literal["base", "end", "avg", "sum", "trapz"] = "avg",
    mc_n: int = 1000,
    delta: float = 0.10,
    lam: float = 0.5,
    alpha: float = 0.90,
    top_frac: float = 0.10,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run advanced analytics pipeline and export manuscript-ready outputs."""
    out = output_dir or config.OUTPUT_DIR
    fig_dir = config.FIGURE_DIR if output_dir is None else Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    long_df, agg_df, species = load_all(data_dir)
    validate_all(long_df, agg_df, output_dir=out, raise_on_fail=True)

    shapley_df = shapley_decompose(
        long_df,
        agg_df,
        species,
        weight_method=weight_method,
        output_dir=out,
    )
    opt_df = run_all_countries(
        long_df,
        delta=delta,
        lam=lam,
        alpha=alpha,
        output_dir=out,
    )

    regions_df = load_region_mapping(data_dir)
    shapley_df = _attach_regions(shapley_df, regions_df)

    # Table: MC global summaries
    table_mc = _mc_global_table(shapley_df, weight_method=weight_method, mc_n=mc_n)
    path_mc = out / "Table_MC_Global.csv"
    table_mc.to_csv(path_mc, index=False)

    # Table: inequality decomposition
    ineq_rows = [
        inequality_decomposition(shapley_df, component="struct"),
        inequality_decomposition(shapley_df, component="within"),
        inequality_decomposition(shapley_df, component="total"),
    ]
    table_ineq = pd.DataFrame(ineq_rows)
    path_ineq = out / "Table_Inequality_Decomposition.csv"
    table_ineq.to_csv(path_ineq, index=False)

    # Table: LOO influence top-20
    loo_total = loo_influence(shapley_df, component="total").assign(component="total")
    loo_struct = loo_influence(shapley_df, component="struct").assign(component="struct")
    loo_top = pd.concat(
        [
            loo_total.reindex(loo_total["influence_abs"].abs().nlargest(20).index),
            loo_struct.reindex(loo_struct["influence_abs"].abs().nlargest(20).index),
        ],
        ignore_index=True,
    )
    path_loo = out / "Table_LOO_Influence_Top20.csv"
    loo_top.to_csv(path_loo, index=False)

    # Table: Pareto tail
    tail_rows = []
    for comp, col in [("total", "delta_total"), ("struct", "delta_struct"), ("within", "delta_within")]:
        x = np.abs(shapley_df["weight_interval"].to_numpy(dtype=float) * shapley_df[col].to_numpy(dtype=float))
        stats = hill_pareto_exponent(x, top_frac=top_frac)
        stats["component"] = comp
        tail_rows.append(stats)
    table_tail = pd.DataFrame(tail_rows)
    path_tail = out / "Table_Pareto_Tail.csv"
    table_tail.to_csv(path_tail, index=False)

    # Figures
    phase_df = build_phase_space(shapley_df, opt_df, weights_col="weight_interval")
    path_phase = fig_dir / "Fig_PhaseDiagram.png"
    _plot_phase_diagram(phase_df, path_phase)

    path_conc = fig_dir / "Fig_ConcentrationCurve.png"
    _plot_concentration_curve(shapley_df, path_conc)

    return {
        "table_mc_global": path_mc,
        "table_inequality": path_ineq,
        "table_loo_top20": path_loo,
        "table_pareto_tail": path_tail,
        "fig_phase_diagram": path_phase,
        "fig_concentration_curve": path_conc,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m methane_portfolio.figures_advanced",
        description="Advanced manuscript exports (tables + figures).",
    )
    p.add_argument(
        "--weight-method",
        choices=("base", "end", "avg", "sum", "trapz"),
        default=config.DEFAULT_WEIGHT_METHOD,
    )
    p.add_argument("--mc-n", type=int, default=1000, help="Number of MC bootstrap draws.")
    p.add_argument("--delta", type=float, default=0.10, help="TV distance budget for optimizer.")
    p.add_argument("--lam", type=float, default=0.5, help="Mean/CVaR tradeoff for optimizer.")
    p.add_argument("--alpha", type=float, default=0.90, help="CVaR confidence level.")
    p.add_argument("--top-frac", type=float, default=0.10, help="Top tail fraction for Hill estimator.")
    p.add_argument("--data-dir", type=Path, default=None, help="Optional data directory override.")
    p.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        config.FIGURE_DIR = args.output_dir / "figures"

    ensure_dirs()

    outputs = run_advanced_exports(
        weight_method=args.weight_method,
        mc_n=args.mc_n,
        delta=args.delta,
        lam=args.lam,
        alpha=args.alpha,
        top_frac=args.top_frac,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
