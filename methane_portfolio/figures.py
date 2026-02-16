# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Publication-ready figures (Nature-style, matplotlib only).

All figures are saved to outputs/figures/ using ``config.FIGURE_FORMAT``.
No seaborn is used. Colors are drawn from a curated palette.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from methane_portfolio import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated colour palette (colour-blind friendly, Nature-style)
# ---------------------------------------------------------------------------
PALETTE = {
    "struct":    "#1B7FA6",   # deep teal-blue
    "within":    "#C0392B",   # rich crimson
    "net":       "#2C3E50",   # dark slate
    "highlight": "#E67E22",   # warm amber
    "secondary": "#27AE60",   # emerald
    "tertiary":  "#8E44AD",   # amethyst
    "bg_light":  "#FFFFFF",   # white (Nature standard)
    "grid":      "#E8E8E8",   # very light grey
}

_SPECIES_COLORS: dict[str, str] = {}   # populated lazily


def _get_species_colors(species_list: list[str]) -> dict[str, str]:
    """Assign a distinct color per species from a curated set."""
    global _SPECIES_COLORS
    if not _SPECIES_COLORS:
        base_colors = [
            "#1B7FA6", "#C0392B", "#E67E22", "#27AE60", "#8E44AD",
            "#2C3E50", "#16A085", "#D35400", "#E74C3C", "#7F8C8D",
        ]
        for i, sp in enumerate(sorted(species_list)):
            _SPECIES_COLORS[sp] = base_colors[i % len(base_colors)]
    return _SPECIES_COLORS


# ---------------------------------------------------------------------------
# Nature-style formatting helpers
# ---------------------------------------------------------------------------

def _apply_nature_style() -> None:
    """Set global matplotlib rcParams for Nature house style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica Neue", "Helvetica"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": config.FIGURE_DPI,
        "pdf.fonttype": 42,   # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def _setup_ax(ax: mpl.axes.Axes, title: str = "") -> None:
    """Apply consistent Nature-like styling to an axes."""
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, color=PALETTE["grid"], linewidth=0.3,
            zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    """Add a bold panel label (a, b, câ€¦) at top-left."""
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def _pick_spread_labels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    score_col: str,
    *,
    max_labels: int = 10,
    min_dist: float = 0.10,
) -> pd.DataFrame:
    """Select high-priority points while keeping labels spatially separated."""
    if df.empty:
        return df.iloc[0:0]

    work = df[
        np.isfinite(df[x_col]) & np.isfinite(df[y_col]) & np.isfinite(df[score_col])
    ].copy()
    if work.empty:
        return work

    work.sort_values(score_col, ascending=False, inplace=True)

    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    x_span = max(x.max() - x.min(), 1e-12)
    y_span = max(y.max() - y.min(), 1e-12)

    selected_idx: list[object] = []
    selected_coords: list[tuple[float, float]] = []
    min_dist_sq = float(min_dist * min_dist)

    for idx, row in work.iterrows():
        xn = float((row[x_col] - x.min()) / x_span)
        yn = float((row[y_col] - y.min()) / y_span)

        if all((xn - sx) ** 2 + (yn - sy) ** 2 >= min_dist_sq for sx, sy in selected_coords):
            selected_idx.append(idx)
            selected_coords.append((xn, yn))
            if len(selected_idx) >= max_labels:
                break

    return work.loc[selected_idx]


def _save(fig: mpl.figure.Figure, name: str, output_dir: Path) -> None:
    """Save figure using configured output format (default PNG)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(config.FIGURE_FORMAT).strip().lower().lstrip(".") or "png"
    fig.savefig(
        output_dir / f"{name}.{fmt}",
        format=fmt,
        dpi=config.FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("Saved figure %s", name)


# ===================================================================
# Figure 1: Global Shapley decomposition bars
# ===================================================================

def fig1_global_shapley(
    shapley_global_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Horizontal bar chart of global structure vs within decomposition."""
    out = output_dir or config.FIGURE_DIR
    gpath = shapley_global_path or (config.OUTPUT_DIR / "shapley_global.json")
    with open(gpath, "r", encoding="utf-8") as f:
        g = json.load(f)

    # Convert to Ă—10â»Âł scale for readability
    scale = 1e3
    vals = [g["global_struct"] * scale, g["global_within"] * scale, g["net"] * scale]
    labels = [
        "Structure\n(species mix)",
        "Within\n(species intensity)",
        f"Net \u0394I",
    ]
    colors = [PALETTE["struct"], PALETTE["within"], PALETTE["net"]]
    hatches = ["//", "\\\\", ""]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    _setup_ax(ax)

    bars = ax.barh(labels, vals, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.55, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
        bar.set_edgecolor("white")

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(
        "\u0394 Emission intensity (\u00d710\u207b\u00b3 kg CO\u2082e / t milk)",
        fontsize=9,
        labelpad=6,
    )
    ax.margins(y=0.15)
    xmin = float(min(vals))
    xmax = float(max(vals))
    span = max(xmax - xmin, 1e-9)
    pad = max(0.35, span * 0.10)
    ax.set_xlim(xmin - pad, xmax + pad)

    # Value labels with safe placement
    for bar, v in zip(bars, vals):
        xoff = 0.15 if v >= 0 else -0.15
        ha = "left" if v >= 0 else "right"
        ax.text(v + xoff, bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}", ha=ha, va="center",
                fontsize=8, fontweight="bold")

    fig.text(
        0.99, 0.01,
        f"Accounting counterfactual ({g['base_year']}\u2192{g['end_year']}); no causal claim.",
        ha="right", va="bottom", fontsize=6, color="grey", style="italic",
    )

    fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.19)
    _panel_label(ax, "a")
    _save(fig, "Fig1_global_shapley", out)


# ===================================================================
# Figure 2: Country quadrant scatter
# ===================================================================

def fig2_country_quadrants(
    shapley_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Scatter of Î”struct vs Î”within, point size = production."""
    out = output_dir or config.FIGURE_DIR
    merged = shapley_df.merge(
        agg_df[agg_df["year"] == config.BASE_YEAR][
            ["country_m49", "milk_total_tonnes"]
        ],
        on="country_m49", how="left",
    )

    # Exclude Mongolia (massive outlier: Î”struct = â’15.19)
    mongolia = merged[merged["country"].str.contains("Mongolia", case=False, na=False)]
    merged_clean = merged[~merged["country"].str.contains("Mongolia", case=False, na=False)].copy()

    prod = merged_clean["milk_total_tonnes"].fillna(1e3)
    sizes = 10 + 250 * np.sqrt(prod / prod.max())

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    _setup_ax(ax)

    sc = ax.scatter(
        merged_clean["delta_struct"], merged_clean["delta_within"],
        s=sizes, alpha=0.6,
        c=merged_clean["delta_total"],
        cmap="RdYlGn_r", edgecolors="grey", linewidth=0.3,
        zorder=3,
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Net \u0394I (kg CO\u2082e / t milk)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Quadrant lines
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.margins(x=0.10, y=0.10)

    ax.text(
        0.01, 0.98,
        "Dashed lines mark zero effect",
        transform=ax.transAxes,
        fontsize=6,
        color="#7a7a7a",
        style="italic",
        ha="left",
        va="top",
    )

    ax.set_xlabel("\u0394 Structure (species mix shift)", fontsize=9)
    ax.set_ylabel("\u0394 Within (species intensity change)", fontsize=9)

    # Label only spread-out high-impact points to reduce overlaps.
    merged_clean = merged_clean.copy()
    merged_clean["label_priority"] = merged_clean["delta_total"].abs()
    label_pool = merged_clean.nlargest(25, "label_priority")
    label_rows = _pick_spread_labels(
        label_pool,
        "delta_struct",
        "delta_within",
        "label_priority",
        max_labels=9,
        min_dist=0.12,
    )
    for _, r in label_rows.iterrows():
        dx = 7 if r["delta_struct"] >= 0 else -7
        dy = 6 if r["delta_within"] >= 0 else -6
        ha = "left" if dx > 0 else "right"
        va = "bottom" if dy > 0 else "top"
        ax.annotate(
            r["country"][:20],
            (r["delta_struct"], r["delta_within"]),
            fontsize=6.5,
            alpha=0.95,
            fontweight="semibold",
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15),
        )

    # Mongolia annotation
    if not mongolia.empty:
        m = mongolia.iloc[0]
        ax.text(0.98, 0.10,
                f"Mongolia excluded\n(\u0394struct={m['delta_struct']:.1f}, "
                f"\u0394within={m['delta_within']:.1f})",
                transform=ax.transAxes, fontsize=6, color="grey",
                style="italic", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                          edgecolor="#cccccc", alpha=0.8))

    ax.text(0.02, 0.02, "Accounting counterfactual",
            transform=ax.transAxes, fontsize=6, color="grey", style="italic")

    _panel_label(ax, "b")
    _save(fig, "Fig2_country_quadrants", out)


# ===================================================================
# Figure 3: Regime shift posteriors
# ===================================================================

def fig3_regime_shift(
    idata=None,
    species_list: list[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Violin-like posterior distributions of Îł_s by species."""
    out = output_dir or config.FIGURE_DIR

    if idata is None:
        logger.warning("No idata supplied; skipping Fig3.")
        return

    gamma = idata.posterior["gamma_s"].values  # (chain, draw, n_species)
    gamma_flat = gamma.reshape(-1, gamma.shape[-1])

    if species_list is None:
        species_list = [f"Species {i}" for i in range(gamma_flat.shape[1])]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    _setup_ax(ax, "Posterior Regime-Shift Parameters (\u03b3\u209b, t \u2265 2022)")

    parts = ax.violinplot(
        [gamma_flat[:, i] for i in range(len(species_list))],
        positions=range(len(species_list)),
        showmeans=True, showmedians=True,
    )
    # Style violins
    colors = _get_species_colors(species_list)
    for i, pc in enumerate(parts["bodies"]):
        sp = sorted(species_list)[i] if i < len(species_list) else None
        pc.set_facecolor(colors.get(sp, PALETTE["struct"]))
        pc.set_alpha(0.55)
        pc.set_edgecolor("grey")
        pc.set_linewidth(0.5)

    # Style mean/median lines
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#2C3E50")
            parts[key].set_linewidth(0.8)

    ax.set_xticks(range(len(species_list)))
    ax.set_xticklabels(
        [s.replace("Raw milk of ", "") for s in sorted(species_list)],
        fontsize=8, rotation=30, ha="right",
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylabel("\u03b3\u209b (log-scale shift)", fontsize=9)

    # Add HDI annotations
    for i in range(len(species_list)):
        q05 = np.percentile(gamma_flat[:, i], 3)
        q97 = np.percentile(gamma_flat[:, i], 97)
        median = np.median(gamma_flat[:, i])
        ax.text(i, q97 + 0.05, f"{median:.2f}",
                ha="center", va="bottom", fontsize=6, fontweight="bold")

    _panel_label(ax, "c")
    _save(fig, "Fig3_regime_shift", out)


# ===================================================================
# Figure 4: Pareto risk frontier
# ===================================================================

def fig4_pareto_risk(
    sensitivity_df: pd.DataFrame,
    countries: list[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Mean reduction vs CVaR reduction across delta for selected countries."""
    out = output_dir or config.FIGURE_DIR

    if countries is None:
        top = (
            sensitivity_df
            .groupby("country")["reduction_mean_pct"]
            .mean()
            .nlargest(6)
            .index.tolist()
        )
        countries = top

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _setup_ax(ax, "Pareto Frontier: Mean vs Risk Reduction")

    palette_list = [PALETTE["struct"], PALETTE["within"], PALETTE["highlight"],
                    PALETTE["secondary"], PALETTE["tertiary"], PALETTE["net"]]
    markers = ["o", "s", "D", "^", "v", "P"]

    for i, c in enumerate(countries):
        csub = sensitivity_df[sensitivity_df["country"] == c]
        if csub.empty:
            continue
        ax.scatter(
            csub["reduction_mean_pct"], csub["reduction_cvar_pct"],
            label=c[:25], alpha=0.75, s=35,
            color=palette_list[i % len(palette_list)],
            marker=markers[i % len(markers)],
            edgecolors="grey", linewidth=0.3, zorder=3,
        )

    ax.set_xlabel("Mean Intensity Reduction (%)", fontsize=9)
    ax.set_ylabel("CVaR Intensity Reduction (%)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9,
              edgecolor="#cccccc")

    # Add diagonal reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="#cccccc", linewidth=0.5, alpha=0.5, zorder=1)

    ax.text(0.02, 0.02, "Accounting counterfactual",
            transform=ax.transAxes, fontsize=6, color="grey", style="italic")

    _panel_label(ax, "d")
    _save(fig, "Fig4_pareto_risk", out)


# ===================================================================
# Figure 5: Low-hanging fruit
# ===================================================================

def fig5_low_hanging_fruit(
    opt_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Scatter: x = reduction_mean_pct, y = baseline intensity, size = production."""
    out = output_dir or config.FIGURE_DIR

    merged = opt_df.merge(
        agg_df[agg_df["year"] == config.END_YEAR][
            ["country_m49", "milk_total_tonnes"]
        ],
        on="country_m49", how="left",
    )

    # Filter to countries with meaningful reduction (> 0.5%)
    merged_plot = merged[merged["reduction_mean_pct"] > 0.5].copy()

    prod = merged_plot["milk_total_tonnes"].fillna(1e3)
    sizes = 10 + 250 * np.sqrt(prod / prod.max())

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    _setup_ax(ax)

    sc = ax.scatter(
        merged_plot["reduction_mean_pct"],
        merged_plot["baseline_intensity_kg_co2e_per_t"],
        s=sizes, alpha=0.55,
        c=merged_plot["reduction_mean_pct"],
        cmap="viridis", edgecolors="grey", linewidth=0.3,
        zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Reduction potential (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_xlim(left=0.0)
    ax.margins(x=0.06, y=0.08)

    ax.set_xlabel("Mean intensity reduction potential (%)", fontsize=9)
    ax.set_ylabel("Baseline Intensity (kg CO\u2082e / t milk)", fontsize=9)

    # Annotate only spread-out high-potential countries to avoid clutter.
    merged_plot = merged_plot.copy()
    merged_plot["label_priority"] = (
        merged_plot["reduction_mean_pct"] * np.sqrt(prod / prod.max())
    )
    label_pool = merged_plot.nlargest(30, "label_priority")
    label_rows = _pick_spread_labels(
        label_pool,
        "reduction_mean_pct",
        "baseline_intensity_kg_co2e_per_t",
        "label_priority",
        max_labels=8,
        min_dist=0.13,
    )
    for _, r in label_rows.iterrows():
        name = r["country"]
        if len(name) > 15:
            name = name[:13] + "\u2026"
        dx = 7 if r["reduction_mean_pct"] >= merged_plot["reduction_mean_pct"].median() else -7
        dy = 5 if r["baseline_intensity_kg_co2e_per_t"] >= merged_plot["baseline_intensity_kg_co2e_per_t"].median() else -5
        ax.annotate(
            name,
            (r["reduction_mean_pct"], r["baseline_intensity_kg_co2e_per_t"]),
            fontsize=6.5,
            fontweight="semibold",
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15),
        )

    # Reference lines for global means
    mean_red = merged_plot["reduction_mean_pct"].mean()
    mean_base = merged_plot["baseline_intensity_kg_co2e_per_t"].mean()
    ax.axvline(mean_red, color=PALETTE["highlight"], linewidth=0.5,
               linestyle=":", alpha=0.5, zorder=1)
    ax.axhline(mean_base, color=PALETTE["highlight"], linewidth=0.5,
               linestyle=":", alpha=0.5, zorder=1)

    ax.text(0.02, 0.02, "Accounting counterfactual",
            transform=ax.transAxes, fontsize=6, color="grey", style="italic")

    _panel_label(ax, "e")
    _save(fig, "Fig5_low_hanging_fruit", out)


# ===================================================================
# Figure 6: Elasticity distribution
# ===================================================================

def fig6_elasticity(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    year: int = config.END_YEAR,
    n_countries: int = 25,
    output_dir: Path | None = None,
) -> None:
    """Species-level distribution of elasticity for top-producing countries."""
    out = output_dir or config.FIGURE_DIR
    sub = long_df[long_df["year"] == year]
    if sub.empty:
        logger.warning("No rows for year=%s; skipping Fig6.", year)
        return

    # Top countries by production
    top = (
        agg_df[agg_df["year"] == year]
        .nlargest(n_countries, "milk_total_tonnes")
    )
    if top.empty:
        logger.warning("No aggregate rows for year=%s; skipping Fig6.", year)
        return

    top_m49 = top["country_m49"].tolist()
    top_total = (
        top.set_index("country_m49")["kg_co2e_per_ton_milk"]
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    for m49 in top_m49:
        csub = sub[sub["country_m49"] == m49]
        total_intensity = top_total.get(m49)
        if not np.isfinite(total_intensity):
            continue
        for _, r in csub.iterrows():
            i_species = r["kg_co2e_per_ton_milk"]
            if not np.isfinite(i_species):
                continue
            rows.append(
                {
                    "milk_species": r["milk_species"],
                    "elasticity": float(i_species - total_intensity),
                },
            )

    elast_df = pd.DataFrame(rows)
    if elast_df.empty:
        logger.warning("No finite elasticity values; skipping Fig6.")
        return

    species_order = (
        elast_df.groupby("milk_species", observed=True)["elasticity"]
        .median()
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    species_short = [s.replace("Raw milk of ", "") for s in species_order]
    data = [
        elast_df[elast_df["milk_species"] == sp]["elasticity"].to_numpy(dtype=float)
        for sp in species_order
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    _setup_ax(ax)

    bp = ax.boxplot(
        data,
        vert=False,
        labels=species_short,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": "#1f1f1f", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
    )
    species_colors = _get_species_colors(species_order)
    for box, sp in zip(bp["boxes"], species_order):
        box.set_facecolor(species_colors.get(sp, PALETTE["struct"]))
        box.set_alpha(0.45)
        box.set_edgecolor("#555555")
        box.set_linewidth(0.8)

    rng = np.random.default_rng(config.RNG_SEED)
    for yi, sp in enumerate(species_order, start=1):
        vals = elast_df[elast_df["milk_species"] == sp]["elasticity"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = yi + rng.uniform(-0.10, 0.10, size=vals.size)
        ax.scatter(
            vals,
            jitter,
            s=10,
            color=species_colors.get(sp, PALETTE["struct"]),
            alpha=0.30,
            edgecolors="none",
            zorder=2,
        )

    lo = float(np.percentile(elast_df["elasticity"], 2))
    hi = float(np.percentile(elast_df["elasticity"], 98))
    if hi <= lo:
        lo = float(np.min(elast_df["elasticity"]))
        hi = float(np.max(elast_df["elasticity"]))
    pad = max(0.15, (hi - lo) * 0.08)
    ax.set_xlim(lo - pad, hi + pad)
    n_out = int(((elast_df["elasticity"] < (lo - pad)) | (elast_df["elasticity"] > (hi + pad))).sum())

    ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.8, zorder=1)
    ax.set_xlabel("Elasticity = species intensity - country intensity (kg CO\u2082e / t milk)", fontsize=9)
    ax.set_ylabel("Species", fontsize=9)
    if n_out > 0:
        ax.text(
            0.01, 0.03,
            f"{n_out} extreme values clipped from x-range",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6,
            color="grey",
            style="italic",
        )
    ax.text(
        0.99, 0.03,
        f"Top {len(top_m49)} producers, year {year}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="grey",
        style="italic",
    )

    _panel_label(ax, "f")
    _save(fig, "Fig6_elasticity", out)


# ===================================================================
# Convenience: generate all figures
# ===================================================================

def make_all_figures(
    shapley_df: pd.DataFrame | None = None,
    opt_df: pd.DataFrame | None = None,
    sensitivity_df: pd.DataFrame | None = None,
    idata=None,
    bayes_data: dict | None = None,
    long_df: pd.DataFrame | None = None,
    agg_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate all 6 figures."""
    out = output_dir or config.FIGURE_DIR

    # Apply Nature-style globally
    _apply_nature_style()

    fig1_global_shapley(output_dir=out)

    if shapley_df is not None and agg_df is not None:
        fig2_country_quadrants(shapley_df, agg_df, output_dir=out)

    if idata is not None and bayes_data is not None:
        fig3_regime_shift(idata, bayes_data.get("species_list"), output_dir=out)

    if sensitivity_df is not None:
        fig4_pareto_risk(sensitivity_df, output_dir=out)

    if opt_df is not None and agg_df is not None:
        fig5_low_hanging_fruit(opt_df, agg_df, output_dir=out)

    if long_df is not None and agg_df is not None:
        fig6_elasticity(long_df, agg_df, output_dir=out)
