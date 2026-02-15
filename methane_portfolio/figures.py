"""Publication-ready figures (Nature-style, matplotlib only).

All figures are saved as PNG (600 DPI) and PDF to outputs/figures/.
No seaborn is used.  Colors are drawn from a curated palette.
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
from methane_portfolio.utils import causal_disclaimer

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
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
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
    """Add a bold panel label (a, b, c…) at top-left."""
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def _save(fig: mpl.figure.Figure, name: str, output_dir: Path) -> None:
    """Save figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{name}.{fmt}",
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

    # Convert to ×10⁻³ scale for readability
    scale = 1e3
    vals = [g["global_struct"] * scale, g["global_within"] * scale, g["net"] * scale]
    labels = [
        "Structure\n(species mix)",
        "Within\n(species intensity)",
        f"Net \u0394I",
    ]
    colors = [PALETTE["struct"], PALETTE["within"], PALETTE["net"]]
    hatches = ["//", "\\\\", ""]

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    _setup_ax(ax, f"Shapley Decomposition of \u0394I ({g['base_year']}\u2192{g['end_year']})")

    bars = ax.barh(labels, vals, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.55, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
        bar.set_edgecolor("white")

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(
        "\u0394 Emission Intensity (\u00d710\u207b\u00b3 kg CO\u2082e / t milk, production-weighted)",
        fontsize=8,
    )

    # Value labels with safe placement
    for bar, v in zip(bars, vals):
        xoff = 0.15 if v >= 0 else -0.15
        ha = "left" if v >= 0 else "right"
        ax.text(v + xoff, bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}", ha=ha, va="center",
                fontsize=8, fontweight="bold")

    ax.text(0.02, 0.02,
            "Accounting counterfactual; does not imply causal effects.",
            transform=ax.transAxes, fontsize=6, color="grey", style="italic")

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
    """Scatter of Δstruct vs Δwithin, point size = production."""
    out = output_dir or config.FIGURE_DIR
    merged = shapley_df.merge(
        agg_df[agg_df["year"] == config.BASE_YEAR][
            ["country_m49", "milk_total_tonnes"]
        ],
        on="country_m49", how="left",
    )

    # Exclude Mongolia (massive outlier: Δstruct = −15.19)
    mongolia = merged[merged["country"].str.contains("Mongolia", case=False, na=False)]
    merged_clean = merged[~merged["country"].str.contains("Mongolia", case=False, na=False)].copy()

    prod = merged_clean["milk_total_tonnes"].fillna(1e3)
    sizes = 10 + 250 * np.sqrt(prod / prod.max())

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    _setup_ax(ax, "Country-Level Shapley Decomposition")

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

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    qprops = dict(fontsize=6, color="#888888", style="italic", alpha=0.7, ha="center")
    ax.text(xlim[1] * 0.6, ylim[1] * 0.85, "Worsening both", **qprops)
    ax.text(xlim[0] * 0.6, ylim[0] * 0.85, "Improving both", **qprops)
    ax.text(xlim[1] * 0.6, ylim[0] * 0.85, "Worsening structure\nImproving intensity", **qprops)
    ax.text(xlim[0] * 0.6, ylim[1] * 0.85, "Improving structure\nWorsening intensity", **qprops)

    ax.set_xlabel("\u0394 Structure (species mix shift)", fontsize=9)
    ax.set_ylabel("\u0394 Within (species intensity change)", fontsize=9)

    # Label outliers (top 5 and bottom 5 by |delta_total|)
    top = merged_clean.nlargest(5, "delta_total", keep="first")
    bot = merged_clean.nsmallest(5, "delta_total", keep="first")
    for _, r in pd.concat([top, bot]).iterrows():
        ax.annotate(
            r["country"][:20],
            (r["delta_struct"], r["delta_within"]),
            fontsize=6, alpha=0.85, fontweight="bold",
            textcoords="offset points", xytext=(6, 6),
            arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"),
        )

    # Mongolia annotation
    if not mongolia.empty:
        m = mongolia.iloc[0]
        ax.text(0.98, 0.98,
                f"Mongolia excluded\n(\u0394struct={m['delta_struct']:.1f}, "
                f"\u0394within={m['delta_within']:.1f})",
                transform=ax.transAxes, fontsize=6, color="grey",
                style="italic", ha="right", va="top",
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
    """Violin-like posterior distributions of γ_s by species."""
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
    """Scatter: x = reduction_mean_pct (log), y = baseline intensity, size = prod."""
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

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    _setup_ax(ax, "Low-Hanging Fruit: Reduction Potential vs Baseline Intensity")

    sc = ax.scatter(
        merged_plot["reduction_mean_pct"],
        merged_plot["baseline_intensity_kg_co2e_per_t"],
        s=sizes, alpha=0.55,
        c=merged_plot["reduction_mean_pct"],
        cmap="viridis", edgecolors="grey", linewidth=0.3,
        zorder=3,
    )

    # Log scale for x-axis to spread clustered data
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%" if x >= 1 else f"{x:.1f}%"))

    ax.set_xlabel("Mean Intensity Reduction (%, log scale)", fontsize=9)
    ax.set_ylabel("Baseline Intensity (kg CO\u2082e / t milk)", fontsize=9)

    # Annotate top-15 countries by reduction potential
    top15 = merged_plot.nlargest(15, "reduction_mean_pct")
    for _, r in top15.iterrows():
        name = r["country"]
        if len(name) > 18:
            name = name[:16] + "\u2026"
        ax.annotate(
            name,
            (r["reduction_mean_pct"], r["baseline_intensity_kg_co2e_per_t"]),
            fontsize=6, fontweight="bold",
            textcoords="offset points", xytext=(8, 4),
            arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"),
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
# Figure 6: Elasticity heatmap
# ===================================================================

def fig6_elasticity(
    long_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    year: int = config.END_YEAR,
    n_countries: int = 25,
    output_dir: Path | None = None,
) -> None:
    """Heatmap of elasticity (dI/dw_s) for top-producing countries."""
    out = output_dir or config.FIGURE_DIR
    sub = long_df[long_df["year"] == year]
    species = sorted(sub["milk_species"].unique())

    # Top countries by production
    top = (
        agg_df[agg_df["year"] == year]
        .nlargest(n_countries, "milk_total_tonnes")
    )
    top_m49 = top["country_m49"].tolist()
    top_names = top.set_index("country_m49")["country"]

    # Build elasticity matrix
    elast = np.full((len(top_m49), len(species)), np.nan)
    for ci, m49 in enumerate(top_m49):
        csub = sub[sub["country_m49"] == m49]
        agg_row = agg_df[
            (agg_df["country_m49"] == m49) & (agg_df["year"] == year)
        ]
        if agg_row.empty:
            continue
        I_total = agg_row.iloc[0]["kg_co2e_per_ton_milk"]
        for _, r in csub.iterrows():
            si = species.index(r["milk_species"])
            elast[ci, si] = r["kg_co2e_per_ton_milk"] - I_total

    # Symmetric colormap limits
    vmax = np.nanmax(np.abs(elast)) * 0.9
    vmin = -vmax

    # Taller figure to fit full country names
    fig_height = max(6, n_countries * 0.38)
    fig, ax = plt.subplots(figsize=(7, fig_height), constrained_layout=True)
    _setup_ax(ax, f"Elasticity dI/dw (Top {n_countries} Producers, {year})")

    im = ax.pcolormesh(
        np.arange(len(species) + 1), np.arange(len(top_m49) + 1),
        elast, cmap="coolwarm", vmin=vmin, vmax=vmax,
        edgecolors="white", linewidth=0.5,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Elasticity (kg CO\u2082e / t milk)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Cell value annotations for significant values
    for ci in range(len(top_m49)):
        for si in range(len(species)):
            val = elast[ci, si]
            if np.isnan(val):
                continue
            if abs(val) > vmax * 0.15:  # only annotate significant
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(si + 0.5, ci + 0.5, f"{val:.3f}",
                        ha="center", va="center", fontsize=5.5,
                        color=color, fontweight="bold")

    # Full species and country names
    short_species = [s.replace("Raw milk of ", "") for s in species]
    ax.set_xticks(np.arange(len(species)) + 0.5)
    ax.set_xticklabels(short_species, fontsize=7, rotation=45, ha="right")

    ylabels = [str(top_names.get(m, m)) for m in top_m49]  # Full names
    ax.set_yticks(np.arange(len(top_m49)) + 0.5)
    ax.set_yticklabels(ylabels, fontsize=7)

    # Invert y-axis so top producer is at top
    ax.invert_yaxis()

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
