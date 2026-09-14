"""
Publication figures for:
"Structural Drivers of Global Methane Intensity in Dairy Systems:
 Multi-Species Bayesian Uncertainty and Robust Portfolio Mitigation"

Regenerates Figures 1-6 from the deposited analytical outputs (see ../evidence).

    python figures_editorial.py --evidence ../evidence --out .

Each figure is a composite: aligned panels, shared axes and a single legend, so
that every figure carries one complete argument rather than one isolated chart.
PNG at 600 dpi is produced for the manuscript; SVG and PDF masters are written
alongside it.

Design rules applied
--------------------
* Categorical species palette is fixed (never cycled) and was validated for
  colour-vision deficiency across all pairwise comparisons
  (worst pair deltaE 9.0 deutan, 15.6 normal vision):
      buffalo #D55E00 | camel #6a3d9a | cattle #1f78b4 | goats #009E73 | sheep #E69F00
* Signed quantities use a two-hue diverging scale with a neutral grey midpoint;
  magnitudes use a single-hue sequential ramp. Never a rainbow.
* Bars start at zero. Logarithmic and symmetric-log axes are stated in the caption.
* Species identity is carried by a label or legend as well as by colour.
* No gradients, shadows, rounded cards or other infographic styling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

# ---------------------------------------------------------------- design tokens
DPI = 600
FORMATS = ("png", "svg", "pdf")

SPECIES_COLOR = {
    "buffalo": "#D55E00",
    "camel": "#6a3d9a",
    "cattle": "#1f78b4",
    "goats": "#009E73",
    "sheep": "#E69F00",
}
SPECIES_ORDER = ["buffalo", "camel", "cattle", "goats", "sheep"]

NEG, POS, MID = "#1f78b4", "#D55E00", "#f1efec"
TOTAL = "#2f3b45"

INK, INK_SOFT, INK_FAINT = "#2e2a26", "#6b655d", "#9b948a"
GRID, BAND, SURFACE = "#e8e3db", "#f6f3ee", "#ffffff"
MARK_EDGE = "#b3aca2"

DIVERGING = LinearSegmentedColormap.from_list("d_div", [NEG, MID, POS])
SEQUENTIAL = LinearSegmentedColormap.from_list("d_seq", ["#adcbe0", "#1f78b4", "#123f5c"])


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c0b9ae",
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def spines(ax, keep=("left", "bottom")) -> None:
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


def panel_tag(ax, tag: str, dx: float = -0.02, dy: float = 1.02) -> None:
    ax.text(dx, dy, f"({tag})", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=INK, ha="left", va="bottom")


def row_bands(ax, positions) -> None:
    for i, y in enumerate(sorted(positions)):
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg / .pdf")


def symlog_fwd(x, lin):
    return np.sign(x) * np.log10(1.0 + np.abs(x) / lin)


CHINA_AGGREGATE_M49 = 159   # FAO reports both "China" (aggregate) and "China, mainland";
                            # the pipeline keeps mainland only, so the figures must too


def dedup_china(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["country_m49"] != CHINA_AGGREGATE_M49].copy()


def short_name(name: str) -> str:
    return {"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "Iran (Islamic Republic of)": "Iran",
            "Netherlands (Kingdom of the)": "Netherlands",
            "Russian Federation": "Russia",
            "United Republic of Tanzania": "Tanzania",
            "China, mainland": "China",
            "United States of America": "United States",
            "Venezuela (Bolivarian Republic of)": "Venezuela",
            "Bolivia (Plurinational State of)": "Bolivia"}.get(name, name)


def species_key(raw: str) -> str:
    s = str(raw).replace("Raw milk of ", "").strip().lower()
    return {"camels": "camel", "buffaloes": "buffalo", "goat": "goats"}.get(s, s)


# ============================================================== Figure 1
def figure1(ev: Path, out: Path) -> None:
    """(a) global decomposition waterfall; (b) why the species mix matters."""
    g = json.loads((ev / "02_processed_results" / "shapley_global.json").read_text())
    struct, within, net = (g["global_struct"], g["global_within"],
                           g["global_total"])

    # per-species distribution of country intensities, 2023 (median + IQR: robust to
    # the single extreme sheep observation that destabilises the arithmetic mean)
    raw = ev / "01_raw_data"
    sp_raw = dedup_china(pd.read_csv(raw / "cercetare-485010.faostat_clean.milk_emission_intensity_2020_2023.csv"))
    sp_raw = sp_raw[sp_raw["year"] == 2023].copy()
    sp_raw["species"] = sp_raw["milk_species"].map(species_key)
    sp_raw = sp_raw[sp_raw["species"].isin(SPECIES_ORDER) & (sp_raw["milk_tonnes"] > 0)]

    stat = (sp_raw.groupby("species")
            .agg(q1=("kg_co2e_per_ton_milk", lambda v: np.nanpercentile(v, 25)),
                 med=("kg_co2e_per_ton_milk", "median"),
                 q3=("kg_co2e_per_ton_milk", lambda v: np.nanpercentile(v, 75)),
                 milk=("milk_tonnes", "sum"),
                 n=("kg_co2e_per_ton_milk", "size"))
            .reindex(SPECIES_ORDER).dropna(subset=["med"]))
    stat["share"] = stat["milk"] / stat["milk"].sum()

    # exact three-factor decomposition of the WORLD aggregate ratio (panel c).
    # Values are already in g CH4 per kg, so they need no unit conversion; look in the
    # figure-input tree first, then fall back to the evidence root.
    cand = [ev / "06_boundary_sensitivity" / "panelE_global_estimand.csv",
            ev.parent.parent / "06_boundary_sensitivity" / "panelE_global_estimand.csv"]
    panel_e = next((p for p in cand if p.exists()), None)
    if panel_e is None:
        raise FileNotFoundError(
            "panelE_global_estimand.csv not found; run "
            "evidence/06_boundary_sensitivity/boundary_sensitivity.py first")
    glob = pd.read_csv(panel_e).set_index("quantity")["value_g_ch4_per_kg"]
    geo = float(glob["shapley3_geography"])
    mix3 = float(glob["shapley3_within_country_species_mix"])
    wit3 = float(glob["shapley3_country_species_intensity"])
    world = float(glob["world_ratio_change"])

    fig = plt.figure(figsize=(7.2, 3.5))
    gs = GridSpec(1, 3, width_ratios=[0.80, 1.16, 0.90], wspace=0.58, figure=fig)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # ---- (a) waterfall
    x = np.arange(3)
    starts, values = [0.0, struct, 0.0], [struct, within, net]
    for xi, s, v, c in zip(x, starts, values, [NEG, POS, TOTAL]):
        ax1.bar(xi, v, bottom=s, width=0.46, color=c, linewidth=0, zorder=3)
    ax1.plot([0.23, 0.77], [struct, struct], color=INK_FAINT, lw=0.9, ls=(0, (3, 3)), zorder=2)
    ax1.plot([1.23, 1.77], [net, net], color=INK_FAINT, lw=0.9, ls=(0, (3, 3)), zorder=2)
    for xi, s, v in zip(x, starts, values):
        end = s + v
        ax1.text(xi, end + (-0.13 if v < 0 else 0.13), f"{v:+.2f}", ha="center",
                 va="top" if v < 0 else "bottom", fontsize=10, color=INK, zorder=4)
    ax1.axhline(0, color="#c0b9ae", lw=0.8, zorder=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Species\nmix", "Within\nspecies", "Net"],
                        fontsize=7.6, color=INK)
    ax1.set_ylabel("$\\Delta$ intensity, 2020$\\rightarrow$2023\n"
                   "(g CH$_4$ kg$^{-1}$ raw milk)")
    lo = min(struct, net, 0.0)
    ax1.set_ylim(lo * 1.26, abs(lo) * 0.22)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    spines(ax1, keep=("left",))
    ax1.tick_params(axis="x", length=0)
    ax1.set_title("Fixed country weights:\nmean of national changes",
                  fontsize=7.4, color=INK_FAINT, pad=6)
    panel_tag(ax1, "a", dx=-0.42)

    # ---- (c) exact three-factor decomposition of the world aggregate ratio
    xs = np.arange(4)
    vals3 = [geo, mix3, wit3, world]
    cols3 = ["#8d8578", NEG, POS, TOTAL]
    ax3.bar(xs, vals3, width=0.56, color=cols3, linewidth=0, zorder=3)
    for xi, v in zip(xs, vals3):
        ax3.text(xi, v + (-0.13 if v < 0 else 0.13), f"{v:+.2f}", ha="center",
                 va="top" if v < 0 else "bottom", fontsize=8.2, color=INK, zorder=4)
    ax3.axhline(0, color="#c0b9ae", lw=0.8, zorder=1)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["Geo-\ngraphy", "Species\nmix", "Within\nspecies",
                         "World\nratio"], fontsize=6.8, color=INK)
    lo3 = min(vals3 + [0.0])
    hi3 = max(vals3 + [0.0])
    ax3.set_ylim(lo3 * 1.30, hi3 * 1.55)
    ax3.yaxis.grid(True, zorder=0)
    ax3.set_axisbelow(True)
    spines(ax3, keep=("left",))
    ax3.tick_params(axis="x", length=0)
    ax3.set_title("World aggregate ratio:\nexact three-factor split",
                  fontsize=7.4, color=INK_FAINT, pad=6)
    panel_tag(ax3, "c", dx=-0.38)

    # ---- (b) intensity distribution per species against the milk it carries
    y = np.arange(len(stat))[::-1]
    sizes = 26 + 300 * stat["share"].to_numpy() ** 0.5
    x_floor = stat["q1"].min() * 0.55

    for yi, (spn, r), sz in zip(y, stat.iterrows(), sizes):
        c = SPECIES_COLOR[spn]
        ax2.plot([r["q1"], r["q3"]], [yi, yi], color=c, lw=2.0, alpha=0.55,
                 solid_capstyle="round", zorder=2)
        ax2.scatter([r["med"]], [yi], s=sz, color=c, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3)
        ax2.text(r["q3"] * 1.45, yi, f"{r['med']:,.1f}", va="center", ha="left",
                 fontsize=8.6, color=INK, zorder=4)
        ax2.text(x_floor, yi + 0.31, f"{r['share'] * 100:.1f}% of world milk  ($n$ = {int(r['n'])})",
                 va="bottom", ha="left", fontsize=7.2, color=INK_FAINT, zorder=4)

    ax2.set_xscale("log")
    ax2.set_yticks(y)
    ax2.set_yticklabels([s.capitalize() for s in stat.index], fontsize=9, color=INK)
    ax2.set_xlim(x_floor * 0.85, stat["q3"].max() * 5.0)
    ax2.set_ylim(-0.72, len(stat) - 0.28)
    ax2.set_xlabel("Country methane intensity, 2023\n"
                   "median and IQR (g CH$_4$ kg$^{-1}$ raw milk, log scale)")
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    spines(ax2, keep=("bottom",))
    ax2.tick_params(axis="y", length=0)
    panel_tag(ax2, "b", dx=-0.24)

    save(fig, out, "Figure 1")


# ============================================================== Figure 2
def figure2(ev: Path, out: Path) -> None:
    """Quadrant map of the country decomposition, with a linear zoom on the core."""
    df = pd.read_csv(ev / "02_processed_results" / "shapley_country.csv").dropna(
        subset=["delta_struct", "delta_within", "delta_total"])
    extreme = df.loc[df["delta_struct"].abs() > 1000.0]
    core = df.drop(extreme.index).reset_index(drop=True)

    lin = 2.0
    fx = symlog_fwd(core["delta_struct"].to_numpy(), lin)
    fy = symlog_fwd(core["delta_within"].to_numpy(), lin)
    prod = core["weight_interval"].to_numpy()
    size = 13 + 230 * (prod / prod.max()) ** 0.42
    vmax = np.nanpercentile(np.abs(core["delta_total"]), 97)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(1, 2, width_ratios=[1, 0.030], wspace=0.04, figure=fig)
    ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    lim = symlog_fwd(np.array([350.0]), lin)[0]
    ax.axhline(0, color="#c0b9ae", lw=0.8, zorder=2)
    ax.axvline(0, color="#c0b9ae", lw=0.8, zorder=2)
    sc = ax.scatter(fx, fy, s=size, c=core["delta_total"], cmap=DIVERGING,
                    vmin=-vmax, vmax=vmax, alpha=0.92, edgecolor=MARK_EDGE,
                    linewidth=0.45, zorder=3)

    q = dict(fontsize=8, color=INK_FAINT, style="italic", zorder=4)
    ax.text(-lim * .96, lim * .95, "mix improved,\nintensity worsened", ha="left", va="top", **q)
    ax.text(lim * .96, lim * .95, "both worsened", ha="right", va="top", **q)
    ax.text(-lim * .96, -lim * .95, "both improved", ha="left", va="bottom", **q)
    ax.text(lim * .96, -lim * .95, "mix worsened,\nintensity improved", ha="right", va="bottom", **q)

    tag = core.reindex(core["delta_total"].abs().sort_values(ascending=False).index).head(5)
    offs = [(-0.34, 0.10), (0.30, 0.12), (-0.34, -0.12), (0.30, -0.14), (-0.30, 0.26)]
    for (dx, dy), (_, r) in zip(offs, tag.iterrows()):
        px = symlog_fwd(np.array([r["delta_struct"]]), lin)[0]
        py = symlog_fwd(np.array([r["delta_within"]]), lin)[0]
        ax.annotate(r["country"], xy=(px, py), xytext=(px + dx, py + dy),
                    ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top",
                    fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_FAINT, lw=0.55,
                                    shrinkA=0, shrinkB=3), zorder=5)

    ticks = np.array([-200.0, -50.0, -10.0, 0.0, 10.0, 50.0, 200.0])
    tpos = symlog_fwd(ticks, lin)
    ax.set_xticks(tpos); ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks(tpos); ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("$\\Delta$ Structure — species-mix shift (g CH$_4$ kg$^{-1}$)")
    ax.set_ylabel("$\\Delta$ Within — species-intensity change (g CH$_4$ kg$^{-1}$)")
    spines(ax, keep=())
    ax.tick_params(length=0)

    # ---- linear zoom on the crowded core
    zoom = 4.0
    m = ((core["delta_struct"].abs() <= zoom) & (core["delta_within"].abs() <= zoom)).to_numpy()
    axz = ax.inset_axes([0.045, 0.115, 0.27, 0.27])
    axz.axhline(0, color="#c0b9ae", lw=0.6)
    axz.axvline(0, color="#c0b9ae", lw=0.6)
    axz.scatter(core.loc[m, "delta_struct"], core.loc[m, "delta_within"],
                s=size[m] * 0.5, c=core.loc[m, "delta_total"], cmap=DIVERGING,
                vmin=-vmax, vmax=vmax, alpha=0.92, edgecolor=MARK_EDGE, linewidth=0.35)
    axz.set_xlim(-zoom, zoom); axz.set_ylim(-zoom, zoom)
    axz.set_xticks([-zoom, 0, zoom]); axz.set_yticks([-zoom, 0, zoom])
    axz.set_xticklabels(["-4", "0", "4"], fontsize=6.4)
    axz.set_yticklabels(["-4", "0", "4"], fontsize=6.4)
    axz.tick_params(length=1.6, pad=1.5)
    for sp_ in axz.spines.values():
        sp_.set(visible=True, color="#c0b9ae", linewidth=0.6)
    axz.set_facecolor("#fbfaf8")
    axz.set_title(f"linear zoom · $n$ = {int(m.sum())}", fontsize=7, color=INK_SOFT, pad=3)

    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Net $\\Delta$ intensity (g CH$_4$ kg$^{-1}$)", fontsize=8.5, color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0, colors=INK_SOFT)

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#d8d4ce",
                      markeredgecolor=MARK_EDGE, label=lab,
                      markersize=np.sqrt(13 + 230 * f ** 0.42) * 0.6)
               for f, lab in ((0.02, "small"), (0.25, "medium"), (1.0, "largest"))]
    ax.legend(handles=handles, title="Milk production", title_fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
              handletextpad=1.0, columnspacing=2.2, borderpad=0.3)

    save(fig, out, "Figure 2")


# ============================================================== Figure 3
def figure3(ev: Path, out: Path) -> None:
    """Model adequacy: (a) posterior predictive fit; (b) how predictable each species is."""
    ppc = pd.read_csv(ev / "03_diagnostics" / "bayes_ppc_summary.csv")
    outl = pd.read_csv(ev / "03_diagnostics" / "bayes_ppc_outliers.csv")
    tab = pd.read_csv(ev / "02_processed_results" / "Table3_bayes_summary.csv")

    sig = tab[tab["parameter"].str.startswith("sigma_s")].reset_index(drop=True)
    sig["species"] = SPECIES_ORDER[: len(sig)]

    inside = ppc["within_90ci"].astype(str).str.lower().isin(["true", "1"])
    cover = inside.mean() * 100

    fig = plt.figure(figsize=(7.3, 3.5))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.12], wspace=0.34, figure=fig)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- (a) observed against posterior predictive mean
    lo = float(min(ppc["y_obs"].min(), ppc["y_rep_mean"].min())) - 0.4
    hi = float(max(ppc["y_obs"].max(), ppc["y_rep_mean"].max())) + 0.4
    ax1.plot([lo, hi], [lo, hi], color=INK_FAINT, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax1.scatter(ppc.loc[inside, "y_rep_mean"], ppc.loc[inside, "y_obs"], s=9,
                color=NEG, alpha=0.30, linewidth=0, zorder=3,
                label=f"within 90% interval ($n$ = {int(inside.sum())})")
    ax1.scatter(ppc.loc[~inside, "y_rep_mean"], ppc.loc[~inside, "y_obs"], s=13,
                color=POS, alpha=0.75, edgecolor=SURFACE, linewidth=0.3, zorder=4,
                label=f"outside ($n$ = {int((~inside).sum())})")

    worst = outl.iloc[outl["abs_residual"].argmax()]
    ax1.annotate(f"{worst['country']}, {species_key(worst['milk_species'])}",
                 xy=(worst["y_rep_mean"], worst["y_obs"]), xytext=(26, 16),
                 textcoords="offset points", fontsize=7.8, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK_FAINT, lw=0.55,
                                 shrinkA=0, shrinkB=3), zorder=5)
    ax1.text(0.03, 0.965, f"90% interval coverage: {cover:.1f}%", transform=ax1.transAxes,
             fontsize=8.2, color=INK_SOFT, style="italic", va="top", ha="left")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Posterior predictive mean, log intensity")
    ax1.set_ylabel("Observed log intensity")
    ax1.grid(True, zorder=0); ax1.set_axisbelow(True)
    spines(ax1, keep=("left", "bottom"))
    ax1.legend(loc="lower right", fontsize=7.8, handletextpad=0.5, borderpad=0.3)
    panel_tag(ax1, "a", dx=-0.20)

    # ---- (b) quantile-quantile check: why a Student-t likelihood was needed
    from scipy import stats

    res = ppc["residual"].to_numpy()
    res = np.sort(res[np.isfinite(res)])
    n = res.size
    nu = float(tab.loc[tab["parameter"] == "nu", "mean"].iloc[0])
    probs = (np.arange(1, n + 1) - 0.5) / n
    scale = np.median(np.abs(res - np.median(res))) / 0.6745   # robust scale

    q_t = stats.t.ppf(probs, df=nu, loc=0, scale=scale)
    q_n = stats.norm.ppf(probs, loc=0, scale=scale)

    span = float(np.nanmax(np.abs(res))) * 1.08
    ax2.plot([-span, span], [-span, span], color=INK_FAINT, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax2.scatter(q_n, res, s=8, color=POS, alpha=0.55, linewidth=0, zorder=3,
                label="Gaussian quantiles")
    ax2.scatter(q_t, res, s=8, color=NEG, alpha=0.55, linewidth=0, zorder=4,
                label=f"Student-$t$ quantiles, $\\nu$ = {nu:.2f}")

    ax2.set_xlim(-span, span)
    ax2.set_ylim(-span, span)
    ax2.set_xlabel("Theoretical quantile")
    ax2.set_ylabel("Observed residual quantile")
    ax2.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    spines(ax2, keep=("left", "bottom"))
    ax2.legend(loc="upper left", fontsize=7.8, handletextpad=0.5, borderpad=0.3,
               scatterpoints=1, markerscale=2.0)
    ax2.text(0.97, 0.05, "the Gaussian reference bends away\nin both tails; the Student-$t$ does not",
             transform=ax2.transAxes, fontsize=7.4, style="italic", color=INK_SOFT,
             ha="right", va="bottom", zorder=5)
    panel_tag(ax2, "b", dx=-0.18)

    save(fig, out, "Figure 3")


# ============================================================== Figure 4
def _budget_ramp(deltas):
    """Graded single-hue ramp: the budget is itself the ordered variable."""
    stops = np.linspace(0.04, 1.0, len(deltas))
    fills = [SEQUENTIAL(v) for v in stops]
    edges = [MARK_EDGE if v < 0.5 else SURFACE for v in stops]
    sizes = np.linspace(21, 47, len(deltas))
    return fills, edges, sizes


def _budget_strip(ax, labels, fills, edges, y=1.030, x0=0.47, step=0.132) -> None:
    """A compact labelled strip above the panel, in place of a boxed legend."""
    ax.text(0.0, y, "Reallocation budget  $\delta$", transform=ax.transAxes,
            fontsize=8.2, color=INK_SOFT, ha="left", va="center", clip_on=False)
    x = x0
    for lab, fc in zip(labels, fills):
        ax.add_patch(mpatches.Rectangle((x, y - 0.010), 0.019, 0.020, facecolor=fc,
                                        edgecolor=MARK_EDGE, linewidth=0.5,
                                        transform=ax.transAxes, clip_on=False, zorder=6))
        ax.text(x + 0.026, y, lab, transform=ax.transAxes, fontsize=8.2,
                color=INK, ha="left", va="center", clip_on=False)
        x += step


def figure4(ev: Path, out: Path) -> None:
    """How far a larger reallocation budget actually gets you: by country, then overall."""
    grid = pd.read_csv(ev / "02_processed_results" / "sensitivity_grid.csv")
    d = grid[(grid["lambda"] == 0.5) & (grid["alpha"] == 0.9)].copy()
    deltas = np.array(sorted(d["delta"].unique()))

    wide = (d.pivot_table(index="country", columns="delta", values="reduction_mean_pct")
            .reindex(columns=deltas))
    fills, edges, sizes = _budget_ramp(deltas)

    # ---- (a) plots only the systems that respond; the rest are named in a note
    # panel (a) is restricted to the 20 largest milk producers of 2023; panel (b)
    # pools every analysed country.
    _agg = dedup_china(pd.read_csv(ev / "01_raw_data" /
        "cercetare-485010.faostat_clean.milk_intensity_country_year.csv"))
    _top = set(_agg[_agg["year"] == 2023]
               .nlargest(20, "milk_total_tonnes")["country"].astype(str))
    w20 = wide.loc[[c for c in wide.index if str(c) in _top]]
    resp = w20[w20[deltas[-1]] > 0.0].sort_values(deltas[-1], ascending=True)
    flat = sorted(short_name(c) for c in w20.index[w20[deltas[-1]] <= 0.0])
    # each bar is the total reduction at the largest budget, split by what each
    # successive increment of the budget actually contributed
    incr = np.column_stack([resp[deltas[0]].to_numpy(dtype=float)] +
                           [(resp[deltas[i]] - resp[deltas[i - 1]]).to_numpy(dtype=float)
                            for i in range(1, len(deltas))])
    totals = resp[deltas[-1]].to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.3, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1.8, 1.0], wspace=0.34, figure=fig)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    y = np.arange(len(resp))
    axa.xaxis.grid(True, color=GRID, lw=0.55, zorder=1)
    left = np.zeros(len(resp))
    for j, (fc, ec) in enumerate(zip(fills, edges)):
        seg = incr[:, j]
        axa.barh(y, seg, left=left, height=0.50, color=fc, edgecolor=SURFACE,
                 linewidth=0.7, zorder=3)
        for yi, v, l in zip(y, seg, left):
            if v >= 6.0:
                axa.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                         fontsize=7.0, color=SURFACE if j >= 1 else INK, zorder=5)
        left += seg
    for yi, tv in zip(y, totals):
        axa.annotate(f"{tv:.1f}", xy=(tv, yi), xytext=(5, 0), textcoords="offset points",
                     va="center", fontsize=7.6, color=INK, zorder=6)

    axa.set_yticks(y)
    axa.set_yticklabels([short_name(c) for c in resp.index], fontsize=8.3, color=INK)
    axa.set_ylim(-0.72, len(resp) - 0.28)
    axa.set_xlim(0, float(totals.max()) * 1.20)
    axa.set_xlabel("Mean intensity reduction (%)")
    spines(axa, keep=("bottom",))
    axa.tick_params(axis="y", length=0)
    panel_tag(axa, "a", dx=-0.265, dy=1.075)
    _budget_strip(axa, [f"{deltas[0]:g}"] + [f"+{dv:g}" for dv in deltas[1:]],
                  fills, edges)

    rows = [", ".join(flat[i:i + 4]) for i in range(0, len(flat), 4)]
    rows = [r + "," for r in rows[:-1]] + rows[-1:]
    note = "No positive gain at any budget under the constraint set:\n" + "\n".join(rows)
    axa.text(0.30 * float(totals.max()), 1.35, note, fontsize=7.4, style="italic",
             color=INK_SOFT, ha="left", va="center", linespacing=1.55, zorder=6)

    # ---- (b) all twenty countries pooled, budget by budget
    xpos = np.arange(len(deltas), dtype=float)
    vals = [wide[dv].to_numpy(dtype=float) for dv in deltas]
    rng = np.random.default_rng(11)

    axb.set_yscale("symlog", linthresh=1.0, linscale=0.55)
    axb.grid(True, axis="y", color=GRID, lw=0.55, zorder=1)

    # A box degenerates here: half the countries are exact zeros, so Q1 = 0 at every
    # budget and any interquartile mark would run to the axis floor. The twenty
    # observations carry the spread directly; a median tick summarises them.
    for xi, v, fc, ec in zip(xpos, vals, fills, edges):
        axb.plot([xi - 0.38, xi - 0.06], [np.median(v)] * 2, color=SEQUENTIAL(0.95),
                 lw=1.6, solid_capstyle="butt", zorder=6)
        jx = xi + 0.12 + rng.uniform(-0.14, 0.14, size=len(v))
        axb.scatter(jx, v, s=12, facecolor=fc, edgecolor=ec, linewidth=0.45,
                    alpha=0.9, zorder=5)
    axb.annotate("median", xy=(xpos[1] - 0.38, np.median(vals[1])), xytext=(-4, 1),
                 textcoords="offset points", ha="right", va="center",
                 fontsize=7.0, color=INK_FAINT, zorder=8)

    means = np.array([v.mean() for v in vals])
    axb.plot(xpos, means, color=TOTAL, lw=1.1, zorder=6)
    axb.scatter(xpos, means, marker="D", s=25, facecolor=SURFACE,
                edgecolor=TOTAL, linewidth=1.15, zorder=7)
    for xi, mv in zip(xpos, means):
        axb.annotate(f"{mv:.1f}", xy=(xi, mv), xytext=(0, 13),
                     textcoords="offset points", ha="center", fontsize=7.6,
                     color=TOTAL, zorder=9,
                     path_effects=[mpe.withStroke(linewidth=2.6, foreground=SURFACE)])
    axb.annotate("mean", xy=(xpos[-1], means[-1]), xytext=(11, 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=7.3, color=TOTAL, zorder=8)

    ticks = [0, 1, 2, 5, 10, 20, 50, 100]
    axb.yaxis.set_major_locator(FixedLocator(ticks))
    axb.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    axb.yaxis.set_minor_locator(NullLocator())
    axb.set_xticks(xpos)
    axb.set_xticklabels([f"{dv:g}" for dv in deltas])
    axb.set_xlim(-0.62, len(deltas) - 0.38)
    axb.set_ylim(0, 130)
    axb.set_xlabel("Reallocation budget  $\delta$")
    axb.set_ylabel("Mean intensity reduction (%)")
    spines(axb, keep=("left", "bottom"))
    panel_tag(axb, "b", dx=-0.30, dy=1.075)
    axb.text(0.52, 0.905, "median saturates by $\delta$ = 0.05; mean does not",
             transform=axb.transAxes, fontsize=7.7, style="italic",
             color=INK_SOFT, ha="center", va="top", zorder=8)

    save(fig, out, "Figure 4")


# ============================================================== Figure 5
def _col(df, *names: str) -> str:
    """First of `names` present in `df`; raises listing the columns if none is."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)}")


def figure5(ev: Path, out: Path) -> None:
    """Where the mitigation potential sits, and how concentrated it is."""
    df = pd.read_csv(ev / "02_processed_results" / "robust_optimization_results.csv")
    abs_col = _col(df, "abs_reduction_mt_ch4", "absolute_reduction_kt")
    base_col = _col(df, "baseline_intensity_kg_co2e_per_t", "baseline_intensity")
    df = df[(df["production_tonnes"] > 0) & (df[base_col] > 0)].copy()

    fig = plt.figure(figsize=(7.3, 6.0))
    gs = GridSpec(2, 2, height_ratios=[0.40, 1.0], width_ratios=[1, 0.030],
                  hspace=0.42, wspace=0.04, figure=fig)
    axc = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # ---- (a) how concentrated the inventory-scaled accounting reduction is
    s = df[abs_col].sort_values(ascending=False).to_numpy()
    cum = np.cumsum(s) / s.sum() * 100
    rank = np.arange(1, len(s) + 1)
    axc.fill_between(rank, 0, cum, color=NEG, alpha=0.10, zorder=2)
    axc.plot(rank, cum, color=NEG, lw=1.8, zorder=3)
    for k in (5, 10, 20):
        if k <= len(cum):
            axc.plot([k, k], [0, cum[k - 1]], color=INK_FAINT, lw=0.7, ls=(0, (3, 3)), zorder=3)
            axc.annotate(f"top {k}: {cum[k - 1]:.0f}%", xy=(k, cum[k - 1]),
                         xytext=(6, -4), textcoords="offset points",
                         fontsize=7.8, color=INK, va="top", zorder=4)
    axc.set_xscale("log")
    axc.set_xlim(1, len(s))
    axc.set_ylim(0, 106)
    axc.set_xlabel("Countries ranked by absolute reduction (log scale)", labelpad=2)
    axc.set_ylabel("Cumulative share of\ntotal reduction (%)")
    axc.grid(True, zorder=0)
    axc.set_axisbelow(True)
    spines(axc, keep=("left", "bottom"))
    panel_tag(axc, "a", dx=-0.105)

    # ---- (b) baseline intensity against reduction potential
    prod = df["production_tonnes"].to_numpy()
    size = 13 + 300 * (prod / prod.max()) ** 0.40
    sc = ax.scatter(df["reduction_mean_pct"], df[base_col],
                    s=size, c=df[abs_col], cmap=SEQUENTIAL,
                    alpha=0.88, edgecolor=MARK_EDGE, linewidth=0.45, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(df[base_col].min() * 0.6,
                df[base_col].max() * 2.8)
    ax.set_xlim(df["reduction_mean_pct"].min() - 3.0,
                df["reduction_mean_pct"].max() * 1.16)
    ax.set_xlabel("Mean intensity reduction potential (%)")
    ax.set_ylabel("Posterior reference intensity (g CH$_4$ kg$^{-1}$, log scale)")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    spines(ax, keep=("left", "bottom"))
    panel_tag(ax, "b", dx=-0.135, dy=1.045)

    short = {"Iran (Islamic Republic of)": "Iran", "China, mainland": "China",
             "Russian Federation": "Russia", "United Republic of Tanzania": "Tanzania"}
    top = df.reindex(df[abs_col].sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        ax.annotate(short.get(r["country"], r["country"]),
                    xy=(r["reduction_mean_pct"], r[base_col]),
                    xytext=(7, 6), textcoords="offset points", fontsize=8, color=INK, zorder=5)

    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Absolute reduction (Mt CH$_4$)", fontsize=8.5, color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0, colors=INK_SOFT)

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#d8d4ce",
                      markeredgecolor=MARK_EDGE, label=lab,
                      markersize=np.sqrt(13 + 300 * f ** 0.40) * 0.58)
               for f, lab in ((0.02, "small"), (0.25, "medium"), (1.0, "largest"))]
    ax.legend(handles=handles, title="Milk production", title_fontsize=8,
              loc="lower right", ncol=1, handletextpad=1.0, borderpad=0.5, labelspacing=0.85)

    save(fig, out, "Figure 5")


# ============================================================== Figure 6
def figure6(ev: Path, out: Path) -> None:
    """The non-cattle portfolio on the left, the intensity it produces on the right."""
    raw = ev / "01_raw_data"
    agg = dedup_china(pd.read_csv(raw / "cercetare-485010.faostat_clean.milk_intensity_country_year.csv"))
    st = dedup_china(pd.read_csv(raw / "cercetare-485010.faostat_clean.milk_species_structure.csv"))
    unc = dedup_china(pd.read_csv(ev / "02_processed_results" / "uncertainty_summary.csv"))

    year = 2023
    agg, st = agg[agg["year"] == year], st[st["year"] == year]
    top = agg.nlargest(25, "milk_total_tonnes")[["country_m49", "country", "milk_total_tonnes"]]

    st = st.copy()
    st["species"] = st["milk_species"].map(species_key)
    comp = (st[st["species"].isin(SPECIES_ORDER)]
            .pivot_table(index="country_m49", columns="species",
                         values="species_share", aggfunc="sum")
            .reindex(columns=SPECIES_ORDER).fillna(0.0))
    comp = comp.div(comp.sum(axis=1), axis=0) * 100

    tab = (top.merge(comp, left_on="country_m49", right_index=True, how="inner")
           .merge(unc[["country_m49", "q05_kg_co2e_per_t", "q50_kg_co2e_per_t",
                       "q95_kg_co2e_per_t"]], on="country_m49", how="inner"))
    tab["non_cattle"] = 100 - tab["cattle"]
    # one shared order for both panels: ascending posterior intensity, highest at the top
    tab = tab.sort_values("q50_kg_co2e_per_t", ascending=True).reset_index(drop=True)

    # cattle is 88-100% of every system, so plotting it buries the informative species;
    # the four minority species get the whole panel and cattle is carried as a number
    minor = [s for s in SPECIES_ORDER if s != "cattle"]

    fig = plt.figure(figsize=(7.3, 6.9))
    gs = GridSpec(1, 2, width_ratios=[2.6, 1.1], wspace=0.05, figure=fig)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    y = np.arange(len(tab)) * 1.0
    bar_h = 0.52
    xmax = 65.0
    xnum = 76.0

    # ---- (a) composition of the non-cattle share
    axa.set_xticks([0, 15, 30, 45, 60])
    axa.xaxis.grid(True, color=GRID, lw=0.55, zorder=1)
    left = np.zeros(len(tab))
    for spn in minor:
        vals = tab[spn].to_numpy(dtype=float)
        axa.barh(y, vals, left=left, height=bar_h, color=SPECIES_COLOR[spn],
                 edgecolor=SURFACE, linewidth=0.7, zorder=3, label=spn.capitalize())
        lab_c = INK if spn == "sheep" else SURFACE   # mustard is too light for white text
        for yi, v, l in zip(y, vals, left):
            if v >= 6:
                axa.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                         fontsize=7.0, color=lab_c, zorder=5)
        left += vals

    # a purely bovine system has no bar; a faint zero mark keeps that distinct from
    # a missing observation
    for yi, nc in zip(y, tab["non_cattle"].to_numpy(dtype=float)):
        if nc < 0.4:
            axa.plot([0.0, 0.55], [yi, yi], color=INK_FAINT, lw=1.4,
                     solid_capstyle="butt", zorder=4)

    # cattle share as a right-hand numeric column, ruled off from the plot
    axa.plot([xnum - 8.0] * 2, [-0.85, len(tab) - 0.15], color=GRID, lw=0.7, zorder=2)
    for yi, cv in zip(y, tab["cattle"].to_numpy(dtype=float)):
        axa.text(xnum, yi, f"{cv:.0f}", ha="right", va="center", fontsize=7.8,
                 color=INK if cv < 99.5 else INK_FAINT, zorder=5)
    axa.text(xnum, len(tab) - 0.05, "cattle (%)", ha="right", va="bottom",
             fontsize=7.6, color=INK_SOFT, zorder=5)

    axa.set_yticks(y)
    axa.set_yticklabels([short_name(c) for c in tab["country"]], fontsize=8.3, color=INK)
    axa.set_xlim(0, xnum + 1.5)
    axa.set_ylim(-0.85, len(tab) - 0.15)
    axa.set_xlabel("Share of national milk output from\nspecies other than cattle (%)")
    spines(axa, keep=("bottom",))
    axa.spines["bottom"].set_bounds(0, xmax)
    axa.tick_params(axis="y", length=0)
    panel_tag(axa, "a", dx=-0.175, dy=1.030)

    handles, labels = axa.get_legend_handles_labels()
    axa.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.004),
               ncol=4, handlelength=0.85, handleheight=0.85, handletextpad=0.45,
               columnspacing=1.6, borderpad=0.0, borderaxespad=0.0)

    # ---- (b) forest plot of the posterior national intensity, same order
    lo = tab["q05_kg_co2e_per_t"].to_numpy(dtype=float)
    mid = tab["q50_kg_co2e_per_t"].to_numpy(dtype=float)
    hi = tab["q95_kg_co2e_per_t"].to_numpy(dtype=float)

    cap = bar_h * 0.32
    for yi, a, b in zip(y, lo, hi):
        axb.plot([a, b], [yi, yi], color="#9aa7b2", lw=0.9, solid_capstyle="butt", zorder=3)
        axb.plot([a, a], [yi - cap, yi + cap], color="#9aa7b2", lw=0.9, zorder=3)
        axb.plot([b, b], [yi - cap, yi + cap], color="#9aa7b2", lw=0.9, zorder=3)
    axb.scatter(mid, y, s=24, facecolor=SEQUENTIAL(0.85), edgecolor=SURFACE,
                linewidth=0.7, zorder=5)

    axb.set_xscale("log")
    axb.set_xlim(lo.min() * 0.72, hi.max() * 1.55)
    axb.set_ylim(-0.85, len(tab) - 0.15)
    axb.set_yticks([])
    ticks = [t for t in (10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
             if lo.min() * 0.72 <= t <= hi.max() * 1.55]
    axb.xaxis.set_major_locator(FixedLocator(ticks))
    axb.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    axb.xaxis.set_minor_locator(NullLocator())
    axb.xaxis.grid(True, color=GRID, lw=0.55, zorder=1)
    axb.set_xlabel("National methane intensity\n(g CH$_4$ kg$^{-1}$, log scale)")
    spines(axb, keep=("bottom",))
    panel_tag(axb, "b", dx=-0.055, dy=1.030)

    # The Pearson correlation between the non-cattle share and the national
    # intensity is deliberately NOT annotated: the share enters the aggregate
    # intensity by construction, so the association is algebraic rather than
    # independent evidence.  Panel (b) is a descriptive structural view.
    axb.text(0.0, 1.006, "Posterior median, 5-95%",
             transform=axb.transAxes, fontsize=7.9, color=INK_SOFT,
             ha="left", va="bottom", clip_on=False)

    save(fig, out, "Figure 6")


# ==============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="../evidence", type=Path)
    ap.add_argument("--out", default=".", type=Path)
    args = ap.parse_args()

    apply_style()
    ev, out = args.evidence.resolve(), args.out.resolve()
    print(f"evidence: {ev}\noutput:   {out}")
    for fn in (figure1, figure2, figure3, figure4, figure5, figure6):
        fn(ev, out)
    print("done")


if __name__ == "__main__":
    main()
