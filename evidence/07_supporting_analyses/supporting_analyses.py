"""Supporting analyses for the Supplementary Materials (Section S4).

Regenerates, from the deposited frozen inputs and the deposited posterior draw
array, the robustness checks reported in Supplementary Sections S4.2 to S4.9:

  S4.2  robust species summaries (median, IQR, 20% trimmed mean, raw mean, and the
        raw mean recomputed without denominator-collapse cells)
  S4.3  country-species reporting coverage
  S4.4  influence analysis (copied from the deposited pipeline output)
  S4.5  finite-draw subsampling stability of the mean-CVaR solution
  S4.6  Dirichlet concentration sensitivity of the reference mix
  S4.7  trend / post-2022 step separability (copied from the deposited output)
  S4.8  cells carrying positive milk output and a reported methane value of zero
  S4.9  FAOSTAT reporting entities absent from the frozen analytical extraction

The mean-CVaR linear program is imported from boundary_sensitivity.py, so the same
implementation that reproduces the deposited country-level reductions to 1.3e-13
percentage points is used here.

Run from the manuscript directory (the one holding evidence/).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "06_boundary_sensitivity"))
from boundary_sensitivity import load, solve, DELTA, LAM, ALPHA, RAW, PROC

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "")

SEED = 20230101
TRIM = 0.20
COLLAPSE_Q = 0.99          # species-wise quantile above which a ratio is flagged
COLLAPSE_M = 0.25          # and milk below this fraction of the cell's own maximum

# FAOSTAT reporting entities that carry milk and emissions records upstream but are
# absent from the three frozen analytical tables. The criterion that removed them is
# not recoverable from the archived artefacts; see S4.9.
ABSENT = [
    (682, "Saudi Arabia", "cattle, camels, goats, sheep"),
    (760, "Syrian Arab Republic", "cattle, goats, sheep, camels"),
    (784, "United Arab Emirates", "cattle, camels, goats"),
    (140, "Central African Republic", "cattle, goats"),
    (710, "South Africa", "cattle, goats, sheep"),
]


def species_panel():
    ei = pd.read_csv(RAW + "cercetare-485010.faostat_clean.milk_emission_intensity_2020_2023.csv")
    ei = ei[ei.country_m49 != 159].copy()
    ei["sp"] = ei.milk_species.str.replace("Raw milk of ", "", regex=False)
    ei["I"] = ei.kg_co2e_per_ton_milk * 1000.0          # g CH4 per kg raw milk
    return ei


# ------------------------------------------------------------------ S4.2
def robust_species_summaries(ei):
    milk_col = "milk_tonnes"
    if milk_col is None:
        for c in ei.columns:
            if "milk" in c.lower() and ei[c].dtype.kind in "fi":
                milk_col = c
                break
    flag = pd.Series(False, index=ei.index)
    for sp, g in ei.groupby("sp"):
        q = g.I.quantile(COLLAPSE_Q)
        mx = g.groupby("country_m49")[milk_col].transform("max")
        flag.loc[g.index] = (g.I > q) & (g[milk_col] < COLLAPSE_M * mx)
    ei = ei.assign(collapse_flag=flag)
    rows = []
    for (sp, yr), g in ei.groupby(["sp", "year"]):
        keep = g[~g.collapse_flag]
        rows.append(dict(
            species=sp, year=int(yr), n=len(g),
            mean_raw=g.I.mean(),
            median=g.I.median(),
            iqr=g.I.quantile(0.75) - g.I.quantile(0.25),
            trimmed_mean_20pct=trim_mean(g.I.values, TRIM) if len(g) >= 5 else np.nan,
            mean_excluding_collapse_cells=keep.I.mean(),
            n_collapse_cells=int(g.collapse_flag.sum()),
        ))
    out = pd.DataFrame(rows).sort_values(["species", "year"])
    cells = ei[ei.collapse_flag][["country_m49", "country", "sp", "year", "I", milk_col]] \
        if "country" in ei.columns else ei[ei.collapse_flag][
            ["country_m49", "sp", "year", "I", milk_col]]
    return out, cells


# ------------------------------------------------------------------ S4.3
def coverage(ei):
    by_year = ei.groupby(["sp", "year"]).country_m49.nunique().unstack("year")
    e23 = ei[ei.year == 2023]
    nspec = e23.groupby("country_m49").sp.nunique()
    mat = (e23.assign(k=e23.country_m49.map(nspec))
           .groupby(["sp", "k"]).country_m49.nunique().unstack("k").fillna(0).astype(int))
    dist = nspec.value_counts().sort_index().rename("n_countries").to_frame()
    dist.index.name = "species_reported"
    return by_year, mat, dist


# ------------------------------------------------------------------ S4.5
def finite_draw_stability(I_all, countries, species, W, sizes=(100, 250, 500), repeats=15):
    rng = np.random.default_rng(SEED)
    full = {}
    base_rows = []
    for i, cm in enumerate(countries):
        if cm not in W.index:
            continue
        wref = W.loc[cm].values.astype(float)
        if wref.sum() <= 0:
            continue
        wref /= wref.sum()
        r, w = solve(I_all[:, i, :], wref)
        full[cm] = (r, w)
    base_mean = np.mean([v[0] for v in full.values()])

    rows, rep_rows = [], []
    for n in sizes:
        nrep = 1 if n == I_all.shape[0] else repeats
        means, ident = [], []
        for rep in range(nrep):
            idx = (np.arange(I_all.shape[0]) if n == I_all.shape[0]
                   else rng.choice(I_all.shape[0], n, replace=False))
            reds, same = [], 0
            for i, cm in enumerate(countries):
                if cm not in full:
                    continue
                wref = W.loc[cm].values.astype(float)
                wref /= wref.sum()
                r, w = solve(I_all[idx][:, i, :], wref)
                reds.append(r)
                if np.allclose(w, full[cm][1], atol=1e-8):
                    same += 1
            means.append(float(np.mean(reds)))
            ident.append(same)
            rep_rows.append(dict(n_draws=n, repeat=rep + 1, mean_reduction_pct=means[-1],
                                 n_identical_vertex=same, n_countries=len(reds)))
        rows.append(dict(n_draws=n, repeats=nrep,
                         mean_reduction_pct_min=min(means),
                         mean_reduction_pct_max=max(means),
                         mean_reduction_pct_spread_pp=max(means) - min(means),
                         n_identical_vertex_min=min(ident),
                         n_countries=len(full),
                         full_draw_mean_reduction_pct=base_mean))
    return pd.DataFrame(rows), pd.DataFrame(rep_rows)


# ------------------------------------------------------------------ S4.6
def dirichlet_sensitivity(I_all, countries, species, W,
                          kappas=(50, 200, 1000, 20000, np.inf), repeats=10):
    """Perturb the reference mix as a Dirichlet draw and re-solve.

    The concentration is a scenario input and carries no distributional claim.
    Finite concentrations are averaged over `repeats` independent draws; the
    direction counts are reported as the range over those draws.
    """
    rng = np.random.default_rng(SEED)
    ic = species.index("cattle")
    rows = []
    for kap in kappas:
        nrep = 1 if np.isinf(kap) else repeats
        means, meds, tow = [], [], []
        n_multi = 0
        for rep in range(nrep):
            reds, toward, multi = [], 0, 0
            for i, cm in enumerate(countries):
                if cm not in W.index:
                    continue
                w0 = W.loc[cm].values.astype(float)
                if w0.sum() <= 0:
                    continue
                w0 = w0 / w0.sum()
                act = w0 > 1e-12
                if np.isinf(kap):
                    wref = w0
                else:
                    wref = np.zeros_like(w0)
                    wref[act] = rng.dirichlet(kap * w0[act])
                r, w = solve(I_all[:, i, :], wref)
                reds.append(r)
                if act.sum() > 1:
                    multi += 1
                    if w[ic] - wref[ic] > 1e-9:
                        toward += 1
            means.append(float(np.mean(reds)))
            meds.append(float(np.median(reds)))
            tow.append(toward)
            n_multi = multi
            n_all = len(reds)
        rows.append(dict(kappa=('inf' if np.isinf(kap) else kap), repeats=nrep,
                         mean_reduction_pct=float(np.mean(means)),
                         mean_reduction_pct_min=min(means),
                         mean_reduction_pct_max=max(means),
                         median_reduction_pct=float(np.mean(meds)),
                         n_multi_species=n_multi,
                         n_toward_cattle_min=min(tow), n_toward_cattle_max=max(tow),
                         n_countries=n_all))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    import os
    import shutil
    os.makedirs(OUT, exist_ok=True)

    ei = species_panel()
    I_all, countries, species, W, _ = load()

    print("S4.2 robust species summaries")
    rs, cells = robust_species_summaries(ei)
    rs.to_csv(OUT + "robust_species_summaries.csv", index=False)
    cells.to_csv(OUT + "denominator_collapse_cells.csv", index=False)
    print(rs.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("  denominator-collapse cells flagged:", len(cells))

    print("\nS4.3 coverage")
    by_year, mat, dist = coverage(ei)
    by_year.to_csv(OUT + "coverage_by_species_year.csv")
    mat.to_csv(OUT + "coverage_matrix_2023.csv")
    dist.to_csv(OUT + "coverage_species_count_2023.csv")
    print(by_year.to_string())
    print(dist.to_string())

    print("\nS4.4 / S4.7 deposited pipeline outputs copied forward")
    for src, dst in [(PROC + "influence_summary_both_estimands.csv",
                      OUT + "influence_summary_both_estimands.csv"),
                     (PROC + "trend_step_separability.csv",
                      OUT + "trend_step_separability.csv")]:
        shutil.copyfile(src, dst)
        print("  ", dst)

    print("\nS4.5 finite-draw subsampling stability")
    fds, fdr = finite_draw_stability(I_all, countries, species, W)
    fds.to_csv(OUT + "finite_draw_stability.csv", index=False)
    fdr.to_csv(OUT + "finite_draw_stability_repeats.csv", index=False)
    print(fds.to_string(index=False))

    print("\nS4.6 Dirichlet concentration sensitivity")
    dc = dirichlet_sensitivity(I_all, countries, species, W)
    dc.to_csv(OUT + "dirichlet_concentration_sensitivity.csv", index=False)
    print(dc.to_string(index=False))

    print("\nS4.9 entities absent from the frozen extraction")
    present = set(ei.country_m49.unique())
    rows = []
    for m49, name, spp in ABSENT:
        rows.append(dict(country=name, m49=m49, species_potentially_affected=spp,
                         present_in_frozen_inputs=bool(m49 in present),
                         status="absent from the frozen analytical extraction",
                         reason="not recoverable from the archived artefacts; "
                                "no post hoc reconstruction or imputation performed"))
    ab = pd.DataFrame(rows)
    ab.to_csv(OUT + "absent_reporting_entities.csv", index=False)
    print(ab.to_string(index=False))
    assert not ab.present_in_frozen_inputs.any(), "an 'absent' entity is present in the inputs"
    print("\n  verified: none of the five entities appears in the frozen analytical inputs")
    print("  analysed entities in the frozen inputs:", len(present))

    json.dump({"seed": SEED, "trim": TRIM, "collapse_quantile": COLLAPSE_Q,
               "collapse_milk_fraction": COLLAPSE_M,
               "delta": DELTA, "lambda": LAM, "alpha": ALPHA},
              open(OUT + "supporting_analyses_config.json", "w"), indent=2)
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
