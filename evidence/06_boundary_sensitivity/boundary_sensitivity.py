"""Boundary and functional-unit sensitivity for the milk-species portfolio analysis.

Adds three families of sensitivity analysis to the deposited R3 pipeline:

  A. Allocation boundary.  For non-bovine species s the milk-allocated intensity is
     phi_s * I_s, and only r_s = phi_s / phi_cattle affects any comparison or optimum.
     Reports the break-even ratio r*_s = I_cattle / I_s and re-solves the mean-CVaR LP
     with the posterior draws of the four non-bovine species scaled by r_s.

  B. Functional unit.  Re-expresses intensities per kg FPCM
     (FPCM = M (0.337 + 0.116 F + 0.06 P)) and per kg total-solids equivalent,
     then re-solves the LP under each unit.

  C. Shapley decomposition under each convention.  Both conventions act
     multiplicatively on the species intensities, so the two-factor decomposition
     stays exact but its COMPONENTS are NOT invariant: only the additive identity
     Delta_struct + Delta_within = Delta_obs is preserved.  Panel D reports the
     production-weighted global structural and within-species components, and the
     maximum absolute reconstruction error, under every scenario.

  D. Global estimand.  The two-factor decomposition above is aggregated with FIXED
     country weights, so it is the production-weighted mean of NATIONAL intensity
     changes, not the change in the world aggregate ratio.  Panel E reports the world
     ratio R(t) = sum_cs CH4_cst / sum_cs M_cst directly and decomposes its change
     exactly into THREE factors by the Shapley value over factor groups:

         R = sum_c p_c sum_s w_cs I_cs

     with p_c the country share of world milk, w_cs the within-country species share
     and I_cs the country-species intensity.  The three values sum to
     R(2023) - R(2020) by construction.

  E. Eligible subgroup, calibration and zero cells.  Panel F reports the LP reduction
     over the 107 multi-species systems separately from the all-181 mean; Panel G gives
     the species-stratified posterior predictive coverage that the pooled 93.6%
     conceals; Panel H lists the cells with positive milk and zero reported methane,
     which carry no defined log-intensity.

The LP is an independent re-implementation of the Rockafellar-Uryasev mean-CVaR
program of Section 2.7; it reproduces the deposited country-level reductions to
~1e-13 percentage points (printed as a check).  The unscaled Shapley run likewise
reproduces the deposited global components -4.2813 / +1.8501 g CH4 per kg.

Run from the manuscript directory (the one holding evidence/).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# --- path resolution -------------------------------------------------------
# The script runs both from the manuscript evidence tree and from a clean clone of
# the public repository, whose layout differs. Resolve each input directory once.
import os as _os


def _pick(*cands):
    for c in cands:
        if _os.path.isdir(c):
            return c if c.endswith("/") else c + "/"
    return cands[0] if cands[0].endswith("/") else cands[0] + "/"


_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RAW = _pick(_os.path.join(_ROOT, "data"), "evidence/01_raw_data", "data")
REL = _pick(_os.path.join(_ROOT, "outputs_R3"), "evidence/05_release_R3",
            "evidence/05_release", "outputs_R3")
DIAG = _pick(_os.path.join(_ROOT, "outputs_R3"), "evidence/03_diagnostics", "outputs_R3")
PROC = _pick(_os.path.join(_ROOT, "outputs_R3"), "evidence/02_processed_results", "outputs_R3")
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "")
DELTA, LAM, ALPHA = 0.10, 0.50, 0.90
NONBOV = ["buffalo", "camel", "goats", "sheep"]
WTOL = 1e-9                                    # share-move tolerance for direction counts

# representative species milk composition: fat %, protein %, total solids %
COMP = {"cattle": (4.0, 3.3, 12.7), "buffalo": (7.5, 4.3, 16.8), "goats": (3.8, 3.4, 12.0),
        "sheep": (7.0, 5.6, 18.5), "camel": (3.5, 3.1, 12.4)}


def fpcm_factor(f, p):
    return 0.337 + 0.116 * f + 0.06 * p


def load():
    z = np.load(REL + "posterior_intensity_draws.npz", allow_pickle=True)
    I_all = z["I_samples"]
    countries = z["country_list"]
    species = [s.replace("Raw milk of ", "") for s in z["species_list"]]
    sh = pd.read_csv(RAW + "cercetare-485010.faostat_clean.milk_species_structure.csv")
    sh = sh[(sh.country_m49 != 159) & (sh.year == 2023)].copy()
    sh["sp"] = sh.milk_species.str.replace("Raw milk of ", "", regex=False)
    W = sh.pivot_table(index="country_m49", columns="sp", values="species_share").reindex(
        columns=species).fillna(0.0)
    ei = pd.read_csv(RAW + "cercetare-485010.faostat_clean.milk_emission_intensity_2020_2023.csv")
    ei = ei[ei.country_m49 != 159].copy()
    ei["I"] = ei.kg_co2e_per_ton_milk * 1000.0          # g CH4 / kg raw milk
    ei["sp"] = ei.milk_species.str.replace("Raw milk of ", "", regex=False)
    return I_all, countries, species, W, ei


def solve(Iscen, wref, delta=DELTA, lam=LAM, alpha=ALPHA):
    """Exact mean-CVaR LP (Rockafellar-Uryasev) over the active species.

    Returns (percentage reduction, optimal weight vector on the full species axis).
    """
    w_out = wref.copy()
    act = np.where(wref > 1e-12)[0]
    if len(act) < 2:
        return 0.0, w_out
    Is = Iscen[:, act]
    wr = wref[act]
    K, S = Is.shape
    Ibar = Is.mean(axis=0)
    coef = 1.0 / ((1.0 - alpha) * K)
    n = S + 1 + K + S                                    # w, zeta, u, d
    c = np.zeros(n)
    c[:S] = lam * Ibar
    c[S] = 1 - lam
    c[S + 1:S + 1 + K] = (1 - lam) * coef
    Aeq = np.zeros((1, n)); Aeq[0, :S] = 1.0
    rows, rhs = [], []
    for k in range(K):                                   # I_k.w - zeta - u_k <= 0
        r = np.zeros(n); r[:S] = Is[k]; r[S] = -1.0; r[S + 1 + k] = -1.0
        rows.append(r); rhs.append(0.0)
    for j in range(S):                                   # |w - wref| <= d
        r = np.zeros(n); r[j] = 1.0; r[S + 1 + K + j] = -1.0; rows.append(r); rhs.append(wr[j])
        r = np.zeros(n); r[j] = -1.0; r[S + 1 + K + j] = -1.0; rows.append(r); rhs.append(-wr[j])
    r = np.zeros(n); r[S + 1 + K:] = 1.0                 # sum d <= 2 delta
    rows.append(r); rhs.append(2.0 * delta)
    bnds = [(0, None)] * S + [(None, None)] + [(0, None)] * K + [(0, None)] * S
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=Aeq, b_eq=[1.0],
                  bounds=bnds, method="highs")
    if not res.success:
        return np.nan, w_out
    w = np.clip(res.x[:S], 0, None); w = w / w.sum()
    ref, opt = float(Ibar @ wr), float(Ibar @ w)
    if opt > ref + 1e-15:                                # do-no-harm second stage
        rows2 = rows + [np.concatenate([Ibar, [0.0], np.zeros(K), np.zeros(S)])]
        r2 = linprog(c, A_ub=np.array(rows2), b_ub=np.array(rhs + [ref]), A_eq=Aeq, b_eq=[1.0],
                     bounds=bnds, method="highs")
        if r2.success:
            w = np.clip(r2.x[:S], 0, None); w = w / w.sum(); opt = float(Ibar @ w)
        else:
            w, opt = wr.copy(), ref
    w_out = np.zeros_like(wref); w_out[act] = w
    return 100.0 * (ref - opt) / ref, w_out


def run(I_all, countries, species, W, scale=None):
    """Re-solve every country under one scenario.

    Returns (reduction series, cattle-share move per multi-species country).
    """
    ic = species.index("cattle")
    red, mov = [], []
    for i, cm in enumerate(countries):
        if cm not in W.index:
            continue
        wref = W.loc[cm].values.astype(float)
        if wref.sum() <= 0:
            continue
        wref = wref / wref.sum()
        Is = I_all[:, i, :].copy()
        if scale:
            for j, s in enumerate(species):
                Is[:, j] *= scale.get(s, 1.0)
        r, wopt = solve(Is, wref)
        red.append((cm, r))
        if (wref > 1e-12).sum() > 1:                     # multi-species systems only
            mov.append((cm, wopt[ic] - wref[ic]))
    red = pd.DataFrame(red, columns=["m49", "red"]).set_index("m49").red
    mov = pd.DataFrame(mov, columns=["m49", "d_cattle"]).set_index("m49").d_cattle
    return red, mov


def direction_counts(mov):
    """How many multi-species solutions move share toward / away from cattle."""
    toward = int((mov > WTOL).sum())
    away = int((mov < -WTOL).sum())
    return toward, away, int(len(mov) - toward - away)


def cattle_lowest(ei23, W, r):
    """# multi-species countries in which cattle is the minimum-intensity species.

    Uses the OBSERVED 2023 species ratios, not the posterior latent draws that the
    optimisation consumes; the two estimands are reported side by side deliberately.
    """
    piv = ei23.pivot_table(index="country_m49", columns="sp", values="I")
    idx = piv.index.intersection(W.index)
    piv, Wc = piv.loc[idx], W.loc[idx]
    multi = (Wc > 0).sum(axis=1) > 1
    keep = tot = 0
    for c in piv.index[multi]:
        row = piv.loc[c]
        if not (Wc.loc[c, "cattle"] > 0 and np.isfinite(row.get("cattle", np.nan))):
            continue
        tot += 1
        vals = [row[s] * r.get(s, 1.0) for s in NONBOV
                if Wc.loc[c, s] > 0 and np.isfinite(row[s])]
        if vals and row["cattle"] <= min(vals):
            keep += 1
    return keep, tot


def shapley(ei, scale=None):
    """Exact two-factor Shapley on I_c = sum_s w_s I_s, production-weighted globally.

    Delta_struct = 0.5 [ (w1 - w0).I0 + (w1 - w0).I1 ]
    Delta_within = 0.5 [ w0.(I1 - I0) + w1.(I1 - I0) ]
    The identity Delta_struct + Delta_within = Delta_obs holds exactly under ANY
    positive rescaling of the species intensities; the two components do not.
    """
    prod = ei.pivot_table(index=["country_m49", "year"], columns="sp",
                          values="milk_tonnes").fillna(0.0)
    Iv = ei.pivot_table(index=["country_m49", "year"], columns="sp", values="I")
    sps = list(prod.columns)
    sc = np.array([1.0 if scale is None else scale.get(s, 1.0) for s in sps])
    cs = sorted(set(prod.xs(2020, level="year").index) & set(prod.xs(2023, level="year").index))
    rows = []
    for c in cs:
        p0 = prod.loc[(c, 2020)].values.astype(float)
        p1 = prod.loc[(c, 2023)].values.astype(float)
        i0 = np.nan_to_num(Iv.loc[(c, 2020)].values.astype(float)) * sc
        i1 = np.nan_to_num(Iv.loc[(c, 2023)].values.astype(float)) * sc
        t0, t1 = p0.sum(), p1.sum()
        if t0 <= 0 or t1 <= 0:
            continue
        w0, w1 = p0 / t0, p1 / t1
        rows.append((0.5 * ((w1 - w0) @ i0 + (w1 - w0) @ i1),
                     0.5 * (w0 @ (i1 - i0) + w1 @ (i1 - i0)),
                     w1 @ i1 - w0 @ i0, 0.5 * (t0 + t1)))
    D = pd.DataFrame(rows, columns=["struct", "within", "obs", "wt"])
    wn = D.wt / D.wt.sum()
    return (float((D.struct * wn).sum()), float((D.within * wn).sum()),
            float((D.obs * wn).sum()), float((D.struct + D.within - D.obs).abs().max()), len(D))


def complete_panel(ei, y0=2020, y1=2023):
    """The country codes holding a record in every year of the interval."""
    n = ei.groupby("country_m49").year.nunique()
    return set(n[n == (y1 - y0 + 1)].index)


def world_ratio(ei, year, keep=None):
    """World aggregate ratio sum(CH4) / sum(milk), in g CH4 per kg raw milk.

    Restricted by default to the complete-series panel, so the reported series is
    the same population as the decomposition of Panel E.
    """
    keep = complete_panel(ei) if keep is None else keep
    d = ei[(ei.year == year) & (ei.country_m49.isin(keep))]
    return d.ch4_ktco2e.sum() * 1000.0 / d.milk_tonnes.sum() * 1000.0


def shapley_three_factor(ei, y0=2020, y1=2023):
    """Exact three-factor Shapley of the change in the WORLD aggregate ratio.

    R = sum_c p_c sum_s w_cs I_cs, with factor groups
        (0) p   country shares of world milk       (geography)
        (1) w   within-country species shares      (species mix)
        (2) I   country-species intensities        (within-species intensity)
    The Shapley value averages each factor's marginal contribution over all 3! = 6
    orderings; the three values sum to R(y1) - R(y0) exactly.
    """
    from itertools import permutations
    prod = ei.pivot_table(index=["country_m49", "year"], columns="sp",
                          values="milk_tonnes").fillna(0.0)
    Iv = ei.pivot_table(index=["country_m49", "year"], columns="sp", values="I")
    cs = sorted(set(prod.xs(y0, level="year").index) & set(prod.xs(y1, level="year").index))
    n_s = prod.shape[1]

    def build(year):
        M = np.zeros((len(cs), n_s))
        I = np.zeros((len(cs), n_s))
        for k, c in enumerate(cs):
            M[k] = prod.loc[(c, year)].values.astype(float)
            I[k] = np.nan_to_num(Iv.loc[(c, year)].values.astype(float))
        Mc = M.sum(axis=1)
        p = Mc / Mc.sum()
        w = np.divide(M, np.where(Mc[:, None] == 0, 1.0, Mc[:, None]))
        return p, w, I

    end = dict(enumerate(zip(build(y0), build(y1))))

    def f(p, w, I):
        return float(np.sum(p[:, None] * w * I))

    def ev(st):
        return f(*[end[k][st[k]] for k in range(3)])

    val = np.zeros(3)
    perms = list(permutations(range(3)))
    for order in perms:
        st = [0, 0, 0]
        prev = ev(st)
        for k in order:
            st[k] = 1
            cur = ev(st)
            val[k] += cur - prev
            prev = cur
    val /= len(perms)
    return val, ev([0, 0, 0]), ev([1, 1, 1]), len(cs)


def ppc_by_species(ei):
    """Species-stratified posterior predictive coverage.

    The deposited PPC table is indexed by model row only, so rows are matched back to
    country-year-species by their log-intensity value, which is unique to 1e-9.
    """
    ppc = pd.read_csv(DIAG + "bayes_ppc_summary.csv")
    sh = pd.read_csv(RAW + "cercetare-485010.faostat_clean.milk_species_structure.csv")
    sh = sh[sh.country_m49 != 159].copy()
    sh["sp"] = sh.milk_species.str.replace("Raw milk of ", "", regex=False)
    d = ei.merge(sh[["country_m49", "year", "sp", "species_share"]],
                 on=["country_m49", "year", "sp"], how="left")
    d = d[(d.milk_tonnes > 0) & (d.species_share > 0) & (d.I > 0)].copy()
    d["k"] = np.log(d.kg_co2e_per_ton_milk).round(9)
    lut = {}
    for row in d.itertuples():
        lut.setdefault(row.k, []).append((row.sp, row.year))
    tag = []
    for row in ppc.itertuples():
        q = lut.get(round(row.y_obs, 9))
        tag.append(q.pop(0) if q else (None, None))
    ppc[["sp", "yr"]] = pd.DataFrame(tag, index=ppc.index)
    g = ppc.dropna(subset=["sp"]).groupby("sp").agg(
        n=("within_90ci", "size"),
        coverage_90ci_pct=("within_90ci", "mean"),
        mean_log_residual=("residual", "mean"),
        median_log_residual=("residual", "median"),
        n_abs_resid_gt_2=("residual", lambda x: int((x.abs() > 2).sum())),
        n_abs_resid_gt_3=("residual", lambda x: int((x.abs() > 3).sum())))
    g["coverage_90ci_pct"] = (g.coverage_90ci_pct * 100).round(2)
    for col in ("mean_log_residual", "median_log_residual"):
        g[col] = g[col].round(4)
    return g.sort_values("coverage_90ci_pct").reset_index(), int(ppc.sp.isna().sum())


def zero_emission_rows(ei):
    """Country-year-species cells with positive milk but zero reported methane.

    These carry no defined log-intensity and are therefore dropped from the
    log-likelihood of Section 2.6 while remaining in the accounting identity.
    """
    z = ei[(ei.milk_tonnes > 0) & (ei.ch4_ktco2e == 0)]
    return z[["country", "country_m49", "year", "sp", "milk_tonnes", "ch4_ktco2e"]]


def scenarios():
    """(label, posterior-draw scale, observed-ratio scale) for every convention tested."""
    fp = dict((s, fpcm_factor(*COMP[s][:2])) for s in COMP)
    ts = dict((s, COMP[s][2] / COMP["cattle"][2]) for s in COMP)
    diff = {"buffalo": 0.70, "goats": 0.45, "sheep": 0.25, "camel": 0.35}
    out = [("Whole-herd numerator, raw milk (reported)", None, {})]
    for r in (0.75, 0.50, 0.25, 0.10):
        out.append(("Uniform allocation ratio r = %.2f" % r,
                    dict((s, r) for s in NONBOV), dict((s, r) for s in NONBOV)))
    out += [
        ("Illustrative species-differentiated allocation", diff, diff),
        ("Per kg FPCM", dict((s, 1.0 / fp[s]) for s in fp),
         dict((s, (1.0 / fp[s]) * fp["cattle"]) for s in fp)),
        ("Per kg total-solids equivalent", dict((s, 1.0 / ts[s]) for s in ts),
         dict((s, 1.0 / ts[s]) for s in ts)),
        ("FPCM + illustrative differentiated allocation",
         dict((s, (1.0 / fp[s]) * diff.get(s, 1.0)) for s in fp),
         dict((s, (1.0 / fp[s]) * fp["cattle"] * diff.get(s, 1.0)) for s in fp)),
    ]
    return out, fp, ts


def main():
    I_all, countries, species, W, ei = load()
    ei23 = ei[ei.year == 2023]
    med = ei23.groupby("sp").I.median()

    pub = pd.read_csv(REL + "robust_optimization_results.csv")
    pub = pub[(pub.delta == DELTA) & (pub["lambda"] == LAM) & (pub.alpha == ALPHA)]
    base, base_mov = run(I_all, countries, species, W)
    chk = (base - pub.set_index("country_m49").reduction_mean_pct).abs().max()
    print("LP reproduction check vs deposited output: max |diff| = %.3e pp" % chk)
    print("baseline mean %.4f%%  median %.4f%%  (deposited: 11.9065 / 2.3785)"
          % (base.mean(), base.median()))
    s0, w0, o0, e0, n0 = shapley(ei)
    print("Shapley reproduction check: struct %+.4f  within %+.4f  net %+.4f  (n = %d)"
          % (s0, w0, s0 + w0, n0))
    print("                 deposited: struct -4.2813  within +1.8501  net -2.4312\n")

    # ---- Panel A: break-even ratios ----
    A = pd.DataFrame([{"species": s, "median_2023_g_per_kg": round(med[s], 1),
                       "ratio_to_cattle": round(med[s] / med["cattle"], 2),
                       "break_even_r_star": round(med["cattle"] / med[s], 4)} for s in NONBOV])
    print("PANEL A - allocation break-even (2023 medians; cattle median %.1f g CH4/kg)"
          % med["cattle"])
    print(A.to_string(index=False), "\n")

    scen, fp, ts = scenarios()

    # ---- Panel B: scenario re-solves, with reallocation-direction counts ----
    rowsB = []
    print("PANEL B - portfolio re-solves (delta = %.2f, lambda = %.2f, alpha = %.2f)"
          % (DELTA, LAM, ALPHA))
    for name, sc, rr in scen:
        red, mov = run(I_all, countries, species, W, scale=sc)
        keep, tot = cattle_lowest(ei23, W, rr if rr else {})
        tw, aw, nc = direction_counts(mov)
        rowsB.append({"scenario": name, "mean_reduction_pct": round(red.mean(), 2),
                      "median_reduction_pct": round(red.median(), 2),
                      "cattle_lowest_n": keep, "cattle_lowest_of": tot,
                      "n_toward_cattle": tw, "n_away_from_cattle": aw, "n_no_move": nc})
        print(" %-46s mean %6.2f%%  median %5.2f%%  cattle lowest %3d/%3d"
              "  toward %3d  away %3d  static %3d"
              % (name, red.mean(), red.median(), keep, tot, tw, aw, nc))
    B = pd.DataFrame(rowsB)

    # ---- Panel C: functional-unit ratios ----
    C = pd.DataFrame([{"species": s, "fat_pct": COMP[s][0], "protein_pct": COMP[s][1],
                       "total_solids_pct": COMP[s][2], "fpcm_factor": round(fp[s], 3),
                       "ts_factor": round(ts[s], 3), "raw_g_per_kg": round(med[s], 1),
                       "per_kg_fpcm": round(med[s] / fp[s], 1),
                       "per_kg_ts_eq": round(med[s] / ts[s], 1)}
                      for s in ["cattle"] + NONBOV])
    print("\nPANEL C - functional unit")
    print(C.to_string(index=False))

    # ---- Panel D: Shapley decomposition under each convention ----
    rowsD = []
    print("\nPANEL D - production-weighted global Shapley components, 2020-2023"
          " (g CH4 per kg)")
    for name, sc, _ in scen:
        st, wi, ob, err, nn = shapley(ei, scale=sc)
        rowsD.append({"scenario": name, "delta_struct_g_per_kg": round(st, 4),
                      "delta_within_g_per_kg": round(wi, 4),
                      "delta_net_g_per_kg": round(st + wi, 4),
                      "max_identity_error": "%.2e" % err, "n_countries": nn})
        print(" %-46s struct %+7.4f  within %+7.4f  net %+7.4f  |id err| %.1e"
              % (name, st, wi, st + wi, err))
    D = pd.DataFrame(rowsD)
    print("\n Note: the additive identity is exact in every scenario (column"
          " max_identity_error);\n the COMPONENTS are not invariant, and the"
          " within-species term changes sign under the\n allocation scenarios.")

    # ---- Panel E: world aggregate ratio and its exact three-factor decomposition ----
    print("\nPANEL E - the global estimand")
    for y in (2020, 2021, 2022, 2023):
        print("  world aggregate ratio %d: %.6f g CH4/kg" % (y, world_ratio(ei, y)))
    v3, R0, R1, n3 = shapley_three_factor(ei)
    names3 = ["country shares of world milk (geography)",
              "within-country species mix",
              "country-species intensities"]
    print("  change in the WORLD aggregate ratio 2020-2023: %+.4f g CH4/kg (%.4f -> %.4f)"
          % (R1 - R0, R0, R1))
    print("  fixed-country-weight mean of national changes: %+.4f g CH4/kg"
          "  -- a DIFFERENT estimand" % (s0 + w0))
    for nm, vv in zip(names3, v3):
        print("    %-42s %+8.4f" % (nm, vv))
    print("    %-42s %+8.4f   (identity error %.1e)"
          % ("sum", v3.sum(), abs(v3.sum() - (R1 - R0))))
    E = pd.DataFrame({
        "quantity": ["world_ratio_2020", "world_ratio_2023", "world_ratio_change",
                     "shapley3_geography", "shapley3_within_country_species_mix",
                     "shapley3_country_species_intensity", "shapley3_sum",
                     "shapley3_identity_error", "fixed_country_weight_struct",
                     "fixed_country_weight_within", "fixed_country_weight_net",
                     "n_countries"],
        "value_g_ch4_per_kg": [R0, R1, R1 - R0, v3[0], v3[1], v3[2], v3.sum(),
                               abs(v3.sum() - (R1 - R0)), s0, w0, s0 + w0, n3]})

    # ---- Panel F: eligible multi-species subgroup ----
    elig = base[base > 1e-9]
    n_single = int((base <= 1e-9).sum())
    print("\nPANEL F - LP reduction: all analysed countries vs the eligible subgroup")
    print("  all analysed       n = %3d  mean %7.4f%%  median %7.4f%%"
          % (len(base), base.mean(), base.median()))
    print("  multi-species only n = %3d  mean %7.4f%%  median %7.4f%%"
          % (len(elig), elig.mean(), elig.median()))
    print("  single-species     n = %3d  mean    0.0000%%  (mix is not a decision variable)"
          % n_single)
    F = pd.DataFrame([
        {"group": "all analysed countries", "n": len(base),
         "mean_reduction_pct": round(base.mean(), 4),
         "median_reduction_pct": round(base.median(), 4)},
        {"group": "multi-species systems (eligible)", "n": len(elig),
         "mean_reduction_pct": round(elig.mean(), 4),
         "median_reduction_pct": round(elig.median(), 4)},
        {"group": "single-species systems", "n": n_single,
         "mean_reduction_pct": 0.0, "median_reduction_pct": 0.0}])

    # ---- Panel G: species-stratified posterior predictive coverage ----
    G, unmatched = ppc_by_species(ei)
    print("\nPANEL G - posterior predictive coverage by species"
          " (pooled 93.56%%; unmatched rows %d)" % unmatched)
    print(G.to_string(index=False))
    print(" Note: cattle carry essentially no mean log bias, while all four non-bovine")
    print(" species are over-predicted in the mean and every |residual| > 3 outlier is")
    print(" non-bovine.")

    # ---- Panel H: zero-methane cells with positive milk ----
    Z = zero_emission_rows(ei)
    print("\nPANEL H - cells with positive milk and zero reported methane"
          " (dropped from the log-likelihood)")
    print(Z.to_string(index=False) if len(Z) else "  none")

    A.to_csv(OUT + "panelA_break_even.csv", index=False)
    E.to_csv(OUT + "panelE_global_estimand.csv", index=False)
    F.to_csv(OUT + "panelF_eligible_subgroup.csv", index=False)
    G.to_csv(OUT + "panelG_ppc_by_species.csv", index=False)
    Z.to_csv(OUT + "panelH_zero_methane_cells.csv", index=False)
    B.to_csv(OUT + "panelB_scenarios.csv", index=False)
    C.to_csv(OUT + "panelC_functional_unit.csv", index=False)
    D.to_csv(OUT + "panelD_shapley.csv", index=False)
    print("\nwritten to " + OUT)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    main()
