#!/usr/bin/env python3
"""
02_analysis.py - revised analysis (Animals manuscript, revision 1)

Input : ../R1_data/faostat_panel_2020_2023_long.csv   (from 01_extract_faostat.py)
Output: ../R1_results/*.csv, results_summary.json

Indicators (g CH4 per kg raw milk)
  I_whole = CH4_whole_herd / M                         whole-herd boundary
  I_milk  = I_whole * delta ,  delta = PAS / TS        milk-allocated boundary (non-bovine species)
                                delta = 1 for cattle   ('Cattle, dairy' = milk cows already)
Decompositions: exact Shapley (two-factor, country level; three-factor, world ratio).
Scenario      : bounded compositional counterfactual (closed form; verified against an LP solver).
"""
import os, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "R1_data")
OUT = os.path.join(HERE, "..", "R1_results")
os.makedirs(OUT, exist_ok=True)
SP = ["cattle", "buffalo", "goats", "sheep", "camel"]
Y0, Y1 = 2020, 2023
YEARS = [2020, 2021, 2022, 2023]
G_PER_KG = 1e6  # kt CH4 / t milk -> g CH4 / kg milk

# representative composition (% fat, % protein) used for the FPCM sensitivity (as in the original submission)
COMP = {"cattle": (4.0, 3.3), "buffalo": (7.5, 4.3), "goats": (3.8, 3.4), "sheep": (7.0, 5.6), "camel": (3.5, 3.1)}
FPCM_F = {s: 0.1226 * f + 0.0776 * p + 0.2534 for s, (f, p) in COMP.items()}  # IDF (2015) / ISO 14040-type FPCM


# ----------------------------------------------------------------------------- data
def build_panel():
    d = pd.read_csv(os.path.join(DATA, "faostat_panel_2020_2023_long.csv"))
    d["m49"] = d["m49"].astype(int)
    d["delta_raw"] = np.where(d.species == "cattle", 1.0, d.milk_animals / d.stock_qcl)  # FAO EI: TS = QCL live animals
    d["delta_gle"] = np.where(d.species == "cattle", 1.0, d.milk_animals / d.stock_gle)
    d["delta_gle"] = pd.Series(d.delta_gle, index=d.index).clip(upper=1.0).fillna(d.delta_raw.clip(upper=1.0))
    d["delta"] = d.delta_raw.clip(upper=1.0)
    d["I_whole"] = d.ch4_whole_kt / d.milk_t * G_PER_KG
    d["I_milk"] = d.I_whole * d.delta
    d["I_milk_gle"] = d.I_whole * d.delta_gle
    ok_cell = (d.milk_t > 0) & (d.ch4_whole_kt > 0) & d.delta.notna() & (d.delta > 0)
    d["valid_cell"] = ok_cell
    g = d.groupby(["m49", "species"]).valid_cell.agg(["sum"])
    complete = g[g["sum"] == len(YEARS)].index
    d["complete"] = [(m, s) in set(complete) for m, s in zip(d.m49, d.species)]
    return d


def coverage_tables(d):
    full = d[d.year.isin(YEARS)]
    # country-species pairs excluded and why
    rows = []
    for (m, s), g in full.groupby(["m49", "species"]):
        if g.complete.all():
            continue
        reasons = []
        if len(g) < len(YEARS):
            reasons.append("milk series incomplete in 2020-2023")
        if g.ch4_whole_kt.isna().any():
            reasons.append("no GLE methane for %d year(s)" % g.ch4_whole_kt.isna().sum())
        if (g.ch4_whole_kt == 0).any():
            reasons.append("zero methane against positive milk")
        if g.milk_animals.isna().any() and s != "cattle":
            reasons.append("no milk-animal count")
        if g.stock_qcl.isna().any() and s != "cattle":
            reasons.append("no QCL total-stock count")
        rows.append(dict(country=g.country.iloc[0], m49=m, species=s, years_with_milk=len(g),
                         milk_2023_t=g[g.year == Y1].milk_t.sum(), reason="; ".join(sorted(set(reasons)))))
    ex = pd.DataFrame(rows).sort_values(["country", "species"])
    ex.to_csv(os.path.join(OUT, "S_excluded_country_species.csv"), index=False)
    return ex


# ----------------------------------------------------------------------------- matrices
def wide(d, boundary, year, drop=None, fpcm=False):
    """countries x species matrices of milk (M), methane charged to milk (C) and intensity (I)."""
    x = d[(d.year == year) & d.complete].copy()
    if drop is not None:
        x = x[~drop(x)]
    icol = {"milk": "I_milk", "milk_gle": "I_milk_gle", "whole": "I_whole"}[boundary]
    x["I"] = x[icol] / (x.species.map(FPCM_F) if fpcm else 1.0)
    x["Mx"] = x.milk_t * (x.species.map(FPCM_F) if fpcm else 1.0)
    M = x.pivot_table(index="m49", columns="species", values="Mx", aggfunc="sum").reindex(columns=SP).fillna(0.0)
    I = x.pivot_table(index="m49", columns="species", values="I", aggfunc="sum").reindex(columns=SP).fillna(0.0)
    return M, I


def endpoints(d, boundary, **kw):
    M0, I0 = wide(d, boundary, Y0, **kw)
    M1, I1 = wide(d, boundary, Y1, **kw)
    common = M0.index.intersection(M1.index)
    # keep countries that still have milk at both endpoints after dropping
    keep = [c for c in common if M0.loc[c].sum() > 0 and M1.loc[c].sum() > 0]
    return M0.loc[keep], I0.loc[keep], M1.loc[keep], I1.loc[keep]


# ----------------------------------------------------------------------------- Shapley
def shapley2(M0, I0, M1, I1):
    """country-level exact two-factor Shapley (structure vs within-species)."""
    W0 = M0.div(M0.sum(1), axis=0).values
    W1 = M1.div(M1.sum(1), axis=0).values
    a, b = I0.values, I1.values
    dW, dI = W1 - W0, b - a
    st = 0.5 * ((dW * a).sum(1) + (dW * b).sum(1))
    wi = 0.5 * ((W0 * dI).sum(1) + (W1 * dI).sum(1))
    obs = (W1 * b).sum(1) - (W0 * a).sum(1)
    res = pd.DataFrame({"struct": st, "within": wi, "total": st + wi, "obs": obs,
                        "wt": 0.5 * (M0.sum(1).values + M1.sum(1).values)}, index=M0.index)
    assert np.abs(res.total - res.obs).max() < 1e-8
    return res


def global2(res):
    w = res.wt / res.wt.sum()
    return dict(struct=float((w * res.struct).sum()), within=float((w * res.within).sum()),
                net=float((w * res.total).sum()), n=int(len(res)))


def shapley3(M0, I0, M1, I1):
    """exact three-factor Shapley of the world aggregate ratio R = sum_c p_c sum_s w_cs I_cs."""
    def fac(M, I):
        p = (M.sum(1) / M.values.sum()).values
        w = M.div(M.sum(1), axis=0).values
        return p, w, I.values

    f0, f1 = fac(M0, I0), fac(M1, I1)

    def R(sel):  # sel[k] in {0,1}: which endpoint each factor takes
        p = f1[0] if sel[0] else f0[0]
        w = f1[1] if sel[1] else f0[1]
        i = f1[2] if sel[2] else f0[2]
        return float((p[:, None] * w * i).sum())

    names = ["country_shares", "species_mix", "species_intensity"]
    phi = {n: 0.0 for n in names}
    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        sel = [0, 0, 0]
        for k in perm:
            before = R(sel)
            sel[k] = 1
            phi[names[k]] += (R(sel) - before) / len(perms)
    r0, r1 = R([0, 0, 0]), R([1, 1, 1])
    phi.update(R0=r0, R1=r1, delta=r1 - r0)
    assert abs(sum(phi[n] for n in names) - phi["delta"]) < 1e-9
    return phi


# ----------------------------------------------------------------------------- scenario
def counterfactual_country(w, I, delta):
    """closed-form optimum of min sum w_s I_s s.t. 0.5||w-w_ref||_1 <= delta, no expansion of absent species."""
    w = np.asarray(w, float)
    I = np.asarray(I, float)
    present = w > 0
    if present.sum() < 2:
        return w.copy(), 0.0, False
    recv = np.argmin(np.where(present, I, np.inf))
    order = [s for s in np.argsort(-I) if present[s] and s != recv and I[s] > I[recv]]
    left = delta
    wn = w.copy()
    for s in order:
        mv = min(left, wn[s])
        wn[s] -= mv
        wn[recv] += mv
        left -= mv
        if left <= 1e-15:
            break
    return wn, delta - left, True


def counterfactual_lp(w, I, delta):
    S = len(w)
    present = np.where(np.asarray(w) > 0)[0]
    n = len(present)
    if n < 2:
        return np.asarray(w, float)
    # variables: w' (n), d+ (n), d- (n); w' - w = d+ - d-
    c = np.r_[np.asarray(I)[present], np.zeros(2 * n)]
    A_eq = np.zeros((n + 1, 3 * n)); b_eq = np.zeros(n + 1)
    for k in range(n):
        A_eq[k, k] = 1; A_eq[k, n + k] = -1; A_eq[k, 2 * n + k] = 1; b_eq[k] = np.asarray(w)[present][k]
    A_eq[n, :n] = 1; b_eq[n] = 1
    A_ub = np.r_[np.zeros(n), np.ones(n), np.ones(n)][None, :] * 0.5
    r = linprog(c, A_ub=A_ub, b_ub=[delta], A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    out = np.zeros(S); out[present] = r.x[:n]
    return out


def scenario(d, boundary, delta=0.10, fpcm=False, drop=None):
    M1, I1 = wide(d, boundary, Y1, drop=drop, fpcm=fpcm)
    W = M1.div(M1.sum(1), axis=0)
    rows = []
    for c in M1.index:
        w = W.loc[c].values; I = I1.loc[c].values
        wn, used, elig = counterfactual_country(w, I, delta)
        iref, icf = float((w * I).sum()), float((wn * I).sum())
        present = w > 0
        recv = int(np.argmin(np.where(present, I, np.inf)))
        donors = [s for s in range(len(SP)) if present[s] and s != recv and I[s] > I[recv]]
        donor_share = float(sum(w[s] for s in donors))
        rows.append(dict(m49=c, milk_t=float(M1.loc[c].sum()), n_species=int(present.sum()), eligible=bool(elig),
                         I_ref=iref, I_cf=icf, reduction_abs=iref - icf,
                         reduction_pct=100 * (iref - icf) / iref if iref > 0 else 0.0,
                         lowest_species=SP[recv], cattle_lowest=(SP[recv] == "cattle"),
                         donor_share=donor_share, saturated=bool(elig and donor_share <= delta + 1e-12),
                         moved=used))
    r = pd.DataFrame(rows)
    return r


def scenario_summary(r):
    wref = (r.milk_t * r.I_ref).sum()
    el = r[r.eligible]
    return dict(
        n_countries=int(len(r)), n_eligible=int(len(el)),
        world_ratio_ref=float(wref / r.milk_t.sum()),
        world_change_pct_production_weighted=float(100 * (r.milk_t * r.reduction_abs).sum() / wref),
        mean_pct_all=float(r.reduction_pct.mean()),
        mean_pct_eligible=float(el.reduction_pct.mean()),
        median_pct_eligible=float(el.reduction_pct.median()),
        q1_pct_eligible=float(el.reduction_pct.quantile(.25)), q3_pct_eligible=float(el.reduction_pct.quantile(.75)),
        n_cattle_lowest=int(el.cattle_lowest.sum()), n_saturated=int(el.saturated.sum()),
        share_of_world_change_top5=float(((r.milk_t * r.reduction_abs).nlargest(5).sum()) / (r.milk_t * r.reduction_abs).sum()))


# ----------------------------------------------------------------------------- main
def main():
    d = build_panel()
    P = d[d.year.isin(YEARS)].copy()
    summary = {}
    names = P.drop_duplicates("m49").set_index("m49").country

    # ---- coverage
    ex = coverage_tables(P)
    cc = P[P.complete]
    summary["n_entities_raw"] = int(P.m49.nunique())
    summary["n_countries_panel"] = int(cc.m49.nunique())
    summary["n_country_species"] = int(cc.drop_duplicates(["m49", "species"]).shape[0])
    summary["n_cells"] = int(len(cc))
    tot_milk = P.groupby("year").milk_t.sum()
    cc_milk = cc.groupby("year").milk_t.sum()
    summary["milk_share_retained_2023_pct"] = float(100 * cc_milk[Y1] / tot_milk[Y1])
    summary["milk_total_reported_2023_Mt"] = float(tot_milk[Y1] / 1e6)
    summary["milk_total_panel_2023_Mt"] = float(cc_milk[Y1] / 1e6)
    summary["ch4_panel_2023_kt_whole"] = float(cc[cc.year == Y1].ch4_whole_kt.sum())
    summary["ch4_panel_2023_kt_milk_allocated"] = float((cc[cc.year == Y1].ch4_whole_kt * cc[cc.year == Y1].delta).sum())
    sp_excl = ex.groupby("species").size().to_dict()
    summary["excluded_country_species_by_species"] = sp_excl
    summary["delta_capped_cells"] = int((cc.delta_raw > 1 + 1e-9).sum())
    cc.to_csv(os.path.join(OUT, "analytical_panel_complete_case.csv"), index=False)

    # ---- descriptive table by species-year
    rows = []
    for (s, y), g in cc.groupby(["species", "year"]):
        rows.append(dict(species=s, year=y, n=len(g), milk_Mt=g.milk_t.sum() / 1e6,
                         ch4_kt=g.ch4_whole_kt.sum(), delta_median=g.delta.median(),
                         I_whole_pooled=g.ch4_whole_kt.sum() / g.milk_t.sum() * G_PER_KG,
                         I_milk_pooled=(g.ch4_whole_kt * g.delta).sum() / g.milk_t.sum() * G_PER_KG,
                         I_whole_median=g.I_whole.median(), I_whole_q1=g.I_whole.quantile(.25), I_whole_q3=g.I_whole.quantile(.75),
                         I_milk_median=g.I_milk.median(), I_milk_q1=g.I_milk.quantile(.25), I_milk_q3=g.I_milk.quantile(.75),
                         I_whole_mean=g.I_whole.mean(), I_milk_mean=g.I_milk.mean()))
    desc = pd.DataFrame(rows)
    desc["species"] = pd.Categorical(desc.species, SP, ordered=True)
    desc = desc.sort_values(["species", "year"])
    desc.to_csv(os.path.join(OUT, "T1_descriptive_by_species_year.csv"), index=False)

    # ---- same descriptive statistics on the unbalanced panel (original convention), to show the composition effect
    ub = P[P.valid_cell].groupby(["species", "year"]).agg(n=("I_whole", "size"), mean_whole=("I_whole", "mean"),
                                                         median_whole=("I_whole", "median"), mean_milk=("I_milk", "mean"),
                                                         median_milk=("I_milk", "median")).reset_index()
    ub.to_csv(os.path.join(OUT, "S_unbalanced_species_year.csv"), index=False)

    # ---- balanced buffalo (and all species) trend
    bal = []
    for s in SP:
        g = cc[cc.species == s]
        for y, gy in g.groupby("year"):
            bal.append(dict(species=s, year=y, n_balanced=len(gy), mean_whole=gy.I_whole.mean(), median_whole=gy.I_whole.median(),
                            pooled_whole=gy.ch4_whole_kt.sum() / gy.milk_t.sum() * G_PER_KG,
                            mean_milk=gy.I_milk.mean(), median_milk=gy.I_milk.median(),
                            pooled_milk=(gy.ch4_whole_kt * gy.delta).sum() / gy.milk_t.sum() * G_PER_KG))
    bal = pd.DataFrame(bal)
    bal.to_csv(os.path.join(OUT, "S_balanced_panel_species_trend.csv"), index=False)
    # buffalo: unbalanced vs balanced, raw frozen-convention (cells with methane, any year-set)
    buf_ub = P[(P.species == "buffalo") & P.milk_t.gt(0) & P.ch4_whole_kt.gt(0)].groupby("year").agg(
        n=("I_whole", "size"), mean_whole=("I_whole", "mean"), median_whole=("I_whole", "median"))
    buf_ub.to_csv(os.path.join(OUT, "S_buffalo_unbalanced_vs_balanced.csv"))

    # ---- Tier 1 diagnostic: ln I vs ln(N/M)
    diag = []
    for s in SP:
        g = cc[cc.species == s].copy()
        for lab, num, icol in [("whole-herd", "stock_gle", "I_whole"), ("milk-allocated", "milk_animals", "I_milk")]:
            if s == "cattle" and lab == "milk-allocated":
                continue
            g["x"] = np.log(g[num] / g.milk_t)
            g["y"] = np.log(g[icol])
            x, y = g.x.values, g.y.values
            b = np.polyfit(x, y, 1)
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            # cluster (country) bootstrap for the slope
            rng = np.random.default_rng(20260101)
            codes = g.m49.unique()
            sl = []
            for _ in range(500):
                pick = rng.choice(codes, len(codes))
                gg = pd.concat([g[g.m49 == c] for c in pick])
                sl.append(np.polyfit(gg.x, gg.y, 1)[0])
            ef = g.ch4_whole_kt / g.stock_gle * 1e6  # kg CH4 per head
            vlnef, vlnn = np.var(np.log(ef)), np.var(g.x)
            diag.append(dict(species=s, boundary=lab, n=len(g), countries=g.m49.nunique(), slope=b[0], slope_ci_lo=np.percentile(sl, 2.5),
                             slope_ci_hi=np.percentile(sl, 97.5), r2=r2, ef_median_kg_head=ef.median(), ef_q1=ef.quantile(.25), ef_q3=ef.quantile(.75),
                             var_ln_ef=vlnef, var_ln_n_over_m=vlnn))
    pd.DataFrame(diag).to_csv(os.path.join(OUT, "T3_tier1_diagnostic.csv"), index=False)
    # within country-species stability of the implied emission factor across years
    cc2 = cc.copy(); cc2["ef"] = cc2.ch4_whole_kt / cc2.stock_gle * 1e6
    cv = cc2.groupby(["m49", "species"]).ef.agg(lambda v: v.std(ddof=0) / v.mean())
    summary["ef_within_series_cv_median_pct"] = float(100 * cv.median())
    summary["ef_within_series_cv_p90_pct"] = float(100 * cv.quantile(.9))

    # ---- Shapley: full panel and influence cases, both boundaries, both estimands
    def run_case(boundary, label, drop=None, fpcm=False):
        M0, I0, M1, I1 = endpoints(d_c, boundary, drop=drop, fpcm=fpcm)
        r2 = shapley2(M0, I0, M1, I1)
        g2 = global2(r2)
        g3 = shapley3(M0, I0, M1, I1)
        return r2, dict(boundary=boundary, case=label, n_countries=g2["n"], two_struct=g2["struct"], two_within=g2["within"], two_net=g2["net"],
                        world_R0=g3["R0"], world_R1=g3["R1"], world_delta=g3["delta"], three_country_shares=g3["country_shares"],
                        three_species_mix=g3["species_mix"], three_species_intensity=g3["species_intensity"])
    d_c = P  # includes 'complete' flag used inside wide()

    mon = int(P[P.country == "Mongolia"].m49.iloc[0])
    cases = []
    country_tables = {}
    for b in ["whole", "milk"]:
        r2, row = run_case(b, "full panel")
        cases.append(row)
        country_tables[b] = r2
        # ranking by |structural contribution| (weight x struct)
        contrib = (r2.wt / r2.wt.sum() * r2.struct).abs().sort_values(ascending=False)
        top = list(contrib.index)
        row_top = contrib.head(5).index.tolist()
        cases.append(run_case(b, "Mongolia sheep series excluded", drop=lambda x: (x.m49 == mon) & (x.species == "sheep"))[1])
        cases.append(run_case(b, "Mongolia excluded", drop=lambda x: x.m49 == mon)[1])
        cases.append(run_case(b, "five largest structural contributors excluded", drop=lambda x, t=row_top: x.m49.isin(t))[1])
        cases.append(run_case(b, "ten largest structural contributors excluded", drop=lambda x, t=top[:10]: x.m49.isin(t))[1])
        # rule-based cell screening (denominator-collapse rule from the original submission)
        thr = P[P.valid_cell].groupby("species").I_whole.quantile(.99)
        P["milk_valid"] = P.milk_t.where(P.valid_cell)
        mx = P.groupby(["m49", "species"]).milk_valid.transform("max")
        P["_flag"] = P.valid_cell & (P.I_whole > P.species.map(thr)) & (P.milk_t < 0.25 * mx)
        flagged = P[P._flag & P.complete]
        fl_pairs = set(zip(flagged.m49, flagged.species))
        cases.append(run_case(b, "denominator-collapse cells screened (series removed)",
                              drop=lambda x, f=fl_pairs: pd.Series([(m, s) in f for m, s in zip(x.m49, x.species)], index=x.index))[1])
        summary["denominator_collapse_cells"] = [dict(country=names[m], species=s) for m, s in sorted(fl_pairs)]
        # leave-one-country-out
        loo = []
        for c in country_tables[b].index:
            _, rr = run_case(b, "loo", drop=lambda x, c=c: x.m49 == c)
            loo.append(dict(boundary=b, m49=c, country=names[c], two_struct=rr["two_struct"], two_within=rr["two_within"],
                            three_species_mix=rr["three_species_mix"], world_delta=rr["world_delta"]))
        pd.DataFrame(loo).to_csv(os.path.join(OUT, f"S_leave_one_country_out_{b}.csv"), index=False)
    cases_gle = run_case("milk_gle", "milk-allocated, GLE stock as TS")[1]
    cases_gle["boundary"] = "milk"
    cases.append(cases_gle)
    cases.append({**run_case("milk_gle", "milk-allocated, GLE stock as TS, Mongolia sheep excluded", drop=lambda x: (x.m49 == mon) & (x.species == "sheep"))[1], "boundary": "milk"})
    # functional-unit sensitivity (FPCM) on the milk-allocated boundary
    cases.append(run_case("milk", "FPCM functional unit (milk-allocated)", fpcm=True)[1])
    cases.append(run_case("milk", "FPCM functional unit, Mongolia sheep excluded", fpcm=True,
                          drop=lambda x: (x.m49 == mon) & (x.species == "sheep"))[1])
    cases = pd.DataFrame(cases)
    cases.to_csv(os.path.join(OUT, "T4_shapley_cases.csv"), index=False)

    # country-level Shapley tables
    for b in ["whole", "milk"]:
        t = country_tables[b].copy()
        t.insert(0, "country", [names[m] for m in t.index])
        t.to_csv(os.path.join(OUT, f"S_shapley_country_{b}.csv"))

    # ---- scenario (closed form on observed intensities), main + budget grid + FPCM + Mongolia
    grid = []
    sc = {}
    for b in ["whole", "milk"]:
        for dl in [0.01, 0.05, 0.10, 0.20]:
            r = scenario(d_c, b, dl)
            s = scenario_summary(r); s.update(boundary=b, delta=dl, case="all data")
            grid.append(s)
            if dl == 0.10:
                sc[b] = r
        r = scenario(d_c, b, 0.10, drop=lambda x: (x.m49 == mon) & (x.species == "sheep"))
        s = scenario_summary(r); s.update(boundary=b, delta=0.10, case="Mongolia sheep excluded"); grid.append(s)
    r = scenario(d_c, "milk", 0.10, fpcm=True)
    s = scenario_summary(r); s.update(boundary="milk", delta=0.10, case="FPCM"); grid.append(s)
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(OUT, "T5_counterfactual_summary.csv"), index=False)
    for b in ["whole", "milk"]:
        t = sc[b].copy(); t.insert(0, "country", [names[m] for m in t.m49])
        t.to_csv(os.path.join(OUT, f"S_counterfactual_country_{b}.csv"), index=False)
    # agreement closed form vs LP
    M1, I1 = wide(d_c, "milk", Y1)
    W1 = M1.div(M1.sum(1), axis=0)
    mx = 0.0
    for b in ["whole", "milk"]:
        M1, I1 = wide(d_c, b, Y1); W1 = M1.div(M1.sum(1), axis=0)
        for c in M1.index:
            w, I = W1.loc[c].values, I1.loc[c].values
            wa, _, _ = counterfactual_country(w, I, 0.10)
            wb = counterfactual_lp(w, I, 0.10)
            mx = max(mx, abs((wa * I).sum() - (wb * I).sum()))
    summary["closed_form_vs_lp_max_abs_diff_objective"] = mx

    # direction of the counterfactual: lowest species by country under both boundaries
    both = sc["whole"][["m49", "lowest_species", "eligible"]].merge(sc["milk"][["m49", "lowest_species"]], on="m49", suffixes=("_whole", "_milk"))
    both = both[both.eligible]
    summary["n_multispecies_countries"] = int(len(both))
    summary["cattle_lowest_whole"] = int((both.lowest_species_whole == "cattle").sum())
    summary["cattle_lowest_milk"] = int((both.lowest_species_milk == "cattle").sum())
    summary["lowest_species_milk_counts"] = both.lowest_species_milk.value_counts().to_dict()

    # ---- country table 2023 (all countries): intensities both boundaries and species detail
    c23 = cc[cc.year == Y1].copy()
    nat = c23.groupby("m49").apply(lambda g: pd.Series(dict(
        country=g.country.iloc[0], milk_t=g.milk_t.sum(), species_n=len(g),
        I_nat_whole=g.ch4_whole_kt.sum() / g.milk_t.sum() * G_PER_KG,
        I_nat_milk=(g.ch4_whole_kt * g.delta).sum() / g.milk_t.sum() * G_PER_KG,
        species=",".join(sorted(g.species)),
        I_cattle=g[g.species == "cattle"].I_whole.sum() if (g.species == "cattle").any() else np.nan)), include_groups=False)
    nat.to_csv(os.path.join(OUT, "S_country_national_intensity_2023.csv"))
    wide_cells = c23.pivot_table(index=["m49", "country"], columns="species",
                                 values=["I_whole", "I_milk", "delta", "milk_t"], aggfunc="first")
    wide_cells.columns = [f"{a}_{b}" for a, b in wide_cells.columns]
    wide_cells.to_csv(os.path.join(OUT, "S_country_species_intensity_2023.csv"))

    # low-end checks requested by the reviewer
    low = []
    for cn in ["Israel", "Cyprus", "Republic of Korea"]:
        g = P[(P.country == cn) & (P.year == Y1)]
        for _, r in g.iterrows():
            low.append(dict(country=cn, species=r.species, in_panel=bool(r.complete), milk_t=r.milk_t, methane_kt=r.ch4_whole_kt,
                            milk_animals=r.milk_animals, stock=r.stock_gle,
                            ef_kg_head=(r.ch4_whole_kt / r.stock_gle * 1e6) if r.stock_gle > 0 else np.nan,
                            yield_t_per_milk_animal=(r.milk_t / r.milk_animals) if r.milk_animals > 0 else np.nan, I_whole=r.I_whole, I_milk=r.I_milk))
    pd.DataFrame(low).to_csv(os.path.join(OUT, "S_low_end_countries_2023.csv"), index=False)

    # Mongolia audit table
    mg = P[P.country == "Mongolia"][["species", "year", "milk_t", "milk_animals", "stock_gle", "ch4_whole_kt", "flag_milk_t",
                                     "flag_milk_animals", "note_milk_t", "I_whole", "I_milk", "delta"]]
    mg.to_csv(os.path.join(OUT, "S_mongolia_audit.csv"), index=False)

    # ---- world aggregates
    for b, col in [("whole", "I_whole"), ("milk", "I_milk")]:
        for y in YEARS:
            g = cc[cc.year == y]
            ch = (g.ch4_whole_kt * (g.delta if b == "milk" else 1)).sum()
            summary[f"world_ratio_{b}_{y}"] = float(ch / g.milk_t.sum() * G_PER_KG)

    # ---- accounting identity checks on the analytical panel (2 identities x 2 boundaries x country-years: shares sum to one; mixture identity)
    chk = []
    for b, col in [("whole", "I_whole"), ("milk", "I_milk")]:
        for (m, y), g in cc.groupby(["m49", "year"]):
            w = g.milk_t / g.milk_t.sum()
            nat_direct = (g.ch4_whole_kt * (g.delta if b == "milk" else 1)).sum() / g.milk_t.sum() * G_PER_KG
            chk.append(dict(boundary=b, m49=m, year=y, share_sum_err=abs(w.sum() - 1),
                            mixture_err=abs((w * g[col]).sum() - nat_direct)))
    chk = pd.DataFrame(chk)
    summary["accounting_checks_country_years"] = int(chk[chk.boundary == "whole"].shape[0])
    summary["accounting_checks_total"] = int(2 * chk.shape[0])
    summary["accounting_checks_max_share_err"] = float(chk.share_sum_err.max())
    summary["accounting_checks_max_mixture_err_g_per_kg"] = float(chk.mixture_err.max())
    json.dump(summary, open(os.path.join(OUT, "results_summary.json"), "w"), indent=2, default=str)
    print(json.dumps(summary, indent=1, default=str)[:3000])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(cases.round(3).to_string())
    print(grid.round(3).to_string())


if __name__ == "__main__":
    main()
