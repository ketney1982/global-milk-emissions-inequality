#!/usr/bin/env python3
"""
12_longer_window.py - sensitivity of the aggregate decomposition to the length of the study window (Reviewer 3, R3.4).

Re-uses the extraction rules of 01_extract_faostat.py and the indicators, complete-case rule and exact Shapley
decompositions of 02_analysis.py unchanged; only the years change. For each window the three-factor decomposition of the
change in the world aggregate ratio and the two-factor (fixed-country-weight) decomposition are computed under both
boundaries, for the full panel and with the series flagged by the denominator-collapse rule removed.

Usage: python 12_longer_window.py [path_to_bulk_dir]      (same bulk files as 01_extract_faostat.py)
Output: ../R1_results/S_longer_window_decomposition.csv, S_longer_window_world_ratio.csv
The 2020-2023 window must reproduce Table 4 of the manuscript (check printed at the end).
"""
import sys, os, importlib.util, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BULK = sys.argv[1] if len(sys.argv) > 1 else "faostat_bulk"
OUT = os.path.join(HERE, "..", "R1_results")


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("0", "m", 1), os.path.join(HERE, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ex, an = load("01_extract_faostat.py"), load("02_analysis.py")
WINDOWS = [(2020, 2023), (2015, 2023), (2010, 2023)]
rows, world = [], []
for y0, y1 in WINDOWS:
    tmp = tempfile.mkdtemp()
    ex.BULK, ex.OUT, ex.YEARS = BULK, tmp, range(y0, y1 + 1)
    ex.main()                                    # writes faostat_panel_2020_2023_long.csv (all years of the window) to tmp
    an.DATA, an.Y0, an.Y1, an.YEARS = tmp, y0, y1, list(range(y0, y1 + 1))
    d = an.build_panel()
    P = d[d.year.isin(an.YEARS)].copy()
    cc = P[P.complete]
    mon = P[P.country == "Mongolia"].m49.iloc[0]
    # denominator-collapse rule exactly as in 02_analysis.py
    thr = P[P.valid_cell].groupby("species").I_whole.quantile(.99)
    P["milk_valid"] = P.milk_t.where(P.valid_cell)
    mx = P.groupby(["m49", "species"]).milk_valid.transform("max")
    P["_flag"] = P.valid_cell & (P.I_whole > P.species.map(thr)) & (P.milk_t < 0.25 * mx)
    fl = P[P._flag & P.complete]
    fl_pairs = set(zip(fl.m49, fl.species))
    names = P.drop_duplicates("m49").set_index("m49").country
    flagged_txt = "; ".join(f"{names[m]} {s}" for m, s in sorted(fl_pairs)) or "none"
    tot1 = P[P.year == y1].milk_t.sum()
    for b in ("whole", "milk"):
        for y in an.YEARS:
            g = cc[cc.year == y]
            ch = (g.ch4_whole_kt * (g.delta if b == "milk" else 1)).sum()
            world.append(dict(window=f"{y0}-{y1}", boundary=b, year=y, world_ratio=ch / g.milk_t.sum() * an.G_PER_KG,
                              n_countries=g.m49.nunique()))
        for label, drop in [("full panel", None),
                            ("flagged series removed", (lambda x, f=fl_pairs: pd.Series([(m, s) in f for m, s in zip(x.m49, x.species)], index=x.index)))]:
            M0, I0, M1, I1 = an.endpoints(P, b, drop=drop)
            g2 = an.global2(an.shapley2(M0, I0, M1, I1))
            g3 = an.shapley3(M0, I0, M1, I1)
            rows.append(dict(window=f"{y0}-{y1}", boundary=b, case=label, n_countries=g2["n"], panel_milk_share_end_year_pct=100 * cc[cc.year == y1].milk_t.sum() / tot1,
                             world_R0=g3["R0"], world_R1=g3["R1"], world_delta=g3["delta"], three_country_shares=g3["country_shares"],
                             three_species_mix=g3["species_mix"], three_species_intensity=g3["species_intensity"],
                             two_struct=g2["struct"], two_within=g2["within"], two_net=g2["net"], flagged=flagged_txt))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "S_longer_window_decomposition.csv"), index=False)
pd.DataFrame(world).to_csv(os.path.join(OUT, "S_longer_window_world_ratio.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
print(res.round(3).to_string())
# check against Table 4 of the manuscript (T4_shapley_cases.csv): window 2020-2023
t4 = pd.read_csv(os.path.join(OUT, "T4_shapley_cases.csv"))
a = t4[(t4.case == "full panel") & (t4.boundary == "milk")].iloc[0]
b = res[(res.window == "2020-2023") & (res.boundary == "milk") & (res.case == "full panel")].iloc[0]
print("check vs Table 4 (milk, full panel): max abs diff =",
      max(abs(a.three_country_shares - b.three_country_shares), abs(a.three_species_mix - b.three_species_mix), abs(a.three_species_intensity - b.three_species_intensity),
          abs(a.two_struct - b.two_struct), abs(a.two_within - b.two_within)))
