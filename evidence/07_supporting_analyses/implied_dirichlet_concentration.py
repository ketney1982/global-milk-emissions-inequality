"""Method-of-moments concentration implied by the observed 2020-2023 variability of
national milk-species shares.  For a Dirichlet with mean w and concentration kappa,
Var(w_s) = w_s (1 - w_s) / (kappa + 1), so kappa = mean_s[w_s(1-w_s)/Var_s] - 1."""
import numpy as np, pandas as pd, sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sh = pd.read_csv(os.path.join(_ROOT, "data",
    "cercetare-485010.faostat_clean.milk_species_structure.csv"))
sh = sh[sh.country_m49 != 159].copy()
sh["sp"] = sh.milk_species.str.replace("Raw milk of ", "", regex=False)
ks = []
for cm, g in sh.groupby("country_m49"):
    piv = g.pivot_table(index="year", columns="sp", values="species_share").fillna(0.0)
    if piv.shape[0] < 2:
        continue
    act = piv.columns[(piv > 1e-12).any()]
    if len(act) < 2:
        continue
    num, den = [], []
    for s in act:
        w = piv[s].mean(); v = piv[s].var(ddof=1)
        if v > 0 and 0 < w < 1:
            num.append(w * (1 - w)); den.append(v)
    if not num:
        continue
    k = float(np.mean(np.array(num) / np.array(den))) - 1.0
    if np.isfinite(k) and k > 0:
        ks.append((cm, k))
k = pd.Series(dict(ks))
print("countries with an identifiable implied concentration:", len(k))
for q in (0.10, 0.25, 0.50, 0.75, 0.90):
    print("  p%-3d %12.0f" % (q * 100, k.quantile(q)))
pd.DataFrame({"country_m49": k.index, "implied_kappa": k.values}).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "implied_dirichlet_concentration.csv"), index=False)
print("written")
