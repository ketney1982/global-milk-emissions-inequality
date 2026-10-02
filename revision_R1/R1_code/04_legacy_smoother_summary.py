#!/usr/bin/env python3
"""04_legacy_smoother_summary.py - posterior/observed ratios of the ORIGINAL hierarchical smoother (Supplement S6).
Reads the deposited outputs of the original submission (portfolio_results_corrected.csv; outputs_R2/ in this repository)."""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, "..", "..", "outputs_R2", "portfolio_results_corrected.csv"),
              os.path.join(HERE, "..", "..", "evidence", "04_corrected_outputs", "portfolio_results_corrected.csv")]
src = next(c for c in CANDIDATES if os.path.exists(c))
d = pd.read_csv(src)
d["ratio"] = d.baseline_posterior / d.baseline_observed
d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
out = os.path.join(HERE, "..", "R1_results")
top = d.sort_values("production_tonnes", ascending=False).head(20)[["country", "production_tonnes", "baseline_observed", "baseline_posterior", "ratio"]].copy()
top["baseline_observed"] *= 1000; top["baseline_posterior"] *= 1000   # g CH4 / kg
top.to_csv(os.path.join(out, "S_legacy_smoother_top20.csv"), index=False)
sm = dict(n=len(d), median_ratio=float(d.ratio.median()), share_within_25pct=float(((d.ratio > .75) & (d.ratio < 1.25)).mean()),
          n_top20_outside_25pct=int(((top.ratio < .75) | (top.ratio > 1.25)).sum()), max_ratio=float(d.ratio.max()), min_ratio=float(d.ratio.min()))
pd.Series(sm).to_csv(os.path.join(out, "S_legacy_smoother_summary.csv"))
print(sm); print(top.round(2).to_string())
