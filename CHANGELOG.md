# Changelog

## R2 — 2 September 2026

The release that reproduces the revised manuscript. Two corrections to the optimisation
layer, one correction to output reporting, and documentation of a unit and an allocation
boundary that were previously implicit. **No change to the input data, and no re-run of
the MCMC**: the corrected optimisation reads the posterior saved by R1.

### Corrected: reference estimand of the portfolio analysis

R1 computed the reference intensity from the **observed** species intensities while
evaluating the optimum on Bayesian **posterior** scenarios:

```python
baseline = w_ref @ i_obs                     # observed
mean_opt = mean_k(I_scen[k] @ w_opt)         # posterior
```

The two sides of every reported percentage therefore came from different estimands, and
hierarchical shrinkage entered the result as apparent mitigation. For China, mainland the
posterior reference is 46.1 against an observed 100.1 g CH4/kg, which is how a
1-percentage-point reallocation budget appeared to deliver a 57% reduction when the
arithmetic ceiling on observed data is 18.1%.

The reference is now `mean_k(I_scen[k] @ w_ref)`, from the same scenarios that evaluate
the optimum. Controlled by `posterior_baseline=True` (default); set `False` for the R1
behaviour.

### Corrected: the problem is an exact linear program

Under the Rockafellar–Uryasev representation of CVaR the problem is linear once the L1
budget is expressed through auxiliary deviation variables. R1 solved it with SLSQP
initialised at `x0 = (w_ref, 0)` — precisely the non-differentiable kink of the L1
constraint. New module `methane_portfolio/lp_optimize.py` builds the LP and solves it
with HiGHS through `scipy.optimize.linprog`.

| | R1 (SLSQP → trust-constr) | R2 (HiGHS LP) |
| --- | --- | --- |
| certified optima, multi-species countries | 98 / 107 | **107 / 107** |
| strictly better objective found by the LP | — | **22 countries** (up to 40.4%) |
| LP worse anywhere | — | **0** |
| wall time, 181 countries | ~285 s | **~2 s** |

Controlled by `use_lp=True` (default). The iteration-limit fallbacks, the post-solver
weight renormalisation and the TV re-projection are no longer needed.

### Corrected: `raw_*` columns

R1 exported `sol_raw_report = sol_final if do_no_harm else sol_raw`, so with the guard
enabled — the default — the `raw_*` columns were written from the **guarded** solution.
In the published R1 output they were byte-identical to the unprefixed columns in all 181
rows while 31 rows had in fact been guard-adjusted, contradicting the README's
description of them as unmodified solver output. `raw_*` now always carries the
unguarded solution.

### Consequences for the reported results

| | R1 | R2 |
| --- | --- | --- |
| mean national reduction | 12.61% | **11.91%** |
| median national reduction | 0.00% | **2.38%** |
| countries with a positive reduction | 77 | **107** (every multi-species system) |
| countries with none | — | **74** (all single-species) |
| certified optima | 178 / 181 | **181 / 181** |
| do-no-harm guard applied | 31 | **0** |
| top-5 share of the total | 69% | **56.9%** |

The guard no longer binding is a consequence of the estimand repair, not a loosening of
the constraint: the reference mix is always feasible under the budget, so a certified
optimum of a convex objective cannot return a worse mean. The guard is retained as an
explicit safeguard and is exercised in the test suite under the R1 configuration.

Two further properties, both now reported in the manuscript:

- `delta` is the only parameter that changes the result — 2.44% → 8.08% → 11.91% →
  15.92% for `delta` = 0.01/0.05/0.10/0.20, with **no saturation** on the full panel. The
  median saturates at `delta` = 0.05. R1's "rapid saturation above 10%" came from the
  20-largest-producer subset.
- The optimal composition is invariant to `lambda` and `alpha` to within 1e-16 in every
  weight, because the species ordering by posterior mean intensity coincides with the
  ordering by upper-tail intensity. The risk term is a robustness check in this panel,
  not a distinct decision rule.

### Documented: units

The column `kg_co2e_per_ton_milk` is a **misnomer**, retained for compatibility with the
R1 artefacts. It holds

```
CH4 (kt) * 1000 / milk (t)  =  t CH4 / t milk  =  kg CH4 / kg raw milk
```

Verified against the live FAOSTAT API: the numerator is domain GLE, elements 72254
(enteric fermentation) plus 72256 (manure management), reported in **kilotonnes of CH4
mass**, with **no GWP conversion applied**. Five country-species reconstructions match
the archived values exactly. The manuscript reports **g CH4 per kg raw milk**, i.e. these
values × 1000; absolute reductions are **Mt CH4**.

### Documented: allocation boundary for non-bovine species

FAOSTAT resolves cattle into `Cattle, dairy` and `Cattle, non-dairy`, so the cattle
numerator is already the dairy herd. It provides no such split for buffalo, goats, sheep
or camels, so for those four species the numerator is the **whole national herd's**
methane while the denominator is milk alone. FAO's own *Emissions intensities* domain
corrects for this with a producing-animals / total-animals scaling that this pipeline
deliberately does not apply. The species ordering is therefore not a biological
efficiency ranking.

### Other

- New: `tests/test_lp_optimize.py` — 11 tests including a brute-force simplex check and
  the "LP never worse than the nonlinear solver" property.
- Updated: the two do-no-harm regression tests now pin `posterior_baseline=False`, since
  their fixtures were built around the reference/optimum scale mismatch that R2 removes.
- New result columns: `baseline_intensity_observed`, `baseline_cvar`,
  `reference_is_posterior`, `solver`, `lp_certified`, `tv_distance`.
- Reported CVaR, for both reference and optimum, is now the empirical CVaR at the
  respective alpha-quantile. R1 used the solver's auxiliary threshold variable for the
  optimum, which produced a −3,210% CVaR reduction for Niger.
- Contact address harmonised to `ketney.otto@ulbsibiu.ro` across all modules, matching
  the manuscript.

## R1 — 18 February 2026

Initial release accompanying the Research Square preprint
(<https://doi.org/10.21203/rs.3.rs-9426691/v1>).
