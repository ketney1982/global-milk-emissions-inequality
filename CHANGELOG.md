# Changelog

## R3 - 3 September 2026

Reproducibility release. R2 corrected the science; R3 makes the **public entry point
regenerate what the manuscript reports**. No change to the input data, no re-run of the
MCMC, and no change to any reported number except one rounding fix (8.46 -> 8.45 Mt CH4).

### Corrected: `run-all` did not reproduce the reported sensitivity grid

The manuscript and Supplementary Table S1 report a 4 x 3 x 3 factorial grid evaluated on
the full panel: 181 countries x 36 configurations = **6,516 solves**. The R2 deposit
contained exactly that. But `run_sensitivity_grid` still carried

```python
n_countries_max: int = 20        # "top-producing countries", for runtime
```

and neither `cmd_run_all` nor `scripts/reproduce.py` overrode it, so the public entry
point emitted a **720-row** grid over 20 countries. The auto-generated methods appendix
said so in writing ("computed on the top 20 producers only"), contradicting the paper.

`n_countries_max` now defaults to `None`, meaning the whole analysed panel. The runtime
argument no longer holds: under the exact LP one full-panel solve takes ~1.5 s and the
entire grid ~12 s. The top-20 restriction survives as a derived summary
(`sensitivity_summary_top20.csv`) for comparability with R1, and as the explicit
`--sensitivity-countries N` flag. `run_sensitivity_grid` now also writes the
per-configuration digests that are Table 8.

### Corrected: `absolute_reduction_kt` was the wrong quantity in the wrong unit

R1 and R2 exported one column:

```python
absolute_reduction_kt = (baseline - optimized_mean) * production_tonnes / 1e6
```

Two defects. The `/1e6` divisor returns **megatonnes** of CH4, not kilotonnes. And the
quantity is the *posterior-scaled* difference, while the manuscript reports the
*inventory-scaled* accounting reduction - the posterior percentage applied to the
observed inventory, `E_obs * R / 100`. These are not the same number, because the
posterior reference is not the observed aggregate ratio: for Ethiopia, 0.509 against
0.541 Mt CH4; across the panel, **10.73 against 8.45 Mt CH4**. `tables.py` ranked
countries by the column the manuscript does not report.

Both quantities are now exported under explicit names, with the inventory they are built
from:

| Column | Definition | Units |
| --- | --- | --- |
| `observed_ch4_t` | reported methane charged to milk in the reference year | t CH4 |
| `abs_reduction_mt_ch4` | `observed_ch4_t * reduction_mean_pct / 100` - **the reported quantity** | Mt CH4 |
| `abs_reduction_mt_ch4_posterior` | `(baseline_intensity - optimized_mean) * production_tonnes` | Mt CH4 |

with `raw_*` counterparts. Rankings and Figure 5 use `abs_reduction_mt_ch4`. The
ambiguous column is removed rather than aliased, so nothing can silently keep reading it.

### Corrected: `u_c_raw` convergence was never assessed

`_compute_diagnostics` defined the directly sampled parameters as `alpha_s, beta_s,
gamma_s, sigma_s, nu` and handled `tau` separately. But the model also samples

```python
u_c_raw = pm.ZeroSumNormal(...)
u_c = tau * u_c_raw
```

and `u_c_raw` was in neither tier, so the claim that every directly sampled parameter
converged had not been checked against it. Computed from the deposited posterior, it
fails by exactly the same margin as `tau` - max R-hat 1.189, min ESS_bulk 65, min
ESS_tail 189, with 173 of 182 components at R-hat >= 1.01. That is expected: under a
non-centred parameterisation `tau` and `u_c_raw` are only *jointly* identified.

The diagnostic that matters downstream is their product, and it was never reported. It is
now:

| Variable | max R-hat | min ESS bulk | min ESS tail | Verdict |
| --- | --- | --- | --- | --- |
| `tau` | 1.190 | 65 | 189 | fails strict and relaxed |
| `u_c_raw` (182 levels) | 1.189 | 65 | 189 | fails strict and relaxed |
| **`u_c = tau * u_c_raw`** (182 levels) | **1.007** | **2,551** | **994** | **passes strict** |

`u_c` is what enters the linear predictor, so the country-level posterior intensities
used as optimisation scenarios are well mixed. What remains genuinely uncertain is the
magnitude of between-country dispersion, which is not interpreted as a substantive result
anywhere. `bayes_diagnostics.json` gains `u_c_raw`, `u_c` and `country_effects_converged`
blocks.

Threshold arithmetic now runs on unrounded summaries. `az.summary` rounds to 2 decimals
by default, which turns 1.00672 into 1.01 (a spurious failure) and 1.00155 into 1.00 (a
spurious pass) at a threshold of 1.01.

### Added: the posterior draws the analysis actually consumes

Everything downstream of the MCMC depends on the posterior only through a
`(500, 182, 5)` array of latent intensities. That array is now written by the bayes stage
and deposited as `outputs_R3/posterior_intensity_draws.npz` - 3.5 MB against 1.6 GB for
the full archive - so the optimisation, the grid, the uncertainty propagation and every
figure and table are exactly reproducible without the archive and without re-sampling.
`--posterior-draws` selects it; `run-all --skip-bayes` finds it automatically.

### Corrected: `optimize` silently ignored the posterior

`cmd_optimize` loaded the posterior only under `--allow-expansion`. A bare
`methane-portfolio optimize` therefore ran on lognormal fallback scenarios and did not
reproduce the reported optimisation, without saying so. The posterior is now used
whenever it is available, and its absence is reported as a warning that names the
consequence.

### Fixed: `run-all` crashed on current matplotlib

`fig6_elasticity` called `ax.boxplot(..., labels=...)`. Matplotlib renamed that argument
to `tick_labels` in 3.9 and removed the old spelling in 3.10, so step 7 of the pipeline
raised `TypeError` on any current install. The spelling is now resolved from the running
version rather than by pinning matplotlib.

### Corrected: two claims stated more strongly than the evidence

* **`lp_optimize` docstring** said the LP needs no "weight renormalisation". It does clip
  at zero and divide by the sum. That is floating-point housekeeping of order 1e-16, not
  a feasibility repair, and it is now described as such rather than denied.
* **README** said slow `tau` mixing "does not affect downstream results". That was
  asserted, not demonstrated, and it was stronger than the manuscript. It is replaced by
  the `u_c` evidence above.

### Documented: the one exception to (lambda, alpha) invariance

The optimal composition is invariant to `lambda` and `alpha` for 180 of the 181 countries
at every budget, to within 1e-15. There is exactly one exception in the whole 6,516-cell
grid: Indonesia at `delta` = 0.20 under `lambda` = 0.20, `alpha` = 0.95, where the LP
selects a different vertex of the budget face, retaining 2.24 pp of buffalo instead of
moving it to goats. Checked against the exact objective, the alternative is genuinely
optimal for that configuration - better by 2.4e-05, worse under the other eight - so it
is a real, if tiny, effect of the risk parameters and not a numerical tie. It moves the
panel mean by 0.0025 percentage points, 15.9230% -> 15.9205%.

### Added: scope regression tests

`tests/test_manuscript_scope.py` fails if the grid stops covering the panel, if the
ambiguous column returns, or if either absolute-reduction definition drifts.
`scripts/reproduce.py` records the same checks in the manifest, alongside input
checksums, and prints a verdict.

### Effect on the reported results

None, except one rounding correction. Regenerated from the public entry point, every
shared quantity matches the R2 deposit to machine precision: the portfolio results
bit-for-bit, the grid to 9e-16 on the absolute reduction and exactly elsewhere. The
manuscript figure of 8.46 Mt CH4 was a rounding error for 8.4546 and reads **8.45**.

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
