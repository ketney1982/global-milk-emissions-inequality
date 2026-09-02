# Decoupling Methane from Milk

**Bayesian Species Portfolio Optimization under Structural Constraints (2020–2023)**

**Author:** Ketney Otto
**Affiliation:** „Lucian Blaga" University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Rațiu Street, no. 7-9, 550012 Sibiu, Romania
**Contact:** ketney.otto@ulbsibiu.ro | [ORCID 0000-0003-1638-1154](https://orcid.org/0000-0003-1638-1154)

## Overview

This repository implements a reproducible research pipeline to:

1. **Validate** the accounting identity $ I_{ct} = \sum_s w_{cts} \cdot I_{cts} $
2. **Decompose** changes in emission intensity via exact two-factor Shapley values
3. **Fit** a Bayesian hierarchical model with species-level partial pooling and regime shift
4. **Optimize** species portfolios via robust (CVaR) optimization with feasibility constraints
5. **Propagate** uncertainty using posterior draws + Dirichlet share perturbation
6. **Generate** publication-quality figures and tables
7. **Auto-write** a Methods Appendix

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install
pip install -e ".[dev]"

# Run the full pipeline without Bayesian model fitting (fast path)
methane-portfolio run-all --skip-bayes
# Optional: allow expansion into species with baseline zero share
methane-portfolio run-all --allow-expansion

# Or step by step
methane-portfolio validate
methane-portfolio shapley
methane-portfolio optimize --delta 0.10
# Optional: allow expansion into species with baseline zero share
methane-portfolio optimize --delta 0.10 --allow-expansion
methane-portfolio uncertainty
methane-portfolio figures
methane-portfolio tables
methane-portfolio report

# Bayesian model (requires PyMC + CPU time)
# Tip: Increase --chains to match your CPU core count for faster parallel processing
methane-portfolio bayes --chains 16 --draws 8000 --tune 15000

# Reproducibility manifest (runs pipeline + writes checksums)
python scripts/reproduce.py --skip-bayes
```

## Project Structure

```
├── data/                              # Input CSVs (FAOSTAT)
├── methane_portfolio/                 # Main package
│   ├── __init__.py
│   ├── config.py                      # Central configuration
│   ├── io.py                          # Data loading + deduplication
│   ├── validate.py                    # Accounting identity checks
│   ├── utils.py                       # Shared utilities
│   ├── shapley.py                     # Exact 2-factor Shapley decomposition
│   ├── bayes.py                       # Bayesian hierarchical model (PyMC)
│   ├── optimize.py                    # Base optimization helpers
│   ├── robust_optimize.py             # CVaR robust portfolio optimization
│   ├── uncertainty.py                 # Dirichlet + posterior propagation
│   ├── figures.py                     # 6 Nature-style figures
│   ├── tables.py                      # 5 manuscript tables
│   ├── report.py                      # Methods appendix generator
│   └── cli.py                         # Command-line interface
├── tests/                             # pytest unit tests
├── scripts/                           # Reproducibility & integrity scripts
│   ├── reproduce.py                   # Pipeline execution + checksum manifest
│   ├── editorial_integrity_check.py   # Raw vs final audit validation
│   └── prepublish_check.py            # Pre-publication hygiene scan
├── outputs/                           # Generated outputs
│   ├── figures/                       # Generated figures (PNG, 600 DPI)
│   ├── *.csv                          # Result tables
│   ├── *.json                         # Diagnostics, audits, manifests
│   └── methods_appendix.md            # Auto-generated appendix
├── pyproject.toml
└── README.md
```

## Key Scientific Features

- **Structural-consistent inference**: National intensity modelled as mixture $I_{ct} = \sum w_{cts} I_{cts}$
- **Bayesian partial pooling + regime shift**: Student-t likelihood with $\gamma_s \cdot \mathbf{1}[t \geq 2022]$
- **Robust portfolio optimization**: Minimizes $\lambda E[I] + (1-\lambda) \text{CVaR}_\alpha(I)$ via Rockafellar-Uryasev
- **Editorial traceability**: `robust_optimization_results.csv` keeps both raw (`raw_*`) and guarded outputs, with `robust_optimization_audit.json` documenting every do-no-harm intervention
- **Causal guardrails**: All results are labelled as accounting counterfactuals; causal language triggers errors

---

## Data Integrity & Transparency

This section documents all data processing decisions to ensure full transparency and reproducibility.  No data points are removed, altered, or imputed without explicit documentation and justification.

### FAOSTAT China Deduplication

FAOSTAT reports both **"China" (M49 = 159)** — an aggregate entry — and **"China, mainland" (M49 = 156)** — the underlying disaggregated data.  Retaining both would double-count China's milk production and emissions in all global totals.

**Decision:** The aggregate "China" entry (M49 = 159) is excluded.  "China, mainland" (M49 = 156) is retained as the authoritative record.

- **Implementation:** `methane_portfolio/io.py`, functions `build_long_df()` and `build_agg_df()`
- **Logging:** Every run logs the exact number of rows dropped (e.g., `"Dropped 20 rows for 'China' aggregate (m49=159)"`)
- **Justification:** This is standard practice in FAOSTAT-based research to avoid double-counting of national aggregates

### No Imputation or Post-Hoc Data Alteration

- All input data are used as-is from the FAOSTAT source CSVs.  No values are imputed, interpolated, or manually overridden.
- Rows with zero intensity (`kg_co2e_per_ton_milk = 0`) or zero species share (`species_share = 0`) are filtered in the Bayesian model because the log-transform `y = log(I)` is undefined at zero.  These exclusions are inherent to the model specification, not ad-hoc data cleaning.

---

## Bayesian Model Convergence Protocol

Convergence is assessed in **two tiers**, following best practices for hierarchical Bayesian models:

### Tier 1 — Direct Parameters (Pipeline-Blocking)

Parameters that directly enter the linear predictor and affect downstream intensity estimates:

| Parameter | Description |
|-----------|-------------|
| `alpha_s` | Species intercepts |
| `beta_s`  | Species time slopes |
| `gamma_s` | Species regime-shift effects (post-2022) |
| `sigma_s` | Species-level observation noise |
| `nu`      | Student-t degrees of freedom |

**Strict thresholds:** R-hat < 1.01, ESS ≥ 400
**Relaxed thresholds (minimum acceptable):** R-hat < 1.10, ESS ≥ 100

If direct parameters fail even the relaxed thresholds, the pipeline raises `ConvergenceError` and refuses to produce downstream results.

### Tier 2 — Hyperparameter `tau` (Warning Only)

The random-effect scale `tau` controls the magnitude of country-level intercepts (`u_c = tau × u_c_raw`).  Slow mixing of `tau` is a **well-documented pathology** in hierarchical models with many groups (181 countries) and sparse data (4 years).

**Why it does not invalidate results:**
- The `ZeroSumNormal` constraint on `u_c_raw` ensures `Σ u_c = 0`, breaking the `tau ↔ alpha_s` ridge
- The non-centered parameterization `u_c = tau × u_c_raw` breaks the funnel geometry
- Relative country effects are well-identified regardless of `tau` mixing quality

**Policy:** Tau diagnostics are logged as `WARNING` but **do not block the pipeline**.

### Diagnostics Output Files

| File | Contents |
|------|----------|
| `bayes_diagnostics.json` | R-hat, ESS bulk/tail, divergences for all parameters |
| `bayes_ppc_summary.csv` | Posterior predictive check: per-observation residuals |
| `bayes_ppc_outliers.csv` | Top 25 observations by absolute residual |
| `bayes_ppc_diagnostics.json` | Aggregate PPC metrics (mean bias, 90% coverage) |
| `bayes_posterior.nc` | Full posterior samples (NetCDF/ArviZ format) |

### Observed Convergence Results

With the reference sampling configuration (16 chains × 8,000 draws, 15,000 tuning steps, nutpie sampler), the following convergence diagnostics are observed:

| Parameter group | R-hat | ESS bulk (min) | ESS tail (min) | Status |
|-----------------|-------|----------------|----------------|--------|
| `alpha_s` (5 species) | 1.00 | 9,719 | 22,163 | ✅ Converged (strict) |
| `beta_s` (5 species)  | 1.00 | 26,679 | 44,926 | ✅ Converged (strict) |
| `gamma_s` (5 species) | 1.00 | 29,614 | 50,338 | ✅ Converged (strict) |
| `sigma_s` (5 species) | 1.00 | 36,153 | 67,695 | ✅ Converged (strict) |
| `nu`                  | 1.00 | 51,176 | 81,742 | ✅ Converged (strict) |
| `tau` *(hyperparameter)* | 1.19 | 65 | 189 | ⚠️ Slow mixing (expected) |

- **Zero divergences** across all 16 chains.
- All direct parameters pass **strict** convergence thresholds (R-hat < 1.01, ESS ≥ 400) by wide margins.
- The `tau` hyperparameter exhibits the expected slow mixing (R-hat = 1.19, ESS bulk = 65), as discussed in Tier 2 above.  This does not affect downstream results.

### Posterior Predictive Check (PPC) Interpretation

The pipeline automatically performs posterior predictive checks on a subsample of 4,000 posterior draws (out of 128,000 total).

| PPC Metric | Value | Interpretation |
|-----------|-------|----------------|
| Residual mean | 0.106 | Slight positive bias — model marginally underestimates |
| Residual **median** | 0.001 | Effectively zero — bias is driven by outliers |
| Trimmed mean (10%) | 0.027 | Near-zero after trimming tail observations |
| 90% CI coverage | 93.6% | Well-calibrated (expected: ~90%) |
| \|residual\| > 2 | 124 / 1,615 (7.7%) | Consistent with heavy-tailed Student-t likelihood |
| \|residual\| > 3 | 50 / 1,615 (3.1%) | Genuine outliers in the data |

**Why the residual mean exceeds the 0.05 warning threshold:**

The bias is driven entirely by a small number of **genuine data outliers** — not model misspecification.  The top outliers are:

1. **Kuwait — sheep milk** (all 4 years): Species share ~0.5% but emission intensity ~13 kg CO₂e/t — among the highest globally for sheep milk.  The model's partial pooling pulls the prediction toward the global sheep mean, producing large positive residuals (|r| ≈ 4.8).
2. **Russia — sheep milk** (all 4 years): Species share ~0.015% with anomalously high intensity (~28–34 kg CO₂e/t).  Same mechanism as Kuwait.
3. **Czechia — sheep milk** (2022): Share effectively zero (0.003%) with elevated intensity.

These outliers share a pattern: **extremely low species shares combined with anomalously high intensities**.  The Student-t likelihood accommodates these heavy tails without distorting the bulk of estimates.  The median residual of 0.001 and the 93.6% coverage confirm that the model is well-calibrated for the vast majority of observations.

Full outlier details are in `outputs/bayes_ppc_outliers.csv`.

---

## Optimization Transparency

### Do-No-Harm Guardrail

The robust portfolio optimizer includes a **do-no-harm constraint**: no country's optimized emission intensity may exceed its baseline.  If the raw solver output violates this constraint (e.g., due to numerical artifacts at the boundary), the result is clamped to the baseline.

### Raw vs. Final Traceability

`robust_optimization_results.csv` retains **both** values:
- `raw_optimized_mean` / `raw_reduction_mean_pct` — the unguarded solver output
- `optimized_mean` / `reduction_mean_pct` — after the do-no-harm guard
- `no_harm_applied` — boolean flag marking which countries were adjusted
- `no_harm_action` — human-readable description of the adjustment

> **Corrected in R2.** Up to and including the R1 commit, the exporter assigned
> `sol_raw_report = sol_final if do_no_harm else sol_raw`, so whenever the guard was
> enabled — the default — the `raw_*` columns were written from the **guarded** solution.
> In the published R1 output they were byte-identical to the unprefixed columns in all
> 181 rows while 31 rows had in fact been guard-adjusted, which contradicted the
> description above. `raw_*` now always carries the unguarded solution.

`robust_optimization_audit.json` provides aggregate statistics: how many countries were affected, and to what extent.

### Observed Do-No-Harm Behavior

With the R2 defaults (posterior-consistent reference, exact LP solve):

| Metric | R1 | R2 |
|--------|----|----|
| Total countries optimized | 181 | 181 |
| Countries attaining a certified optimum | 178 | **181** |
| Countries with do-no-harm applied | 31 (17.1%) | **0** |
| Negative raw reductions | 0 | 0 |
| Negative final reductions | 0 | 0 |

**Why the guard no longer fires.** R1 attributed the 31 triggers to "near-boundary
numerical artifacts". That diagnosis was wrong. The reference intensity was computed
from the *observed* species intensities, `w_ref @ i_obs`, while the optimum was
evaluated on Bayesian *posterior* scenarios. For 47% of countries the posterior mean of
the reference mix exceeds the observed ratio — hierarchical shrinkage pulls sparse-data
countries toward the species means — so the optimum could appear to exceed the reference
even when it was optimal. With both sides measured on the same scenarios, the reference
mix is always feasible under the budget constraint, so a certified optimum can never
return a worse mean, and the guard is redundant. It is retained as an explicit
safeguard.

Single-species countries (74 of 181) still have no decision freedom: with only one
active species the mix is not a decision variable. They are fixed at the reference and
report a reduction of exactly zero. They do not enter the do-no-harm count.

### Sensitivity Grid

The pipeline evaluates 36 parameter combinations over `delta` (TV-distance budget),
`lambda` (mean vs CVaR weight) and `alpha` (CVaR confidence). Because the LP solves each
country in milliseconds, R2 runs the grid on the full 181-country panel rather than on a
20-country subset. The R1 warning `"30% of countries required do-no-harm revert (6/20)"`
no longer occurs, for the reason given above.

Two properties of the grid are worth noting, both reported in the manuscript:
- `delta` is the only parameter that changes the result. The mean reduction rises
  2.44% → 8.08% → 11.91% → 15.92% for `delta` = 0.01/0.05/0.10/0.20 and does **not**
  saturate on the full panel; the median saturates at `delta` = 0.05.
- The optimal composition is invariant to `lambda` and `alpha` to within 1e-16 in every
  weight, because the species ordering by posterior mean intensity coincides with the
  ordering by upper-tail intensity. The risk term therefore acts as a robustness check
  rather than as a distinct decision rule in this panel.

### Verification

```bash
python scripts/editorial_integrity_check.py
```

This script validates that:
- All required transparency columns exist
- Do-no-harm semantics are satisfied (guarded intensity ≤ baseline)
- Audit JSON is consistent with the CSV output
- Methods appendix references the audit trail

---

## Causal Guardrails

All results are **accounting counterfactuals / scenario-based analyses**.  They quantify "what if country X shifted its species portfolio" under the structural model — they do **not** imply causal effects.  No covariates supporting causal identification (e.g., policy instruments, farm-level interventions) are present in the data.

The `CAUSAL_DISCLAIMER` is embedded in all generated reports and appendices.

---

## Known Runtime Warnings

During pipeline execution, several warnings may appear in the console.  All are **expected** and **do not affect the correctness of results**.

### PyTensor / g++ Compiler Warnings (Windows)

On Windows, the PyTensor backend compiles C code using `g++` (from MinGW or MSYS2).  This produces verbose compiler warnings such as:

```
WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
c:\...\g++.exe: warning: ...MinGW\lib\libpthread.a: linker input file unused because linking not done
```

These warnings are **purely cosmetic** — they reflect MinGW linker configuration details and have no impact on numerical results.  They appear every time PyTensor compiles its computation graphs and cannot be suppressed without modifying the PyTensor source.

### Bayesian Model Warnings

| Warning | Severity | Explanation |
|---------|----------|-------------|
| `tau` R-hat > 1.01 / low ESS | ⚠️ Expected | Hierarchical hyperparameter slow mixing — see [Bayesian Model Convergence Protocol](#bayesian-model-convergence-protocol) |
| PPC residual mean > 0.05 | ⚠️ Expected | Driven by outliers (Kuwait, Russia sheep milk) — see [PPC Interpretation](#posterior-predictive-check-ppc-interpretation) |
| "Do-no-harm guard triggered" | ℹ️ By design | Conservative safety constraint — see [Optimization Transparency](#optimization-transparency) |
| "30% of countries required do-no-harm revert" | ℹ️ By design | Specific to tight sensitivity grid cells — see [Sensitivity Grid Warning](#sensitivity-grid-warning-30-of-countries-required-do-no-harm-revert) |

All warnings are documented in detail in their respective sections above.  If a **new** warning appears that is not listed here, please open an issue.

---

## Reproducibility for Manuscript

### Full Pipeline Execution

```bash
# Full pipeline (Bayesian + all downstream)
methane-portfolio run-all --chains 16 --draws 8000 --tune 15000
methane-portfolio run-all --chains 16 --draws 8000 --tune 15000 --target-accept 0.95

# Fast path (skip Bayesian, uses cached posterior if available)
methane-portfolio run-all --skip-bayes
```

### Reproducibility Manifest

```bash
python scripts/reproduce.py --skip-bayes
```

**Manifest output:** `outputs/reproducibility_manifest.json`
- Run parameters (chains, draws, tune, seeds)
- Git commit hash + dirty state
- Full package versions (`pip freeze`)
- SHA-256 checksums for all key output files

### Fixed Seeds

| Seed | Value | Purpose |
|------|-------|---------|
| `PYMC_SEED` | 42 | Bayesian sampler reproducibility |
| `RNG_SEED` | 20230101 | NumPy RNG for Dirichlet, subsampling |

---

## Dependencies

- numpy, pandas, scipy, matplotlib, pymc, arviz, xarray
- **Optional:** nutpie (Rust-based NUTS sampler, 5–20× faster)
- Python ≥ 3.11

## Testing

```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"
```

## Public Release Hygiene

Before making the repository public:
1. Verify no personal or institutional contact details remain in source headers/comments.
2. Verify no credentials/secrets are present (`.env`, API tokens, private keys).
3. Regenerate `outputs/reproducibility_manifest.json` in the final public commit.

Automated checks:

```bash
# Pre-publish scan (credentials, secrets, personal data)
python scripts/prepublish_check.py

# Editorial integrity (raw vs final optimisation traceability)
python scripts/editorial_integrity_check.py
```

## License

MIT

## Acknowledgements

This research was supported by the Google Cloud Research Credits program.  All computations and data processing for this project were performed using infrastructure provided by Google Cloud.
