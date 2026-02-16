# Decoupling Methane from Milk

**Bayesian Species Portfolio Optimization under Structural Constraints (2020–2023)**

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

# Bayesian model (requires PyMC + GPU/CPU time)
# Tip: Increase --chains to match your CPU core count for faster parallel processing
methane-portfolio bayes --chains 8 --draws 10000 --tune 20000 --target-accept 0.995

# Reproducibility manifest (runs pipeline + writes checksums)
python scripts/reproduce.py --skip-bayes
```

## Project Structure

```
├── data/                              # Input CSVs
├── methane_portfolio/                 # Main package
│   ├── __init__.py
│   ├── config.py                      # Central configuration
│   ├── io.py                          # Data loading
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
├── tests/                             # pytest tests
├── scripts/                           # Shell scripts
├── outputs/                           # Generated outputs
│   ├── figures/                       # Generated figures (PNG by default)
│   ├── *.csv                          # Result tables
│   ├── *.json                         # Summaries and manifests
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

## Dependencies

- numpy, pandas, scipy, matplotlib, pymc, arviz, xarray, rich
- Python ≥ 3.11

## Testing

```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"
```

## Reproducibility for Manuscript

- Run a controlled pipeline execution and generate a checksum manifest:

```bash
python scripts/reproduce.py --skip-bayes
```

- Manifest output: `outputs/reproducibility_manifest.json`
- It includes: run parameters, git commit/dirty state, package versions (`pip freeze`), and SHA-256 checksums for key outputs.

## Public Release Hygiene

- Before making the repository public:
1. Verify no personal or institutional contact details remain in source headers/comments.
2. Verify no credentials/secrets are present (`.env`, API tokens, private keys).
3. Regenerate `outputs/reproducibility_manifest.json` in the final public commit.

- Automated prepublish scan:

```bash
python scripts/prepublish_check.py
```

- Editorial integrity check (raw vs final optimisation traceability):

```bash
python scripts/editorial_integrity_check.py
```

## License

MIT

This research was supported by the Google Cloud Research Credits program. All computations and data processing for this project were performed using infrastructure provided by Google Cloud.
