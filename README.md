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

# Run the full pipeline (without Bayesian model fitting)
methane-portfolio run-all

# Or step by step
methane-portfolio validate
methane-portfolio shapley
methane-portfolio optimize --delta 0.10
methane-portfolio uncertainty
methane-portfolio figures
methane-portfolio tables
methane-portfolio report

# Bayesian model (requires PyMC + GPU/CPU time)
methane-portfolio bayes --chains 2 --draws 1000
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
│   ├── figures/                       # PNG + PDF figures
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
- **Causal guardrails**: All results are labelled as accounting counterfactuals; causal language triggers errors

## Dependencies

- numpy, pandas, scipy, matplotlib, pymc, arviz, xarray, rich
- Python ≥ 3.11

## Testing

```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"
```

## License

MIT
