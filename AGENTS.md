# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `methane_portfolio/` (CLI, validation, Shapley decomposition, Bayesian modeling, optimization, uncertainty, figures, tables, report generation).  
Tests are in `tests/` with shared fixtures in `tests/conftest.py`.  
Input datasets are stored in `data/` as CSV files.  
Generated artifacts go to `outputs/` (tables, JSON summaries, figures, appendix); treat this directory as build output unless a result is intentionally versioned.  
Automation helpers live in `scripts/` (notably `scripts/run_all.sh`).

## Build, Test, and Development Commands
- `python -m venv .venv` then `.venv\Scripts\activate` (Windows): create/activate local environment.
- `pip install -e ".[dev]"`: install package + dev dependencies in editable mode.
- `methane-portfolio run-all --skip-bayes`: run full fast pipeline without Bayesian fitting.
- `methane-portfolio bayes --chains 4 --draws 1000`: run Bayesian model explicitly (heavier runtime).
- `pytest tests/ -v`: run complete test suite.
- `pytest tests/ -v -m "not slow"`: run quick subset.

## Coding Style & Naming Conventions
Target Python `>=3.11` and follow PEP 8 with 4-space indentation.  
Use `snake_case` for modules, functions, and variables; use `PascalCase` for test classes (for example, `TestValidateIdentity`).  
Prefer explicit type hints and short, single-purpose functions.  
Keep column names and domain terms consistent with existing data schema (`country_m49`, `milk_species`, `kg_co2e_per_ton_milk`).

## Testing Guidelines
Use `pytest` and keep tests under `tests/test_<feature>.py`.  
Reuse fixtures from `tests/conftest.py` or module fixtures for real-data loading.  
For numeric logic, assert with tolerances (`assert_allclose`, bounded relative/absolute error).  
When adding features, include at least one regression test covering expected outputs or failure modes.

## Commit & Pull Request Guidelines
Current history contains a single bootstrap commit (`Initial`), so conventions are lightweight.  
Use concise imperative commit subjects (example: `Add sensitivity-grid validation guard`).  
PRs should include: purpose, changed modules, test evidence (`pytest ...`), and any pipeline/runtime impact.  
If figures/tables change, mention affected output files (for example, `outputs/figures/fig3_*.png`).

## Configuration & Reproducibility Notes
Avoid editing raw files in `data/`; write derived artifacts to `outputs/`.  
For Bayesian runs, document `--chains`, `--draws`, and whether `--skip-bayes` was used so results remain reproducible.
