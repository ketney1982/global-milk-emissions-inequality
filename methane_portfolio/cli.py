# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Command-line interface for the methane-portfolio pipeline.

Usage examples::

    methane-portfolio validate
    methane-portfolio shapley
    methane-portfolio bayes --chains 4 --draws 5000 --tune 10000 --target-accept 0.98
    methane-portfolio optimize --delta 0.15
    methane-portfolio figures
    methane-portfolio tables
    methane-portfolio report
    methane-portfolio run-all              # everything in order
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

from methane_portfolio import config
from methane_portfolio.utils import ensure_dirs, write_manifest


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from optional ArviZ preview subpackages.
    logging.getLogger("arviz.preview").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message="ArviZ is undergoing a major refactor.*",
        category=FutureWarning,
    )


def _assert_posterior_convergence(*, allow_weak_convergence: bool) -> None:
    """Validate saved Bayesian diagnostics before reuse downstream."""
    diag_path = config.OUTPUT_DIR / "bayes_diagnostics.json"
    if not diag_path.exists():
        return
    try:
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"Could not parse {diag_path}: {exc}") from exc

    converged_relaxed = bool(diag.get("converged_relaxed", diag.get("converged", False)))
    if converged_relaxed:
        return

    msg = (
        "Saved Bayesian posterior failed relaxed convergence checks "
        f"(max_rhat={diag.get('max_rhat', 'NA')}, "
        f"min_ess_bulk={diag.get('min_ess_bulk', 'NA')}, "
        f"min_ess_tail={diag.get('min_ess_tail', 'NA')}, "
        f"divergences={diag.get('divergences', 'NA')})."
    )
    if allow_weak_convergence:
        print(f"  [WARN] {msg} Continuing because --allow-weak-convergence was set.")
        return
    raise RuntimeError(
        f"{msg} Rerun Bayes with stronger settings or pass --allow-weak-convergence."
    )


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> None:
    from methane_portfolio.io import load_all
    from methane_portfolio.validate import validate_all

    long_df, agg_df, species = load_all(args.data_dir)
    report = validate_all(long_df, agg_df, raise_on_fail=not args.no_fail)
    n_pass = (report["status"] == "PASS").sum()
    n_total = len(report)
    print(f"[OK] Validation: {n_pass}/{n_total} checks passed")


def cmd_shapley(args: argparse.Namespace) -> None:
    from methane_portfolio.io import load_all
    from methane_portfolio.shapley import shapley_decompose

    long_df, agg_df, species = load_all(args.data_dir)
    result = shapley_decompose(
        long_df,
        agg_df,
        species,
        weight_method=args.weight_method,
    )
    print(f"[OK] Shapley decomposition for {len(result)} countries saved")


def cmd_bayes(args: argparse.Namespace) -> None:
    from methane_portfolio.bayes import fit_model, posterior_intensity_samples
    from methane_portfolio.io import load_all

    long_df, agg_df, species = load_all(args.data_dir)
    idata, data = fit_model(
        long_df,
        chains=args.chains,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        cores=args.cores,
        fail_on_weak_convergence=not args.allow_weak_convergence,
    )
    # Also extract posterior intensity samples for downstream use
    I_samples, country_list, species_list = posterior_intensity_samples(
        idata, data, n_samples=500,
    )
    print(f"[OK] Bayesian model fitted ({args.chains} chains x {args.draws} draws)")
    print(f"  Posterior samples: {I_samples.shape[0]} draws × {len(country_list)} countries × {len(species_list)} species")


def cmd_optimize(args: argparse.Namespace) -> None:
    import arviz as az
    from methane_portfolio.bayes import posterior_intensity_samples, prepare_bayes_data
    from methane_portfolio.io import load_all
    from methane_portfolio.robust_optimize import run_all_countries

    long_df, agg_df, species = load_all(args.data_dir)
    I_samples, country_list, species_list = None, None, None
    if args.allow_expansion:
        nc_path = config.OUTPUT_DIR / "bayes_posterior.nc"
        if nc_path.exists():
            _assert_posterior_convergence(
                allow_weak_convergence=args.allow_weak_convergence,
            )
            idata = az.from_netcdf(str(nc_path))
            bayes_data = prepare_bayes_data(long_df)
            I_samples, country_list, species_list = posterior_intensity_samples(
                idata, bayes_data, n_samples=500,
            )
            print("  Loaded existing Bayesian posterior for expansion-enabled optimisation")
        else:
            print("  [WARN] --allow-expansion requested but bayes_posterior.nc is missing; expansion may be disabled")

    result = run_all_countries(
        long_df,
        I_samples=I_samples,
        country_list=country_list,
        species_list=species_list,
        lam=args.lam,
        alpha=args.alpha,
        delta=args.delta,
        allow_expansion=args.allow_expansion,
    )
    print(f"[OK] Robust optimisation for {len(result)} countries saved")


def cmd_uncertainty(args: argparse.Namespace) -> None:
    import arviz as az
    from methane_portfolio.io import load_all
    from methane_portfolio.uncertainty import propagate_uncertainty, run_sensitivity_grid

    long_df, agg_df, species = load_all(args.data_dir)

    # Try to load Bayesian posterior for posterior-informed uncertainty
    I_samples, country_list, species_list = None, None, None
    nc_path = config.OUTPUT_DIR / "bayes_posterior.nc"
    if nc_path.exists():
        _assert_posterior_convergence(
            allow_weak_convergence=args.allow_weak_convergence,
        )
        from methane_portfolio.bayes import prepare_bayes_data, posterior_intensity_samples
        idata = az.from_netcdf(str(nc_path))
        data = prepare_bayes_data(long_df)
        I_samples, country_list, species_list = posterior_intensity_samples(
            idata, data, n_samples=500,
        )
        print("  Using Bayesian posterior for uncertainty propagation")

    result = propagate_uncertainty(
        long_df, I_samples=I_samples,
        country_list=country_list, species_list=species_list,
    )
    print(f"[OK] Uncertainty propagation for {len(result)} countries saved")

    # Also run the sensitivity grid
    grid = run_sensitivity_grid(
        long_df, I_samples=I_samples,
        country_list=country_list, species_list=species_list,
        workers=args.sensitivity_workers,
    )
    print(f"[OK] Sensitivity grid: {len(grid)} rows saved")


def cmd_figures(args: argparse.Namespace) -> None:
    import pandas as pd
    from methane_portfolio.figures import make_all_figures
    from methane_portfolio.io import load_all

    long_df, agg_df, species = load_all(args.data_dir)

    # Load intermediate results if available
    shapley_path = config.OUTPUT_DIR / "shapley_country.csv"
    shapley_df = None
    if shapley_path.exists():
        shapley_df = pd.read_csv(shapley_path)

    opt_path = config.OUTPUT_DIR / "robust_optimization_results.csv"
    opt_df = None
    if opt_path.exists():
        opt_df = pd.read_csv(opt_path)

    sens_path = config.OUTPUT_DIR / "sensitivity_grid.csv"
    sensitivity_df = None
    if sens_path.exists():
        sensitivity_df = pd.read_csv(sens_path)

    # Load Bayesian posterior if available
    idata, bayes_data = None, None
    nc_path = config.OUTPUT_DIR / "bayes_posterior.nc"
    if nc_path.exists():
        import arviz as az
        from methane_portfolio.bayes import prepare_bayes_data
        idata = az.from_netcdf(str(nc_path))
        bayes_data = prepare_bayes_data(long_df)
        print("  Loaded Bayesian posterior for Fig3/Fig4")

    make_all_figures(
        shapley_df=shapley_df,
        opt_df=opt_df,
        sensitivity_df=sensitivity_df,
        idata=idata,
        bayes_data=bayes_data,
        long_df=long_df,
        agg_df=agg_df,
    )
    print("[OK] Figures generated")


def cmd_tables(args: argparse.Namespace) -> None:
    import pandas as pd
    from methane_portfolio.io import load_all
    from methane_portfolio.tables import make_all_tables

    long_df, agg_df, species = load_all(args.data_dir)

    shapley_path = config.OUTPUT_DIR / "shapley_country.csv"
    shapley_df = pd.read_csv(shapley_path) if shapley_path.exists() else None

    opt_path = config.OUTPUT_DIR / "robust_optimization_results.csv"
    opt_df = pd.read_csv(opt_path) if opt_path.exists() else None

    sens_path = config.OUTPUT_DIR / "sensitivity_grid.csv"
    sensitivity_df = pd.read_csv(sens_path) if sens_path.exists() else None

    make_all_tables(
        long_df=long_df,
        agg_df=agg_df,
        shapley_df=shapley_df,
        opt_df=opt_df,
        sensitivity_df=sensitivity_df,
    )
    print("[OK] Tables generated")


def cmd_report(args: argparse.Namespace) -> None:
    from methane_portfolio.io import load_all
    from methane_portfolio.report import generate_appendix

    long_df, agg_df, species = load_all(args.data_dir)
    path = generate_appendix(long_df=long_df)
    print(f"[OK] Methods appendix written to {path}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the full pipeline in order."""

    t0 = time.time()
    n_steps = 9 if not args.skip_bayes else 8
    step = 0

    from methane_portfolio.io import load_all

    long_df, agg_df, species = load_all(args.data_dir)

    # 1. Validate
    step += 1
    from methane_portfolio.validate import validate_all
    validate_all(long_df, agg_df, raise_on_fail=not args.no_fail)
    print(f"[OK] Step {step}/{n_steps}: Validation passed")

    # 2. Shapley
    step += 1
    from methane_portfolio.shapley import shapley_decompose
    shapley_df = shapley_decompose(
        long_df,
        agg_df,
        species,
        weight_method=args.weight_method,
    )
    print(f"[OK] Step {step}/{n_steps}: Shapley decomposition ({len(shapley_df)} countries)")

    # 3. Bayesian model fitting (optional)
    idata, bayes_data = None, None
    I_samples, country_list, species_list = None, None, None
    if not args.skip_bayes:
        step += 1
        from methane_portfolio.bayes import fit_model, posterior_intensity_samples
        idata, bayes_data = fit_model(
            long_df,
            chains=args.chains, draws=args.draws,
            tune=args.tune, target_accept=args.target_accept,
            cores=args.cores,
            fail_on_weak_convergence=not args.allow_weak_convergence,
        )
        I_samples, country_list, species_list = posterior_intensity_samples(
            idata, bayes_data, n_samples=500,
        )
        print(f"[OK] Step {step}/{n_steps}: Bayesian model fitted ({args.chains} chains x {args.draws} draws)")
    else:
        print("  [SKIP] Bayesian model fitting (--skip-bayes)")
        nc_path = config.OUTPUT_DIR / "bayes_posterior.nc"
        if nc_path.exists():
            _assert_posterior_convergence(
                allow_weak_convergence=args.allow_weak_convergence,
            )
            import arviz as az
            from methane_portfolio.bayes import prepare_bayes_data, posterior_intensity_samples

            idata = az.from_netcdf(str(nc_path))
            bayes_data = prepare_bayes_data(long_df)
            I_samples, country_list, species_list = posterior_intensity_samples(
                idata, bayes_data, n_samples=500,
            )
            print(
                "  Loaded existing Bayesian posterior for downstream "
                "optimisation/uncertainty/sensitivity/figures"
            )
        else:
            print(
                "  [WARN] bayes_posterior.nc is missing; downstream uncertainty and "
                "sensitivity will use fallback scenarios, and Fig3 may be skipped"
            )
            if args.allow_expansion:
                print(
                    "  [WARN] --allow-expansion requested but posterior is unavailable; "
                    "expansion may be disabled for some countries"
                )

    # 4. Robust optimisation (posterior-informed if Bayes ran)
    step += 1
    from methane_portfolio.robust_optimize import run_all_countries
    opt_df = run_all_countries(
        long_df,
        I_samples=I_samples,
        country_list=country_list,
        species_list=species_list,
        lam=args.lam, alpha=args.alpha, delta=args.delta,
        allow_expansion=args.allow_expansion,
    )
    print(f"[OK] Step {step}/{n_steps}: Robust optimisation ({len(opt_df)} countries)")

    # 5. Uncertainty propagation
    step += 1
    from methane_portfolio.uncertainty import propagate_uncertainty
    unc_df = propagate_uncertainty(
        long_df, I_samples=I_samples,
        country_list=country_list, species_list=species_list,
    )
    print(f"[OK] Step {step}/{n_steps}: Uncertainty propagation ({len(unc_df)} countries)")

    # 6. Sensitivity grid
    step += 1
    from methane_portfolio.uncertainty import run_sensitivity_grid
    sensitivity_df = run_sensitivity_grid(
        long_df, I_samples=I_samples,
        country_list=country_list, species_list=species_list,
        allow_expansion=args.allow_expansion,
        workers=args.sensitivity_workers,
    )
    print(f"[OK] Step {step}/{n_steps}: Sensitivity grid ({len(sensitivity_df)} rows)")

    # 7. Tables
    step += 1
    from methane_portfolio.tables import make_all_tables
    make_all_tables(
        long_df=long_df, agg_df=agg_df,
        shapley_df=shapley_df, opt_df=opt_df,
        sensitivity_df=sensitivity_df,
    )
    print(f"[OK] Step {step}/{n_steps}: Tables generated")

    # 8. Figures
    step += 1
    from methane_portfolio.figures import make_all_figures
    make_all_figures(
        shapley_df=shapley_df,
        opt_df=opt_df,
        sensitivity_df=sensitivity_df,
        idata=idata,
        bayes_data=bayes_data,
        long_df=long_df,
        agg_df=agg_df,
    )
    print(f"[OK] Step {step}/{n_steps}: Figures generated")

    # 9. Manifest (written before report so the appendix can read actual params)
    write_manifest({
        "pipeline": "run_all",
        "skip_bayes": args.skip_bayes,
        "weight_method": args.weight_method,
        "chains": args.chains,
        "draws": args.draws,
        "tune": args.tune,
        "target_accept": args.target_accept,
        "lam": args.lam,
        "alpha": args.alpha,
        "delta": args.delta,
        "allow_expansion": args.allow_expansion,
        "n_countries": int(shapley_df["country_m49"].nunique()),
        "n_countries_input": int(long_df["country_m49"].nunique()),
        "n_countries_optimized": int(opt_df["country_m49"].nunique()),
        "n_species": int(long_df["milk_species"].nunique()),
    })

    # 10. Report
    step += 1
    from methane_portfolio.report import generate_appendix
    generate_appendix(
        long_df=long_df,
        chains=args.chains,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
    )
    print(f"[OK] Step {step}/{n_steps}: Methods appendix written")

    elapsed = time.time() - t0
    print(f"\n[OK] Pipeline complete in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="methane-portfolio",
        description="Bayesian Species Portfolio Optimisation for Milk CH₄",
    )

    # Suppress PyTensor C++ warnings on Windows (since we use nutpie or python fallback)
    import logging
    logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)
    import os
    os.environ["PYTENSOR_FLAGS"] = os.environ.get("PYTENSOR_FLAGS", "") + ",cxx="

    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Override data directory",
    )

    sub = p.add_subparsers(dest="command", help="Pipeline sub-commands")
    weight_choices = ("base", "end", "avg", "sum", "trapz")

    # validate
    v = sub.add_parser("validate", help="Run data validation checks")
    v.add_argument("--no-fail", action="store_true",
                   help="Don't raise on validation failure")

    # shapley
    sp = sub.add_parser("shapley", help="Shapley decomposition")
    sp.add_argument(
        "--weight-method",
        choices=weight_choices,
        default=config.DEFAULT_WEIGHT_METHOD,
        help="Global weighting method across the interval.",
    )

    # bayes
    b = sub.add_parser("bayes", help="Fit Bayesian hierarchical model")
    b.add_argument("--chains", type=int, default=config.CHAINS)
    b.add_argument("--draws", type=int, default=config.DRAWS)
    b.add_argument("--tune", type=int, default=config.TUNE)
    b.add_argument("--target-accept", type=float, default=config.TARGET_ACCEPT)
    b.add_argument("--cores", type=int, default=None,
                   help="CPU cores for parallel sampling (default: auto)")
    b.add_argument(
        "--allow-weak-convergence",
        action="store_true",
        help="Allow saving/using posterior even if relaxed convergence checks fail.",
    )

    # optimize
    o = sub.add_parser("optimize", help="Robust portfolio optimisation")
    o.add_argument("--lam", type=float, default=0.5)
    o.add_argument("--alpha", type=float, default=0.90)
    o.add_argument("--delta", type=float, default=0.10)
    o.add_argument(
        "--allow-expansion",
        action="store_true",
        help="Allow expansion into species with baseline share 0.",
    )
    o.add_argument(
        "--allow-weak-convergence",
        action="store_true",
        help="Allow using saved posterior even if relaxed convergence checks fail.",
    )

    # uncertainty
    u = sub.add_parser("uncertainty", help="Uncertainty propagation")
    u.add_argument(
        "--sensitivity-workers",
        type=int,
        default=None,
        help="Parallel workers for sensitivity grid (default: auto).",
    )
    u.add_argument(
        "--allow-weak-convergence",
        action="store_true",
        help="Allow using saved posterior even if relaxed convergence checks fail.",
    )

    # figures
    sub.add_parser("figures", help="Generate publication figures")

    # tables
    sub.add_parser("tables", help="Generate manuscript tables")

    # report
    sub.add_parser("report", help="Generate methods appendix")

    # run-all
    ra = sub.add_parser("run-all", help="Run full pipeline")
    ra.add_argument("--no-fail", action="store_true")
    ra.add_argument("--skip-bayes", action="store_true",
                    help="Skip Bayesian model fitting (faster)")
    ra.add_argument("--chains", type=int, default=config.CHAINS)
    ra.add_argument("--draws", type=int, default=config.DRAWS)
    ra.add_argument("--tune", type=int, default=config.TUNE)
    ra.add_argument("--target-accept", type=float, default=config.TARGET_ACCEPT)
    ra.add_argument("--cores", type=int, default=None,
                    help="CPU cores for parallel sampling (default: auto)")
    ra.add_argument(
        "--allow-weak-convergence",
        action="store_true",
        help="Allow using posterior even if relaxed convergence checks fail.",
    )
    ra.add_argument("--lam", type=float, default=0.5)
    ra.add_argument("--alpha", type=float, default=0.90)
    ra.add_argument("--delta", type=float, default=0.10)
    ra.add_argument(
        "--sensitivity-workers",
        type=int,
        default=None,
        help="Parallel workers for sensitivity grid (default: auto).",
    )
    ra.add_argument(
        "--allow-expansion",
        action="store_true",
        help="Allow expansion into species with baseline share 0.",
    )
    ra.add_argument(
        "--weight-method",
        choices=weight_choices,
        default=config.DEFAULT_WEIGHT_METHOD,
        help="Global weighting method for Shapley aggregation.",
    )

    return p


def main() -> None:
    import pandas as pd  # noqa: F811

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _setup_logging(args.verbose)

    if args.data_dir:
        config.DATA_DIR = args.data_dir

    ensure_dirs()

    dispatch = {
        "validate": cmd_validate,
        "shapley": cmd_shapley,
        "bayes": cmd_bayes,
        "optimize": cmd_optimize,
        "uncertainty": cmd_uncertainty,
        "figures": cmd_figures,
        "tables": cmd_tables,
        "report": cmd_report,
        "run-all": cmd_run_all,
    }

    try:
        dispatch[args.command](args)
    except Exception as exc:
        logging.error("Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
