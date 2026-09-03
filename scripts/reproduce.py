#!/usr/bin/env python

# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: ketney.otto@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Reproducibility helper for manuscript-ready pipeline runs.

Usage examples
--------------
python scripts/reproduce.py --skip-bayes
python scripts/reproduce.py --allow-expansion
python scripts/reproduce.py --no-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


def _run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _capture(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    return proc.stdout


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_outputs() -> dict[str, str]:
    files: list[Path] = []
    patterns = [
        "robust_optimization_results.csv",
        "robust_optimization_audit.json",
        "shapley_country.csv",
        "shapley_global.json",
        "uncertainty_summary.csv",
        "sensitivity_grid.csv",
        "sensitivity_summary_all181.csv",
        "sensitivity_summary_top20.csv",
        "bayes_diagnostics.json",
        "posterior_intensity_draws.npz",
        "methods_appendix.md",
    ]
    for name in patterns:
        p = OUT_DIR / name
        if p.exists():
            files.append(p)

    if FIG_DIR.exists():
        files.extend(sorted(FIG_DIR.glob("*.png")))

    checksums: dict[str, str] = {}
    for p in sorted(files):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        checksums[rel] = _sha256(p)
    return checksums


def _collect_inputs() -> dict[str, str]:
    """Checksum the frozen analytical inputs, so a run is pinned to its data."""
    data_dir = ROOT / "data"
    checksums: dict[str, str] = {}
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.csv")):
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            checksums[rel] = _sha256(p)
    return checksums


def _verify_manuscript_scope() -> dict[str, object]:
    """Check that the run produced the analysis the manuscript reports.

    Two scope regressions were shipped in R1/R2 and neither was caught by anything
    that ran automatically:

    * the sensitivity grid silently covered 20 countries instead of the panel, and
    * the optimiser exported one ambiguous absolute-reduction column.

    This returns a verdict per check and never raises, so the manifest records the
    state of the run rather than aborting it.
    """
    import csv

    checks: dict[str, object] = {}

    opt = OUT_DIR / "robust_optimization_results.csv"
    if opt.exists():
        with opt.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        required = {
            "observed_ch4_t",
            "abs_reduction_mt_ch4",
            "abs_reduction_mt_ch4_posterior",
        }
        checks["optimisation_columns"] = {
            "required": sorted(required),
            "missing": sorted(required - set(header)),
            "ambiguous_column_removed": "absolute_reduction_kt" not in header,
            "pass": not (required - set(header)) and "absolute_reduction_kt" not in header,
        }

    grid = OUT_DIR / "sensitivity_grid.csv"
    if grid.exists():
        countries: set[str] = set()
        configs: set[tuple[str, str, str]] = set()
        n_rows = 0
        with grid.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                countries.add(row["country_m49"])
                configs.add((row["delta"], row["lambda"], row["alpha"]))
        expected = len(countries) * len(configs)
        checks["sensitivity_grid_scope"] = {
            "n_rows": n_rows,
            "n_countries": len(countries),
            "n_configurations": len(configs),
            "expected_rows": expected,
            "pass": n_rows == expected and len(configs) == 36,
        }

    checks["all_pass"] = all(
        v.get("pass", True) for v in checks.values() if isinstance(v, dict)
    )
    return checks


def _redact_home(text: str) -> str:
    """Replace the running user's home directory with ``~``.

    The manifest is deposited publicly, and an absolute interpreter path leaks the
    local account name for no reproducibility benefit: what matters is the
    interpreter version and the package set, both of which are recorded separately.
    """
    home = str(Path.home())
    out = text.replace(home, "~")
    return out.replace(home.replace("\\", "/"), "~")


def _git_info() -> dict[str, str | bool]:
    out: dict[str, str | bool] = {"commit": "unknown", "dirty": True}
    try:
        out["commit"] = _capture(["git", "rev-parse", "HEAD"]).strip()
        dirty = _capture(["git", "status", "--porcelain"]).strip()
        out["dirty"] = bool(dirty)
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible pipeline and emit checksum manifest.")
    parser.add_argument("--skip-bayes", action="store_true", help="Skip Bayesian fitting in pipeline run.")
    parser.add_argument("--allow-expansion", action="store_true", help="Allow expansion into zero-share species.")
    parser.add_argument(
        "--weight-method",
        default="avg",
        choices=("base", "end", "avg", "sum", "trapz"),
        help="Weight method passed to run-all.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run the pipeline; only hash existing outputs.",
    )
    parser.add_argument(
        "--manifest",
        default=str(OUT_DIR / "reproducibility_manifest.json"),
        help="Path for reproducibility manifest JSON.",
    )
    args = parser.parse_args()

    if not args.no_run:
        cmd = [sys.executable, "-m", "methane_portfolio.cli", "run-all", "--weight-method", args.weight_method]
        if args.skip_bayes:
            cmd.append("--skip-bayes")
        if args.allow_expansion:
            cmd.append("--allow-expansion")
        _run(cmd)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": _redact_home(sys.executable),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git": _git_info(),
        "parameters": {
            "skip_bayes": args.skip_bayes,
            "allow_expansion": args.allow_expansion,
            "weight_method": args.weight_method,
            "no_run": args.no_run,
        },
        "pip_freeze": _capture([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "input_sha256": _collect_inputs(),
        "output_sha256": _collect_outputs(),
        "manuscript_scope_checks": _verify_manuscript_scope(),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rel = manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path
    print(f"[OK] Reproducibility manifest written to {rel}")

    scope = manifest["manuscript_scope_checks"]
    grid = scope.get("sensitivity_grid_scope")
    if grid:
        verdict = "OK" if grid["pass"] else "FAIL"
        print(
            f"[{verdict}] Sensitivity grid: {grid['n_rows']} rows = "
            f"{grid['n_countries']} countries x {grid['n_configurations']} configurations"
        )
    cols = scope.get("optimisation_columns")
    if cols:
        verdict = "OK" if cols["pass"] else "FAIL"
        print(f"[{verdict}] Optimisation export carries both absolute-reduction quantities")
    if not scope.get("all_pass", True):
        print(
            "[WARN] This run does NOT match the scope reported in the manuscript. "
            "See manuscript_scope_checks in the manifest."
        )


if __name__ == "__main__":
    main()
