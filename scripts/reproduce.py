#!/usr/bin/env python

# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

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
        "shapley_country.csv",
        "shapley_global.json",
        "uncertainty_summary.csv",
        "sensitivity_grid.csv",
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
        "python_executable": sys.executable,
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
        "output_sha256": _collect_outputs(),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rel = manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path
    print(f"[OK] Reproducibility manifest written to {rel}")


if __name__ == "__main__":
    main()
