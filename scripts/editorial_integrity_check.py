#!/usr/bin/env python

"""Editorial integrity checks for optimisation outputs.

This script validates that optimisation results preserve raw vs final values
and that the transparency audit file is coherent with the CSV outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run editorial integrity checks on optimisation outputs.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory containing robust_optimization_results.csv and robust_optimization_audit.json",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Tolerance for numeric consistency checks.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    csv_path = out / "robust_optimization_results.csv"
    audit_path = out / "robust_optimization_audit.json"
    appendix_path = out / "methods_appendix.md"

    if not csv_path.exists():
        _fail(f"Missing file: {csv_path}")
    if not audit_path.exists():
        _fail(f"Missing file: {audit_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        _fail("robust_optimization_results.csv is empty.")

    required_cols = {
        "baseline_intensity",
        "raw_optimized_mean",
        "optimized_mean",
        "raw_reduction_mean_pct",
        "reduction_mean_pct",
        "no_harm_applied",
        "no_harm_action",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        _fail(f"Missing required transparency columns in CSV: {missing_cols}")

    # Final reduction should never be worse than raw after guard.
    neg_raw = int((df["raw_reduction_mean_pct"] < 0).sum())
    neg_final = int((df["reduction_mean_pct"] < 0).sum())
    if neg_final > neg_raw:
        _fail(
            f"Final reductions have more negatives than raw outputs ({neg_final} > {neg_raw}).",
        )

    # Guard-applied rows must satisfy no-harm semantics.
    guard_rows = df[df["no_harm_applied"] == True]  # noqa: E712
    if not guard_rows.empty:
        if not (guard_rows["optimized_mean"] <= guard_rows["baseline_intensity"] + args.tol).all():
            _fail("Some no_harm_applied rows still exceed baseline intensity.")
        if (guard_rows["no_harm_action"].astype(str).str.strip() == "").any():
            _fail("Some no_harm_applied rows have empty no_harm_action.")

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    if int(audit.get("n_countries", -1)) != len(df):
        _fail(
            f"Audit n_countries ({audit.get('n_countries')}) does not match CSV rows ({len(df)}).",
        )
    if int(audit.get("n_no_harm_applied", -1)) != int((df["no_harm_applied"] == True).sum()):  # noqa: E712
        _fail("Audit n_no_harm_applied does not match CSV.")
    if int(audit.get("n_negative_raw_reductions", -1)) != neg_raw:
        _fail("Audit n_negative_raw_reductions does not match CSV.")
    if int(audit.get("n_negative_final_reductions", -1)) != neg_final:
        _fail("Audit n_negative_final_reductions does not match CSV.")

    if appendix_path.exists():
        appendix = appendix_path.read_text(encoding="utf-8")
        if "raw (`raw_*`) and final (`optimized_*`)" not in appendix:
            _fail("methods_appendix.md is missing the raw vs final transparency statement.")
        if "robust_optimization_audit.json" not in appendix:
            _fail("methods_appendix.md does not mention robust_optimization_audit.json.")

    print("[OK] Editorial integrity checks passed.")
    print(f"[OK] Countries: {len(df)} | no_harm_applied: {int((df['no_harm_applied'] == True).sum())} | negative raw/final: {neg_raw}/{neg_final}")  # noqa: E712


if __name__ == "__main__":
    main()
