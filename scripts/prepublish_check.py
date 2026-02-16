#!/usr/bin/env python

# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Pre-publication hygiene checks for secrets and personal contact data."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INCLUDE_EXTS = {
    ".py", ".md", ".toml", ".yaml", ".yml", ".json", ".ini", ".sh", ".txt",
}
EXCLUDE_DIRS = {
    ".git", ".venv", ".pytest_cache", ".mypy_cache", "outputs", "data",
    "methane_portfolio.egg-info", "__pycache__",
}

PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "orcid": re.compile(r"orcid\.org/\d{4}-\d{4}-\d{4}-\d{4}", re.IGNORECASE),
    "github_token": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "private_key": re.compile(r"BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY"),
}


def should_scan(path: Path) -> bool:
    if path.suffix.lower() not in INCLUDE_EXTS:
        return False
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return False
    return True


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings
    for line_no, line in enumerate(text.splitlines(), start=1):
        for key, rx in PATTERNS.items():
            if rx.search(line):
                findings.append((key, line_no, line.strip()))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan repository for potential secrets/PII before public release.")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Repository root path (default: current project root).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    all_findings: list[tuple[Path, str, int, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if not should_scan(rel):
            continue
        for key, line_no, line in scan_file(path):
            all_findings.append((rel, key, line_no, line))

    if all_findings:
        print("[FAIL] Potential sensitive content detected:")
        for rel, key, line_no, line in all_findings:
            print(f"- {rel}:{line_no} [{key}] {line}")
        sys.exit(1)

    print("[OK] No obvious secrets or personal contact details detected in scanned files.")


if __name__ == "__main__":
    main()
