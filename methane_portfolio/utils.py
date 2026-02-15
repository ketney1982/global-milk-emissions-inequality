"""Shared utilities used across the methane_portfolio package."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from methane_portfolio import config


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------
def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a reproducible numpy Generator."""
    return np.random.default_rng(seed or config.RNG_SEED)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Causal guardrails
# ---------------------------------------------------------------------------
_CAUSAL_WORDS = re.compile(
    r"\b(causes?|leads?\s+to|will achieve|results?\s+in|drives?|induces?)\b",
    re.IGNORECASE,
)


def check_causal_language(text: str) -> None:
    """Raise if causal language appears and the causal flag is off."""
    if config.CAUSAL_FLAG:
        return
    match = _CAUSAL_WORDS.search(text)
    if match:
        raise ValueError(
            f"Causal language detected ('{match.group()}') but CAUSAL_FLAG "
            f"is False.  Either remove causal claims or set "
            f"config.CAUSAL_FLAG = True (requires causal covariates)."
        )


def causal_disclaimer() -> str:
    """Return the standard accounting-counterfactual disclaimer."""
    return config.CAUSAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def write_manifest(
    params: dict[str, Any],
    max_validation_errors: int = 0,
    path: Path | None = None,
) -> Path:
    """Write a JSON run manifest recording parameters and environment."""
    import platform
    import sys

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "rng_seed": config.RNG_SEED,
        "pymc_seed": config.PYMC_SEED,
        "parameters": _jsonify(params),
        "max_validation_errors": max_validation_errors,
    }
    p = path or (config.OUTPUT_DIR / "run_manifest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return p


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy/Path types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj
