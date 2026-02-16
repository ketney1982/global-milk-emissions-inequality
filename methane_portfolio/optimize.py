# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Base portfolio optimization helpers.

Provides simplex-constrained optimisation utilities used by
``robust_optimize.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def simplex_constraint(n: int) -> LinearConstraint:
    """Return ÎŁ w_s = 1 as a LinearConstraint."""
    return LinearConstraint(np.ones(n), lb=1.0, ub=1.0)


def tv_distance_constraint(w_ref: np.ndarray, delta: float) -> dict:
    """Total-variation distance â‰¤ Î´:  0.5 Â· â€–w' â’ w_refâ€–â‚ â‰¤ Î´.

    Returns a dict suitable for ``scipy.optimize.minimize``
    (inequality constraint: returns value â‰Ą 0 when satisfied).
    """
    def tv_ineq(w: np.ndarray) -> float:
        return delta - 0.5 * np.sum(np.abs(w - w_ref))

    return {"type": "ineq", "fun": tv_ineq}


def mean_intensity(w: np.ndarray, I_matrix: np.ndarray) -> float:
    """Expected portfolio intensity E[I'(w)] over scenarios.

    Parameters
    ----------
    w : (S,) portfolio vector
    I_matrix : (K, S) â€“ K posterior/scenario draws of species intensities

    Returns
    -------
    float
    """
    return float(np.mean(I_matrix @ w))


def elasticity_vector(I_species_mean: np.ndarray) -> np.ndarray:
    """Compute dI/dw_s = I_s â’ I_total.

    Since I = ÎŁ w_s I_s â†’ â‚I/â‚w_s = I_s  (holding other weights fixed
    up to the simplex).  More precisely, on the simplex, shifting weight
    from species j to species s changes total by I_s â’ I_j.  We report
    the marginal I_s â’ I_total as a convenient first-order summary.
    """
    I_total = I_species_mean.mean()
    return I_species_mean - I_total
