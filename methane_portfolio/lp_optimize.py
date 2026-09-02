# Autor: Ketney Otto
# Affiliation: Lucian Blaga University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: ketney.otto@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Exact linear-programming form of the mean-CVaR species-portfolio problem.

Why an LP
---------
With the Rockafellar-Uryasev representation of CVaR, the whole problem is linear:

    minimise   lam * (Ibar . w)  +  (1-lam) * ( t + 1/(K(1-alpha)) * sum_k u_k )

    s.t.       u_k >= I_k . w - t ,          u_k >= 0            (CVaR epigraph)
               z_s >= | w_s - w_ref,s |                          (L1 linearisation)
               sum_s z_s <= 2 * delta                            (TV budget)
               sum_s w_s  = 1 ,  w_s >= 0                         (simplex)
               w_s = 0 for species absent from the reference mix  (no expansion)
               Ibar . w <= ceiling                                (optional do-no-harm)

The original implementation solved this with SLSQP started at x0 = (w_ref, 0),
which is exactly the non-differentiable kink of |w - w_ref|; that is the source of
the reported convergence failures. HiGHS solves the LP form exactly, so every
country attains a certified optimum and no "best feasible solution" fallback,
weight renormalisation or TV re-projection is needed.

Variable layout:  x = [ w (S) | t (1) | u (K) | z (S) ]
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy import sparse


def empirical_cvar(vals: np.ndarray, alpha: float) -> float:
    """CVaR_alpha of an empirical sample, evaluated at its own alpha-quantile.

    Used for *reporting* both the reference and the optimised portfolio, so the
    two are computed the same way. (The RU auxiliary variable t is an artefact of
    the optimisation and is not a reliable quantile estimate for the optimum.)
    """
    t = float(np.percentile(vals, alpha * 100.0))
    return t + float(np.maximum(0.0, vals - t).mean()) / (1.0 - alpha)


def solve_lp(w_ref, I_scen, *, lam=0.5, alpha=0.90, delta=0.10,
             allow_expansion=False, ceiling=None):
    """Solve one country's mean-CVaR portfolio LP exactly.

    Returns dict: w_opt, mean_opt, cvar_opt, tv, status, success, message
    """
    S = len(w_ref)
    K = I_scen.shape[0]
    n = 2 * S + 1 + K
    Ibar = I_scen.mean(axis=0)

    iw = slice(0, S)
    it = S
    iu = slice(S + 1, S + 1 + K)
    iz = slice(S + 1 + K, n)

    c = np.zeros(n)
    c[iw] = lam * Ibar
    c[it] = (1.0 - lam)
    c[iu] = (1.0 - lam) / (K * (1.0 - alpha))

    rows, bub = [], []

    # CVaR epigraph:  I_k . w - t - u_k <= 0
    A_cv = sparse.hstack([
        sparse.csr_matrix(I_scen),                       # w
        sparse.csr_matrix(-np.ones((K, 1))),             # t
        -sparse.identity(K, format='csr'),               # u
        sparse.csr_matrix((K, S)),                       # z
    ], format='csr')
    rows.append(A_cv); bub.append(np.zeros(K))

    # z_s >= w_s - w_ref_s   ->   w_s - z_s <= w_ref_s
    A_z1 = sparse.hstack([
        sparse.identity(S, format='csr'),
        sparse.csr_matrix((S, 1)),
        sparse.csr_matrix((S, K)),
        -sparse.identity(S, format='csr'),
    ], format='csr')
    rows.append(A_z1); bub.append(w_ref.copy())

    # z_s >= w_ref_s - w_s   ->   -w_s - z_s <= -w_ref_s
    A_z2 = sparse.hstack([
        -sparse.identity(S, format='csr'),
        sparse.csr_matrix((S, 1)),
        sparse.csr_matrix((S, K)),
        -sparse.identity(S, format='csr'),
    ], format='csr')
    rows.append(A_z2); bub.append(-w_ref.copy())

    # budget:  sum z <= 2 delta
    a_b = np.zeros((1, n)); a_b[0, iz] = 1.0
    rows.append(sparse.csr_matrix(a_b)); bub.append(np.array([2.0 * delta]))

    # optional do-no-harm ceiling on the posterior mean
    if ceiling is not None:
        a_c = np.zeros((1, n)); a_c[0, iw] = Ibar
        rows.append(sparse.csr_matrix(a_c)); bub.append(np.array([float(ceiling)]))

    A_ub = sparse.vstack(rows, format='csr')
    b_ub = np.concatenate(bub)

    a_eq = np.zeros((1, n)); a_eq[0, iw] = 1.0
    A_eq = sparse.csr_matrix(a_eq)
    b_eq = np.array([1.0])

    bounds = []
    for s in range(S):
        if not allow_expansion and w_ref[s] == 0.0:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, 1.0))
    bounds.append((None, None))            # t free
    bounds += [(0.0, None)] * K            # u >= 0
    bounds += [(0.0, None)] * S            # z >= 0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return {"w_opt": w_ref.copy(),
                "mean_opt": float(Ibar @ w_ref),
                "cvar_opt": empirical_cvar(I_scen @ w_ref, alpha),
                "tv": 0.0, "status": int(res.status), "success": False,
                "message": str(res.message)}

    w = np.asarray(res.x[iw], dtype=float)
    w = np.clip(w, 0.0, None)
    if not allow_expansion:
        w[w_ref == 0.0] = 0.0
    ssum = w.sum()
    if ssum > 0:
        w = w / ssum

    port = I_scen @ w
    return {"w_opt": w,
            "mean_opt": float(port.mean()),
            "cvar_opt": empirical_cvar(port, alpha),
            "tv": 0.5 * float(np.abs(w - w_ref).sum()),
            "status": int(res.status), "success": True,
            "message": str(res.message)}
