"""
PageRankEngine: Industrial-grade iterative PageRank via power iteration.

The Google matrix is:

    G = (1 - p) * A_hat  +  (p / N) * e * e^T

where
    A_hat[i, j] = A[i, j]   if j is not dangling
                = 1/N        if j is a dangling node

    A is the column-stochastic link matrix (built by WebGraphLoader)
    p is the teleportation probability  (0 < p <= 1)
    e is the all-ones column vector of length N
    N is the number of nodes

PageRank vector r satisfies  r = G * r  (principal eigenvector).

Because G is dense we NEVER materialise it.  Instead the matrix-vector
product is computed as three cheap steps:

    y  = A * r                                (sparse MV, O(nnz))
    y += (dangling_correction / N) * e       (rank-1 dangling fix, O(N))
    y  = (1-p) * y  +  (p/N) * e            (teleportation, O(N))

Convergence is measured with the L1 norm of the residual.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass
class PageRankResult:
    scores: np.ndarray            # shape (N,), sums to 1
    iterations: int
    residuals: list[float] = field(default_factory=list)
    elapsed_sec: float = 0.0
    converged: bool = False

    def top_k(self, k: int = 10, idx_to_id: Optional[list] = None):
        """Return list of (node_id, score) for top-k nodes."""
        indices = np.argsort(-self.scores)[:k]
        result = []
        for idx in indices:
            node_id = idx_to_id[idx] if idx_to_id else idx
            result.append((node_id, float(self.scores[idx])))
        return result


class PageRankEngine:
    """
    Compute PageRank via power iteration on a column-stochastic sparse matrix.

    Parameters
    ----------
    A             : column-stochastic CSR matrix  (shape N x N)
    dangling_mask : bool array of length N; True where out-degree == 0
    p             : teleportation probability (damping = 1 - p)
    tol           : L1 convergence tolerance
    max_iter      : iteration cap
    """

    def __init__(
        self,
        A: sp.csr_matrix,
        dangling_mask: np.ndarray,
        p: float = 0.15,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.A = A
        self.dangling_mask = dangling_mask.astype(bool)
        self.p = p
        self.tol = tol
        self.max_iter = max_iter
        self.N = A.shape[0]

    # ------------------------------------------------------------------
    # Power iteration
    # ------------------------------------------------------------------

    def run(self, r0: Optional[np.ndarray] = None) -> PageRankResult:
        """Run power iteration and return a PageRankResult."""
        N, p = self.N, self.p
        alpha = 1.0 - p  # link-following weight

        # Initial distribution
        r = np.full(N, 1.0 / N, dtype=np.float64) if r0 is None else r0.copy()
        r /= r.sum()

        residuals: list[float] = []
        converged = False
        t0 = time.perf_counter()

        for iteration in range(1, self.max_iter + 1):
            # --- sparse link-following step ---
            r_new = self.A.dot(r)  # (N,)

            # --- dangling node correction (rank-1 term) ---
            # dangling mass that would be lost: redistribute uniformly
            dangling_sum = r[self.dangling_mask].sum()
            r_new += (dangling_sum / N)

            # --- teleportation ---
            r_new = alpha * r_new + (p / N)

            # --- convergence check (L1 norm) ---
            residual = float(np.abs(r_new - r).sum())
            residuals.append(residual)

            r = r_new

            if iteration % 10 == 0 or residual < self.tol:
                logger.debug("iter=%4d  residual=%.3e", iteration, residual)

            if residual < self.tol:
                converged = True
                logger.info(
                    "PageRank converged at iteration %d (residual=%.3e, p=%.3f)",
                    iteration, residual, p,
                )
                break
        else:
            logger.warning(
                "PageRank did NOT converge in %d iterations (final residual=%.3e)",
                self.max_iter, residuals[-1],
            )

        elapsed = time.perf_counter() - t0
        return PageRankResult(
            scores=r,
            iterations=iteration,
            residuals=residuals,
            elapsed_sec=elapsed,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Sweep over multiple p values
    # ------------------------------------------------------------------

    def run_sweep(self, p_values: list[float]) -> dict[float, PageRankResult]:
        """Compute PageRank for each teleportation probability in p_values."""
        results: dict[float, PageRankResult] = {}
        for pv in p_values:
            self.p = pv
            logger.info("--- Running PageRank with p=%.3f ---", pv)
            results[pv] = self.run()
        return results
