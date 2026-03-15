"""
AnalyticalPageRank: Closed-form PageRank via direct linear solve.

============================================================
MATHEMATICAL DERIVATION
============================================================

The Google matrix for a web graph with N pages is:

    G = (1 - p) * A_hat  +  (p / N) * e * e^T         ... (1)

where:
    A_hat  -- column-stochastic transition matrix (dangling nodes
              redistributed uniformly: A_hat[:, j] = 1/N for dangling j)
    p      -- teleportation probability  (0 < p ≤ 1)
    e      -- all-ones column vector  (length N)
    N      -- number of web pages

PageRank r is the UNIQUE stationary distribution:

    r = G * r ,   ||r||_1 = 1                          ... (2)

Substituting (1) into (2):

    r = (1-p) * A_hat * r  +  (p/N) * e * (e^T * r)

Since e^T * r = 1  (r is a probability vector):

    r = (1-p) * A_hat * r  +  (p/N) * e

Rearranging:

    r  -  (1-p) * A_hat * r  =  (p/N) * e
    [I - (1-p) * A_hat] * r  =  (p/N) * e             ... (3)

CLOSED FORM:

    r = (p/N) * [I - (1-p) * A_hat]^{-1} * e          ... (4)

The matrix [I - (1-p)*A_hat] is strictly diagonally dominant for p > 0,
so it is always invertible.  We solve the linear system (3) directly
using scipy's sparse direct solver (SuperLU) instead of inverting (4)
explicitly — this is O(N^1.5) to O(N^2) but exact for small graphs.

Effect of p on PageRank:
  p -> 0 : r concentrates on the highest-authority nodes (may not be
            unique for reducible graphs).  Power-law distribution.
  p -> 1 : r -> uniform  (1/N for all pages).  The graph structure
            is completely ignored.
  Intermediate p (≈ 0.15 in Google's original paper) balances authority
  concentration with uniform exploration, guaranteeing convergence and
  uniqueness via the Perron–Frobenius theorem.
============================================================
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

# Maximum graph size for which we attempt the direct solve.
# For N > this threshold we fall back to a note (matrix too large to invert).
_MAX_DIRECT_N = 50_000


class AnalyticalPageRank:
    """
    Compute the closed-form PageRank solution for a web graph.

    For small graphs (N ≤ 50 k) we solve  [I-(1-p)*A_hat]*r = (p/N)*e
    directly using scipy's sparse LU factorisation.

    For larger graphs we document why direct inversion is infeasible and
    return None so callers can fall back to power iteration.
    """

    def __init__(
        self,
        A: sp.csr_matrix,
        dangling_mask: np.ndarray,
        p: float = 0.15,
    ) -> None:
        self.A = A
        self.dangling_mask = dangling_mask.astype(bool)
        self.p = p
        self.N = A.shape[0]
        self._A_hat: Optional[sp.csr_matrix] = None  # cached

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> Optional[np.ndarray]:
        """
        Return the analytical PageRank vector, or None if N is too large.

        The returned vector sums to 1 (probability distribution).
        """
        if self.N > _MAX_DIRECT_N:
            logger.warning(
                "N=%d > %d: direct solve skipped (use power iteration instead).",
                self.N, _MAX_DIRECT_N,
            )
            return None

        t0 = time.perf_counter()
        A_hat = self._get_A_hat()

        # Build  B = I - (1-p) * A_hat
        alpha = 1.0 - self.p
        B = sp.eye(self.N, format="csr") - alpha * A_hat

        # Right-hand side: (p/N) * e
        rhs = np.full(self.N, self.p / self.N, dtype=np.float64)

        logger.info("Solving %dx%d linear system (direct sparse LU) ...", self.N, self.N)
        try:
            r = spla.spsolve(B, rhs)
        except Exception as exc:
            logger.error("Direct solve failed: %s", exc)
            return None

        # Normalise to a proper probability vector
        r = np.maximum(r, 0.0)
        r /= r.sum()

        elapsed = time.perf_counter() - t0
        logger.info("Analytical solve completed in %.3fs", elapsed)
        return r

    def compare_with_iterative(
        self,
        iterative_scores: np.ndarray,
        top_k: int = 10,
    ) -> dict:
        """
        Compare analytical vs. iterative PageRank scores.

        Returns a dict with:
          l1_error, l_inf_error, top_k_rank_overlap, top_k lists.
        """
        analytical = self.compute()
        if analytical is None:
            return {"error": "Analytical solution not available (graph too large)"}

        l1 = float(np.abs(analytical - iterative_scores).sum())
        linf = float(np.abs(analytical - iterative_scores).max())

        top_analytical = set(np.argsort(-analytical)[:top_k])
        top_iterative = set(np.argsort(-iterative_scores)[:top_k])
        overlap = len(top_analytical & top_iterative)

        return {
            "l1_error": l1,
            "l_inf_error": linf,
            "top_k": top_k,
            "top_k_rank_overlap": overlap,
            "top_k_analytical": sorted(top_analytical),
            "top_k_iterative": sorted(top_iterative),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_A_hat(self) -> sp.csr_matrix:
        """Build A_hat: A with dangling columns replaced by 1/N."""
        if self._A_hat is not None:
            return self._A_hat

        N = self.N
        A_hat = self.A.copy().astype(np.float64)

        # For each dangling node j (out-degree=0), add a 1/N column
        dangling_indices = np.where(self.dangling_mask)[0]
        if len(dangling_indices) > 0:
            # Build a sparse correction: each dangling column j -> all rows get 1/N
            rows = np.tile(np.arange(N), len(dangling_indices))
            cols = np.repeat(dangling_indices, N)
            vals = np.full(len(rows), 1.0 / N)
            correction = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
            A_hat = A_hat + correction

        self._A_hat = A_hat
        return self._A_hat
