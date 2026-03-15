"""
GraphStructuralMetrics: Validates that PageRank reflects graph structure.

Metrics computed:
  - Entropy (Shannon)           : H = -Σ r_i log(r_i)  [nats]
  - Gini coefficient            : inequality of score distribution
  - In-degree Spearman ρ        : correlation between PR and in-degree
  - In-degree Pearson r         : linear correlation
  - Power-law fit               : exponent α of P(r) ~ r^{-α} (MLE)
  - Power-law R²                : goodness of fit on log-log scale
  - Score concentration (top-1/10/1% of nodes hold what fraction of total PR?)
  - Max / mean score ratio      : how much does the top node dominate?
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats
import scipy.sparse as sp


@dataclass
class GraphStructuralResult:
    entropy: float
    gini: float
    in_degree_spearman_rho: float
    in_degree_spearman_p: float
    in_degree_pearson_r: float
    in_degree_pearson_p: float
    powerlaw_alpha: float
    powerlaw_r2: float
    top1pct_mass: float         # fraction of total PR in top 1% of nodes
    top10pct_mass: float
    top1node_mass: float        # single highest-ranked node's share
    max_mean_ratio: float

    def as_dict(self) -> dict:
        return {
            "Entropy (nats)":           self.entropy,
            "Gini Coefficient":         self.gini,
            "In-Degree Spearman ρ":     self.in_degree_spearman_rho,
            "In-Degree Spearman p":     self.in_degree_spearman_p,
            "In-Degree Pearson r":      self.in_degree_pearson_r,
            "In-Degree Pearson p":      self.in_degree_pearson_p,
            "Power-law α":              self.powerlaw_alpha,
            "Power-law R²":             self.powerlaw_r2,
            "Top-1% PR Mass":           self.top1pct_mass,
            "Top-10% PR Mass":          self.top10pct_mass,
            "Top-1 Node PR Mass":       self.top1node_mass,
            "Max/Mean Ratio":           self.max_mean_ratio,
        }


class GraphStructuralMetrics:
    """
    Compute structural quality metrics for a PageRank score vector.

    Parameters
    ----------
    scores : PageRank probability vector (sums to 1)
    A      : column-stochastic adjacency matrix (to compute in-degrees)
    """

    def __init__(self, scores: np.ndarray, A: sp.csr_matrix):
        self.scores = scores
        self.A = A
        self.N = len(scores)

    def compute(self) -> GraphStructuralResult:
        r = self.scores
        N = self.N

        # In-degree of each node = sum of column of A^T = sum of row of A
        in_degree = np.array(self.A.sum(axis=1)).flatten()   # weighted in-degree

        # Shannon entropy
        safe = r[r > 0]
        entropy = float(-np.sum(safe * np.log(safe)))

        # Gini coefficient
        gini = self._gini(r)

        # In-degree correlations
        rho, rho_p = stats.spearmanr(r, in_degree)
        pearson_r, pearson_p = stats.pearsonr(r, in_degree)

        # Power-law fit on log-log scale
        alpha, r2 = self._powerlaw_fit(r)

        # Concentration
        sorted_r = np.sort(r)[::-1]
        cumsum = np.cumsum(sorted_r)
        top1_n = max(1, N // 100)
        top10_n = max(1, N // 10)
        top1pct_mass = float(sorted_r[:top1_n].sum())
        top10pct_mass = float(sorted_r[:top10_n].sum())
        top1node_mass = float(sorted_r[0])

        return GraphStructuralResult(
            entropy=entropy,
            gini=gini,
            in_degree_spearman_rho=float(rho),
            in_degree_spearman_p=float(rho_p),
            in_degree_pearson_r=float(pearson_r),
            in_degree_pearson_p=float(pearson_p),
            powerlaw_alpha=alpha,
            powerlaw_r2=r2,
            top1pct_mass=top1pct_mass,
            top10pct_mass=top10pct_mass,
            top1node_mass=top1node_mass,
            max_mean_ratio=float(r.max() / r.mean()),
        )

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        """Compute Gini coefficient of array (0=perfect equality, 1=maximum inequality)."""
        s = np.sort(np.abs(arr))
        n = len(s)
        idx = np.arange(1, n + 1)
        return float((2 * np.sum(idx * s) / (n * s.sum())) - (n + 1) / n)

    @staticmethod
    def _powerlaw_fit(r: np.ndarray) -> tuple[float, float]:
        """Fit P(r) ~ r^{-alpha} via linear regression on log-log rank-frequency."""
        sorted_r = np.sort(r[r > 0])[::-1]
        n = len(sorted_r)
        if n < 10:
            return float("nan"), float("nan")
        ranks = np.log(np.arange(1, n + 1))
        scores_log = np.log(sorted_r)
        slope, intercept, r_value, p_value, _ = stats.linregress(scores_log, ranks)
        return float(-slope), float(r_value ** 2)
