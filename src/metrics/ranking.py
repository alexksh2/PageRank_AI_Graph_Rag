"""
RankingMetrics: Evaluates the quality of PageRank node rankings.

Metrics computed:
  - Top-k overlap (precision)    : |top_k_A ∩ top_k_B| / k
  - Kendall's τ                  : rank correlation (concordant - discordant pairs)
  - Spearman's ρ                 : rank correlation (Pearson on ranks)
  - NDCG@k                       : Normalised Discounted Cumulative Gain
  - Precision@k / Recall@k       : against a labelled relevant set (if provided)
  - Mean Reciprocal Rank (MRR)   : for labelled relevant nodes
  - Rank displacement             : mean |rank_A[i] - rank_B[i]| over top-k
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class RankingResult:
    k: int
    top_k_overlap: float           # intersection fraction
    kendall_tau: float
    kendall_p: float
    spearman_rho: float
    spearman_p: float
    ndcg_at_k: float
    precision_at_k: Optional[float]
    recall_at_k: Optional[float]
    mrr: Optional[float]
    mean_rank_displacement: float  # average |rank_A - rank_B| in top-k

    def as_dict(self) -> dict:
        return {
            f"Top-{self.k} Overlap":       self.top_k_overlap,
            "Kendall τ":                   self.kendall_tau,
            "Kendall p-value":             self.kendall_p,
            "Spearman ρ":                  self.spearman_rho,
            "Spearman p-value":            self.spearman_p,
            f"NDCG@{self.k}":              self.ndcg_at_k,
            f"Precision@{self.k}":         self.precision_at_k,
            f"Recall@{self.k}":            self.recall_at_k,
            "MRR":                         self.mrr,
            "Mean Rank Displacement":      self.mean_rank_displacement,
        }


class RankingMetrics:
    """
    Compare two rankings (e.g. iterative vs analytical PageRank).

    Parameters
    ----------
    scores_a       : scores from method A (e.g. power iteration)
    scores_b       : scores from method B (e.g. analytical)
    k              : cutoff for top-k metrics
    relevant_nodes : optional set of node indices known to be "relevant"
    """

    def __init__(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        k: int = 20,
        relevant_nodes: Optional[set[int]] = None,
    ):
        self.a = scores_a
        self.b = scores_b
        self.k = k
        self.relevant = relevant_nodes

    def compute(self) -> RankingResult:
        N = len(self.a)
        k = min(self.k, N)

        rank_a = np.argsort(-self.a)       # descending order of scores
        rank_b = np.argsort(-self.b)

        top_k_a = set(rank_a[:k])
        top_k_b = set(rank_b[:k])
        overlap = len(top_k_a & top_k_b) / k

        # Position arrays (1-indexed rank of each node)
        pos_a = np.empty(N, dtype=np.int64)
        pos_b = np.empty(N, dtype=np.int64)
        pos_a[rank_a] = np.arange(1, N + 1)
        pos_b[rank_b] = np.arange(1, N + 1)

        # Kendall τ and Spearman ρ on the full ranking
        # (subsample for speed if N is large)
        sample = min(N, 10_000)
        idx = np.random.RandomState(42).choice(N, sample, replace=False)
        tau, tau_p = stats.kendalltau(pos_a[idx], pos_b[idx])
        rho, rho_p = stats.spearmanr(pos_a[idx], pos_b[idx])

        # NDCG@k: treat scores_b as "relevance" ground truth
        ndcg = self._ndcg(self.b, rank_a, k)

        # Mean rank displacement in top-k (using method A's top-k)
        top_k_nodes = rank_a[:k]
        displacement = float(np.abs(pos_a[top_k_nodes] - pos_b[top_k_nodes]).mean())

        # Optional: precision@k, recall@k, MRR against labelled relevant set
        prec, rec, mrr = None, None, None
        if self.relevant:
            hits = [1 if rank_a[i] in self.relevant else 0 for i in range(k)]
            prec = sum(hits) / k
            rec = sum(hits) / len(self.relevant) if self.relevant else 0.0
            # MRR: 1/rank of first relevant node in ranking A
            for rank_pos, node in enumerate(rank_a, 1):
                if node in self.relevant:
                    mrr = 1.0 / rank_pos
                    break

        return RankingResult(
            k=k,
            top_k_overlap=overlap,
            kendall_tau=float(tau),
            kendall_p=float(tau_p),
            spearman_rho=float(rho),
            spearman_p=float(rho_p),
            ndcg_at_k=ndcg,
            precision_at_k=prec,
            recall_at_k=rec,
            mrr=mrr,
            mean_rank_displacement=displacement,
        )

    @staticmethod
    def _ndcg(relevance: np.ndarray, ranked_indices: np.ndarray, k: int) -> float:
        """Compute NDCG@k using relevance scores as ground truth."""
        gains = relevance[ranked_indices[:k]]
        # Normalise relevance to [0,1]
        max_rel = relevance.max()
        if max_rel == 0:
            return 0.0
        gains = gains / max_rel

        discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)
        dcg = float((gains / discounts).sum())

        # Ideal DCG: sort relevance descending
        ideal_gains = np.sort(relevance)[::-1][:k] / max_rel
        idcg = float((ideal_gains / discounts[:len(ideal_gains)]).sum())

        return dcg / idcg if idcg > 0 else 0.0
