"""
GraphRAGMetrics: Evaluates retrieval quality in the GraphRAG pipeline.

Metrics computed:
  - Mean Reciprocal Rank (MRR)       : 1/rank of first relevant entity
  - Hit@k (k=1,3,5,10)               : was a relevant entity in top-k?
  - Precision@k / Recall@k           : fraction of top-k that are relevant
  - Mean Hop Distance                : avg BFS distance of top-k from seeds
  - Score Decay by Hop               : mean PPR score per hop distance (should decrease)
  - Personalisation Fidelity         : Spearman ρ between score and proximity to seeds
  - Seed Score vs Non-Seed Score     : ratio (seeds should rank highest for high p)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats

from ..graphrag.pagerank_retrieval import RetrievalResult


@dataclass
class GraphRAGResult:
    mrr: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mean_hop_distance: float
    score_by_hop: dict[int, float]         # hop -> mean score
    personalisation_fidelity: float        # Spearman ρ(score, -hop_distance)
    seed_vs_nonseed_ratio: float

    def as_dict(self) -> dict:
        d = {
            "MRR":                          self.mrr,
            "Hit@1":                        self.hit_at_1,
            "Hit@3":                        self.hit_at_3,
            "Hit@5":                        self.hit_at_5,
            "Hit@10":                       self.hit_at_10,
            "Mean Hop Distance (top-10)":   self.mean_hop_distance,
            "Personalisation Fidelity ρ":   self.personalisation_fidelity,
            "Seed/Non-Seed Score Ratio":    self.seed_vs_nonseed_ratio,
        }
        for k, v in self.precision_at_k.items():
            d[f"Precision@{k}"] = v
        for k, v in self.recall_at_k.items():
            d[f"Recall@{k}"] = v
        for h, s in self.score_by_hop.items():
            d[f"Mean Score Hop={h}"] = s
        return d


class GraphRAGMetrics:
    """
    Evaluate a GraphRAG retrieval run.

    Parameters
    ----------
    results          : list of RetrievalResult from PageRankRetrieval.retrieve()
    relevant_entities: set of entity names considered ground-truth relevant
    seeds            : list of seed entity names used in the query
    """

    def __init__(
        self,
        results: list[RetrievalResult],
        relevant_entities: set[str],
        seeds: list[str],
    ):
        self.results = results
        self.relevant = relevant_entities
        self.seeds = set(seeds)

    def compute(self) -> GraphRAGResult:
        names = [r.entity_name for r in self.results]
        scores = np.array([r.score for r in self.results])
        hops = [r.hop_distance for r in self.results]

        # MRR
        mrr = 0.0
        for i, name in enumerate(names):
            if name in self.relevant:
                mrr = 1.0 / (i + 1)
                break

        # Hit@k
        def hit_at(k):
            return float(any(n in self.relevant for n in names[:k]))

        # Precision@k and Recall@k
        prec_k, rec_k = {}, {}
        for k in [1, 3, 5, 10]:
            top = names[:k]
            hits = sum(1 for n in top if n in self.relevant)
            prec_k[k] = hits / k
            rec_k[k] = hits / len(self.relevant) if self.relevant else 0.0

        # Mean hop distance of top-10
        valid_hops = [h for h in hops[:10] if h is not None]
        mean_hop = float(np.mean(valid_hops)) if valid_hops else float("nan")

        # Score by hop distance
        score_by_hop: dict[int, list[float]] = {}
        for r in self.results:
            if r.hop_distance is not None:
                score_by_hop.setdefault(r.hop_distance, []).append(r.score)
        score_by_hop_mean = {h: float(np.mean(v)) for h, v in sorted(score_by_hop.items())}

        # Personalisation fidelity: Spearman ρ between score and proximity (-hop)
        valid = [(r.score, r.hop_distance) for r in self.results if r.hop_distance is not None]
        if len(valid) > 2:
            sc = [v[0] for v in valid]
            neg_hop = [-v[1] for v in valid]  # higher = closer = should have higher score
            rho, _ = stats.spearmanr(sc, neg_hop)
            fidelity = float(rho)
        else:
            fidelity = float("nan")

        # Seed vs non-seed score ratio
        seed_scores = [r.score for r in self.results if r.entity_name in self.seeds]
        nonseed_scores = [r.score for r in self.results if r.entity_name not in self.seeds]
        if seed_scores and nonseed_scores:
            ratio = float(np.mean(seed_scores) / np.mean(nonseed_scores))
        else:
            ratio = float("nan")

        return GraphRAGResult(
            mrr=mrr,
            hit_at_1=hit_at(1),
            hit_at_3=hit_at(3),
            hit_at_5=hit_at(5),
            hit_at_10=hit_at(10),
            precision_at_k=prec_k,
            recall_at_k=rec_k,
            mean_hop_distance=mean_hop,
            score_by_hop=score_by_hop_mean,
            personalisation_fidelity=fidelity,
            seed_vs_nonseed_ratio=ratio,
        )
