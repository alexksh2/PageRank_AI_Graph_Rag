"""
PageRankRetrieval: Query-aware PageRank over a knowledge graph.

============================================================
ALGORITHM: Personalised / Query-Biased PageRank
============================================================

Standard PageRank ranks nodes by global authority.  For GraphRAG we need
QUERY-AWARE importance: given seed entities (e.g. "Marie Curie",
"medical imaging") we personalise the teleportation distribution so that
the random surfer always teleports INTO the seed set rather than to any
random node.

The personalised Google matrix is:

    G_q = (1 - p) * A_hat  +  p * v_q * e^T

where
    v_q  -- query-specific personalisation vector (sums to 1)
             v_q[i] = 1/|seeds| if entity i is a seed, else 0
    e    -- all-ones vector
    A_hat -- column-stochastic adjacency matrix (dangling handled)

PageRank equation:

    r_q = G_q * r_q
        = (1-p) * A_hat * r_q  +  p * v_q

Closed form (same derivation as global PR):

    r_q = p * [I - (1-p) * A_hat]^{-1} * v_q         ... (*)

Power iteration:

    r^{(t+1)} = (1-p) * A_hat * r^{(t)}  +  p * v_q

This converges to (*) and biases the stationary distribution towards the
seed entities and their graph neighbourhood, effectively performing
multi-hop reasoning:

    "Marie Curie"  -[discovered]->  "radium"
    "radium"       -[used_in]->     "radiotherapy"
    "radiotherapy" -[advances]->    "medical imaging"

Nodes closer (in link-distance) to seeds accumulate more score, naturally
surfacing the multi-hop reasoning chain relevant to the query.

============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    entity_name: str
    score: float
    rank: int
    entity_type: str
    hop_distance: Optional[int] = None  # BFS distance from nearest seed


class PageRankRetrieval:
    """
    Personalised PageRank retrieval engine for GraphRAG.

    Parameters
    ----------
    kg          : KnowledgeGraph instance
    p           : teleportation probability (default 0.15)
    tol         : L1 convergence tolerance
    max_iter    : maximum power iterations
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        p: float = 0.15,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.kg = kg
        self.p = p
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        seed_entities: list[str],
        k: int = 10,
        exclude_seeds: bool = False,
    ) -> list[RetrievalResult]:
        """
        Run personalised PageRank from seed_entities and return top-k nodes.

        Parameters
        ----------
        seed_entities  : list of entity names to use as query seeds
        k              : number of results to return
        exclude_seeds  : if True, seeds themselves are excluded from results
        """
        N = self.kg.n_entities
        if N == 0:
            logger.warning("Empty knowledge graph.")
            return []

        v_q = self._build_personalisation(seed_entities, N)
        r = self._power_iterate(v_q, N)

        # Sort by score descending
        order = np.argsort(-r)
        hop_distances = self._bfs_distances(seed_entities)

        results: list[RetrievalResult] = []
        rank = 0
        for idx in order:
            name = self.kg.idx_to_name(idx)
            if exclude_seeds and name in seed_entities:
                continue
            entity = self.kg.entity(name)
            etype = entity.entity_type if entity else "entity"
            results.append(
                RetrievalResult(
                    entity_name=name,
                    score=float(r[idx]),
                    rank=rank + 1,
                    entity_type=etype,
                    hop_distance=hop_distances.get(name),
                )
            )
            rank += 1
            if rank >= k:
                break

        return results

    def explain_path(self, source: str, target: str, max_hops: int = 5) -> list[list[str]]:
        """
        Return all simple paths from source to target up to max_hops.

        Used to show the multi-hop reasoning chain that PageRank surfaces.
        """
        paths: list[list[str]] = []
        self._dfs_paths(source, target, [], paths, max_hops)
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_personalisation(self, seeds: list[str], N: int) -> np.ndarray:
        """Build the query personalisation vector v_q."""
        v_q = np.zeros(N, dtype=np.float64)
        valid_seeds = []
        for name in seeds:
            ent = self.kg.entity(name)
            if ent is not None:
                v_q[ent.idx] = 1.0
                valid_seeds.append(name)
            else:
                logger.warning("Seed entity '%s' not found in KG.", name)

        if v_q.sum() == 0:
            logger.warning("No valid seeds — falling back to uniform personalisation.")
            v_q[:] = 1.0 / N
        else:
            v_q /= v_q.sum()

        return v_q

    def _power_iterate(self, v_q: np.ndarray, N: int) -> np.ndarray:
        """Personalised PageRank power iteration."""
        A = self.kg.adjacency_matrix()  # column-stochastic, dangling handled
        p = self.p
        alpha = 1.0 - p

        r = v_q.copy()  # warm-start with personalisation vector

        for it in range(1, self.max_iter + 1):
            r_new = alpha * A.dot(r) + p * v_q
            residual = float(np.abs(r_new - r).sum())
            r = r_new
            if residual < self.tol:
                logger.debug("PPR converged at iter %d (residual=%.2e)", it, residual)
                break

        return r

    def _bfs_distances(self, seeds: list[str]) -> dict[str, int]:
        """BFS from all seeds simultaneously; return {entity_name: min_distance}."""
        from collections import deque
        dist: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for seed in seeds:
            if self.kg.entity(seed) is not None:
                dist[seed] = 0
                queue.append((seed, 0))

        while queue:
            node, d = queue.popleft()
            for neighbor in self.kg.neighbors(node):
                if neighbor not in dist:
                    dist[neighbor] = d + 1
                    queue.append((neighbor, d + 1))

        return dist

    def _dfs_paths(
        self,
        current: str,
        target: str,
        path: list[str],
        paths: list[list[str]],
        max_hops: int,
    ) -> None:
        path = path + [current]
        if current == target:
            paths.append(path)
            return
        if len(path) > max_hops + 1:
            return
        for neighbor in self.kg.neighbors(current):
            if neighbor not in path:  # avoid cycles
                self._dfs_paths(neighbor, target, path, paths, max_hops)
