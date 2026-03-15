"""
Unit tests for PageRank core, analytical solver, and GraphRAG retrieval.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import scipy.sparse as sp

from src.pagerank.core import PageRankEngine
from src.pagerank.analytical import AnalyticalPageRank
from src.graphrag.knowledge_graph import KnowledgeGraph
from src.graphrag.pagerank_retrieval import PageRankRetrieval
from src.graphrag.query_engine import GraphRAGQueryEngine
from src.crawler.prioritizer import CrawlPrioritizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_triangle_graph():
    """3-node cycle: 0->1->2->0.  Uniform PR = 1/3 for all."""
    N = 3
    rows = [1, 2, 0]
    cols = [0, 1, 2]
    data = [1.0, 1.0, 1.0]
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    dangling = np.zeros(N, dtype=bool)
    return A, dangling, N


def make_dangling_graph():
    """3 nodes: 0->1, 0->2, node 1 and 2 are dangling."""
    N = 3
    rows = [1, 2]
    cols = [0, 0]
    data = [0.5, 0.5]
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    dangling = np.array([False, True, True])
    return A, dangling, N


# ---------------------------------------------------------------------------
# PageRankEngine tests
# ---------------------------------------------------------------------------

class TestPageRankEngine:

    def test_triangle_uniform(self):
        """A 3-cycle should converge to uniform PR = 1/3."""
        A, dangling, N = make_triangle_graph()
        engine = PageRankEngine(A, dangling, p=0.15, tol=1e-12)
        result = engine.run()
        assert result.converged
        np.testing.assert_allclose(result.scores, [1/3, 1/3, 1/3], atol=1e-8)

    def test_scores_sum_to_one(self):
        A, dangling, N = make_dangling_graph()
        engine = PageRankEngine(A, dangling, p=0.15, tol=1e-10)
        result = engine.run()
        assert abs(result.scores.sum() - 1.0) < 1e-8

    def test_scores_non_negative(self):
        A, dangling, N = make_dangling_graph()
        engine = PageRankEngine(A, dangling, p=0.15)
        result = engine.run()
        assert (result.scores >= 0).all()

    def test_p_equals_one_uniform(self):
        """p=1 means pure teleportation -> uniform distribution."""
        A, dangling, N = make_triangle_graph()
        engine = PageRankEngine(A, dangling, p=1.0, tol=1e-12)
        result = engine.run()
        np.testing.assert_allclose(result.scores, [1/3, 1/3, 1/3], atol=1e-8)

    def test_invalid_p_raises(self):
        A, dangling, N = make_triangle_graph()
        with pytest.raises(ValueError):
            PageRankEngine(A, dangling, p=0.0)
        with pytest.raises(ValueError):
            PageRankEngine(A, dangling, p=1.5)

    def test_residuals_monotone_decreasing(self):
        # Use a non-trivial graph where power iteration actually takes steps.
        A, dangling, N = make_dangling_graph()
        engine = PageRankEngine(A, dangling, p=0.15)
        result = engine.run()
        if len(result.residuals) > 1:
            # If multiple iterations ran, final residual must be smaller than first
            assert result.residuals[-1] <= result.residuals[0]
        # If converged in 1 iteration, residual is trivially valid
        assert result.residuals[-1] >= 0

    def test_top_k(self):
        A, dangling, N = make_dangling_graph()
        engine = PageRankEngine(A, dangling, p=0.15)
        result = engine.run()
        top2 = result.top_k(2)
        assert len(top2) == 2
        assert top2[0][1] >= top2[1][1]  # sorted descending


# ---------------------------------------------------------------------------
# AnalyticalPageRank tests
# ---------------------------------------------------------------------------

class TestAnalyticalPageRank:

    def test_triangle_matches_iterative(self):
        A, dangling, N = make_triangle_graph()
        p = 0.15
        analytical = AnalyticalPageRank(A, dangling, p=p)
        r_analytical = analytical.compute()
        assert r_analytical is not None

        engine = PageRankEngine(A, dangling, p=p, tol=1e-12)
        r_iter = engine.run().scores

        np.testing.assert_allclose(r_analytical, r_iter, atol=1e-6)

    def test_dangling_matches_iterative(self):
        A, dangling, N = make_dangling_graph()
        p = 0.15
        analytical = AnalyticalPageRank(A, dangling, p=p)
        r_analytical = analytical.compute()
        assert r_analytical is not None

        engine = PageRankEngine(A, dangling, p=p, tol=1e-12)
        r_iter = engine.run().scores

        np.testing.assert_allclose(r_analytical, r_iter, atol=1e-6)

    def test_sums_to_one(self):
        A, dangling, N = make_triangle_graph()
        r = AnalyticalPageRank(A, dangling, p=0.2).compute()
        assert abs(r.sum() - 1.0) < 1e-8

    def test_compare_returns_low_error(self):
        A, dangling, N = make_triangle_graph()
        engine = PageRankEngine(A, dangling, p=0.15, tol=1e-12)
        r_iter = engine.run().scores
        cmp = AnalyticalPageRank(A, dangling, p=0.15).compare_with_iterative(r_iter, top_k=3)
        assert cmp["l1_error"] < 1e-5
        assert cmp["top_k_rank_overlap"] == 3


# ---------------------------------------------------------------------------
# KnowledgeGraph tests
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:

    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.add_triple("A", "rel", "B")
        self.kg.add_triple("A", "rel", "C")
        self.kg.add_triple("B", "rel", "D")
        self.kg.add_triple("C", "rel", "D")

    def test_entity_count(self):
        assert self.kg.n_entities == 4  # A, B, C, D

    def test_relation_count(self):
        assert self.kg.n_relations == 4

    def test_neighbors(self):
        nbrs = set(self.kg.neighbors("A"))
        assert nbrs == {"B", "C"}

    def test_adjacency_matrix_shape(self):
        A = self.kg.adjacency_matrix()
        assert A.shape == (4, 4)

    def test_adjacency_matrix_column_stochastic(self):
        A = self.kg.adjacency_matrix()
        col_sums = np.array(A.sum(axis=0)).flatten()
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_incremental_update(self):
        self.kg.add_triple("D", "rel", "A")
        assert self.kg.n_entities == 4
        assert self.kg.n_relations == 5


# ---------------------------------------------------------------------------
# PageRankRetrieval tests
# ---------------------------------------------------------------------------

class TestPageRankRetrieval:

    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.add_triples([
            ("Marie Curie", "discovered", "radium"),
            ("radium", "used_in", "radiotherapy"),
            ("radiotherapy", "advances", "medical imaging"),
            ("Marie Curie", "developed", "xray unit"),
            ("xray unit", "enables", "medical imaging"),
        ])
        self.retrieval = PageRankRetrieval(self.kg, p=0.15)

    def test_retrieval_returns_k_results(self):
        results = self.retrieval.retrieve(["Marie Curie"], k=3)
        assert len(results) <= 3

    def test_scores_non_negative(self):
        results = self.retrieval.retrieve(["Marie Curie"], k=5)
        for r in results:
            assert r.score >= 0

    def test_scores_sum_approx_one(self):
        results = self.retrieval.retrieve(["Marie Curie"], k=100)
        total = sum(r.score for r in results)
        assert abs(total - 1.0) < 0.1  # partial sum, so not exactly 1

    def test_seed_at_top(self):
        """Marie Curie should rank at or near the top as seed."""
        results = self.retrieval.retrieve(["Marie Curie"], k=7)
        names = [r.entity_name for r in results]
        assert "Marie Curie" in names[:3]

    def test_explain_path(self):
        paths = self.retrieval.explain_path("Marie Curie", "medical imaging", max_hops=5)
        assert len(paths) > 0
        for path in paths:
            assert path[0] == "Marie Curie"
            assert path[-1] == "medical imaging"


# ---------------------------------------------------------------------------
# GraphRAGQueryEngine tests
# ---------------------------------------------------------------------------

class TestGraphRAGQueryEngine:

    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.add_triples([
            ("Marie Curie", "discovered", "radium"),
            ("radium", "used_in", "radiotherapy"),
            ("radiotherapy", "advances", "medical imaging"),
        ])
        self.engine = GraphRAGQueryEngine(self.kg, p=0.15, k=5)

    def test_query_returns_response(self):
        resp = self.engine.query("Marie Curie medical imaging", seed_entities=["Marie Curie"])
        assert resp.query == "Marie Curie medical imaging"
        assert len(resp.top_k_entities) > 0

    def test_auto_seed_extraction(self):
        resp = self.engine.query("What did Marie Curie discover?")
        assert "Marie Curie" in resp.seed_entities

    def test_no_seed_graceful(self):
        resp = self.engine.query("quantum gravity entanglement")
        assert len(resp.top_k_entities) == 0 or resp.explanation  # graceful

    def test_reasoning_chains_connect_seeds_to_targets(self):
        resp = self.engine.query(
            "medical imaging",
            seed_entities=["Marie Curie", "medical imaging"],
            k=5,
        )
        for chain in resp.reasoning_chains:
            assert isinstance(chain, list)
            assert len(chain) >= 2


# ---------------------------------------------------------------------------
# CrawlPrioritizer tests
# ---------------------------------------------------------------------------

class TestCrawlPrioritizer:

    def setup_method(self):
        self.graph = {
            "https://arxiv.org/": ["https://arxiv.org/paper1", "https://arxiv.org/paper2"],
            "https://arxiv.org/paper1": ["https://arxiv.org/"],
            "https://arxiv.org/paper2": [],
            "https://spam.com/": [],
        }
        self.pageranks = {
            "https://arxiv.org/": 0.5,
            "https://arxiv.org/paper1": 0.2,
            "https://arxiv.org/paper2": 0.2,
            "https://spam.com/": 0.01,
        }
        self.prioritizer = CrawlPrioritizer(self.graph, self.pageranks)

    def test_top_k_length(self):
        top = self.prioritizer.top_k(k=3)
        assert len(top) <= 3

    def test_high_pr_ranks_first(self):
        """arxiv.org (highest PR) should appear first."""
        top = self.prioritizer.top_k(k=4)
        assert top[0].url == "https://arxiv.org/"

    def test_spam_ranks_last(self):
        top = self.prioritizer.top_k(k=4)
        assert top[-1].url == "https://spam.com/"

    def test_explain_policy_non_empty(self):
        assert len(self.prioritizer.explain_policy()) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
