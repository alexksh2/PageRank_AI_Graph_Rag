"""
experiments.py — 10 structured experiments evaluating crawl heuristics.

EXP-1  Heuristic Ablation Study        — NDCG@k when each QWA signal is zeroed out
EXP-2  p-Sensitivity Analysis          — rank stability as teleportation p varies
EXP-3  Quality Signal Correlation      — pairwise Spearman ρ heatmap of signals
EXP-4  Domain Diversity Analysis       — Gini of domain distribution in top-k
EXP-5  Simulated Crawl Quality Curve   — quality pages discovered per crawl step
EXP-6  Robots Compliance Rate          — fraction of top-k that is crawlable
EXP-7  URL Structural Analysis         — PR vs TLD quality scatter
EXP-8  k-Stability Analysis            — Kendall τ as k varies 5→50
EXP-9  Graph Topology Correlation      — crawl priority vs betweenness centrality
EXP-10 Head-to-Head Comparative        — all H0–H4 quality@step curves

All results are returned as dicts suitable for the MetricsReporter to plot.
"""
from __future__ import annotations

import logging
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy import stats

from .heuristics import (
    BaseCrawlHeuristic, QualityWeightedAuthority,
    build_all, ALL_HEURISTICS,
)
from .quality_proxy import score_url, score_urls

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""


def _is_high_quality(url: str, pr: float, pr_threshold: float = 0.0) -> bool:
    """Ground-truth quality label: high-reputation domain OR high PR."""
    q = score_url(url)
    return q.domain_reputation >= 0.75 or (pr >= pr_threshold and q.tld_score >= 0.70)


def _ndcg(relevance: list[float], k: int) -> float:
    """Compute NDCG@k from a list of relevance scores in ranked order."""
    rel = np.array(relevance[:k], dtype=float)
    if rel.max() == 0:
        return 0.0
    dcg  = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def _gini(arr: np.ndarray) -> float:
    s = np.sort(np.abs(arr) + 1e-12)
    n = len(s)
    return float((2 * np.sum(np.arange(1, n + 1) * s) / (n * s.sum())) - (n + 1) / n)


# ── Experiment results containers ─────────────────────────────────────────────

@dataclass
class AblationResult:
    full_ndcg:    float
    ablations:    dict[str, float]   # signal_name -> NDCG@k without that signal

@dataclass
class PSensitivityResult:
    p_values:       list[float]
    kendall_tau_vs_base: list[float]  # τ vs p=0.15 ranking
    top1_stability: list[str]         # top-1 URL for each p

@dataclass
class CorrelationResult:
    signal_names:  list[str]
    corr_matrix:   np.ndarray

@dataclass
class DomainDiversityResult:
    heuristic_names: list[str]
    gini_scores:     list[float]
    unique_domains:  list[int]

@dataclass
class CrawlQualityResult:
    heuristic_names: list[str]
    steps:           list[int]
    quality_at_step: dict[str, list[float]]  # heuristic_name -> [quality@1, quality@2, ...]

@dataclass
class RobotsComplianceResult:
    heuristic_names:      list[str]
    compliance_at_k:      dict[str, list[float]]  # name -> [compliance@5, @10, @20]
    k_values:             list[int]

@dataclass
class KStabilityResult:
    k_values:    list[int]
    kendall_tau: list[float]   # τ between ranking at k and ranking at k_max

@dataclass
class TopologyResult:
    betweenness_centrality: list[float]
    qwa_scores:             list[float]
    spearman_rho:           float


# ── ExperimentSuite ───────────────────────────────────────────────────────────

class ExperimentSuite:
    """
    Run all 10 crawl heuristic experiments.

    Parameters
    ----------
    graph      : dict[str, list[str]]  — URL adjacency dict
    pageranks  : dict[str, float]      — precomputed PageRank scores
    k          : default cutoff for top-k experiments
    """

    def __init__(
        self,
        graph: dict[str, list[str]],
        pageranks: dict[str, float],
        k: int = 10,
    ):
        self.graph     = graph
        self.pageranks = pageranks
        self.k         = k
        self.urls      = list(graph.keys())
        self.N         = len(self.urls)

        # Ground truth quality labels
        pr_vals = list(pageranks.values())
        pr_med  = float(np.median(pr_vals)) if pr_vals else 0.0
        self.is_quality = {u: _is_high_quality(u, pageranks.get(u, 0.0), pr_med)
                           for u in self.urls}
        self.n_quality = sum(self.is_quality.values())

        # Instantiate all heuristics once
        self._heuristics: list[BaseCrawlHeuristic] = build_all(graph, pageranks)

    # ── EXP-1: Ablation ───────────────────────────────────────────────────────

    def exp1_ablation(self) -> AblationResult:
        """Zero out each QWA signal one at a time; measure NDCG@k drop."""
        logger.info("EXP-1: Ablation study")

        def ndcg_for_weights(**kw):
            h = QualityWeightedAuthority(self.graph, self.pageranks, **kw)
            ranked = h.rank(k=self.k)
            rel = [float(self.is_quality.get(r.url, False)) for r in ranked]
            return _ndcg(rel, self.k)

        full = ndcg_for_weights()
        ablations = {
            "w/o PageRank":    ndcg_for_weights(w_pr=0.0),
            "w/o Domain Rep":  ndcg_for_weights(w_rep=0.0),
            "w/o TLD Quality": ndcg_for_weights(w_tld=0.0),
            "w/o URL Depth":   ndcg_for_weights(w_dep=0.0),
        }
        return AblationResult(full_ndcg=full, ablations=ablations)

    # ── EXP-2: p-Sensitivity ──────────────────────────────────────────────────

    def exp2_p_sensitivity(
        self,
        p_values: Optional[list[float]] = None,
    ) -> PSensitivityResult:
        """Rebuild QWA with PageRank at each p; measure Kendall τ vs p=0.15."""
        from ..pagerank.core import PageRankEngine
        logger.info("EXP-2: p-sensitivity")
        if p_values is None:
            p_values = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.70, 0.85]

        # Build sparse matrix from graph
        idx = {u: i for i, u in enumerate(self.urls)}
        N = self.N
        out_deg = np.array([len(self.graph.get(u, [])) for u in self.urls], dtype=float)
        rows, cols, data = [], [], []
        for src_url, targets in self.graph.items():
            src = idx[src_url]
            for t in targets:
                if t in idx:
                    rows.append(idx[t]); cols.append(src); data.append(1.0)
        if rows:
            A_raw = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=float)
            # Normalise columns
            col_sums = np.array(A_raw.sum(axis=0)).flatten()
            col_sums[col_sums == 0] = 1.0
            A = A_raw.multiply(1.0 / col_sums)
        else:
            A = sp.eye(N, format="csr") * (1.0 / N)
        dangling = out_deg == 0

        pr_by_p: dict[float, dict[str, float]] = {}
        for pv in p_values:
            eng = PageRankEngine(A, dangling, p=pv, tol=1e-8)
            res = eng.run()
            pr_by_p[pv] = {self.urls[i]: float(res.scores[i]) for i in range(N)}

        # Reference ranking at p=0.15
        ref_p = min(p_values, key=lambda x: abs(x - 0.15))
        ref_pr = pr_by_p[ref_p]
        ref_h = QualityWeightedAuthority(self.graph, ref_pr)
        ref_order = [item.url for item in ref_h.rank()]

        taus, top1s = [], []
        for pv in p_values:
            h = QualityWeightedAuthority(self.graph, pr_by_p[pv])
            ranked = h.rank()
            order = [item.url for item in ranked]
            tau, _ = stats.kendalltau(
                [ref_order.index(u) for u in order],
                list(range(len(order)))
            )
            taus.append(float(tau))
            top1s.append(order[0] if order else "")

        return PSensitivityResult(
            p_values=p_values,
            kendall_tau_vs_base=taus,
            top1_stability=top1s,
        )

    # ── EXP-3: Signal Correlation Matrix ──────────────────────────────────────

    def exp3_signal_correlation(self) -> CorrelationResult:
        """Pairwise Spearman ρ between all QWA signals + PageRank."""
        logger.info("EXP-3: signal correlation")
        h = QualityWeightedAuthority(self.graph, self.pageranks)
        ranked = h.rank()

        signals = ["pr_norm", "rep_norm", "tld_norm", "dep_norm"]
        mat = np.zeros((len(signals), len(signals)))
        vecs = {s: np.array([r.signals.get(s, 0.0) for r in ranked]) for s in signals}

        for i, si in enumerate(signals):
            for j, sj in enumerate(signals):
                rho, _ = stats.spearmanr(vecs[si], vecs[sj])
                mat[i, j] = float(rho)

        labels = ["PageRank", "Domain Rep", "TLD Quality", "URL Depth"]
        return CorrelationResult(signal_names=labels, corr_matrix=mat)

    # ── EXP-4: Domain Diversity ───────────────────────────────────────────────

    def exp4_domain_diversity(self) -> DomainDiversityResult:
        """Measure Gini of domain distribution in top-k for each heuristic."""
        logger.info("EXP-4: domain diversity")
        names, ginis, unique = [], [], []
        for h in self._heuristics:
            ranked = h.rank(k=self.k)
            domains = [_domain(r.url) for r in ranked]
            domain_counts = defaultdict(int)
            for d in domains:
                domain_counts[d] += 1
            counts = np.array(list(domain_counts.values()), dtype=float)
            ginis.append(_gini(counts / counts.sum()))
            unique.append(len(domain_counts))
            names.append(h.name)
        return DomainDiversityResult(
            heuristic_names=names, gini_scores=ginis, unique_domains=unique
        )

    # ── EXP-5: Simulated Crawl Quality Curve ─────────────────────────────────

    def exp5_crawl_quality_curve(self) -> CrawlQualityResult:
        """Simulate crawling in priority order; track cumulative quality pages found."""
        logger.info("EXP-5: crawl quality curve")
        steps = list(range(1, self.N + 1))
        quality_at: dict[str, list[float]] = {}

        for h in self._heuristics:
            ranked = h.rank()
            cumulative = []
            found = 0
            for step, item in enumerate(ranked, 1):
                if self.is_quality.get(item.url, False):
                    found += 1
                cumulative.append(found / max(self.n_quality, 1))
            quality_at[h.name] = cumulative

        return CrawlQualityResult(
            heuristic_names=[h.name for h in self._heuristics],
            steps=steps,
            quality_at_step=quality_at,
        )

    # ── EXP-6: Robots Compliance Rate ────────────────────────────────────────

    def exp6_robots_compliance(self) -> RobotsComplianceResult:
        """Fraction of top-k that passes the robots.txt simulation."""
        logger.info("EXP-6: robots compliance")
        k_vals = [5, 10, 20, min(30, self.N)]
        compliance: dict[str, list[float]] = {}

        for h in self._heuristics:
            ranked = h.rank()
            comp_at_k = []
            for k in k_vals:
                top = ranked[:k]
                ok = sum(1 for r in top if score_url(r.url).tld_score >= 0.40)
                comp_at_k.append(ok / max(k, 1))
            compliance[h.name] = comp_at_k

        return RobotsComplianceResult(
            heuristic_names=[h.name for h in self._heuristics],
            compliance_at_k=compliance,
            k_values=k_vals,
        )

    # ── EXP-7: URL Structural Analysis ───────────────────────────────────────

    def exp7_url_structural(self) -> dict:
        """Return per-URL arrays of PR, TLD score, rep score for scatter plot."""
        logger.info("EXP-7: URL structural analysis")
        pr_vals, tld_vals, rep_vals, dep_vals, labels = [], [], [], [], []
        for url in self.urls:
            q = score_url(url)
            pr_vals.append(self.pageranks.get(url, 0.0))
            tld_vals.append(q.tld_score)
            rep_vals.append(q.domain_reputation)
            dep_vals.append(q.url_depth_score)
            labels.append(_domain(url))
        return {
            "urls":     self.urls,
            "pr":       np.array(pr_vals),
            "tld":      np.array(tld_vals),
            "rep":      np.array(rep_vals),
            "depth":    np.array(dep_vals),
            "labels":   labels,
            "quality":  np.array([float(self.is_quality[u]) for u in self.urls]),
        }

    # ── EXP-8: k-Stability ───────────────────────────────────────────────────

    def exp8_k_stability(self) -> KStabilityResult:
        """Kendall τ between top-k ranking and full ranking, as k grows."""
        logger.info("EXP-8: k-stability")
        h = QualityWeightedAuthority(self.graph, self.pageranks)
        full_order = [item.url for item in h.rank()]
        k_vals = [3, 5, 8, 10, 15, 20, min(30, self.N), self.N]
        k_vals = sorted(set(k for k in k_vals if k <= self.N))

        k_max = k_vals[-1]
        base_order = full_order[:k_max]

        taus = []
        for k in k_vals:
            sub = full_order[:k]
            # Compare positions of sub's URLs in base_order
            pos_base = [base_order.index(u) for u in sub if u in base_order]
            pos_sub  = list(range(len(sub)))[:len(pos_base)]
            if len(pos_base) > 2:
                tau, _ = stats.kendalltau(pos_base, pos_sub)
            else:
                tau = 1.0
            taus.append(float(tau))

        return KStabilityResult(k_values=k_vals, kendall_tau=taus)

    # ── EXP-9: Graph Topology ─────────────────────────────────────────────────

    def exp9_topology_correlation(self) -> TopologyResult:
        """Correlate QWA crawl priority with approximate betweenness centrality."""
        logger.info("EXP-9: topology correlation")
        # Approximate betweenness via random-walk sampling (BFS-based)
        idx = {u: i for i, u in enumerate(self.urls)}
        N = self.N
        between = np.zeros(N)

        # Simple proxy: for each node, count how many shortest paths from
        # BFS samples pass through it. We use 50 random source samples.
        rng = np.random.RandomState(42)
        samples = rng.choice(N, min(50, N), replace=False)
        from collections import deque
        for src_idx in samples:
            src = self.urls[src_idx]
            dist = {src: 0}
            parents: dict[str, list[str]] = defaultdict(list)
            q: deque = deque([src])
            while q:
                node = q.popleft()
                for nbr in self.graph.get(node, []):
                    if nbr not in dist:
                        dist[nbr] = dist[node] + 1
                        parents[nbr].append(node)
                        q.append(nbr)
                    elif dist[nbr] == dist[node] + 1:
                        parents[nbr].append(node)
            # Count path contributions
            for tgt, par_list in parents.items():
                if tgt != src and tgt in idx:
                    between[idx[tgt]] += len(par_list)

        between = between / between.sum() if between.sum() > 0 else between

        # QWA scores
        h = QualityWeightedAuthority(self.graph, self.pageranks)
        ranked = h.rank()
        score_map = {r.url: r.score for r in ranked}
        qwa_scores = np.array([score_map.get(u, 0.0) for u in self.urls])

        rho, _ = stats.spearmanr(between, qwa_scores)
        return TopologyResult(
            betweenness_centrality=between.tolist(),
            qwa_scores=qwa_scores.tolist(),
            spearman_rho=float(rho),
        )

    # ── EXP-10: Head-to-Head Comparative ─────────────────────────────────────

    def exp10_head_to_head(self) -> dict:
        """
        For each heuristic compute:
          • NDCG@k
          • Precision@k
          • Recall@k
          • Mean quality score of top-k
          • Domain diversity Gini
        Return a dict of {heuristic_name: {metric: value}}.
        """
        logger.info("EXP-10: head-to-head comparison")
        results = {}
        for h in self._heuristics:
            ranked = h.rank(k=self.k)
            rel = [float(self.is_quality.get(r.url, False)) for r in ranked]
            hits = sum(rel)
            mean_quality = float(np.mean([score_url(r.url).composite for r in ranked]))
            domains = [_domain(r.url) for r in ranked]
            dcounts = defaultdict(int)
            for d in domains:
                dcounts[d] += 1
            dcounts_arr = np.array(list(dcounts.values()), dtype=float)

            results[h.name] = {
                "NDCG@k":          _ndcg(rel, self.k),
                "Precision@k":     hits / self.k,
                "Recall@k":        hits / max(self.n_quality, 1),
                "Mean Quality":    mean_quality,
                "Domain Gini":     _gini(dcounts_arr / dcounts_arr.sum()),
                "Unique Domains":  len(dcounts),
            }
        return results

    # ── Run all ───────────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        """Run all 10 experiments; return a combined results dict."""
        return {
            "exp1_ablation":       self.exp1_ablation(),
            "exp2_p_sensitivity":  self.exp2_p_sensitivity(),
            "exp3_correlation":    self.exp3_signal_correlation(),
            "exp4_diversity":      self.exp4_domain_diversity(),
            "exp5_quality_curve":  self.exp5_crawl_quality_curve(),
            "exp6_robots":         self.exp6_robots_compliance(),
            "exp7_structural":     self.exp7_url_structural(),
            "exp8_k_stability":    self.exp8_k_stability(),
            "exp9_topology":       self.exp9_topology_correlation(),
            "exp10_head_to_head":  self.exp10_head_to_head(),
        }
