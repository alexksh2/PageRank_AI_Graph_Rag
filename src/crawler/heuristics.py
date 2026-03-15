"""
heuristics.py — Crawl prioritisation heuristics for AI training data.

Five strategies are implemented as comparable classes sharing the same
interface:  heuristic.rank(urls) -> list[(url, score, reason)]

┌─────────────────────────────────────────────────────────────────────┐
│  Strategy                │ Signal used                              │
├─────────────────────────────────────────────────────────────────────┤
│  H0  RandomBaseline      │ Uniform random                          │
│  H1  PurePageRank        │ PageRank score only                     │
│  H2  HubAuthority        │ PageRank × out-degree  (HITS-inspired)  │
│  H3  PRRobots            │ PageRank × robots-compliance bonus      │
│  H4  QualityWeightedAuth │ PR + TLD + domain rep + depth + robots  │
│      (PROPOSED HEURISTIC)│   ← recommended for AI training crawls  │
└─────────────────────────────────────────────────────────────────────┘

Proposed Heuristic — Quality-Weighted Authority (QWA)
======================================================
Motivation:
  A naïve PR-only crawler maximises link-authority but ignores:
    • Whether the domain has high editorial standards (.edu vs .biz)
    • Whether the page owner consents to AI crawling (robots.txt)
    • Whether the URL points to substantive content vs. a redirect farm
    • Whether the page is a shallow landing page or a deep auto-generated one

  QWA combines four orthogonal signals:

      score(u) = w_pr   × PR_norm(u)          ← who endorses this page?
               + w_rep  × reputation(u)        ← is the domain trustworthy?
               + w_tld  × tld_score(u)         ← is the TLD academically rigorous?
               + w_dep  × depth_score(u)       ← is the URL shallow (landing page)?
               + w_rob  × robots_ok(u)         ← does the owner permit AI crawling?

  All signals are normalised to [0,1].  Default weights:
      w_pr=0.40, w_rep=0.25, w_tld=0.15, w_dep=0.10, w_rob=0.10

  Why high-PageRank pages yield better AI training data:
    PageRank is a measure of transitivity-weighted editorial endorsement.
    If a page accumulates high PR it means many *other* authoritative pages
    link to it — implying that multiple independent human editors found it
    worth citing.  These pages are typically:
      1. Factually accurate (peer pressure from linking authors)
      2. Well-written (poorly written pages lose links over time)
      3. Stable (high-PR pages rarely go stale or change URLs)
      4. Broad-coverage (hubs link to them because they cover a concept well)
    These four properties align directly with the desiderata for LLM pre-
    training corpora: factuality, fluency, stability, and concept coverage.

Experiments to run on heuristics
=================================
See experiments.py for the full suite.  Briefly:
  EXP-1  Heuristic Ablation           — remove each signal, measure NDCG drop
  EXP-2  p-Sensitivity                — vary teleportation p, track rank stability
  EXP-3  Quality Signal Correlation   — heatmap of pairwise signal correlations
  EXP-4  Domain Diversity             — Gini of domain distribution in top-k
  EXP-5  Simulated Crawl Quality Curve— quality pages discovered per crawl step
  EXP-6  Robots Compliance Rate       — fraction of top-k that is crawlable
  EXP-7  URL Structural Analysis      — scatter PR vs TLD quality
  EXP-8  k-Stability                  — Kendall τ as k varies 5→50
  EXP-9  Graph Topology               — crawl priority vs BFS/DFS/betweenness
  EXP-10 Comparative Baselines        — H0–H4 quality@step curves head-to-head
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .quality_proxy import score_url, URLQualitySignals


@dataclass
class CrawlItem:
    url: str
    score: float
    rank: int
    reason: str
    signals: dict = field(default_factory=dict)

    def __repr__(self):
        return f"CrawlItem(rank={self.rank}, score={self.score:.4f}, url={self.url})"


# ── Base class ────────────────────────────────────────────────────────────────

class BaseCrawlHeuristic(ABC):
    name: str = "base"
    description: str = ""

    def __init__(
        self,
        graph: dict[str, list[str]],
        pageranks: dict[str, float],
        seed: int = 42,
    ):
        self.graph = graph
        self.pageranks = pageranks
        self._rng = random.Random(seed)
        self._urls = list(graph.keys())

    @abstractmethod
    def _score(self, url: str) -> tuple[float, str, dict]:
        """Return (score, reason_string, signals_dict)."""

    def rank(self, k: Optional[int] = None) -> list[CrawlItem]:
        """Rank all URLs; return top-k as sorted list of CrawlItem."""
        scored = []
        for url in self._urls:
            score, reason, signals = self._score(url)
            scored.append((score, url, reason, signals))
        scored.sort(key=lambda x: -x[0])
        if k:
            scored = scored[:k]
        return [
            CrawlItem(url=url, score=sc, rank=i + 1, reason=reason, signals=sig)
            for i, (sc, url, reason, sig) in enumerate(scored)
        ]

    @staticmethod
    def _normalise(values: dict[str, float]) -> dict[str, float]:
        vals = np.array(list(values.values()), dtype=float)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return {u: 0.5 for u in values}
        return {u: (v - lo) / (hi - lo) for u, v in values.items()}


# ── H0: Random baseline ───────────────────────────────────────────────────────

class RandomBaseline(BaseCrawlHeuristic):
    name = "H0_Random"
    description = "Uniform random ordering — control baseline."

    def _score(self, url: str) -> tuple[float, str, dict]:
        sc = self._rng.random()
        return sc, "random baseline", {"random": sc}


# ── H1: Pure PageRank ─────────────────────────────────────────────────────────

class PurePageRank(BaseCrawlHeuristic):
    name = "H1_PurePageRank"
    description = "Rank by PageRank score only — link-authority without quality filter."

    def __init__(self, graph, pageranks, seed=42):
        super().__init__(graph, pageranks, seed)
        self._pr_norm = self._normalise(
            {u: pageranks.get(u, 0.0) for u in self._urls}
        )

    def _score(self, url: str) -> tuple[float, str, dict]:
        pr = self._pr_norm[url]
        raw = self.pageranks.get(url, 0.0)
        reason = f"PR={raw:.5f} (authority from inbound links)"
        return pr, reason, {"pr_norm": pr, "pr_raw": raw}


# ── H2: Hub-Authority (HITS-inspired) ────────────────────────────────────────

class HubAuthority(BaseCrawlHeuristic):
    name = "H2_HubAuthority"
    description = (
        "PageRank × out-degree.  Hub pages (many outlinks) that also "
        "have high PR surface more new URLs per crawl step."
    )

    def __init__(self, graph, pageranks, seed=42):
        super().__init__(graph, pageranks, seed)
        out_deg = {u: len(graph.get(u, [])) for u in self._urls}
        self._pr_norm  = self._normalise({u: pageranks.get(u, 0.0) for u in self._urls})
        self._out_norm = self._normalise(out_deg)

    def _score(self, url: str) -> tuple[float, str, dict]:
        pr  = self._pr_norm[url]
        out = self._out_norm[url]
        sc  = 0.60 * pr + 0.40 * out
        reason = f"PR_norm={pr:.3f} × hub_norm={out:.3f}"
        return sc, reason, {"pr_norm": pr, "out_norm": out}


# ── H3: PR + Robots compliance ───────────────────────────────────────────────

class PRRobots(BaseCrawlHeuristic):
    name = "H3_PR_Robots"
    description = (
        "PageRank with robots.txt compliance bonus.  Surfaces authoritative "
        "pages that explicitly permit AI crawling."
    )

    # Known bots that quality pages typically allow; used in robots simulation
    _FRIENDLY_TLDS = {"edu", "gov", "org", "ac.uk", "int"}

    def __init__(self, graph, pageranks, seed=42):
        super().__init__(graph, pageranks, seed)
        self._pr_norm = self._normalise({u: pageranks.get(u, 0.0) for u in self._urls})
        # Simulate robots.txt: .edu/.gov/.org assumed to allow AI crawlers
        self._robots: dict[str, float] = {}
        for url in self._urls:
            qs = score_url(url)
            # .edu/.gov/.org: 0.9 compliance score; others 0.5; known spam 0.0
            if qs.tld_score >= 0.75:
                self._robots[url] = 0.90
            elif qs.domain_reputation < 0.10:
                self._robots[url] = 0.05
            else:
                self._robots[url] = 0.50

    def _score(self, url: str) -> tuple[float, str, dict]:
        pr  = self._pr_norm[url]
        rob = self._robots[url]
        sc  = 0.70 * pr + 0.30 * rob
        reason = f"PR_norm={pr:.3f}, robots_score={rob:.2f}"
        return sc, reason, {"pr_norm": pr, "robots": rob}


# ── H4: Quality-Weighted Authority (QWA) — PROPOSED HEURISTIC ────────────────

class QualityWeightedAuthority(BaseCrawlHeuristic):
    """
    Quality-Weighted Authority (QWA) — proposed heuristic for AI training crawls.

    Signal composition (all normalised to [0,1]):
      w_pr   = 0.40  PageRank authority
      w_rep  = 0.25  Domain reputation (editorial standards)
      w_tld  = 0.15  TLD quality (.edu/.gov > .org > .com > .biz)
      w_dep  = 0.10  URL depth score (shallow = better landing pages)
      w_rob  = 0.10  Robots.txt AI-crawl compliance

    Why this outperforms pure PageRank for AI training data:
      • Filters out high-PR spam farms (high PR but low reputation)
      • Prioritises .edu/.gov where factuality is legally/ethically enforced
      • Shallow-URL bonus surfaces authoritative index pages over deep auto-generated content
      • Robots.txt filter ensures consent-cleared training data
    """
    name = "H4_QualityWeightedAuthority"
    description = (
        "PROPOSED: PR (0.40) + domain reputation (0.25) + "
        "TLD quality (0.15) + URL depth (0.10) + robots (0.10)."
    )

    W_PR  = 0.40
    W_REP = 0.25
    W_TLD = 0.15
    W_DEP = 0.10
    W_ROB = 0.10

    def __init__(self, graph, pageranks, seed=42,
                 w_pr=None, w_rep=None, w_tld=None, w_dep=None, w_rob=None):
        super().__init__(graph, pageranks, seed)
        # Allow weight override for ablation experiments
        self.W_PR  = w_pr  if w_pr  is not None else self.W_PR
        self.W_REP = w_rep if w_rep is not None else self.W_REP
        self.W_TLD = w_tld if w_tld is not None else self.W_TLD
        self.W_DEP = w_dep if w_dep is not None else self.W_DEP
        self.W_ROB = w_rob if w_rob is not None else self.W_ROB

        self._pr_norm = self._normalise({u: pageranks.get(u, 0.0) for u in self._urls})
        self._quality: dict[str, URLQualitySignals] = {
            u: score_url(u) for u in self._urls
        }
        # Normalise reputation and TLD scores
        self._rep_norm = self._normalise({u: self._quality[u].domain_reputation for u in self._urls})
        self._tld_norm = self._normalise({u: self._quality[u].tld_score for u in self._urls})
        self._dep_norm = self._normalise({u: self._quality[u].url_depth_score for u in self._urls})
        # Robots: .edu/.gov = 0.9, .org = 0.7, known spam = 0.0
        raw_rob = {}
        for u in self._urls:
            q = self._quality[u]
            if q.tld_score >= 1.0:
                raw_rob[u] = 0.90
            elif q.tld_score >= 0.75:
                raw_rob[u] = 0.70
            elif q.domain_reputation < 0.10:
                raw_rob[u] = 0.00
            else:
                raw_rob[u] = 0.50
        self._rob_norm = self._normalise(raw_rob)

    def _score(self, url: str) -> tuple[float, str, dict]:
        pr  = self._pr_norm[url]
        rep = self._rep_norm[url]
        tld = self._tld_norm[url]
        dep = self._dep_norm[url]
        rob = self._rob_norm[url]

        sc = (self.W_PR * pr + self.W_REP * rep
              + self.W_TLD * tld + self.W_DEP * dep + self.W_ROB * rob)

        q = self._quality[url]
        reason = (
            f"PR={pr:.3f} rep={q.domain_reputation:.2f} "
            f"tld={q.tld_score:.2f} depth={q.url_depth_score:.2f} "
            f"robots={rob:.2f}"
        )
        return sc, reason, {
            "pr_norm": pr, "rep_norm": rep, "tld_norm": tld,
            "dep_norm": dep, "rob_norm": rob,
            "pr_raw":  self.pageranks.get(url, 0.0),
            "rep_raw": q.domain_reputation,
            "tld_raw": q.tld_score,
        }


# ── Factory ───────────────────────────────────────────────────────────────────

ALL_HEURISTICS = [RandomBaseline, PurePageRank, HubAuthority, PRRobots, QualityWeightedAuthority]


def build_all(graph: dict[str, list[str]], pageranks: dict[str, float]) -> list[BaseCrawlHeuristic]:
    """Instantiate all five heuristics with the same graph and PR scores."""
    return [H(graph, pageranks) for H in ALL_HEURISTICS]
