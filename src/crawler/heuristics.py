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
│  H3  PRRobots            │ Hard robots gate → rank by PageRank     │
│  H4  QualityWeightedAuth │ Hard robots gate → PR + TLD + rep + dep │
│      (PROPOSED HEURISTIC)│   ← recommended for AI training crawls  │
└─────────────────────────────────────────────────────────────────────┘

Robots.txt policy
=================
Robots compliance is treated as a HARD GATE, not a soft bonus.
H3 and H4 first fetch robots.txt for each domain and discard any URL
whose domain blocks AI crawlers (GPTBot, CCBot, anthropic-ai, ClaudeBot).
Only permitted URLs proceed to scoring. A blocked URL can never appear
in the output regardless of its PageRank.
H0–H2 do not apply a robots filter (used as baselines for comparison).

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
from .prioritizer import fetch_robots_permitted


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

    def _robots_permitted(self, url: str) -> bool:
        """Hard robots gate: fetch real robots.txt and check AI crawler permissions."""
        return fetch_robots_permitted(url)

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

    def rank_gated(self, k: Optional[int] = None) -> list[CrawlItem]:
        """Rank with hard robots gate applied before scoring."""
        permitted = [u for u in self._urls if self._robots_permitted(u)]
        scored = []
        for url in permitted:
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
    """
    H2 — Hub-Authority (HITS-inspired).

    score(u) = 0.60 × PR_norm(u) + 0.40 × OutDegree_norm(u)

    Rationale: a page that is both authoritative (high PageRank) and a hub
    (many outlinks) is prioritised because crawling it surfaces more new URLs
    per step while still being endorsed by the link graph.
    No robots gate applied — used as a baseline for comparison with H3/H4.
    """
    name = "H2_HubAuthority"
    description = (
        "0.60 × PageRank_norm + 0.40 × OutDegree_norm. "
        "Prioritises authoritative hub pages that surface many new URLs per crawl step."
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


# ── H3: PR + Hard Robots Gate ─────────────────────────────────────────────────

class PRRobots(BaseCrawlHeuristic):
    name = "H3_PR_Robots"
    description = (
        "Hard robots gate then rank by PageRank. "
        "Blocked domains are excluded entirely before scoring."
    )

    def __init__(self, graph, pageranks, seed=42):
        super().__init__(graph, pageranks, seed)
        self._pr_norm = self._normalise({u: pageranks.get(u, 0.0) for u in self._urls})

    def _score(self, url: str) -> tuple[float, str, dict]:
        pr = self._pr_norm[url]
        reason = f"PR_norm={pr:.3f} (robots-permitted)"
        return pr, reason, {"pr_norm": pr}

    def rank(self, k: Optional[int] = None) -> list[CrawlItem]:
        """Apply hard robots gate before ranking."""
        return self.rank_gated(k)


# ── H4: Quality-Weighted Authority (QWA) — PROPOSED HEURISTIC ────────────────

class QualityWeightedAuthority(BaseCrawlHeuristic):
    """
    Quality-Weighted Authority (QWA) — proposed heuristic for AI training crawls.

    Two-stage design:
      Stage 1 — Hard robots gate: fetch robots.txt and discard domains blocking AI crawlers
      Stage 2 — Score permitted URLs by four orthogonal signals (all normalised [0,1]):
        w_pr   = 0.45  PageRank authority
        w_rep  = 0.30  Domain reputation (editorial standards)
        w_tld  = 0.15  TLD quality (.edu/.gov > .org > .com > .biz)
        w_dep  = 0.10  URL depth score (shallow = better landing pages)

    Robots is a hard gate, not a soft bonus — a blocked URL can never appear
    in the output regardless of how high its PageRank is.
    """
    name = "H4_QualityWeightedAuthority"
    description = (
        "PROPOSED: Hard robots gate → PR (0.45) + domain reputation (0.30) + "
        "TLD quality (0.15) + URL depth (0.10)."
    )

    W_PR  = 0.45
    W_REP = 0.30
    W_TLD = 0.15
    W_DEP = 0.10

    def __init__(self, graph, pageranks, seed=42,
                 w_pr=None, w_rep=None, w_tld=None, w_dep=None):
        super().__init__(graph, pageranks, seed)
        # Allow weight override for ablation experiments
        self.W_PR  = w_pr  if w_pr  is not None else self.W_PR
        self.W_REP = w_rep if w_rep is not None else self.W_REP
        self.W_TLD = w_tld if w_tld is not None else self.W_TLD
        self.W_DEP = w_dep if w_dep is not None else self.W_DEP

        self._pr_norm = self._normalise({u: pageranks.get(u, 0.0) for u in self._urls})
        self._quality: dict[str, URLQualitySignals] = {
            u: score_url(u) for u in self._urls
        }
        self._rep_norm = self._normalise({u: self._quality[u].domain_reputation for u in self._urls})
        self._tld_norm = self._normalise({u: self._quality[u].tld_score for u in self._urls})
        self._dep_norm = self._normalise({u: self._quality[u].url_depth_score for u in self._urls})

    def _score(self, url: str) -> tuple[float, str, dict]:
        pr  = self._pr_norm[url]
        rep = self._rep_norm[url]
        tld = self._tld_norm[url]
        dep = self._dep_norm[url]

        sc = self.W_PR * pr + self.W_REP * rep + self.W_TLD * tld + self.W_DEP * dep

        q = self._quality[url]
        reason = (
            f"PR={pr:.3f} rep={q.domain_reputation:.2f} "
            f"tld={q.tld_score:.2f} depth={q.url_depth_score:.2f}"
        )
        return sc, reason, {
            "pr_norm": pr, "rep_norm": rep, "tld_norm": tld, "dep_norm": dep,
            "pr_raw":  self.pageranks.get(url, 0.0),
            "rep_raw": q.domain_reputation,
            "tld_raw": q.tld_score,
        }

    def rank(self, k: Optional[int] = None) -> list[CrawlItem]:
        """Apply hard robots gate before scoring."""
        return self.rank_gated(k)


# ── Factory ───────────────────────────────────────────────────────────────────

ALL_HEURISTICS = [RandomBaseline, PurePageRank, HubAuthority, PRRobots, QualityWeightedAuthority]


def build_all(graph: dict[str, list[str]], pageranks: dict[str, float]) -> list[BaseCrawlHeuristic]:
    """Instantiate all five heuristics with the same graph and PR scores."""
    return [H(graph, pageranks) for H in ALL_HEURISTICS]
