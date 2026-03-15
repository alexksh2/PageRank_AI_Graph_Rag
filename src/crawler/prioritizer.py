"""
CrawlPrioritizer: PageRank-aware URL prioritisation for AI training data crawling.

============================================================
DESIGN RATIONALE
============================================================

When building training corpora for large language models (GPTBot-style),
not all web pages are equally valuable.  High-PageRank pages tend to be:

  1. Authoritative  -- many trusted sites link to them.
  2. Information-dense -- well-curated hubs (Wikipedia, arXiv, official docs).
  3. Low-noise  -- spammy pages rarely accumulate many quality backlinks.

We combine three signals into a composite PRIORITY SCORE:

    priority(u) = w_pr * norm(PR(u))                  (PageRank authority)
               + w_out * norm(out_degree(u))           (hub potential: many links to follow)
               + w_robots * robots_bonus(u)            (page respects/signals crawlability)

All components are normalised to [0, 1] before weighting so the weights
are interpretable percentages.

Heuristic for "high quality AND permits crawling":
  - Download robots.txt for each domain.
  - Pages whose domain does NOT block GPTBot/CCBot/OAI-SearchBot receive a
    positive robots_bonus.
  - This surfaces authoritative pages that actively welcome AI crawlers,
    which are more likely to contain consent-cleared training data.
============================================================
"""

from __future__ import annotations

import heapq
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np

logger = logging.getLogger(__name__)

# Known AI crawler user-agent tokens to check in robots.txt
_AI_BOTS = ["GPTBot", "CCBot", "OAI-SearchBot", "anthropic-ai", "Googlebot"]
_ROBOTS_DISALLOW_RE = re.compile(r"^Disallow\s*:\s*(.+)$", re.IGNORECASE)
_USERAGENT_RE = re.compile(r"^User-agent\s*:\s*(.+)$", re.IGNORECASE)


@dataclass(order=True)
class CrawlCandidate:
    """A URL candidate with its composite crawl-priority score."""

    priority: float = field(compare=True)  # negated for min-heap use
    url: str = field(compare=False)
    pagerank: float = field(compare=False)
    out_degree: int = field(compare=False)
    robots_ok: bool = field(compare=False)
    reason: str = field(compare=False, default="")


class CrawlPrioritizer:
    """
    Rank a set of discovered URLs by crawl priority for AI training data.

    Parameters
    ----------
    graph       : dict[str, list[str]]  — adjacency dict  {url: [outlink_urls]}
    pageranks   : dict[str, float]      — precomputed PageRank scores
    w_pr        : weight for PageRank signal   (default 0.6)
    w_out       : weight for out-degree signal (default 0.2)
    w_robots    : weight for robots.txt signal (default 0.2)
    check_robots: whether to fetch robots.txt files (default False; requires network)
    """

    def __init__(
        self,
        graph: dict[str, list[str]],
        pageranks: dict[str, float],
        w_pr: float = 0.6,
        w_out: float = 0.2,
        w_robots: float = 0.2,
        check_robots: bool = False,
    ) -> None:
        self.graph = graph
        self.pageranks = pageranks
        self.w_pr = w_pr
        self.w_out = w_out
        self.w_robots = w_robots
        self.check_robots = check_robots
        self._robots_cache: dict[str, bool] = {}  # domain -> True if crawlable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def top_k(self, k: int = 10) -> list[CrawlCandidate]:
        """Return the top-k URLs to crawl next, ranked by composite priority."""
        urls = list(self.graph.keys())
        if not urls:
            return []

        pr_arr = np.array([self.pageranks.get(u, 0.0) for u in urls], dtype=float)
        od_arr = np.array([len(self.graph.get(u, [])) for u in urls], dtype=float)

        pr_norm = self._normalise(pr_arr)
        od_norm = self._normalise(od_arr)

        # Build robots signal
        robots_arr = np.zeros(len(urls))
        if self.check_robots:
            for i, url in enumerate(urls):
                robots_arr[i] = 1.0 if self._is_crawlable(url) else 0.0
        else:
            robots_arr[:] = 1.0  # assume all crawlable if not checking

        priority_arr = (
            self.w_pr * pr_norm
            + self.w_out * od_norm
            + self.w_robots * robots_arr
        )

        # Use a heap to get top-k efficiently (O(N log k))
        heap: list[tuple[float, int]] = []
        for i, score in enumerate(priority_arr):
            if len(heap) < k:
                heapq.heappush(heap, (score, i))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, i))

        top_indices = sorted([i for _, i in heap], key=lambda i: -priority_arr[i])

        candidates = []
        for i in top_indices:
            url = urls[i]
            reason = self._explain(pr_norm[i], od_norm[i], robots_arr[i])
            candidates.append(
                CrawlCandidate(
                    priority=float(priority_arr[i]),
                    url=url,
                    pagerank=float(pr_arr[i]),
                    out_degree=int(od_arr[i]),
                    robots_ok=bool(robots_arr[i] > 0),
                    reason=reason,
                )
            )
        return candidates

    def explain_policy(self) -> str:
        """Return a human-readable explanation of the crawl policy."""
        return (
            "Crawl Priority Policy\n"
            "=====================\n"
            f"  PageRank weight  : {self.w_pr:.0%}  — authoritative pages first\n"
            f"  Out-degree weight: {self.w_out:.0%}  — hubs surface more new URLs\n"
            f"  Robots bonus     : {self.w_robots:.0%}  — prefer consent-cleared pages\n\n"
            "Why high-PageRank pages yield better AI training data:\n"
            "  PageRank is a proxy for editorial endorsement — many trusted\n"
            "  sites must have found a page worth linking to.  High-PR pages\n"
            "  tend to be factually accurate, well-written, and frequently\n"
            "  updated (e.g. Wikipedia, arXiv, official documentation).\n"
            "  These properties directly correlate with training data quality\n"
            "  for generative models.\n\n"
            "Robots.txt heuristic:\n"
            "  Pages whose domain does NOT block known AI bots (GPTBot,\n"
            "  CCBot, anthropic-ai) in robots.txt are more likely to have\n"
            "  consented to AI training use.  Combining PageRank authority\n"
            "  with robots-crawlability surfaces high-quality, consent-cleared\n"
            "  training pages."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1]; return zeros if all values identical."""
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def _is_crawlable(self, url: str) -> bool:
        """Check robots.txt for the domain. Cached per domain."""
        try:
            parsed = urllib.parse.urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return True  # assume crawlable on parse error

        if domain in self._robots_cache:
            return self._robots_cache[domain]

        robots_url = f"{domain}/robots.txt"
        try:
            import urllib.request
            with urllib.request.urlopen(robots_url, timeout=5) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            result = self._parse_robots(content)
        except Exception:
            result = True  # network error → assume crawlable

        self._robots_cache[domain] = result
        return result

    @staticmethod
    def _parse_robots(content: str) -> bool:
        """
        Return True if none of the known AI bots are globally disallowed.

        Simple parser: looks for User-agent: <bot> followed by Disallow: /
        """
        lines = content.splitlines()
        current_agents: list[str] = []
        for line in lines:
            line = line.strip()
            ua_match = _USERAGENT_RE.match(line)
            if ua_match:
                current_agents = [ua_match.group(1).strip()]
                continue
            dis_match = _ROBOTS_DISALLOW_RE.match(line)
            if dis_match:
                path = dis_match.group(1).strip()
                if path in ("/", "*"):
                    for agent in current_agents:
                        for bot in _AI_BOTS:
                            if bot.lower() in agent.lower() or agent == "*":
                                return False  # globally disallowed
        return True

    @staticmethod
    def _explain(pr_norm: float, od_norm: float, robots_ok: float) -> str:
        parts = []
        if pr_norm > 0.7:
            parts.append("very high PageRank authority")
        elif pr_norm > 0.4:
            parts.append("moderate PageRank authority")
        if od_norm > 0.6:
            parts.append("high out-degree hub")
        if robots_ok:
            parts.append("robots.txt allows AI crawlers")
        return "; ".join(parts) if parts else "low priority"
