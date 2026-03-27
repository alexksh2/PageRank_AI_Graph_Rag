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

Two-stage strategy:

  Stage 1 — Hard gate (robots.txt):
    Discard any URL whose domain blocks known AI crawlers (GPTBot, CCBot,
    anthropic-ai) in robots.txt.  A blocked page is NEVER crawled regardless
    of how high its PageRank is.  This ensures all candidates are
    consent-cleared before any quality ranking begins.

  Stage 2 — Rank permitted pages by composite PRIORITY SCORE:

    priority(u) = w_pr  * norm(PR(u))           (PageRank authority)
               + w_out * norm(out_degree(u))     (hub potential: many links to follow)

All components are normalised to [0, 1] before weighting so the weights
are interpretable percentages.

Heuristic for "high quality AND permits crawling":
  robots.txt is treated as a hard filter, not a soft bonus.
  Only pages that explicitly permit AI crawlers are ranked.
  Among those, PageRank authority is the primary quality signal.
============================================================
"""

from __future__ import annotations

import heapq
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np

logger = logging.getLogger(__name__)

# Known AI crawler user-agent tokens to check in robots.txt
_AI_BOTS = ["GPTBot", "CCBot", "OAI-SearchBot", "anthropic-ai", "ClaudeBot"]
_ROBOTS_DISALLOW_RE = re.compile(r"^Disallow\s*:\s*(.+)$", re.IGNORECASE)
_USERAGENT_RE = re.compile(r"^User-agent\s*:\s*(.+)$", re.IGNORECASE)

# Module-level cache shared across all instances and heuristics
_ROBOTS_CACHE: dict[str, bool] = {}


def fetch_robots_permitted(url: str) -> bool:
    """
    Fetch and parse robots.txt for the domain of url.
    Returns True if AI crawlers are permitted, False if blocked.
    Results are cached per domain to avoid repeated network calls.
    On network error, defaults to True (assume crawlable).
    """
    try:
        parsed = urllib.parse.urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return True

    if domain in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[domain]

    robots_url = f"{domain}/robots.txt"
    try:
        with urllib.request.urlopen(robots_url, timeout=5) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        result = _parse_robots_content(content)
    except Exception:
        result = True  # network error → assume crawlable

    _ROBOTS_CACHE[domain] = result
    logger.debug("robots.txt %s → %s", domain, "allowed" if result else "blocked")
    return result


def _parse_robots_content(content: str) -> bool:
    """
    Return True if none of the known AI bots are globally disallowed.

    Handles both single and grouped User-agent blocks, e.g.:
        User-agent: GPTBot       User-agent: GPTBot
        Disallow: /              User-agent: ClaudeBot
                                 Disallow: /

    Consecutive User-agent lines are accumulated into one group; a blank
    line resets the group (standard robots.txt group separator).
    """
    lines = content.splitlines()
    current_agents: list[str] = []
    last_was_ua = False
    for line in lines:
        line = line.strip()
        if not line:  # blank line = group separator
            current_agents = []
            last_was_ua = False
            continue
        ua_match = _USERAGENT_RE.match(line)
        if ua_match:
            if not last_was_ua:
                current_agents = []  # start of a new group
            current_agents.append(ua_match.group(1).strip())
            last_was_ua = True
            continue
        last_was_ua = False
        dis_match = _ROBOTS_DISALLOW_RE.match(line)
        if dis_match:
            path = dis_match.group(1).strip()
            if path in ("/", "*"):
                for agent in current_agents:
                    for bot in _AI_BOTS:
                        if bot.lower() in agent.lower() or agent == "*":
                            return False
    return True


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
        w_pr: float = 0.75,
        w_out: float = 0.25,
        check_robots: bool = False,
    ) -> None:
        self.graph = graph
        self.pageranks = pageranks
        self.w_pr = w_pr
        self.w_out = w_out
        self.check_robots = check_robots
        self._robots_cache: dict[str, bool] = {}  # domain -> True if crawlable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def top_k(self, k: int = 10) -> list[CrawlCandidate]:
        """Return the top-k URLs to crawl next, ranked by composite priority.

        Two-stage strategy:
          Stage 1 — Hard gate: discard URLs blocked by robots.txt for AI crawlers.
          Stage 2 — Rank:      among permitted pages, score by PageRank + out-degree.
        """
        urls = list(self.graph.keys())
        if not urls:
            return []

        # ── Stage 1: hard robots.txt gate ────────────────────────────────────
        if self.check_robots:
            permitted = [u for u in urls if self._is_crawlable(u)]
        else:
            permitted = urls  # assume all crawlable if not checking

        if not permitted:
            logger.warning("All URLs blocked by robots.txt — no candidates to return.")
            return []

        # ── Stage 2: rank permitted URLs by PageRank + out-degree ────────────
        pr_arr = np.array([self.pageranks.get(u, 0.0) for u in permitted], dtype=float)
        od_arr = np.array([len(self.graph.get(u, [])) for u in permitted], dtype=float)

        pr_norm = self._normalise(pr_arr)
        od_norm = self._normalise(od_arr)

        priority_arr = self.w_pr * pr_norm + self.w_out * od_norm

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
            url = permitted[i]
            reason = self._explain(pr_norm[i], od_norm[i])
            candidates.append(
                CrawlCandidate(
                    priority=float(priority_arr[i]),
                    url=url,
                    pagerank=float(pr_arr[i]),
                    out_degree=int(od_arr[i]),
                    robots_ok=True,  # all candidates passed the gate
                    reason=reason,
                )
            )
        return candidates

    def explain_policy(self) -> str:
        """Return a human-readable explanation of the crawl policy."""
        return (
            "Crawl Priority Policy\n"
            "=====================\n"
            "  Stage 1 — Hard gate : robots.txt must permit AI crawlers\n"
            "              (GPTBot, CCBot, anthropic-ai).  Blocked pages are\n"
            "              excluded entirely regardless of PageRank.\n\n"
            f"  Stage 2 — Ranking  (among permitted pages only):\n"
            f"    PageRank weight  : {self.w_pr:.0%}  — authoritative pages first\n"
            f"    Out-degree weight: {self.w_out:.0%}  — hubs surface more new URLs\n\n"
            "Why high-PageRank pages yield better AI training data:\n"
            "  PageRank is a proxy for editorial endorsement — many trusted\n"
            "  sites must have found a page worth linking to.  High-PR pages\n"
            "  tend to be factually accurate, well-written, and frequently\n"
            "  updated (e.g. Wikipedia, arXiv, official documentation).\n"
            "  These properties directly correlate with training data quality\n"
            "  for generative models.\n\n"
            "Robots.txt heuristic:\n"
            "  robots.txt is treated as a hard filter, not a soft bonus.\n"
            "  Only pages that explicitly permit AI crawlers are considered.\n"
            "  This ensures all returned candidates are consent-cleared,\n"
            "  then ranked purely by content quality signals."
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

        if domain in _ROBOTS_CACHE:
            return _ROBOTS_CACHE[domain]
        return fetch_robots_permitted(url)

    @staticmethod
    def _explain(pr_norm: float, od_norm: float) -> str:
        parts = ["robots.txt permits AI crawlers"]  # all candidates passed the gate
        if pr_norm > 0.7:
            parts.append("very high PageRank authority")
        elif pr_norm > 0.4:
            parts.append("moderate PageRank authority")
        if od_norm > 0.6:
            parts.append("high out-degree hub")
        return "; ".join(parts)
