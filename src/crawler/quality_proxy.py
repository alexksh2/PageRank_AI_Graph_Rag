"""
quality_proxy.py — URL-level quality signals for AI training data.

Each signal is normalised to [0, 1] so they can be combined linearly.

Signals
-------
1. tld_score          — editorial rigour implied by top-level domain
                         .edu / .gov = 1.0  (peer-reviewed, official)
                         .org         = 0.75 (non-profit, often curated)
                         .com / .net  = 0.45 (commercial, variable quality)
                         .io / .ai    = 0.35 (tech startups, less curation)
                         other / junk = 0.10

2. domain_reputation  — explicit whitelist of known high-quality domains
                         scored 0–1; unknown domains default to 0.30.

3. url_depth_penalty  — shallower URLs tend to be index / landing pages,
                         which are higher quality than auto-generated deep
                         paths (e.g. /category/page/123/item/456).
                         score = exp(−0.35 × depth)

4. path_cleanliness   — penalise URLs with query strings, session IDs,
                         tracking params, and very long paths.

5. content_type_prior — heuristic based on path keywords
                         (.pdf, /paper/, /abstract/, /wiki/ → high)
                         (/ads/, /click/, /redirect/ → low)
"""
from __future__ import annotations

import math
import re
import urllib.parse
from dataclasses import dataclass


# ── Domain reputation table (score 0–1) ─────────────────────────────────────
DOMAIN_REPUTATION: dict[str, float] = {
    # Academic / research
    "arxiv.org":              1.00,
    "pubmed.ncbi.nlm.nih.gov":1.00,
    "scholar.google.com":     0.95,
    "semanticscholar.org":    0.95,
    "researchgate.net":       0.85,
    "jstor.org":              0.90,
    "nature.com":             0.95,
    "sciencedirect.com":      0.90,
    "acm.org":                0.92,
    "ieee.org":               0.92,
    # Universities
    "mit.edu":                1.00,
    "stanford.edu":           1.00,
    "berkeley.edu":           1.00,
    "oxford.ac.uk":           1.00,
    "cambridge.org":          0.95,
    # Reference
    "en.wikipedia.org":       0.90,
    "britannica.com":         0.88,
    # Government / health
    "nih.gov":                1.00,
    "cdc.gov":                0.98,
    "nasa.gov":               0.98,
    "who.int":                0.97,
    # Quality tech / AI
    "openai.com":             0.80,
    "anthropic.com":          0.80,
    "deepmind.com":           0.80,
    "huggingface.co":         0.78,
    # Quality news
    "reuters.com":            0.80,
    "bbc.co.uk":              0.80,
    "nytimes.com":            0.72,
    "theguardian.com":        0.72,
    # Moderate quality
    "medium.com":             0.45,
    "substack.com":           0.42,
    "github.com":             0.70,
    "stackoverflow.com":      0.72,
    # Low quality / spam
    "spammy-seo.com":         0.02,
    "click-farm.net":         0.01,
    "scraped-content.biz":    0.01,
    "ad-network.io":          0.02,
    "keyword-stuffed.org":    0.05,
}

# ── TLD quality table ─────────────────────────────────────────────────────────
TLD_SCORES: dict[str, float] = {
    "edu":   1.00,
    "gov":   1.00,
    "mil":   0.90,
    "ac.uk": 0.95,
    "ac.jp": 0.90,
    "org":   0.75,
    "int":   0.85,
    "com":   0.45,
    "net":   0.40,
    "co.uk": 0.42,
    "io":    0.35,
    "ai":    0.35,
    "info":  0.25,
    "biz":   0.15,
    "xyz":   0.10,
    "club":  0.08,
}

# ── Content-type path patterns ────────────────────────────────────────────────
HIGH_QUALITY_PATH_RE = re.compile(
    r"(/paper|/abstract|/article|/wiki/|/research|"
    r"/publication|/journal|/preprint|/pdf|\.pdf$|"
    r"/docs?/|/documentation|/report|/whitepaper)",
    re.IGNORECASE,
)
LOW_QUALITY_PATH_RE = re.compile(
    r"(/ad|/ads/|/click|/redirect|/track|/affiliate|"
    r"/promo|/coupon|/download-now|/sign-up|"
    r"/subscribe\?|/checkout|/cart)",
    re.IGNORECASE,
)
DIRTY_QUERY_RE = re.compile(
    r"(utm_|sessionid|PHPSESSID|ref=|fbclid|gclid|affiliate)", re.IGNORECASE
)


@dataclass
class URLQualitySignals:
    url: str
    tld_score: float
    domain_reputation: float
    url_depth_score: float
    path_cleanliness: float
    content_type_prior: float

    @property
    def composite(self) -> float:
        """Weighted composite quality score in [0, 1]."""
        return (
            0.30 * self.domain_reputation
            + 0.25 * self.tld_score
            + 0.20 * self.url_depth_score
            + 0.15 * self.path_cleanliness
            + 0.10 * self.content_type_prior
        )

    def as_dict(self) -> dict:
        return {
            "url":               self.url,
            "tld_score":         round(self.tld_score, 4),
            "domain_reputation": round(self.domain_reputation, 4),
            "url_depth_score":   round(self.url_depth_score, 4),
            "path_cleanliness":  round(self.path_cleanliness, 4),
            "content_type_prior":round(self.content_type_prior, 4),
            "composite":         round(self.composite, 4),
        }


def _extract_tld(hostname: str) -> str:
    """Return the effective TLD (handles .ac.uk, .co.uk style)."""
    parts = hostname.split(".")
    if len(parts) >= 3:
        two_part = ".".join(parts[-2:])
        if two_part in TLD_SCORES:
            return two_part
    return parts[-1] if parts else ""


def score_url(url: str) -> URLQualitySignals:
    """Compute all quality signals for a single URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        path     = parsed.path or "/"
        query    = parsed.query or ""
    except Exception:
        hostname, path, query = "", "/", ""

    # 1. TLD score
    tld      = _extract_tld(hostname)
    tld_scr  = TLD_SCORES.get(tld, 0.20)

    # 2. Domain reputation (exact match, then subdomain strip)
    rep = DOMAIN_REPUTATION.get(hostname, None)
    if rep is None:
        # Try stripping one subdomain level
        bare = ".".join(hostname.split(".")[-2:]) if hostname.count(".") >= 2 else hostname
        rep = DOMAIN_REPUTATION.get(bare, 0.30)

    # 3. URL depth  (number of non-empty path segments)
    depth     = max(0, len([s for s in path.split("/") if s]) - 1)
    depth_scr = math.exp(-0.35 * depth)

    # 4. Path cleanliness
    dirty_query   = bool(DIRTY_QUERY_RE.search(query))
    very_long_url = len(url) > 200
    path_clean    = 1.0
    if dirty_query:   path_clean -= 0.35
    if very_long_url: path_clean -= 0.25
    if len(query) > 80: path_clean -= 0.20
    path_clean = max(0.0, path_clean)

    # 5. Content-type prior
    if HIGH_QUALITY_PATH_RE.search(path):
        ctype = 0.90
    elif LOW_QUALITY_PATH_RE.search(path):
        ctype = 0.05
    else:
        ctype = 0.50   # neutral

    return URLQualitySignals(
        url=url,
        tld_score=tld_scr,
        domain_reputation=rep,
        url_depth_score=depth_scr,
        path_cleanliness=path_clean,
        content_type_prior=ctype,
    )


def score_urls(urls: list[str]) -> dict[str, URLQualitySignals]:
    """Score a list of URLs; returns {url: URLQualitySignals}."""
    return {url: score_url(url) for url in urls}
