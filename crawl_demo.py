#!/usr/bin/env python3
"""
crawl_demo.py — AI Training Data Crawl Prioritisation Demo
===========================================================

Demonstrates the Quality-Weighted Authority (QWA) heuristic and runs
all 10 experiments on a realistic 42-node synthetic web graph that
includes academic, government, news, tech, and spam pages.

Usage:
    python3.11 crawl_demo.py            # full demo + all experiments
    python3.11 crawl_demo.py --k 15     # change top-k cutoff
    python3.11 crawl_demo.py --out results/crawl
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("crawl_demo")
sys.path.insert(0, str(Path(__file__).parent))

from src.pagerank.core import PageRankEngine
from src.crawler.heuristics import build_all, QualityWeightedAuthority
from src.crawler.experiments import ExperimentSuite
from src.crawler.visualiser import CrawlVisualiser
from src.crawler.quality_proxy import score_url
from src.crawler.prioritizer import CrawlPrioritizer


# ── Synthetic web graph ───────────────────────────────────────────────────────
# 42 URLs with realistic link topology.
# Link patterns:
#   • Wikipedia pages link to primary sources (arXiv, PubMed, .gov)
#   • Academic hubs (arXiv, Stanford, MIT) link to each other heavily
#   • Spam sites link only to each other (sink cluster)
#   • News sites link to government and academic sources
#   • Tech AI labs link to arXiv papers and each other

GRAPH: dict[str, list[str]] = {
    # ── Wikipedia (high PR hub) ──────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/PageRank": [
        "https://arxiv.org/abs/1706.03762",
        "https://en.wikipedia.org/wiki/Graph_theory",
        "https://scholar.google.com/",
        "https://stanford.edu/",
    ],
    "https://en.wikipedia.org/wiki/Graph_theory": [
        "https://en.wikipedia.org/wiki/PageRank",
        "https://acm.org/",
        "https://arxiv.org/abs/2005.14165",
    ],
    "https://en.wikipedia.org/wiki/Machine_learning": [
        "https://arxiv.org/abs/1706.03762",
        "https://deepmind.com/research",
        "https://openai.com/research/gpt-3",
        "https://en.wikipedia.org/wiki/PageRank",
        "https://pubmed.ncbi.nlm.nih.gov/",
    ],
    "https://en.wikipedia.org/wiki/Radioactivity": [
        "https://nih.gov/",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "https://britannica.com/science/radioactivity",
    ],

    # ── arXiv (authoritative academic) ──────────────────────────────────────
    "https://arxiv.org/": [
        "https://arxiv.org/abs/1706.03762",
        "https://arxiv.org/abs/2005.14165",
        "https://scholar.google.com/",
    ],
    "https://arxiv.org/abs/1706.03762": [
        "https://arxiv.org/",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://acm.org/",
        "https://stanford.edu/",
    ],
    "https://arxiv.org/abs/2005.14165": [
        "https://arxiv.org/",
        "https://openai.com/research/gpt-3",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ],

    # ── Universities ─────────────────────────────────────────────────────────
    "https://stanford.edu/": [
        "https://arxiv.org/",
        "https://scholar.google.com/",
        "https://openai.com/research/gpt-3",
        "https://mit.edu/",
    ],
    "https://mit.edu/": [
        "https://arxiv.org/",
        "https://scholar.google.com/",
        "https://acm.org/",
        "https://stanford.edu/",
    ],
    "https://berkeley.edu/": [
        "https://arxiv.org/",
        "https://scholar.google.com/",
        "https://mit.edu/",
    ],

    # ── Government / health ──────────────────────────────────────────────────
    "https://nih.gov/": [
        "https://pubmed.ncbi.nlm.nih.gov/",
        "https://cdc.gov/",
        "https://who.int/",
    ],
    "https://pubmed.ncbi.nlm.nih.gov/": [
        "https://nih.gov/",
        "https://arxiv.org/",
        "https://en.wikipedia.org/wiki/Radioactivity",
    ],
    "https://cdc.gov/": [
        "https://nih.gov/",
        "https://who.int/",
    ],
    "https://nasa.gov/": [
        "https://arxiv.org/",
        "https://nih.gov/",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ],
    "https://who.int/": [
        "https://nih.gov/",
        "https://pubmed.ncbi.nlm.nih.gov/",
    ],

    # ── Academic journals ────────────────────────────────────────────────────
    "https://acm.org/": [
        "https://arxiv.org/",
        "https://stanford.edu/",
        "https://scholar.google.com/",
    ],
    "https://scholar.google.com/": [
        "https://arxiv.org/",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "https://acm.org/",
    ],
    "https://nature.com/articles/ai-review": [
        "https://arxiv.org/abs/2005.14165",
        "https://deepmind.com/research",
        "https://nih.gov/",
    ],
    "https://britannica.com/science/radioactivity": [
        "https://en.wikipedia.org/wiki/Radioactivity",
        "https://nih.gov/",
    ],

    # ── AI / Tech ────────────────────────────────────────────────────────────
    "https://openai.com/research/gpt-3": [
        "https://arxiv.org/abs/2005.14165",
        "https://openai.com/",
        "https://stanford.edu/",
    ],
    "https://openai.com/": [
        "https://openai.com/research/gpt-3",
        "https://arxiv.org/",
    ],
    "https://anthropic.com/": [
        "https://arxiv.org/",
        "https://stanford.edu/",
        "https://openai.com/",
    ],
    "https://deepmind.com/research": [
        "https://arxiv.org/abs/1706.03762",
        "https://nature.com/articles/ai-review",
        "https://deepmind.com/",
    ],
    "https://deepmind.com/": [
        "https://deepmind.com/research",
        "https://arxiv.org/",
    ],
    "https://huggingface.co/": [
        "https://arxiv.org/",
        "https://openai.com/",
        "https://anthropic.com/",
    ],

    # ── Quality news ─────────────────────────────────────────────────────────
    "https://reuters.com/technology/ai": [
        "https://openai.com/",
        "https://deepmind.com/",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ],
    "https://bbc.co.uk/news/technology": [
        "https://reuters.com/technology/ai",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://who.int/",
    ],
    "https://theguardian.com/technology": [
        "https://bbc.co.uk/news/technology",
        "https://openai.com/",
    ],

    # ── Medium quality ────────────────────────────────────────────────────────
    "https://medium.com/@researcher/attention-is-all-you-need": [
        "https://arxiv.org/abs/1706.03762",
        "https://medium.com/",
    ],
    "https://medium.com/": [
        "https://medium.com/@researcher/attention-is-all-you-need",
    ],
    "https://github.com/huggingface/transformers": [
        "https://huggingface.co/",
        "https://arxiv.org/abs/1706.03762",
    ],
    "https://stackoverflow.com/questions/pagerank": [
        "https://en.wikipedia.org/wiki/PageRank",
        "https://stackoverflow.com/",
    ],
    "https://stackoverflow.com/": [
        "https://stackoverflow.com/questions/pagerank",
    ],

    # ── AI-blocked news sites (robots.txt blocks GPTBot/CCBot/anthropic-ai) ───
    # All three verified: explicit ClaudeBot/CCBot/anthropic-ai + Disallow: /
    "https://www.nytimes.com/section/technology": [
        "https://openai.com/research/gpt-3",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ],
    "https://www.cnn.com/tech": [
        "https://reuters.com/technology/ai",
        "https://openai.com/",
    ],
    "https://www.washingtonpost.com/technology": [
        "https://www.nytimes.com/section/technology",
        "https://openai.com/",
    ],

    # ── Developer / open-source (robots.txt allows all crawlers) ─────────────
    "https://www.python.org/": [
        "https://github.com/huggingface/transformers",
        "https://stackoverflow.com/questions/pagerank",
        "https://arxiv.org/",
    ],
    "https://docs.python.org/3/": [
        "https://www.python.org/",
        "https://github.com/huggingface/transformers",
    ],
    "https://news.ycombinator.com/": [
        "https://arxiv.org/",
        "https://github.com/huggingface/transformers",
        "https://openai.com/research/gpt-3",
    ],
    "https://substack.com/ai-weekly": [
        "https://openai.com/",
        "https://medium.com/",
    ],
}


def compute_pagerank(graph: dict[str, list[str]], p: float = 0.15) -> dict[str, float]:
    """Compute PageRank on the graph dict."""
    urls = list(graph.keys())
    idx  = {u: i for i, u in enumerate(urls)}
    N    = len(urls)
    out_deg = np.array([len(graph.get(u, [])) for u in urls], dtype=float)
    rows, cols, data = [], [], []
    for src_url, targets in graph.items():
        src = idx[src_url]
        for t in targets:
            if t in idx:
                rows.append(idx[t]); cols.append(src); data.append(1.0)
    if rows:
        A_raw = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=float)
        col_s = np.array(A_raw.sum(axis=0)).flatten()
        col_s[col_s == 0] = 1.0
        A = A_raw.multiply(1.0 / col_s).tocsr()
    else:
        A = sp.eye(N, format="csr") / N
    dangling = out_deg == 0
    result   = PageRankEngine(A, dangling, p=p, tol=1e-10).run()
    return {urls[i]: float(result.scores[i]) for i in range(N)}


def print_top_k_table(ranked, title: str, k: int = 10):
    """Pretty-print top-k crawl candidates."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Rank':>4}  {'Score':>7}  {'URL':<55}  Reason")
    print(f"  {'-'*4}  {'-'*7}  {'-'*55}  {'-'*20}")
    for item in ranked[:k]:
        url_short = item.url[:55] if len(item.url) > 55 else item.url
        print(f"  {item.rank:>4}  {item.score:>7.4f}  {url_short:<55}  {item.reason[:60]}")
    print()


def print_experiment_table(exp_results: dict):
    """Print EXP-10 head-to-head summary table."""
    from src.crawler.heuristics import ALL_HEURISTICS
    short = {
        "H0_Random": "H0 Random",
        "H1_PurePageRank": "H1 Pure PR",
        "H2_HubAuthority": "H2 Hub+PR",
        "H3_PR_Robots": "H3 PR+Robots",
        "H4_QualityWeightedAuth": "H4 QWA ★",
    }
    metrics = ["NDCG@k", "Precision@k", "Recall@k", "Mean Quality", "Unique Domains"]
    print(f"\n{'='*80}")
    print("  EXP-10: Head-to-Head Summary")
    print(f"{'='*80}")
    header = f"  {'Heuristic':<20}" + "".join(f"  {m:>14}" for m in metrics)
    print(header)
    print("  " + "-" * (20 + 16 * len(metrics)))
    for hname, vals in exp_results.items():
        sname = short.get(hname, hname[:20])
        row = f"  {sname:<20}"
        for m in metrics:
            v = vals[m]
            row += f"  {v:>14.4f}" if isinstance(v, float) else f"  {v:>14}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",   type=int, default=10)
    parser.add_argument("--p",   type=float, default=0.15)
    parser.add_argument("--out", type=str, default="results/crawl")
    args, _ = parser.parse_known_args()  # ignore Jupyter kernel args

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = CrawlVisualiser(output_dir=str(out))

    # ── Step 1: compute PageRank ─────────────────────────────────────────────
    logger.info("Computing PageRank on %d-node synthetic web graph ...", len(GRAPH))
    pageranks = compute_pagerank(GRAPH, p=args.p)

    # ── Step 2: show all five heuristics ─────────────────────────────────────
    print("\n" + "="*80)
    print("  SYNTHETIC WEB GRAPH — CRAWL PRIORITISATION")
    print(f"  {len(GRAPH)} URLs  ·  p = {args.p}  ·  Top-k = {args.k}")
    print("="*80)

    all_h = build_all(GRAPH, pageranks)
    for h in all_h:
        ranked = h.rank(k=args.k)
        print_top_k_table(ranked, f"{h.name}  —  {h.description}", k=args.k)

    # ── Step 2b: CrawlPrioritizer top-k ──────────────────────────────────────
    prioritizer = CrawlPrioritizer(GRAPH, pageranks)
    print("="*80)
    print("  CRAWL PRIORITIZER (PageRank + Out-Degree frontier queue)")
    print(f"  Policy: {prioritizer.w_pr:.0%} PageRank  +  {prioritizer.w_out:.0%} Out-Degree")
    print("="*80)
    print(prioritizer.explain_policy())
    candidates = prioritizer.top_k(k=args.k)
    print(f"\n  {'Rank':>4}  {'Priority':>8}  {'PageRank':>9}  {'OutDeg':>6}  URL")
    print(f"  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*50}")
    for i, c in enumerate(candidates, 1):
        url_short = c.url[:50] if len(c.url) > 50 else c.url
        print(f"  {i:>4}  {c.priority:>8.4f}  {c.pagerank:>9.6f}  {c.out_degree:>6}  {url_short}")
    print()

    # ── Step 3: explain QWA heuristic ────────────────────────────────────────
    print("="*80)
    print("  PROPOSED HEURISTIC: Quality-Weighted Authority (QWA)")
    print("="*80)
    print("""
  Signal Composition
  ──────────────────
  Stage 1 — Hard robots gate: fetch robots.txt and discard domains that block AI crawlers
  Stage 2 — Score permitted URLs:
  score(u) = 0.45 × PageRank_norm(u)       ← link-authority endorsement
           + 0.30 × DomainReputation(u)    ← editorial standards
           + 0.15 × TLDQuality(u)          ← .edu/.gov > .org > .com > .biz
           + 0.10 × URLDepthScore(u)       ← shallow pages = better content

  Why High-PageRank Pages Yield Better AI Training Data
  ──────────────────────────────────────────────────────
  1. FACTUAL ACCURACY  — PageRank is weighted editorial endorsement.
     Many independent trusted authors must have linked to a page, implying
     fact-checking by multiple domain experts.

  2. WRITING QUALITY   — Poorly-written pages lose inbound links over time.
     High-PR pages have survived competitive selection pressure for prose clarity.

  3. STABILITY         — High-PR pages rarely disappear or change URLs.
     Training data from stable sources is less likely to contain broken
     references, hallucination triggers, or stale facts.


  Why QWA Outperforms Pure PageRank
  ───────────────────────────────────
  • Filters spam farms that game PageRank with link schemes
  • Prioritises .edu/.gov where factuality is legally/ethically mandated
  • Robots.txt filter ensures training data is consent-cleared
  • URL depth penalty avoids auto-generated pagination noise
""")

    # ── Step 4: quality signal breakdown for QWA top-10 ─────────────────────
    qwa = QualityWeightedAuthority(GRAPH, pageranks)
    top_k = qwa.rank(k=args.k)
    print("="*80)
    print(f"  QWA Top-{args.k} — Full Signal Breakdown")
    print("="*80)
    print(f"  {'#':>3}  {'PR':>7}  {'Rep':>5}  {'TLD':>5}  {'Dep':>5}  URL")
    print(f"  {'-'*3}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*50}")
    for item in top_k:
        q = score_url(item.url)
        pr = pageranks.get(item.url, 0.0)
        print(
            f"  {item.rank:>3}  {pr:>7.5f}  {q.domain_reputation:>5.2f}  "
            f"{q.tld_score:>5.2f}  {q.url_depth_score:>5.2f}  "
            f"{item.url[:55]}"
        )
    print()

    # ── Step 5: run all 10 experiments ───────────────────────────────────────
    logger.info("Running 10 experiments ...")
    t0    = time.perf_counter()
    suite = ExperimentSuite(GRAPH, pageranks, k=args.k)
    all_r = suite.run_all()
    elapsed = time.perf_counter() - t0
    logger.info("All experiments completed in %.2fs", elapsed)

    # ── Step 6: print summaries ───────────────────────────────────────────────
    r1 = all_r["exp1_ablation"]
    print(f"\n{'='*80}")
    print("  EXP-1: Ablation Study")
    print(f"  Full QWA NDCG@{args.k} = {r1.full_ndcg:.4f}")
    for sig, ndcg in r1.ablations.items():
        drop = r1.full_ndcg - ndcg
        bar  = "▇" * int(drop * 80) if drop > 0 else ""
        print(f"    w/o {sig:<20s}  NDCG={ndcg:.4f}  drop={drop:+.4f}  {bar}")

    r3 = all_r["exp3_correlation"]
    print(f"\n{'='*80}")
    print("  EXP-3: Signal Correlation Summary (most complementary pairs)")
    mat = r3.corr_matrix
    names = r3.signal_names
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pairs.append((abs(mat[i,j]), names[i], names[j], mat[i,j]))
    pairs.sort()
    for _, n1, n2, rho in pairs[:3]:
        print(f"    {n1} ↔ {n2}: ρ = {rho:.3f}  (complementary)")

    print_experiment_table(all_r["exp10_head_to_head"])

    # ── Step 7: generate all plots ────────────────────────────────────────────
    logger.info("Generating visualisations ...")
    created = [
        vis.plot_ablation(all_r["exp1_ablation"]),
        vis.plot_p_sensitivity(all_r["exp2_p_sensitivity"]),
        vis.plot_signal_correlation(all_r["exp3_correlation"]),
        vis.plot_domain_diversity(all_r["exp4_diversity"]),
        vis.plot_quality_curve(all_r["exp5_quality_curve"]),
        vis.plot_robots_compliance(all_r["exp6_robots"]),
        vis.plot_url_structural(all_r["exp7_structural"]),
        vis.plot_k_stability(all_r["exp8_k_stability"]),
        vis.plot_topology(all_r["exp9_topology"]),
        vis.plot_head_to_head(all_r["exp10_head_to_head"]),
    ]

    print(f"\n{'='*80}")
    print(f"  Generated {len(created)} figures in {out.resolve()}/")
    for f in created:
        kb = Path(f).stat().st_size / 1024
        print(f"    {Path(f).name:<45s}  {kb:6.1f} KB")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
