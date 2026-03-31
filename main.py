#!/usr/bin/env python3
"""
main.py — Industrial PageRank + GraphRAG Driver
================================================

Runs four experiments sequentially:

  1. Analytical PageRank derivation demonstration (small toy graph)
  2. Large-scale power-iteration PageRank on SNAP web-Google dataset
     with closed-form comparison and p-sweep analysis
  3. AI web crawl prioritisation demo (GPTBot-style crawler)
  4. GraphRAG multi-hop query answering via Personalised PageRank
     (Marie Curie -> medical imaging example)

Usage
-----
    # Download dataset first (only needed for experiment 2):
    #   wget https://hunglvosu.github.io/posts/2020/07/PA3/web-Google_10k.txt -P data/
    # or use the full dataset:
    #   wget https://snap.stanford.edu/data/web-Google.txt.gz -P data/
    #   gunzip data/web-Google.txt.gz

    python main.py                     # run all experiments
    python main.py --exp 2             # run only experiment 2
    python main.py --exp 2 --p 0.1    # custom teleportation prob
    python main.py --exp 2 --data data/web-Google.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Add src/ to path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from src.pagerank import PageRankEngine, WebGraphLoader, AnalyticalPageRank
from src.crawler import CrawlPrioritizer
from src.graphrag import KnowledgeGraph, PageRankRetrieval, GraphRAGQueryEngine


# ===========================================================================
# Experiment 1: Analytical PageRank on a small toy graph
# ===========================================================================

def exp1_analytical_demo():
    """
    Demonstrate the closed-form PageRank on a 6-node toy graph and compare
    with power-iteration.  Also sweeps over p to show how distribution changes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Analytical PageRank (Toy Graph)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Print the mathematical derivation
    # -----------------------------------------------------------------------
    print("""
MATHEMATICAL DERIVATION
-----------------------
The Google matrix for N web pages with teleportation probability p:

    G = (1 - p) * A_hat  +  (p / N) * e * e^T

where A_hat is column-stochastic (dangling nodes redistributed uniformly),
e is the all-ones vector.

PageRank r is the stationary distribution:

    r = G * r  ,  ||r||_1 = 1

Substituting G and using e^T * r = 1:

    r = (1-p) * A_hat * r  +  (p/N) * e

Rearranging:

    [I - (1-p) * A_hat] * r = (p/N) * e

CLOSED FORM:

    r = (p/N) * [I - (1-p) * A_hat]^{-1} * e          ... (*)

Effect of p on PageRank distribution:
  p -> 0   : r concentrates on highest-authority nodes (power-law).
             Structure dominates; may not converge for reducible graphs.
  p = 0.15 : Google's original value. Balances authority and exploration.
  p -> 1   : r -> uniform (1/N). Graph structure is completely ignored.
""")

    # -----------------------------------------------------------------------
    # Build a 6-node toy web graph
    # Nodes: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F
    # Edges: A->B, A->C, B->D, C->D, C->E, D->F, E->F, F->A
    # -----------------------------------------------------------------------
    import scipy.sparse as sp

    N = 6
    labels = ["A", "B", "C", "D", "E", "F"]
    edges = [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5),(5,0)]

    # Build column-stochastic matrix
    out_degree = np.zeros(N, dtype=np.float64)
    for src, _ in edges:
        out_degree[src] += 1

    rows, cols, data = [], [], []
    for src, dst in edges:
        rows.append(dst)
        cols.append(src)
        data.append(1.0 / out_degree[src])

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    dangling = out_degree == 0

    print("Toy graph (6 nodes):")
    print("  Nodes:", labels)
    print("  Edges:", [(labels[u], labels[v]) for u, v in edges])
    print()

    # Power-iteration result for several p values
    p_values = [0.05, 0.10, 0.15, 0.25, 0.50, 0.85, 1.00]
    print(f"{'p':>6}  " + "  ".join(f"{l:>8}" for l in labels))
    print("-" * (6 + 12 * N))

    for p in p_values:
        engine = PageRankEngine(A, dangling, p=p, tol=1e-12)
        result = engine.run()
        row = f"{p:>6.2f}  " + "  ".join(f"{result.scores[i]:>8.5f}" for i in range(N))
        print(row)

    print()

    # Closed-form vs power-iteration comparison at p=0.15
    p = 0.15
    analytical_pr = AnalyticalPageRank(A, dangling, p=p)
    r_analytical = analytical_pr.compute()

    engine = PageRankEngine(A, dangling, p=p, tol=1e-12)
    r_iterative = engine.run().scores

    print(f"\nClosed-form vs Power-Iteration comparison (p={p}):")
    print(f"{'Node':>6}  {'Analytical':>12}  {'Iterative':>12}  {'|Diff|':>10}")
    print("-" * 50)
    for i in range(N):
        print(f"{labels[i]:>6}  {r_analytical[i]:>12.8f}  {r_iterative[i]:>12.8f}  {abs(r_analytical[i]-r_iterative[i]):>10.2e}")

    l1_err = float(np.abs(r_analytical - r_iterative).sum())
    print(f"\nL1 error (analytical vs iterative): {l1_err:.2e}")
    print("\n[INTERPRETATION]")
    print("  - As p increases, scores become more uniform (converge to 1/N=0.1667)")
    print("  - At low p, node F gets highest PR (all paths flow through it)")
    print("  - Closed-form and power-iteration agree to machine precision")


# ===========================================================================
# Experiment 2: Large-scale PageRank on SNAP web-Google dataset
# ===========================================================================

def exp2_large_scale(data_path: str, p: float = 0.15):
    """Load web-Google dataset, run PageRank, compare with analytical."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Large-Scale PageRank (SNAP web-Google)")
    print("=" * 70)

    path = Path(data_path)
    if not path.exists():
        print(f"\n[WARNING] Dataset not found at '{data_path}'.")
        print("  Please download it:")
        print("    wget https://hunglvosu.github.io/posts/2020/07/PA3/web-Google_10k.txt -O data/web-Google-10k.txt")
        print("  Skipping experiment 2.")
        return

    # -----------------------------------------------------------------------
    # Load graph
    # -----------------------------------------------------------------------
    loader = WebGraphLoader(path)
    loader.load()
    print("\nGraph summary:")
    print(loader.summary())

    # -----------------------------------------------------------------------
    # Power iteration at default p
    # -----------------------------------------------------------------------
    print(f"\nRunning power-iteration PageRank (p={p}) ...")
    engine = PageRankEngine(loader.A, loader.dangling_mask, p=p, tol=1e-8)
    result = engine.run()

    print(f"  Converged : {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time      : {result.elapsed_sec:.3f}s")
    print(f"  Sum(r)    : {result.scores.sum():.8f}  (should be 1.0)")

    print(f"\nTop-20 PageRank nodes (p={p}):")
    print(f"  {'Rank':>5}  {'Node ID':>10}  {'PR Score':>14}")
    print("  " + "-" * 36)
    for rank, (node_id, score) in enumerate(result.top_k(20, loader.idx_to_id), 1):
        print(f"  {rank:>5}  {node_id:>10}  {score:>14.8f}")

    # -----------------------------------------------------------------------
    # Analytical comparison (small graphs only)
    # -----------------------------------------------------------------------
    analytical_pr = AnalyticalPageRank(loader.A, loader.dangling_mask, p=p)
    cmp = analytical_pr.compare_with_iterative(result.scores, top_k=20)
    if "error" not in cmp:
        print(f"\nAnalytical vs Iterative Comparison:")
        print(f"  L1 error     : {cmp['l1_error']:.4e}")
        print(f"  L∞ error     : {cmp['l_inf_error']:.4e}")
        print(f"  Top-{cmp['top_k']} rank overlap: {cmp['top_k_rank_overlap']}/{cmp['top_k']}")
    else:
        print(f"\nAnalytical comparison: {cmp['error']}")

    # -----------------------------------------------------------------------
    # p-sweep: show how ranking changes
    # -----------------------------------------------------------------------
    p_vals = [0.05, 0.15, 0.30, 0.50, 0.85]
    print("\np-sweep: entropy of PageRank distribution (higher = more uniform)")
    print(f"  {'p':>6}  {'Entropy':>10}  {'Top-1 score':>12}  {'Top-1 node':>12}")
    print("  " + "-" * 50)
    for pv in p_vals:
        eng = PageRankEngine(loader.A, loader.dangling_mask, p=pv, tol=1e-7)
        res = eng.run()
        r = res.scores
        # Shannon entropy
        r_safe = r[r > 0]
        entropy = float(-np.sum(r_safe * np.log(r_safe)))
        top_idx = int(np.argmax(r))
        top_id = loader.idx_to_id[top_idx]
        top_score = float(r[top_idx])
        print(f"  {pv:>6.2f}  {entropy:>10.4f}  {top_score:>12.8f}  {top_id:>12}")


# ===========================================================================
# Experiment 3: AI Web Crawl Prioritisation
# ===========================================================================

def exp3_crawl_prioritizer():
    """Demonstrate GPTBot-style crawl prioritisation using PageRank."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: AI Web Crawl Prioritisation (GPTBot-style)")
    print("=" * 70)

    # Small web graph: URL -> [outlinks]
    graph = {
        "https://arxiv.org/":           ["https://arxiv.org/abs/1706.03762", "https://arxiv.org/abs/2005.14165"],
        "https://arxiv.org/abs/1706.03762": ["https://arxiv.org/", "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"],
        "https://arxiv.org/abs/2005.14165": ["https://arxiv.org/", "https://openai.com/research/gpt-3"],
        "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)": [
            "https://arxiv.org/abs/1706.03762",
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
        ],
        "https://en.wikipedia.org/wiki/Deep_learning": [
            "https://en.wikipedia.org/wiki/Natural_language_processing",
            "https://arxiv.org/",
        ],
        "https://en.wikipedia.org/wiki/Natural_language_processing": [
            "https://en.wikipedia.org/wiki/Deep_learning",
        ],
        "https://openai.com/research/gpt-3": [
            "https://arxiv.org/abs/2005.14165",
            "https://openai.com/",
        ],
        "https://openai.com/": [
            "https://openai.com/research/gpt-3",
        ],
        "https://spam-site-example.com/cheap-seo": [],   # low PR, dangling
        "https://spam-site-example.com/click-here":  ["https://spam-site-example.com/cheap-seo"],
    }

    # Map URLs to integer indices and build a loader-compatible structure
    import scipy.sparse as sp

    urls = list(graph.keys())
    url_to_idx = {u: i for i, u in enumerate(urls)}
    N = len(urls)

    out_degree = np.zeros(N, dtype=np.float64)
    for src_url, targets in graph.items():
        out_degree[url_to_idx[src_url]] += len(targets)

    rows, cols, data = [], [], []
    for src_url, targets in graph.items():
        src = url_to_idx[src_url]
        for tgt_url in targets:
            if tgt_url in url_to_idx:
                dst = url_to_idx[tgt_url]
                rows.append(dst)
                cols.append(src)
                data.append(1.0 / len(targets))

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    dangling = out_degree == 0

    engine = PageRankEngine(A, dangling, p=0.15, tol=1e-10)
    result = engine.run()

    pageranks = {url: float(result.scores[url_to_idx[url]]) for url in urls}

    prioritizer = CrawlPrioritizer(graph, pageranks, check_robots=False)
    top_candidates = prioritizer.top_k(k=7)

    print("\nCrawl Priority Policy:")
    print(prioritizer.explain_policy())

    print("\nTop-7 URLs to crawl (ranked by composite priority):")
    print(f"  {'Rank':>4}  {'Priority':>8}  {'PageRank':>10}  {'OutDeg':>6}  URL")
    print("  " + "-" * 90)
    for i, c in enumerate(top_candidates, 1):
        print(f"  {i:>4}  {c.priority:>8.4f}  {c.pagerank:>10.6f}  {c.out_degree:>6}  {c.url}")
        print(f"        Reason: {c.reason}")


# ===========================================================================
# Experiment 4: GraphRAG — Marie Curie multi-hop query
# ===========================================================================

def exp4_graphrag_query():
    """
    Build a knowledge graph about Marie Curie and medical imaging,
    then answer a multi-hop query using Personalised PageRank.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: GraphRAG Multi-Hop Query via Personalised PageRank")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Build knowledge graph
    # -----------------------------------------------------------------------
    kg = KnowledgeGraph()

    # Core Curie facts
    kg.add_triples([
        ("Marie Curie", "is_a", "physicist"),
        ("Marie Curie", "is_a", "chemist"),
        ("Marie Curie", "discovered", "polonium"),
        ("Marie Curie", "discovered", "radium"),
        ("Marie Curie", "coined_term", "radioactivity"),
        ("Marie Curie", "developed", "mobile_xray_unit"),
        ("Marie Curie", "awarded", "Nobel_Prize_Physics_1903"),
        ("Marie Curie", "awarded", "Nobel_Prize_Chemistry_1911"),
        ("Pierre Curie", "co_discovered", "polonium"),
        ("Pierre Curie", "co_discovered", "radium"),
        ("Pierre Curie", "married", "Marie Curie"),

        # Radium properties and uses
        ("radium", "is_a", "radioactive_element"),
        ("radium", "emits", "alpha_radiation"),
        ("radium", "emits", "gamma_radiation"),
        ("radium", "property_of", "radioactivity"),
        ("radioactivity", "enables", "radiation_therapy"),
        ("radium", "used_in", "radium_therapy"),
        ("radium_therapy", "is_a_form_of", "radiation_therapy"),
        ("radiation_therapy", "treats", "cancer"),
        ("radiation_therapy", "led_to", "radiotherapy"),

        # X-ray and medical imaging chain
        ("mobile_xray_unit", "is_a", "X-ray_device"),
        ("X-ray_device", "enables", "X-ray_imaging"),
        ("X-ray_imaging", "is_a_form_of", "medical_imaging"),
        ("X-ray_imaging", "led_to", "CT_scan"),
        ("CT_scan", "is_a_form_of", "medical_imaging"),
        ("radium", "inspired", "nuclear_medicine"),
        ("nuclear_medicine", "uses", "radioactive_tracers"),
        ("radioactive_tracers", "enable", "PET_scan"),
        ("PET_scan", "is_a_form_of", "medical_imaging"),

        # Isotope connection
        ("radium", "led_to_study_of", "isotopes"),
        ("isotopes", "used_in", "radioactive_tracers"),
        ("isotopes", "used_in", "MRI_contrast_agents"),
        ("MRI_contrast_agents", "enhance", "MRI_scan"),
        ("MRI_scan", "is_a_form_of", "medical_imaging"),

        # Modern medical imaging
        ("medical_imaging", "includes", "CT_scan"),
        ("medical_imaging", "includes", "MRI_scan"),
        ("medical_imaging", "includes", "PET_scan"),
        ("medical_imaging", "includes", "X-ray_imaging"),
        ("medical_imaging", "advances", "cancer_diagnosis"),
        ("medical_imaging", "advances", "neurology"),
        ("radiotherapy", "is_a_form_of", "medical_imaging"),
        ("radiotherapy", "uses", "gamma_radiation"),
        ("gamma_radiation", "emitted_by", "radium"),

        # Higher-order connections
        ("Nobel_Prize_Physics_1903", "recognises", "radioactivity"),
        ("radioactivity", "foundation_of", "nuclear_physics"),
        ("nuclear_physics", "enables", "nuclear_medicine"),
        ("nuclear_medicine", "is_a_form_of", "medical_imaging"),
    ])

    print(f"\nKnowledge Graph built: {kg.n_entities} entities, {kg.n_relations} relations")
    print("\nSample entity neighbours:")
    for entity in ["Marie Curie", "radium", "radioactivity", "medical_imaging"]:
        nbrs = kg.neighbors(entity)
        print(f"  {entity!r:30s} -> {nbrs[:4]}")

    # -----------------------------------------------------------------------
    # Multi-hop query
    # -----------------------------------------------------------------------
    engine = GraphRAGQueryEngine(kg, p=0.15, k=15)

    query = "What discoveries by Marie Curie led to later advances in medical imaging?"
    seeds = ["Marie Curie", "medical_imaging"]

    print(f"\nQuery : {query!r}")
    print(f"Seeds : {seeds}")

    response = engine.query(query, seed_entities=seeds, k=12, max_chain_hops=8)
    print()
    print(response.pretty_print())

    # -----------------------------------------------------------------------
    # p-sensitivity analysis for GraphRAG
    # -----------------------------------------------------------------------
    print("\nSensitivity to teleportation parameter p:")
    print(f"  {'p':>6}  Top-5 retrieved entities")
    print("  " + "-" * 70)
    for pv in [0.05, 0.15, 0.30, 0.60]:
        engine_pv = GraphRAGQueryEngine(kg, p=pv, k=5)
        resp_pv = engine_pv.query(query, seed_entities=seeds, k=5)
        top5 = ", ".join(r.entity_name for r in resp_pv.top_k_entities[:5])
        print(f"  {pv:>6.2f}  {top5}")

    print("\n[INTERPRETATION]")
    print("  Low p  : seeds dominate; only direct neighbours rank highly")
    print("  Mid p  : balanced; multi-hop reasoning chains emerge")
    print("  High p : uniform spread; distant entities rank near seeds")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PageRank + GraphRAG Industrial Implementation"
    )
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run only experiment N (default: run all)"
    )
    parser.add_argument(
        "--data", type=str, default="data/web-Google-10k.txt",
        help="Path to SNAP web-Google edge-list file"
    )
    parser.add_argument(
        "--p", type=float, default=0.15,
        help="Teleportation probability (default: 0.15)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exps = [args.exp] if args.exp else [1, 2, 3, 4]

    t_total = time.perf_counter()

    if 1 in exps:
        exp1_analytical_demo()
    if 2 in exps:
        exp2_large_scale(args.data, p=args.p)
    if 3 in exps:
        exp3_crawl_prioritizer()
    if 4 in exps:
        exp4_graphrag_query()

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
