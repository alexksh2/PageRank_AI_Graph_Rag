#!/usr/bin/env python3
"""
evaluate.py — Full metrics evaluation and report generation.

Runs every metric category against the real web-Google graph and
the GraphRAG Marie Curie demo, then publishes all figures and tables
to results/.

Usage:
    python3.11 evaluate.py                         # uses data/web-Google.txt
    python3.11 evaluate.py --data data/web-Google.txt
    python3.11 evaluate.py --small                 # toy graph only (no large dataset needed)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")

sys.path.insert(0, str(Path(__file__).parent))

from src.pagerank import PageRankEngine, WebGraphLoader, AnalyticalPageRank
from src.graphrag import KnowledgeGraph, GraphRAGQueryEngine
from src.graphrag.pagerank_retrieval import PageRankRetrieval

from src.metrics import (
    CorrectnessMetrics,
    ConvergenceMetrics,
    RankingMetrics,
    ScalabilityMetrics,
    GraphStructuralMetrics,
    GraphRAGMetrics,
    MetricsReporter,
)


# ===========================================================================
# Section A: Build toy or real graph, run PageRank, collect metrics
# ===========================================================================

def evaluate_pagerank(data_path: str, reporter: MetricsReporter, use_small: bool = False):
    import scipy.sparse as sp

    if use_small:
        logger.info("Using 6-node toy graph (--small mode)")
        N = 6
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (5, 0)]
        out_degree = np.zeros(N, dtype=np.float64)
        for src, _ in edges:
            out_degree[src] += 1
        rows, cols, data = [], [], []
        for src, dst in edges:
            rows.append(dst); cols.append(src); data.append(1.0 / out_degree[src])
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        dangling = out_degree == 0
        n_edges = len(edges)
    else:
        logger.info("Loading web-Google graph from %s", data_path)
        loader = WebGraphLoader(data_path)
        loader.load()
        print(loader.summary())
        A = loader.A
        dangling = loader.dangling_mask
        N = loader.n_nodes
        n_edges = loader.n_edges

    p_values = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.70, 0.85]
    convergence_sweep: dict = {}
    p_sweep_data: dict = {}

    # -----------------------------------------------------------------------
    # Run PageRank for each p value
    # -----------------------------------------------------------------------
    results_by_p: dict[float, np.ndarray] = {}
    for pv in p_values:
        label = f"p={pv:.2f}"
        logger.info("Running PageRank p=%.2f ...", pv)

        scalability = ScalabilityMetrics(N, n_edges)
        engine = PageRankEngine(A, dangling, p=pv, tol=1e-8, max_iter=300)
        pr_result, scale_result = scalability.measure(engine.run)
        results_by_p[pv] = pr_result.scores

        # Convergence metrics
        conv = ConvergenceMetrics(
            pr_result.residuals, pv, pr_result.elapsed_sec, pr_result.converged
        ).compute()
        convergence_sweep[label] = conv

        # p-sweep structural data
        r = pr_result.scores
        safe = r[r > 0]
        entropy = float(-np.sum(safe * np.log(safe)))
        p_sweep_data[pv] = {
            "entropy":   entropy,
            "gini":      GraphStructuralMetrics._gini(r),
            "top1_score": float(r.max()),
        }

        # Save scalability for default p
        if abs(pv - 0.15) < 1e-9:
            reporter.add_scalability(scale_result)

    reporter.add_convergence_sweep(convergence_sweep)
    reporter.add_p_sweep(p_sweep_data)

    # -----------------------------------------------------------------------
    # Focus analysis at p=0.15
    # -----------------------------------------------------------------------
    r_iterative = results_by_p[0.15]
    p_ref = 0.15

    # Analytical comparison (small graphs only)
    analytical = AnalyticalPageRank(A, dangling, p=p_ref)
    r_analytical = analytical.compute()
    if r_analytical is not None:
        logger.info("Computing correctness metrics (analytical vs iterative) ...")
        correctness = CorrectnessMetrics(r_iterative, r_analytical, p_ref, N).compute()
        reporter.add_correctness(correctness, r_iterative, r_analytical)
        ranking = RankingMetrics(r_iterative, r_analytical, k=20).compute()
        reporter.add_ranking(ranking)
        logger.info("L1 error: %.3e | Top-20 overlap: %d/20",
                    correctness.l1_error, ranking.top_k_overlap * 20)
    else:
        logger.info("Graph too large for analytical comparison; skipping correctness/ranking.")
        # Use two different p values as proxy for ranking comparison
        r_a = results_by_p[0.15]
        r_b = results_by_p[0.10]
        ranking = RankingMetrics(r_a, r_b, k=20).compute()
        reporter.add_ranking(ranking)

    # Structural metrics
    logger.info("Computing structural metrics ...")
    struct = GraphStructuralMetrics(r_iterative, A).compute()
    reporter.add_structural(struct, r_iterative, A)
    logger.info(
        "Entropy=%.3f | Gini=%.3f | In-deg Spearman ρ=%.3f | Power-law α=%.3f",
        struct.entropy, struct.gini,
        struct.in_degree_spearman_rho, struct.powerlaw_alpha,
    )


# ===========================================================================
# Section B: GraphRAG evaluation on the Marie Curie KG
# ===========================================================================

def evaluate_graphrag(reporter: MetricsReporter):
    logger.info("Building Marie Curie knowledge graph ...")

    kg = KnowledgeGraph()
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
        ("radium", "is_a", "radioactive_element"),
        ("radium", "emits", "alpha_radiation"),
        ("radium", "emits", "gamma_radiation"),
        ("radium", "property_of", "radioactivity"),
        ("radioactivity", "enables", "radiation_therapy"),
        ("radium", "used_in", "radium_therapy"),
        ("radium_therapy", "is_a_form_of", "radiation_therapy"),
        ("radiation_therapy", "treats", "cancer"),
        ("radiation_therapy", "led_to", "radiotherapy"),
        ("mobile_xray_unit", "is_a", "X-ray_device"),
        ("X-ray_device", "enables", "X-ray_imaging"),
        ("X-ray_imaging", "is_a_form_of", "medical_imaging"),
        ("X-ray_imaging", "led_to", "CT_scan"),
        ("CT_scan", "is_a_form_of", "medical_imaging"),
        ("radium", "inspired", "nuclear_medicine"),
        ("nuclear_medicine", "uses", "radioactive_tracers"),
        ("radioactive_tracers", "enable", "PET_scan"),
        ("PET_scan", "is_a_form_of", "medical_imaging"),
        ("radium", "led_to_study_of", "isotopes"),
        ("isotopes", "used_in", "radioactive_tracers"),
        ("isotopes", "used_in", "MRI_contrast_agents"),
        ("MRI_contrast_agents", "enhance", "MRI_scan"),
        ("MRI_scan", "is_a_form_of", "medical_imaging"),
        ("medical_imaging", "includes", "CT_scan"),
        ("medical_imaging", "includes", "MRI_scan"),
        ("medical_imaging", "includes", "PET_scan"),
        ("medical_imaging", "includes", "X-ray_imaging"),
        ("medical_imaging", "advances", "cancer_diagnosis"),
        ("medical_imaging", "advances", "neurology"),
        ("radiotherapy", "is_a_form_of", "medical_imaging"),
        ("radiotherapy", "uses", "gamma_radiation"),
        ("gamma_radiation", "emitted_by", "radium"),
        ("Nobel_Prize_Physics_1903", "recognises", "radioactivity"),
        ("radioactivity", "foundation_of", "nuclear_physics"),
        ("nuclear_physics", "enables", "nuclear_medicine"),
        ("nuclear_medicine", "is_a_form_of", "medical_imaging"),
    ])

    seeds = ["Marie Curie", "medical_imaging"]
    retrieval = PageRankRetrieval(kg, p=0.15)
    results = retrieval.retrieve(seeds, k=30)

    # Ground-truth relevant entities for the query
    relevant = {
        "radium", "radioactivity", "radiation_therapy", "radiotherapy",
        "X-ray_imaging", "CT_scan", "PET_scan", "MRI_scan",
        "nuclear_medicine", "medical_imaging", "isotopes",
        "mobile_xray_unit", "radioactive_tracers",
    }

    # Expected reasoning chains
    expected_chains = [
        ["Marie Curie", "discovered", "radium"],
        ["Marie Curie", "coined_term", "radioactivity"],
        ["Marie Curie", "developed", "mobile_xray_unit"],
    ]

    # Build chains
    engine = GraphRAGQueryEngine(kg, p=0.15, k=15)
    resp = engine.query(
        "What discoveries by Marie Curie led to later advances in medical imaging?",
        seed_entities=seeds, k=15,
    )

    rag_metrics = GraphRAGMetrics(
        results=results,
        relevant_entities=relevant,
        seeds=seeds,
        expected_chains=expected_chains,
        found_chains=resp.reasoning_chains,
    ).compute()

    reporter.add_graphrag(rag_metrics, retrieval_results=results)

    logger.info(
        "GraphRAG — MRR=%.3f | Hit@5=%.2f | Hit@10=%.2f | PPR fidelity ρ=%.3f",
        rag_metrics.mrr, rag_metrics.hit_at_5, rag_metrics.hit_at_10,
        rag_metrics.personalisation_fidelity,
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="PageRank Metrics Evaluation")
    parser.add_argument("--data", default="data/web-Google.txt")
    parser.add_argument("--small", action="store_true",
                        help="Use toy 6-node graph (no dataset needed)")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    reporter = MetricsReporter(output_dir=args.out)

    use_small = args.small or not Path(args.data).exists()
    if not Path(args.data).exists() and not args.small:
        logger.warning("Dataset not found at '%s'; falling back to toy graph.", args.data)
        use_small = True

    t0 = time.perf_counter()

    logger.info("=== Phase 1: PageRank Evaluation ===")
    evaluate_pagerank(args.data, reporter, use_small=use_small)

    logger.info("=== Phase 2: GraphRAG Evaluation ===")
    evaluate_graphrag(reporter)

    logger.info("=== Phase 3: Publishing Results ===")
    created = reporter.publish()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 65}")
    print(f"Evaluation complete in {elapsed:.1f}s")
    print(f"{'=' * 65}")
    print(f"Output directory: {Path(args.out).resolve()}")
    print(f"\nGenerated {len(created)} file(s):")
    for f in sorted(created):
        size_kb = Path(f).stat().st_size / 1024
        print(f"  {Path(f).name:<45s}  {size_kb:6.1f} KB")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
