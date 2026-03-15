"""
GraphRAGQueryEngine: End-to-end multi-hop query answering via PageRank.

Pipeline
--------
1. Parse query -> extract seed entities (via simple keyword matching or
   injected NER; extendable to an LLM-based extractor).
2. Run Personalised PageRank from seeds over the knowledge graph.
3. Retrieve top-k most relevant entities with scores and hop distances.
4. Reconstruct the multi-hop reasoning chain from seeds to top entities.
5. Return a structured QueryResponse with the reasoning chain, ranked
   entities, and a natural-language explanation.

This design mirrors the fast-graphrag framework's approach of using
PageRank to guide agent retrieval in a KG, but is self-contained and
requires no external API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .knowledge_graph import KnowledgeGraph
from .pagerank_retrieval import PageRankRetrieval, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Structured response from a GraphRAG query."""
    query: str
    seed_entities: list[str]
    top_k_entities: list[RetrievalResult]
    reasoning_chains: list[list[str]]            # multi-hop paths
    explanation: str
    metadata: dict = field(default_factory=dict)

    def pretty_print(self) -> str:
        lines = [
            "=" * 70,
            f"Query   : {self.query}",
            f"Seeds   : {', '.join(self.seed_entities)}",
            "=" * 70,
            "",
            "Top Entities (PageRank-ranked):",
            "-" * 40,
        ]
        for r in self.top_k_entities:
            hop = f"  hop={r.hop_distance}" if r.hop_distance is not None else ""
            lines.append(
                f"  #{r.rank:02d}  {r.entity_name:<35s}  score={r.score:.6f}{hop}"
            )

        if self.reasoning_chains:
            lines += ["", "Reasoning Chains (multi-hop paths):"]
            for i, chain in enumerate(self.reasoning_chains, 1):
                lines.append(f"  Chain {i}: {' -> '.join(chain)}")

        lines += ["", "Explanation:", self.explanation]
        lines.append("=" * 70)
        return "\n".join(lines)


class GraphRAGQueryEngine:
    """
    High-level query engine combining KG retrieval with PageRank scoring.

    Parameters
    ----------
    kg              : KnowledgeGraph
    p               : teleportation probability for PPR
    k               : default number of results to return
    seed_extractor  : callable(query) -> list[str] for extracting seed entities.
                      Defaults to a simple keyword matcher against KG entity names.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        p: float = 0.15,
        k: int = 10,
        seed_extractor=None,
    ) -> None:
        self.kg = kg
        self.k = k
        self.retrieval = PageRankRetrieval(kg, p=p)
        self._seed_extractor = seed_extractor or self._default_seed_extractor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        seed_entities: Optional[list[str]] = None,
        k: Optional[int] = None,
        max_chain_hops: int = 6,
    ) -> QueryResponse:
        """
        Answer a multi-hop query using personalised PageRank.

        Parameters
        ----------
        query_text    : natural-language query string
        seed_entities : explicit seeds (overrides auto-extraction if provided)
        k             : override default top-k
        max_chain_hops: maximum path length for reasoning chains
        """
        k = k or self.k

        # Step 1: Extract seeds
        if seed_entities is None:
            seed_entities = self._seed_extractor(query_text)
            logger.info("Auto-extracted seeds: %s", seed_entities)
        else:
            logger.info("Using provided seeds: %s", seed_entities)

        if not seed_entities:
            return QueryResponse(
                query=query_text,
                seed_entities=[],
                top_k_entities=[],
                reasoning_chains=[],
                explanation="No seed entities found in the knowledge graph for this query.",
            )

        # Step 2: Personalised PageRank retrieval
        top_entities = self.retrieval.retrieve(seed_entities, k=k)

        # Step 3: Reconstruct reasoning chains (seeds -> top non-seed entities)
        chains = self._build_chains(seed_entities, top_entities, max_chain_hops)

        # Step 4: Generate explanation
        explanation = self._build_explanation(query_text, seed_entities, top_entities, chains)

        return QueryResponse(
            query=query_text,
            seed_entities=seed_entities,
            top_k_entities=top_entities,
            reasoning_chains=chains,
            explanation=explanation,
            metadata={
                "n_entities": self.kg.n_entities,
                "n_relations": self.kg.n_relations,
                "p_teleport": self.retrieval.p,
            },
        )

    def add_knowledge(self, triples: list[tuple]) -> "GraphRAGQueryEngine":
        """Incrementally add (subject, relation, object) triples."""
        self.kg.add_triples(triples)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_seed_extractor(self, query: str) -> list[str]:
        """
        Simple keyword-based seed extraction.

        Checks for exact (case-insensitive) substring matches of KG entity
        names within the query string.  Production systems would use an NER
        model or LLM-based entity linker here.
        """
        query_lower = query.lower()
        matched = []
        for name in self.kg.all_entity_names():
            if name.lower() in query_lower:
                matched.append(name)
        # Prefer longer matches (more specific entities first)
        matched.sort(key=lambda x: -len(x))
        return matched[:5]  # cap at 5 seeds

    def _build_chains(
        self,
        seeds: list[str],
        top_entities: list[RetrievalResult],
        max_hops: int,
    ) -> list[list[str]]:
        """Find shortest multi-hop paths from any seed to top non-seed entities."""
        chains: list[list[str]] = []
        seed_set = set(seeds)
        targets = [r.entity_name for r in top_entities if r.entity_name not in seed_set]

        for seed in seeds:
            for target in targets[:5]:  # limit to 5 targets per seed
                paths = self.retrieval.explain_path(seed, target, max_hops)
                chains.extend(paths)
                if len(chains) >= 10:  # cap total chains
                    return chains

        return chains

    def _build_explanation(
        self,
        query: str,
        seeds: list[str],
        top_entities: list[RetrievalResult],
        chains: list[list[str]],
    ) -> str:
        top_names = [r.entity_name for r in top_entities[:3]]
        seed_str = ", ".join(f'"{s}"' for s in seeds)
        top_str = ", ".join(f'"{n}"' for n in top_names)

        chain_str = ""
        if chains:
            best = min(chains, key=len)
            chain_str = (
                f"\nThe most direct reasoning path found is:\n"
                f"  {' -> '.join(best)}"
            )

        return (
            f'Query: "{query}"\n\n'
            f"Personalised PageRank was seeded from {seed_str} and propagated\n"
            f"importance through the knowledge graph connection edges.  The\n"
            f"highest-ranked entities are {top_str}, which are most reachable\n"
            f"from the query seeds via multi-hop graph traversal.{chain_str}\n\n"
            f"Entities are ranked by their personalised PageRank score, which\n"
            f"combines proximity to seeds (low hop-distance) with structural\n"
            f"authority (many entities point to them in the KG)."
        )
