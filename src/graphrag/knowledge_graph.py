"""
KnowledgeGraph: Directed entity-relation graph for GraphRAG.

Nodes  = entities  (e.g. "Marie Curie", "radium", "radiotherapy")
Edges  = typed relations  (e.g. "discovered", "used_in", "led_to")

The graph is backed by an adjacency list and supports:
  - Incremental ingestion of (subject, relation, object) triples
  - Persistence to / from a simple JSON format
  - Materialisation of a scipy sparse adjacency matrix for PageRank

The knowledge graph is intentionally schema-free so it can be populated
from arbitrary text corpora, structured databases, or external KGs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""
    idx: int
    name: str
    entity_type: str = "entity"
    metadata: dict = field(default_factory=dict)


@dataclass
class Relation:
    """A directed edge in the knowledge graph."""
    src_idx: int
    dst_idx: int
    relation_type: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


class KnowledgeGraph:
    """
    Mutable, incrementally updatable directed knowledge graph.

    Usage
    -----
    kg = KnowledgeGraph()
    kg.add_triple("Marie Curie", "discovered", "polonium")
    kg.add_triple("Marie Curie", "discovered", "radium")
    kg.add_triple("radium", "used_in", "radiotherapy")
    kg.add_triple("radiotherapy", "advances", "medical imaging")
    A = kg.adjacency_matrix()
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}       # name -> Entity
        self._relations: list[Relation] = []
        self._adj: dict[int, list[int]] = {}          # src_idx -> [dst_idx]
        self._rev_adj: dict[int, list[int]] = {}      # dst_idx -> [src_idx]
        self._dirty = True                             # matrix needs rebuild

        # cached sparse matrix
        self._A: Optional[sp.csr_matrix] = None
        self._N: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        weight: float = 1.0,
        subject_type: str = "entity",
        object_type: str = "entity",
        metadata: Optional[dict] = None,
    ) -> "KnowledgeGraph":
        """Add a (subject, relation, object) triple to the graph."""
        s_entity = self._get_or_create(subject, subject_type)
        o_entity = self._get_or_create(obj, object_type)

        rel = Relation(
            src_idx=s_entity.idx,
            dst_idx=o_entity.idx,
            relation_type=relation,
            weight=weight,
            metadata=metadata or {},
        )
        self._relations.append(rel)
        self._adj.setdefault(s_entity.idx, []).append(o_entity.idx)
        self._rev_adj.setdefault(o_entity.idx, []).append(s_entity.idx)
        self._dirty = True
        return self

    def add_triples(self, triples: list[tuple]) -> "KnowledgeGraph":
        """Bulk-add (subject, relation, object) or (subject, relation, object, weight)."""
        for triple in triples:
            if len(triple) == 3:
                self.add_triple(*triple)
            elif len(triple) == 4:
                self.add_triple(*triple)
        return self

    # ------------------------------------------------------------------
    # Graph accessors
    # ------------------------------------------------------------------

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    @property
    def n_relations(self) -> int:
        return len(self._relations)

    def entity(self, name: str) -> Optional[Entity]:
        return self._entities.get(name)

    def idx_to_name(self, idx: int) -> str:
        for name, ent in self._entities.items():
            if ent.idx == idx:
                return name
        return f"<unknown:{idx}>"

    def neighbors(self, name: str) -> list[str]:
        """Return out-neighbors (entities this entity points to)."""
        ent = self._entities.get(name)
        if ent is None:
            return []
        return [self.idx_to_name(j) for j in self._adj.get(ent.idx, [])]

    def in_neighbors(self, name: str) -> list[str]:
        """Return in-neighbors (entities that point to this entity)."""
        ent = self._entities.get(name)
        if ent is None:
            return []
        return [self.idx_to_name(j) for j in self._rev_adj.get(ent.idx, [])]

    def relations_from(self, name: str) -> list[tuple[str, str]]:
        """Return list of (relation_type, target_entity_name) from name."""
        ent = self._entities.get(name)
        if ent is None:
            return []
        result = []
        for rel in self._relations:
            if rel.src_idx == ent.idx:
                result.append((rel.relation_type, self.idx_to_name(rel.dst_idx)))
        return result

    def all_entity_names(self) -> list[str]:
        return list(self._entities.keys())

    # ------------------------------------------------------------------
    # Sparse adjacency matrix
    # ------------------------------------------------------------------

    def adjacency_matrix(self, weighted: bool = False) -> sp.csr_matrix:
        """
        Return the column-stochastic adjacency matrix A (N x N).

        A[i, j] = 1 / out_degree(j)   if j -> i exists
               = 1/N                   if j is dangling

        If weighted=True, edge weights are used instead of binary adjacency.
        """
        if not self._dirty and self._A is not None:
            return self._A

        N = self.n_entities
        if N == 0:
            return sp.csr_matrix((0, 0))

        # Count out-degrees (weighted or unweighted)
        out_degree = np.zeros(N, dtype=np.float64)
        for rel in self._relations:
            out_degree[rel.src_idx] += rel.weight if weighted else 1.0

        rows, cols, data = [], [], []
        for rel in self._relations:
            src, dst = rel.src_idx, rel.dst_idx
            w = rel.weight if weighted else 1.0
            if out_degree[src] > 0:
                rows.append(dst)
                cols.append(src)
                data.append(w / out_degree[src]) #normalise the weight

        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)

        # Handle dangling nodes
        dangling = np.where(out_degree == 0)[0]
        if len(dangling) > 0:
            dan_rows = np.tile(np.arange(N), len(dangling))
            dan_cols = np.repeat(dangling, N)
            dan_data = np.full(len(dan_rows), 1.0 / N)
            A = A + sp.csr_matrix(
                (dan_data, (dan_rows, dan_cols)), shape=(N, N)
            )

        self._A = A.tocsr()
        self._N = N
        self._dirty = False
        return self._A

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "entities": [
                {"idx": e.idx, "name": e.name, "type": e.entity_type, "meta": e.metadata}
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "src": r.src_idx,
                    "dst": r.dst_idx,
                    "type": r.relation_type,
                    "weight": r.weight,
                    "meta": r.metadata,
                }
                for r in self._relations
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("KG saved to %s (%d entities, %d relations)", path, self.n_entities, self.n_relations)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        kg = cls() #same as kg = KnowledgeGraph(), class method
        for e in data["entities"]:
            entity = Entity(idx=e["idx"], name=e["name"], entity_type=e["type"], metadata=e.get("meta", {}))
            kg._entities[e["name"]] = entity
        for r in data["relations"]:
            rel = Relation(
                src_idx=r["src"], dst_idx=r["dst"],
                relation_type=r["type"], weight=r.get("weight", 1.0),
                metadata=r.get("meta", {}),
            )
            kg._relations.append(rel)
            kg._adj.setdefault(rel.src_idx, []).append(rel.dst_idx)
            kg._rev_adj.setdefault(rel.dst_idx, []).append(rel.src_idx)
        logger.info("KG loaded from %s", path)
        return kg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, name: str, entity_type: str) -> Entity:
        if name not in self._entities:
            idx = len(self._entities)
            self._entities[name] = Entity(idx=idx, name=name, entity_type=entity_type)
        return self._entities[name]

    def __repr__(self) -> str:
        return f"KnowledgeGraph(entities={self.n_entities}, relations={self.n_relations})"
