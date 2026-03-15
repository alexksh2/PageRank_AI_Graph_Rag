"""
WebGraphLoader: Loads Stanford SNAP-format web graphs into sparse CSR matrices.

Supported formats:
  - SNAP edge-list  (lines: "from_id\\tto_id" or "from_id to_id", comments with #)
  - Any whitespace-delimited edge list

The loader:
  1. Reads every edge once (streaming) to stay memory-efficient on large files.
  2. Remaps raw node IDs to a contiguous [0, N) index.
  3. Builds a column-stochastic transition matrix A in CSR format:
       A[i, j] = 1 / out_degree(j)   if j -> i is an edge
       A[:, j] = 1/N                  if j is a dangling node (out-degree 0)
     Dangling nodes are handled lazily inside PageRankEngine so that the
     sparse matrix itself stays sparse (dangling correction is a dense rank-1
     update applied only during iteration).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class WebGraphLoader:
    """Load a web graph from a SNAP-style edge-list file."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {self.filepath}")

        # populated after load()
        self.n_nodes: int = 0
        self.n_edges: int = 0
        self.id_to_idx: dict[int, int] = {}
        self.idx_to_id: list[int] = []
        self.A: Optional[sp.csr_matrix] = None  # column-stochastic sparse matrix
        self.dangling_mask: Optional[np.ndarray] = None  # bool array, True = dangling

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "WebGraphLoader":
        """Parse the file and build the transition matrix. Returns self."""
        t0 = time.perf_counter()
        logger.info("Loading graph from %s ...", self.filepath)

        edges = self._read_edges()
        self._build_matrix(edges)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Graph loaded: %d nodes, %d edges in %.2fs",
            self.n_nodes,
            self.n_edges,
            elapsed,
        )
        return self

    def summary(self) -> str:
        lines = [
            f"File            : {self.filepath.name}",
            f"Nodes           : {self.n_nodes:,}",
            f"Edges           : {self.n_edges:,}",
            f"Dangling nodes  : {int(self.dangling_mask.sum()):,}",
            f"Density         : {self.n_edges / max(self.n_nodes**2, 1):.2e}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_edges(self) -> list[tuple[int, int]]:
        """Stream the file and collect (src_idx, dst_idx) integer pairs."""
        raw_id_set: set[int] = set()
        raw_edges: list[tuple[int, int]] = []

        file_size = os.path.getsize(self.filepath)
        logger.info("File size: %.1f MB", file_size / 1e6)

        with open(self.filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                raw_id_set.add(u)
                raw_id_set.add(v)
                raw_edges.append((u, v))

        # Build contiguous index mapping (sorted for reproducibility)
        sorted_ids = sorted(raw_id_set)
        self.idx_to_id = sorted_ids
        self.id_to_idx = {raw_id: idx for idx, raw_id in enumerate(sorted_ids)}
        self.n_nodes = len(sorted_ids)
        self.n_edges = len(raw_edges)

        # Remap to contiguous indices
        mapped = [(self.id_to_idx[u], self.id_to_idx[v]) for u, v in raw_edges]
        return mapped

    def _build_matrix(self, edges: list[tuple[int, int]]) -> None:
        """Construct the column-stochastic sparse matrix in CSR format."""
        N = self.n_nodes

        # Count out-degrees
        out_degree = np.zeros(N, dtype=np.int64)
        for u, _ in edges:
            out_degree[u] += 1

        # dangling nodes: out-degree == 0
        self.dangling_mask = out_degree == 0

        # Build COO data for A[dst, src] = 1/out_degree[src]
        rows, cols, data = [], [], []
        for src, dst in edges:
            if out_degree[src] > 0:
                rows.append(dst)
                cols.append(src)
                data.append(1.0 / out_degree[src])

        self.A = sp.csr_matrix(
            (data, (rows, cols)), shape=(N, N), dtype=np.float64
        )
        logger.info(
            "Sparse matrix built: shape=%s, nnz=%d", self.A.shape, self.A.nnz
        )
