"""
ScalabilityMetrics: Measures runtime and memory performance of PageRank.

Metrics computed:
  - Wall-clock time (total, load, iterate)
  - Time per iteration
  - Edges processed per second
  - Peak memory usage (RSS via tracemalloc)
  - Memory per edge
  - Theoretical memory (sparse matrix NNZ + dense vector)
  - Scalability index: time / (N + nnz) -- should be roughly constant
"""
from __future__ import annotations
import tracemalloc
import time
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np


@dataclass
class ScalabilityResult:
    n_nodes: int
    n_edges: int
    n_iterations: int
    time_total_sec: float
    time_per_iter_sec: float
    edges_per_sec: float
    peak_memory_mb: float
    memory_per_edge_bytes: float
    scalability_index: float        # time / (N + nnz) in microseconds

    def as_dict(self) -> dict:
        return {
            "Nodes (N)":               self.n_nodes,
            "Edges (nnz)":             self.n_edges,
            "Iterations":              self.n_iterations,
            "Total Time (s)":          self.time_total_sec,
            "Time/Iteration (ms)":     self.time_per_iter_sec * 1000,
            "Edges/sec":               self.edges_per_sec,
            "Peak Memory (MB)":        self.peak_memory_mb,
            "Memory/Edge (bytes)":     self.memory_per_edge_bytes,
            "Scalability Index (µs/edge)": self.scalability_index * 1e6,
        }


class ScalabilityMetrics:
    """Benchmark memory and time performance of a PageRank run."""

    def __init__(self, n_nodes: int, n_edges: int):
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def measure(self, fn: Callable, *args, **kwargs) -> tuple[Any, "ScalabilityResult"]:
        """
        Run fn(*args, **kwargs), measure time and peak memory, return (result, metrics).
        """
        tracemalloc.start()
        t0 = time.perf_counter()

        result = fn(*args, **kwargs)

        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        n_iter = getattr(result, "iterations", 1)

        return result, ScalabilityResult(
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            n_iterations=n_iter,
            time_total_sec=elapsed,
            time_per_iter_sec=elapsed / max(n_iter, 1),
            edges_per_sec=self.n_edges * n_iter / max(elapsed, 1e-9),
            peak_memory_mb=peak_bytes / 1e6,
            memory_per_edge_bytes=peak_bytes / max(self.n_edges, 1),
            scalability_index=elapsed / max(self.n_nodes + self.n_edges, 1),
        )

    @staticmethod
    def theoretical_memory_mb(n_nodes: int, nnz: int) -> float:
        """
        Estimate theoretical minimum memory for CSR matrix + two dense vectors.
        CSR: (nnz * 8 bytes data) + (nnz * 4 bytes col_ind) + ((N+1) * 4 bytes row_ptr)
        Vectors: 2 * N * 8 bytes (float64)
        """
        csr_bytes = nnz * 12 + (n_nodes + 1) * 4
        vec_bytes = 2 * n_nodes * 8
        return (csr_bytes + vec_bytes) / 1e6
