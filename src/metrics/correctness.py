"""
CorrectnessMetrics: Validates PageRank scores against analytical ground truth.

Metrics computed:
  - L1 error   : sum of absolute differences
  - L2 / RMSE  : root mean squared error
  - L∞ error   : max absolute difference (worst-case node)
  - Sum check   : |sum(r) - 1| (must be ~0)
  - Non-negativity rate : fraction of r[i] >= 0
  - Floor check : fraction of r[i] >= p/N (teleportation lower bound)
  - Pearson correlation : linear correlation between iterative and analytical
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class CorrectnessResult:
    l1_error: float
    l2_error: float          # RMSE
    l_inf_error: float
    sum_deviation: float     # |sum(r) - 1|
    non_negative_rate: float # fraction >= 0
    floor_rate: float        # fraction >= p/N
    pearson_r: float
    pearson_p: float
    worst_node_idx: int      # index of node with max absolute error

    def as_dict(self) -> dict:
        return {
            "L1 Error":             self.l1_error,
            "L2 / RMSE":            self.l2_error,
            "L∞ Error":             self.l_inf_error,
            "Sum Deviation":        self.sum_deviation,
            "Non-Negative Rate":    self.non_negative_rate,
            "Floor Rate (≥p/N)":    self.floor_rate,
            "Pearson r":            self.pearson_r,
            "Pearson p-value":      self.pearson_p,
            "Worst Node Index":     self.worst_node_idx,
        }


class CorrectnessMetrics:
    """Compare iterative PageRank scores against analytical ground truth."""

    def __init__(self, r_iterative: np.ndarray, r_analytical: np.ndarray, p: float, N: int):
        self.r_iter = r_iterative
        self.r_anal = r_analytical
        self.p = p
        self.N = N

    def compute(self) -> CorrectnessResult:
        diff = np.abs(self.r_iter - self.r_anal)
        pearson_r, pearson_p = stats.pearsonr(self.r_iter, self.r_anal)
        return CorrectnessResult(
            l1_error=float(diff.sum()),
            l2_error=float(np.sqrt(np.mean(diff ** 2))),
            l_inf_error=float(diff.max()),
            sum_deviation=float(abs(self.r_iter.sum() - 1.0)),
            non_negative_rate=float((self.r_iter >= 0).mean()),
            floor_rate=float((self.r_iter >= self.p / self.N).mean()),
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            worst_node_idx=int(diff.argmax()),
        )
