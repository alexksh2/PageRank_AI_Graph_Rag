"""
ConvergenceMetrics: Analyses the iterative convergence behaviour of PageRank.

Metrics computed:
  - Residuals per iteration        (L1 norm of r^(t+1) - r^(t))
  - Iterations to reach tolerance
  - Empirical convergence rate     (geometric decay factor per step)
  - Theoretical convergence rate   (1 - p, from spectral gap)
  - Ratio empirical / theoretical  (how close to worst-case?)
  - Area under residual curve      (total convergence work)
  - First iteration below each threshold (1e-3, 1e-6, 1e-8)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ConvergenceResult:
    residuals: list[float]
    n_iterations: int
    converged: bool
    final_residual: float
    theoretical_rate: float          # (1 - p)
    empirical_rate: float            # geometric fit of residual decay
    rate_ratio: float                # empirical / theoretical
    auc_residual: float              # area under residual curve (trapezoidal)
    iters_to_1e3: int | None
    iters_to_1e6: int | None
    iters_to_1e8: int | None
    elapsed_sec: float

    def as_dict(self) -> dict:
        return {
            "Iterations":            self.n_iterations,
            "Converged":             self.converged,
            "Final Residual":        self.final_residual,
            "Theoretical Rate (1-p)": self.theoretical_rate,
            "Empirical Rate":        self.empirical_rate,
            "Rate Ratio (emp/th)":   self.rate_ratio,
            "AUC Residual":          self.auc_residual,
            "Iters to <1e-3":        self.iters_to_1e3,
            "Iters to <1e-6":        self.iters_to_1e6,
            "Iters to <1e-8":        self.iters_to_1e8,
            "Elapsed (s)":           self.elapsed_sec,
        }


class ConvergenceMetrics:
    """Analyse convergence behaviour from the residual sequence."""

    def __init__(self, residuals: list[float], p: float, elapsed_sec: float, converged: bool):
        self.residuals = residuals
        self.p = p
        self.elapsed_sec = elapsed_sec
        self.converged = converged

    def compute(self) -> ConvergenceResult:
        res = np.array(self.residuals, dtype=np.float64)
        n = len(res)

        # Empirical convergence rate: fit log(residual) ~ a + rate*iter
        # geometric decay: r_t = r_0 * rate^t => log(r_t) = log(r_0) + t*log(rate)
        safe_res = res[res > 0]
        if len(safe_res) > 3:
            iters = np.arange(len(safe_res), dtype=float)
            log_res = np.log(safe_res)
            # Linear fit: slope = log(rate)
            slope, _, _, _, _ = np.polyfit(iters, log_res, 1, full=False) if False else \
                (np.polyfit(iters, log_res, 1)[0], None, None, None, None)
            empirical_rate = float(np.exp(slope))
        else:
            empirical_rate = float(1 - self.p)

        theoretical_rate = float(1 - self.p)
        rate_ratio = empirical_rate / theoretical_rate if theoretical_rate > 0 else float("nan")

        # Area under residual curve (trapezoidal integration)
        auc = float(np.trapz(res))

        def first_below(threshold):
            idxs = np.where(res < threshold)[0]
            return int(idxs[0]) + 1 if len(idxs) > 0 else None

        return ConvergenceResult(
            residuals=self.residuals,
            n_iterations=n,
            converged=self.converged,
            final_residual=float(res[-1]),
            theoretical_rate=theoretical_rate,
            empirical_rate=empirical_rate,
            rate_ratio=rate_ratio,
            auc_residual=auc,
            iters_to_1e3=first_below(1e-3),
            iters_to_1e6=first_below(1e-6),
            iters_to_1e8=first_below(1e-8),
            elapsed_sec=self.elapsed_sec,
        )
