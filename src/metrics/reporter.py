"""
MetricsReporter — Goldman Sachs Design System
==============================================

All figures follow Goldman Sachs brand guidelines:
  Primary Navy   #003366   — titles, primary data series, spines
  GS Gold        #C9A84C   — accent lines, secondary series, highlights
  Slate          #4A5568   — axis labels, body text
  Pale Blue      #EBF2FA   — axes background, alternate fills
  Mid Gray       #A0AEC0   — grid lines, minor elements
  White          #FFFFFF   — figure background, annotation boxes
  Soft Red       #C0392B   — warnings, negative signals (Bland-Altman bias)
  Soft Green     #1E7145   — positive signals, Hit@k bars

Typography conventions:
  - Titles      : 13 pt, bold, GS Navy
  - Subtitles   : 10 pt, medium, Slate
  - Axis labels : 10 pt, Slate
  - Tick labels : 8.5 pt, Slate
  - Annotations : 8.5 pt, white-on-navy or navy-on-white

Layout:
  - No top/right spines
  - Bottom/left spines: GS Navy, linewidth 1.0
  - Grid: horizontal only (where sensible), Mid Gray, alpha 0.4
  - Figure background: white; axes background: Pale Blue
  - All figures exported at 180 dpi (press quality)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

logger = logging.getLogger(__name__)

# ── Goldman Sachs colour palette ──────────────────────────────────────────────
GS_NAVY       = "#003366"
GS_GOLD       = "#C9A84C"
GS_SLATE      = "#4A5568"
GS_PALE_BLUE  = "#EBF2FA"
GS_MID_GRAY   = "#A0AEC0"
GS_LIGHT_GRAY = "#E2E8F0"
GS_WHITE      = "#FFFFFF"
GS_RED        = "#C0392B"
GS_GREEN      = "#1E7145"
GS_NAVY_LIGHT = "#1A5276"   # secondary navy (gradient mid-point)
GS_GOLD_LIGHT = "#F0D080"   # lighter gold for fills

# Ordered series palette for multi-line / multi-bar plots
# Cycles navy → gold → variants of each
GS_SERIES = [
    "#003366",   # GS Navy
    "#C9A84C",   # GS Gold
    "#1A5276",   # Navy-2
    "#E8A820",   # Gold-2
    "#2E86C1",   # Blue accent
    "#7D6608",   # Dark gold
    "#5D6D7E",   # Slate-blue
    "#A9CCE3",   # Pale blue accent
]

# ── Shared rcParams ───────────────────────────────────────────────────────────
GS_STYLE = {
    "figure.facecolor":     GS_WHITE,
    "axes.facecolor":       GS_PALE_BLUE,
    "axes.edgecolor":       GS_NAVY,
    "axes.linewidth":       0.9,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.titlecolor":      GS_NAVY,
    "axes.labelsize":       10,
    "axes.labelcolor":      GS_SLATE,
    "axes.labelweight":     "medium",
    "axes.grid":            True,
    "axes.axisbelow":       True,
    "grid.color":           GS_MID_GRAY,
    "grid.linewidth":       0.5,
    "grid.alpha":           0.45,
    "xtick.labelsize":      8.5,
    "ytick.labelsize":      8.5,
    "xtick.color":          GS_SLATE,
    "ytick.color":          GS_SLATE,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "legend.fontsize":      8.5,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     GS_LIGHT_GRAY,
    "legend.facecolor":     GS_WHITE,
    "lines.linewidth":      2.0,
    "lines.solid_capstyle": "round",
    "patch.linewidth":      0.6,
    "savefig.dpi":          180,
    "savefig.facecolor":    GS_WHITE,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.15,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Arial", "DejaVu Sans"],
}

# ── Custom navy→gold colormap for scatter plots ───────────────────────────────
_GS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "gs_navy_gold",
    [GS_NAVY, GS_NAVY_LIGHT, "#2E86C1", GS_GOLD_LIGHT, GS_GOLD],
    N=256,
)


def _gs_spine(ax, bottom_color=GS_NAVY, lw=0.9):
    """Apply GS spine styling: only bottom + left, navy coloured."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(bottom_color)
        ax.spines[spine].set_linewidth(lw)


def _gs_title(ax, title: str, subtitle: str = ""):
    """Set a two-line title: bold navy main title + slate subtitle."""
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, loc="left", pad=10,
                 fontsize=13 if not subtitle else 12,
                 fontweight="bold", color=GS_NAVY)


def _gs_watermark(fig, text="Goldman Sachs  |  PageRank Analytics"):
    """Add a discreet footer watermark."""
    fig.text(0.99, 0.01, text,
             ha="right", va="bottom", fontsize=6.5,
             color=GS_MID_GRAY, style="italic")


def _gs_stat_box(ax, text: str, xy=(0.04, 0.93)):
    """Navy stat annotation box."""
    ax.annotate(
        text, xy=xy, xycoords="axes fraction",
        fontsize=8.5, color=GS_WHITE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=GS_NAVY, ec="none", alpha=0.88),
    )


# ── Reporter ─────────────────────────────────────────────────────────────────

class MetricsReporter:
    """
    Collect all metric results and publish GS-styled figures and tables.
    """

    def __init__(self, output_dir: str = "results"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self._correctness = None
        self._convergence_sweep: dict = {}
        self._ranking = None
        self._scalability = None
        self._structural = None
        self._structural_scores: Optional[np.ndarray] = None
        self._structural_A = None
        self._graphrag = None
        self._graphrag_results = None
        self._p_sweep_data: dict = {}
        self._scores_iter: Optional[np.ndarray] = None
        self._scores_anal: Optional[np.ndarray] = None

    # ── Data registration ────────────────────────────────────────────────────

    def add_correctness(self, result, scores_iter=None, scores_anal=None):
        self._correctness = result
        self._scores_iter = scores_iter
        self._scores_anal = scores_anal

    def add_convergence_sweep(self, sweep: dict):
        self._convergence_sweep = sweep

    def add_ranking(self, result):
        self._ranking = result

    def add_scalability(self, result):
        self._scalability = result

    def add_structural(self, result, scores, A):
        self._structural = result
        self._structural_scores = scores
        self._structural_A = A

    def add_graphrag(self, result, retrieval_results=None):
        self._graphrag = result
        self._graphrag_results = retrieval_results

    def add_p_sweep(self, p_sweep: dict):
        self._p_sweep_data = p_sweep

    # ── Publish ──────────────────────────────────────────────────────────────

    def publish(self) -> list[str]:
        plt.rcParams.update(GS_STYLE)
        created = []
        created += self._write_tables()
        if self._convergence_sweep:
            created.append(self._fig_convergence())
        if self._structural_scores is not None:
            created.append(self._fig_score_distribution())
        if self._structural_A is not None and self._structural_scores is not None:
            created.append(self._fig_indegree_scatter())
        if self._scores_iter is not None and self._scores_anal is not None:
            created.append(self._fig_analytical_vs_iterative())
        if self._p_sweep_data:
            created.append(self._fig_p_sweep())
        if self._structural_scores is not None:
            created.append(self._fig_top20_bar())
        if self._graphrag_results is not None:
            created.append(self._fig_graphrag_hop_scores())
        if self._graphrag is not None:
            created.append(self._fig_graphrag_precision_recall())
        logger.info("MetricsReporter: %d files written to %s", len(created), self.out)
        return created

    # ── Tables ───────────────────────────────────────────────────────────────

    def _write_tables(self) -> list[str]:
        created = []
        tables = [
            ("table1_correctness.txt",  "Table 1 — Correctness Metrics",       self._correctness),
            ("table2_convergence.txt",  "Table 2 — Convergence Metrics",
             next(iter(self._convergence_sweep.values())) if self._convergence_sweep else None),
            ("table3_ranking.txt",      "Table 3 — Ranking Quality Metrics",   self._ranking),
            ("table4_scalability.txt",  "Table 4 — Scalability Metrics",       self._scalability),
            ("table5_structural.txt",   "Table 5 — Graph Structural Metrics",  self._structural),
            ("table6_graphrag.txt",     "Table 6 — GraphRAG Retrieval Metrics", self._graphrag),
        ]
        for fname, title, result in tables:
            if result is None:
                continue
            path = self.out / fname
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._format_table(title, result.as_dict()))
            created.append(str(path))
            logger.info("Written: %s", path)
        if len(self._convergence_sweep) > 1:
            path = self.out / "table2b_convergence_sweep.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._format_convergence_sweep_table())
            created.append(str(path))
        return created

    @staticmethod
    def _format_table(title: str, data: dict) -> str:
        lines = [title, "=" * 60]
        max_key = max(len(str(k)) for k in data)
        for k, v in data.items():
            vs = "N/A" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
            lines.append(f"  {str(k):<{max_key}}  :  {vs}")
        lines.append("")
        return "\n".join(lines)

    def _format_convergence_sweep_table(self) -> str:
        lines = ["Table 2b — Convergence Sweep across p values", "=" * 72]
        keys = ["n_iterations", "final_residual", "empirical_rate",
                "theoretical_rate", "auc_residual", "elapsed_sec"]
        header = f"  {'Metric':<28}" + "".join(f"  {lb:>10}" for lb in self._convergence_sweep)
        lines.append(header)
        lines.append("  " + "-" * (28 + 12 * len(self._convergence_sweep)))
        for key in keys:
            row = f"  {key:<28}"
            for result in self._convergence_sweep.values():
                val = getattr(result, key, None)
                row += f"  {'N/A':>10}" if val is None else \
                       f"  {val:>10.4g}" if isinstance(val, float) else f"  {val:>10}"
            lines.append(row)
        lines.append("")
        return "\n".join(lines)

    # ── Fig 1: Convergence curves ─────────────────────────────────────────────

    def _fig_convergence(self) -> str:
        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Navy → Gold gradient across p values
        n = len(self._convergence_sweep)
        colors = [_GS_CMAP(i / max(n - 1, 1)) for i in range(n)]

        for (label, result), color in zip(self._convergence_sweep.items(), colors):
            res = np.array(result.residuals)
            iters = np.arange(1, len(res) + 1)
            ax.semilogy(iters, res, color=color, label=label, linewidth=2.0,
                        solid_capstyle="round")
            # Mark convergence point with a gold dot
            ax.scatter(iters[-1], res[-1], color=GS_GOLD, s=40, zorder=5,
                       linewidths=0.8, edgecolors=GS_NAVY)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("L1 Residual  ‖r⁽ᵗ⁺¹⁾ − r⁽ᵗ⁾‖₁")
        _gs_title(ax, "Fig 1 — PageRank Convergence", "Power Iteration · p-sweep")

        legend = ax.legend(title="Teleportation p", loc="upper right",
                           ncol=2, framealpha=0.92)
        legend.get_title().set_color(GS_NAVY)
        legend.get_title().set_fontweight("bold")

        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.yaxis.grid(True, which="both")
        ax.xaxis.grid(False)
        _gs_spine(ax)
        _gs_watermark(fig)
        fig.tight_layout()
        path = str(self.out / "fig1_convergence.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 2: Score distribution CCDF ────────────────────────────────────────

    def _fig_score_distribution(self) -> str:
        r = np.sort(self._structural_scores)[::-1]
        r = r[r > 0]
        N = len(r)
        ccdf = np.arange(1, N + 1) / N

        fig, ax = plt.subplots(figsize=(9, 5.5))

        # Filled area under empirical CCDF
        ax.fill_between(r, ccdf, alpha=0.12, color=GS_NAVY)
        ax.loglog(r, ccdf, color=GS_NAVY, linewidth=2.2, label="Empirical CCDF",
                  solid_capstyle="round")

        if self._structural and not math.isnan(self._structural.powerlaw_alpha):
            alpha = self._structural.powerlaw_alpha
            x_fit = np.logspace(np.log10(r.min()), np.log10(r.max()), 300)
            y_fit = (x_fit / r[0]) ** (-alpha + 1)
            y_fit = y_fit / y_fit[0]
            ax.loglog(x_fit, y_fit, "--", color=GS_GOLD, linewidth=2.0,
                      label=f"Power-law fit  α = {alpha:.2f}")

        ax.set_xlabel("PageRank Score")
        ax.set_ylabel("CCDF  P(R > r)")
        _gs_title(ax, "Fig 2 — PageRank Score Distribution",
                  "Complementary CDF · log-log scale")

        if self._structural:
            _gs_stat_box(ax,
                f"Top 1% nodes hold {self._structural.top1pct_mass:.1%} of PR mass\n"
                f"Gini = {self._structural.gini:.3f}  |  Power-law R² = {self._structural.powerlaw_r2:.3f}")

        legend = ax.legend(loc="lower left")
        _gs_spine(ax)
        _gs_watermark(fig)
        fig.tight_layout()
        path = str(self.out / "fig2_score_distribution.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 3: In-degree vs PageRank scatter ──────────────────────────────────

    def _fig_indegree_scatter(self) -> str:
        r = self._structural_scores
        in_deg = np.array(self._structural_A.sum(axis=1)).flatten()

        N = len(r)
        sample = min(N, 6_000)
        idx = np.random.RandomState(0).choice(N, sample, replace=False)

        fig, ax = plt.subplots(figsize=(8, 6.5))

        sc = ax.scatter(
            in_deg[idx], r[idx],
            c=np.log1p(r[idx]),
            cmap=_GS_CMAP,
            alpha=0.55, s=6, linewidths=0,
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("log(1 + PageRank)", color=GS_SLATE, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=GS_SLATE)
        cbar.outline.set_edgecolor(GS_LIGHT_GRAY)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Weighted In-Degree")
        ax.set_ylabel("PageRank Score")
        _gs_title(ax, "Fig 3 — In-Degree vs PageRank Score", "log-log scatter · sampled 6 k nodes")

        if self._structural:
            _gs_stat_box(ax,
                f"Spearman  ρ = {self._structural.in_degree_spearman_rho:.3f}\n"
                f"Pearson   r = {self._structural.in_degree_pearson_r:.3f}")

        _gs_spine(ax)
        _gs_watermark(fig)
        fig.tight_layout()
        path = str(self.out / "fig3_indegree_vs_pr.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 4: Analytical vs Iterative ────────────────────────────────────────

    def _fig_analytical_vs_iterative(self) -> str:
        a = self._scores_anal
        b = self._scores_iter
        N = len(a)
        sample = min(N, 5_000)
        idx = np.random.RandomState(1).choice(N, sample, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle("Fig 4 — Analytical vs Iterative PageRank",
                     fontsize=13, fontweight="bold", color=GS_NAVY, x=0.02, ha="left")

        # Left: identity scatter
        ax = axes[0]
        ax.scatter(a[idx], b[idx], alpha=0.35, s=5,
                   c=np.log1p(a[idx]), cmap=_GS_CMAP, linewidths=0)
        lim = max(a.max(), b.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color=GS_GOLD, linewidth=1.8, label="y = x  (perfect)")
        ax.set_xlabel("Analytical PageRank")
        ax.set_ylabel("Iterative PageRank")
        _gs_title(ax, "Score Scatter", "Analytical ground truth vs power iteration")
        ax.legend()
        if self._correctness:
            _gs_stat_box(ax,
                f"L1 error  = {self._correctness.l1_error:.2e}\n"
                f"L∞ error  = {self._correctness.l_inf_error:.2e}\n"
                f"Pearson r = {self._correctness.pearson_r:.6f}")
        _gs_spine(ax)

        # Right: Bland-Altman
        ax = axes[1]
        mean_ab = (a[idx] + b[idx]) / 2
        diff_ab = b[idx] - a[idx]
        bias  = diff_ab.mean()
        loa   = 1.96 * diff_ab.std()
        ax.scatter(mean_ab, diff_ab, alpha=0.3, s=5,
                   c=np.log1p(mean_ab), cmap=_GS_CMAP, linewidths=0)
        ax.axhline(bias,       color=GS_RED,  linewidth=1.8, label=f"Bias = {bias:.2e}")
        ax.axhline(bias + loa, color=GS_GOLD, linewidth=1.4, linestyle="--",
                   label=f"+1.96σ = {bias+loa:.2e}")
        ax.axhline(bias - loa, color=GS_GOLD, linewidth=1.4, linestyle="--",
                   label=f"−1.96σ = {bias-loa:.2e}")
        ax.axhline(0,          color=GS_NAVY, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Mean  (Analytical + Iterative) / 2")
        ax.set_ylabel("Difference  (Iterative − Analytical)")
        _gs_title(ax, "Bland-Altman Agreement", "Limits of agreement at 95%")
        ax.legend(fontsize=7.5)
        _gs_spine(ax)

        _gs_watermark(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = str(self.out / "fig4_analytical_vs_iterative.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 5: p-sweep triptych ───────────────────────────────────────────────

    def _fig_p_sweep(self) -> str:
        p_vals  = sorted(self._p_sweep_data.keys())
        entropy = [self._p_sweep_data[p]["entropy"]   for p in p_vals]
        gini    = [self._p_sweep_data[p]["gini"]      for p in p_vals]
        top1    = [self._p_sweep_data[p]["top1_score"] for p in p_vals]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Fig 5 — Effect of Teleportation Probability p on PageRank Distribution",
            fontsize=13, fontweight="bold", color=GS_NAVY, x=0.02, ha="left",
        )

        marker_kw = dict(markersize=7, markerfacecolor=GS_GOLD,
                         markeredgecolor=GS_NAVY, markeredgewidth=0.8)

        # Entropy
        ax = axes[0]
        ax.plot(p_vals, entropy, "o-", color=GS_NAVY, linewidth=2.2, **marker_kw)
        ax.fill_between(p_vals, entropy, min(entropy), alpha=0.10, color=GS_NAVY)
        ax.set_xlabel("Teleportation Probability p")
        ax.set_ylabel("Shannon Entropy (nats)")
        _gs_title(ax, "Entropy vs p", "↑ higher = more uniform distribution")
        _gs_spine(ax)

        # Gini
        ax = axes[1]
        ax.plot(p_vals, gini, "s-", color=GS_NAVY, linewidth=2.2, **marker_kw)
        ax.fill_between(p_vals, gini, min(gini), alpha=0.10, color=GS_NAVY)
        ax.set_xlabel("Teleportation Probability p")
        ax.set_ylabel("Gini Coefficient")
        _gs_title(ax, "Gini vs p", "↓ lower = more equal score distribution")
        _gs_spine(ax)

        # Top-1 score (log scale)
        ax = axes[2]
        ax.semilogy(p_vals, top1, "^-", color=GS_NAVY, linewidth=2.2, **marker_kw)
        # GS Gold reference line at 1/N (uniform)
        if self._scalability:
            uniform = 1.0 / self._scalability.n_nodes
            ax.axhline(uniform, color=GS_GOLD, linewidth=1.4, linestyle="--",
                       label=f"Uniform 1/N = {uniform:.2e}")
            ax.legend(fontsize=8)
        ax.set_xlabel("Teleportation Probability p")
        ax.set_ylabel("Top-1 Node PageRank Score")
        _gs_title(ax, "Top-1 Score vs p", "log scale — authority concentration")
        _gs_spine(ax)

        _gs_watermark(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = str(self.out / "fig5_p_sweep.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 6: Top-20 bar chart ───────────────────────────────────────────────

    def _fig_top20_bar(self) -> str:
        r = self._structural_scores
        k = min(20, len(r))
        top_idx    = np.argsort(-r)[:k]
        top_scores = r[top_idx]
        labels = [str(i) for i in top_idx]

        # Gradient: top bar is GS Navy, fades to a lighter navy-blue
        bar_colors = [_GS_CMAP(1.0 - (i / max(k - 1, 1)) * 0.65) for i in range(k)]

        fig, ax = plt.subplots(figsize=(13, 5.5))
        bars = ax.bar(range(k), top_scores, color=bar_colors,
                      edgecolor=GS_WHITE, linewidth=0.6, width=0.72)

        # Gold accent stripe on the #1 bar
        bars[0].set_edgecolor(GS_GOLD)
        bars[0].set_linewidth(2.0)

        ax.set_xticks(range(k))
        ax.set_xticklabels([f"Node\n{l}" for l in labels], fontsize=7.5)
        ax.set_ylabel("PageRank Score")
        _gs_title(ax, "Fig 6 — Top-20 PageRank Nodes",
                  f"web-Google  ·  N = {len(r):,} nodes  ·  p = 0.15")

        # Value labels
        for bar, score in zip(bars, top_scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top_scores.max() * 0.005,
                    f"{score:.5f}", ha="center", va="bottom",
                    fontsize=6.2, color=GS_SLATE)

        # Rank badge on first bar
        ax.text(0, top_scores[0] / 2, "#1", ha="center", va="center",
                fontsize=10, fontweight="bold", color=GS_WHITE)

        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _gs_spine(ax)
        _gs_watermark(fig)
        fig.tight_layout()
        path = str(self.out / "fig6_top20_nodes.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 7: GraphRAG hop scores ────────────────────────────────────────────

    def _fig_graphrag_hop_scores(self) -> str:
        results = self._graphrag_results
        hop_scores: dict[int, list[float]] = {}
        for r in results:
            if r.hop_distance is not None:
                hop_scores.setdefault(r.hop_distance, []).append(r.score)

        hops   = sorted(hop_scores.keys())
        means  = [np.mean(hop_scores[h]) for h in hops]
        stds   = [np.std(hop_scores[h])  for h in hops]
        counts = [len(hop_scores[h])     for h in hops]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(
            "Fig 7 — GraphRAG: Personalised PageRank Score by Hop Distance",
            fontsize=13, fontweight="bold", color=GS_NAVY, x=0.02, ha="left",
        )

        # ── Left: mean PPR score per hop ──────────────────────────────────────
        ax = axes[0]
        bar_cols = [GS_NAVY if h == 0 else
                    GS_NAVY_LIGHT if h == 1 else
                    "#2E86C1" if h == 2 else GS_MID_GRAY
                    for h in hops]
        bars = ax.bar(hops, means, color=bar_cols, width=0.55,
                      edgecolor=GS_WHITE, linewidth=0.8)
        ax.errorbar(hops, means, yerr=stds, fmt="none",
                    ecolor=GS_GOLD, elinewidth=1.8, capsize=5, capthick=1.8)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.08,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=8, color=GS_SLATE)

        ax.set_xlabel("Hop Distance from Seeds")
        ax.set_ylabel("Mean PPR Score")
        _gs_title(ax, "Mean PPR Score by Hop Distance", "Error bar = ±1 std dev  ·  Gold = uncertainty")
        ax.set_xticks(hops)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _gs_spine(ax)

        # ── Right: entity count per hop ───────────────────────────────────────
        ax = axes[1]
        gold_bars = ax.bar(hops, counts, color=GS_GOLD, width=0.55,
                           edgecolor=GS_NAVY, linewidth=0.8)
        for bar, c in zip(gold_bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    str(c), ha="center", va="bottom", fontsize=9,
                    color=GS_NAVY, fontweight="bold")
        ax.set_xlabel("Hop Distance from Seeds")
        ax.set_ylabel("Number of Entities")
        _gs_title(ax, "Entity Count by Hop Distance", "Knowledge graph breadth per hop")
        ax.set_xticks(hops)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _gs_spine(ax)

        _gs_watermark(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = str(self.out / "fig7_graphrag_hop_scores.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Fig 8: GraphRAG Precision / Recall / Hit@k ───────────────────────────

    def _fig_graphrag_precision_recall(self) -> str:
        g    = self._graphrag
        ks   = sorted(g.precision_at_k.keys())
        prec = [g.precision_at_k[k] for k in ks]
        rec  = [g.recall_at_k[k]    for k in ks]

        x     = np.arange(len(ks))
        width = 0.36

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle("Fig 8 — GraphRAG Retrieval Quality Metrics",
                     fontsize=13, fontweight="bold", color=GS_NAVY, x=0.02, ha="left")

        # ── Left: Precision & Recall grouped ─────────────────────────────────
        ax = axes[0]
        b1 = ax.bar(x - width / 2, prec, width, label="Precision@k",
                    color=GS_NAVY, edgecolor=GS_WHITE, linewidth=0.7)
        b2 = ax.bar(x + width / 2, rec,  width, label="Recall@k",
                    color=GS_GOLD, edgecolor=GS_NAVY,  linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k = {k}" for k in ks])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        _gs_title(ax, "Precision@k  &  Recall@k", "PageRank-ranked retrieval vs ground truth")
        ax.legend()
        for bar, val in [(b, v) for bars, vals in [(b1, prec), (b2, rec)]
                         for b, v in zip(bars, vals)]:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=8.5, color=GS_SLATE)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _gs_spine(ax)

        # ── Right: Hit@k + MRR indicator ─────────────────────────────────────
        ax = axes[1]
        hit_ks   = [1, 3, 5, 10]
        hit_vals = [g.hit_at_1, g.hit_at_3, g.hit_at_5, g.hit_at_10]
        hit_cols = [GS_GREEN if v == 1.0 else GS_GOLD for v in hit_vals]
        hit_bars = ax.bar([f"Hit@{k}" for k in hit_ks], hit_vals,
                          color=hit_cols, edgecolor=GS_NAVY, linewidth=0.8, width=0.5)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Hit Rate")
        _gs_title(ax, "Hit@k",
                  f"MRR = {g.mrr:.3f}  ·  Personalisation fidelity ρ = {g.personalisation_fidelity:.3f}")
        for bar, val in zip(hit_bars, hit_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    "✓" if val == 1.0 else f"{val:.2f}",
                    ha="center", fontsize=11,
                    color=GS_GREEN if val == 1.0 else GS_SLATE,
                    fontweight="bold")

        # MRR annotation box
        _gs_stat_box(ax, f"MRR = {g.mrr:.3f}\nSeed/Non-Seed = {g.seed_vs_nonseed_ratio:.1f}×",
                     xy=(0.60, 0.88))
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _gs_spine(ax)

        _gs_watermark(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = str(self.out / "fig8_graphrag_precision_recall.png")
        fig.savefig(path)
        plt.close(fig)
        return path
