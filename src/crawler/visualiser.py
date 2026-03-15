"""
visualiser.py — Goldman Sachs-styled plots for all 10 crawl experiments.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ── GS Design tokens (shared with reporter.py) ───────────────────────────────
GS_NAVY      = "#003366"
GS_GOLD      = "#C9A84C"
GS_SLATE     = "#4A5568"
GS_PALE_BLUE = "#EBF2FA"
GS_MID_GRAY  = "#A0AEC0"
GS_LIGHT     = "#E2E8F0"
GS_WHITE     = "#FFFFFF"
GS_RED       = "#C0392B"
GS_GREEN     = "#1E7145"
GS_NAVY2     = "#1A5276"
GS_GOLD2     = "#F0D080"

GS_SERIES = [GS_NAVY, GS_GOLD, GS_NAVY2, "#E8A820", "#2E86C1",
             "#7D6608", "#5D6D7E", "#A9CCE3"]

GS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "gs", [GS_NAVY, GS_NAVY2, "#2E86C1", GS_GOLD2, GS_GOLD], N=256
)

GS_STYLE = {
    "figure.facecolor":  GS_WHITE,
    "axes.facecolor":    GS_PALE_BLUE,
    "axes.edgecolor":    GS_NAVY,
    "axes.linewidth":    0.9,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   GS_NAVY,
    "axes.labelsize":    9.5,
    "axes.labelcolor":   GS_SLATE,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        GS_MID_GRAY,
    "grid.linewidth":    0.45,
    "grid.alpha":        0.45,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.color":       GS_SLATE,
    "ytick.color":       GS_SLATE,
    "legend.fontsize":   8,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  GS_LIGHT,
    "lines.linewidth":   2.0,
    "savefig.dpi":       180,
    "savefig.facecolor": GS_WHITE,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
}

SHORT_NAMES = {
    "H0_Random":               "H0 Random",
    "H1_PurePageRank":         "H1 Pure PR",
    "H2_HubAuthority":         "H2 Hub+PR",
    "H3_PR_Robots":            "H3 PR+Robots",
    "H4_QualityWeightedAuth":  "H4 QWA ★",
}

def _short(name: str) -> str:
    for k, v in SHORT_NAMES.items():
        if k in name:
            return v
    return name


def _wm(fig):
    fig.text(0.99, 0.01, "Goldman Sachs  |  Crawl Heuristics",
             ha="right", va="bottom", fontsize=6, color=GS_MID_GRAY, style="italic")


def _spine(ax):
    for s in ("top","right"):   ax.spines[s].set_visible(False)
    for s in ("bottom","left"): ax.spines[s].set_color(GS_NAVY); ax.spines[s].set_linewidth(0.9)


def _title(ax, main, sub=""):
    ax.set_title(f"{main}\n{sub}" if sub else main, loc="left",
                 fontsize=12, fontweight="bold", color=GS_NAVY, pad=8)


def _stat(ax, text, xy=(0.04, 0.93)):
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=8,
                color=GS_WHITE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=GS_NAVY, ec="none", alpha=0.88))


class CrawlVisualiser:
    def __init__(self, output_dir: str = "results"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(GS_STYLE)

    def save(self, fig, name: str) -> str:
        path = str(self.out / name)
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ── EXP-1: Ablation bar chart ─────────────────────────────────────────────
    def plot_ablation(self, result) -> str:
        labels = ["Full QWA"] + list(result.ablations.keys())
        values = [result.full_ndcg] + list(result.ablations.values())
        colors = [GS_GOLD if i == 0 else GS_NAVY for i in range(len(labels))]
        drop   = [0.0] + [result.full_ndcg - v for v in result.ablations.values()]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       edgecolor=GS_WHITE, linewidth=0.6, height=0.55)
        ax.axvline(result.full_ndcg, color=GS_GOLD, linewidth=1.5,
                   linestyle="--", label=f"Full QWA NDCG={result.full_ndcg:.3f}")
        ax.set_xlabel("NDCG@k")
        _title(ax, "EXP-1: Heuristic Ablation Study", "NDCG@k when each signal is removed")
        ax.legend(fontsize=7.5)
        for bar, val in zip(bars, values[::-1]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color=GS_SLATE)
        ax.yaxis.grid(False); ax.xaxis.grid(True)
        _spine(ax)

        ax = axes[1]
        drop_labels = list(result.ablations.keys())
        drop_vals   = [result.full_ndcg - v for v in result.ablations.values()]
        bar_cols    = [GS_RED if d > 0.02 else GS_GOLD for d in drop_vals]
        ax.bar(drop_labels, drop_vals, color=bar_cols, edgecolor=GS_NAVY, linewidth=0.6)
        ax.set_xticklabels(drop_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("NDCG Drop (Full − Ablated)")
        _title(ax, "NDCG Drop by Removed Signal", "Larger = signal is more important")
        for i, (lbl, d) in enumerate(zip(drop_labels, drop_vals)):
            ax.text(i, d + 0.001, f"−{d:.3f}", ha="center", fontsize=8, color=GS_SLATE)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _spine(ax)

        _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp1_ablation.png")

    # ── EXP-2: p-Sensitivity line chart ───────────────────────────────────────
    def plot_p_sensitivity(self, result) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(result.p_values, result.kendall_tau_vs_base, "o-",
                color=GS_NAVY, linewidth=2.2, markersize=8,
                markerfacecolor=GS_GOLD, markeredgecolor=GS_NAVY)
        ax.axhline(1.0, color=GS_MID_GRAY, linewidth=0.8, linestyle=":")
        ax.fill_between(result.p_values, result.kendall_tau_vs_base,
                        min(result.kendall_tau_vs_base), alpha=0.12, color=GS_NAVY)
        ax.set_xlabel("Teleportation Probability p")
        ax.set_ylabel("Kendall τ  (vs p = 0.15 ranking)")
        _title(ax, "EXP-2: p-Sensitivity Analysis",
               "Ranking stability of QWA as teleportation probability varies")
        _stat(ax, f"Stable near p≈0.15\nτ range: "
              f"[{min(result.kendall_tau_vs_base):.3f}, {max(result.kendall_tau_vs_base):.3f}]")
        _spine(ax); _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp2_p_sensitivity.png")

    # ── EXP-3: Signal correlation heatmap ────────────────────────────────────
    def plot_signal_correlation(self, result) -> str:
        fig, ax = plt.subplots(figsize=(7, 6))
        mat = result.corr_matrix
        n   = len(result.signal_names)
        im  = ax.imshow(mat, cmap=GS_CMAP, vmin=-1, vmax=1, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Spearman ρ", color=GS_SLATE, fontsize=9)
        cbar.outline.set_edgecolor(GS_LIGHT)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(result.signal_names, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(result.signal_names, fontsize=8.5)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                col = GS_WHITE if abs(val) > 0.5 else GS_SLATE
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=col, fontweight="bold")
        _title(ax, "EXP-3: Quality Signal Correlation Matrix",
               "Pairwise Spearman ρ — complementary signals improve coverage")
        ax.grid(False)
        _spine(ax); _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp3_signal_correlation.png")

    # ── EXP-4: Domain Diversity ───────────────────────────────────────────────
    def plot_domain_diversity(self, result) -> str:
        names  = [_short(n) for n in result.heuristic_names]
        ginis  = result.gini_scores
        unique = result.unique_domains
        x = np.arange(len(names))
        w = 0.38

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        bars = ax.bar(x, ginis, width=w*2, color=GS_SERIES[:len(names)],
                      edgecolor=GS_NAVY, linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Gini Coefficient  (higher = more concentrated)")
        _title(ax, "EXP-4: Domain Gini in Top-k", "Lower is better — more domain diversity")
        for bar, g in zip(bars, ginis):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{g:.3f}", ha="center", fontsize=8.5, color=GS_SLATE)
        ax.yaxis.grid(True); ax.xaxis.grid(False); _spine(ax)

        ax = axes[1]
        bars2 = ax.bar(x, unique, width=w*2, color=GS_SERIES[:len(names)],
                       edgecolor=GS_NAVY, linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Unique Domains in Top-k")
        _title(ax, "Unique Domains in Top-k", "Higher is better — broader source coverage")
        for bar, u in zip(bars2, unique):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    str(u), ha="center", fontsize=8.5, color=GS_SLATE, fontweight="bold")
        ax.yaxis.grid(True); ax.xaxis.grid(False); _spine(ax)

        _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp4_domain_diversity.png")

    # ── EXP-5: Crawl Quality Curve ───────────────────────────────────────────
    def plot_quality_curve(self, result) -> str:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, name in enumerate(result.heuristic_names):
            curve = result.quality_at_step[name]
            steps = list(range(1, len(curve)+1))
            lw = 2.8 if "QWA" in name or "H4" in name else 1.6
            ls = "-" if "QWA" in name or "H4" in name else "--"
            ax.plot(steps, curve, color=GS_SERIES[i % len(GS_SERIES)],
                    label=_short(name), linewidth=lw, linestyle=ls)

        ax.set_xlabel("Crawl Step (URLs visited in priority order)")
        ax.set_ylabel("Fraction of Quality Pages Recovered")
        _title(ax, "EXP-5: Simulated Crawl Quality Curve",
               "Steeper = finds high-quality pages earlier")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        _spine(ax); _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp5_quality_curve.png")

    # ── EXP-6: Robots Compliance ─────────────────────────────────────────────
    def plot_robots_compliance(self, result) -> str:
        k_vals = result.k_values
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(k_vals))
        w = 0.15
        for i, name in enumerate(result.heuristic_names):
            comp = result.compliance_at_k[name]
            offset = (i - len(result.heuristic_names)/2) * w + w/2
            bars = ax.bar(x + offset, comp, width=w,
                          color=GS_SERIES[i % len(GS_SERIES)],
                          edgecolor=GS_WHITE, linewidth=0.5, label=_short(name))
        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-{k}" for k in k_vals])
        ax.set_ylabel("Robots-Compliant Fraction")
        ax.set_ylim(0, 1.15)
        _title(ax, "EXP-6: Robots.txt Compliance Rate",
               "Fraction of top-k URLs estimated to permit AI crawling")
        ax.legend(ncol=2)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        _spine(ax); _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp6_robots_compliance.png")

    # ── EXP-7: URL Structural scatter ────────────────────────────────────────
    def plot_url_structural(self, data: dict) -> str:
        pr  = data["pr"];   tld = data["tld"]
        rep = data["rep"];  qual = data["quality"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        ax = axes[0]
        sc = ax.scatter(pr, tld, c=qual, cmap=GS_CMAP, alpha=0.75,
                        s=55, edgecolors=GS_NAVY, linewidths=0.4,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Quality Label (1=high)").outline.set_edgecolor(GS_LIGHT)
        ax.set_xlabel("PageRank Score")
        ax.set_ylabel("TLD Quality Score")
        _title(ax, "EXP-7: PageRank vs TLD Quality",
               "Gold = high-quality pages — QWA targets upper-right cluster")
        _spine(ax)

        ax = axes[1]
        sc2 = ax.scatter(pr, rep, c=qual, cmap=GS_CMAP, alpha=0.75,
                         s=55, edgecolors=GS_NAVY, linewidths=0.4,
                         vmin=0, vmax=1)
        plt.colorbar(sc2, ax=ax, label="Quality Label (1=high)").outline.set_edgecolor(GS_LIGHT)
        ax.set_xlabel("PageRank Score")
        ax.set_ylabel("Domain Reputation Score")
        _title(ax, "PageRank vs Domain Reputation",
               "Pure PR misses high-rep low-PR pages (left side)")
        _spine(ax)

        _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp7_structural.png")

    # ── EXP-8: k-Stability ───────────────────────────────────────────────────
    def plot_k_stability(self, result) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(result.k_values, result.kendall_tau, "o-",
                color=GS_NAVY, linewidth=2.2, markersize=8,
                markerfacecolor=GS_GOLD, markeredgecolor=GS_NAVY)
        ax.fill_between(result.k_values, result.kendall_tau,
                        min(result.kendall_tau), alpha=0.10, color=GS_NAVY)
        ax.axhline(1.0, color=GS_MID_GRAY, linewidth=0.8, linestyle=":")
        ax.set_xlabel("k  (list cutoff)")
        ax.set_ylabel("Kendall τ  (vs full ranking)")
        _title(ax, "EXP-8: k-Stability of QWA Ranking",
               "How stable is the top-k list as k grows?")
        _stat(ax, f"Min τ = {min(result.kendall_tau):.3f}\nMax τ = {max(result.kendall_tau):.3f}")
        _spine(ax); _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp8_k_stability.png")

    # ── EXP-9: Topology ──────────────────────────────────────────────────────
    def plot_topology(self, result) -> str:
        bc  = np.array(result.betweenness_centrality)
        qwa = np.array(result.qwa_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(bc, qwa, c=qwa, cmap=GS_CMAP, alpha=0.65,
                        s=45, edgecolors=GS_NAVY, linewidths=0.4)
        plt.colorbar(sc, ax=ax, label="QWA Score").outline.set_edgecolor(GS_LIGHT)
        ax.set_xlabel("Approximate Betweenness Centrality")
        ax.set_ylabel("QWA Crawl Priority Score")
        _title(ax, "EXP-9: Crawl Priority vs Graph Betweenness",
               f"Spearman ρ = {result.spearman_rho:.3f}")
        _stat(ax, f"Spearman ρ = {result.spearman_rho:.3f}")
        _spine(ax); _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp9_topology.png")

    # ── EXP-10: Head-to-Head table + radar ───────────────────────────────────
    def plot_head_to_head(self, results: dict) -> str:
        metrics = ["NDCG@k", "Precision@k", "Recall@k", "Mean Quality", "Unique Domains"]
        heuristics = list(results.keys())
        short_names = [_short(n) for n in heuristics]

        # Normalise Unique Domains to [0,1] for radar
        max_ud = max(results[h]["Unique Domains"] for h in heuristics)

        # Build normalised data matrix
        mat = []
        for h in heuristics:
            row = [
                results[h]["NDCG@k"],
                results[h]["Precision@k"],
                results[h]["Recall@k"],
                results[h]["Mean Quality"],
                results[h]["Unique Domains"] / max(max_ud, 1),
            ]
            mat.append(row)
        mat = np.array(mat)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: grouped bar ─────────────────────────────────────────────────
        ax = axes[0]
        bar_metrics = ["NDCG@k", "Precision@k", "Recall@k", "Mean Quality"]
        x = np.arange(len(bar_metrics))
        w = 0.14
        for i, (h, sname) in enumerate(zip(heuristics, short_names)):
            vals = [results[h][m] for m in bar_metrics]
            offset = (i - len(heuristics)/2) * w + w/2
            ax.bar(x + offset, vals, width=w,
                   color=GS_SERIES[i % len(GS_SERIES)],
                   edgecolor=GS_WHITE, linewidth=0.5, label=sname)
        ax.set_xticks(x); ax.set_xticklabels(bar_metrics, rotation=12, ha="right")
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
        _title(ax, "EXP-10: Head-to-Head Quality Metrics",
               "All heuristics compared across retrieval quality metrics")
        ax.legend(fontsize=7.5, ncol=2)
        ax.yaxis.grid(True); ax.xaxis.grid(False); _spine(ax)

        # ── Right: domain diversity vs NDCG scatter ───────────────────────────
        ax = axes[1]
        ndcg_vals  = [results[h]["NDCG@k"]        for h in heuristics]
        div_vals   = [results[h]["Unique Domains"] for h in heuristics]
        for i, (ndcg, div, sname) in enumerate(zip(ndcg_vals, div_vals, short_names)):
            color = GS_GOLD if "QWA" in sname or "★" in sname else GS_SERIES[i]
            ax.scatter(div, ndcg, color=color, s=180,
                       edgecolors=GS_NAVY, linewidths=1.2, zorder=5)
            ax.annotate(sname, (div, ndcg), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=GS_SLATE)
        ax.set_xlabel("Unique Domains in Top-k  (diversity)")
        ax.set_ylabel("NDCG@k  (relevance quality)")
        _title(ax, "Quality vs Diversity Trade-off",
               "Upper-right = best: high quality AND diverse sources")
        _spine(ax)

        _wm(fig); fig.tight_layout()
        return self.save(fig, "crawl_exp10_head_to_head.png")
