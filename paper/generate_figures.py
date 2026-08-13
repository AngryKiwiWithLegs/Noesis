#!/usr/bin/env python3
"""
generate_figures.py

Generates publication-quality figures for the Noesis paper from the
experiment JSON data in noesis_experiment/results/.

Outputs PNG (300 DPI) + PDF to paper/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/Users/mac27ssd/Noesis/experiments/results")
OUT = Path("/Users/mac27ssd/Noesis/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "noesis": "#22d3ee",
    "noesis_light": "#67e8f9",
    "naive": "#f87171",
    "naive_light": "#fca5a5",
    "recent": "#fbbf24",
    "isolated": "#9aa0b4",
    "without": "#6b7280",
    "gemini": "#4285f4",
    "gemma": "#34a853",
    "qwen": "#ea4335",
}


def save(fig, name):
    """Save as both PNG and PDF."""
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}.png + {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: A/B Comparison — Hit Rates by Model
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_ab_comparison():
    """Bar chart: with-memory vs without-memory hit rates for 3 models."""
    import math

    # All three models truncated to n=110 (same profiles, same question order)
    files = [
        ("Gemini\nFlash", RESULTS / "ab_n110_gemini_flash.json"),
        ("Gemma3\n4B", RESULTS / "ab_n110_gemma3_4b.json"),
        ("Qwen2.5\n3B", RESULTS / "ab_n110_qwen2.5_3b.json"),
    ]

    def load_hits(path):
        data = json.loads(Path(path).read_text())
        details = data.get("details", [])
        pairs = []
        for d in details:
            w = d.get("with_mem_hit", d.get("with_memory_hit", False))
            wo = d.get("without_mem_hit", d.get("without_memory_hit", False))
            pairs.append((bool(w), bool(wo)))
        with_hit = sum(1 for w, _ in pairs if w)
        without_hit = sum(1 for _, wo in pairs if wo)
        n = len(pairs)
        b = sum(1 for w, wo in pairs if w and not wo)
        c = sum(1 for w, wo in pairs if not w and wo)
        chi2 = (abs(b - c) - 1) ** 2 / max(1, b + c) if (b + c) > 0 else 0
        return with_hit, without_hit, n, chi2

    models = []
    with_rates = []
    without_rates = []
    ns = []
    p_stars = []

    for label, path in files:
        if not path.exists():
            print(f"  ⚠ {path.name} not found, skipping")
            continue
        wh, woh, n, chi2 = load_hits(path)
        models.append(label)
        with_rates.append(wh / n * 100)
        without_rates.append(woh / n * 100)
        ns.append(n)
        # Significance stars
        if chi2 > 10.83: p_stars.append("***")
        elif chi2 > 6.63: p_stars.append("**")
        elif chi2 > 3.84: p_stars.append("*")
        else: p_stars.append("")

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars1 = ax.bar(x - width/2, without_rates, width, label="Without memory",
                   color=COLORS["without"], alpha=0.8, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, with_rates, width, label="With Noesis",
                   color=COLORS["noesis"], alpha=0.9, edgecolor="white", linewidth=0.5)

    # Value labels — placed well above bars, no overlap
    for bar, rate in zip(bars1, without_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, color=COLORS["without"])
    for bar, rate, star in zip(bars2, with_rates, p_stars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{rate:.0f}%{star}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLORS["noesis"])

    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Memory Injection Improves Answer Accuracy", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={n})" for m, n in zip(models, ns)], fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.2)

    # Footnote below the figure (using fig.text, not ax.text)
    fig.text(0.5, 0.01, "*** p < 0.001  (McNemar's test, df=1)",
             ha="center", fontsize=9, color="gray")

    plt.subplots_adjust(bottom=0.15)
    save(fig, "fig3_ab_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Scale Degradation — THE HERO FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_scale_degradation():
    """Dual-axis: token cost (left) + MRR (right) vs memory scale."""
    # Use the latest scale degradation run
    data = json.loads((RESULTS / "scale_degradation_20260630_142946.json").read_text())

    # Extract data — try different structures
    if "results" in data:
        points = data["results"]
    elif "scales" in data:
        points = data["scales"]
    else:
        points = data if isinstance(data, list) else data.get("data", [])

    scales = []
    noesis_tok, naive_tok, recent_tok = [], [], []
    noesis_mrr, naive_mrr, recent_mrr = [], [], []

    for p in points:
        if not isinstance(p, dict):
            continue
        n = p.get("n", p.get("scale", p.get("num_memories")))
        if n is None:
            continue
        scales.append(n)

        for strategy, tok_list, mrr_list in [
            ("noesis", noesis_tok, noesis_mrr),
            ("naive", naive_tok, naive_mrr),
            ("recent", recent_tok, recent_mrr),
        ]:
            s_data = p.get(strategy, p.get(strategy.title(), p))
            if isinstance(s_data, dict):
                tok_list.append(s_data.get("tokens", s_data.get("avg_tokens", 0)))
                mrr_list.append(s_data.get("mrr", 0))
            else:
                tok_list.append(0)
                mrr_list.append(0)

    # If parsing failed, use known report numbers
    if not scales:
        print("  ⚠ Parsing failed, using report numbers from SCALE_DEGRADATION_REPORT")
        scales = [10, 50, 100, 200, 500]
        noesis_tok = [144, 202, 205, 212, 207]
        naive_tok = [120, 677, 1380, 2827, 7192]
        recent_tok = [57, 73, 73, 75, 75]
        noesis_mrr = [0.90, 0.90, 0.90, 0.90, 0.90]
        naive_mrr = [0.28, 0.02, 0.01, 0.00, 0.002]
        recent_mrr = [0.23, 0.00, 0.00, 0.00, 0.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: Token cost ──
    ax1.plot(scales, noesis_tok, "o-", color=COLORS["noesis"], linewidth=2.5,
             markersize=8, label="Noesis", zorder=3)
    ax1.plot(scales, naive_tok, "s-", color=COLORS["naive"], linewidth=2.5,
             markersize=8, label="Naive-RAG", zorder=2)
    ax1.plot(scales, recent_tok, "^--", color=COLORS["recent"], linewidth=2,
             markersize=7, label="Recent-window", alpha=0.7, zorder=1)

    # Annotate the 35× gap
    if len(scales) >= 5:
        gap_x = scales[-1]
        gap_n = noesis_tok[-1]
        gap_r = naive_tok[-1]
        ax1.annotate(f"{gap_r//gap_n}× more\n tokens",
                     xy=(gap_x, gap_r), xytext=(gap_x * 0.4, gap_r * 0.7),
                     fontsize=12, fontweight="bold", color=COLORS["naive"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["naive"], lw=1.5))
        ax1.annotate(f"~{gap_n} tokens",
                     xy=(gap_x, gap_n), xytext=(gap_x * 0.4, gap_n * 2.5),
                     fontsize=11, fontweight="bold", color=COLORS["noesis"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["noesis"], lw=1.5))

    ax1.set_xlabel("Number of Memories in Store")
    ax1.set_ylabel("Avg Tokens Injected per Query")
    ax1.set_title("(a) Token Cost: Noesis Stays Flat")
    ax1.legend(loc="upper left")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.2)

    # ── Right panel: MRR ──
    ax2.plot(scales, noesis_mrr, "o-", color=COLORS["noesis"], linewidth=2.5,
             markersize=8, label="Noesis", zorder=3)
    ax2.plot(scales, naive_mrr, "s-", color=COLORS["naive"], linewidth=2.5,
             markersize=8, label="Naive-RAG", zorder=2)
    ax2.plot(scales, recent_mrr, "^--", color=COLORS["recent"], linewidth=2,
             markersize=7, label="Recent-window", alpha=0.7, zorder=1)

    ax2.set_xlabel("Number of Memories in Store")
    ax2.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax2.set_title("(b) Retrieval Quality: Noesis Maintains Rank-1")
    ax2.legend(loc="center right")
    ax2.set_xscale("log")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Scale Degradation: Noesis vs Baselines (10–500 Memories)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig4_scale_degradation")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Cross-Tool Consistency
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_cross_tool():
    """Bar chart: Noesis shared vs isolated per-tool memory."""
    # From report: Noesis 20/40 (50%), Isolated 3/40 (7.5%) at n=40
    # Also cross-tool-only: Noesis 10/27 (37%), Isolated 0/27 (0%)

    categories = ["All scenarios\n(n=40)", "Cross-tool only\n(n=27)"]
    noesis_rates = [50.0, 37.0]
    isolated_rates = [7.5, 0.0]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, isolated_rates, width, label="Isolated (per-tool)",
                   color=COLORS["isolated"], alpha=0.8, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, noesis_rates, width, label="Noesis (shared)",
                   color=COLORS["noesis"], alpha=0.9, edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars1, isolated_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{rate}%", ha="center", va="bottom", fontsize=11, color=COLORS["isolated"])
    for bar, rate in zip(bars2, noesis_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{rate}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=COLORS["noesis"])

    ax.set_ylabel("Retrieval Hit Rate (%)")
    ax.set_title("Cross-Tool Memory: Shared vs Isolated", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 65)

    # Annotation moved to the right side so it doesn't overlap bars
    ax.annotate("0%: structurally\nimpossible for\nper-tool systems",
                xy=(1 - width/2, 0), xytext=(1.15, 28),
                fontsize=9, fontstyle="italic", color=COLORS["isolated"],
                ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["isolated"], lw=1.2))

    plt.tight_layout()
    save(fig, "fig6_cross_tool")


# ═══════════════════════════════════════════════════════════════════════════════
# Table data export (for LaTeX tables)
# ═══════════════════════════════════════════════════════════════════════════════

def export_table_data():
    """Export the ablation and supersession data as LaTeX tables."""
    tables_dir = OUT.parent / "tables"
    tables_dir.mkdir(exist_ok=True)

    # ── Table 3: Ablation results (from report) ──
    ablation_latex = r"""\begin{table}[ht]
\centering
\caption{Ablation study: retrieval-layer evaluation (50 queries, English). Removing each component shows its contribution.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Recall@5} & \textbf{MRR} & \textbf{Prec@5} & \textbf{Tentative Leak} \\
\midrule
\textbf{Noesis (full)}  & \textbf{100\%} & \textbf{1.000} & \textbf{96\%} & \textbf{10} \\
Noesis w/o core-fact    & 100\% & 1.000 & 96\% & 10 \\
Noesis w/o recency      & 100\% & 1.000 & 96\% & 10 \\
Noesis w/o gating        & 100\% & 1.000 & 88\% & 29 \\
Noesis w/o semantic     & 49\%  & 0.160 & 100\% & 0 \\
\midrule
Naive-RAG               & 100\% & 1.000 & 88\% & 29 \\
Recent-window           & 22\%  & 0.099 & 100\% & 0 \\
Random                  & 45\%  & 0.206 & 94\% & 15 \\
\bottomrule
\end{tabular}
\end{table}"""

    (tables_dir / "table3_ablation.tex").write_text(ablation_latex)

    # ── Table 5: Supersession fix (from report) ──
    supersession_latex = r"""\begin{table}[ht]
\centering
\caption{Supersession fix: before/after stale-stance leak rate (n=30 scenarios).}
\label{tab:supersession}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Before fix} & \textbf{After fix} & \textbf{Change} \\
\midrule
Old stance leaked into context & 28/30 (93\%) & 2/30 (7\%) & $-86$pp \\
New stance surfaced            & 12/30 (40\%) & 28/30 (93\%) & $+53$pp \\
Ideal outcome (new + suppress old) & 1/30 (3\%) & 26/30 (87\%) & $+84$pp \\
\bottomrule
\end{tabular}
\end{table}"""

    (tables_dir / "table5_supersession.tex").write_text(supersession_latex)

    # ── Table 1: Thought types ──
    types_latex = r"""\begin{table}[ht]
\centering
\caption{Thought taxonomy: seven types with type-specific half-lives and injection priorities.}
\label{tab:thought_types}
\begin{tabular}{llcc}
\toprule
\textbf{Type} & \textbf{Definition} & \textbf{Half-life (days)} & \textbf{Injection priority} \\
\midrule
identity       & Who the user is               & $\infty$ & Highest (always injected) \\
preference     & Explicit preference           & 90       & High \\
position       & Stance on a topic             & 90       & High \\
question       & Open question being explored   & 30       & Medium (triggers knowledge ingestion) \\
synthesis      & Insight from merging ideas    & 180      & Medium \\
event          & Something that happened/decided & 7        & Low (recent priority) \\
contradiction  & Self-contradictory judgment   & $\infty$ & Flagged for resolution \\
\bottomrule
\end{tabular}
\end{table}"""

    (tables_dir / "table1_thought_types.tex").write_text(types_latex)

    print("  ✓ LaTeX tables exported to paper/tables/")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Noesis paper figures...")
    print()
    print("Figure 3: A/B comparison...")
    fig3_ab_comparison()
    print()
    print("Figure 4: Scale degradation (hero)...")
    fig4_scale_degradation()
    print()
    print("Figure 6: Cross-tool consistency...")
    fig6_cross_tool()
    print()
    print("Exporting LaTeX tables...")
    export_table_data()
    print()
    print(f"Done! Figures in: {OUT}")
    print(f"Tables in: {OUT.parent / 'tables'}")
