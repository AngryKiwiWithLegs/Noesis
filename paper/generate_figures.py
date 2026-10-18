#!/usr/bin/env python3
"""
generate_figures.py

Generates publication-quality figures for the Noesis paper from the
experiment JSON data in experiments/results/.

Each figure is drawn at the exact size of its frame in the paper (inches), so
text prints at its true point size. Outputs PNG (300 DPI) + PDF to
paper/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

RESULTS = Path("/Users/mac27ssd/Noesis/experiments/results")
OUT = Path("/Users/mac27ssd/Noesis/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette (validated with the dataviz palette validator, light mode) ───────
# Noesis is always blue; baselines are gray, or orange/aqua when there are two.
COLORS = {
    "noesis": "#2a78d6",     # categorical slot 1
    "naive": "#eb6834",      # categorical slot 2
    "recent": "#1baf7a",     # categorical slot 3 (<3:1 → always direct-labelled)
    "topk": "#eda100",       # categorical slot 4 (<3:1 → always direct-labelled)
    "baseline": "#87857e",   # neutral gray for "without memory" / "isolated"
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "ytick.major.pad": 4,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "pdf.fonttype": 3,   # subset glyphs (mpl 3.4 embeds whole fonts with 42)
    "savefig.bbox": "standard",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}.png + {name}.pdf")


def ygrid(ax):
    """Recessive hairline gridlines behind the data."""
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def mcnemar_chi2(b, c):
    return (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: A/B Comparison — accuracy with vs without memory, per model
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_ab_comparison():
    """Grouped columns: without-memory (gray) vs with-Noesis (blue)."""
    # All three models truncated to n=110 (same profiles, same question order)
    files = [
        ("Gemini Flash-Lite", "cloud", RESULTS / "ab_n110_gemini_flash.json"),
        ("gemma3:4b", "local", RESULTS / "ab_n110_gemma3_4b.json"),
        ("qwen2.5:3b", "local", RESULTS / "ab_n110_qwen2.5_3b.json"),
    ]

    labels, without, with_, sig, ns = [], [], [], [], []
    for name, where, path in files:
        details = json.loads(path.read_text())["details"]
        w = [bool(d["with_mem_hit"]) for d in details]
        wo = [bool(d["without_mem_hit"]) for d in details]
        n = len(details)
        b = sum(x and not y for x, y in zip(w, wo))
        c = sum(y and not x for x, y in zip(w, wo))
        labels.append(f"{name}\n{where}")
        with_.append(100 * sum(w) / n)
        without.append(100 * sum(wo) / n)
        sig.append(mcnemar_chi2(b, c) > 10.83)   # p < 0.001, df = 1
        ns.append(n)

    W, H = 5.25, 3.9089
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.125, 0.15, 0.855, 0.72])

    x = np.arange(len(labels))
    bw, gap = 0.30, 0.02
    ax.bar(x - (bw + gap) / 2, without, bw, color=COLORS["baseline"], zorder=2)
    ax.bar(x + (bw + gap) / 2, with_, bw, color=COLORS["noesis"], zorder=2)

    for i in range(len(x)):
        ax.text(x[i] - (bw + gap) / 2, without[i] + 1.8, f"{without[i]:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.text(x[i] + (bw + gap) / 2, with_[i] + 1.8,
                f"{with_[i]:.1f}%" + ("***" if sig[i] else ""),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for t in ax.get_xticklabels():
        t.set_color(INK)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Accuracy (keyword hit)")
    ygrid(ax)

    fig.legend(handles=[Patch(color=COLORS["baseline"], label="Without memory"),
                        Patch(color=COLORS["noesis"], label="With Noesis")],
               loc="upper left", bbox_to_anchor=(0.115, 0.995), ncol=2,
               handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    fig.text(0.98, 0.012,
             f"n = {ns[0]} paired questions per model   ·   *** p < 0.001, McNemar's test",
             ha="right", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "fig3_ab_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Scale degradation — token cost and ranking vs memory size
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_scale_degradation():
    """Two panels: (a) tokens injected per query, (b) MRR, vs memories stored."""
    # 2026-10-09 re-run: reproduces the June run exactly and adds a top-5 RAG baseline
    data = json.loads((RESULTS / "scale_degradation_20261009_190657.json").read_text())
    curve = sorted(data["curve"], key=lambda p: p["n_memories"])
    scales = [p["n_memories"] for p in curve]
    series = [  # (json key, label, color, marker)
        ("Noesis", "Noesis", COLORS["noesis"], "o"),
        ("naive-RAG", "Naive RAG (inject all)", COLORS["naive"], "s"),
        ("recent-window", "Recent window", COLORS["recent"], "^"),
        ("topk-RAG", "Top-5 RAG", COLORS["topk"], "D"),
    ]
    tok = {k: [p[k]["avg_tokens_per_query"] for p in curve] for k, *_ in series}
    mrr = {k: [p[k]["mrr"] for p in curve] for k, *_ in series}

    W, H = 5.8333, 2.3270
    fig = plt.figure(figsize=(W, H))
    ax1 = fig.add_axes([0.085, 0.19, 0.355, 0.60])
    ax2 = fig.add_axes([0.585, 0.19, 0.355, 0.60])

    def style_x(ax):
        ax.set_xscale("log")
        ax.set_xlim(7, 1400)
        ax.xaxis.set_major_locator(FixedLocator(scales))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Memories in store")
        ygrid(ax)

    for k, label, color, marker in series:
        for ax, ys in ((ax1, tok[k]), (ax2, mrr[k])):
            ax.plot(scales, ys, color=color, linewidth=0.9 if k == "topk-RAG" else 1.6, marker=marker,
                    markersize=2.8 if k == "topk-RAG" else 5.4,
                    markeredgecolor="white", markeredgewidth=0 if k == "topk-RAG" else 0.9,
                    solid_capstyle="round", solid_joinstyle="round",
                    zorder={"topk-RAG": 4, "Noesis": 3}.get(k, 2))

    # (a) tokens — log y, direct end labels, 35x bracket
    style_x(ax1)
    ax1.set_yscale("log")
    ax1.set_ylim(40, 14000)
    ax1.yaxis.set_major_locator(FixedLocator([50, 100, 200, 500, 1000, 2000, 5000, 10000]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.set_title("(a) Tokens injected per query", loc="left", fontsize=9,
                  fontweight="bold", color=INK, pad=6)
    end = scales[-1]
    for k, dy in (("naive-RAG", 1.0), ("Noesis", 1.0), ("recent-window", 1.32), ("topk-RAG", 1 / 1.32)):
        ax1.text(end * 1.18, tok[k][-1] * dy, f"{tok[k][-1]:,}", ha="left", va="center",
                 fontsize=8.2, color=INK2)
    ratio = tok["naive-RAG"][-1] / tok["Noesis"][-1]
    bx = end * 3.4
    ax1.annotate("", xy=(bx, tok["naive-RAG"][-1]), xytext=(bx, tok["Noesis"][-1]),
                 arrowprops=dict(arrowstyle="<|-|>", color=INK2, lw=0.8,
                                 mutation_scale=6, shrinkA=0, shrinkB=0))
    ax1.text(bx * 1.12, np.sqrt(tok["naive-RAG"][-1] * tok["Noesis"][-1]),
             f"{ratio:.0f}×", ha="left", va="center", fontsize=9,
             fontweight="bold", color=INK)
    ax1.set_xlim(7, 4600)

    # (b) MRR — linear, label Noesis plateau and naive-RAG floor
    style_x(ax2)
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}" if v else "0"))
    ax2.set_title("(b) Rank of the target memory (MRR)", loc="left", fontsize=9,
                  fontweight="bold", color=INK, pad=6)
    ax2.text(end * 1.18, mrr["Noesis"][-1], f"{mrr['Noesis'][-1]:.2f}",
             ha="left", va="center", fontsize=8.2, color=INK2)
    ax2.text(scales[0], mrr["Noesis"][0] - 0.08, "Noesis and top-5 RAG overlap",
             ha="left", va="top", fontsize=7.6, color=MUTED)
    ax2.text(end * 1.18, 0.0, f"{mrr['naive-RAG'][-1]:.3f}".rstrip("0"),
             ha="left", va="center", fontsize=8.2, color=INK2)

    handles = [Line2D([], [], color=c, marker=m, linewidth=1.6, markersize=5,
                      markeredgecolor="white", markeredgewidth=0.9, label=l)
               for _, l, c, m in series]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, handlelength=2.0, columnspacing=1.4)
    save(fig, "fig4_scale_degradation")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Cross-Tool Consistency (Figure 5 in the paper)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_cross_tool():
    """Grouped columns: isolated per-tool memory (gray) vs Noesis shared (blue)."""
    det = json.loads((RESULTS / "cross_tool_20260630_205622.json").read_text())["details"]
    # "Strictly cross-tool": the stance was stated in exactly one tool, and a
    # different tool asks about it.
    strict = [d for d in det if len(d["source_tools"]) == 1
              and d["query_tool"] not in d["source_tools"]]
    groups = [("All scenarios", det), ("Strictly cross-tool", strict)]

    W, H = 4.9583, 3.5071
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.11, 0.15, 0.86, 0.71])

    x = np.arange(len(groups))
    bw, gap = 0.30, 0.02
    for i, (name, rows) in enumerate(groups):
        n = len(rows)
        iso = sum(r["isolated_hit"] for r in rows)
        noe = sum(r["noesis_hit"] for r in rows)
        xi, xn = x[i] - (bw + gap) / 2, x[i] + (bw + gap) / 2
        ax.bar(xi, 100 * iso / n, bw, color=COLORS["baseline"], zorder=2)
        ax.bar(xn, 100 * noe / n, bw, color=COLORS["noesis"], zorder=2)
        if iso == 0:
            # a zero-height column is invisible: mark its slot on the baseline
            ax.plot([xi - bw / 2, xi + bw / 2], [0, 0], color=COLORS["baseline"],
                    linewidth=2.2, solid_capstyle="butt", zorder=3, clip_on=False)
            ax.text(xi, 1.6, f"0%\n(0/{n})", ha="center", va="bottom",
                    fontsize=8.5, color=INK2, linespacing=1.15)
            ax.text(xi, 11.5, "no access to\nother tools'\nmemories", ha="center",
                    va="bottom", fontsize=7.6, color=MUTED, linespacing=1.15)
        else:
            ax.text(xi, 100 * iso / n + 1.6, f"{100 * iso / n:.1f}%\n({iso}/{n})",
                    ha="center", va="bottom", fontsize=8.5, color=INK2, linespacing=1.15)
        ax.text(xn, 100 * noe / n + 1.6, f"{100 * noe / n:.1f}%\n({noe}/{n})",
                ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold",
                linespacing=1.15)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(n = {len(rows)})" for name, rows in groups])
    for t in ax.get_xticklabels():
        t.set_color(INK)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Retrieval hit rate")
    ygrid(ax)

    fig.legend(handles=[Patch(color=COLORS["baseline"], label="Isolated (per-tool memory)"),
                        Patch(color=COLORS["noesis"], label="Noesis (shared memory)")],
               loc="upper left", bbox_to_anchor=(0.10, 0.995), ncol=2,
               handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    save(fig, "fig6_cross_tool")


# ═══════════════════════════════════════════════════════════════════════════════
# Table data export (for LaTeX tables)
# ═══════════════════════════════════════════════════════════════════════════════

def export_table_data():
    """Export the ablation and supersession data as LaTeX tables."""
    tables_dir = OUT.parent / "tables"
    tables_dir.mkdir(exist_ok=True)

    # ── Table 3: Ablation results (exact-identity scoring, October 2026) ──
    abl = {r["config"]: r for r in json.load(
        open(RESULTS / "ablation_baseline_en_20261017_102429.json"))["results"]}
    rows = [("Noesis-full", r"\textbf{Noesis (full)}"), ("Noesis-no-core", "w/o core-fact signal"),
            ("Noesis-no-recency", "w/o recency signal"), ("Noesis-no-gating", "w/o confidence gating"),
            ("Noesis-no-semantic", "w/o semantic retrieval"), None, ("naive-RAG", "Top-5 RAG"),
            ("recent-window", "Recent-window"), ("random", "Random")]
    body = []
    for row in rows:
        if row is None:
            body.append(r"\midrule")
            continue
        r = abl[row[0]]
        body.append(f"{row[1]} & {100 * r['recall@5']:.0f}\\% & {r['mrr']:.3f} & "
                    f"{100 * r['context_recall']:.0f}\\% & {r['context_items']:.1f} & "
                    f"{r['tentative_leak_total']}/{r['context_slots_total']} \\\\")
    ablation_latex = r"""\begin{table}[ht]
\centering
\caption{Ablation and baselines (50 queries over 28 memories, 5 tentative; retrieval-layer, 2{,}000-token budget). A retrieved memory counts only if it is the ground-truth memory. Recall@5 and MRR use the first five memories in injection order (or all, if fewer are returned); context recall, items, and leak cover everything injected, and leak counts tentative memories among those slots.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Config} & \textbf{Recall@5} & \textbf{MRR} & \textbf{Context recall} & \textbf{Items} & \textbf{Leak} \\
\midrule
""" + "\n".join(body) + r"""
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
Old stance leaked into context & 28/30 (93\%) & 2/30 (7\%) & $-87$pp \\
New stance surfaced            & 12/30 (40\%) & 28/30 (93\%) & $+53$pp \\
Ideal outcome (new + suppress old) & 1/30 (3\%) & 27/30 (90\%) & $+87$pp \\
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
