#!/usr/bin/env python3
"""
generate_diagrams.py — Publication-quality architecture diagrams for Noesis paper.
Redesigned with generous spacing to prevent overlapping text and arrows.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("/Users/mac27ssd/Noesis/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}.png + {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: System Architecture — clean vertical flow with generous spacing
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=10, text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2",
            facecolor=color, edgecolor="none", alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                    connectionstyle="arc3,rad=0"))

    # ── Title ──
    ax.text(5, 8.7, "Noesis System Architecture", ha="center",
            fontsize=15, fontweight="bold")

    # ── Layer 1: Input tools (y=7.2-8.0) ──
    ax.text(5, 8.25, "AI Tools", ha="center", fontsize=12, color="#333",
            fontweight="bold")
    box(0.5, 7.2, 2.5, 0.7, "Claude Desktop\n(MCP Server)", "#d97706", fontsize=9)
    box(3.75, 7.2, 2.5, 0.7, "Any API Tool\n(Proxy :8080)", "#2563eb", fontsize=9)
    box(7.0, 7.2, 2.5, 0.7, "Browser Extension\n(WebSocket :8082)", "#7c3aed", fontsize=9)

    # Arrows down to daemon (all converge to center)
    for x in [1.75, 5.0, 8.25]:
        arrow(x, 7.2, 5.0, 6.2)

    # ── Layer 2: Noesis Daemon (y=5.4-6.2) ──
    box(1.5, 5.4, 7.0, 0.8, "Noesis Daemon", "#0f172a", fontsize=14)

    # Arrows from daemon to two phases
    arrow(3.0, 5.4, 2.8, 4.7)
    arrow(7.0, 5.4, 7.2, 4.7)

    # ── Layer 3: Two-phase pipeline (y=3.8-4.6) ──
    box(0.8, 3.8, 4.0, 0.8, "Phase 1 (<15ms)\nEmbed → Insert → Tentative",
        "#0891b2", fontsize=9)
    box(5.2, 3.8, 4.0, 0.8, "Phase 2 (async)\nExtract → Score → Store",
        "#0e7490", fontsize=9)

    # Confidence scorer on the left side
    arrow(0.8, 4.2, 0.5, 3.0)  # to scorer
    box(0.2, 2.2, 2.2, 0.7, "Confidence\n4-signal + decay", "#059669", fontsize=9)
    arrow(1.3, 2.9, 1.3, 3.8, "#059669")  # back up to phase 1

    # ── Layer 4: Storage (y=1.0-1.8) ──
    # Arrows from phases to storage
    arrow(2.8, 3.8, 3.2, 1.8, "#dc2626")
    arrow(7.2, 3.8, 6.8, 1.8, "#7c2d12")

    box(2.0, 1.0, 2.8, 0.7, "Hot Store\n(sqlite-vec)\n<1ms retrieval",
        "#dc2626", fontsize=9)
    box(5.2, 1.0, 2.8, 0.7, "Cold Store\n(Obsidian vault)\nHuman-readable",
        "#7c2d12", fontsize=9)

    # Bidirectional sync between stores
    ax.annotate("", xy=(5.2, 1.35), xytext=(4.8, 1.35),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.5))
    ax.text(5.0, 1.55, "sync", ha="center", fontsize=8, color="#666")

    # ── Labels on the left ──
    ax.text(0.1, 7.55, "Input", fontsize=9, color="#999", rotation=90,
            ha="center", va="center")
    ax.text(0.1, 5.8, "Core", fontsize=9, color="#999", rotation=90,
            ha="center", va="center")
    ax.text(0.1, 4.2, "Pipeline", fontsize=9, color="#999", rotation=90,
            ha="center", va="center")
    ax.text(0.1, 1.35, "Storage", fontsize=9, color="#999", rotation=90,
            ha="center", va="center")

    save(fig, "fig1_architecture")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Retrieval Pipeline — horizontal flow with clear spacing
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9, text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2",
            facecolor=color, edgecolor="none", alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

    ax.text(7, 4.2, "Retrieval and Injection Pipeline", ha="center",
            fontsize=14, fontweight="bold")

    # Query input (left)
    box(0.3, 1.6, 1.6, 1.0, "User\nQuery", "#2563eb", fontsize=11)

    # Three parallel retrieval channels (middle-left)
    box(2.6, 3.0, 2.0, 0.7, "BM25\n(keyword)", "#0891b2", fontsize=9)
    box(2.6, 1.8, 2.0, 0.7, "Vector\n(semantic)", "#0e7490", fontsize=9)
    box(2.6, 0.6, 2.0, 0.7, "Signals\n(recency + core)", "#059669", fontsize=9)

    # Arrows from query to channels (spread out)
    arrow(1.9, 2.3, 2.6, 3.35)
    arrow(1.9, 2.1, 2.6, 2.15)
    arrow(1.9, 1.9, 2.6, 0.95)

    # RRF Fusion (middle-right)
    box(5.3, 1.6, 1.8, 1.0, "RRF Fusion\n+ Time Decay", "#7c3aed", fontsize=9)

    # Arrows from channels to RRF (converge)
    arrow(4.6, 3.35, 5.3, 2.3)
    arrow(4.6, 2.15, 5.3, 2.1)
    arrow(4.6, 0.95, 5.3, 1.9)

    # Confidence gate
    box(7.7, 1.6, 1.6, 1.0, "Confidence\nGate\n(≥provisional)", "#dc2626", fontsize=9)
    arrow(7.1, 2.1, 7.7, 2.1)

    # Token budget
    box(9.9, 1.6, 1.4, 1.0, "Token\nBudget\n(1200)", "#d97706", fontsize=9)
    arrow(9.3, 2.1, 9.9, 2.1)

    # Output
    box(12.0, 1.6, 1.6, 1.0, "System\nPrompt\nInjection", "#0f172a", fontsize=9)
    arrow(11.3, 2.1, 12.0, 2.1)

    save(fig, "fig2_pipeline")


if __name__ == "__main__":
    print("Generating diagrams...")
    print("\nFigure 1: Architecture...")
    fig1_architecture()
    print("\nFigure 2: Pipeline...")
    fig2_pipeline()
    print(f"\nDone! Figures in: {OUT}")
