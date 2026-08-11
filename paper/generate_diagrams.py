#!/usr/bin/env python3
"""
generate_diagrams.py

Creates architecture and pipeline diagrams for the Noesis paper using
matplotlib patches (boxes, arrows, labels). Clean, publication-quality.
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


def draw_box(ax, x, y, w, h, text, color="#22d3ee", text_color="white", fontsize=10, alpha=0.9):
    """Draw a rounded box with centered text."""
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transData
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            transform=ax.transData)


def draw_arrow(ax, x1, y1, x2, y2, color="#666", style="->", lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}.png + {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: System Architecture
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ── Input layer (top) ──
    ax.text(6, 6.7, "AI Tools (User's daily conversations)", ha="center",
            fontsize=13, fontweight="bold", color="#333")

    draw_box(ax, 0.5, 5.5, 3, 0.8, "Claude Desktop\n(MCP Server)", color="#d97706")
    draw_box(ax, 4.5, 5.5, 3, 0.8, "ChatGPT / Any API\n(Proxy :8080)", color="#2563eb")
    draw_box(ax, 8.5, 5.5, 3, 0.8, "Browser Extension\n(WebSocket :8082)", color="#7c3aed")

    # Arrows from tools to Noesis
    for x in [2, 6, 10]:
        draw_arrow(ax, x, 5.5, 6, 4.7, color="#888")

    # ── Noesis core (middle) ──
    draw_box(ax, 2, 3.5, 8, 1.0, "Noesis Daemon", color="#0f172a", fontsize=14)

    # Two-phase pipeline inside
    draw_box(ax, 2.3, 2.3, 3.5, 0.9,
             "Phase 1 (<15ms)\nEmbed → Insert → Tentative",
             color="#0891b2", fontsize=9)
    draw_box(ax, 6.2, 2.3, 3.5, 0.9,
             "Phase 2 (async)\nExtract → Score → Store",
             color="#0e7490", fontsize=9)

    draw_arrow(ax, 4, 3.5, 4, 3.2, color="#666")
    draw_arrow(ax, 8, 3.5, 8, 3.2, color="#666")

    # ── Confidence scorer (side) ──
    draw_box(ax, 0.2, 1.0, 2.5, 0.9,
             "Confidence Scorer\n4-signal + decay",
             color="#059669", fontsize=9)
    draw_arrow(ax, 2.7, 1.4, 4.0, 2.3, color="#059669", style="<->")

    # ── Storage layer (bottom) ──
    draw_box(ax, 3.0, 0.2, 2.8, 0.8,
             "Hot Store\n(sqlite-vec)\n<1ms retrieval",
             color="#dc2626", fontsize=9)
    draw_box(ax, 6.2, 0.2, 2.8, 0.8,
             "Cold Store\n(Obsidian vault)\nHuman-readable",
             color="#7c2d12", fontsize=9)

    draw_arrow(ax, 4.0, 2.3, 4.4, 1.0, color="#dc2626")
    draw_arrow(ax, 8.0, 2.3, 7.6, 1.0, color="#7c2d12")

    # Bidirectional sync arrow
    draw_arrow(ax, 5.8, 0.6, 6.2, 0.6, color="#666", style="<->")
    ax.text(6.0, 0.9, "sync", ha="center", fontsize=8, color="#666")

    ax.set_title("Noesis System Architecture", fontsize=15, fontweight="bold", pad=15)
    save(fig, "fig1_architecture")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Retrieval Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Query input
    draw_box(ax, 0.2, 1.5, 1.8, 1.0, "User\nQuery", color="#2563eb", fontsize=11)

    # Three parallel retrieval channels
    draw_box(ax, 2.5, 2.8, 2.2, 0.8, "BM25\n(keyword match)", color="#0891b2", fontsize=9)
    draw_box(ax, 2.5, 1.5, 2.2, 0.8, "Vector\n(semantic sim.)", color="#0e7490", fontsize=9)
    draw_box(ax, 2.5, 0.2, 2.2, 0.8, "Signals\n(recency + core)", color="#059669", fontsize=9)

    # Arrows from query to channels
    for y in [3.2, 1.9, 0.6]:
        draw_arrow(ax, 2.0, 2.0, 2.5, y, color="#666")

    # RRF Fusion
    draw_box(ax, 5.3, 1.5, 2.0, 1.0, "RRR Fusion\n+ Time Decay\nWeighting",
             color="#7c3aed", fontsize=9)

    for y in [3.2, 1.9, 0.6]:
        draw_arrow(ax, 4.7, y, 5.3, 2.0, color="#666")

    # Confidence gate
    draw_box(ax, 7.8, 1.5, 2.0, 1.0, "Confidence\nGate\n(≥ provisional)",
             color="#dc2626", fontsize=9)
    draw_arrow(ax, 7.3, 2.0, 7.8, 2.0, color="#666")

    # Token budget
    draw_box(ax, 10.2, 1.5, 1.6, 1.0, "Token\nBudget\n(1200 tok)",
             color="#d97706", fontsize=9)
    draw_arrow(ax, 9.8, 2.0, 10.2, 2.0, color="#666")

    # Output
    ax.text(11.0, 0.8, "System\nPrompt\nInjection", ha="center",
            fontsize=10, fontweight="bold", color="#333")
    draw_arrow(ax, 11.0, 1.5, 11.0, 1.1, color="#333")

    ax.set_title("Retrieval and Injection Pipeline", fontsize=14, fontweight="bold", pad=10)
    save(fig, "fig2_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating architecture diagrams...")
    print()
    print("Figure 1: System architecture...")
    fig1_architecture()
    print()
    print("Figure 2: Retrieval pipeline...")
    fig2_pipeline()
    print()
    print(f"Done! Figures in: {OUT}")
