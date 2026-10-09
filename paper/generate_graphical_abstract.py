#!/usr/bin/env python3
"""
generate_graphical_abstract.py — original graphical abstract for preprint
servers (not a copy of any figure in the paper). Every number shown is
printed in the paper and checked by experiments/verify_paper.py.

Output: paper/figures/graphical_abstract.png (2200 x 1120 px).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/Users/mac27ssd/Noesis/paper/figures")
INK, INK2, MUTED, HAIR = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, BLUE_TINT, NEUTRAL, GRAY = "#2a78d6", "#eaf2fc", "#f4f3ef", "#87857e"

plt.rcParams.update({"font.family": "Arial", "savefig.bbox": "standard"})

W, H = 7.3333, 3.7333                       # inches; 300 dpi -> 2200 x 1120 px
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x0, y0, x1, y1, fill, edge, lw=0.9, r=0.07):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(p0, p1, color=INK2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color,
                                 linewidth=1.1, shrinkA=0, shrinkB=0, zorder=3))


# Title
ax.text(0.22, H - 0.30, "Noesis: memory that earns its way into the prompt, shared across AI tools",
        ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)

# ── Left: how it works ────────────────────────────────────────────────────────
tools = ["Claude Desktop", "Any API tool", "Browser extension"]
ty = [2.62, 2.12, 1.62]
for name, y in zip(tools, ty):
    box(0.22, y - 0.19, 1.40, y + 0.19, NEUTRAL, HAIR)
    ax.text(0.81, y, name, ha="center", va="center", fontsize=8.4, color=INK, zorder=4)
    arrow((1.40 + 0.04, y), (1.66 - 0.04, 2.12 + (y - 2.12) * 0.35))

box(1.66, 1.42, 2.92, 2.82, BLUE_TINT, BLUE)
ax.text(2.29, 2.58, "Shared local store", ha="center", va="center", fontsize=9.0, fontweight="bold", color=INK, zorder=4)
for i, (label, col) in enumerate([("tentative", GRAY), ("provisional", BLUE), ("settled", BLUE)]):
    y = 2.24 - i * 0.30
    ax.add_patch(plt.Circle((1.88, y), 0.055, color=col, zorder=4))
    ax.text(2.00, y, label, ha="left", va="center", fontsize=8.2, color=INK2, zorder=4)

arrow((2.92 + 0.04, 2.12), (3.14 - 0.04, 2.12))
box(3.14, 1.84, 3.92, 2.40, BLUE, BLUE)
ax.text(3.53, 2.21, "confidence", ha="center", va="center", fontsize=8.4, fontweight="bold", color="white", zorder=4)
ax.text(3.53, 2.03, "gate", ha="center", va="center", fontsize=8.4, fontweight="bold", color="white", zorder=4)

arrow((3.92 + 0.04, 2.12), (4.14 - 0.04, 2.12))
box(4.14, 1.84, 4.86, 2.40, NEUTRAL, HAIR)
ax.text(4.50, 2.21, "System", ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK, zorder=4)
ax.text(4.50, 2.03, "prompt", ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK, zorder=4)
ax.text(4.50, 1.70, "≈207 tokens at\n500 memories", ha="center", va="top", fontsize=7.0, color=MUTED, linespacing=1.2)

ax.text(0.22, 1.12, "A thought passes the gate once it is firmly stated, repeated, or confirmed\n"
        "in another tool; tentative thoughts are stored but never injected.",
        ha="left", va="top", fontsize=7.8, color=INK2, linespacing=1.35)

# ── Right: three results ─────────────────────────────────────────────────────
tiles = [
    ("Answer accuracy with memory", "+25.5 to +48.2 pp",
     "3 models × 110 questions, all p < 0.001\ne.g. gemma3:4b: 34.5% → 82.7%"),
    ("Unverified statements injected", "0 vs. 19",
     "Noesis vs. ungated retrieval (ablation);\nplanted statements: 0 vs. 13"),
    ("Recall on strictly cross-tool queries", "37% vs. 0%",
     "shared store vs. per-tool memory"),
]
x0, x1 = 5.06, W - 0.22
for i, (label, value, note) in enumerate(tiles):
    top = 2.98 - i * 0.98
    box(x0, top - 0.86, x1, top, "white", HAIR, lw=0.8)
    ax.text(x0 + 0.13, top - 0.15, label, ha="left", va="center", fontsize=8.0, color=INK2, zorder=4)
    ax.text(x0 + 0.13, top - 0.42, value, ha="left", va="center", fontsize=14.5, fontweight="bold", color=INK, zorder=4)
    ax.text(x0 + 0.13, top - 0.70, note, ha="left", va="center", fontsize=6.9, color=MUTED,
            linespacing=1.25, zorder=4)

fig.savefig(OUT / "graphical_abstract.png", dpi=300, facecolor="white")
print("  ✓ graphical_abstract.png")
