#!/usr/bin/env python3
"""
generate_diagrams.py — Architecture and pipeline diagrams for the Noesis paper.

Each figure is drawn at the exact size of its frame in the paper (inches), so
text prints at its true point size. Arrows run edge-to-edge between boxes and
never pass through another box.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path("/Users/mac27ssd/Noesis/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette (validated: see dataviz reference palette) ──────────────────────
INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text, connectors
MUTED = "#898781"        # annotations
HAIR = "#c3c2b7"         # hairline borders
BLUE = "#2a78d6"         # Noesis accent
BLUE_TINT = "#eaf2fc"    # Noesis internal components
NEUTRAL = "#f4f3ef"      # external endpoints / storage
WASH = "#f8f9fb"         # local-system container

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.6,
    "pdf.fonttype": 3,   # subset glyphs (mpl 3.4 embeds whole fonts with 42)
    "savefig.bbox": "standard",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}.png + {name}.pdf")


def canvas(w, h):
    """A figure whose data coordinates are inches, origin bottom-left."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, x1, y1, title, lines=(), style="tint", title_size=8.6,
        line_size=7.6):
    fill, edge, tcol, lcol = {
        "tint":    (BLUE_TINT, BLUE, INK, INK2),
        "solid":   (BLUE, BLUE, "white", "white"),
        "neutral": (NEUTRAL, HAIR, INK, INK2),
    }[style]
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=2))
    # stack title + lines around the vertical centre
    rows = ([(t, title_size, "bold", tcol) for t in title.split("\n")]
            + [(l, line_size, "normal", lcol) for l in lines])
    heights = [s * 1.32 / 72 for _, s, _, _ in rows]
    y = (y0 + y1) / 2 + sum(heights) / 2
    for (text, size, weight, color), hgt in zip(rows, heights):
        y -= hgt
        ax.text((x0 + x1) / 2, y + hgt * 0.52, text, ha="center", va="center",
                fontsize=size, fontweight=weight, color=color, zorder=3)


def arrow(ax, p0, p1, color=INK2, both=False, lw=0.9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7,
        color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=4))


def path(ax, points, color=INK2, lw=0.9):
    """Orthogonal connector: plain segments, arrowhead on the last one."""
    xs, ys = zip(*points[:-1])
    ax.add_line(Line2D(xs, ys, color=color, linewidth=lw,
                       solid_capstyle="butt", solid_joinstyle="miter", zorder=4))
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: System architecture
# ═══════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    W, H = 5.5417, 4.9701
    fig, ax = canvas(W, H)
    g = 0.025  # breathing room between arrow tips and box edges

    # External tools
    tools = [
        (0.25, 1.80, "Claude Desktop", "MCP server"),
        (2.00, 3.55, "Any API tool", "proxy :8080"),
        (3.75, 5.30, "Browser extension", "WebSocket :8082"),
    ]
    for x0, x1, t, sub in tools:
        box(ax, x0, 4.30, x1, 4.84, t, [sub], style="neutral")

    # Local system container
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.12), W - 0.20, 3.90 - 0.12,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=WASH, edgecolor=HAIR, linewidth=0.7, zorder=1))
    ax.text(W - 0.22, 0.26, "Noesis — runs locally", ha="right", va="center",
            fontsize=7.6, fontweight="bold", color=MUTED, zorder=3)

    # Core daemon
    box(ax, 0.60, 3.00, 4.95, 3.55, "Noesis daemon",
        ["captures every turn · injects memory into the system prompt"],
        style="solid", title_size=10)
    for x0, x1, *_ in tools:
        cx = (x0 + x1) / 2
        arrow(ax, (cx, 4.30 - g), (cx, 3.55 + g))

    # Two-phase capture + confidence scorer
    box(ax, 0.60, 1.95, 2.20, 2.62, "Phase 1 · sync",
        ["embed → dedupe → insert", "as tentative · < 15 ms"])
    box(ax, 2.55, 1.95, 3.95, 2.62, "Phase 2 · async",
        ["extract → score →", "supersede → store"])
    box(ax, 4.20, 1.95, 5.30, 2.62, "Confidence",
        ["4-signal scorer", "+ per-type decay"])
    arrow(ax, (1.40, 3.00 - g), (1.40, 2.62 + g))
    arrow(ax, (3.25, 3.00 - g), (3.25, 2.62 + g))
    arrow(ax, (3.95 + g, 2.285), (4.20 - g, 2.285), both=True)

    # Storage
    box(ax, 0.60, 0.40, 2.20, 1.02, "Hot store",
        ["sqlite-vec · serves retrieval"], style="neutral")
    box(ax, 2.55, 0.40, 3.95, 1.02, "Cold store",
        ["Obsidian vault (Markdown)", "human-readable, editable"], style="neutral")
    arrow(ax, (1.40, 1.95 - g), (1.40, 1.02 + g))
    arrow(ax, (3.25, 1.95 - g), (3.25, 1.02 + g))
    ax.text(1.47, 1.485, "insert", ha="left", va="center", fontsize=7.4, color=MUTED)
    ax.text(3.32, 1.485, "write", ha="left", va="center", fontsize=7.4, color=MUTED)
    arrow(ax, (2.20 + g, 0.68), (2.55 - g, 0.68), both=True)
    ax.text(2.375, 0.77, "sync", ha="center", va="bottom", fontsize=7.2, color=MUTED)

    # Retrieval path: hot store → daemon, routed along the left gutter
    path(ax, [(0.60 - g, 0.71), (0.36, 0.71), (0.36, 3.275), (0.60 - g, 3.275)],
         color=BLUE, lw=1.0)
    ax.text(0.255, 1.95, "retrieval → injection", rotation=90, ha="center",
            va="center", fontsize=7.4, color=INK2)

    save(fig, "fig1_architecture")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Retrieval and injection pipeline
# ═══════════════════════════════════════════════════════════════════════════

def fig2_pipeline():
    W, H = 5.8333, 1.9298
    fig, ax = canvas(W, H)
    g = 0.02
    yc = 0.92                      # main flow line
    ylo, yhi = yc - 0.30, yc + 0.30

    box(ax, 0.04, ylo, 0.58, yhi, "User\nquery", style="neutral", title_size=8.2)

    # Three parallel channels
    chans = [(1.57, "BM25", ["exact terms"]),
             (yc,   "Dense vector", ["paraphrase"]),
             (0.27, "Signals", ["recency, core"])]
    cx0, cx1, ch = 0.86, 1.80, 0.21
    for y, t, sub in chans:
        box(ax, cx0, y - ch, cx1, y + ch, t, sub, title_size=8.2, line_size=7.4)

    # Fan-out: query → bus → channels
    bus1 = 0.72
    ax.add_line(Line2D([0.58 + g, bus1], [yc, yc], color=INK2, lw=0.9, zorder=4))
    ax.add_line(Line2D([bus1, bus1], [chans[2][0], chans[0][0]], color=INK2, lw=0.9, zorder=4))
    for y, *_ in chans:
        arrow(ax, (bus1, y), (cx0 - g, y))

    # Fan-in: channels → bus → RRF
    bus2 = 1.95
    for y, *_ in chans:
        ax.add_line(Line2D([cx1 + g, bus2], [y, y], color=INK2, lw=0.9, zorder=4))
    ax.add_line(Line2D([bus2, bus2], [chans[2][0], chans[0][0]], color=INK2, lw=0.9, zorder=4))

    stages = [
        (2.10, 2.84, "RRF fusion", ["+ time decay"], "tint"),
        (3.06, 3.84, "Confidence\ngate", ["≥ provisional"], "solid"),
        (4.06, 4.88, "Token\nbudget", ["≤ 1,200 tokens"], "solid"),
        (5.10, 5.79, "System\nprompt", ["injection"], "neutral"),
    ]
    arrow(ax, (bus2, yc), (stages[0][0] - g, yc))
    for (x0, x1, t, sub, st) in stages:
        box(ax, x0, ylo, x1, yhi, t, sub, style=st, title_size=8.2, line_size=7.4)
    for a, b in zip(stages, stages[1:]):
        arrow(ax, (a[1] + g, yc), (b[0] - g, yc))

    save(fig, "fig2_pipeline")


if __name__ == "__main__":
    print("Generating diagrams...")
    fig1_architecture()
    fig2_pipeline()
    print(f"Done! Figures in: {OUT}")
