#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle


def draw_maze_panel(ax):
    ax.set_title("Environment & Graph", fontsize=12, pad=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    walls = [
        Rectangle((0.2, 0.2), 9.6, 0.4, color="#333333"),
        Rectangle((0.2, 7.4), 9.6, 0.4, color="#333333"),
        Rectangle((0.2, 0.2), 0.4, 7.6, color="#333333"),
        Rectangle((9.4, 0.2), 0.4, 7.6, color="#333333"),
        Rectangle((2.0, 1.5), 0.5, 4.8, color="#666666"),
        Rectangle((4.3, 2.0), 0.5, 5.2, color="#666666"),
        Rectangle((6.6, 0.8), 0.5, 4.8, color="#666666"),
    ]
    for w in walls:
        ax.add_patch(w)

    nodes = [
        (1.0, 1.0), (1.6, 2.2), (2.7, 2.7), (3.6, 3.2), (5.2, 3.7),
        (6.2, 4.4), (7.3, 5.1), (8.2, 6.5), (7.8, 2.0), (5.8, 1.3),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (4, 8), (8, 9)]

    for i, j in edges:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        ax.plot([x1, x2], [y1, y2], color="#2c7fb8", lw=2.0, alpha=0.9)

    for idx, (x, y) in enumerate(nodes):
        c = "#1f78b4"
        if idx == 0:
            c = "#2ca25f"
        if idx == 7:
            c = "#d95f0e"
        ax.add_patch(Circle((x, y), 0.18, facecolor=c, edgecolor="white", lw=0.8))

    ax.text(0.8, 0.55, "source", color="#2ca25f", fontsize=9)
    ax.text(7.7, 6.9, "goal source", color="#d95f0e", fontsize=9)
    ax.text(0.5, 7.95, "maze-constrained living graph", fontsize=9, color="#444444")


def draw_nn_panel(ax):
    ax.set_title("Local Neural Controller (node_nn)", fontsize=12, pad=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.add_patch(Rectangle((0.8, 1.0), 2.0, 6.0, facecolor="#e8f1fa", edgecolor="#4c78a8", lw=1.5))
    ax.add_patch(Rectangle((4.0, 1.7), 2.0, 4.6, facecolor="#f2e8ff", edgecolor="#7a5195", lw=1.5))
    ax.add_patch(Rectangle((7.2, 1.3), 2.0, 5.4, facecolor="#e9f7ef", edgecolor="#2a9d8f", lw=1.5))

    ax.text(1.8, 7.35, "8 inputs", ha="center", fontsize=10, color="#2f4b7c")
    ax.text(5.0, 6.65, "8 hidden", ha="center", fontsize=10, color="#5c2a9d")
    ax.text(8.2, 7.0, "7 outputs", ha="center", fontsize=10, color="#1f6f63")

    input_labels = [
        "wall pressure", "crowding", "target vec", "energy", "edge stats", "local angle", "source bias", "clamped features"
    ]
    for i, label in enumerate(input_labels):
        y = 6.6 - i * 0.72
        ax.text(0.95, y, f"• {label}", fontsize=8, color="#1d3557")

    output_labels = [
        "move", "prune", "thicken", "sprout", "apoptosis", "snap", "fusion tendency"
    ]
    for i, label in enumerate(output_labels):
        y = 6.2 - i * 0.75
        ax.text(7.35, y, f"• {label}", fontsize=8, color="#0f4c45")

    for y0 in [6.2, 5.4, 4.6, 3.8, 3.0, 2.2]:
        ax.add_patch(FancyArrowPatch((2.9, y0), (3.9, 4.0), arrowstyle="->", mutation_scale=8, lw=0.9, color="#6c757d", alpha=0.7))
        ax.add_patch(FancyArrowPatch((6.1, 4.0), (7.1, y0), arrowstyle="->", mutation_scale=8, lw=0.9, color="#6c757d", alpha=0.7))


def draw_dynamics_panel(ax):
    ax.set_title("C++ Simulation Dynamics", fontsize=12, pad=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    boxes = [
        (0.6, 5.8, 3.6, 1.4, "energy diffusion\n+ maintenance"),
        (5.0, 5.8, 3.8, 1.4, "pulse source/sink\nperiodic forcing"),
        (0.6, 3.5, 3.6, 1.4, "growth / pruning\nedge weight update"),
        (5.0, 3.5, 3.8, 1.4, "fusion + cleanup\nwall-safe topology"),
        (2.8, 1.1, 4.0, 1.5, "connectivity metric\npersistent ratio"),
    ]

    colors = ["#edf8fb", "#fee8c8", "#e5f5e0", "#f2f0f7", "#fff7bc"]
    edges = ["#2b8cbe", "#e34a33", "#31a354", "#756bb1", "#b8860b"]

    for (x, y, w, h, txt), fc, ec in zip(boxes, colors, edges):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    arrows = [
        ((4.2, 6.5), (5.0, 6.5)),
        ((2.4, 5.8), (2.4, 4.9)),
        ((6.9, 5.8), (6.9, 4.9)),
        ((4.2, 4.2), (5.0, 4.2)),
        ((2.4, 3.5), (4.2, 2.6)),
        ((6.9, 3.5), (5.8, 2.6)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, lw=1.4, color="#555555"))


def draw_metric_strip(ax):
    ax.set_title("Observed Behavior", fontsize=12, pad=8)
    ax.set_xlim(0, 50)
    ax.set_ylim(0.6, 1.0)
    ax.grid(alpha=0.25)
    ax.set_xlabel("step")
    ax.set_ylabel("persistent ratio")

    x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 43, 50]
    y = [1.0, 0.98, 0.95, 0.92, 0.89, 0.87, 0.855, 0.845, 0.84, 0.836, 0.82]
    ax.plot(x, y, color="#1f78b4", lw=2.2)
    ax.axvline(43, color="#d62728", ls="--", lw=1.5)
    ax.scatter([43], [0.836], color="#d62728", zorder=3)
    ax.text(43.5, 0.846, "x=43, 83.6%", fontsize=9, color="#b22222")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a poster-style conceptual figure for the node_nn C++ engine.")
    parser.add_argument("--out-png", default="results/poster/node_nn_engine_poster.png")
    parser.add_argument("--out-svg", default="results/poster/node_nn_engine_poster.svg")
    parser.add_argument("--show", action="store_true", help="Show the figure window before saving")
    args = parser.parse_args()

    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    draw_maze_panel(ax1)
    draw_nn_panel(ax2)
    draw_dynamics_panel(ax3)
    draw_metric_strip(ax4)

    fig.suptitle("Mycelium-Intelligence Engine: Local Neural Decisions + Bio-inspired Graph Dynamics", fontsize=16, y=1.01)

    if args.show:
        plt.show()

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Poster figure generated.")
    print(f"- PNG: {out_png.resolve()}")
    print(f"- SVG: {out_svg.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
