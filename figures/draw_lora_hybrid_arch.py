#!/usr/bin/env python3
"""
Generate a publication-style block diagram: LoRA + bucket-head hybrid (XS combo-bucket factor pipeline).

Outputs (same directory as this script):
  - lora_hybrid_arch.svg  (vector, good for papers)
  - lora_hybrid_arch.png  (raster preview)

Requires: matplotlib

Example:
  python3 draw_lora_hybrid_arch.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_box(ax, xy, w, h, text, facecolor, fontsize=9, text_color="black"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        wrap=True,
    )
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color="#333333",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Title
    ax.text(
        5,
        5.25,
        "LoRA–hybrid architecture (trainable vs frozen)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#111111",
    )

    # Row 1: input
    draw_box(
        ax,
        (0.35, 3.85),
        2.0,
        0.95,
        "Tokenized prompt\n(combo events → text)",
        "#E8F4FC",
        fontsize=8.5,
    )

    # Backbone: split frozen vs LoRA
    draw_box(
        ax,
        (2.85, 3.55),
        3.3,
        1.55,
        "Pretrained causal LM\n(Mistral-7B, …)",
        "#F5F5F5",
        fontsize=9,
    )
    ax.text(4.5, 4.55, "frozen weights", ha="center", va="center", fontsize=7, color="#555555", style="italic")
    draw_box(
        ax,
        (3.05, 3.7),
        1.35,
        0.55,
        "LoRA on\nq/k/v/o_proj",
        "#FFE8CC",
        fontsize=7.5,
    )
    ax.text(3.72, 3.52, "trainable", ha="center", fontsize=6.5, color="#884400")

    # Hidden states
    draw_box(
        ax,
        (6.55, 3.85),
        1.55,
        0.95,
        "Last-layer\nhidden state",
        "#E8F8E8",
        fontsize=8.5,
    )

    # Pool
    draw_box(
        ax,
        (8.35, 3.85),
        1.3,
        0.95,
        "Last-token\npooling",
        "#E8F8E8",
        fontsize=8.5,
    )

    # Head
    draw_box(
        ax,
        (6.9, 2.05),
        2.2,
        0.9,
        "Linear head\nH → 5 buckets",
        "#E8D4F0",
        fontsize=9,
    )
    ax.text(8.0, 1.85, "trainable", ha="center", fontsize=6.5, color="#552266")

    # Output
    draw_box(
        ax,
        (6.75, 0.55),
        2.5,
        0.85,
        "Softmax expected bucket → Factor ∈ [0,100]",
        "#FFF8E1",
        fontsize=8.5,
    )

    # Arrows
    arrow(ax, 2.4, 4.325, 2.82, 4.325)
    arrow(ax, 6.18, 4.325, 6.52, 4.325)
    arrow(ax, 8.13, 4.325, 8.32, 4.325)
    arrow(ax, 9.0, 3.82, 8.5, 3.15)
    arrow(ax, 8.0, 2.02, 8.0, 1.45)

    # Side note: supervision
    ax.text(
        0.35,
        2.35,
        "Supervision:\n5-class CE vs\nuniverse quintile\nlabel y ∈ {0,…,4}",
        fontsize=7.5,
        color="#333333",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FAFAFA", edgecolor="#CCCCCC"),
    )

    # Loss arrow (conceptual)
    ax.annotate(
        "",
        xy=(6.85, 2.45),
        xytext=(2.0, 2.45),
        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8, linestyle="--"),
    )
    ax.text(4.4, 2.62, "L = cross-entropy(logits, y)", fontsize=7, color="#666666", ha="center")

    plt.tight_layout()
    svg_path = os.path.join(out_dir, "lora_hybrid_arch.svg")
    png_path = os.path.join(out_dir, "lora_hybrid_arch.png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close()
    print("Wrote:", svg_path)
    print("Wrote:", png_path)


if __name__ == "__main__":
    main()
