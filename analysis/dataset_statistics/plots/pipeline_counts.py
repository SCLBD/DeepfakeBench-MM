#!/usr/bin/env python3
"""Sample counts per pipeline for Mega-MMDF.

Produces:
  - pipeline_counts_bar.png : bar chart, pipelines sorted by descending count
  - pipeline_counts_pie.png : pie chart, 23 slices with legend
  - pipeline_counts.txt     : "pipelineN  <count>" lines, sorted descending
"""

import matplotlib.pyplot as plt
import os

import plot_common as pc


def main():
    df = pc.load_df()
    counts = df["pipeline"].value_counts().sort_values(ascending=False)
    total = int(counts.sum())

    # Short x-axis labels: strip the "pipeline" prefix, keep only the number.
    short_names = [str(p).replace("pipeline", "") for p in counts.index]

    # ---------------------------------------------------------------- bar chart
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(counts)), counts.values, color="#4c72b0")

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=9)
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Number of samples")
    ax.set_title("Samples per pipeline")

    # Value labels above each bar (small font).
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    ax.set_ylim(0, counts.max() * 1.10)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    pc.save(fig, "pipeline_counts_bar.png")

    # ----------------------------------------------------------------- pie chart
    # 23 distinct categorical colors: tab20 (20) + tab20b (first 3).
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)[:3]

    fig, ax = plt.subplots(figsize=(11, 8))
    wedges, _ = ax.pie(
        counts.values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=0.6),
    )

    legend_labels = [
        "%s  %s  (%.1f%%)" % (p, f"{v:,}", 100.0 * v / total)
        for p, v in zip(counts.index, counts.values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Pipeline",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title_fontsize=9,
    )
    ax.set_title("Samples per pipeline (share)")
    ax.axis("equal")

    pc.save(fig, "pipeline_counts_pie.png")

    # ------------------------------------------------------------- text summary
    txt_path = os.path.join(pc.OUTDIR, "pipeline_counts.txt")
    lines = ["%s  %d" % (p, int(v)) for p, v in counts.items()]
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", txt_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
