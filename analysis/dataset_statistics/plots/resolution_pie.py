#!/usr/bin/env python3
"""Video resolution (width x height) distribution pie chart for Mega-MMDF.

Reads the resumable ffprobe output `resolution.jsonl` (one {path,width,height,
resolution} per line) and plots the share of each distinct resolution. Slices
below 1% are grouped into an "Other" bucket for readability; the full per-
resolution breakdown is always written to `resolution_counts.txt`.
"""

import collections
import json
import matplotlib.pyplot as plt
import os

import plot_common as pc

RES_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolution.jsonl")
MIN_PCT = 0.5  # group slices smaller than this into "Other"


def load_resolutions():
    counts = collections.Counter()
    n_error = 0
    with open(RES_JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                n_error += 1
                continue
            counts[rec["resolution"]] += 1
    return counts, n_error


def main():
    counts, n_error = load_resolutions()
    total = sum(counts.values())
    print("total resolution entries:", total, "| errors:", n_error)

    items = counts.most_common()
    main = [(k, v) for k, v in items if 100.0 * v / total >= MIN_PCT]
    other = [(k, v) for k, v in items if 100.0 * v / total < MIN_PCT]
    other_total = sum(v for _, v in other)

    labels = [k for k, _ in main]
    vals = [v for _, v in main]
    if other_total > 0:
        labels.append("Other (%d types)" % len(other))
        vals.append(other_total)

    # Full breakdown to a text file.
    lines = ["# Video resolution (width x height) distribution",
             "# total = %d   errors = %d" % (total, n_error)]
    for k, v in items:
        lines.append("%s\t%d\t%.2f%%" % (k, v, 100.0 * v / total))
    txt_path = os.path.join(pc.OUTDIR, "resolution_counts.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", txt_path)

    # Pie chart with legend.
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    fig, ax = plt.subplots(figsize=(11, 8))
    wedges, _ = ax.pie(
        vals,
        colors=palette[:len(vals)],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=0.6),
    )
    legend_labels = [
        "%s   %s   (%.1f%%)" % (lbl, f"{v:,}", 100.0 * v / total)
        for lbl, v in zip(labels, vals)
    ]
    ax.legend(wedges, legend_labels, title="Resolution (WxH)",
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_title("Video resolution distribution (Mega-MMDF)")
    ax.axis("equal")
    pc.save(fig, "resolution_pie.png")


if __name__ == "__main__":
    main()
