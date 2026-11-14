#!/usr/bin/env python3
"""Demographic-group composition for Mega-MMDF.

Produces three side-by-side pies (one per train/val/test split), each showing
the 8 demographic groups (Female/Male x Black/Brown/White/Yellow). Also writes
``demographic_counts.txt`` with the group x split count matrix.

Colours are the Okabe-Ito colorblind-safe palette (Okabe & Ito 2008).
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import pandas as pd

import plot_common as pc

GROUPS = [
    "Female_Black", "Female_Brown", "Female_White", "Female_Yellow",
    "Male_Black", "Male_Brown", "Male_White", "Male_Yellow",
]

LABEL_1LINE = {
    "Female_Black": "Female Black",
    "Female_Brown": "Female Brown",
    "Female_White": "Female White",
    "Female_Yellow": "Female Yellow",
    "Male_Black": "Male Black",
    "Male_Brown": "Male Brown",
    "Male_White": "Male White",
    "Male_Yellow": "Male Yellow",
}

# Okabe-Ito palette, one abstract hue per group (deliberately not mapped to
# skin tone).
GROUP_COLORS = {
    "Female_Black": "#0072B2",
    "Female_Brown": "#E69F00",
    "Female_White": "#009E73",
    "Female_Yellow": "#D55E00",
    "Male_Black": "#56B4E9",
    "Male_Brown": "#CC79A7",
    "Male_White": "#F0E442",
    "Male_Yellow": "#000000",
}

WEDGE_KW = dict(edgecolor="white", linewidth=0.6)


def _autopct_white(pct):
    """Percentage label (white text); caller re-tints light slices to black."""
    return "%.1f%%" % pct


def build_crosstab(df):
    ct = pd.crosstab(df["demographic_group"], df["split"])
    return ct.reindex(index=GROUPS, columns=pc.SPLIT_ORDER).fillna(0).astype(int)


def write_counts(ct, total):
    lines = [
        "# Sample counts: demographic_group x split (Mega-MMDF)",
        "# total samples = %d" % total,
        "demographic_group\ttrain\tval\ttest\ttotal",
    ]
    for g in GROUPS:
        vals = [int(ct.loc[g, s]) for s in pc.SPLIT_ORDER]
        lines.append("%s\t%d\t%d\t%d\t%d"
                     % (g, vals[0], vals[1], vals[2], sum(vals)))
    col_totals = [int(ct[s].sum()) for s in pc.SPLIT_ORDER]
    lines.append("total\t%d\t%d\t%d\t%d"
                 % (col_totals[0], col_totals[1], col_totals[2],
                    sum(col_totals)))
    text = "\n".join(lines) + "\n"
    path = os.path.join(pc.OUTDIR, "demographic_counts.txt")
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)
    print(text)
    return text


def _tint_autotexts(wedges, autotexts):
    """Black text on light slices, white on dark ones (readability on yellow)."""
    for w, at in zip(wedges, autotexts):
        r, g, b = w.get_facecolor()[:3]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        at.set_color("black" if lum > 0.6 else "white")


def by_split_pies(ct):
    group_colors = [GROUP_COLORS[g] for g in GROUPS]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, s in zip(axes, pc.SPLIT_ORDER):
        vals = [int(ct.loc[g, s]) for g in GROUPS]
        wedges, _, autotexts = ax.pie(
            vals, colors=group_colors,
            autopct="%.1f%%", pctdistance=0.80,
            startangle=90, counterclock=False,
            wedgeprops=dict(**WEDGE_KW),
            textprops=dict(fontsize=8, fontweight="bold"),
        )
        _tint_autotexts(wedges, autotexts)
        ax.set_aspect("equal")
        ax.set_title(s.capitalize())

    group_handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[g], edgecolor="white",
                       label=LABEL_1LINE[g])
        for g in GROUPS
    ]
    fig.legend(handles=group_handles, loc="lower center", ncol=4,
               frameon=False, title="Demographic group", title_fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    pc.save(fig, "demographic_groups_by_split.png")


def main():
    df = pc.load_df()
    total = int(len(df))
    ct = build_crosstab(df)
    write_counts(ct, total)
    by_split_pies(ct)


if __name__ == "__main__":
    main()
