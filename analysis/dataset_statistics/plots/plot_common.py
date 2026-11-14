"""Shared plotting helpers for Mega-MMDF dataset statistics.

Every plot script in this directory does:

    import plot_common as pc
    df = pc.load_df()
    fig, ax = plt.subplots(...)
    ...
    pc.save(fig, "my_plot.png")

and uses `pc.write_stats(...)` for the numeric summary text files.

The parquet is resolved relative to this file so the directory is self-
contained and re-runnable from any location.
"""

import matplotlib

matplotlib.use("Agg")  # headless server, no display

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(_THIS_DIR, "dataset_statistics.parquet")
OUTDIR = os.path.join(_THIS_DIR)

# Consistent, publication-friendly defaults. English labels only (no CJK font).
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.autolayout": True,
})

MODALITY_ORDER = ["RARV", "RAFV", "FARV", "FAFV"]
MODALITY_LABELS = {
    "RARV": "Real audio + Real video",
    "RAFV": "Real audio + Fake video",
    "FARV": "Fake audio + Real video",
    "FAFV": "Fake audio + Fake video",
}
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

# Okabe-Ito colorblind-safe categorical palette (Okabe & Ito 2008) -- the
# standard 8-color qualitative palette recommended for publication figures.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
OKABE_BLUE = "#0072B2"
OKABE_VERMILION = "#D55E00"


def load_df():
    """Load the per-sample statistics parquet into a DataFrame."""
    return pd.read_parquet(PARQUET)


def save(fig, name):
    """Save `fig` into OUTDIR/name and close it. Returns the full path."""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


def write_stats(series, title, name):
    """Write a compact summary (min/max/mean/median/std/quantiles) to OUTDIR/name.

    Returns the text content (also printed).
    """
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    lines = [
        "# %s statistics" % title,
        "count  : %d" % int(series.count()),
        "min    : %.6f" % float(series.min()),
        "max    : %.6f" % float(series.max()),
        "mean   : %.6f" % float(series.mean()),
        "median : %.6f" % float(series.median()),
        "std    : %.6f" % float(series.std()),
    ]
    for q in qs:
        lines.append("q%.2f   : %.6f" % (q, float(series.quantile(q))))
    text = "\n".join(lines) + "\n"
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)
    print(text)
    return text


def hist_with_cdf(values, bins, xmax, xlabel, title, outname, stats_name):
    """Histogram (count, left axis) with a CDF overlay (right axis), no in-figure stats.

    The x axis is clipped to [0, xmax] to cover the bulk of a right-skewed
    distribution; the tail beyond xmax is captured in the written stats file,
    not annotated on the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(values, bins=bins, range=(0.0, xmax), color=OKABE_BLUE,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Mean / median as dashed vertical markers (location only, no numeric label).
    mean = float(values.mean())
    med = float(values.median())
    ax.axvline(mean, color="#000000", linestyle="--", linewidth=1.4, label="Mean")
    ax.axvline(med, color="#666666", linestyle=":", linewidth=1.4, label="Median")
    ax.legend(loc="center right", frameon=False)

    # CDF (cumulative fraction <= x) on a twin axis.
    xs = np.sort(values)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    mask = xs <= xmax
    ax2 = ax.twinx()
    ax2.plot(xs[mask], cdf[mask], drawstyle="steps-post",
             color=OKABE_VERMILION, linewidth=1.6)
    ax2.set_ylabel("Cumulative fraction  P(≤ x)")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(False)

    ax.set_title(title)
    save(fig, outname)
    write_stats(values, title, stats_name)
