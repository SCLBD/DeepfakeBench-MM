#!/usr/bin/env python3
"""File size distribution: histogram + CDF (P(<=x)) overlay."""

import plot_common as pc

XMAX = 6.0    # MB; covers ~99.3% of samples (q0.99 = 5.33 MB)
BINS = 60


def main():
    df = pc.load_df()
    mb = (df["file_size_bytes"] / 1e6).dropna()
    pc.hist_with_cdf(
        mb, bins=BINS, xmax=XMAX, xlabel="File size (MB)",
        title="File size distribution (Mega-MMDF)",
        outname="file_size_distribution.png",
        stats_name="file_size_statistics.txt",
    )


if __name__ == "__main__":
    main()
