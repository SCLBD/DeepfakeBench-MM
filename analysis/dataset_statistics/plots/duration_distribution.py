#!/usr/bin/env python3
"""Duration distribution: histogram + CDF (P(<=x)) overlay."""

import plot_common as pc

XMAX = 20.0   # s; covers ~99% of samples (q0.99 = 18.4 s)
BINS = 60


def main():
    df = pc.load_df()
    d = df["duration_sec"].dropna()
    pc.hist_with_cdf(
        d, bins=BINS, xmax=XMAX, xlabel="Duration (s)",
        title="Duration distribution (Mega-MMDF)",
        outname="duration_distribution.png",
        stats_name="duration_statistics.txt",
    )


if __name__ == "__main__":
    main()
