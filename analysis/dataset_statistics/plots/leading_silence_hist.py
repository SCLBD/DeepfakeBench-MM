#!/usr/bin/env python3
"""Leading-silence distribution: binned histogram + KDE fitted curve.

Blue   = real audio (RARV + RAFV)
Orange = fake audio (FARV + FAFV)

Both the bars and the fitted curve are density-normalized (area == 1) so the
two groups are comparable despite different sample counts. The fitted curve is
a Gaussian KDE evaluated on a fine grid and sits on the same density scale as
the bars.
"""

import matplotlib.pyplot as plt
import numpy as np

import plot_common as pc

REAL_AUDIO = ["RARV", "RAFV"]
FAKE_AUDIO = ["FARV", "FAFV"]
XMAX_MS = 100.0

BAR_MS = 5.0       # visible binned histogram (5 ms bars)
CURVE_MS = 1.0     # fine grid for the smooth fitted (KDE) curve
SIGMA_MS = 2.5     # KDE gaussian bandwidth (ms)

BLUE = "#0072B2"    # Okabe-Ito blue   -> real audio
ORANGE = "#E69F00"  # Okabe-Ito orange -> fake audio

try:
    from scipy.ndimage import gaussian_filter1d as _gauss1d
except Exception:  # numpy-only fallback
    def _gauss1d(y, s):
        k = int(np.ceil(3 * s))
        if k < 1:
            return y
        x = np.arange(-k, k + 1)
        g = np.exp(-(x ** 2) / (2 * s ** 2))
        g /= g.sum()
        return np.convolve(y, g, mode="same")


def main():
    df = pc.load_df()
    ls = df["leading_silence_sec"]
    real = ls[df["modality"].isin(REAL_AUDIO)].dropna().to_numpy() * 1000.0
    fake = ls[df["modality"].isin(FAKE_AUDIO)].dropna().to_numpy() * 1000.0

    bar_bins = np.arange(0.0, XMAX_MS + BAR_MS, BAR_MS)
    bar_centers = (bar_bins[:-1] + bar_bins[1:]) / 2.0
    curve_bins = np.arange(0.0, XMAX_MS + CURVE_MS, CURVE_MS)
    curve_centers = (curve_bins[:-1] + curve_bins[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(8, 4.6))

    for data, color, label in (
        (real, BLUE, "Real audio (RARV + RAFV)"),
        (fake, ORANGE, "Fake audio (FARV + FAFV)"),
    ):
        hb, _ = np.histogram(data, bins=bar_bins, density=True)
        ax.bar(bar_centers, hb, width=BAR_MS * 0.9, color=color, alpha=0.4,
               edgecolor="none", align="center")

        hc, _ = np.histogram(data, bins=curve_bins, density=True)
        ax.plot(curve_centers, _gauss1d(hc, SIGMA_MS / CURVE_MS),
                color=color, linewidth=2.0, label=label)

    ax.set_xlabel("Leading silence (ms)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, XMAX_MS)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Leading silence distribution (real vs fake audio)")

    pc.save(fig, "leading_silence_hist_real_vs_fake.png")

    for label, s in (("real audio", real), ("fake audio", fake)):
        print(f"{label}: n={len(s)} mean={s.mean():.3f} ms "
              f"median={np.median(s):.3f} ms >0={(s > 0).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
