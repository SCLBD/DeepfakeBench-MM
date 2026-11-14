#!/usr/bin/env python3
"""Resolution analysis for Mega-MMDF (from resolution.jsonl).

Modality and pipeline are parsed straight out of the path prefix, which is
always `<modality>/<pipeline>/<demographic_group>/<file>`.

Produces:
  - resolution_by_modality.png       stacked horizontal bar, top resolutions
                                     broken down by the 4 modalities
  - resolution_real_vs_fake_video.png  two pies: real-video (RARV+FARV) vs
                                     fake-video (RAFV+FAFV) resolution mix
  - resolution_by_pipeline.txt       per-pipeline resolution mapping (shows
                                     which pipelines are single-resolution)
"""

import collections
import json
import matplotlib.pyplot as plt
import os

import plot_common as pc

RES_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolution.jsonl")
MIN_PCT = 0.5  # resolutions below this share are grouped into "Other"

VIDEO_REAL = ["RARV", "FARV"]
VIDEO_FAKE = ["RAFV", "FAFV"]

MODALITY_COLORS = {
    "RARV": "#2c7bb6",  # real audio + real video
    "RAFV": "#d7191c",  # real audio + fake video
    "FARV": "#1a9641",  # fake audio + real video
    "FAFV": "#fdae61",  # fake audio + fake video
}


def load():
    mod_res = collections.defaultdict(collections.Counter)
    pipe_res = collections.defaultdict(collections.Counter)
    n = 0
    with open(RES_JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            modality, pipeline = rec["path"].split("/")[:2]
            res = rec["resolution"]
            mod_res[modality][res] += 1
            pipe_res[pipeline][res] += 1
            n += 1
    return mod_res, pipe_res, n


def overall_counter(mod_res):
    c = collections.Counter()
    for m in mod_res:
        c.update(mod_res[m])
    return c


def split_top(overall, total):
    """Split resolutions into those >= MIN_PCT (ordered desc) and the rest."""
    top, other = [], []
    for r, c in overall.most_common():
        (top if 100.0 * c / total >= MIN_PCT else other).append(r)
    return top, other


def by_modality_bar(mod_res, top, other, total):
    rows = [(r, [mod_res[m].get(r, 0) for m in pc.MODALITY_ORDER]) for r in top]
    if other:
        rows.append((
            "Other (%d types)" % len(other),
            [sum(c for r, c in mod_res[m].items() if r in other)
             for m in pc.MODALITY_ORDER],
        ))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = list(range(len(rows)))
    left = [0] * len(rows)
    for mi, m in enumerate(pc.MODALITY_ORDER):
        vals = [row[1][mi] for row in rows]
        ax.barh(y, vals, left=left, color=MODALITY_COLORS[m],
                label=pc.MODALITY_LABELS[m], edgecolor="white", linewidth=0.5)
        left = [l + v for l, v in zip(left, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Number of samples")
    ax.set_title("Video resolution by modality (Mega-MMDF)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    pc.save(fig, "resolution_by_modality.png")


def real_vs_fake_pies(mod_res):
    real = collections.Counter()
    fake = collections.Counter()
    for m in VIDEO_REAL:
        real.update(mod_res[m])
    for m in VIDEO_FAKE:
        fake.update(mod_res[m])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    for ax, counts, title in [(axes[0], real, "Real video (RARV+FARV)"),
                              (axes[1], fake, "Fake video (RAFV+FAFV)")]:
        g_total = sum(counts.values())
        top, other = split_top(counts, g_total)
        labels = top[:]
        vals = [counts[r] for r in top]
        if other:
            labels.append("Other (%d types)" % len(other))
            vals.append(sum(counts[r] for r in other))
        wedges, _ = ax.pie(vals, colors=palette[:len(vals)], startangle=90,
                           counterclock=False,
                           wedgeprops=dict(edgecolor="white", linewidth=0.5))
        ax.legend(wedges,
                  ["%s  %s" % (l, f"{v:,}") for l, v in zip(labels, vals)],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.set_title("%s\n(n=%s)" % (title, f"{g_total:,}"))
        ax.axis("equal")
    pc.save(fig, "resolution_real_vs_fake_video.png")


def modality_table(mod_res):
    """Write the full resolution x modality count matrix to a text file."""
    overall = overall_counter(mod_res)
    lines = [
        "# Resolution x modality sample counts (Mega-MMDF)",
        "resolution\t" + "\t".join(pc.MODALITY_ORDER) + "\ttotal",
    ]
    for r, c in overall.most_common():
        row = [str(mod_res[m].get(r, 0)) for m in pc.MODALITY_ORDER]
        lines.append("%s\t%s\t%d" % (r, "\t".join(row), c))
    lines.append("total\t" + "\t".join(
        str(sum(mod_res[m].values())) for m in pc.MODALITY_ORDER)
        + "\t" + str(sum(overall.values())))
    path = os.path.join(pc.OUTDIR, "resolution_by_modality.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def pipeline_table(pipe_res):
    lines = ["# Resolution mapping per pipeline (Mega-MMDF)",
             "# 'single' = pipeline outputs exactly one resolution",
             "pipeline\tn_resolutions\tresolutions(count)"]
    single = 0
    for p in sorted(pipe_res, key=lambda x: int(x.replace("pipeline", ""))):
        items = pipe_res[p].most_common()
        if len(items) == 1:
            single += 1
        detail = ", ".join("%s=%d" % (r, c) for r, c in items)
        lines.append("%s\t%d\t%s" % (p, len(items), detail))
    lines.append("# %d / %d pipelines are single-resolution"
                 % (single, len(pipe_res)))
    path = os.path.join(pc.OUTDIR, "resolution_by_pipeline.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


def main():
    mod_res, pipe_res, total = load()
    print("total samples:", total)
    overall = overall_counter(mod_res)
    top, other = split_top(overall, total)
    print("top resolutions (>=%.1f%%):" % MIN_PCT, top)
    print("other types:", len(other))
    by_modality_bar(mod_res, top, other, total)
    real_vs_fake_pies(mod_res)
    modality_table(mod_res)
    pipeline_table(pipe_res)


if __name__ == "__main__":
    main()
