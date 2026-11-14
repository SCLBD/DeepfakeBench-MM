#!/usr/bin/env python3
"""Extract video resolution (width x height) for every Mega-MMDF sample via ffprobe.

ffprobe-only (no audio decode), so this is much faster than the full stats pass.
Output is a resumable JSONL: one {"path", "width", "height", "resolution"} per line.
"""

import argparse
import json
import os
import subprocess

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = None


def _init(root):
    global ROOT
    ROOT = root


def get_res(rel):
    abs_path = os.path.join(ROOT, rel)
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", abs_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    rec = {"path": rel}
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        rec["error"] = "ffprobe: %s" % r.stderr.strip()[:200]
        return rec
    try:
        w, h = [int(x) for x in out.split(",")[:2]]
    except ValueError:
        rec["error"] = "bad output: %s" % out
        return rec
    rec["width"] = w
    rec["height"] = h
    rec["resolution"] = "%dx%d" % (w, h)
    return rec


def load_done(output):
    done = set()
    if not os.path.exists(output):
        return done
    with open(output) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Original Mega-MMDF root")
    ap.add_argument("--parquet", default=str(Path(__file__).with_name("dataset_statistics.parquet")))
    ap.add_argument("--output",
                    default=str(Path(__file__).with_name("resolution.jsonl")))
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet)
    paths = df["path"].tolist()
    print("total paths    :", len(paths))

    done = load_done(args.output)
    remaining = [p for p in paths if p not in done]
    print("done (resume)  :", len(done))
    print("remaining      :", len(remaining))
    if not remaining:
        print("nothing to do.")
        return

    n = 0
    from tqdm import tqdm
    with open(args.output, "a") as fh:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init, initargs=(args.root,)) as ex:
            for rec in tqdm(ex.map(get_res, remaining, chunksize=64),
                            total=len(remaining), desc="res"):
                fh.write(json.dumps(rec) + "\n")
                n += 1
                if n % 500 == 0:
                    fh.flush()
    print("done. wrote", n)


if __name__ == "__main__":
    main()
