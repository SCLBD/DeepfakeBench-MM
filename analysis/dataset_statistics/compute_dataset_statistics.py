#!/usr/bin/env python3
"""Compute per-sample statistics for the Mega-MMDF dataset.

For each audiovisual sample this records:
  1) duration          -> duration_sec           (container duration)
  2) frame count       -> frame_count            (video nb_frames, else round(duration*fps))
  3) video FPS         -> fps
  4) audio sample rate -> sample_rate
  5) file size         -> file_size_bytes
  6) demographic group -> demographic_group      (e.g. "Female_Black"; also gender/skin_tone)
  7) leading silence   -> leading_silence_sec    (per Smeu et al., CVPR 2025)

Plus auxiliary fields derived from the file path with no extra cost:
  path, split, modality (RARV/RAFV/FARV/FAFV), pipeline.

Leading silence definition (Smeu et al. 2025, "Circumventing Shortcuts in
Audio-Visual Deepfake Detection Datasets with Unsupervised Learning"):
the duration from the start of the audio until the first sample whose absolute
amplitude exceeds a threshold tau (default 5e-4), measured at 16 kHz mono
float32 ([-1, 1]). If no sample ever exceeds tau, the full audio length is
returned. This exactly mirrors the reference implementation
`compute_starting_silence` (threshold=0.0005, sr=16000).

Design notes:
  * Output is a single JSONL file, appended line-by-line. On re-run, already
    processed paths are read back from the JSONL and skipped, so the run is
    resumable after interruption (no separate checkpoint / intermediate files).
  * `--finalize` reads the JSONL and writes a `.parquet` for fast analysis,
    printing a compact per-modality + overall summary.

Usage:
  python3 compute_dataset_statistics.py --root <dataset_dir> [--workers 96]
  python3 compute_dataset_statistics.py --root <dataset_dir> --finalize
"""

import argparse
import json
import numpy as np
import os
import subprocess
import sys

from concurrent.futures import ProcessPoolExecutor

# Silence threshold on normalized float32 audio (Smeu et al. 2025).
DEFAULT_THRESHOLD = 5e-4
# Audio is decoded to this sample rate for the silence computation.
SILENCE_SR = 16000
# Path prefix used inside the split JSONs, stripped to get dataset-relative paths.
PREFIX = "Mega-MMDF/"

# Per-worker globals set by the pool initializer (works under fork and spawn).
ROOT = None
THRESHOLD = DEFAULT_THRESHOLD


def _init_worker(root, threshold):
    global ROOT, THRESHOLD
    ROOT = root
    THRESHOLD = threshold


def parse_fps(value):
    """Parse an ffprobe r_frame_rate string like '25/1' or '29.97' into a float."""
    if not value:
        return None
    value = value.strip()
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        try:
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def ffprobe_meta(path):
    """Return (duration_sec, fps, nb_frames, sample_rate) via a single ffprobe call."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError("ffprobe: %s" % r.stderr.strip()[:300])
    info = json.loads(r.stdout)
    fmt = info.get("format", {})
    duration = fmt.get("duration")
    duration = float(duration) if duration is not None else None
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = parse_fps(video.get("r_frame_rate")) if video else None
    nb_frames = None
    if video and video.get("nb_frames"):
        try:
            nb_frames = int(video["nb_frames"])
        except (TypeError, ValueError):
            nb_frames = None
    sample_rate = None
    if audio and audio.get("sample_rate"):
        try:
            sample_rate = int(audio["sample_rate"])
        except (TypeError, ValueError):
            sample_rate = None
    return duration, fps, nb_frames, sample_rate


def leading_silence_sec(path):
    """First time (s) where |amplitude| > THRESHOLD; full length if never exceeds it.

    Decodes the whole audio stream to float32 mono at SILENCE_SR Hz via ffmpeg and
    scans it. This matches the reference librosa-based implementation and is fast
    in practice (subprocess startup dominates over decode for ~5 s clips).
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(SILENCE_SR), "-ac", "1", "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: %s" % r.stderr.strip()[:300])
    raw = r.stdout
    if not raw:
        return None  # no audio stream
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return None
    above = np.abs(samples) > THRESHOLD
    if above.any():
        # argmax of a bool array is the index of the first True.
        return float(np.argmax(above)) / SILENCE_SR
    return float(samples.size) / SILENCE_SR


def parse_rel(rel):
    """Split a dataset-relative path into (modality, pipeline, subgroup, gender, skin_tone)."""
    parts = rel.split("/")  # [modality, pipeline, subgroup, filename]
    modality = parts[0]
    pipeline = parts[1]
    subgroup = parts[2]
    gender, _, skin_tone = subgroup.partition("_")
    return modality, pipeline, subgroup, gender, skin_tone


def process_one(task):
    rel, split = task
    abs_path = os.path.join(ROOT, rel)
    rec = {"path": rel, "split": split}
    try:
        modality, pipeline, subgroup, gender, skin_tone = parse_rel(rel)
        rec["modality"] = modality
        rec["pipeline"] = pipeline
        rec["demographic_group"] = subgroup
        rec["gender"] = gender
        rec["skin_tone"] = skin_tone

        duration, fps, nb_frames, sample_rate = ffprobe_meta(abs_path)
        rec["duration_sec"] = duration
        rec["fps"] = fps
        rec["sample_rate"] = sample_rate
        if nb_frames is not None:
            rec["frame_count"] = nb_frames
        elif duration is not None and fps:
            rec["frame_count"] = int(round(duration * fps))
        else:
            rec["frame_count"] = None

        rec["file_size_bytes"] = os.path.getsize(abs_path)
        rec["leading_silence_sec"] = leading_silence_sec(abs_path)
    except Exception as exc:  # noqa: BLE001 - record failure without dropping the row
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    return rec


def build_tasks(root):
    tasks = []
    for name in ("train", "val", "test"):
        path = os.path.join(root, name + ".json")
        with open(path) as fh:
            full_paths = json.load(fh)
        for full in full_paths:
            rel = full[len(PREFIX):] if full.startswith(PREFIX) else full
            tasks.append((rel, name))
    return tasks


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
                # Tolerate a truncated trailing line from an interrupted run.
                continue
    return done


def _read_jsonl_tolerant(jsonl_path):
    """Read JSONL rows, dropping a possibly-truncated trailing line."""
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def finalize(jsonl_path, parquet_path):
    import pandas as pd
    df = pd.DataFrame(_read_jsonl_tolerant(jsonl_path))
    df.to_parquet(parquet_path, index=False)
    print("wrote %s (%d rows)" % (parquet_path, len(df)))

    # Compact summary useful for the paper.
    print("\n=== Overall ===")
    print("total samples : %d" % len(df))
    n_err = int(df["error"].notna().sum()) if "error" in df.columns else 0
    print("with errors   : %d" % n_err)
    print("duration_sec  : mean=%.3f median=%.3f min=%.3f max=%.3f"
          % (df["duration_sec"].mean(), df["duration_sec"].median(),
             df["duration_sec"].min(), df["duration_sec"].max()))
    print("file_size_MB  : mean=%.3f median=%.3f min=%.3f max=%.3f"
          % (df["file_size_bytes"].mean() / 1e6, df["file_size_bytes"].median() / 1e6,
             df["file_size_bytes"].min() / 1e6, df["file_size_bytes"].max() / 1e6))
    print("file_size<1MB : %.2f%%" % (100.0 * (df["file_size_bytes"] < 1e6).mean()))
    print("fps values    : %s" % dict(df["fps"].value_counts(dropna=False).to_dict()))
    print("sample_rate   : %s" % dict(df["sample_rate"].value_counts(dropna=False).to_dict()))
    sil = df["leading_silence_sec"]
    print("lead_silence_s: mean=%.4f median=%.4f max=%.4f  >0: %.2f%%"
          % (sil.mean(), sil.median(), sil.max(), 100.0 * (sil > 0).mean()))

    print("\n=== Per modality ===")
    cols = ["duration_sec", "file_size_bytes", "leading_silence_sec"]
    g = df.groupby("modality").agg(
        n=("path", "size"),
        dur_mean=("duration_sec", "mean"),
        size_mean_MB=("file_size_bytes", lambda s: s.mean() / 1e6),
        silence_mean=("leading_silence_sec", "mean"),
        silence_gt0=("leading_silence_sec", lambda s: 100.0 * (s > 0).mean()),
    )
    print(g.to_string())


def main():
    ap = argparse.ArgumentParser(description="Compute per-sample Mega-MMDF statistics.")
    ap.add_argument("--root", default=None,
                    help="Dataset root (default: current directory).")
    ap.add_argument("--output", default=None,
                    help="Output JSONL path (default: <root>/dataset_statistics.jsonl).")
    ap.add_argument("--workers", type=int, default=96,
                    help="Number of worker processes (default: 96).")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Leading-silence amplitude threshold (default: %s)." % DEFAULT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N (not-yet-done) samples (for testing).")
    ap.add_argument("--finalize", action="store_true",
                    help="Convert the JSONL to parquet and print a summary, then exit.")
    args = ap.parse_args()

    root = args.root or os.getcwd()
    output = args.output or os.path.join(root, "dataset_statistics.jsonl")

    if args.finalize:
        finalize(output, os.path.splitext(output)[0] + ".parquet")
        return

    tasks = build_tasks(root)
    print("total samples listed in split JSONs : %d" % len(tasks))

    done = load_done(output)
    remaining = [t for t in tasks if t[0] not in done]
    print("already done (resume)              : %d" % len(done))
    print("remaining to process               : %d" % len(remaining))
    if args.limit is not None:
        remaining = remaining[: args.limit]
        print("--limit applied                    : %d" % len(remaining))
    if not remaining:
        print("nothing to do.")
        return

    n_errors = 0
    n_written = 0
    try:
        with open(output, "a") as fh:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(root, args.threshold),
            ) as ex:
                from tqdm import tqdm
                for rec in tqdm(ex.map(process_one, remaining, chunksize=32),
                                total=len(remaining), desc="stats"):
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1
                    if rec.get("error"):
                        n_errors += 1
                    # Flush periodically regardless of errors so a kill loses at
                    # most a few hundred records (the rest are already on disk).
                    if n_written % 500 == 0:
                        fh.flush()
    except KeyboardInterrupt:
        print("\nInterrupted; progress is saved. Re-run the same command to resume.")
        sys.exit(130)

    print("done. errors=%d  output=%s" % (n_errors, output))
    print("run again with --finalize to write the parquet and print the summary.")


if __name__ == "__main__":
    main()
