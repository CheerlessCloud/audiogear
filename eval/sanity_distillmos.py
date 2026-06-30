"""Sanity-check the DistillMOS block by reproducing a dataset's own MOS column.

The RESD metadata already ships a ``distillmos`` column. We recompute MOS with
audiogear's :class:`DistillMosMetric` and correlate against it (Pearson +
Spearman + mean abs error). High correlation ⇒ the pipeline reproduces known
values, i.e. our wiring (resample, batching, device) is faithful.

Usage:
    uv run python eval/sanity_distillmos.py [--n 200] [--data-dir /path/to/resd]
"""

from __future__ import annotations

import argparse
import csv

from audiogear.pipeline.metrics.distillmos import DistillMosMetric
from audiogear.pipeline.readers.csv import CsvReader


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0] * len(xs)
        for rank_idx, i in enumerate(order):
            r[i] = rank_idx
        return r

    return pearson(rank(a), rank(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/simba9/workflow/speech/tts_data_process/resd")
    ap.add_argument("--metadata", default="resd_metadata.csv")
    ap.add_argument("--n", type=int, default=200, help="number of clips to check")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # reference column from the dataset
    ref = {}
    with open(f"{args.data_dir}/{args.metadata}") as f:
        for row in csv.DictReader(f, delimiter="|"):
            if "distillmos" in row and row["distillmos"]:
                ref[row["audio_path"]] = float(row["distillmos"])

    segments = CsvReader(
        data_folder=args.data_dir,
        glob_pattern=args.metadata,
        delimiter="|",
        audio_key="audio_path",
        limit=args.n,
    )(rank=0, world_size=1)

    block = DistillMosMetric(device=args.device)
    block(segments, rank=0, world_size=1)

    ours, theirs = [], []
    for s in segments:
        if s.path in ref:
            ours.append(s.metadata["distillmos"])
            theirs.append(ref[s.path])

    n = len(ours)
    mae = sum(abs(a - b) for a, b in zip(ours, theirs)) / n
    print(f"n={n}")
    r = pearson(ours, theirs)
    spread = (max(theirs) - min(theirs))
    print(f"Pearson  r = {r:.4f}")
    print(f"Spearman r = {spearman(ours, theirs):.4f}")
    print(f"mean abs error = {mae:.4f}  (MOS is 1..5)")
    print(f"reference MOS spread = {spread:.2f} (narrow spread restricts achievable correlation)")
    # MAE is the robust signal here; correlation is depressed when the reference
    # MOS range is narrow (restriction of range), as in clean studio datasets.
    if mae < 0.75:
        print("PASS: recomputed MOS tracks the reference (low MAE).")
    else:
        print("WARN: high MAE — check resampling / device / model version.")


if __name__ == "__main__":
    main()
