#!/usr/bin/env python
"""Run audiogear over a whole collection of datasets, smallest-first, with resume.

For each dataset that has a ``configs/feat_<ds>.yaml`` (see gen_configs.py), this:
  1. runs ``process.py --config-name feat_<ds>`` as a subprocess,
  2. merges the per-shard ``outputs/<ds>/ext_*.csv`` into
     ``<root>/<ds>/extended_metadata.csv``,
  3. writes a ``.FEAT_DONE`` marker so reruns skip finished datasets.

It is a **template** — adapt the merge target / markers to your storage.

Usage:
    export AUDIOGEAR_DATA_DIR=/path/to/data_root
    python examples/run_batch.py                 # all datasets with a feat_ config
    python examples/run_batch.py resd dialogs    # only the named ones

Run detached so it survives the session:
    setsid nohup python examples/run_batch.py > run_batch.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
import time

csv.field_size_limit(10**9)

DATA_ROOT = os.environ.get("AUDIOGEAR_DATA_DIR", "")
if not DATA_ROOT or not os.path.isdir(DATA_ROOT):
    sys.exit("Set AUDIOGEAR_DATA_DIR to your data root. Got: " + repr(DATA_ROOT))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def datasets() -> list[str]:
    ds = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
        and os.path.exists(os.path.join(DATA_ROOT, d, "metadata.csv"))
        and os.path.exists(os.path.join(REPO, "configs", f"feat_{d}.yaml"))
    ]
    ds.sort(key=lambda d: os.path.getsize(os.path.join(DATA_ROOT, d, "metadata.csv")))  # small first
    return ds


def merge_shards(ds: str) -> int:
    shards = sorted(glob.glob(os.path.join(REPO, "outputs", ds, "ext_*.csv")))
    if not shards:
        return 0
    out = os.path.join(DATA_ROOT, ds, "extended_metadata.csv")
    n = 0
    with open(out, "w", encoding="utf-8", newline="") as fo:
        writer = None
        for sh in shards:
            with open(sh, encoding="utf-8") as fi:
                for row in csv.DictReader(fi, delimiter="|"):
                    if writer is None:
                        writer = csv.DictWriter(fo, fieldnames=list(row.keys()), delimiter="|", extrasaction="ignore")
                        writer.writeheader()
                    writer.writerow(row)
                    n += 1
    return n


def main() -> None:
    only = set(sys.argv[1:])
    todo = [d for d in datasets() if (not only or d in only)]
    # HF_TOKEN etc. come from .env, loaded by `import audiogear` in the subprocess.
    env = dict(os.environ)
    print(f"[{time.strftime('%F %T')}] datasets to process: {len(todo)}", flush=True)
    for ds in todo:
        done = os.path.join(DATA_ROOT, ds, ".FEAT_DONE")
        ext = os.path.join(DATA_ROOT, ds, "extended_metadata.csv")
        if os.path.exists(done):
            print(f"[skip] {ds} (.FEAT_DONE)", flush=True)
            continue
        if os.path.exists(ext):  # already merged on a previous run -> reuse
            print(f"[skip] {ds} (extended_metadata.csv exists — reusing)", flush=True)
            with open(done, "w") as f:
                f.write(time.strftime("%F %T"))
            continue
        print(f"\n[{time.strftime('%F %T')}] ===== {ds} =====", flush=True)
        rc = subprocess.run(
            [sys.executable, "process.py", "--config-name", f"feat_{ds}"], cwd=REPO, env=env
        ).returncode
        if rc != 0:
            print(f"[FAIL] {ds} rc={rc} — skipping merge, continuing", flush=True)
            continue
        n = merge_shards(ds)
        if n > 0:
            with open(done, "w") as f:
                f.write(time.strftime("%F %T"))
            print(f"[OK] {ds}: extended_metadata.csv {n} rows", flush=True)
        else:
            print(f"[WARN] {ds}: no shards to merge", flush=True)
    print(f"\n[{time.strftime('%F %T')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
