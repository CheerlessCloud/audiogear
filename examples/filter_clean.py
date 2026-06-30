#!/usr/bin/env python
"""Select "good" clips from a computed ``extended_metadata.csv`` into
``clean_metadata.csv`` per dataset, logging how many were dropped and why.

A **template** for the filtering step after feature extraction — the thresholds
below are starting points; tune them to your distributions. Notes baked into the
defaults:
  - WER/CER is only trustworthy where `text` is independent of the ASR model.
    For auto-ASR datasets, `whisper_cer≈0` is meaningless — lean on
    MOS/SNR/STOI/asr_agreement there instead.
  - A speaker id is required (speaker-wise concatenation): non-empty `speaker_id`
    OR `speaker_conf >= 0.75`.
  - 16 kHz-native sets have bandwidth ≈ 7-8 kHz — don't drop them hard (min 6000).

Usage:
    export AUDIOGEAR_DATA_DIR=/path/to/data_root
    python examples/filter_clean.py             # all datasets with extended_metadata.csv
    python examples/filter_clean.py resd ...    # only the named ones
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter

csv.field_size_limit(10**9)

DATA_ROOT = os.environ.get("AUDIOGEAR_DATA_DIR", "")
if not DATA_ROOT or not os.path.isdir(DATA_ROOT):
    sys.exit("Set AUDIOGEAR_DATA_DIR to your data root. Got: " + repr(DATA_ROOT))

# Where `text` is human-authored (CER usable as a filter) vs ASR-derived.
# EDIT for your collection.
HUMAN_TEXT: set[str] = set()
ASR_TEXT: set[str] = set()

TH = {
    "distillmos": 3.0, "wada_snr": 15.0, "stoi": 0.85, "dur_min": 1.0, "dur_max": 20.0,
    "cer_max": 0.15, "asr_agree_min": 0.6, "bandwidth_min": 6000.0, "speaker_conf_min": 0.75,
    "srate_min": 5.0, "srate_max": 30.0,
}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def reasons_to_drop(ds: str, r: dict) -> list[str]:
    """Return the list of failed checks (empty => keep the clip)."""
    bad = []
    mos, snr, stoi = fnum(r.get("distillmos")), fnum(r.get("wada_snr")), fnum(r.get("pyt_stoi"))
    dur, bw = fnum(r.get("duration")), fnum(r.get("bandwidth_hz"))
    cer, agree, srate = fnum(r.get("whisper_cer")), fnum(r.get("asr_agreement")), fnum(r.get("speaking_rate"))
    if mos is not None and mos < TH["distillmos"]:
        bad.append("mos")
    if snr is not None and snr < TH["wada_snr"]:
        bad.append("snr")
    if stoi is not None and stoi < TH["stoi"]:
        bad.append("stoi")
    if dur is not None and not (TH["dur_min"] <= dur <= TH["dur_max"]):
        bad.append("dur")
    if bw is not None and bw < TH["bandwidth_min"]:
        bad.append("bandwidth")
    if srate is not None and not (TH["srate_min"] <= srate <= TH["srate_max"]):
        bad.append("srate")
    if ds in HUMAN_TEXT:
        if cer is not None and cer > TH["cer_max"]:
            bad.append("cer")
    elif ds in ASR_TEXT:
        if agree is not None and agree < TH["asr_agree_min"]:
            bad.append("asr_agree")
    spk = r.get("speaker_id") or r.get("speaker") or ""
    sconf = fnum(r.get("speaker_conf"))
    if not spk and not (sconf is not None and sconf >= TH["speaker_conf_min"]):
        bad.append("no_speaker")
    if not (r.get("text") or "").strip():
        bad.append("no_text")
    return bad


def process(ds: str) -> None:
    ext = os.path.join(DATA_ROOT, ds, "extended_metadata.csv")
    if not os.path.exists(ext):
        print(f"[skip] {ds}: no extended_metadata.csv")
        return
    with open(ext, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    if not rows:
        print(f"[skip] {ds}: empty")
        return
    keep, reasons = [], Counter()
    for r in rows:
        bad = reasons_to_drop(ds, r)
        if bad:
            reasons.update(bad)
        else:
            keep.append(r)
    out = os.path.join(DATA_ROOT, ds, "clean_metadata.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="|", extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    pct = 100 * len(keep) // max(1, len(rows))
    print(f"[{ds}] {len(rows)} -> kept {len(keep)} ({pct}%). dropped by: {dict(reasons.most_common())}")


def main() -> None:
    only = set(sys.argv[1:])
    dss = [
        d for d in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and (not only or d in only)
    ]
    for ds in dss:
        process(ds)


if __name__ == "__main__":
    main()
