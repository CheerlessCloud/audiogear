#!/usr/bin/env python
"""Generate a per-dataset audiogear config (``feat_<ds>.yaml``) for every dataset
under a data root, choosing the metric set per dataset from simple profiles.

This is a **template** for processing a whole *collection* of datasets, not a
fixed CLI. Copy it, then edit the profile sets below to match your data. The
emitted configs use the same `reader: ru` / `writer: csv` / `executor: local`
presets as the rest of the repo.

Usage:
    export AUDIOGEAR_DATA_DIR=/path/to/data_root      # holds <dataset>/metadata.csv
    python examples/gen_configs.py                    # writes configs/feat_<ds>.yaml

The layout expected under the data root (see README "Dataset structure"):
    <root>/<dataset>/metadata.csv
    <root>/<dataset>/audio/...
"""

from __future__ import annotations

import os
import sys

# --- where the data and configs live --------------------------------------
DATA_ROOT = os.environ.get("AUDIOGEAR_DATA_DIR") or (sys.argv[1] if len(sys.argv) > 1 else "")
if not DATA_ROOT or not os.path.isdir(DATA_ROOT):
    sys.exit(
        "Set AUDIOGEAR_DATA_DIR (or pass the data root as arg 1) to a directory of "
        "<dataset>/metadata.csv folders. Got: " + repr(DATA_ROOT)
    )
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "configs")

# --- per-dataset profiles (EDIT THESE for your collection) ----------------
# Datasets whose `text` column is empty -> transcribe first via consensus ASR.
NO_TEXT: set[str] = set()
# Datasets with no speaker ids -> run SpeakerLabeler (embed + global cluster).
NO_SPEAKER: set[str] = set()
# WER is only meaningful where `text` is independent of the ASR model. If the
# transcript itself came from Whisper, WER≈0 — useless and expensive, so skip it.
HUMAN_TEXT: set[str] = set()
# Datasets to also tag with predicted gender / emotion.
WANT_GENDER: set[str] = set()
WANT_EMOTION: set[str] = set()

NGPU = int(os.environ.get("AUDIOGEAR_NGPU", "2"))  # GPUs to pin workers across


def block(target: str, indent: int = 2, **kw) -> str:
    pad = " " * indent
    s = f"{pad}- _target_: {target}\n"
    for k, v in kw.items():
        s += f"{pad}  {k}: {v}\n"
    return s


def consensus_block() -> str:
    return (
        "  - _target_: audiogear.pipeline.transcribers.consensus.ConsensusTranscriber\n"
        "    only_missing: true\n    overwrite_text: true\n    min_agreement: 0.5\n"
        "    backends:\n"
        "      - _target_: audiogear.pipeline.transcribers.backends.GigaAMBackend\n"
        "        model_name: v2_rnnt\n        device: ${device}\n"
        "      - _target_: audiogear.pipeline.transcribers.backends.WhisperBackend\n"
        "        model_name: large-v3\n        language: ru\n        device: ${device}\n"
    )


def metrics_yaml(ds: str, has_text: bool, want_speaker: bool) -> str:
    m = ""
    if not has_text:  # ASR is a shared producer of `text` -> run it BEFORE the lanes
        m += consensus_block()

    # Parallel lanes: CPU DSP metrics ∥ GPU metrics (overlap -> ~max, not sum).
    # Lane columns are disjoint; dependencies stay inside a lane (pitch -> style;
    # `text` -> WER / speaking_rate).
    cpu = block("audiogear.pipeline.metrics.bandwidth.BandwidthMetric", indent=8)
    cpu += block("audiogear.pipeline.metrics.pitch.PitchMetric", indent=8, backend="pyin")
    cpu += block("audiogear.pipeline.metrics.style.StyleMetric", indent=8)
    cpu += block("audiogear.pipeline.metrics.wada_snr.SnrMetric", indent=8)
    cpu += block("audiogear.pipeline.metrics.speaking_rate.SpeakingRateMetric", indent=8, language="ru")

    gpu = block("audiogear.pipeline.metrics.distillmos.DistillMosMetric", indent=8, device="${device}")
    gpu += block("audiogear.pipeline.metrics.squim.SquimMetrics", indent=8, device="${device}")
    if ds in HUMAN_TEXT:  # WER only where text is human-authored
        gpu += block(
            "audiogear.pipeline.metrics.wer.WhisperWer",
            indent=8, whisper_model="large-v3", language="ru", device="${device}",
        )
    if ds in WANT_GENDER:
        gpu += block("audiogear.pipeline.metrics.gender.GenderMetric", indent=8, device="${device}")
    if ds in WANT_EMOTION:
        gpu += block("audiogear.pipeline.metrics.emotion.EmotionMetric", indent=8, device="${device}")

    m += "  - _target_: audiogear.pipeline.parallel.ParallelLanes\n    lanes:\n      cpu:\n" + cpu + "      gpu:\n" + gpu
    if want_speaker:  # embeddings run in parallel on all GPUs -> FS barrier -> one global clustering
        m += block(
            "audiogear.pipeline.metrics.speaker.SpeakerLabeler",
            device="${device}", assign_threshold=0.75, margin=0.10, cache_dir=f"outputs/{ds}/_spk_emb",
        )
    return m


def write_config(ds: str) -> None:
    has_text = ds not in NO_TEXT
    want_speaker = ds in NO_SPEAKER
    # Speaker datasets need every shard's embeddings before the global clustering,
    # so run one shard per GPU at once; others shard more finely for resume.
    tasks, workers = (NGPU, NGPU) if want_speaker else (8, 2)
    cfg = f"""# AUTO-GENERATED by examples/gen_configs.py. Dataset: {ds}
defaults:
  - reader: ru
  - writer: csv
  - executor: local
  - _self_
hydra:
  job:
    chdir: false
  run:
    dir: .
  output_subdir: null
device: cuda
reader:
  data_folder: {os.path.join(DATA_ROOT, ds)}
  glob_pattern: metadata.csv
  limit: -1
writer:
  output_folder: outputs/{ds}
  output_filename: ext_$rank.csv
  sep: "|"
executor:
  tasks: {tasks}
  workers: {workers}
  skip_completed: true
  logging_dir: logs/{ds}
metrics:
{metrics_yaml(ds, has_text, want_speaker)}"""
    with open(os.path.join(CFG, f"feat_{ds}.yaml"), "w") as f:
        f.write(cfg)
    print(f"feat_{ds}.yaml  text={has_text} speaker={want_speaker} tasks={tasks} "
          f"gender={ds in WANT_GENDER} emotion={ds in WANT_EMOTION}")


def main() -> None:
    datasets = [
        d for d in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, d))
        and os.path.exists(os.path.join(DATA_ROOT, d, "metadata.csv"))
    ]
    for ds in datasets:
        write_config(ds)
    print(f"\ntotal configs: {len(datasets)}")


if __name__ == "__main__":
    main()
