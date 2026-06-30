# examples — processing a whole collection of datasets

`process.py --config-name <name>` handles a single dataset. When you have *many*
datasets to annotate, these three scripts show the end-to-end pattern. They are
**templates to copy and adapt**, not a stable CLI — all paths come from
`AUDIOGEAR_DATA_DIR`, and the per-dataset profiles are plain Python sets you edit.

Expected data layout (see the main README, "Dataset structure"):

```
$AUDIOGEAR_DATA_DIR/
  <dataset>/metadata.csv
  <dataset>/audio/...
```

## Workflow

```bash
export AUDIOGEAR_DATA_DIR=/path/to/data_root

# 1. Generate a per-dataset config (configs/feat_<ds>.yaml) for each dataset,
#    picking the metric set from the profiles at the top of the script.
python examples/gen_configs.py

# 2. Run everything smallest-first, with resume; merges per-shard CSVs into
#    <root>/<ds>/extended_metadata.csv. Launch detached for long runs:
setsid nohup python examples/run_batch.py > run_batch.log 2>&1 < /dev/null &

# 3. Filter each extended_metadata.csv down to a clean subset.
python examples/filter_clean.py
```

## The scripts

| Script | What it does | Edit before running |
|--------|--------------|---------------------|
| `gen_configs.py` | One `feat_<ds>.yaml` per dataset; chooses metrics per profile (which sets have human text, need transcription, want gender/emotion, lack speaker ids). | The profile sets (`NO_TEXT`, `HUMAN_TEXT`, `WANT_GENDER`, …). |
| `run_batch.py` | Runs each `feat_<ds>` config, merges shards, writes a `.FEAT_DONE` marker, skips finished datasets on rerun. | Merge target / markers if your storage differs. |
| `filter_clean.py` | `extended_metadata.csv` → `clean_metadata.csv` per dataset, logging drop reasons. | The thresholds in `TH` and the `HUMAN_TEXT` / `ASR_TEXT` sets. |

## Why these choices

- **Smallest-first** surfaces config mistakes in seconds instead of after a
  multi-hour giant.
- **Resume by marker + existing output** means a crash mid-collection costs only
  the in-flight dataset.
- **WER only on human text.** If `text` itself came from Whisper, WER≈0 — skipping
  it on auto-transcribed sets saves hours and avoids a meaningless column.
- **Profiles, not flags.** The metric set per dataset is data, kept in one place,
  so the collection's policy is auditable at a glance.
