# Testing & a verified run

## Unit suite (CPU-only, no model downloads)

```bash
uv sync --extra dev
uv run pytest            # 46 tests
```

It targets the plumbing where data silently corrupts: CSV writer/reader
round-trips and column alignment, result↔segment pairing across **every**
scheduling path (batched / prefetch / plain-GPU / parallel-CPU), the runtime
helpers (length bucketing, windowing, OOM detection), and text normalization. No
weights are downloaded, so it runs in a couple of seconds. Qwen tests mock the
optional package and don't download model weights.

## Qwen3 RTX 3060 smoke

Install the dedicated extra in the Python 3.12, Torch 2.8 runtime and validate
both presets before model execution:

```bash
uv sync --python 3.12 --extra dev --extra qwen3
uv pip check
uv run python process.py --config-name annotate_qwen3 --cfg job
uv run python process.py --config-name align_qwen3 --cfg job
```

Run ASR and alignment separately on a 12 GB card. Keep
`executor.workers=1`, `executor.gpus=1`, BF16, and batch size 1. Don't combine
ASR, alignment, or diarization in one invocation because worker-process model
caches retain GPU models until exit.

```bash
uv run python process.py --config-name annotate_qwen3 reader.limit=10
uv run python process.py --config-name align_qwen3 reader.limit=10
uv run python process.py --config-name annotate_qwen3 reader.limit=10 \
  metrics.0.backends.0.model_name_or_path=Qwen/Qwen3-ASR-0.6B
```

The established RTX 3060 baseline used Python 3.12, torch 2.8.0+cu128,
transformers 4.57.6, and qwen-asr 0.0.6. `Qwen/Qwen3-ASR-1.7B` BF16 reserved
about 4492 MiB and returned `сделать заказ с доставкой.`;
`Qwen/Qwen3-ForcedAligner-0.6B` reserved about 1810 MiB.

For ASR, verify ordered one-result-per-input mapping, nonempty Russian speech,
`asr_text_qwen3`/chosen/agreement columns, and byte-for-byte preservation of every
nonempty reference in `text`. Both presets disable rank-only completion skipping
and use separate logging directories; kill/restart validation must show that the
config- and audio-aware per-clip checkpoints resume work instead. For alignment,
include empty text,
Russian punctuation, a corrupt clip, and a partial-checkpoint restart. Parse
every `qwen3_alignment` cell with `json.loads`, require statuses `ok`,
`empty_text`, or `error`, and verify finite monotonic word spans. The alignment
text must be the existing CSV `text`; Qwen hypothesis timestamps must never
replace it. Change the model/revision, audio bytes, reference text, and
CPU/CUDA device between restarts and verify each incompatible row is recomputed.
When testing a pinned Hub revision, confirm qwen-asr receives a local snapshot
path and no `revision` keyword.

## End-to-end run on a small subset (2× RTX 4090)

A self-contained smoke test on 12 synthetic clips (varying length / sample rate,
**some with empty `text`/`speaker_id` and a ragged `bit_rate` column** — the exact
shape that used to corrupt CSV output). Config: a CPU DSP lane
(`bandwidth, pitch, style, wada_snr`) ∥ a GPU lane (`distillmos, squim`).

### Single GPU, 2 shards

```bash
python process.py --config-name feat_minitest executor.tasks=2 executor.workers=1
```

```
🔀 - PARALLEL: [cpu:Bandwidth→Pitch→Style→Snr || gpu:DistillMOS→Squim]
Running 2 lanes concurrently over 6 segments: ['cpu', 'gpu']
... Processing done for rank=0 / rank=1
Pipeline finished.
```

Output validation across the two shards:

```
ext_00000.csv: rows=6 col-counts={20}
ext_00001.csv: rows=6 col-counts={20}
identical header across shards: True
total rows: 12 | out-of-range values: NONE        # distillmos∈[0,5], stoi∈[0,1], pesq∈[1,5]
empty-text row OK -> id=mini/clip_00.wav bit_rate='256000'
                     distillmos=1.58  pyt_si_sdr=-13.85
```

Every row has the same 21 columns, and the ragged empty-text/`bit_rate` row maps
correctly — `pyt_si_sdr=-13.85` sits under its own header, not shifted into
`whisper_wer`. That shift was the original BUG-1 corruption; the
schema-locked `CsvWriter` prevents it. (The model values are produced on a real
GPU forward; the inputs are synthetic tones, so the absolute numbers aren't
meaningful — only their ranges and alignment are.)

### Multi-GPU, 4 shards

```bash
python process.py --config-name feat_minitest executor.tasks=4 executor.workers=2
```

```
Worker local_rank=0 pinned to physical GPU 0
Worker local_rank=1 pinned to physical GPU 1
1/4 ... 4/4 tasks completed.
-> ext_00000.csv .. ext_00003.csv   (4 shards)   logs/.../completions/00000 .. 00003
```

### Multi-node (simulated) + resume

```bash
AUDIOGEAR_NODE_RANK=0 AUDIOGEAR_NUM_NODES=2 python process.py --config-name feat_minitest executor.tasks=4 executor.workers=2
AUDIOGEAR_NODE_RANK=1 AUDIOGEAR_NUM_NODES=2 python process.py --config-name feat_minitest executor.tasks=4 executor.workers=2
```

```
Multi-node: node 0/2 runs ranks [0, 2)
Multi-node: node 1/2 runs ranks [2, 4)
-> all 4 shards in the shared output dir
# re-running node 0:
Not doing anything as all 2 local tasks are already completed.   # resume via shared logging_dir
```

See [multi-node.md](multi-node.md) for the topology details and [storage.md](storage.md)
for putting the shared `logging_dir` (and outputs) in S3.

### Outputs + logs to S3 (live, Yandex Object Storage)

Same run with audio local and the outputs/logs in a real S3-compatible bucket:

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ru-central1
# endpoint via ~/.config/fsspec/conf.json: {"s3":{"client_kwargs":{"endpoint_url":"https://storage.yandexcloud.net"}}}
python process.py --config-name feat_minitest executor.tasks=2 executor.workers=1 \
  reader.data_folder=/data/mini \
  writer.output_folder=s3://my-bucket/ag_test/out \
  executor.logging_dir=s3://my-bucket/ag_test/logs
```

```
# objects created in the bucket:
ag_test/out/ext_00000.csv  ag_test/out/ext_00001.csv
ag_test/logs/completions/00000  .../00001  ag_test/logs/executor.json  ag_test/logs/logs/task_*.log
# reading the output back FROM S3:
identical header across shards: True | total rows: 12 | out-of-range values: NONE
# re-running -> resume off the S3 markers:
Not doing anything as all 2 local tasks are already completed.
```

## Reproducing the synthetic subset

```python
# writes ~/ag_test_data/mini/{metadata.csv, audio/clip_*.wav}
import os, csv, math, struct, wave, random
random.seed(0)
ROOT = os.path.expanduser("~/ag_test_data/mini"); AUD = f"{ROOT}/audio"
os.makedirs(AUD, exist_ok=True)
def tone(path, secs, sr, freq):
    n = int(secs * sr)
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2*math.pi*freq*i/sr))) for i in range(n)))
rows = []
for i in range(12):
    secs = round(random.uniform(1.2, 3.5), 2); sr = random.choice([16000, 22050, 44100])
    tone(f"{AUD}/clip_{i:02d}.wav", secs, sr, random.choice([110, 180, 240]))
    r = {"id": f"mini/clip_{i:02d}.wav", "audio_path": f"audio/clip_{i:02d}.wav",
         "text": (f"пример текста номер {i}" if i % 3 else ""),
         "speaker_id": (f"spk_{i%3}" if i % 2 else ""), "duration": secs, "sample_rate": sr}
    if i % 4 == 0: r["bit_rate"] = "256000"      # ragged column on purpose
    rows.append(r)
cols = ["id","audio_path","text","speaker_id","duration","sample_rate","bit_rate"]
with open(f"{ROOT}/metadata.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="|", extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
```
