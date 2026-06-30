# Remote storage (S3 / GCS / fsspec)

audiogear's table I/O goes through [fsspec](https://filesystem-spec.readthedocs.io/),
so paths can be object-store URLs. **One caveat decides the layout** (verified, see
below): the metadata table, the output CSVs, and the logging/checkpoint dir can be
remote, but the **audio is decoded with torchaudio's local loader and cannot be
read from `s3://` directly.**

## What can be remote

| Path | Remote (`s3://`, `gcs://`, …)? | Why |
|---|---|---|
| `reader.data_folder` (the `metadata.csv`) | ✅ | read via fsspec |
| `writer.output_folder` (result CSVs) | ✅ | written via fsspec |
| `executor.logging_dir` (completion markers, `executor.json`) | ✅ | written via fsspec |
| the **audio files** themselves | ❌ direct | `torchaudio.load()` is local-path only |

### Verified

Driving the same code path with fsspec's in-memory backend (`memory://`, no
credentials needed):

```
reader over a non-local fsspec FS (memory://):  read 12 segments    ✅
CsvWriter over a non-local fsspec FS:           ext_00000.csv written ✅
completion marker over a non-local logging_dir: isfile -> True       ✅
torchaudio decode of a non-local path:          RuntimeError "Protocol not found" ❌
```

So the audio-decode limitation is real and the table/log I/O genuinely works over
a non-local filesystem.

**Live run against real object storage (Yandex Object Storage, S3-compatible):**
with audio on local disk and `writer.output_folder` + `executor.logging_dir`
pointed at `s3://…`, a 2-shard run wrote the result CSVs, completion markers,
`executor.json`, and task logs to the bucket; the output read back from S3 had
identical headers across shards, all metric values in range, and the ragged
empty-text row correctly aligned. Re-running skipped everything via the markers in
the S3 `logging_dir` (resume), confirming the recommended layout below end to end.

## Recommended layout

Keep the **audio local** (or FUSE-mounted) and stream **outputs + logs to S3**:

```yaml
reader:
  data_folder: /data/dataset                 # local disk, or a FUSE mount of the bucket
writer:
  output_folder: s3://my-bucket/out/dataset
executor:
  logging_dir: s3://my-bucket/logs/dataset   # shared -> cross-node / cross-rerun resume
```

To read audio *from* a bucket, mount it with FUSE
(`geesefs` / `goofys` / `s3fs-fuse`) and point `reader.data_folder` at the mount —
then every path, audio included, is "local" as far as torchaudio is concerned.

## Install & credentials

```bash
uv sync --extra ru-pipeline --extra s3        # adds s3fs (sync all extras together)

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=ru-central1          # your bucket's region
```

Credentials follow the standard AWS resolution chain (env vars,
`~/.aws/credentials`, instance role).

### Custom / S3-compatible endpoint (Yandex, MinIO, Ceph)

The plain `s3://` string in a config builds an `S3FileSystem` with no extra
kwargs, so the endpoint must come from the environment or fsspec config. Most
robust for the config-driven path is an fsspec config file
`~/.config/fsspec/conf.json`:

```json
{
  "s3": {
    "client_kwargs": { "endpoint_url": "https://storage.yandexcloud.net",
                       "region_name": "ru-central1" }
  }
}
```

(`AWS_ENDPOINT_URL` also works with recent botocore.) Programmatically you can
instead hand `get_datafolder` a `(url, storage_options)` tuple:

```python
from audiogear.io import get_datafolder
df = get_datafolder(("s3://bucket/ds",
                     {"client_kwargs": {"endpoint_url": "https://storage.yandexcloud.net"}}))
```

### Sanity-check the connection

```python
import s3fs
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://storage.yandexcloud.net",
                                      "region_name": "ru-central1"})
print(fs.ls("my-bucket"))
with fs.open("my-bucket/probe.txt", "w") as f: f.write("ok")
print(fs.cat("my-bucket/probe.txt").decode()); fs.rm("my-bucket/probe.txt")
```

A `PermissionError: The request signature we calculated does not match` means the
**secret key (or region) is wrong** — the request reached the endpoint and was
rejected at auth, so connectivity/endpoint are fine.
