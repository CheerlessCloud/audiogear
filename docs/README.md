# audiogear documentation

Operational guides that go beyond the top-level [README](../README.md).

- [Distributed runs (multi-GPU & multi-node)](multi-node.md) — sharding, GPU
  pinning, the env-driven node topology, and resume.
- [Remote storage (S3 / fsspec)](storage.md) — what can live in object storage,
  the audio-decode caveat, credentials, and custom endpoints.
- [Testing & a verified run](testing.md) — the unit suite, and an end-to-end run
  on a small subset (2× RTX 4090) with the output validated.

All three were exercised on real hardware; the commands and outputs in them are
copied from actual runs, not invented.
