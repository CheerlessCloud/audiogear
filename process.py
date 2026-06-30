#!/usr/bin/env python
"""Entry point for processing a dataset with audiogear.

Pick a dataset config from ``configs/`` by name:

    uv run python process.py --config-name resd
    uv run python process.py --config-name resd reader.limit=10          # dry run
    uv run python process.py --config-name resd executor.tasks=16 executor.workers=2
    uv run python process.py --config-name resd --cfg job                # print resolved config

The chosen config declares the reader, the ordered ``metrics`` list, the writer,
and the executor; everything is resolved via ``hydra.utils.instantiate`` in
``audiogear.build``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from audiogear.build import build_and_run

    build_and_run(cfg)


if __name__ == "__main__":
    main()
