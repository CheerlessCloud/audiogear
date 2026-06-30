import json
from typing import IO, Callable

from audiogear.io import DataFolderLike
from audiogear.pipeline.writers.base_disk import DiskWriter


class JsonlWriter(DiskWriter):
    default_output_filename: str = "metadata.jsonl"
    name = "🐿 Jsonl"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "gzip",
        adapter: Callable = None,
    ):
        super().__init__(output_folder, output_filename=output_filename, compression=compression, adapter=adapter)

    def _write(self, segment: dict, file_handler: IO, _filename: str):
        file_handler.write(json.dumps(segment, ensure_ascii=False) + "\n")
