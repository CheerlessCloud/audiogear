import dataclasses
from abc import ABC, abstractmethod
from string import Template
from typing import IO, Callable

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.io import DataFolderLike, get_datafolder
from audiogear.pipeline.base import PipelineStep


class DiskWriter(PipelineStep, ABC):
    default_output_filename: str = None
    type = "💽 - WRITER"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "infer",
        adapter: Callable = None,
        mode: str = "wt",
        expand_metadata: bool = False,
        max_file_size: int = -1
    ):
        super().__init__()
        self.compression = compression
        self.output_folder = get_datafolder(output_folder)
        output_filename = output_filename or self.default_output_filename
        if self.compression == "gzip" and not output_filename.endswith(".gz"):
            output_filename += ".gz"
        self.max_file_size = max_file_size
        self.output_filename = Template(output_filename)
        self.output_mg = self.output_folder.get_output_file_manager(mode=mode, compression=compression)
        self.adapter = adapter if adapter else self._default_adapter
        self.expand_metadata = expand_metadata

    def _default_adapter(self, segment: AudioSegment) -> dict:
        data = {key: val for key, val in dataclasses.asdict(segment).items() if val}
        if self.expand_metadata and "metadata" in data:
            data |= data.pop("metadata")

        del data['path'] # remove this because only needed it to remember the original relative path
        return data

    def __enter__(self):
        return self

    def close(self):
        self.output_mg.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_output_filename(self, segment: AudioSegment, rank: int | str = 0, **kwargs) -> str:
        return self.output_filename.substitute(
            {"rank": str(rank).zfill(5), "id": segment.id, **segment.metadata, **kwargs}
        )

    @abstractmethod
    def _write(self, segment: dict, file_handler: IO, filename: str):
        raise NotImplementedError

    def write(self, segment: AudioSegment, rank: int = 0, **kwargs):
        # forward rank so a "${rank}" in the filename template yields one output
        # file per shard (required when running with workers>1 / multi-node).
        original_name = output_filename = self._get_output_filename(segment, rank=rank, **kwargs)
        # we possibly have to change file
        if self.max_file_size > 0:
            # get size of current file
            output_filename = self._get_filename_with_file_id(original_name)
            # we have to switch file!
            if self.output_mg.get_file(output_filename).tell() >= self.max_file_size:
                self.file_id_counter[original_name] += 1
                new_output_filename = self._get_filename_with_file_id(original_name)
                self._on_file_switch(original_name, output_filename, new_output_filename)
                output_filename = new_output_filename
        # actually write
        segment.audio_file = segment.path # restore audio file to relative path
        self._write(self.adapter(segment), self.output_mg.get_file(output_filename), original_name)


    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        with self:
            for segment in data:
                self.write(segment, rank)
            return data
