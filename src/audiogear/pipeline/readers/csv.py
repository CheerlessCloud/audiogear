import csv
from typing import Callable, Literal

from audiogear.data import AudioPipeline
from audiogear.io import DataFolderLike
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.utils.progress import tqdm


class CsvReader(BaseDiskReader):
    name = "🔢 Csv"

    def __init__(
        self,
        data_folder: DataFolderLike,
        delimiter: str = ",",
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        audio_key: str = "audio_file",
        audio_format: str = "wav",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = "metadata.csv",
        workers=1,
        get_audio_information=False,
    ):
        super().__init__(
            data_folder,
            limit,
            progress,
            adapter,
            audio_key,
            audio_format,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            workers,
            get_audio_information,
        )
        self.compression = compression
        self.delimiter = delimiter
        self.empty_warning = False

    def process_data(self, filepath: str) -> AudioPipeline:
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            segments = []
            for si, s in tqdm(enumerate(reader)):
                segment = self.get_segment_from_dict(s, filepath, si)
                if segment is not None:
                    segments.append(segment)
                # `limit` caps the number of rows read (handy for dry runs).
                # The base reader only checks `limit` per file, so enforce it here.
                if self.limit > 0 and len(segments) >= self.limit:
                    break
        return segments

    def run(self, data: AudioPipeline = None, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        """Row-level sharding: every rank reads the same metadata file(s) but
        keeps only rows where ``row_index % world_size == rank``.

        File-level sharding (the base reader's default) cannot split a single
        metadata CSV across GPUs/nodes — only rank 0 would receive the file.
        Sharding by row distributes the dataset evenly instead.
        """
        segments: AudioPipeline = data if data is not None else []
        files = self.data_folder.list_files(recursive=self.recursive, glob_pattern=self.glob_pattern)
        for filepath in files:
            with self.data_folder.open(filepath, "r", compression=self.compression) as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                for si, row in enumerate(reader):
                    if si % world_size != rank:
                        continue
                    segment = self.get_segment_from_dict(row, filepath, si)
                    if segment is not None:
                        segments.append(segment)
                    if self.limit > 0 and len(segments) >= self.limit:
                        break
        if self.get_audio_information:
            segments = self.get_audios_info(segments)
        return segments
