import json
from typing import Callable, Literal

from audiogear.data import AudioPipeline
from audiogear.io import DataFolderLike
from audiogear.pipeline.readers.base import BaseDiskReader


class JsonlReader(BaseDiskReader):
    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        audio_key: str = "text",
        audio_format: str = "wav",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        workers: int = 1,
        get_audio_information: bool = False,
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

    def process_data(self, filepath: str) -> AudioPipeline:
        segments = []
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                for li, line in enumerate(f):
                    try:
                        segment = self.get_segment_from_dict(json.loads(line), filepath, li)
                        if segment is None:
                            print(segment)
                            continue
                    except json.JSONDecodeError as e:
                        print(f"Error reading line {li} in {filepath}: {e}")
                        continue
                    segments.append(segment)
            except UnicodeDecodeError as e:
                print(f"Error reading {filepath}: {e}")
        return segments
