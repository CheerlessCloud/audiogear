from abc import abstractmethod
from itertools import chain
from typing import Callable

from tqdm.contrib.concurrent import process_map

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.io import DataFolderLike, get_datafolder
from audiogear.pipeline.base import PipelineStep


class BaseReader(PipelineStep):
    type = "📖 - READER"

    def __init__(
        self,
        progress: bool = False,
        audio_key: str = "audio_file",
        audio_format: str = "wav",
        id_key: str = "id",
        adapter: Callable = None,
        additional_metadata: dict = None,
    ):
        self.progress = progress
        self.audio_key = audio_key
        self.audio_format = audio_format
        self.id_key = id_key
        self.adapter = adapter if adapter is not None else self._default_adapter
        self.additional_metadata = additional_metadata
        self._empty_warning = False

    def _default_adapter(self, data: dict, path: str, id_in_file: int | str):
        """
        The default data adapter to adapt input data into the datatrove Document format

        Args:
            data: a dictionary with the "raw" representation of the data
            path: file path or source for this sample
            id_in_file: its id in this particular file or source

        Returns: a dictionary with text, id, media and metadata fields

        """
         # Attempt to convert metadata from string to dict if necessary
        import ast

        metadata = data.pop("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = ast.literal_eval(metadata)
            except (ValueError, SyntaxError) as e:
                print(f"Error converting metadata string to dict: {e}")
                metadata = {}  # Default to empty dict in case of error

        def _num(v, cast):
            # CSV values arrive as strings; cast numeric fields so downstream
            # metrics (librosa.load(sr=...), duration filters) get real numbers.
            if v is None or v == "":
                return None
            try:
                return cast(v)
            except (ValueError, TypeError):
                return None

        return {
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "audio_file": data.pop(self.audio_key, ""),
            "path": data.pop("path", ""),
            "sample_rate": _num(data.pop("sample_rate", None), int),
            "channels": _num(data.pop("channels", None), int),
            "format": data.pop("format", self.audio_format),
            "bit_rate": data.pop("bit_rate", None),
            "duration": _num(data.pop("duration", None), float),
            "text": data.pop("text", ""),
            "metadata": data.pop("metadata", {}) | data,  # remaining data goes into metadata
        }

    def get_segment_from_dict(self, data: dict, source_audio: str, id_in_file: int | str) -> AudioSegment:
        parsed_data = self.adapter(data, source_audio, id_in_file)
        if not parsed_data.get("audio_file", None):
            if not self._empty_warning:
                self._empty_warning = True
                print(
                    f"Found segment without audio file, skipping. "
                    f'Is your `audio_key` ("{self.audio_key}") correct?'
                )
            return None
        segment = AudioSegment(**parsed_data)
        if self.additional_metadata is not None:
            segment.metadata = self.additional_metadata
        return segment

    @abstractmethod
    def run(self, data: AudioPipeline = None, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        raise NotImplementedError("This method should be implemented in a subclass")


class BaseDiskReader(BaseReader):
    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        audio_key: str = "audio_file",
        audio_format: str = "wav",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        workers: int = 1,
        get_audio_information: bool = False,
    ):
        super().__init__(progress, audio_key, audio_format, id_key, adapter, default_metadata)
        self.data_folder = get_datafolder(data_folder)
        self.limit = limit
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.workers = workers
        self.get_audio_information = get_audio_information

    def get_segment_from_dict(self, data: dict, source_file: str, id_in_file: int | str) -> AudioSegment:
        data['path'] = data[self.audio_key] # get the placeholder to the relative path
        x = self.data_folder.resolve_paths(data[self.audio_key])  # get the full path to the audio file
        data[self.audio_key] = x
        document = super().get_segment_from_dict(data, source_file, id_in_file)
        return document

    def audio_info_wrapper(self, sample: AudioSegment) -> AudioSegment:
        sample.get_audio_info()
        return sample

    def get_audios_info(self, data: AudioPipeline) -> AudioPipeline:
        return process_map(self.audio_info_wrapper, data, max_workers=self.workers, chunksize=self.workers // 2)

    @abstractmethod
    def process_data(self, data: str) -> AudioPipeline:
        """data should either be a string path to a metadat file or list of audio files"""
        raise NotImplementedError("This method should be implemented in a subclass")

    def run(self, data: AudioPipeline = None, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        if data is not None:
            segments = data
        else:
            segments = []

        self.files = self.data_folder.get_shard(
            rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern
        )
        for file in self.files:
            segments.append(self.process_data(file))
            if self.limit > 0 and len(segments) >= self.limit:
                break

        combined_segments = list(chain.from_iterable(segments))
        if self.get_audio_information:
            combined_segments = self.get_audios_info(combined_segments)

        return combined_segments
