from abc import abstractmethod
from typing import Callable

from tqdm.contrib.concurrent import process_map

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep


class BaseSegmenter(PipelineStep):
    """
        A very general class for constructing a segmentation pipeline.
        just does some setting up like getting the root data path
        and handling some general formatting that all segmenters will need
    """
    def __init__(
        self,
        audio_key: str = "audio_file",
        id_key: str = "id",
        additional_metadata: dict = None,
        adapter: Callable = None,
        limit: int = -1,
        progress: bool = False,
        get_audio_information: bool = False,
        workers: int = 1,

    ):
        self.audio_key = audio_key
        self.id_key = id_key
        self.additional_metadata = additional_metadata
        self.adapter = adapter if adapter is not None else self._default_adapter
        self.limit = limit
        self.progress = progress
        self.get_audio_information = get_audio_information
        self.workers = workers

    def _default_adapter(self, data: dict, path: str, id_in_file: int | str):
        """
        The default data adapter to adapt input data into the datatrove Document format

        Args:
            data: a dictionary with the "raw" representation of the data
            path: file path or source for this sample
            id_in_file: its id in this particular file or source

        Returns: a dictionary with text, id, media and metadata fields

        """
        return {
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "audio_file": data.pop(self.audio_key, ""),
            "sample_rate": data.pop("sample_rate", None),
            "channels": data.pop("channels", None),
            "format": data.pop("format", self.audio_format),
            "bit_rate": data.pop("bit_rate", None),
            "duration": data.pop("duration", None),
            "text": data.pop("text", ""),
            "metadata": data.pop("metadata", {}) | data,  # remaining data goes into metadata
        }

    def get_segment_from_dict(self, data: dict, source_audio: str, id_in_file: int | str) -> AudioSegment:
        parsed_data = self.adapter(data, source_audio, id_in_file)
        if not parsed_data.get("audio_file", None):
            if not self._empty_warning:
                self._empty_warning = True
                print(f"Found segment without audio file, skipping. " f'Is your `audio_key` ("{self.audio_key}") correct?')
            return None
        segment = AudioSegment(**parsed_data)
        if self.additional_metadata is not None:
            segment.metadata = self.additional_metadata
        return segment

    def audio_info_wrapper(self, sample: AudioSegment) -> AudioSegment:
        sample.get_audio_info()
        return sample

    def get_audios_info(self, data: AudioPipeline) -> AudioPipeline:
        return process_map(self.audio_info_wrapper, data, max_workers=self.workers, chunksize=self.workers // 2)


    @abstractmethod
    def segment_audio(self, filepath: str) -> AudioPipeline:
        """
            This is the function that should wrap all the steps to segment one audio file
            this should also return an AudioPipeline of all the segments that the file created
            the final `run` function should combine all the list of AudioPipelines into one
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    @abstractmethod
    def run(self) -> AudioPipeline:
        """
            This needs to return a list of AudioSegments/AudioPipeline
        """
        raise NotImplementedError("This method should be implemented in a subclass")
