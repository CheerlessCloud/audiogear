import os
from typing import Callable, Literal

from audiogear.data import AudioPipeline
from audiogear.io import DataFolderLike
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.utils.progress import tqdm


class AudioFolderReader(BaseDiskReader):
    name = "📁 Folder"
    """This fins all the audios in a single root folder. it can also find and read text files with the same base filename"""

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
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
        find_matching_text_files: bool = False,
        textfile_extension: str = ".txt",
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
        self.find_matching_text_files = find_matching_text_files
        self.textfile_extension = textfile_extension

    def _get_textfile_list(self):
        tl = self.data_folder.list_files(recursive=self.recursive, glob_pattern=f"**/*{self.textfile_extension}")
        self.text_files = [self.data_folder.resolve_paths(f) for f in tl]

    def _match_text_files_list(self, audio_files, text_files):
        files_map = {}
        for audio_file in tqdm(audio_files):
            # base_name = os.path.basename(audio_file).split(".")[0]
            base_name = os.path.basename(audio_file)
            base_name = os.path.splitext(base_name)[0]

            files_map[base_name] = {"audio_file": audio_file, "text": None}

        matched_files = []
        for tf, text_file in enumerate(text_files):
            # base_name = os.path.basename(text_file).split(".")[0]
            base_name = os.path.basename(text_file)
            base_name = os.path.splitext(base_name)[0]

            if base_name in files_map:
                with open(text_file, "r", encoding="utf-8") as text:
                    transcript = text.read().replace("\n", "")
                    files_map[base_name]["text"] = transcript
                    seg = self.get_segment_from_dict(files_map[base_name], files_map[base_name]["audio_file"], tf)
                    matched_files.append(seg)
        return matched_files

    def process_data(self, filepath: str, num: int = 0) -> AudioPipeline:
        """This is the process for if the audio files don't have a corresponding text file"""
        audio_dict = {"audio_file": filepath, "format": self.audio_format}
        return self.get_segment_from_dict(audio_dict, filepath, num)

    def run(self, data: AudioPipeline = None, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        """I needed to rewrite this"""
        if data is not None:
            segments = data
        else:
            segments = []

        self.audio_files = self.data_folder.get_shard(
            rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern
        )
        if self.find_matching_text_files is True:
            self._get_textfile_list()
            segments = self._match_text_files_list(self.audio_files, self.text_files)
        else:
            for file in self.audio_files:
                segments.append(self.process_data(file))
                if self.limit > 0 and len(segments) >= self.limit:
                    break

        if self.get_audio_information:
            segments = self.get_audios_info(segments)
        return segments
