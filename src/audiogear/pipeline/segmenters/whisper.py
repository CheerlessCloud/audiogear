
import os
from contextlib import nullcontext
from typing import Callable

from tqdm import tqdm

from audiogear.data import AudioPipeline
from audiogear.io import DataFolderLike, get_datafolder
from audiogear.pipeline.segmenters.base import BaseSegmenter


class WhisperSegmenter(BaseSegmenter):
    _requires_dependencies = ["torchaudio", "faster_whisper", "pandas", "sox"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        output_folder_path: str,
        audio_key: str = "audio_file",
        id_key: str = "id",
        addtional_metadata: dict = None,
        adapter: Callable = None,
        limit: int = -1,
        progress: bool = False,
        get_audio_information: bool = False,
        workers: int = 1,
        recursive: bool = False,
        audio_format: str = "wav",
        whisper_model: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float16",
        target_language: str = "en",
        timestamp_buffer = 0.5,
        punc_list = [".", "!", "?"]

    ):
        super().__init__(
            audio_key, id_key, addtional_metadata, adapter, limit, progress, get_audio_information, workers
        )
        self.data_folder = get_datafolder(data_folder)
        self.audio_format = audio_format
        self.output_folder_path = output_folder_path
        self.target_language = target_language
        self.timestamp_buffer = timestamp_buffer
        self.punc_list = punc_list
        self.recursive = recursive

        from faster_whisper import WhisperModel
        self.whisper_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)

        self.glob_pattern = f"**/*.{self.audio_format}"


    @staticmethod
    def _create_audio_file(wav, sr, out_path, start, end):
        import torchaudio
        torchaudio.backend.sox_io_backend.save(
            out_path,
            wav[int(sr*start):int(sr*end)].unsqueeze(0),
            sr,
            compression=128.0
        )

    def _load_audio(self, audio_path):
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.squeeze()
        return wav, sr

    def _transcribe_audio(self, audio_path):
        print("transcribing")
        segments, _ =  self.whisper_model.transcribe(audio_path, word_timestamps=True, language=self.target_language)
        return segments

    def _create_sentence_split(
        self,
        audio_path,
        wav,
        sr,
        segments,
    ):
        print("splitting")
        sentence_segments = []
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        words_list = []

        for seg_idx, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        for word_idx, word in tqdm(enumerate(words_list)):
            if first_word:
                sentence_start = word.start
                if word_idx == 0:
                    sentence_start = max(sentence_start - self.timestamp_buffer, 0)  # Add buffer to the sentence start
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - self.timestamp_buffer, (previous_word_end + sentence_start)/2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in self.punc_list:
                sentence = sentence[1:]
                audio_file_name, ext = os.path.splitext(os.path.basename(audio_path))
                audio_file = f"{self.output_folder_path}/{audio_file_name}_{str(i).zfill(8)}{ext}"
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    next_word_start = (wav.shape[0] - 1) / sr

                word_end = min((word.end + next_word_start) / 2, word.end + self.timestamp_buffer)
                sentence_start = max(sentence_start - self.timestamp_buffer, 0)

                self._create_audio_file(wav, sr, audio_file, sentence_start, word_end)
                sentence_segments.append(
                    self.get_segment_from_dict(
                        {self.audio_key: audio_file, "format": self.audio_format, "text": sentence, "start_timestamp": sentence_start, "end_timestamp": word_end},
                        audio_path,
                        i
                    )
                )
                i += 1
                first_word = True

        return sentence_segments

    def segment_audio(self, filepath: str) -> AudioPipeline:
        wav, sr = self._load_audio(filepath)
        segments = self._transcribe_audio(filepath)
        sentence_segments = self._create_sentence_split(filepath, wav, sr, segments)
        if self.get_audio_information:
            sentence_segments = self.get_audios_info(sentence_segments)
        return sentence_segments

    def run(self):
        self.audio_files = self.data_folder.get_shard(0, 1, recursive=self.recursive, glob_pattern=self.glob_pattern)
        self.audio_files = [self.data_folder.resolve_paths(file) for file in self.audio_files]
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            segments = []
            for file in self.audio_files:
                segments.extend(self.segment_audio(file))
                if self.limit > 0 and len(segments) >= self.limit:
                    break
                if self.progress:
                    pbar.update()
        return segments
