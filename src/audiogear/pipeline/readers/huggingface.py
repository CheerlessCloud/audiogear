from contextlib import nullcontext
from typing import Callable

from tqdm import tqdm

from audiogear.data import AudioPipeline
from audiogear.pipeline.readers.base import BaseReader


class HuggingfaceReader(BaseReader):
    name = "🤗 Huggingface"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        dataset: str,
        subset: str | None = None,
        dataset_options: dict | None = None,
        limit: int = -1,
        batch_size: int = 1000,
        progress: bool = False,
        adapter: Callable = None,
        audio_key: str = "audio_file",
        id_key: str = "id",
        default_metadata: dict = None,
        get_audio_information: bool = False,
    ):
        super().__init__(progress, audio_key, None, id_key, adapter, default_metadata)
        self.dataset = dataset
        self.dataset_options = dataset_options
        self.limit = limit
        self.batch_size = batch_size
        self.subset = subset
        self.get_audio_information = get_audio_information

    def get_segment_from_dict(self, data: dict, source: str, id_in_file: int | str):
        document = super().get_segment_from_dict(data, source, id_in_file)
        if document is not None:
            document.metadata.setdefault("dataset", source)
        return document

    def run(self, data: AudioPipeline = None, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        from datasets import load_dataset

        if data is not None:
            segments = data
        else:
            segments = []

            if self.subset is not None:
                ds = load_dataset(self.dataset, self.subset, **self.dataset_options)
            else:
                ds = load_dataset(self.dataset, **self.dataset_options)

        shard = ds.shard(world_size, rank, contiguous=True)
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            li = 0
            for batch in shard.iter(self.batch_size):
                if self.limit != -1 and li >= self.limit:
                    break
                for line in (dict(zip(batch, t)) for t in zip(*batch.values())):
                    if self.limit != -1 and li >= self.limit:
                        break
                    line[self.audio_key] = line["audio"]["path"]
                    # line["audio_array"] = line["audio"]["array"] # this kills your ram
                    line["sample_rate"] = line["audio"]["sampling_rate"]
                    del line["audio"]
                    segment = self.get_segment_from_dict(line, self.dataset, f"{rank:05d}/{li}")
                    segments.append(segment)
                    li += 1
                    if self.progress:
                        pbar.update()

        if self.get_audio_information:
            segments = self.get_audios_info(segments)
        return segments
