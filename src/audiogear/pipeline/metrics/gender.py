from audiogear.pipeline.metrics.hf import HFAudioModelMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter


class GenderMetric(HFAudioModelMetric):
    """Binary speaker-gender classification via a wav2vec2 audio classifier.

    A thin preset over :class:`HFAudioModelMetric` (multilingual xlsr-53 backbone,
    transfers to Russian). Emits the top-1 label into ``gender_pred``. Swap
    ``model_id`` to use any other audio-classification checkpoint.
    """

    name = "👫 Gender"

    def __init__(
        self,
        model_id: str = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        device: str = "cuda",
        batch_size: int = 16,
        max_batch_seconds: float = 480.0,
        chunk_seconds: float = 20.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(
            model_id=model_id,
            metric="gender_pred",
            mode="classification",
            output="label",
            device=device,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
            chunk_seconds=chunk_seconds,
            file_writer=file_writer,
            file_reader=file_reader,
        )
