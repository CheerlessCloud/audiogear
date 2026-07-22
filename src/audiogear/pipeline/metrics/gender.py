from audiogear.pipeline.metrics.hf import HFAudioModelMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter

genderRevision = "7a28165f33e1dbb37adbce09c0a9afcd6095dd4d"
genderAllowPatterns = ("config.json", "preprocessor_config.json", "model.safetensors")


class GenderMetric(HFAudioModelMetric):
    """Binary speaker-gender classification via a wav2vec2 audio classifier.

    A thin preset over :class:`HFAudioModelMetric` (multilingual xlsr-53 backbone,
    transfers to Russian). Emits the top-1 label into ``gender_pred``. Set
    ``model_id`` and its full ``revision`` to use another checkpoint.
    """

    name = "👫 Gender"

    def __init__(
        self,
        model_id: str = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        revision: str = genderRevision,
        local_files_only: bool = False,
        allow_patterns: tuple[str, ...] | list[str] | None = genderAllowPatterns,
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
            revision=revision,
            mode="classification",
            output="label",
            device=device,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
            chunk_seconds=chunk_seconds,
            file_writer=file_writer,
            file_reader=file_reader,
        )
