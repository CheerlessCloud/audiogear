from audiogear.pipeline.metrics.hf import HFAudioModelMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter


class EmotionMetric(HFAudioModelMetric):
    """Speech-emotion classification (opt-in).

    A thin preset over :class:`HFAudioModelMetric`. Defaults to a Russian HuBERT
    model fine-tuned on DUSHA; emits the top label and its score into
    ``emotion_pred`` / ``emotion_score``. Swap ``model_id`` for another
    checkpoint (e.g. an English IEMOCAP model) via config.
    """

    name = "🥴 Emotion"

    def __init__(
        self,
        model_id: str = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
        device: str = "cuda",
        batch_size: int = 16,
        max_batch_seconds: float = 480.0,
        chunk_seconds: float = 20.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(
            model_id=model_id,
            metric=("emotion_pred", "emotion_score"),
            mode="classification",
            output="label_score",
            device=device,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
            chunk_seconds=chunk_seconds,
            file_writer=file_writer,
            file_reader=file_reader,
        )
