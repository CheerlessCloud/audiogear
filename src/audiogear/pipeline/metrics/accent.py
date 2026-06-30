from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter


class AccentMetric(BaseMetric):
    """English-accent identification via a SpeechBrain ECAPA classifier (opt-in).

    English-only (CommonAccent), so it is NOT part of the default Russian
    pipeline; kept for English datasets. Requires ``speechbrain`` (not in the
    default install — ``uv pip install speechbrain``).
    """

    name = "🔮 Accent"
    _requires_dependencies = ("speechbrain",)

    def __init__(
        self,
        model_id: str = "Jzuluaga/accent-id-commonaccent_ecapa",
        device: str = "cuda",
        savedir: str = "pretrained_models/accent-id-commonaccent_ecapa",
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(metric="accent", file_writer=file_writer, file_reader=file_reader)
        self.model_id = model_id
        self.device = device
        self.savedir = savedir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from speechbrain.inference.classifiers import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self.model_id, savedir=self.savedir, run_opts={"device": self.device}
            )
        return self._model

    def compute_metric(self, segment: AudioSegment):
        _, _, _, label = self.model.classify_file(segment.audio_file)
        return label[0]
