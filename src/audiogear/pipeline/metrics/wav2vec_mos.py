from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter


class MosMetric(BaseMetric):
    """Legacy wav2vec2-based MOS (wvmos), kept opt-in.

    DistillMOS (``distillmos.py``) is the default MOS block now; this remains
    for comparison. Requires the `wvmos` package
    (``uv pip install git+https://github.com/AndreevP/wvmos``).
    """

    name = "🌊 Wvmos"
    _requires_dependencies = ("wvmos",)

    def __init__(
        self,
        device: str = "cuda",
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(metric="wv_mos", file_writer=file_writer, file_reader=file_reader)
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from wvmos import get_wvmos

            self._model = get_wvmos(cuda=str(self.device).startswith("cuda"))
        return self._model

    def compute_metric(self, segment: AudioSegment) -> float:
        try:
            return float(self.model.calculate_one(segment.audio_file))
        except Exception as e:  # noqa: BLE001
            print(f"Could not compute wv_mos for {segment.audio_file}: {e}")
            return -1.0
