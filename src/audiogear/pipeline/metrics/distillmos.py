from audiogear.pipeline.metrics.base import PrefetchGPUMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model


class DistillMosMetric(PrefetchGPUMetric):
    """Reference-free MOS prediction via DistillMOS (ConvTransformer SQA model).

    DistillMOS is a small, pip-installable, no-reference speech-quality model
    that predicts a 1..5 MOS from a 16 kHz waveform. It is the default MOS block
    in audiogear (replacing the heavier NISQA `.tar` / wvmos git dependency) and
    matches the `distillmos` column already present in some datasets.

    Runs in ``prefetch`` mode (decode-ahead + single-thread inference), NOT
    batched: ``segmenting_in_forward=True`` makes the model window+average
    internally, so zero-padding a mixed-length batch would inject silent segments
    and bias the MOS. Long clips that OOM are recovered window-by-window and the
    per-window MOS averaged (inherited from ``PrefetchGPUMetric``).
    """

    name = "⭐ DistillMOS"
    sample_rate = 16000
    _requires_dependencies = ("distillmos", "torch", "torchaudio")

    def __init__(
        self,
        device: str = "cuda",
        chunk_seconds: float = 20.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(
            metric="distillmos",
            device=device,
            chunk_seconds=chunk_seconds,
            file_writer=file_writer,
            file_reader=file_reader,
        )

    def _model_on(self, device: str):
        def build():
            import distillmos

            model = distillmos.ConvTransformerSQAModel(load_weights=True, segmenting_in_forward=True)
            return model.to(device).eval()

        return cached_model(("DistillMosMetric", device), build)

    def _run(self, audio, device: str) -> float:
        import torch

        audio = audio.to(device)  # shape (1, samples) == (batch=1, seq_len)
        with torch.no_grad():
            mos = self._model_on(device)(audio)
        return float(mos.squeeze().item())
