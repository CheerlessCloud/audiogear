from audiogear.pipeline.metrics.base import PrefetchGPUMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model


class SquimMetrics(PrefetchGPUMetric):
    """Reference-free objective speech-quality metrics via torchaudio SQUIM.

    Produces STOI (intelligibility), PESQ (perceptual quality) and SI-SDR.
    Input is resampled to 16 kHz mono, which SQUIM requires.

    Runs in ``prefetch`` mode (decode-ahead + single-thread inference), NOT
    batched: the SQUIM objective model takes no attention mask, so zero-padding a
    mixed-length batch shifts the scores of the shorter clips (verified on real
    weights: PESQ off by ~1.2, SI-SDR by ~5 dB even with tight length buckets).
    Long clips that OOM are recovered window-by-window and the per-window scores
    averaged (inherited from ``PrefetchGPUMetric``).
    """

    name = "🦑 Squim"
    sample_rate = 16000
    _requires_dependencies = ("torch", "torchaudio")

    def __init__(
        self,
        device: str = "cuda",
        chunk_seconds: float = 20.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(
            metric=("pyt_stoi", "pyt_pesq", "pyt_si_sdr"),
            device=device,
            chunk_seconds=chunk_seconds,
            file_writer=file_writer,
            file_reader=file_reader,
        )

    def _model_on(self, device: str):
        def build():
            from torchaudio.pipelines import SQUIM_OBJECTIVE

            return SQUIM_OBJECTIVE.get_model().to(device)

        return cached_model(("SquimMetrics", device), build)

    def _failed_value(self):
        # -1 = "clip could not be scored" (too short, corrupt, model failure —
        # the base per-clip guard routes unexpected errors here). Matches the
        # convention in already-computed datasets.
        return -1.0, -1.0, -1.0

    def _run(self, audio, device: str):
        import torch

        # SQUIM's conv stack needs a minimum input length; ultra-short/empty clips
        # (~2 ms) crash it ("kernel size can't be greater than actual input size").
        # Cheap semantic pre-check — such clips are unscorable, not an error.
        if audio.shape[-1] < 1024:
            return self._failed_value()
        audio = audio.to(device)
        with torch.no_grad():
            stoi, pesq, si_sdr = self._model_on(device)(audio)
        return float(stoi), float(pesq), float(si_sdr)
