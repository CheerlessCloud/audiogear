from pathlib import Path

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.utils.runtime import cached_model


class SnrReverbMetrics(BaseMetric):
    """Speech SNR and C50 reverberation via the Brouhaha regressive VAD.

    Uses the vendored ``RegressiveActivityDetectionPipeline`` (under
    ``metrics/models/brouhaha_vad.py``) on top of pyannote.audio — no separate
    `brouhaha` pip package needed. ``model_path`` may be a local checkpoint or a
    HuggingFace id (default ``pyannote/brouhaha``, which is gated and needs
    ``HF_TOKEN``). Emits mean SNR (dB) and mean C50 (dB; higher = drier).
    """

    name = "🛜 SnrReverb"
    gpu = True
    _requires_dependencies = ("pyannote.audio", "numpy")

    def __init__(
        self,
        model_path: str = "pyannote/brouhaha",
        device: str = "cuda",
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(metric=("snr", "c50"), file_writer=file_writer, file_reader=file_reader)
        self.model_path = model_path
        self.device = device

    def _pipeline_on(self, device: str):
        def build():
            import torch
            from pyannote.audio import Model

            from audiogear.pipeline.metrics.models.brouhaha_vad import RegressiveActivityDetectionPipeline

            # Accept a local path or a HuggingFace model id. HF_TOKEN is read
            # from the environment automatically by huggingface_hub.
            local = Path(self.model_path)
            source = local if local.exists() else self.model_path
            model = Model.from_pretrained(source, strict=False)
            model.to(torch.device(device))
            return RegressiveActivityDetectionPipeline(segmentation=model)

        return cached_model(("SnrReverbMetrics", self.model_path, device), build)

    def _run(self, segment: AudioSegment, device: str):
        import numpy as np

        results = self._pipeline_on(device)(segment.audio_file)
        return float(np.mean(results["snr"])), float(np.mean(results["c50"]))

    def compute_metric(self, segment: AudioSegment):
        from audiogear.utils.runtime import normalize_device

        return self._run(segment, normalize_device(self.device))

    def compute_metric_cpu(self, segment: AudioSegment):
        return self._run(segment, "cpu")
