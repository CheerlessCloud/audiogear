from audiogear.audio import load_audio
from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric


class PitchMetric(BaseMetric):
    """Fundamental-frequency (pitch) statistics: mean and std over voiced frames.

    Mirrors DataSpeech's pitch + "speech monotony" signal. Two backends:

    - ``pyin`` (default): librosa's probabilistic YIN. CPU-only, no extra
      dependencies (librosa is a core dep), robust across all hardware. Good
      enough for pitch *statistics*.
    - ``penn``: the neural `penn` estimator (GPU, more accurate). Needs the
      ``pitch`` extra; note `penn`'s compiled ``torbi`` dependency can be
      fragile against specific torch builds.

    Emits ``pitch_mean`` and ``pitch_std`` (Hz), computed over voiced frames
    only (0.0 if the clip has no voiced content).
    """

    name = "🎵 Pitch"

    parallel_cpu = True

    def __init__(
        self,
        backend: str = "pyin",
        fmin: float = 65.0,
        fmax: float = 1000.0,
        hopsize: float = 0.01,
        gpu: int | None = None,
        model_path: str = None,
        center: str = "half-hop",
        batch_size: int = 1,
        num_threads: int = -1,
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(metric=("pitch_mean", "pitch_std"), file_writer=file_writer, file_reader=file_reader, num_threads=num_threads)
        self.backend = backend
        self.fmin = fmin
        self.fmax = fmax
        self.hopsize = hopsize
        self.gpu = gpu
        self.model_path = model_path
        self.center = center
        self.batch_size = batch_size

    def _compute_pyin(self, segment: AudioSegment):
        import librosa
        import numpy as np

        y, sr = librosa.load(segment.audio_file, sr=None, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr, fmin=self.fmin, fmax=self.fmax, frame_length=2048
        )
        voiced = f0[voiced_flag & ~np.isnan(f0)]
        if voiced.size == 0:
            return 0.0, 0.0
        return float(np.mean(voiced)), float(np.std(voiced))

    def _compute_penn(self, segment: AudioSegment):
        import penn
        import torch

        audio, sr = load_audio(segment.audio_file, mono=True)
        pitch, periodicity = penn.from_audio(
            audio,
            sr,
            hopsize=self.hopsize,
            fmin=self.fmin,
            fmax=self.fmax,
            checkpoint=self.model_path,
            center=self.center,
            gpu=self.gpu,
            batch_size=self.batch_size,
        )
        return torch.mean(pitch).item(), torch.std(pitch).item()

    def compute_metric(self, segment: AudioSegment):
        if self.backend == "penn":
            return self._compute_penn(segment)
        return self._compute_pyin(segment)
