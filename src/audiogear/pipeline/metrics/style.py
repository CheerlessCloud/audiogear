import numpy as np

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric


class StyleMetric(BaseMetric):
    """Model-free speaking-style / expressiveness descriptors.

    A lightweight, dependency-light block (librosa only) that captures prosodic
    dynamics useful for TTS data tagging and as a worked example of writing a
    custom metric (see the README extension guide). Computes:

    - ``energy_db``: mean loudness (dBFS-ish, RMS in dB).
    - ``energy_dynamics``: std of frame loudness (how much the volume varies).
    - ``expressiveness``: pitch-variation coefficient (std/mean of voiced f0) —
      a rough monotone↔expressive axis (DataSpeech's "speech monotony").

    Note: if a PitchMetric has already populated ``pitch_mean``/``pitch_std``,
    expressiveness reuses them to avoid recomputing pitch.
    """

    name = "🎭 Style"

    parallel_cpu = True
    _requires_dependencies = ("librosa", "numpy")

    def __init__(self, fmin: float = 65.0, fmax: float = 1000.0, num_threads: int = -1, file_writer=None, file_reader=None):
        super().__init__(
            metric=("energy_db", "energy_dynamics", "expressiveness"),
            file_writer=file_writer,
            file_reader=file_reader,
            num_threads=num_threads,
        )
        self.fmin = fmin
        self.fmax = fmax

    def compute_metric(self, segment: AudioSegment):
        import librosa

        y, sr = librosa.load(segment.audio_file, sr=None, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        energy_db = float(np.mean(rms_db))
        energy_dynamics = float(np.std(rms_db))

        pm = segment.metadata.get("pitch_mean")
        ps = segment.metadata.get("pitch_std")
        if pm is None or ps is None:
            f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=self.fmin, fmax=self.fmax)
            v = f0[voiced & ~np.isnan(f0)]
            pm = float(np.mean(v)) if v.size else 0.0
            ps = float(np.std(v)) if v.size else 0.0
        expressiveness = float(ps / pm) if pm and pm > 0 else 0.0
        return energy_db, energy_dynamics, expressiveness
