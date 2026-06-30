"""Effective audio bandwidth via spectral roll-off.

Detects the real upper frequency content of a clip — independent of the file's
container sample rate. Critical for spotting **upsampled** audio (e.g. 16 kHz
content saved in a 44.1 kHz file): such clips have a hard low-pass edge well
below Nyquist. Pure DSP (FFT), no model.

Columns:
  bandwidth_hz      effective upper frequency (Hz) where averaged spectrum
                    last exceeds ``thresh_db`` below its peak
  is_upsampled_est  True if bandwidth_hz < ``upsample_ratio`` * (file_sr/2)
                    (content doesn't fill the available band -> likely upsampled)
"""
import numpy as np

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter


class BandwidthMetric(BaseMetric):
    name = "📶 Bandwidth"
    parallel_cpu = True
    _requires_dependencies = ("librosa", "numpy")

    def __init__(self, thresh_db: float = -50.0, upsample_ratio: float = 0.92,
                 n_fft: int = 2048,
        num_threads: int = -1, file_writer: DiskWriter = None,
                 file_reader: BaseDiskReader = None):
        super().__init__(metric=("bandwidth_hz", "is_upsampled_est"),
                         file_writer=file_writer, file_reader=file_reader, num_threads=num_threads)
        self.thresh_db = thresh_db
        self.upsample_ratio = upsample_ratio
        self.n_fft = n_fft

    def compute_metric(self, segment: AudioSegment):
        import librosa
        # load at NATIVE sample rate (sr=None) — we need the file's real Nyquist
        wav, sr = librosa.load(segment.audio_file, sr=None, mono=True)
        if wav.size < self.n_fft:
            return (0.0, False)
        # averaged magnitude spectrum (power), in dB relative to peak
        S = np.abs(librosa.stft(wav, n_fft=self.n_fft)) ** 2
        psd = S.mean(axis=1)
        psd_db = 10.0 * np.log10(psd + 1e-12)
        peak = psd_db.max()
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sr)
        above = np.where(psd_db >= peak + self.thresh_db)[0]
        bw = float(freqs[above[-1]]) if above.size else 0.0
        is_up = bool(bw < self.upsample_ratio * (sr / 2.0))
        return (round(bw, 1), is_up)
