import math
import wave
from array import array

import torch

from audiogear.audio import audio_duration, load_audio


def _write_wav(path, channels=1, sample_rate=16000, frame_count=1600):
    samples = array("h")
    for frame in range(frame_count):
        sample = round(math.sin(2 * math.pi * 440 * frame / sample_rate) * 16000)
        samples.extend([sample] * channels)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def test_load_audio_decodes_pcm_with_soundfile(tmp_path):
    path = tmp_path / "mono.wav"
    _write_wav(path)

    waveform, sample_rate = load_audio(str(path))

    assert sample_rate == 16000
    assert waveform.shape == (1, 1600)
    assert waveform.dtype == torch.float32
    assert -1 <= waveform.min() <= waveform.max() <= 1
    assert audio_duration(str(path)) == 0.1


def test_load_audio_downmixes_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_wav(path, channels=2)

    waveform, sample_rate = load_audio(str(path), mono=True)

    assert sample_rate == 16000
    assert waveform.shape == (1, 1600)
