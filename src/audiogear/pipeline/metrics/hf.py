"""Config-driven base for metrics backed by a 🤗 transformers audio model.

One class, ``HFAudioModelMetric``, covers both **classification** (gender,
emotion, accent, …) and **regression** (MOS, arousal/valence, age, …) audio
models. You pick the model and the output mapping entirely from config — no code
needed for a new model:

    - _target_: audiogear.pipeline.metrics.hf.HFAudioModelMetric
      model_id: alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech
      metric: gender_pred
      mode: classification
      output: label            # top-1 label string

    - _target_: audiogear.pipeline.metrics.hf.HFAudioModelMetric
      model_id: <some-regression-model>
      metric: [arousal, dominance, valence]
      mode: regression         # 3 head outputs -> 3 columns

It loads the model with ``AutoModelForAudioClassification`` + an
``AutoFeatureExtractor`` (the extractor pads a batch and builds the attention
mask, so batched inference is accuracy-preserving). Batching, the process-global
model cache, and the CUDA-OOM ladder (windowed GPU → CPU) are all inherited from
``BaseMetric``. ``GenderMetric`` / ``EmotionMetric`` are thin presets below.
"""

from __future__ import annotations

from audiogear.audio import load_audio
from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model, iter_windows


class HFAudioModelMetric(BaseMetric):
    """A HuggingFace audio model applied per clip, configured from YAML.

    Args:
        model_id: HF hub id or local path.
        metric: output column name (str) or names (tuple, for multi-output).
        mode: ``"classification"`` (logits -> label/score) or ``"regression"``
            (logits -> raw float values).
        output: mapping strategy for classification — ``"label"`` (top label
            str), ``"label_score"`` (``(label, score)`` -> 2-col metric),
            ``"score"`` (top softmax prob), or ``"prob"`` (prob of ``label``).
            Ignored for regression (outputs map to ``metric`` by position).
        label: target class for ``output="prob"``.
        sampling_rate: rate the feature extractor expects (16 kHz for wav2vec2/
            hubert).
        trust_remote_code: pass through for models shipping a custom architecture.
    """

    gpu = True
    supports_batch = True
    _requires_dependencies = ("transformers", "torch")

    def __init__(
        self,
        model_id: str,
        metric,
        mode: str = "classification",
        output: str = "label",
        label: str | None = None,
        device: str = "cuda",
        sampling_rate: int = 16000,
        trust_remote_code: bool = False,
        chunk_seconds: float = 20.0,
        batch_size: int = 16,
        max_batch_seconds: float = 480.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(
            metric=metric,
            file_writer=file_writer,
            file_reader=file_reader,
            chunk_seconds=chunk_seconds,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
        )
        if mode not in ("classification", "regression"):
            raise ValueError(f"mode must be 'classification' or 'regression', got {mode!r}")
        self.model_id = model_id
        self.mode = mode
        self.output = output
        self.label = label
        self.device = device
        self.sampling_rate = sampling_rate
        self.trust_remote_code = trust_remote_code

    # --- model / feature extractor (process-global cache) ---------------------
    def _model_on(self, device: str):
        def build():
            import torch
            from transformers import AutoModelForAudioClassification

            model = AutoModelForAudioClassification.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            return model.to(device).eval() if device != "cpu" else model.to(torch.device("cpu")).eval()

        return cached_model((type(self).__name__, "model", self.model_id, device), build)

    @property
    def extractor(self):
        def build():
            from transformers import AutoFeatureExtractor

            return AutoFeatureExtractor.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )

        return cached_model((type(self).__name__, "extractor", self.model_id), build)

    # --- forward + output mapping --------------------------------------------
    def _forward(self, audios: list, device: str):
        """Run the model on a list of 1-D float waveforms; return logits (B, K)."""
        import torch

        inputs = self.extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            return self._model_on(device)(**inputs).logits  # (B, K)

    def _map(self, logits) -> list:
        """Map a (B, K) logits tensor to one metric value per row."""
        import torch

        if self.mode == "regression":
            vals = logits.float().cpu().tolist()  # list[list[float]]
            if isinstance(self.metric, tuple):
                k = len(self.metric)
                return [tuple(float(row[j]) for j in range(k)) for row in vals]
            return [float(row[0]) for row in vals]

        # classification
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        id2label = self._model_on(self.device).config.id2label
        out = []
        for row in probs:
            top = int(row.argmax())
            label = id2label[top]
            if self.output == "label":
                out.append(label)
            elif self.output == "label_score":
                out.append((label, float(row[top])))
            elif self.output == "score":
                out.append(float(row[top]))
            elif self.output == "prob":
                # probability mass on the configured target class
                idx = next((i for i, lab in id2label.items() if lab == self.label), None)
                if idx is None:
                    raise ValueError(
                        f"output='prob' needs label= one of {list(id2label.values())}, got {self.label!r}"
                    )
                out.append(float(row[idx]))
            else:
                raise ValueError(f"unknown output mapping {self.output!r}")
        return out

    def _load_1d(self, segment: AudioSegment):
        audio, _ = load_audio(segment.audio_file, target_sr=self.sampling_rate, mono=True)
        return audio.squeeze(0).numpy()

    # --- BaseMetric hooks -----------------------------------------------------
    def compute_batch(self, segments: list[AudioSegment]):
        logits = self._forward([self._load_1d(s) for s in segments], self.device)
        return self._map(logits)

    def compute_metric(self, segment: AudioSegment):
        return self.compute_batch([segment])[0]

    def _windowed(self, segment: AudioSegment, device: str):
        # OOM recovery: forward each window alone (bounded VRAM) and mean-pool the
        # logits before mapping — works for both classification and regression.
        audio, _ = load_audio(segment.audio_file, target_sr=self.sampling_rate, mono=True)
        acc = None
        n = 0
        for w in iter_windows(audio, self.sampling_rate, self.chunk_seconds):
            lg = self._forward([w.squeeze(0).numpy()], device)
            acc = lg if acc is None else acc + lg
            n += 1
        if n == 0:  # empty/zero-length clip
            return self._failed_value()
        return self._map(acc / n)[0]

    def compute_metric_recover(self, segment: AudioSegment):
        return self._windowed(segment, self.device)

    def compute_metric_cpu(self, segment: AudioSegment):
        return self._windowed(segment, "cpu")
