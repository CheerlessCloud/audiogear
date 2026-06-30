"""Speaker labeling for datasets that lack speaker IDs.

This is a *dataset-level* block: it embeds every clip, clusters the embeddings,
and assigns a speaker id to each clip — but only when it is confident, to avoid
the costly error of merging two speakers under one id (or splitting one speaker).

Confidence-thresholded assignment (precision-first):
  A clip is assigned to its cluster only if
    (1) cosine similarity to the cluster centroid >= ``assign_threshold``  AND
    (2) the margin between the best and second-best centroid similarity
        >= ``margin``.
  Otherwise the clip is labeled ``"unknown"``. ``speaker_conf`` (centroid
  similarity) and ``speaker_margin`` are emitted so downstream filtering can
  drop low-confidence labels or re-tune thresholds without recomputing.

**Multi-GPU embedding with a barrier.** Embedding is per-clip and embarrassingly
parallel, but clustering needs *all* embeddings together. So the work is split:
each shard embeds its own clips on its own GPU (run with ``tasks == workers ==
n_gpus`` so all shards run at once), writes them to ``cache_dir``, then every
shard waits at a filesystem barrier until all shards are done, loads the *global*
embedding set, and runs the SAME deterministic clustering — so speaker ids are
globally consistent across shards even though the GPU work was parallel. With a
single shard (``world_size == 1``) it just embeds + clusters in place, no barrier.

Embedding model: ``pyannote/wespeaker-voxceleb-resnet34-LM`` (not gated; loaded
via pyannote so it is unaffected by the transformers torch.load gate).
"""

from __future__ import annotations

import os
import time

from loguru import logger

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.utils.progress import tqdm
from audiogear.utils.runtime import cached_model, free_cuda, is_oom_error


class SpeakerLabeler(PipelineStep):
    type = "🔻 - METRIC"
    name = "🧑‍🤝‍🧑 SpeakerLabeler"
    _requires_dependencies = ("pyannote.audio", "sklearn", "numpy")

    def __init__(
        self,
        embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        device: str = "cuda",
        distance_threshold: float = 0.5,
        assign_threshold: float = 0.75,
        margin: float = 0.10,
        prefix: str = "spk",
        cache_dir: str | None = None,
        barrier_timeout: float = 7200.0,
        file_writer=None,
        file_reader=None,
    ):
        """
        cache_dir: directory shared by all shards where per-shard embeddings are
            staged for the barrier. REQUIRED when running multi-shard
            (``tasks > 1``); ignored for a single shard.
        barrier_timeout: max seconds to wait for the other shards before erroring
            out (instead of hanging forever).
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.device = device
        self.distance_threshold = distance_threshold
        self.assign_threshold = assign_threshold
        self.margin = margin
        self.prefix = prefix
        self.cache_dir = cache_dir
        self.barrier_timeout = barrier_timeout
        self.file_writer = file_writer
        self.file_reader = file_reader

    def _inference_on(self, device: str):
        def build():
            import torch
            from pyannote.audio import Inference, Model

            # huggingface_hub reads HF_TOKEN from the environment automatically
            # (loaded from .env at import); newer hub rejects use_auth_token here.
            model = Model.from_pretrained(self.embedding_model)
            model.to(torch.device(device))
            return Inference(model, window="whole")

        return cached_model(("SpeakerLabeler", self.embedding_model, device), build)

    def _embed_on(self, audio_file: str, device: str):
        import numpy as np

        emb = np.asarray(self._inference_on(device)(audio_file)).reshape(-1)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _embed(self, audio_file: str):
        # window="whole" embeds the entire clip at once, so a very long file can
        # OOM. Fall back to CPU for that clip rather than killing the whole run.
        try:
            return self._embed_on(audio_file, self.device)
        except Exception as e:
            if not is_oom_error(e):
                raise
            free_cuda()
            logger.warning(f"CUDA OOM embedding {audio_file} — retrying on CPU")
            return self._embed_on(audio_file, "cpu")

    def _embed_shard(self, data: AudioPipeline):
        import numpy as np

        ids = [s.id for s in data]
        emb = np.stack([self._embed(s.audio_file) for s in tqdm(data, desc="embedding")])
        return ids, emb

    # --- filesystem barrier (shared cache_dir) -------------------------------
    def _emb_path(self, rank: int) -> str:
        return os.path.join(self.cache_dir, f"emb_{rank:05d}.npz")

    def _done_path(self, rank: int) -> str:
        return os.path.join(self.cache_dir, f"done_{rank:05d}")

    def _write_shard(self, rank: int, ids, emb):
        import numpy as np

        os.makedirs(self.cache_dir, exist_ok=True)
        tmp = self._emb_path(rank) + ".tmp"
        # write via a file handle so np.savez doesn't re-append ".npz" to tmp
        with open(tmp, "wb") as f:
            np.savez(f, ids=np.array(ids, dtype=object), emb=emb)
        os.replace(tmp, self._emb_path(rank))  # atomic publish, then the marker
        open(self._done_path(rank), "w").close()

    def _wait_for_all(self, world_size: int):
        deadline = time.monotonic() + self.barrier_timeout
        pending = set(range(world_size))
        while pending:
            pending = {r for r in pending if not os.path.exists(self._done_path(r))}
            if not pending:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"SpeakerLabeler barrier timed out after {self.barrier_timeout}s "
                    f"waiting for shards {sorted(pending)} in {self.cache_dir}"
                )
            time.sleep(2.0)

    def _load_all(self, world_size: int):
        import numpy as np

        ids, embs = [], []
        for r in range(world_size):
            d = np.load(self._emb_path(r), allow_pickle=True)
            ids.extend(d["ids"].tolist())
            embs.append(d["emb"])
        emb = np.concatenate(embs, axis=0)
        # deterministic global order by id -> identical clustering on every shard
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        return [ids[i] for i in order], emb[order]

    # --- clustering + confidence assignment (global) -------------------------
    def _cluster_and_assign(self, ids, embeddings) -> dict:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        if len(ids) == 1:
            labels = np.array([0])
        else:
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average",
            ).fit_predict(embeddings)

        uniq = sorted(set(labels.tolist()))
        centroids = []
        for c in uniq:
            v = embeddings[labels == c].mean(axis=0)
            n = np.linalg.norm(v)
            centroids.append(v / n if n > 0 else v)
        cmat = np.stack(centroids)  # (n_clusters, D), unit vectors

        out = {}
        for i, cid in enumerate(ids):
            sims = cmat @ embeddings[i]
            order = np.argsort(sims)[::-1]
            conf = float(sims[uniq.index(labels[i])])
            best = float(sims[order[0]])
            second = float(sims[order[1]]) if len(order) > 1 else -1.0
            mrg = best - second
            confident = conf >= self.assign_threshold and (len(uniq) == 1 or mrg >= self.margin)
            speaker = f"{self.prefix}_{int(labels[i]):03d}" if confident else "unknown"
            out[cid] = (speaker, round(conf, 4), round(mrg, 4))
        return out, len(uniq)

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        if not data:
            # still mark this shard done so the others' barrier doesn't hang
            if world_size > 1 and self.cache_dir:
                import numpy as np

                self._write_shard(rank, [], np.empty((0, 1)))
            return data

        logger.info(f"Embedding {len(data)} clips for speaker clustering (rank {rank}/{world_size})")
        ids, emb = self._embed_shard(data)

        if world_size > 1:
            if not self.cache_dir:
                raise ValueError("SpeakerLabeler needs cache_dir when running multi-shard (tasks>1)")
            self._write_shard(rank, ids, emb)
            logger.info(f"Rank {rank}: embeddings written, waiting at barrier for {world_size} shards")
            self._wait_for_all(world_size)
            all_ids, all_emb = self._load_all(world_size)
        else:
            all_ids, all_emb = ids, emb

        id2label, n_clusters = self._cluster_and_assign(all_ids, all_emb)

        n_assigned = 0
        for seg in data:
            speaker, conf, mrg = id2label[seg.id]
            seg.metadata["speaker"] = speaker
            seg.metadata["speaker_conf"] = conf
            seg.metadata["speaker_margin"] = mrg
            n_assigned += int(speaker != "unknown")
            if self.file_writer:
                self.file_writer.write(seg)

        logger.info(
            f"Speaker labeling (rank {rank}): {n_clusters} global clusters, "
            f"{n_assigned}/{len(data)} clips assigned in this shard"
        )
        return data


class DiarizationMetric(BaseMetric):
    """Per-file speaker diarization (how many speakers, who speaks when).

    Wraps ``pyannote/speaker-diarization-3.1`` (GATED: accept its conditions on
    the Hub and set ``HF_TOKEN``). Useful to flag/skip multi-speaker clips for
    single-speaker TTS, or to drive a segmenter. Emits ``num_speakers`` and the
    dominant speaker's share of speech time (``top_speaker_ratio``).
    """

    name = "🗣️ Diarization"
    gpu = True
    _requires_dependencies = ("pyannote.audio", "numpy")

    def __init__(
        self,
        pipeline_id: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(
            metric=("num_speakers", "top_speaker_ratio"), file_writer=file_writer, file_reader=file_reader
        )
        self.pipeline_id = pipeline_id
        self.device = device

    def _pipeline_on(self, device: str):
        def build():
            import torch
            from pyannote.audio import Pipeline

            p = Pipeline.from_pretrained(self.pipeline_id)
            p.to(torch.device(device))
            return p

        return cached_model(("DiarizationMetric", self.pipeline_id, device), build)

    def _diarize(self, segment: AudioSegment, device: str):
        from collections import defaultdict

        diarization = self._pipeline_on(device)(segment.audio_file)
        durations = defaultdict(float)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            durations[speaker] += turn.duration
        if not durations:
            return 0, 0.0
        total = sum(durations.values())
        return len(durations), round(max(durations.values()) / total, 4)

    def compute_metric(self, segment: AudioSegment):
        from audiogear.utils.runtime import normalize_device

        return self._diarize(segment, normalize_device(self.device))

    def compute_metric_cpu(self, segment: AudioSegment):
        return self._diarize(segment, "cpu")
