import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from loguru import logger  # loguru 0.7.2

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep

# from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.utils.progress import tqdm
from audiogear.utils.runtime import (
    NotRecoverable,
    free_cuda,
    is_oom_error,
    iter_windows,
    length_buckets,
    resolve_num_threads,
)


class BaseMetric(PipelineStep, ABC):
    type = "🔻 - METRIC"

    # CPU-bound DSP metrics set parallel_cpu=True to fan compute_metric across
    # threads (librosa/numpy release the GIL on heavy FFT/array ops, so threads
    # give real multi-core speedup). GPU/model metrics keep it False (a single
    # CUDA stream per process; concurrency comes from multiple worker processes).
    parallel_cpu = False

    # GPU/model metrics set gpu=True. It enables the CUDA-OOM recovery ladder in
    # run(): a clip that blows up VRAM (usually a very long one) is retried with
    # a degraded strategy instead of crashing the whole shard. CPU metrics leave
    # it False and never pay for the guard.
    gpu = False

    # GPU metrics that implement compute_batch (a real batched forward) set this
    # True. run() then groups clips into length-bucketed, VRAM-bounded batches
    # (see batch_size / max_batch_seconds) instead of going clip-by-clip — the
    # big GPU-utilisation win. Metrics that leave it False fall back to the
    # per-clip path automatically.
    supports_batch = False

    # bs=1 GPU metrics (model can't batch, e.g. internal segmenting) set this True
    # and split work into _prepare (CPU decode, thread-parallel) + _infer (GPU,
    # single consumer). run() then decodes clips ahead on a thread pool while the
    # GPU stays busy — overlap without a data race, because exactly one thread
    # ever touches the model. No effect for metrics whose decode is internal to
    # the model call (whisper/pyannote) — leave it False there.
    prefetch = False

    def __init__(
        self,
        metric,
        file_writer=None,
        file_reader: BaseDiskReader = None,
        num_threads: int = -1,
        chunk_seconds: float = 20.0,
        cpu_overflow_threads: int = 1,
        batch_size: int = 16,
        max_batch_seconds: float = 480.0,
        max_consecutive_failures: int = 50,
    ):
        """
        the file writer helps save progress to disk while you compute metrics.
        the reader will let you load that file and continue from where you left off.

        num_threads: CPU worker threads for ``parallel_cpu`` metrics. ``-1``/``0``
            => all available cores (the default — maximise throughput).
        chunk_seconds: window length used by the OOM ladder when a long clip is
            re-decoded piecewise on the GPU (see ``compute_metric_recover``).
        cpu_overflow_threads: threads used to drain the CPU-overflow bucket of
            clips that even chunked-GPU could not handle. ``1`` (serial) is the
            safe default — most model objects are not thread-safe; raise it only
            for metrics whose CPU path is independent per call.
        batch_size: hard cap on clips per GPU batch (``supports_batch`` metrics).
            ``1`` disables batching for that block.
        max_batch_seconds: VRAM budget for a batch, expressed as
            ``batch_size * max_padded_clip_seconds``. Activation memory scales
            roughly with this product, so it is the knob to turn down on OOM (the
            binary-backoff ladder also auto-recovers, so it can be aggressive).
        max_consecutive_failures: abort the shard when this many clips fail in an
            unbroken row. Isolated failures (corrupt files) become sentinel rows,
            but an unbroken streak means the failure is systematic (model load,
            bad config, dead CUDA context) and sentinels would silently produce a
            garbage run.
        """
        self.metric = metric
        self.file_writer = file_writer
        self.file_reader = file_reader
        self.num_threads = resolve_num_threads(num_threads)
        self.chunk_seconds = chunk_seconds
        self.cpu_overflow_threads = cpu_overflow_threads
        self.batch_size = batch_size
        self.max_batch_seconds = max_batch_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self._failures = 0
        self._consecutive_failures = 0

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Metadata columns this metric writes — lets the pipeline builder
        pre-declare the writer schema so a column is never dropped just because
        the first row happened to miss it."""
        return self.metric if isinstance(self.metric, tuple) else (self.metric,)

    @abstractmethod
    def compute_metric(self, segment: AudioSegment) -> int | float | bool:
        raise NotImplementedError

    def compute_batch(self, segments: list[AudioSegment]) -> list:
        """Compute the metric for a batch of clips in one GPU forward.

        Default: loop ``compute_metric`` (correct, no speedup). GPU metrics that
        can pad+stack a batch override this and set ``supports_batch = True``.
        Returns one result per input segment, in order."""
        return [self.compute_metric(s) for s in segments]

    # --- prefetch split (override in ``prefetch = True`` metrics) -------------
    def _prepare(self, segment: AudioSegment):
        """CPU-side preprocessing (decode/resample) — must be thread-safe; runs
        on the decode thread pool. Default: pass the segment through."""
        return segment

    def _infer(self, prepared):
        """GPU-side inference on one prepared input — runs on a single thread, so
        the model is never touched concurrently. Default: not implemented."""
        raise NotImplementedError

    # --- OOM recovery hooks (override in GPU metrics that can degrade) ---------
    def compute_metric_recover(self, segment: AudioSegment):
        """Cheaper GPU retry for a clip that OOM'd at full size.

        Typically re-decodes the clip in ``chunk_seconds`` windows and aggregates
        the per-window values, so peak VRAM is bounded by one window. Metrics
        that cannot be windowed leave this as-is (skip to the CPU strategy)."""
        raise NotRecoverable

    def compute_metric_cpu(self, segment: AudioSegment):
        """Exact CPU computation for a clip the GPU could not handle at all.

        Slow but reliable (host RAM is plentiful). Implemented by metrics whose
        model can run on CPU; others skip to the sentinel value."""
        raise NotRecoverable

    def _failed_value(self):
        """Sentinel written when every strategy fails — keeps the row, flags it.

        ``float('nan')`` (per sub-metric for tuple metrics) so downstream
        filtering can spot and drop these clips without crashing the run."""
        if isinstance(self.metric, tuple):
            return tuple(float("nan") for _ in self.metric)
        return float("nan")

    def _load_data(self):
        logger.info(f"Loading saved data from {self.file_reader.data_folder.path}")
        self.prev_data = self.file_reader()
        self.ids_in_list2 = {item.id for item in self.prev_data}

    def _assign(self, segment, results):
        if isinstance(self.metric, tuple):
            assert len(self.metric) == len(results), "The number of results should equal the number of expected metrics"
            for i in range(len(self.metric)):
                segment.metadata[self.metric[i]] = results[i]
        else:
            segment.metadata[self.metric] = results

    def _write(self, segment):
        if self.file_writer:
            self.file_writer.write(segment)

    # --- per-clip failure guard -------------------------------------------------
    # One corrupt/empty clip must never kill a shard (a single bad mp3 used to
    # take down a 549k-clip run). Every execution path routes per-clip exceptions
    # here: the clip gets a sentinel row and the run continues. Only an unbroken
    # failure streak (systematic breakage, not bad data) aborts the shard.
    # The counters are heuristic — parallel_cpu threads may race on them, which
    # only makes the abort threshold approximate.
    def _record_failure(self, segment: AudioSegment, exc: BaseException):
        """Log one clip's failure and return the sentinel value to write."""
        self._failures += 1
        self._consecutive_failures += 1
        logger.warning(
            f"{self.metric} failed on id={segment.id} ({segment.audio_file}): "
            f"{type(exc).__name__}: {exc} — writing sentinel"
        )
        if self._consecutive_failures >= self.max_consecutive_failures:
            raise RuntimeError(
                f"{self.metric}: {self._consecutive_failures} clips failed in an unbroken "
                f"row — aborting the shard (this is systematic, not bad data)"
            ) from exc
        return self._failed_value()

    def _record_success(self):
        self._consecutive_failures = 0

    def _guarded_compute(self, segment: AudioSegment):
        """``compute_metric`` wrapped in the per-clip guard (CPU paths; the GPU
        paths get the same guard inside ``_gpu_compute`` / ``_process_batch`` /
        the prefetch loop)."""
        try:
            res = self.compute_metric(segment)
        except Exception as e:
            return self._record_failure(segment, e)
        self._record_success()
        return res

    # --- GPU path: per-clip OOM ladder ----------------------------------------
    def _gpu_compute(self, segment: AudioSegment):
        """Run one clip on the GPU with OOM recovery and the per-clip guard.

        Returns ``(value, deferred)``. When the full-size and chunked-GPU
        attempts both OOM, returns ``(None, True)`` so run() can batch the clip
        into the CPU-overflow bucket instead of stalling the GPU on it. A
        non-OOM error (corrupt audio, model quirk on an odd input) becomes a
        sentinel value instead of killing the shard.
        """
        try:
            res = self.compute_metric(segment)
        except Exception as e:
            if not is_oom_error(e):
                return self._record_failure(segment, e), False
            free_cuda()
            logger.warning(
                f"CUDA OOM on {self.metric} for id={segment.id} "
                f"(duration={segment.duration}s) — retrying chunked on GPU"
            )
            try:
                res = self.compute_metric_recover(segment)
            except NotRecoverable:
                return None, True
            except Exception as e2:
                # OOM again or the windowed retry itself broke — either way the
                # CPU drain is the last strategy that can still finish the clip
                # (it guards everything and writes a sentinel at worst).
                free_cuda()
                logger.warning(
                    f"Chunked GPU retry failed ({type(e2).__name__}) for id={segment.id} "
                    f"— deferring to CPU"
                )
                return None, True
        self._record_success()
        return res, False

    # --- GPU path: length-bucketed batched inference --------------------------
    def _segment_seconds(self, segment: AudioSegment) -> float:
        """Clip length for batching — the reader's ``duration`` if present, else
        a header-only probe (cached back onto the segment)."""
        if segment.duration:
            return float(segment.duration)
        from audiogear.audio import audio_duration

        segment.duration = audio_duration(segment.audio_file)
        return segment.duration

    def _durations(self, data: AudioPipeline) -> list[float]:
        # Most rows already carry duration (no I/O). Probe the rest in parallel
        # across cores so the length pass never serialises on disk.
        missing = [i for i, s in enumerate(data) if not s.duration]
        if missing:
            logger.info(f"Probing duration of {len(missing)} clips for batching")
            with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
                list(ex.map(self._segment_seconds, (data[i] for i in missing)))
        # read durations directly now (probe wrote them back) — avoid re-probing
        # clips whose true/unreadable duration is 0.0 (falsy) a second time.
        return [s.duration or 0.0 for s in data]

    def _run_gpu_batched(self, data: AudioPipeline):
        durations = self._durations(data)
        buckets = length_buckets(durations, self.max_batch_seconds, self.batch_size)
        logger.info(
            f"Batched GPU: {len(data)} clips -> {len(buckets)} batches "
            f"(<= {self.batch_size} clips / {self.max_batch_seconds:.0f}s padded each)"
        )
        overflow: list[AudioSegment] = []
        with tqdm(total=len(data)) as pbar:
            for idxs in buckets:
                self._process_batch([data[i] for i in idxs], overflow, pbar)
        if overflow:
            self._drain_overflow(overflow)

    def _run_gpu_prefetch(self, data: AudioPipeline):
        """bs=1 GPU pass that overlaps decode with inference, race-free.

        A thread pool runs ``_prepare`` (decode) for clips ahead of the GPU while
        the main thread runs ``_infer`` one clip at a time — so the model is only
        ever called from this single thread (no lock, no race), and the GPU rarely
        waits on the CPU to decode the next clip. Per-clip CUDA OOM drops into the
        usual ladder (chunked-GPU → CPU-overflow)."""
        from collections import deque

        depth = max(self.num_threads * 2, 8)
        overflow: list[AudioSegment] = []
        n = len(data)
        i = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            inflight = deque()
            while i < n and len(inflight) < depth:
                inflight.append((data[i], ex.submit(self._prepare, data[i])))
                i += 1
            with tqdm(total=n) as pbar:
                while inflight:
                    seg, fut = inflight.popleft()
                    try:
                        res = self._infer(fut.result())  # single-thread GPU use
                        self._assign(seg, res)
                        self._write(seg)
                        self._record_success()
                    except Exception as e:
                        if is_oom_error(e):
                            free_cuda()
                            r, deferred = self._gpu_compute(seg)  # re-decode + ladder
                            if deferred:
                                overflow.append(seg)
                            else:
                                self._assign(seg, r)
                                self._write(seg)
                        else:
                            # decode (fut.result) or inference broke on this one
                            # clip — sentinel, keep the shard going
                            self._assign(seg, self._record_failure(seg, e))
                            self._write(seg)
                    pbar.update(1)
                    if i < n:
                        inflight.append((data[i], ex.submit(self._prepare, data[i])))
                        i += 1
        if overflow:
            self._drain_overflow(overflow)

    def _process_batch(self, batch: list[AudioSegment], overflow: list[AudioSegment], pbar):
        """Run one batch; on CUDA OOM split in half and retry (binary backoff).

        A size-1 batch that still OOMs drops into the per-clip ladder
        (chunked-GPU -> CPU-overflow), so batching and the long-clip recovery are
        one continuous mechanism. A non-OOM batch error (usually one corrupt clip
        poisoning the whole forward) retries the batch clip-by-clip so only the
        culprit gets a sentinel."""
        try:
            results = self.compute_batch(batch)
            for seg, res in zip(batch, results):
                self._assign(seg, res)
                self._write(seg)
            self._record_success()
            pbar.update(len(batch))
            return
        except Exception as e:
            free_cuda()
            if not is_oom_error(e):
                if len(batch) == 1:
                    self._assign(batch[0], self._record_failure(batch[0], e))
                    self._write(batch[0])
                    pbar.update(1)
                    return
                logger.warning(
                    f"{type(e).__name__} on batch of {len(batch)} for {self.metric} "
                    f"— retrying per clip"
                )
                for seg in batch:
                    res, deferred = self._gpu_compute(seg)
                    if deferred:
                        overflow.append(seg)
                    else:
                        self._assign(seg, res)
                        self._write(seg)
                    pbar.update(1)
                return
        if len(batch) == 1:
            res, deferred = self._gpu_compute(batch[0])
            if deferred:
                overflow.append(batch[0])
            else:
                self._assign(batch[0], res)
                self._write(batch[0])
            pbar.update(1)
            return
        logger.warning(f"CUDA OOM on batch of {len(batch)} for {self.metric} — splitting")
        mid = len(batch) // 2
        self._process_batch(batch[:mid], overflow, pbar)
        self._process_batch(batch[mid:], overflow, pbar)

    def _drain_overflow(self, overflow: list[AudioSegment]):
        """Compute the deferred (OOM) clips on CPU, then assign + write them.

        These are the few pathological clips no GPU strategy could handle. CPU
        has the RAM to finish them; ``cpu_overflow_threads`` controls how many
        run at once (serial by default for model thread-safety)."""
        logger.warning(
            f"Draining {len(overflow)} OOM clip(s) for {self.metric} on CPU "
            f"({self.cpu_overflow_threads} thread(s))"
        )

        def work(seg):
            try:
                return self.compute_metric_cpu(seg)
            except NotRecoverable:
                self._failures += 1
                logger.error(f"No CPU fallback for {self.metric} id={seg.id}; writing sentinel")
                return self._failed_value()
            except Exception as e:  # CPU should not OOM; log and keep going
                self._failures += 1
                logger.exception(f"CPU fallback failed for {self.metric} id={seg.id}: {e}")
                return self._failed_value()

        if self.cpu_overflow_threads > 1:
            with ThreadPoolExecutor(max_workers=self.cpu_overflow_threads) as ex:
                results = list(tqdm(ex.map(work, overflow), total=len(overflow), desc="cpu-overflow"))
        else:
            results = [work(seg) for seg in tqdm(overflow, desc="cpu-overflow")]
        for seg, res in zip(overflow, results):
            self._assign(seg, res)
            self._write(seg)

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        if self.file_reader:
            self._load_data()
            data = [item for item in data if item.id not in self.ids_in_list2]

        logger.info(f"Computing {self.metric} for {len(data)} segments")
        self._failures = 0
        self._consecutive_failures = 0

        if self.gpu and self.supports_batch and self.batch_size > 1 and len(data) > 1:
            # Length-bucketed, VRAM-bounded batched GPU pass (the throughput win).
            self._run_gpu_batched(data)
        elif self.gpu and self.prefetch and len(data) > 1:
            # bs=1 GPU pass with parallel decode prefetch (single-thread inference).
            self._run_gpu_prefetch(data)
        elif self.gpu:
            # GPU pass with per-clip OOM recovery; un-recoverable clips are
            # batched and finished on CPU after the GPU pass.
            overflow: list[AudioSegment] = []
            for segment in tqdm(data):
                res, deferred = self._gpu_compute(segment)
                if deferred:
                    overflow.append(segment)
                    continue
                self._assign(segment, res)
                self._write(segment)
            if overflow:
                self._drain_overflow(overflow)
        elif self.parallel_cpu and self.num_threads > 1 and len(data) > 1:
            # Parallel CPU map. Submit in bounded batches (instead of one giant
            # ex.map over millions of clips) so memory/progress stay sane and
            # results are written incrementally.
            self._run_parallel_cpu(data)
        else:
            for segment in tqdm(data):
                self._assign(segment, self._guarded_compute(segment))
                self._write(segment)

        if self._failures:
            logger.warning(
                f"{self.metric}: {self._failures}/{len(data)} clip(s) failed and "
                f"carry sentinel values"
            )
        if self.file_reader:
            data = self.prev_data + data
        return data

    def _run_parallel_cpu(self, data: AudioPipeline):
        batch = max(self.num_threads * 8, 256)
        n_batches = math.ceil(len(data) / batch)
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            with tqdm(total=len(data)) as pbar:
                for b in range(n_batches):
                    chunk = data[b * batch : (b + 1) * batch]
                    for segment, res in zip(chunk, ex.map(self._guarded_compute, chunk)):
                        self._assign(segment, res)
                        self._write(segment)
                        pbar.update(1)


class PrefetchGPUMetric(BaseMetric):
    """Base for a bs=1 GPU metric: decode-prefetch + single-thread inference,
    with windowed-mean OOM recovery (GPU → CPU).

    Captures the pattern shared by SQUIM and DistillMOS (models that can't safely
    pad-batch). A subclass provides ``name``, ``sample_rate``,
    ``_requires_dependencies``, ``_model_on(device)``, and ``_run(audio, device)``
    returning the metric value (a float, or a tuple for multi-column metrics).
    Decode/recovery/cache plumbing is inherited."""

    gpu = True
    prefetch = True
    sample_rate = 16000

    def __init__(self, metric, device: str = "cuda", chunk_seconds: float = 20.0, file_writer=None, file_reader=None):
        super().__init__(metric=metric, file_writer=file_writer, file_reader=file_reader, chunk_seconds=chunk_seconds)
        self.device = device

    def _model_on(self, device: str):
        raise NotImplementedError

    def _run(self, audio, device: str):
        """Model forward on one decoded waveform; returns the metric value(s)."""
        raise NotImplementedError

    @staticmethod
    def _mean(vals: list):
        """Mean over per-window values; handles both scalar and tuple metrics."""
        if isinstance(vals[0], tuple):
            n, k = len(vals), len(vals[0])
            return tuple(sum(v[i] for v in vals) / n for i in range(k))
        return sum(vals) / len(vals)

    def _prepare(self, segment: AudioSegment):
        from audiogear.audio import load_audio

        audio, _ = load_audio(segment.audio_file, target_sr=self.sample_rate, mono=True)
        return audio

    def _infer(self, prepared):
        return self._run(prepared, self.device)

    def _windowed(self, segment: AudioSegment, device: str):
        from audiogear.audio import load_audio

        audio, _ = load_audio(segment.audio_file, target_sr=self.sample_rate, mono=True)
        vals = [self._run(w, device) for w in iter_windows(audio, self.sample_rate, self.chunk_seconds)]
        if not vals:  # empty/zero-length clip
            return self._failed_value()
        return self._mean(vals)

    def compute_metric(self, segment: AudioSegment):
        return self._run(self._prepare(segment), self.device)

    def compute_metric_recover(self, segment: AudioSegment):
        return self._windowed(segment, self.device)

    def compute_metric_cpu(self, segment: AudioSegment):
        return self._windowed(segment, "cpu")
