"""Run independent metric lanes concurrently to overlap CPU and GPU work.

The default pipeline runs metrics one after another over the whole shard, so
while the CPU-only DSP metrics (bandwidth/pitch/style/wada/speaking_rate) run the
GPU sits idle, and while the GPU metrics run the cores sit idle. ``ParallelLanes``
runs several sub-pipelines ("lanes") concurrently over the *same* segment
objects — put the CPU metrics in one lane and the GPU metrics in another and the
two overlap, so wall-clock drops from ``sum(lanes)`` toward ``max(lanes)`` (the
CPU work effectively hides under the GPU work).

It works because CPU DSP metrics (librosa/numpy) and GPU metrics (waiting on CUDA
kernels) both release the GIL, so two Python threads genuinely run in parallel.

Config shape — lanes is a mapping of name -> ordered metric list::

    - _target_: audiogear.pipeline.parallel.ParallelLanes
      lanes:
        cpu:
          - {_target_: ...BandwidthMetric}
          - {_target_: ...PitchMetric, backend: pyin}
          - {_target_: ...StyleMetric}        # after pitch (reuses pitch_*)
        gpu:
          - {_target_: ...DistillMosMetric, device: ${device}}
          - {_target_: ...SquimMetrics, device: ${device}}

Requirements (the config author owns these):
  * Lanes must write **disjoint** metadata columns. They share segment objects;
    concurrent ``dict.__setitem__`` to *different* keys is atomic under the GIL,
    so disjoint columns are safe — overlapping columns would race.
  * No cross-lane data dependency. A metric that consumes ``text`` (WER,
    speaking_rate) or ``pitch_mean`` (StyleMetric) must sit in the SAME lane,
    after its producer. Producers shared by both lanes (e.g. ConsensusTranscriber
    writing ``text``) must run as ordinary sequential steps BEFORE this block.
  * Lane steps must not use ``file_reader`` resume (each lane would then diverge
    onto its own list); checkpoint on the sequential writer instead.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from audiogear.data import AudioPipeline
from audiogear.pipeline.base import PipelineStep


class ParallelLanes(PipelineStep):
    type = "🔀 - PARALLEL"
    name = "lanes"

    def __init__(self, lanes):
        super().__init__()
        from hydra.utils import instantiate

        # ``lanes`` is a mapping {name: [steps...]} (Hydra builds the leaf steps
        # but wraps them in config containers — unwrap to plain lists). A bare
        # sequence of lanes is also accepted (Python construction).
        if hasattr(lanes, "items"):
            names, lane_lists = list(lanes.keys()), list(lanes.values())
        else:
            lane_lists = list(lanes)
            names = [str(i) for i in range(len(lane_lists))]

        def to_step(s):
            return s if isinstance(s, PipelineStep) else instantiate(s)

        self.lane_names = names
        # each lane is an ordered sequence of steps (Hydra already built the leaf
        # objects; a raw config leaf, from Python, is instantiated here)
        self.lanes = [[to_step(s) for s in lane] for lane in lane_lists]

    def __repr__(self):
        inner = " || ".join(
            f"{nm}:" + "→".join(getattr(s, "name", type(s).__name__) for s in lane)
            for nm, lane in zip(self.lane_names, self.lanes)
        )
        return f"{self.type}: [{inner}]"

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Union of the metadata columns declared by the lane steps (see
        ``BaseMetric.output_columns``) — lets the pipeline builder pre-declare
        the writer schema across all lanes."""
        cols: list[str] = []
        for lane in self.lanes:
            for step in lane:
                cols.extend(getattr(step, "output_columns", ()) or ())
        return tuple(cols)

    def _run_lane(self, lane, data, rank, world_size):
        for step in lane:
            data = step(data, rank, world_size)
        return data

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        if len(self.lanes) <= 1:
            for lane in self.lanes:
                self._run_lane(lane, data, rank, world_size)
            return data
        logger.info(
            f"Running {len(self.lanes)} lanes concurrently over {len(data)} segments: {self.lane_names}"
        )
        # Join ALL lanes before raising: with a bare f.result() the first error
        # would surface while sibling lanes keep running to completion behind a
        # dying shard, and their own errors would be silently lost.
        errors: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=len(self.lanes)) as ex:
            futures = [
                (nm, ex.submit(self._run_lane, lane, data, rank, world_size))
                for nm, lane in zip(self.lane_names, self.lanes)
            ]
            for nm, f in futures:
                try:
                    f.result()
                except Exception as e:
                    logger.exception(f"Lane '{nm}' failed: {e}")
                    errors.append((nm, e))
        if errors:
            failed = ", ".join(nm for nm, _ in errors)
            raise RuntimeError(f"{len(errors)} lane(s) failed: {failed}") from errors[0][1]
        return data
