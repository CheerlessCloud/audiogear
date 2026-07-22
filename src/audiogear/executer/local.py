import os
from copy import deepcopy
from functools import partial
from typing import Callable

import multiprocess
from loguru import logger

from audiogear.executer.base import PipelineExecutor
from audiogear.io import DataFolderLike
from audiogear.pipeline.base import PipelineStep


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        tasks: int = 1,
        workers: int = -1,
        logging_dir: DataFolderLike = None,
        skip_completed: bool = True,
        start_method: str = "spawn",
        gpus: int | None = None,
        local_tasks: int = -1,
        local_rank_offset: int = 0,
    ):
        """Execute a pipeline locally, with optional GPU pinning and multi-node sharding.

        The dataset is split into ``tasks`` shards (rank in ``[0, tasks)``).
        ``workers`` of them run concurrently, and each running worker is pinned
        to one GPU (round-robin over the visible GPUs) so model blocks land on
        distinct devices. This single executor covers all single-machine modes:

          - 1 GPU:        tasks=N, workers=1
          - many GPUs:    tasks=N, workers=<n_gpus>   (one GPU per worker)

        For multi-node, launch this SAME executor once per node with the same
        ``tasks``; each node auto-claims a node-local slice via ``local_tasks`` /
        ``local_rank_offset``, computed in ``build._detect_node_topology`` from the
        launcher env (``SLURM_NODEID``/``SLURM_NNODES``, torchrun ``GROUP_RANK``/
        ``NNODES``, or ``AUDIOGEAR_NODE_RANK``/``AUDIOGEAR_NUM_NODES``). Node i then
        runs ranks ``[i*local_tasks, (i+1)*local_tasks)``. Any launcher works —
        slurm, torchrun, or plain ssh; see the README for ready-made recipes. Put
        ``logging_dir`` on shared storage (NFS/S3) so completion markers — and thus
        ``skip_completed`` resume — are visible across nodes.

        Args:
            tasks: total number of shards (the world size).
            workers: concurrent shards. -1 => all of this node's tasks at once.
            gpus: number of GPUs to pin across (None => auto-detect via
                ``torch.cuda.device_count()``; 0 => CPU only, no pinning).
            local_tasks: how many of ``tasks`` this node runs (-1 => all).
            local_rank_offset: index of this node's first rank within ``tasks``.
        """
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.local_tasks = local_tasks if local_tasks != -1 else tasks
        self.local_rank_offset = local_rank_offset
        self.workers = workers if workers != -1 else self.local_tasks
        self.start_method = start_method
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.visible_device_tokens = (
            None
            if visible_devices is None
            else tuple(token.strip() for token in visible_devices.split(",") if token.strip())
        )
        self.gpus = gpus if gpus is not None else self._detect_gpus()

    @staticmethod
    def _detect_gpus() -> int:
        """Count GPUs WITHOUT importing torch / initializing CUDA in the parent.

        Initializing a CUDA context in the parent and then spawning workers that
        also use CUDA deadlocks. We respect CUDA_VISIBLE_DEVICES, then fall back
        to ``nvidia-smi``.
        """
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            return len([d for d in visible.split(",") if d.strip() != ""])
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
            )
            return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
        except Exception:  # pragma: no cover
            return 0

    def _pin_gpu(self, local_rank: int) -> None:
        """Pin this worker process to one GPU. Must run before any CUDA init,
        which is why model blocks load their models lazily (on first use)."""
        if not self.gpus or self.gpus <= 0:
            return
        if self.visible_device_tokens is not None:
            if not self.visible_device_tokens:
                return
            visible_count = min(self.gpus, len(self.visible_device_tokens))
            selected_device = self.visible_device_tokens[local_rank % visible_count]
        else:
            selected_device = str(local_rank % self.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
        logger.info(f"Worker local_rank={local_rank} pinned to visible GPU {selected_device}")

    def _launch_run_for_rank(self, rank: int, ranks_q, completed=None, completed_lock=None) -> None:
        local_rank = ranks_q.get()
        try:
            self._pin_gpu(local_rank)
            return self._run_for_rank(rank, local_rank)
        finally:
            if completed and completed_lock:
                with completed_lock:
                    completed.value += 1
                    logger.info(f"{completed.value}/{self.world_size} tasks completed.")
            ranks_q.put(local_rank)  # free up used rank

    def _local_ranks(self) -> list[int]:
        """The subset of global ranks this node is responsible for."""
        start = self.local_rank_offset
        end = min(self.tasks, start + self.local_tasks)
        return list(range(start, end))

    def run(self):
        local_ranks = self._local_ranks()
        if all(map(self.is_rank_completed, local_ranks)):
            logger.info(f"Not doing anything as all {len(local_ranks)} local tasks are already completed.")
            return

        self.save_executor_as_json()
        mg = multiprocess.Manager()
        ranks_q = mg.Queue()
        for i in range(self.workers):
            ranks_q.put(i)

        ranks_to_run = [r for r in local_ranks if not self.is_rank_completed(r)]
        if (skipped := len(local_ranks) - len(ranks_to_run)) > 0:
            logger.info(f"Skipping {skipped} already completed tasks")

        if self.workers == 1:
            pipeline = self.pipeline
            for rank in ranks_to_run:
                self.pipeline = deepcopy(pipeline)
                self._launch_run_for_rank(rank, ranks_q)
        else:
            completed_counter = mg.Value("i", skipped)
            completed_lock = mg.Lock()
            ctx = multiprocess.get_context(self.start_method)
            with ctx.Pool(self.workers) as pool:
                list(
                    pool.imap_unordered(
                        partial(
                            self._launch_run_for_rank,
                            ranks_q=ranks_q,
                            completed=completed_counter,
                            completed_lock=completed_lock,
                        ),
                        ranks_to_run,
                    )
                )

    @property
    def world_size(self) -> int:
        return self.tasks
