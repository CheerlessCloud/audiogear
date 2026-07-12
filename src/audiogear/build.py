"""Build a runnable pipeline + executor from a composed Hydra config.

The config has three parts: a ``reader``, an ordered ``metrics`` list (metric /
transcriber / labeler blocks), and a ``writer`` — each a node with a ``_target_``
that ``hydra.utils.instantiate`` turns into the corresponding object. The
resulting ``list[PipelineStep]`` is handed to an ``executor`` (also instantiated
from config) which shards and runs it.
"""

from __future__ import annotations

import math
import os

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from audiogear.pipeline.base import PipelineStep


def build_pipeline(cfg: DictConfig) -> list[PipelineStep]:
    """reader -> [metrics...] -> writer."""
    steps: list[PipelineStep] = [instantiate(cfg.reader)]
    for metric_cfg in cfg.get("metrics", []) or []:
        steps.append(instantiate(metric_cfg))
    writer = instantiate(cfg.writer)
    # Pre-declare the metric columns to the writer: its CSV schema is locked
    # from the first row, so a column the first clip happens to miss (skipped
    # clip, lane ordering) would otherwise silently drop for the whole shard.
    declared = [col for step in steps for col in (getattr(step, "output_columns", ()) or ())]
    if declared and hasattr(writer, "ensure_columns") and not writer.ensure_columns:
        writer.ensure_columns = list(dict.fromkeys(declared))
    steps.append(writer)
    return steps


def _detect_node_topology() -> tuple[int, int]:
    """Return ``(node_rank, num_nodes)`` from the launcher's environment.

    Supports SLURM (``SLURM_NODEID``/``SLURM_NNODES``), torchrun/MPI-style
    (``GROUP_RANK``/``NNODES`` or ``NODE_RANK``/``NUM_NODES``), and an explicit
    ``AUDIOGEAR_NODE_RANK``/``AUDIOGEAR_NUM_NODES`` override. Defaults to a
    single node. This lets the SAME ``audiogear`` command, launched once per
    node, automatically claim a disjoint slice of the shards.
    """
    for rank_key, n_key in [
        ("AUDIOGEAR_NODE_RANK", "AUDIOGEAR_NUM_NODES"),
        ("SLURM_NODEID", "SLURM_NNODES"),
        ("GROUP_RANK", "NNODES"),
        ("NODE_RANK", "NUM_NODES"),
    ]:
        if rank_key in os.environ and n_key in os.environ:
            return int(os.environ[rank_key]), int(os.environ[n_key])
    return 0, 1


def build_executor(cfg: DictConfig, pipeline: list[PipelineStep]):
    """Instantiate the executor, injecting the built pipeline and, for
    multi-node launches, this node's shard slice (computed from the launcher
    environment unless the config already pins ``local_tasks``)."""
    overrides = {}
    node_rank, num_nodes = _detect_node_topology()
    if num_nodes > 1 and int(cfg.executor.get("local_tasks", -1)) == -1:
        tasks = int(cfg.executor.tasks)
        per_node = math.ceil(tasks / num_nodes)
        overrides["local_tasks"] = per_node
        overrides["local_rank_offset"] = node_rank * per_node
        logger.info(
            f"Multi-node: node {node_rank}/{num_nodes} runs ranks "
            f"[{node_rank * per_node}, {min(tasks, (node_rank + 1) * per_node)})"
        )
    return instantiate(cfg.executor, pipeline=pipeline, **overrides)


def build_and_run(cfg: DictConfig):
    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    pipeline = build_pipeline(cfg)
    logger.info(f"Built pipeline with {len(pipeline)} steps: {[repr(s) for s in pipeline]}")
    executor = build_executor(cfg, pipeline)
    executor.run()
    logger.success("Pipeline finished.")
