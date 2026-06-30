# Distributed runs: multi-GPU and multi-node

audiogear has **one** sharding model that covers a single GPU, many GPUs, and many
machines. The dataset is split into `tasks` shards (the world size); `workers`
shards run concurrently in separate processes; each running worker pins one GPU
(round-robin). Each shard writes its own `*_${rank}.csv`, which you concatenate
afterward. Completion markers under `logging_dir` give resume.

The examples below were run on a 2× RTX 4090 box.

## Single machine

```bash
# 1 GPU
uv run python process.py --config-name <cfg> executor.tasks=8 executor.workers=1

# many GPUs — one model stack per card
uv run python process.py --config-name <cfg> executor.tasks=4 executor.workers=2
```

With `tasks=4 workers=2` the two workers pin distinct GPUs:

```
Worker local_rank=0 pinned to physical GPU 0
Worker local_rank=1 pinned to physical GPU 1
...
1/4 tasks completed.  2/4 tasks completed.  3/4 tasks completed.  4/4 tasks completed.
-> outputs/.../ext_00000.csv ... ext_00003.csv     (4 shards)
-> logs/.../completions/00000 ... 00003            (4 markers)
```

GPU detection never initializes CUDA in the parent process and workers use the
`spawn` start method, so multi-GPU does not deadlock.

## Many machines — there is no Slurm executor class

Multi-node is the **same** `LocalPipelineExecutor` launched once per node. On
startup, `build._detect_node_topology()` reads the launcher environment and gives
each node a disjoint, contiguous slice of `tasks` by setting `local_tasks` and
`local_rank_offset` for you. Recognized sources (first match wins):

| Launcher | Rank var | Count var |
|---|---|---|
| explicit override | `AUDIOGEAR_NODE_RANK` | `AUDIOGEAR_NUM_NODES` |
| SLURM | `SLURM_NODEID` | `SLURM_NNODES` |
| torchrun / MPI | `GROUP_RANK` | `NNODES` |
| generic | `NODE_RANK` | `NUM_NODES` |

Node *i* runs ranks `[i·ceil(tasks/N), (i+1)·ceil(tasks/N))`.

```bash
# SLURM: the SAME command on every node; each claims its slice
srun -N4 --gpus-per-node=8 \
  uv run python process.py --config-name <cfg> executor.tasks=256 executor.workers=8

# manual / ssh: set the node env per node
AUDIOGEAR_NODE_RANK=$i AUDIOGEAR_NUM_NODES=$N \
  uv run python process.py --config-name <cfg> executor.tasks=256 executor.workers=8
```

### Verified on one box (two simulated nodes)

Running the same config twice with `AUDIOGEAR_NUM_NODES=2` and `tasks=4`:

```
# node 0
Multi-node: node 0/2 runs ranks [0, 2)
# node 1
Multi-node: node 1/2 runs ranks [2, 4)
-> outputs/.../ext_00000.csv ... ext_00003.csv   (all 4 shards, shared output dir)
```

## Resume (`skip_completed`)

A finished shard writes `logging_dir/completions/<rank>`. On rerun those ranks are
skipped. Verified: re-running node 0 after it finished printed

```
Not doing anything as all 2 local tasks are already completed.
```

**For cross-node resume, put `logging_dir` on shared storage** (NFS, or S3 — see
[storage.md](storage.md)). With node-local logs, `skip_completed` only sees that
node's own markers, so a different node can't tell a shard is already done.

## Picking `tasks` / `workers`

- `workers` = number of GPUs you want busy (one model stack per GPU). Because the
  CUDA-OOM ladder spills pathological clips to CPU instead of crashing, you can
  push `workers` past one-stack-per-GPU if VRAM at load time allows.
- `tasks` = resume granularity (and the unit of cross-node distribution). More
  shards = finer resume; thanks to the process-global model cache, more shards do
  **not** mean more model reloads within a worker.
- Process **smallest datasets first** when sweeping a collection (see
  [`examples/`](../examples/)) so a config mistake surfaces in seconds.
