# Plan: Ray Distributed Training (DDP) for Transformer Architectures

## Context

Training currently runs on a single GPU. A Ray cluster is available and we want to leverage it for multi-device DDP training. The key constraint is that all nodes share a `/data` mount, but the mount is **much slower** than inter-node communication. This means we need to minimize reads from `/data` during training — ideally only rank 0 reads from the mount, then broadcasts over the fast interconnect.

The goal is to start with the vanilla transformer training loop but build the infrastructure modularly so ViT/BERT/future architectures can reuse it.

---

## Step 1: Add `ray[train]` dependency

**File:** `pyproject.toml`

Add `"ray[train]>=2.40.0,<3"` to dependencies. This provides `TorchTrainer`, `prepare_model()`, and all DDP orchestration.

---

## Step 2: Create `training/distributed.py` — Core DDP abstractions

**New file:** `transformer_architectures/training/distributed.py`

This is the shared backbone. Contains:

- **`DistributedContext`** dataclass — wraps `rank`, `world_size`, `local_rank`, `is_primary` (rank 0). Built from `ray.train.get_context()`.

- **`TrainableArchitecture` Protocol** — the interface each architecture implements:
  ```python
  build_model() -> nn.Module
  build_datasets() -> (train, val, test)
  build_train_dataloader(dataset, rank, world_size) -> DataLoader
  build_eval_dataloader(dataset) -> DataLoader
  build_optimizer(model) -> Optimizer
  build_scheduler(optimizer, steps_per_epoch) -> LRScheduler
  build_criterion() -> nn.Module
  train_step(model, batch, criterion, autocast_ctx) -> Tensor  # returns loss
  evaluate(model, dataloader, criterion, autocast_ctx) -> dict[str, float]
  ```

- **`distributed_train_loop(config: dict)`** — the function passed to `TorchTrainer` as `train_loop_per_worker`. Orchestrates:
  1. Build `DistributedContext` from Ray
  2. Call `arch.build_model()`, wrap with `ray.train.torch.prepare_model()` (handles DDP)
  3. Call `arch.build_datasets()` (uses broadcast — see Step 3)
  4. Build dataloaders, optimizer, scheduler, criterion
  5. Rank 0 only: MLflow setup (`set_tracking_uri`, `start_run`, `log_params`)
  6. Epoch loop with **`model.no_sync()`** during gradient accumulation steps (avoid unnecessary all-reduce)
  7. Rank 0 only: log metrics, evaluate, checkpoint
  8. Report metrics via `ray.train.report()` for Ray's tracking

- **`unwrap_ddp(model)`** — returns `model.module` if DDP-wrapped, else `model`

- **`make_autocast_ctx(precision, device_type)`** — extracted from the duplicate implementations in vanilla/vit training

---

## Step 3: Create `training/data_broadcast.py` — Slow-mount data strategy

**New file:** `transformer_architectures/training/data_broadcast.py`

Solves the slow `/data` mount problem. Strategy: **rank 0 loads and tokenizes, then broadcasts the compact numpy arrays to all ranks over the fast interconnect.**

- **`load_and_broadcast(load_fn, rank)`**:
  - Rank 0: calls `load_fn()` to read from `/data` + tokenize → 4 numpy arrays (~200MB for 7M pairs)
  - Rank 0: broadcasts via `torch.distributed.broadcast_object_list()`
  - Other ranks: receive the arrays (sub-second over fast interconnect)

- **Why this approach over alternatives:**
  - Pre-sharded files: requires an offline preprocessing step, adds operational complexity
  - Memory-mapped files: still hits the slow `/data` mount on every page fault
  - Streaming/IterableDataset: random access needed for shuffling; streaming limits this
  - Broadcast: simple, one-time cost at startup, leverages the fast interconnect

The vanilla `TransformerDataset` already stores data as 4 flat numpy arrays. We add a `TransformerDataset.from_arrays()` classmethod so ranks receiving the broadcast can construct a dataset without re-tokenizing.

---

## Step 4: Create `training/ray_launcher.py` — Ray job submission

**New file:** `transformer_architectures/training/ray_launcher.py`

- **`RayConfig` (Pydantic model)**: `num_workers`, `use_gpu`, `backend` (nccl)

- **`launch(arch, ray_config, train_config)`**:
  - `ray.init(address="auto")` — connects to the existing cluster
  - Creates `TorchTrainer` with `ScalingConfig(num_workers=N, use_gpu=True, resources_per_worker={"GPU": 1})`
  - Sets `TorchConfig(backend="nccl")`
  - Calls `trainer.fit()` → returns `ray.train.Result`

---

## Step 5: Add `DistributedTokenBudgetBatchSampler`

**File to modify:** `transformer_architectures/samplers/bucket_sampler.py`

**Problem:** `TokenBudgetBatchSampler` produces variable-length batch lists. In DDP, all ranks must iterate the same number of batches per epoch (otherwise some ranks hang at the all-reduce barrier).

**Solution:** New subclass `DistributedTokenBudgetBatchSampler`:
- Takes `rank`, `world_size` in addition to existing params
- All ranks build the full batch list (deterministic given same seed + same data)
- Pads batch count to be divisible by `world_size`
- Each rank takes every `world_size`-th batch: `all_batches[rank::world_size]`
- `set_epoch(epoch)` reseeds the generator (same as `DistributedSampler` pattern)

This ensures all ranks process exactly the same number of optimizer steps.

---

## Step 6: Modify `vanilla/data.py` — Support broadcast + distributed sampling

**File to modify:** `transformer_architectures/architectures/vanilla/data.py`

Changes:
1. **`TransformerDataset.from_arrays(cls, arrays, tokenizer)`** — classmethod that sets the 4 numpy arrays directly, bypassing `_setup_arrays()`. Used by non-rank-0 workers.

2. **`TransformerDataModule.train_dataloader()`** — accept optional `rank`/`world_size` params. When provided and `token_budget` is set, use `DistributedTokenBudgetBatchSampler`. When provided without token_budget, use a `DistributedSampler`.

---

## Step 7: Modify `training/checkpointing.py` — DDP compatibility

**File to modify:** `transformer_architectures/training/checkpointing.py`

Changes:
1. `save_checkpoint()`: auto-unwrap DDP model (`model.module` if present) before calling `state_dict()`
2. `save_checkpoint()`: accept optional `run_name`/`run_id` params with fallback to `mlflow.active_run()`. This lets the distributed loop pass these explicitly.
3. `load_checkpoint()`: works as-is (loads into unwrapped model before DDP wrapping)

The rank-0-only guard is applied at the **call site** in `distributed_train_loop()`, not inside `save_checkpoint()` itself.

---

## Step 8: Modify `training/grad_logging.py` — DDP compatibility

**File to modify:** `transformer_architectures/training/grad_logging.py`

Changes:
1. `log_grads()`: unwrap DDP model before iterating `named_parameters()`
2. The rank-0-only guard is applied at the call site in `distributed_train_loop()`

---

## Step 9: Create `vanilla/distributed_training.py` — Vanilla entry point

**New file:** `transformer_architectures/architectures/vanilla/distributed_training.py`

Implements `VanillaTrainable` (the `TrainableArchitecture` protocol):
- `build_model()` → delegates to existing `make_tokenizer_and_model()`
- `build_datasets()` → uses `load_and_broadcast()` with rank 0 calling `load_data()` + tokenization, then all ranks split via `train_val_test_split()` with the same seed
- `build_train_dataloader()` → uses `DistributedTokenBudgetBatchSampler` or `DistributedSampler`
- `train_step()` → the forward/loss computation currently in `train_epoch()`
- `evaluate()` → the BLEU/GLEU logic currently in `evaluate()`

Entry point:
```bash
uv run python -m transformer_architectures.architectures.vanilla.distributed_training
```

---

## Step 10: Create distributed config

**New file:** `configs/vanilla_large_distributed.yaml`

Same as `vanilla_large.yaml` plus a `Distributed:` section:
```yaml
Distributed:
  num_workers: 4      # adjust to cluster size
  use_gpu: true
  backend: nccl
```

Note: with N GPUs, effective batch = `N * batch_size * grad_accumulation_steps`. Adjust `grad_accumulation_steps` to maintain the same effective batch size (e.g., 4 GPUs → reduce from 24 to 6).

---

## Key Design Decisions

### Gradient accumulation + DDP
Use `model.no_sync()` context manager during accumulation micro-steps. This skips the all-reduce on non-step iterations, saving significant communication overhead.

### MLflow
Only rank 0 interacts with MLflow. All `mlflow.*` calls in `distributed_train_loop()` are guarded by `if ctx.is_primary`. The existing `grad_logging.py` is only called from rank 0.

### Existing single-GPU scripts untouched
`vanilla/training.py` and `vit/training.py` remain as-is. Distributed training is a separate entry point. No risk of breaking existing workflows.

### Ray Train `TorchTrainer` (not raw Ray Core)
Handles process group init, device assignment, and failure recovery automatically. Each architecture just provides a `train_loop_per_worker` function.

---

## New/Modified Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Modify | Add `ray[train]` dependency |
| `training/distributed.py` | **Create** | `DistributedContext`, `TrainableArchitecture` protocol, `distributed_train_loop` |
| `training/data_broadcast.py` | **Create** | `load_and_broadcast()` for slow-mount data distribution |
| `training/ray_launcher.py` | **Create** | `RayConfig`, `launch()` with `TorchTrainer` |
| `samplers/bucket_sampler.py` | Modify | Add `DistributedTokenBudgetBatchSampler` |
| `vanilla/data.py` | Modify | Add `from_arrays()`, distributed sampler support |
| `training/checkpointing.py` | Modify | DDP unwrap, optional MLflow run params |
| `training/grad_logging.py` | Modify | DDP unwrap |
| `vanilla/distributed_training.py` | **Create** | `VanillaTrainable` + entry point |
| `configs/vanilla_large_distributed.yaml` | **Create** | Config with `Distributed:` section |

---

## Verification

1. **Smoke test (2 workers, small data):**
   ```bash
   # Override num_samples to 1000 for quick validation
   uv run python -m transformer_architectures.architectures.vanilla.distributed_training
   ```
   Verify: both workers start, data broadcasts, loss decreases, MLflow shows one run, checkpoint saves.

2. **Batch sync test:** Confirm all ranks process the same number of optimizer steps per epoch (log `len(dataloader)` from each rank).

3. **Equivalence test:** Compare loss curves: 1 GPU with `grad_accum=24` vs 4 GPUs with `grad_accum=6` (same effective batch of 768). Should be nearly identical.

4. **Checkpoint round-trip:** Save checkpoint mid-training, restart from checkpoint, verify training resumes correctly.
