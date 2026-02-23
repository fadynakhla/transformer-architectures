# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands use `uv run` for the virtual environment managed by `uv`.

```bash
# Format code
make format          # runs isort then black on transformer_architectures/

# Type checking
make typecheck       # runs mypy on transformer_architectures/

# Run training (vanilla transformer - EN→FR translation)
uv run python -m transformer_architectures.architectures.vanilla.training

# Run training (Vision Transformer - image classification)
uv run python -m transformer_architectures.architectures.vit.training

# Install dependencies
uv sync
uv sync --group dev   # include dev dependencies
```

There are no automated tests in this repo.

## Architecture

This is a from-scratch implementation of transformer architectures in PyTorch, structured as a library with reusable primitives and concrete architecture implementations.

### Layer Hierarchy (bottom-up)

```
attention_functions.py         # ScaledDotProductAttention, AdditiveAttention
multihead_attention.py         # MultiHeadAttention, SelfAttentionSubLayer, CrossAttentionSubLayer
feedforward/feedforward.py     # FeedForwardSubLayer
transformer_blocks.py          # TransformerBlock (self-attn + optional cross-attn + FF)
encoder.py / decoder.py        # Encoder / Decoder stacks (N × TransformerBlock)
```

### Concrete Architectures (`transformer_architectures/architectures/`)

Each architecture subdirectory has:
- `transformer.py` — the `nn.Module` model class
- `data.py` — `DataModule`, `Dataset`, `DataCollator`, and batch dataclasses
- `training.py` — training loop (run as `__main__`)
- `__init__.py` — re-exports for the public API

| Directory | Architecture | Task |
|-----------|-------------|------|
| `vanilla/` | Encoder-decoder Transformer | EN→FR translation (WMT) |
| `vit/` | Vision Transformer | Multi-label image classification (OpenImages) |
| `bert/` | BERT-style encoder | Seq/token classification (no training script yet) |

### Key Design Decisions

- **Pre-LayerNorm**: All current architectures use `pre_layernorm=True` (normalize before attention/FF, with a final norm after the stack), matching GPT-style training stability.
- **`TransformerBlock` is shared**: Both encoder and decoder use the same `TransformerBlock`; cross-attention is only added when `is_decoder=True`.
- **Attention mask convention**: Masks are `1` for real tokens, `0` for padding. Causal masks are combined with padding masks in `Transformer.suplement_causal_mask()`. The mask is expanded to `(batch, heads, seq, seq)` inside `MultiHeadAttention.broadcast_mask()`.
- **DataModule pattern**: Inspired by PyTorch Lightning — call `.setup()` to split data, then `.train_dataloader()` / `.test_dataloader()`.

### Configuration

Configs live in `configs/*.yaml` with `Model:` and `Training:` sections. Loaded via `config.load_config(path, section, PydanticModelClass)`. Supports `${ENV_VAR}` interpolation.

### Experiment Tracking

Both architectures use **MLflow** at `http://10.9.9.249:5000`. Each `training.py` sets the tracking URI and experiment at module level, then wraps the training loop in `with mlflow.start_run():`. Gradient/parameter histograms are logged asynchronously via `training/grad_logging.py` (`async_log(model, step)`). Eval metrics are logged with `step=global_step` (not epoch).

### Checkpoints

Saved to `/data/trained/<model_name>/<run_name>_<run_id[:6]>/epoch_<n>.pt` via `training/checkpointing.py`. The `save_checkpoint` function reads the active MLflow run internally via `mlflow.active_run()` — it must be called inside a `with mlflow.start_run():` block. `load_checkpoint` returns `(epoch, global_step, run_id)` and takes a direct `checkpoint_path` string. Use `resume_mlflow_run(run_id, epoch, global_step)` to resume a previous run.

### Data Paths

- WMT EN-FR: `/data/datasets/wmt/en-fr`
- ViT training data: configured in `configs/vision_large.yaml`
