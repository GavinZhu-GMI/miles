# LoRA (Low-Rank Adaptation) for Miles/Megatron

This document describes the LoRA implementation in Miles for parameter-efficient fine-tuning with Megatron-LM.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Configuration Reference](#configuration-reference)
4. [Architecture Overview](#architecture-overview)
5. [Tensor Parallelism Details](#tensor-parallelism-details)
6. [Training Flow](#training-flow)
7. [SGLang Integration](#sglang-integration)
8. [Checkpoint Management](#checkpoint-management)
9. [API Reference](#api-reference)
10. [Usage Examples](#usage-examples)
11. [Troubleshooting](#troubleshooting)
12. [File Reference](#file-reference)

---

## Introduction

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each layer. Instead of fine-tuning all parameters, LoRA only trains the small adapter matrices:

```
h = W₀x + (α/r) × B × A × x
```

Where:
- `W₀` is the frozen pre-trained weight matrix
- `A ∈ ℝ^(r×d)` and `B ∈ ℝ^(k×r)` are the trainable low-rank matrices
- `r` is the LoRA rank (typically 8-64)
- `α` is the scaling factor

### Benefits

| Benefit | Description |
|---------|-------------|
| **Memory Efficiency** | Only adapter parameters (< 1% of model) require gradients and optimizer states |
| **Faster Training** | Smaller parameter footprint reduces memory bandwidth and communication |
| **Adapter Modularity** | Multiple adapters can be trained and swapped for different tasks |
| **Easy Deployment** | Merge adapters into base weights for zero-overhead inference |

### Memory Savings Example

| Model Size | Full Fine-tuning | LoRA (rank=64) | Savings |
|------------|------------------|----------------|---------|
| 0.5B params | ~4GB optimizer | ~40MB adapters | ~99% |
| 7B params | ~56GB optimizer | ~500MB adapters | ~99% |
| 70B params | ~560GB optimizer | ~5GB adapters | ~99% |

### Supported Models

- Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
- Llama 3/3.1/3.2 (all sizes)
- DeepSeek-R1-Distill-Qwen variants
- Any Megatron GPTModel with standard attention

---

## Quick Start

### Minimal Example

Enable LoRA training by adding the `--lora-rank` argument:

```bash
python train.py \
    --hf-checkpoint Qwen/Qwen2.5-0.5B-Instruct \
    --pretrained-checkpoint /data/models/Qwen2.5-0.5B-Instruct_torch_dist \
    --lora-rank 64 \
    # ... other training args
```

### What Happens

1. Model loads with frozen base weights
2. LoRA adapters injected into attention layers (`linear_qkv`, `linear_proj`)
3. Only LoRA parameters (~0.5% of model) are trained
4. After each optimizer step, LoRA weights sync to SGLang for inference

### Expected Output

```
INFO - Injecting LoRA adapters: rank=64, alpha=64, dropout=0.0, targets={'linear_qkv', 'linear_proj'}
INFO - Injected 48 LoRA adapters. LoRA params: 2,359,296 (0.48% of total)
INFO - Parameter freezing complete: Frozen 493,035,520 params, 2,359,296 LoRA params trainable
```

---

## Configuration Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lora-rank` | int | 0 | LoRA rank (r). Set > 0 to enable LoRA. Typical values: 8, 16, 32, 64 |
| `--lora-alpha` | float | rank | Scaling factor (α). Effective scaling = α/r. Default equals rank for scaling=1.0 |
| `--lora-dropout` | float | 0.0 | Dropout applied to LoRA path during training |
| `--lora-checkpoint` | str | None | Path to load existing LoRA checkpoint for resuming |

### Recommended Settings

```bash
# Conservative (small adapters, minimal risk)
--lora-rank 16 --lora-alpha 16

# Balanced (good capacity vs efficiency)
--lora-rank 64 --lora-alpha 64

# High capacity (more expressive, higher memory)
--lora-rank 128 --lora-alpha 128

# With dropout (regularization for small datasets)
--lora-rank 64 --lora-alpha 128 --lora-dropout 0.05
```

### Scaling Factor

The effective scaling is `α/r`. Higher values = stronger LoRA contribution:

| Rank | Alpha | Scaling | Effect |
|------|-------|---------|--------|
| 64 | 64 | 1.0 | Neutral |
| 64 | 128 | 2.0 | Stronger adaptation |
| 64 | 32 | 0.5 | More conservative |

---

## Architecture Overview

### Target Modules

LoRA is applied to attention projections only:

| Module | Layer Type | Description |
|--------|------------|-------------|
| `linear_qkv` | ColumnParallelLinear | Combined Q/K/V projection |
| `linear_proj` | RowParallelLinear | Output projection (O) |

MLP layers are not targeted by default (can be extended if needed).

### LoRA Layer Wrappers

```
┌─────────────────────────────────────────────────────────┐
│                   LoRA Wrapped Layer                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input x ──┬──────────────────────────────────────┐     │
│            │                                       │     │
│            ▼                                       ▼     │
│   ┌─────────────────┐                    ┌─────────────┐│
│   │ Base Layer (W₀) │                    │ LoRA Path   ││
│   │    (frozen)     │                    │ x→A→B→scale ││
│   └────────┬────────┘                    └──────┬──────┘│
│            │                                     │       │
│            └──────────────┬──────────────────────┘       │
│                           ▼                              │
│                     Output (h)                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Initialization

- **LoRA A**: Kaiming uniform initialization (same as original LoRA paper)
- **LoRA B**: Zero initialization (starts as identity, no initial change to base model)

---

## Tensor Parallelism Details

Miles implements TP-aware LoRA that correctly shards adapter matrices to match Megatron's parallel linear layers.

### ColumnParallelLinear (QKV Projection)

```
Input: [seq, batch, hidden]

            LoRA A                    LoRA B
        ┌──────────────┐         ┌──────────────┐
        │              │         │  (sharded)   │
        │  Full Input  │         │ out/TP x rank│
        │  rank x in   │    →    │              │
        │ (replicated) │         │   TP rank 0  │
        │              │         │   TP rank 1  │
        └──────────────┘         └──────────────┘

Dimensions:
- lora_A: (rank, in_features) - Full input dim, replicated across TP ranks
- lora_B: (out_features/TP, rank) - Sharded output dim, matches base layer
```

### RowParallelLinear (Output Projection)

```
Input: [seq, batch, hidden/TP] (already sharded)

            LoRA A                    LoRA B
        ┌──────────────┐         ┌──────────────┐
        │  (sharded)   │         │              │
        │ rank x in/TP │         │  Full Output │
        │              │    →    │  out x rank  │
        │   TP rank 0  │         │              │
        │   TP rank 1  │         │ + all-reduce │
        └──────────────┘         └──────────────┘

Dimensions:
- lora_A: (rank, in_features/TP) - Sharded input dim, matches base layer
- lora_B: (out_features, rank) - Full output dim, requires all-reduce
```

### TP Sharding Summary

| Layer Type | LoRA A | LoRA B | Communication |
|------------|--------|--------|---------------|
| ColumnParallel | Replicated | Sharded | None |
| RowParallel | Sharded | Replicated | All-reduce on B output |

---

## Training Flow

### Model Creation

```python
# In model_provider.py after GPTModel creation
model = GPTModel(**kwargs)

if getattr(args, 'lora_rank', 0) > 0:
    from .lora import inject_lora_adapters
    model = inject_lora_adapters(model, args)  # Wraps attention layers
```

### Parameter Freezing

```python
# In model.py during setup
if getattr(args, 'lora_rank', 0) > 0:
    from .lora import freeze_base_model_parameters
    for m in model:
        freeze_base_model_parameters(m)  # Only LoRA params have requires_grad=True
```

### Optimizer Configuration

LoRA parameters are automatically excluded from weight decay:

```python
# LoRA params should not have weight decay
no_wd_decay_cond = lambda name, param: "lora_" in name
```

### Gradient Flow

```
Forward:
  input → frozen_base_layer → base_output
  input → lora_A → lora_B → scaled_lora_output
  output = base_output + scaled_lora_output

Backward:
  gradients flow only through LoRA path (base weights frozen)
  ∂L/∂A and ∂L/∂B computed

Optimizer Step:
  Only A and B matrices updated
  Base weights W₀ unchanged
```

---

## SGLang Integration

### Dynamic Adapter Loading

SGLang supports runtime LoRA adapter loading via HTTP API. Miles syncs updated LoRA weights to SGLang after each optimizer step using a tmpfs-based approach.

### Weight Sync Flow

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Forward-Backward (Megatron)                         │
│     └── Compute gradients for LoRA params               │
│                                                          │
│  2. Optimizer Step                                       │
│     └── Update A and B matrices                         │
│                                                          │
│  3. update_lora_weights() ←─────────────────────────────│
│     │                                                    │
│     ▼                                                    │
│  ┌───────────────────────────────────────────────┐      │
│  │ Rank 0 Only:                                  │      │
│  │                                               │      │
│  │ a) Collect LoRA params from model             │      │
│  │    └── decoder.layers.*.self_attention.*.lora_*     │
│  │                                               │      │
│  │ b) Convert to PEFT naming format              │      │
│  │    └── base_model.model.model.layers.*...    │      │
│  │                                               │      │
│  │ c) Write to tmpfs (/dev/shm)                  │      │
│  │    ├── adapter_config.json                   │      │
│  │    └── adapter_model.bin                     │      │
│  │                                               │      │
│  │ d) Call SGLang load_lora_adapter()           │      │
│  │    └── HTTP POST /load_lora_adapter          │      │
│  │                                               │      │
│  │ e) Unload previous adapter version           │      │
│  │    └── HTTP POST /unload_lora_adapter        │      │
│  │                                               │      │
│  │ f) Cleanup old tmpfs directory               │      │
│  └───────────────────────────────────────────────┘      │
│                                                          │
│  4. dist.barrier() - Sync all ranks                     │
│                                                          │
│  5. Sample with updated model (SGLang)                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### SGLang Configuration

SGLang must be configured with `max_lora_rank` at startup:

```python
# In sglang_engine.py _compute_server_args()
if getattr(args, "lora_rank", 0) > 0:
    kwargs["max_lora_rank"] = args.lora_rank
```

### Adapter Versioning

Adapters are versioned (`lora_v1`, `lora_v2`, ...) to enable:
- Atomic switching between versions
- Cleanup of old adapters to free GPU memory
- Rollback if loading fails

---

## Checkpoint Management

### Checkpoint Format

LoRA checkpoints are stored in Megatron native format:

```
lora_iter_0001000/
├── lora_weights.pt     # LoRA parameter tensors
└── lora_config.json    # Configuration metadata
```

### lora_config.json Example

```json
{
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,
  "iteration": 1000,
  "num_lora_params": 48,
  "total_lora_elements": 2359296
}
```

### Save Checkpoint

```python
from miles.backends.megatron_utils.lora import save_lora_checkpoint

save_lora_checkpoint(
    model=model,
    save_dir="/checkpoints/my_training",
    args=args,
    iteration=1000
)
# Creates: /checkpoints/my_training/lora_iter_0001000/
```

### Load Checkpoint

```python
from miles.backends.megatron_utils.lora import load_lora_checkpoint

result = load_lora_checkpoint(
    model=model,
    load_path="/checkpoints/my_training/lora_iter_0001000",
    args=args,
    strict=False  # Allow missing/extra keys
)
print(f"Loaded {result['loaded_count']} parameters")
```

### Export for SGLang/PEFT

```python
from miles.backends.megatron_utils.lora import export_lora_for_sglang

export_lora_for_sglang(
    model=model,
    args=args,
    output_path="/exports/my_adapter",
    model_name="Qwen/Qwen2.5-7B-Instruct"
)
# Creates:
#   /exports/my_adapter/adapter_config.json
#   /exports/my_adapter/adapter_model.bin
```

### Merge into Base Model

```python
from miles.backends.megatron_utils.lora import merge_lora_weights

merge_lora_weights(model)  # In-place merge
# Base weights now contain LoRA contribution
# Model can be used without LoRA overhead
```

---

## API Reference

### Injection Functions

#### `inject_lora_adapters(model, args, target_modules=None) -> GPTModel`

Inject LoRA adapters into a GPTModel's attention layers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | GPTModel | Model to inject LoRA into |
| `args` | Namespace | Training args with `lora_rank`, `lora_alpha`, `lora_dropout` |
| `target_modules` | Set[str] | Modules to target (default: `{"linear_qkv", "linear_proj"}`) |

**Returns**: Same model with LoRA adapters injected

#### `freeze_base_model_parameters(model) -> tuple[int, int]`

Freeze all parameters except LoRA adapters.

**Returns**: Tuple of (frozen_count, trainable_count)

### State Dict Functions

#### `get_lora_state_dict(model) -> dict`

Extract only LoRA parameters from model.

#### `load_lora_state_dict(model, lora_state_dict, strict=True)`

Load LoRA parameters into model.

### Checkpoint Functions

#### `save_lora_checkpoint(model, save_dir, args, iteration) -> Optional[str]`

Save LoRA weights in Megatron native format (rank 0 only).

#### `load_lora_checkpoint(model, load_path, args=None, strict=False) -> dict`

Load LoRA weights from checkpoint.

#### `export_lora_for_sglang(model, args, output_path, model_name=None) -> str`

Export in PEFT format for SGLang.

#### `merge_lora_weights(model) -> None`

Merge LoRA weights into base model (in-place).

### Debug Functions

#### `print_trainable_parameters(model)`

Print trainable parameter statistics.

#### `get_lora_param_count(model) -> tuple[int, int]`

Count LoRA and total parameters.

---

## Usage Examples

### RLVE Training with LoRA

```bash
python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    --hf-checkpoint Qwen/Qwen2.5-7B-Instruct \
    --pretrained-checkpoint /data/models/Qwen2.5-7B-Instruct_torch_dist \
    --save /data/checkpoints/rlve-lora \
    --save-interval 100 \
    --rlve \
    --environment-list Multiplication Sorting Division \
    --lora-rank 64 \
    --lora-alpha 64 \
    --tensor-model-parallel-size 2 \
    --context-parallel-size 2 \
    --lr 1e-5 \
    --num-rollout 100
```

### Resume from LoRA Checkpoint

```bash
python train.py \
    --hf-checkpoint Qwen/Qwen2.5-7B-Instruct \
    --pretrained-checkpoint /data/models/Qwen2.5-7B-Instruct_torch_dist \
    --lora-rank 64 \
    --lora-checkpoint /data/checkpoints/rlve-lora/lora_iter_0001000 \
    # ... other args
```

### Export Adapter for Inference

```python
import torch
from miles.backends.megatron_utils.lora import (
    load_lora_checkpoint,
    export_lora_for_sglang,
)

# Load trained LoRA checkpoint
load_lora_checkpoint(model, "/checkpoints/lora_iter_1000")

# Export for SGLang
export_lora_for_sglang(
    model=model,
    args=args,
    output_path="/adapters/my_rlve_adapter",
    model_name="Qwen/Qwen2.5-7B-Instruct"
)
```

### Merge LoRA into Base Model

```python
from miles.backends.megatron_utils.lora import (
    load_lora_checkpoint,
    merge_lora_weights,
)

# Load LoRA weights
load_lora_checkpoint(model, "/checkpoints/lora_iter_1000")

# Merge into base weights
merge_lora_weights(model)

# Save merged model (now standard Megatron checkpoint)
torch.save(model.state_dict(), "/merged_model/model.pt")
```

---

## Troubleshooting

### Common Issues

#### "No LoRA parameters found"

**Cause**: `--lora-rank` not set or set to 0.

**Fix**: Ensure `--lora-rank 64` (or other positive value) is in your command.

#### Shape Mismatch on Checkpoint Load

**Cause**: Different `--lora-rank` between training and loading.

**Fix**: Use the same `--lora-rank` value, or use `strict=False` to skip mismatched params.

```python
load_lora_checkpoint(model, path, strict=False)
```

#### SGLang Adapter Loading Fails

**Cause**: SGLang not configured with `max_lora_rank`.

**Fix**: Ensure `--lora-rank` is passed when starting Miles (configures SGLang automatically).

**Debug**: Check SGLang logs for LoRA-related errors:
```bash
grep -i lora /data/logs/sglang.log
```

#### Memory Error with LoRA

**Cause**: `--lora-rank` too high for available GPU memory.

**Fix**: Reduce `--lora-rank` (e.g., 64 → 32 → 16).

#### "LoRA rank mismatch" Warning

**Cause**: Checkpoint was saved with different `--lora-rank`.

**Fix**:
1. Use matching `--lora-rank` value
2. Or retrain from scratch
3. Or use `strict=False` if intentional

### Performance Tips

1. **Rank Selection**: Start with rank 64, increase if underfitting, decrease if overfitting or OOM
2. **Alpha Tuning**: Higher alpha = more aggressive adaptation
3. **Dropout**: Add 0.05-0.1 dropout for small datasets to prevent overfitting

### Debug Logging

Enable debug logging for LoRA operations:

```python
import logging
logging.getLogger("miles.backends.megatron_utils.lora").setLevel(logging.DEBUG)
```

---

## File Reference

### Implementation Files

| File | Purpose |
|------|---------|
| `miles/backends/megatron_utils/lora/__init__.py` | Package exports |
| `miles/backends/megatron_utils/lora/lora_layers.py` | TP-aware LoRA layer wrappers |
| `miles/backends/megatron_utils/lora/lora_injector.py` | Injection and parameter freezing |
| `miles/backends/megatron_utils/lora/lora_checkpoint.py` | Checkpoint save/load/export utilities |

### Integration Points

| File | Changes |
|------|---------|
| `miles/backends/megatron_utils/model_provider.py` | Inject LoRA after GPTModel creation |
| `miles/backends/megatron_utils/model.py` | Freeze base params, configure optimizer |
| `miles/backends/megatron_utils/checkpoint.py` | Load LoRA checkpoint on resume |
| `miles/utils/arguments.py` | LoRA CLI arguments (`--lora-*`) |
| `miles/backends/sglang_utils/sglang_engine.py` | `max_lora_rank`, `load_lora_adapter()`, `unload_lora_adapter()` |
| `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | `update_lora_weights()` method |
| `miles/backends/megatron_utils/actor.py` | `update_lora_weights()` Ray method |
| `miles/ray/actor_group.py` | `update_lora_weights()` group method |

### Parameter Naming Convention

| Format | Example |
|--------|---------|
| Megatron | `decoder.layers.0.self_attention.linear_qkv.lora_A` |
| PEFT/HF | `base_model.model.model.layers.0.self_attn.qkv_proj.lora_A` |

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Original paper
- [PEFT Library](https://github.com/huggingface/peft) - HuggingFace PEFT implementation
- [SGLang LoRA Documentation](https://sgl-project.github.io/) - SGLang adapter loading
