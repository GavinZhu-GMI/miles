# Copyright 2024 Miles Authors
# SPDX-License-Identifier: Apache-2.0
"""
LoRA (Low-Rank Adaptation) layer wrappers for Megatron's tensor-parallel linear layers.

Design principle: Wrap existing layers rather than modify Megatron core.
LoRA formula: h = W_0 * x + (alpha/r) * B * A * x

Tensor Parallelism Strategy:
- ColumnParallelLinear: LoRA A is full (replicated), LoRA B is sharded (matches output)
- RowParallelLinear: LoRA A is sharded (matches input), LoRA B is full (needs all-reduce)

Supports both:
- Megatron base layers (ColumnParallelLinear, RowParallelLinear)
- Transformer Engine layers (TEColumnParallelLinear, TERowParallelLinear)
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core import mpu
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region


def _is_te_layer(layer: nn.Module) -> bool:
    """Check if layer is a Transformer Engine layer."""
    return 'TE' in type(layer).__name__


def _get_column_parallel_dims(base_layer: nn.Module) -> Tuple[int, int]:
    """Get (in_features, out_features_per_partition) from either Megatron or TE layer."""
    if hasattr(base_layer, 'input_size'):  # Megatron
        return base_layer.input_size, base_layer.output_size_per_partition
    else:  # TE - uses in_features/out_features, output is already partitioned
        return base_layer.in_features, base_layer.out_features


def _get_row_parallel_dims(base_layer: nn.Module) -> Tuple[int, int]:
    """Get (in_features_per_partition, out_features) from either Megatron or TE layer."""
    if hasattr(base_layer, 'input_size_per_partition'):  # Megatron
        return base_layer.input_size_per_partition, base_layer.output_size
    else:  # TE - uses in_features/out_features, input is already partitioned
        return base_layer.in_features, base_layer.out_features


def _get_sequence_parallel(base_layer: nn.Module) -> bool:
    """Get sequence_parallel setting from either Megatron or TE layer.

    Megatron layers store sequence_parallel as a direct instance attribute.
    TE layers store it in base_layer.config.sequence_parallel.
    """
    # Try direct attribute first (Megatron layers)
    if hasattr(base_layer, 'sequence_parallel'):
        return base_layer.sequence_parallel
    # TE layers store it in config
    if hasattr(base_layer, 'config') and hasattr(base_layer.config, 'sequence_parallel'):
        return base_layer.config.sequence_parallel
    return False


class LoRAColumnParallelLinear(nn.Module):
    """Wrap ColumnParallelLinear (or TEColumnParallelLinear) with LoRA adapter.

    ColumnParallelLinear: output is sharded along columns (output_dim / TP).

    LoRA tensor shapes with TP:
    - lora_A: (rank, in_features) - Full input dimension, replicated across TP ranks
    - lora_B: (out_features_per_partition, rank) - Sharded output dimension, matches base

    Args:
        base_layer: The original ColumnParallelLinear or TEColumnParallelLinear layer (will be frozen)
        rank: LoRA rank (r)
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA path
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Detect if this is a TE layer
        self._is_te = _is_te_layer(base_layer)

        # Get dimensions from base layer (handles both Megatron and TE)
        in_features, out_features_per_partition = _get_column_parallel_dims(base_layer)
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # LoRA A: down-projection (in_features -> rank)
        # Full input dimension, replicated across TP ranks
        # Initialize with Kaiming uniform (same as original LoRA paper)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA B: up-projection (rank -> out_features_per_partition)
        # Sharded output dimension to match base layer
        # Initialize to zero so LoRA starts as identity (no change to base model)
        self.lora_B = nn.Parameter(
            torch.zeros(out_features_per_partition, rank, dtype=dtype, device=device)
        )

        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Copy attributes from base layer that may be needed (with defaults for TE)
        self.gather_output = getattr(base_layer, 'gather_output', False)
        self.skip_bias_add = getattr(base_layer, 'skip_bias_add', False)

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """Forward pass with LoRA.

        Args:
            input_: 3D tensor [sequence, batch, hidden]
            weight: Optional weight tensor (for skip_weight_param_allocation, Megatron only)
            runtime_gather_output: Override gather_output at runtime (Megatron only)

        Returns:
            Tuple of (output, bias)
        """
        # Run base layer (frozen) - handle different forward signatures
        if self._is_te:
            # TE layers: forward(x) returns output directly or (output, bias) tuple
            base_result = self.base_layer(input_)
            if isinstance(base_result, tuple):
                output_parallel, bias = base_result
            else:
                output_parallel, bias = base_result, None
        else:
            # Megatron layers: forward(input_, weight, runtime_gather_output) returns (output, bias)
            output_parallel, bias = self.base_layer(input_, weight, runtime_gather_output)

        # LoRA contribution: scaling * (dropout(x) @ A.T @ B.T)
        # input_: [seq, batch, in_features]
        # lora_A.T: [in_features, rank]
        # lora_B.T: [rank, out_per_partition]
        # Result: [seq, batch, out_per_partition]
        lora_input = self.lora_dropout(input_)
        lora_out = lora_input @ self.lora_A.T @ self.lora_B.T
        lora_out = self.scaling * lora_out

        # Add LoRA contribution to base output
        output_parallel = output_parallel + lora_out

        return output_parallel, bias

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"base={self.base_layer}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f})"
        )


class LoRARowParallelLinear(nn.Module):
    """Wrap RowParallelLinear (or TERowParallelLinear) with LoRA adapter.

    RowParallelLinear: input is sharded (input_dim / TP), output is gathered via all-reduce.

    LoRA tensor shapes with TP:
    - lora_A: (rank, in_features_per_partition) - Sharded input dimension, matches base
    - lora_B: (out_features, rank) - Full output dimension, requires all-reduce

    Args:
        base_layer: The original RowParallelLinear or TERowParallelLinear layer (will be frozen)
        rank: LoRA rank (r)
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA path
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Detect if this is a TE layer
        self._is_te = _is_te_layer(base_layer)

        # Get dimensions from base layer (handles both Megatron and TE)
        in_features_per_partition, out_features = _get_row_parallel_dims(base_layer)
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # LoRA A: down-projection (in_features_per_partition -> rank)
        # Sharded input dimension to match base layer
        # Initialize with Kaiming uniform
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features_per_partition, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA B: up-projection (rank -> out_features)
        # Full output dimension - will be all-reduced like base layer
        # Initialize to zero so LoRA starts as identity
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=device)
        )

        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Copy attributes from base layer (with defaults for TE)
        self.skip_bias_add = getattr(base_layer, 'skip_bias_add', False)
        self.input_is_parallel = getattr(base_layer, 'input_is_parallel', True)
        self.sequence_parallel = _get_sequence_parallel(base_layer)

    def forward(self, input_: torch.Tensor):
        """Forward pass with LoRA.

        Args:
            input_: 3D tensor [sequence, batch, hidden] (may be sharded for TP)

        Returns:
            Tuple of (output, bias)
        """
        # Run base layer (handles all-reduce internally) - handle different forward signatures
        if self._is_te:
            # TE layers: forward(x) returns output directly or (output, bias) tuple
            base_result = self.base_layer(input_)
            if isinstance(base_result, tuple):
                output, bias = base_result
            else:
                output, bias = base_result, None
        else:
            # Megatron layers: forward(input_) returns (output, bias)
            output, bias = self.base_layer(input_)

        # LoRA contribution on sharded input
        # input_: [seq, batch, in_per_partition] (already parallel if input_is_parallel)
        # lora_A.T: [in_per_partition, rank]
        # lora_B.T: [rank, out_features]
        # Result before reduce: [seq, batch, out_features]
        if self.input_is_parallel:
            lora_input = input_
        else:
            # Need to scatter input to match base layer behavior
            # This case is rare - usually input_is_parallel=True for RowParallel
            from megatron.core.tensor_parallel import scatter_to_tensor_model_parallel_region
            lora_input = scatter_to_tensor_model_parallel_region(input_)

        lora_input = self.lora_dropout(lora_input)
        lora_out = lora_input @ self.lora_A.T @ self.lora_B.T

        # Reduce LoRA output to match base layer's reduction strategy
        # (each TP rank has partial sum, need to combine)
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            if self.sequence_parallel:
                # Sequence parallel: reduce-scatter partitions sequence dimension
                from megatron.core.tensor_parallel import reduce_scatter_to_sequence_parallel_region
                lora_out = reduce_scatter_to_sequence_parallel_region(lora_out)
            else:
                # Non-sequence parallel: all-reduce keeps full sequence
                lora_out = reduce_from_tensor_model_parallel_region(lora_out)

        # Scale and add to base output
        lora_out = self.scaling * lora_out
        output = output + lora_out

        return output, bias

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"base={self.base_layer}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f})"
        )


def get_lora_param_count(model: nn.Module) -> tuple[int, int]:
    """Count LoRA and total parameters in a model.

    Args:
        model: The model to analyze

    Returns:
        Tuple of (lora_params, total_params)
    """
    lora_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            lora_params += param.numel()
    return lora_params, total_params
