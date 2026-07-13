# Copyright 2024 Miles Authors
# SPDX-License-Identifier: Apache-2.0
"""
LoRA (Low-Rank Adaptation) injection for Megatron GPTModel.

This module provides functions to:
1. Inject LoRA adapters into attention layers of an existing GPTModel
2. Freeze base model parameters for LoRA training

Supports both:
- Megatron base layers (ColumnParallelLinear, RowParallelLinear)
- Transformer Engine layers (TEColumnParallelLinear, TERowParallelLinear)
"""

import logging
from argparse import Namespace
from typing import Set, Optional

import torch.nn as nn
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

# Import TE layers with fallback if TE not available
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None
    HAVE_TE = False

from .lora_layers import LoRAColumnParallelLinear, LoRARowParallelLinear, get_lora_param_count

logger = logging.getLogger(__name__)

# Default target modules for attention (Q/K/V/O projections)
DEFAULT_TARGET_MODULES = {"linear_qkv", "linear_proj"}


def _get_column_parallel_types():
    """Get tuple of column parallel layer types (Megatron + TE if available)."""
    types = (ColumnParallelLinear,)
    if HAVE_TE and TEColumnParallelLinear is not None:
        types = types + (TEColumnParallelLinear,)
    return types


def _get_row_parallel_types():
    """Get tuple of row parallel layer types (Megatron + TE if available)."""
    types = (RowParallelLinear,)
    if HAVE_TE and TERowParallelLinear is not None:
        types = types + (TERowParallelLinear,)
    return types


def inject_lora_adapters(
    model: GPTModel,
    args: Namespace,
    target_modules: Optional[Set[str]] = None,
) -> GPTModel:
    """Inject LoRA adapters into a GPTModel's attention layers.

    This function wraps attention linear layers with LoRA adapters.
    The base weights remain unchanged but will be frozen separately.

    Args:
        model: The GPTModel to inject LoRA into
        args: Namespace containing:
            - lora_rank: LoRA rank (r)
            - lora_alpha: LoRA alpha scaling factor (defaults to lora_rank)
            - lora_dropout: LoRA dropout rate
        target_modules: Set of module names to apply LoRA to.
            Default is {"linear_qkv", "linear_proj"} for attention.

    Returns:
        The same model with LoRA adapters injected
    """
    lora_rank = getattr(args, "lora_rank", 0)
    if lora_rank <= 0:
        logger.info("LoRA disabled (lora_rank <= 0)")
        return model

    lora_alpha = getattr(args, "lora_alpha", None)
    if lora_alpha is None:
        lora_alpha = lora_rank  # Default: scaling = 1.0

    lora_dropout = getattr(args, "lora_dropout", 0.0)

    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    logger.info(
        f"Injecting LoRA adapters: rank={lora_rank}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules}"
    )

    lora_count = 0

    # Access the decoder (TransformerBlock) which contains the layers
    if not hasattr(model, "decoder"):
        logger.warning("Model has no 'decoder' attribute, skipping LoRA injection")
        return model

    decoder = model.decoder

    if not hasattr(decoder, "layers"):
        logger.warning("Decoder has no 'layers' attribute, skipping LoRA injection")
        return model

    # Iterate through all transformer layers
    num_layers = len(decoder.layers)
    logger.info(f"LoRA injection: Found {num_layers} decoder layers to process")

    for layer_idx, layer in enumerate(decoder.layers):
        # Check for self_attention module
        if not hasattr(layer, "self_attention"):
            if layer_idx == 0:  # Only log once to avoid spam
                logger.warning(f"Layer 0 has no 'self_attention'. Available attrs: {[a for a in dir(layer) if not a.startswith('_')][:20]}")
            continue

        attn = layer.self_attention

        # Debug: log what we find in attention module for first layer
        if layer_idx == 0:
            attn_attrs = [a for a in dir(attn) if not a.startswith('_')]
            logger.info(f"Attention module attrs (first 20): {attn_attrs[:20]}")
            if hasattr(attn, "linear_qkv"):
                logger.info(f"linear_qkv type: {type(attn.linear_qkv)}")
            if hasattr(attn, "linear_proj"):
                logger.info(f"linear_proj type: {type(attn.linear_proj)}")

        # Get layer types to check (includes TE types if available)
        column_parallel_types = _get_column_parallel_types()
        row_parallel_types = _get_row_parallel_types()

        # Inject LoRA into linear_qkv (Q/K/V combined projection)
        if "linear_qkv" in target_modules:
            if hasattr(attn, "linear_qkv") and isinstance(attn.linear_qkv, column_parallel_types):
                attn.linear_qkv = LoRAColumnParallelLinear(
                    base_layer=attn.linear_qkv,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                lora_count += 1
                logger.debug(f"Layer {layer_idx}: Wrapped linear_qkv with LoRA")

            # Handle gated attention variant (linear_qgkv)
            elif hasattr(attn, "linear_qgkv") and isinstance(attn.linear_qgkv, column_parallel_types):
                attn.linear_qgkv = LoRAColumnParallelLinear(
                    base_layer=attn.linear_qgkv,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                lora_count += 1
                logger.debug(f"Layer {layer_idx}: Wrapped linear_qgkv with LoRA")

        # Inject LoRA into linear_proj (output projection)
        if "linear_proj" in target_modules:
            if hasattr(attn, "linear_proj") and isinstance(attn.linear_proj, row_parallel_types):
                attn.linear_proj = LoRARowParallelLinear(
                    base_layer=attn.linear_proj,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                lora_count += 1
                logger.debug(f"Layer {layer_idx}: Wrapped linear_proj with LoRA")

    # Log parameter counts
    lora_params, total_params = get_lora_param_count(model)
    logger.info(
        f"Injected {lora_count} LoRA adapters. "
        f"LoRA params: {lora_params:,} ({100*lora_params/total_params:.2f}% of total)"
    )

    return model


def freeze_base_model_parameters(model: nn.Module) -> tuple[int, int]:
    """Freeze all parameters except LoRA adapters.

    After calling this function, only parameters with 'lora_' in their name
    will have requires_grad=True.

    Args:
        model: The model to freeze

    Returns:
        Tuple of (frozen_count, trainable_count) parameter counts
    """
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(
        f"Parameter freezing complete: "
        f"Frozen {frozen_count:,} params, {trainable_count:,} LoRA params trainable"
    )

    return frozen_count, trainable_count


def unfreeze_all_parameters(model: nn.Module) -> None:
    """Unfreeze all parameters (for full fine-tuning or debugging).

    Args:
        model: The model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters from model state dict.

    Args:
        model: The model to extract LoRA params from

    Returns:
        Dictionary containing only LoRA parameters
    """
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict, strict: bool = True) -> None:
    """Load LoRA parameters into model.

    Args:
        model: The model to load LoRA params into
        lora_state_dict: Dictionary containing LoRA parameters
        strict: If True, raise error for missing/unexpected keys
    """
    model_lora_keys = {name for name, _ in model.named_parameters() if "lora_" in name}
    dict_keys = set(lora_state_dict.keys())

    if strict:
        missing = model_lora_keys - dict_keys
        unexpected = dict_keys - model_lora_keys
        if missing:
            raise KeyError(f"Missing LoRA keys: {missing}")
        if unexpected:
            raise KeyError(f"Unexpected LoRA keys: {unexpected}")

    for name, param in model.named_parameters():
        if name in lora_state_dict:
            param.data.copy_(lora_state_dict[name])
            logger.debug(f"Loaded LoRA param: {name}")


def print_trainable_parameters(model: nn.Module) -> None:
    """Print trainable parameter statistics.

    Args:
        model: The model to analyze
    """
    trainable_params = 0
    all_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                lora_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_params:,} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}% || "
        f"lora params: {lora_params:,}"
    )
