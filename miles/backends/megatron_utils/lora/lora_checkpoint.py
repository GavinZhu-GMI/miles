# Copyright 2024 Miles Authors
# SPDX-License-Identifier: Apache-2.0
"""
LoRA checkpoint save/load utilities.

This module provides functions to save and load LoRA adapter weights separately
from the base model, enabling efficient checkpoint storage.
"""

import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from megatron.core import mpu

logger = logging.getLogger(__name__)


def save_lora_checkpoint(
    model: nn.Module,
    save_dir: str,
    args: Namespace,
    iteration: int,
) -> Optional[str]:
    """Save LoRA adapter weights in Megatron native format.

    Only saves parameters with 'lora_' in their name. The checkpoint is saved
    by the data parallel rank 0 process.

    Args:
        model: The model containing LoRA adapters
        save_dir: Base directory to save checkpoints
        args: Training arguments (for LoRA config info)
        iteration: Current training iteration

    Returns:
        Path to saved checkpoint, or None if not rank 0
    """
    # Only rank 0 of data parallel group saves
    if mpu.get_data_parallel_rank() != 0:
        return None

    save_path = Path(save_dir) / f"lora_iter_{iteration:07d}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            # Detach and clone to CPU
            lora_state_dict[name] = param.data.detach().cpu().clone()

    if not lora_state_dict:
        logger.warning("No LoRA parameters found to save")
        return None

    # Save weights
    weights_path = save_path / "lora_weights.pt"
    torch.save(lora_state_dict, weights_path)

    # Save config
    config = {
        "lora_rank": getattr(args, "lora_rank", 0),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", 0.0),
        "iteration": iteration,
        "num_lora_params": len(lora_state_dict),
        "total_lora_elements": sum(p.numel() for p in lora_state_dict.values()),
    }
    config_path = save_path / "lora_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Saved LoRA checkpoint to {save_path}: "
        f"{config['num_lora_params']} params, {config['total_lora_elements']:,} elements"
    )

    return str(save_path)


def load_lora_checkpoint(
    model: nn.Module,
    load_path: str,
    args: Optional[Namespace] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load LoRA adapter weights from checkpoint.

    Args:
        model: The model to load LoRA weights into
        load_path: Path to LoRA checkpoint directory
        args: Optional training arguments (for validation)
        strict: If True, raise error for missing/unexpected keys

    Returns:
        Dictionary with loaded config and statistics

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If strict=True and keys don't match
    """
    load_path = Path(load_path)
    weights_path = load_path / "lora_weights.pt"
    config_path = load_path / "lora_config.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

    # Load config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Validate config if args provided
    if args is not None and config:
        if getattr(args, "lora_rank", 0) != config.get("lora_rank", 0):
            logger.warning(
                f"LoRA rank mismatch: args={args.lora_rank}, checkpoint={config.get('lora_rank')}"
            )

    # Load weights
    lora_state_dict = torch.load(weights_path, map_location="cpu")

    # Get model LoRA parameter names
    model_lora_keys = {name for name, _ in model.named_parameters() if "lora_" in name}
    checkpoint_keys = set(lora_state_dict.keys())

    # Check for mismatches
    missing = model_lora_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_lora_keys

    if strict:
        if missing:
            raise KeyError(f"Missing LoRA keys in checkpoint: {missing}")
        if unexpected:
            raise KeyError(f"Unexpected LoRA keys in checkpoint: {unexpected}")
    else:
        if missing:
            logger.warning(f"Missing LoRA keys (will be initialized randomly): {missing}")
        if unexpected:
            logger.warning(f"Unexpected LoRA keys (will be ignored): {unexpected}")

    # Load weights into model
    loaded_count = 0
    for name, param in model.named_parameters():
        if name in lora_state_dict:
            loaded_tensor = lora_state_dict[name]

            # Handle device and dtype
            loaded_tensor = loaded_tensor.to(device=param.device, dtype=param.dtype)

            # Validate shape
            if loaded_tensor.shape != param.shape:
                logger.error(
                    f"Shape mismatch for {name}: "
                    f"model={param.shape}, checkpoint={loaded_tensor.shape}"
                )
                continue

            param.data.copy_(loaded_tensor)
            loaded_count += 1

    logger.info(
        f"Loaded LoRA checkpoint from {load_path}: "
        f"{loaded_count} params loaded"
    )

    return {
        "config": config,
        "loaded_count": loaded_count,
        "missing": missing,
        "unexpected": unexpected,
    }


def export_lora_for_sglang(
    model: nn.Module,
    args: Namespace,
    output_path: str,
    model_name: Optional[str] = None,
) -> str:
    """Export LoRA weights in PEFT-compatible format for SGLang.

    SGLang expects LoRA adapters in HuggingFace PEFT format:
    - adapter_config.json
    - adapter_model.safetensors (or .bin)

    Args:
        model: The model containing LoRA adapters
        args: Training arguments
        output_path: Directory to save adapter files
        model_name: Name of the base model (for config)

    Returns:
        Path to the exported adapter directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights and convert names to PEFT format
    peft_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            # Convert Megatron naming to PEFT naming
            # Example: decoder.layers.0.self_attention.linear_qkv.lora_A
            # -> base_model.model.model.layers.0.self_attn.q_proj.lora_A
            peft_name = _megatron_to_peft_name(name)
            peft_state_dict[peft_name] = param.data.detach().cpu().clone()

    if not peft_state_dict:
        logger.warning("No LoRA parameters found to export")
        return str(output_path)

    # Save weights
    weights_path = output_path / "adapter_model.bin"
    torch.save(peft_state_dict, weights_path)

    # Create PEFT config
    peft_config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": model_name or getattr(args, "hf_checkpoint", "unknown"),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": getattr(args, "lora_alpha", args.lora_rank),
        "lora_dropout": getattr(args, "lora_dropout", 0.0),
        "modules_to_save": None,
        "r": args.lora_rank,
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": None,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    }

    config_path = output_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(peft_config, f, indent=2)

    logger.info(f"Exported LoRA adapter to {output_path} for SGLang")
    return str(output_path)


def _megatron_to_peft_name(megatron_name: str) -> str:
    """Convert Megatron parameter name to HuggingFace PEFT name.

    Args:
        megatron_name: Parameter name in Megatron format

    Returns:
        Parameter name in PEFT format

    Examples:
        decoder.layers.0.self_attention.linear_qkv.lora_A
        -> base_model.model.model.layers.0.self_attn.qkv_proj.lora_A

        decoder.layers.0.self_attention.linear_proj.lora_B
        -> base_model.model.model.layers.0.self_attn.o_proj.lora_B
    """
    name = megatron_name

    # Replace common patterns
    name = name.replace("decoder.layers.", "base_model.model.model.layers.")
    name = name.replace("self_attention.linear_qkv.", "self_attn.qkv_proj.")
    name = name.replace("self_attention.linear_proj.", "self_attn.o_proj.")
    name = name.replace("self_attention.linear_qgkv.", "self_attn.qgkv_proj.")
    name = name.replace("mlp.linear_fc1.", "mlp.gate_up_proj.")
    name = name.replace("mlp.linear_fc2.", "mlp.down_proj.")

    return name


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA weights into base model weights (in-place).

    After merging, the LoRA adapters are no longer needed and the model
    behaves as a standard model with merged weights.

    This is useful for inference when you want to avoid the LoRA overhead.

    Args:
        model: The model with LoRA adapters to merge
    """
    from .lora_layers import LoRAColumnParallelLinear, LoRARowParallelLinear

    merged_count = 0

    for name, module in model.named_modules():
        if isinstance(module, (LoRAColumnParallelLinear, LoRARowParallelLinear)):
            # Compute merged weight: W_new = W_base + scaling * B @ A
            base_weight = module.base_layer.weight.data
            lora_A = module.lora_A.data
            lora_B = module.lora_B.data
            scaling = module.scaling

            # LoRA contribution: B @ A (output_dim x rank) @ (rank x input_dim)
            # = (output_dim x input_dim)
            lora_weight = scaling * (lora_B @ lora_A)

            # Add to base weight
            base_weight.add_(lora_weight)

            # Zero out LoRA weights to free memory (optional)
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()

            merged_count += 1

    logger.info(f"Merged LoRA weights into {merged_count} layers")
