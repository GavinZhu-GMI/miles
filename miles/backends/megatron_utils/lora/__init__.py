# Copyright 2024 Miles Authors
# SPDX-License-Identifier: Apache-2.0
"""
LoRA (Low-Rank Adaptation) support for Miles/Megatron training.

This module provides:
- LoRA layer wrappers for Megatron's tensor-parallel linear layers
- Injection functions to add LoRA to existing models
- Checkpoint save/load utilities for LoRA adapters
"""

from .lora_layers import LoRAColumnParallelLinear, LoRARowParallelLinear, get_lora_param_count
from .lora_injector import (
    inject_lora_adapters,
    freeze_base_model_parameters,
    unfreeze_all_parameters,
    get_lora_state_dict,
    load_lora_state_dict,
    print_trainable_parameters,
)
from .lora_checkpoint import (
    save_lora_checkpoint,
    load_lora_checkpoint,
    export_lora_for_sglang,
    merge_lora_weights,
)

__all__ = [
    # Layers
    "LoRAColumnParallelLinear",
    "LoRARowParallelLinear",
    "get_lora_param_count",
    # Injection
    "inject_lora_adapters",
    "freeze_base_model_parameters",
    "unfreeze_all_parameters",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "print_trainable_parameters",
    # Checkpoints
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "export_lora_for_sglang",
    "merge_lora_weights",
]
