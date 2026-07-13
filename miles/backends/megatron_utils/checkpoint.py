import logging
import os
import re
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args
from miles.utils import megatron_bridge_utils

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # Load LoRA checkpoint if configured
    if getattr(args, "lora_rank", 0) > 0 and getattr(args, "lora_checkpoint", None):
        _load_lora_checkpoint(ddp_model, args)

    return result


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    from megatron.bridge import AutoBridge
    import miles_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")
    bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)


def _load_lora_checkpoint(ddp_model, args):
    """Load LoRA checkpoint into the model.

    Args:
        ddp_model: List of DDP-wrapped model chunks
        args: Training arguments with lora_checkpoint path
    """
    from .lora import load_lora_checkpoint

    lora_path = args.lora_checkpoint
    logger.info(f"Loading LoRA checkpoint from {lora_path}")

    for model_chunk in ddp_model:
        # Access the underlying module (unwrap DDP)
        model = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        load_lora_checkpoint(model, lora_path, args=args, strict=False)
