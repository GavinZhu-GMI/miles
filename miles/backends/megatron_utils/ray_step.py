"""
Helper routines for Miles Ray actors.

NOTE: This duplicates logic that also lives inside `model.py`/`train_one_step`.
We extracted it here to share code between gradient-accumulation helpers, but
the implementation still needs real refactoring to avoid copy/paste.
"""

import math
import os
from functools import partial
from typing import Callable, Dict, Tuple

import torch
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.global_vars import get_args

from .data import get_batch
from .loss import get_log_probs_and_entropy, loss_function
from .model import forward_only


def _build_forward_step_fn(actor, args, num_microbatches_for_step):
    """
    Build the forward_step_fn closure used by Megatron's pipeline engine.

    This is shared by the training path and the forward_backward_only helper.

    Args:
        actor: The training actor
        args: Megatron args
        num_microbatches_for_step: Number of microbatches for THIS gradient accumulation step
                                   (scalar, not the full list)
    """

    def forward_step(
        iterator: "DataIterator",
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> Tuple[torch.Tensor, Callable]:
        batch = get_batch(
            iterator,
            [
                "tokens",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                # NOTE: _loss_type_override is scalar metadata, propagated via
                # rollout_data in get_batch() (like _actual_global_batch_size)
            ],
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
        else:
            old_stage = None

        def build_loss_mask_for_mtp(batch_data: dict[str, object]) -> torch.Tensor | None:
            tokens_tensor: torch.Tensor = batch_data["tokens"]

            mask_chunks: list[torch.Tensor] = []
            for total_len, response_len, resp_mask in zip(
                batch_data["total_lengths"],
                batch_data["response_lengths"],
                batch_data["loss_masks"],
            ):
                assert (
                    resp_mask.numel() == response_len
                ), f"Unexpected loss mask size {resp_mask.numel()} (expected {response_len})."
                prompt_len = total_len - response_len
                full_mask = resp_mask.new_zeros(total_len)
                full_mask[prompt_len:] = resp_mask

                from .cp_utils import slice_with_cp  # local import to avoid cycles

                mask_chunks.append(slice_with_cp(full_mask, 0.0))

            flattened_mask = torch.cat(mask_chunks, dim=0)
            seq_len = tokens_tensor.size(-1)
            assert flattened_mask.numel() <= seq_len, (
                f"MTP loss mask ({flattened_mask.numel()}) exceeds token length ({seq_len})."
            )

            loss_mask_tensor = flattened_mask.new_zeros(seq_len)
            loss_mask_tensor[: flattened_mask.numel()] = flattened_mask
            return loss_mask_tensor.unsqueeze(0)

        loss_mask = None
        mtp_kwargs = None

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should be disabled with combined 1f1b"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
            )
        else:
            if args.enable_mtp_training:
                loss_mask = build_loss_mask_for_mtp(batch)
                assert loss_mask.shape == batch["tokens"].shape, (
                    f"loss_mask shape {loss_mask.shape} mismatches token shape {batch['tokens'].shape}"
                )
                mtp_kwargs = {
                    "mtp_labels": batch["tokens"],
                }

            output_tensor = model(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=loss_mask,
                **(dict(mtp_kwargs=mtp_kwargs) if mtp_kwargs is not None else {}),
            )

        if old_stage is not None:
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches_for_step)

    return forward_step


def run_forward_backward_only(actor, rollout_id, data_iterator, num_microbatches, zero_grads):
    """Execute forward/backward without optimizer.step() for gradient accumulation.

    This function handles multiple gradient accumulation steps when batch size exceeds
    global_batch_size. It loops over all steps in num_microbatches and aggregates
    losses and logprobs from all steps.

    IMPORTANT: When using dynamic batch sizing with sequence-length balancing,
    samples are reordered for efficiency. This function reorders logprobs back
    to original sample order before returning.
    """
    args = get_args()
    num_steps = len(num_microbatches)
    total_microbatches = sum(num_microbatches)
    # print(f"[MILES DEBUG] run_forward_backward_only: num_microbatches={num_microbatches}, "
    #       f"num_steps={num_steps}, total_microbatches={total_microbatches}", flush=True)

    # Get processing order for reordering logprobs later
    # data_iterator is a list of DataIterator (one per VPP stage)
    processing_order = None
    if data_iterator and len(data_iterator) > 0:
        processing_order = data_iterator[0].get_processing_order()
        if processing_order is not None:
            # print(f"[MILES DEBUG] Processing order (first 10): {processing_order[:10]}...", flush=True)
            pass

    if zero_grads:
        for model_chunk in actor.model:
            model_chunk.zero_grad_buffer()
        actor.optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from miles.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, 0, actor.model, actor.optimizer, actor.opt_param_scheduler)

    forward_backward_func = get_forward_backward_func()

    # Aggregate results across all gradient accumulation steps
    all_losses_reduced = []

    for step_id in range(num_steps):
        num_mbs = num_microbatches[step_id]
        # print(f"[MILES DEBUG] Step {step_id}/{num_steps}: num_microbatches={num_mbs}", flush=True)

        # Build forward_step_fn with the correct microbatch count for this step
        forward_step = _build_forward_step_fn(actor, args, num_mbs)

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=actor.model,
            num_microbatches=num_mbs,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
        all_losses_reduced.extend(losses_reduced)
        # print(f"[MILES DEBUG] Step {step_id}: got {len(losses_reduced)} loss entries", flush=True)

    valid_step = True
    grad_norm = None
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = actor.optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = actor.optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    loss_dict: Dict[str, float | list[torch.Tensor]] = {}
    if mpu.is_pipeline_last_stage(ignore_virtual=True) and mpu.get_tensor_model_parallel_rank() == 0:
        # Aggregate losses from all steps
        keys = all_losses_reduced[0]["keys"]
        # Check if data was gathered (Tinker path) - if so, don't multiply by cp_size
        # because all CP ranks already have the same full values after gathering
        with_tinker = all_losses_reduced[0].get("_with_tinker", False)
        # print(f"[MILES DEBUG] loss_dict keys from loss_function: {keys}", flush=True)
        values = None
        for item in all_losses_reduced:
            if values is None:
                values = item["values"]
            else:
                values += item["values"]
        assert len(keys) + 1 == values.numel()
        torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))

        values = values.tolist()
        num_samples_or_tokens = values[0]
        cp_size = mpu.get_context_parallel_world_size()
        for key, value in zip(keys, values[1:]):
            if with_tinker and cp_size > 1:
                # Tinker path with CP: data was gathered, all ranks have same full values
                # After all_reduce, values are summed (cp_size * original_value)
                # Don't multiply by cp_size, just divide by the summed count
                loss_dict[key] = value / num_samples_or_tokens
            else:
                # Native path: each rank has partial data, need to scale by cp_size
                loss_dict[key] = value * cp_size / num_samples_or_tokens

        # Aggregate logprobs from ALL steps (not just first)
        if "log_probs" in all_losses_reduced[0]:
            all_log_probs = []
            # print(f"[MILES DEBUG] all_losses_reduced has {len(all_losses_reduced)} entries (across {num_steps} steps)", flush=True)
            for idx, entry in enumerate(all_losses_reduced):
                lp = entry.get("log_probs", [])
                # print(f"[MILES DEBUG] Entry {idx}: log_probs count = {len(lp) if lp else 0}", flush=True)
                if "log_probs" in entry and entry["log_probs"]:
                    all_log_probs.extend(entry["log_probs"])
            # print(f"[MILES DEBUG] Total all_log_probs: {len(all_log_probs)}", flush=True)

            # REORDER logprobs from processed order back to original sample order
            # This is necessary because sequence-length balancing reorders samples
            # for efficiency, but callers expect logprobs in original order.
            if all_log_probs and processing_order is not None:
                num_samples = len(all_log_probs)
                if len(processing_order) == num_samples:
                    # processing_order[processed_pos] = original_idx
                    # We need: reordered[original_idx] = all_log_probs[processed_pos]
                    reordered_log_probs = [None] * num_samples
                    for processed_pos, original_idx in enumerate(processing_order):
                        reordered_log_probs[original_idx] = all_log_probs[processed_pos]

                    # Verify no None entries (all samples accounted for)
                    if all(lp is not None for lp in reordered_log_probs):
                        all_log_probs = reordered_log_probs
                        # print(f"[MILES DEBUG] Reordered {num_samples} logprobs from processed to original order", flush=True)
                    else:
                        # print(f"[MILES DEBUG] WARNING: Some logprobs missing after reorder, keeping processed order", flush=True)
                        pass
                else:
                    # print(f"[MILES DEBUG] WARNING: processing_order len ({len(processing_order)}) != "
                    #       f"logprobs len ({num_samples}), keeping processed order", flush=True)
                    pass

            if all_log_probs:
                loss_dict["log_probs"] = all_log_probs

    return loss_dict, grad_norm, valid_step


def run_forward_only(actor, data_iterator, num_microbatches):
    """Forward-only pass to collect log-probs/entropy for DPO-style flows.

    IMPORTANT: When using dynamic batch sizing with sequence-length balancing,
    samples are reordered for efficiency. This function reorders logprobs back
    to original sample order before returning.
    """
    # Get processing order for reordering logprobs later
    processing_order = None
    if data_iterator and len(data_iterator) > 0:
        processing_order = data_iterator[0].get_processing_order()

    rollout_data_result = forward_only(
        get_log_probs_and_entropy,
        actor.args,
        actor.model,
        data_iterator,
        num_microbatches,
        store_prefix="",
    )

    loss_dict: Dict[str, list[torch.Tensor]] = {}
    # Only TP rank 0 at pipeline-last stage returns log_probs to avoid duplicates
    # across TP ranks (all TP ranks have identical values due to all-reduce).
    # This is important for external APIs (tinkercloud) that aggregate results.
    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        if "log_probs" in rollout_data_result:
            # Move to CPU to avoid device mismatch when aggregated across actors
            log_probs_list = [lp.cpu() for lp in rollout_data_result["log_probs"]]

            # REORDER logprobs from processed order back to original sample order
            if log_probs_list and processing_order is not None:
                num_samples = len(log_probs_list)
                if len(processing_order) == num_samples:
                    reordered = [None] * num_samples
                    for processed_pos, original_idx in enumerate(processing_order):
                        reordered[original_idx] = log_probs_list[processed_pos]
                    if all(lp is not None for lp in reordered):
                        log_probs_list = reordered
                        # print(f"[MILES DEBUG] run_forward_only: Reordered {num_samples} logprobs to original order", flush=True)

            loss_dict["log_probs"] = log_probs_list
        if "entropy" in rollout_data_result:
            entropy_list = [e.cpu() for e in rollout_data_result["entropy"]]

            # REORDER entropy from processed order back to original sample order
            if entropy_list and processing_order is not None:
                num_samples = len(entropy_list)
                if len(processing_order) == num_samples:
                    reordered = [None] * num_samples
                    for processed_pos, original_idx in enumerate(processing_order):
                        reordered[original_idx] = entropy_list[processed_pos]
                    if all(e is not None for e in reordered):
                        entropy_list = reordered

            loss_dict["entropy"] = entropy_list
    return loss_dict
