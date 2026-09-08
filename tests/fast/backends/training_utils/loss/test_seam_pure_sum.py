"""Tinker seam reduction.

With `_loss_norm_total` in the batch every datum contributes a token SUM over
its mask (client weights ride in the mask, advantages in the term), the
normalizer handed to Megatron is 1, and the log count stays the sample count.
Without the key the native sample-mean path is untouched (the snapshot tests
cover its values).
"""

from __future__ import annotations

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.loss_hub.losses import policy_loss_function, sft_loss_function

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

SEED = 7
VOCAB_SIZE = 128
LOSS_FNS = {"sft_loss": sft_loss_function, "policy_loss": policy_loss_function}


def _build(loss_type: str):
    batch_size, prompt_lens, response_lens = 3, [20, 64, 40], [10, 48, 32]
    args = make_args(global_batch_size=batch_size, advantage_estimator="grpo", loss_type=loss_type)
    make_parallel_state()
    inputs = make_inputs(
        seed=SEED,
        batch_size=batch_size,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        vocab_size=VOCAB_SIZE,
        args=args,
    )
    return args, inputs


def _run(args, inputs, loss_type, masks=None, seam=True):
    batch = make_batch(inputs, loss_type)
    if masks is not None:
        batch["loss_masks"] = masks
    if seam:
        batch["_loss_norm_total"] = 1
    loss, normalizer, log = loss_function(args, batch, 1, deep_clone(inputs["policy_logits"]))
    return loss, normalizer, log, batch


def _reference(args, inputs, batch, loss_type, per_token: bool):
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        per_token,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )
    loss, _ = LOSS_FNS[loss_type](args, batch, deep_clone(inputs["policy_logits"]), reducer)
    return loss


@pytest.mark.parametrize("loss_type", ["sft_loss", "policy_loss"])
def test_seam_reduces_each_datum_by_token_sum(loss_type):
    args, inputs = _build(loss_type)
    loss, normalizer, log, batch = _run(args, inputs, loss_type)
    torch.testing.assert_close(loss, _reference(args, inputs, batch, loss_type, per_token=True))
    assert normalizer == 1
    assert int(log["values"][0]) == len(batch["response_lengths"])


@pytest.mark.parametrize("loss_type", ["sft_loss", "policy_loss"])
def test_seam_is_linear_in_client_weights(loss_type):
    # Doubling one datum's weights adds exactly that datum's contribution once
    # more; a per-datum mean would leave the loss unchanged.
    args, inputs = _build(loss_type)
    base, *_ = _run(args, inputs, loss_type)
    doubled_masks = deep_clone(inputs["loss_masks"])
    doubled_masks[0] = doubled_masks[0] * 2
    doubled, *_ = _run(args, inputs, loss_type, masks=doubled_masks)
    only0_masks = [m if i == 0 else torch.zeros_like(m) for i, m in enumerate(inputs["loss_masks"])]
    only0, *_ = _run(args, inputs, loss_type, masks=only0_masks)
    assert only0.abs().item() > 0
    torch.testing.assert_close(doubled - base, only0)


def test_seam_zero_mask_datum_is_inert():
    args, inputs = _build("sft_loss")
    base, *_ = _run(args, inputs, "sft_loss")
    without2_masks = [torch.zeros_like(m) if i == 2 else m for i, m in enumerate(inputs["loss_masks"])]
    without2, *_ = _run(args, inputs, "sft_loss", masks=without2_masks)
    only2_masks = [m if i == 2 else torch.zeros_like(m) for i, m in enumerate(inputs["loss_masks"])]
    only2, *_ = _run(args, inputs, "sft_loss", masks=only2_masks)
    torch.testing.assert_close(without2 + only2, base)


@pytest.mark.parametrize("loss_type", ["sft_loss", "policy_loss"])
def test_native_path_keeps_sample_mean(loss_type):
    args, inputs = _build(loss_type)
    loss, normalizer, _, batch = _run(args, inputs, loss_type, seam=False)
    # native: sum of per-datum means, divided by global_batch_size (dp=1 here)
    expected = _reference(args, inputs, batch, loss_type, per_token=False) / args.global_batch_size
    torch.testing.assert_close(loss, expected)
    assert normalizer == 1
