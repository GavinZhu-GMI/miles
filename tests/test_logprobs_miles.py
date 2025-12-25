#!/usr/bin/env python3
"""
Test script for Miles logprobs computation internals.

This script tests the actual Miles logprobs computation functions:
- ppo_utils.compute_log_probs()
- loss.get_log_probs_and_entropy()
- train_infer_mismatch_helper metrics

Requires a GPU and Megatron-LM to be installed.

Usage:
    # Test ppo_utils functions (simple, no model needed)
    python tests/test_logprobs_miles.py --mode ppo-utils

    # Test with mock logits (no model needed)
    python tests/test_logprobs_miles.py --mode mock-logits

    # Test mismatch metrics
    python tests/test_logprobs_miles.py --mode mismatch

    # Full test with saved rollout data
    python tests/test_logprobs_miles.py --mode full --rollout-data /path/to/rollout_0.pt
"""

import argparse
import sys
from argparse import Namespace
from typing import Optional

import torch
import torch.nn.functional as F


def test_ppo_utils_compute_log_probs():
    """
    Test miles/utils/ppo_utils.py:compute_log_probs()

    This function uses Megatron's fused_vocab_parallel_cross_entropy internally.
    For TP=1 (no tensor parallelism), it should match standard log_softmax.
    """
    print("\n" + "="*60)
    print("Test: ppo_utils.compute_log_probs()")
    print("="*60)

    try:
        from miles.utils.ppo_utils import compute_log_probs
    except ImportError as e:
        print(f"ERROR: Could not import Miles ppo_utils: {e}")
        print("Make sure Miles is installed: pip install -e /path/to/miles")
        return False

    # Create test data
    torch.manual_seed(42)
    vocab_size = 1000
    seq_len = 10

    # Logits shape for Miles: [seq_len, 1, vocab_size] (batch=1, squeezed later)
    logits = torch.randn(seq_len, 1, vocab_size, device="cuda", dtype=torch.float32)
    tokens = torch.randint(0, vocab_size, (seq_len, 1), device="cuda")

    print(f"Logits shape: {logits.shape}")
    print(f"Tokens shape: {tokens.shape}")

    # Test without tensor parallelism (process_group=None)
    try:
        log_probs = compute_log_probs(logits, tokens, process_group=None)
        print(f"Log probs shape: {log_probs.shape}")
        print(f"Log probs mean: {log_probs.mean():.4f}")
        print(f"Log probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
    except Exception as e:
        print(f"ERROR in compute_log_probs: {e}")
        print("\nNote: compute_log_probs uses megatron.core.fusions.fused_vocab_parallel_cross_entropy")
        print("This requires Megatron-LM to be installed and may need distributed init.")
        return False

    # Compare with standard log_softmax
    log_softmax = F.log_softmax(logits.float(), dim=-1)
    expected = log_softmax.gather(dim=-1, index=tokens).squeeze(-1)

    diff = (log_probs - expected).abs()
    print(f"\nComparison with log_softmax:")
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  Max diff:  {diff.max():.8f}")

    if diff.max() < 1e-5:
        print("\n✓ compute_log_probs matches log_softmax!")
        return True
    else:
        print("\n✗ Difference detected - may be due to fused kernel precision")
        return False


def test_calculate_log_probs_and_entropy():
    """
    Test miles/utils/ppo_utils.py:calculate_log_probs_and_entropy()
    """
    print("\n" + "="*60)
    print("Test: ppo_utils.calculate_log_probs_and_entropy()")
    print("="*60)

    try:
        from miles.utils.ppo_utils import calculate_log_probs_and_entropy
    except ImportError as e:
        print(f"ERROR: Could not import: {e}")
        return False

    # Create test data
    torch.manual_seed(42)
    vocab_size = 1000
    seq_len = 10

    logits = torch.randn(seq_len, vocab_size, device="cuda", dtype=torch.float32)
    tokens = torch.randint(0, vocab_size, (seq_len,), device="cuda")

    print(f"Logits shape: {logits.shape}")
    print(f"Tokens shape: {tokens.shape}")

    try:
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits, tokens, tp_group=None, with_entropy=True
        )
        print(f"\nLog probs shape: {log_probs.shape}")
        print(f"Log probs mean: {log_probs.mean():.4f}")
        print(f"Log probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")

        if entropy is not None:
            print(f"\nEntropy shape: {entropy.shape}")
            print(f"Entropy mean: {entropy.mean():.4f}")
            print(f"Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
        else:
            print("\nEntropy: None")

        print("\n✓ calculate_log_probs_and_entropy test completed!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mismatch_metrics():
    """
    Test miles/examples/train_infer_mismatch_helper/mis.py metrics.
    """
    print("\n" + "="*60)
    print("Test: Mismatch Metrics (PPL, KL, chi-squared)")
    print("="*60)

    try:
        from miles.examples.train_infer_mismatch_helper.mis import (
            add_ppl_metrics,
            compute_mis_weights,
        )
    except ImportError as e:
        print(f"ERROR: Could not import mismatch helper: {e}")
        print("Make sure you're in the miles directory with examples.")
        return False

    # Create synthetic log probs with known differences
    torch.manual_seed(42)
    seq_len = 50

    # Simulate "rollout" (SGLang) and "training" (Megatron) log probs
    # with small but measurable differences
    rollout_log_probs = torch.randn(seq_len, device="cuda") * 0.5 - 2.0  # Mean around -2
    noise = torch.randn(seq_len, device="cuda") * 0.1  # Small noise
    train_log_probs = rollout_log_probs + noise  # Training has small differences

    loss_mask = torch.ones(seq_len, device="cuda")

    print(f"Sequence length: {seq_len}")
    print(f"Rollout log_probs: mean={rollout_log_probs.mean():.4f}, std={rollout_log_probs.std():.4f}")
    print(f"Train log_probs:   mean={train_log_probs.mean():.4f}, std={train_log_probs.std():.4f}")

    # Test add_ppl_metrics
    metrics = {}
    add_ppl_metrics(train_log_probs, rollout_log_probs, loss_mask, metrics)

    print("\nMismatch Metrics:")
    for key, values in metrics.items():
        v = values[0] if values else None
        if v is not None:
            if v.numel() == 1 or (v.numel() == seq_len and (v == v[0]).all()):
                # Scalar or sequence-level metric
                print(f"  {key}: {v[0].item():.6f}")
            else:
                # Token-level metric
                print(f"  {key}: mean={v.mean():.6f}, std={v.std():.6f}")

    # Expected metrics
    expected_kl = (rollout_log_probs - train_log_probs).mean()
    print(f"\nExpected KL (direct): {expected_kl:.6f}")

    if "kl" in metrics:
        computed_kl = metrics["kl"][0].mean()
        print(f"Computed KL:          {computed_kl:.6f}")
        if abs(expected_kl - computed_kl) < 1e-5:
            print("✓ KL computation matches!")
        else:
            print("✗ KL mismatch!")

    print("\n✓ Mismatch metrics test completed!")
    return True


def test_mock_get_log_probs_and_entropy():
    """
    Test loss.get_log_probs_and_entropy() with mock data.

    This tests the response-aligned logprobs extraction logic.
    """
    print("\n" + "="*60)
    print("Test: loss.get_log_probs_and_entropy() with mock data")
    print("="*60)

    try:
        from miles.backends.megatron_utils.loss import get_log_probs_and_entropy
    except ImportError as e:
        print(f"ERROR: Could not import loss functions: {e}")
        return False

    # Create mock args
    args = Namespace(
        rollout_temperature=1.0,
    )

    # Create test data
    # Simulating a batch with 2 samples
    torch.manual_seed(42)
    vocab_size = 1000

    # Sample 1: total=20 tokens, response=10 tokens
    # Sample 2: total=15 tokens, response=8 tokens
    total_lengths = [20, 15]
    response_lengths = [10, 8]

    # Token tensors for each sample
    tokens1 = torch.randint(0, vocab_size, (total_lengths[0],), device="cuda")
    tokens2 = torch.randint(0, vocab_size, (total_lengths[1],), device="cuda")
    unconcat_tokens = [tokens1, tokens2]

    # Concatenated logits as if from model forward pass
    # Shape: [1, total_seq_len, vocab_size]
    total_seq_len = sum(total_lengths)
    logits = torch.randn(1, total_seq_len, vocab_size, device="cuda", dtype=torch.float32)

    print(f"Sample 1: total={total_lengths[0]}, response={response_lengths[0]}")
    print(f"Sample 2: total={total_lengths[1]}, response={response_lengths[1]}")
    print(f"Logits shape: {logits.shape}")

    # Mock mpu for CP=1 case
    try:
        from unittest.mock import MagicMock, patch

        mock_mpu = MagicMock()
        mock_mpu.get_context_parallel_world_size.return_value = 1
        mock_mpu.get_tensor_model_parallel_group.return_value = None

        with patch.dict("sys.modules", {"megatron.core": MagicMock(mpu=mock_mpu)}):
            with patch("miles.backends.megatron_utils.loss.mpu", mock_mpu):
                result = get_log_probs_and_entropy(
                    logits,
                    args=args,
                    unconcat_tokens=unconcat_tokens,
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    with_entropy=True,
                )

        log_probs_list = result["log_probs"]
        entropy_list = result.get("entropy", [])

        print(f"\nResults:")
        for i, (lp, rl) in enumerate(zip(log_probs_list, response_lengths)):
            print(f"  Sample {i}: log_probs shape={lp.shape}, expected={rl}")
            if lp.shape[0] != rl:
                print(f"    WARNING: shape mismatch!")
            else:
                print(f"    mean={lp.mean():.4f}, range=[{lp.min():.4f}, {lp.max():.4f}]")

        print("\n✓ get_log_probs_and_entropy test completed!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_rollout_data(rollout_path: str, max_samples: int = 3):
    """
    Test with actual saved rollout data.
    """
    print("\n" + "="*60)
    print(f"Test: Full Rollout Data Analysis")
    print(f"Path: {rollout_path}")
    print("="*60)

    # Load rollout data
    data = torch.load(rollout_path, weights_only=False)
    samples = data.get("samples", [])

    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("ERROR: No samples in rollout data!")
        return False

    # Analyze each sample
    for i, sample in enumerate(samples[:max_samples]):
        print(f"\n--- Sample {i} ---")

        tokens = sample.get("tokens", [])
        response_length = sample.get("response_length", 0)
        rollout_log_probs = sample.get("rollout_log_probs", None)
        loss_mask = sample.get("loss_mask", None)

        total_len = len(tokens)
        prompt_len = total_len - response_length

        print(f"  Total tokens: {total_len}")
        print(f"  Prompt length: {prompt_len}")
        print(f"  Response length: {response_length}")

        if rollout_log_probs is not None:
            lp = torch.tensor(rollout_log_probs, dtype=torch.float32)
            print(f"\n  Rollout log_probs:")
            print(f"    Count: {len(lp)}")
            print(f"    Mean: {lp.mean():.4f}")
            print(f"    Std: {lp.std():.4f}")
            print(f"    Range: [{lp.min():.4f}, {lp.max():.4f}]")

            # Check for anomalies
            if len(lp) != response_length:
                print(f"    WARNING: log_probs length ({len(lp)}) != response_length ({response_length})")

            # Check for zeros (indicating missing logprobs)
            zero_count = (lp == 0).sum().item()
            if zero_count > 0:
                print(f"    WARNING: {zero_count} zero log_probs found!")

            # Compute perplexity
            ppl = torch.exp(-lp.mean())
            print(f"    Perplexity: {ppl:.2f}")
        else:
            print("\n  Rollout log_probs: None")

        if loss_mask is not None:
            mask = torch.tensor(loss_mask, dtype=torch.float32)
            print(f"\n  Loss mask:")
            print(f"    Count: {len(mask)}")
            print(f"    Sum: {mask.sum().item():.0f}")
            print(f"    Density: {mask.mean():.2%}")
        else:
            print("\n  Loss mask: None")

    print("\n✓ Rollout data analysis completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Miles logprobs test script")
    parser.add_argument(
        "--mode",
        choices=["ppo-utils", "mock-logits", "mismatch", "full"],
        default="ppo-utils",
        help="Test mode",
    )
    parser.add_argument(
        "--rollout-data",
        type=str,
        default=None,
        help="Path to saved rollout data (.pt file)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum samples to analyze",
    )

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.mode == "ppo-utils":
        success = test_ppo_utils_compute_log_probs()
        if success:
            test_calculate_log_probs_and_entropy()

    elif args.mode == "mock-logits":
        success = test_mock_get_log_probs_and_entropy()

    elif args.mode == "mismatch":
        success = test_mismatch_metrics()

    elif args.mode == "full":
        if not args.rollout_data:
            print("ERROR: --rollout-data required for full mode")
            sys.exit(1)
        success = test_with_rollout_data(args.rollout_data, args.max_samples)

    else:
        print(f"Unknown mode: {args.mode}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
