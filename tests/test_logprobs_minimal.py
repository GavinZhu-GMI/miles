#!/usr/bin/env python3
"""
Minimal standalone test script for debugging logprobs calculation in Miles.

This script can be used to:
1. Test logprobs computation with a simple HuggingFace model (no Megatron)
2. Load saved rollout data and verify logprobs
3. Compare SGLang rollout logprobs vs recomputed logprobs

Usage:
    # Basic sanity check (no saved data needed)
    python tests/test_logprobs_minimal.py --mode sanity

    # Load saved rollout data and verify
    python tests/test_logprobs_minimal.py --mode verify --rollout-data /path/to/rollout_0.pt

    # Compare with HF model
    python tests/test_logprobs_minimal.py --mode hf --model Qwen/Qwen2.5-0.5B-Instruct

    # Full comparison: HF model vs saved rollout logprobs
    python tests/test_logprobs_minimal.py --mode compare \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --rollout-data /path/to/rollout_0.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def compute_log_probs_simple(logits: torch.Tensor, tokens: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute log probabilities from logits and tokens.

    This is a simplified version of what Miles does in:
    - miles/utils/ppo_utils.py:compute_log_probs()
    - miles/backends/megatron_utils/loss.py:get_log_probs_and_entropy()

    Args:
        logits: Shape [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        tokens: Shape [seq_len] or [batch, seq_len] - the target tokens
        temperature: Sampling temperature (default 1.0)

    Returns:
        log_probs: Shape [seq_len] or [batch, seq_len]
    """
    # Handle batched vs unbatched
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        tokens = tokens.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Apply temperature
    logits = logits / temperature

    # Compute log softmax
    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    # Gather the log probs for the actual tokens
    # tokens shape: [batch, seq_len] -> [batch, seq_len, 1]
    tokens_expanded = tokens.unsqueeze(-1)
    log_probs = log_probs_all.gather(dim=-1, index=tokens_expanded).squeeze(-1)

    if squeeze:
        log_probs = log_probs.squeeze(0)

    return log_probs


def compute_log_probs_from_hf_model(
    model,
    tokenizer,
    tokens: list[int],
    response_length: int,
    temperature: float = 1.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities using a HuggingFace model.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        tokens: Full token sequence (prompt + response)
        response_length: Length of the response portion
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        response_log_probs: Log probs for response tokens
        all_log_probs: Log probs for all tokens (excluding first)
    """
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Logits at position i predict token at position i+1
    # So logits[:-1] predicts tokens[1:]
    shift_logits = logits[:, :-1, :].squeeze(0)  # [seq_len-1, vocab_size]
    shift_tokens = tokens_tensor[:, 1:].squeeze(0)  # [seq_len-1]

    # Compute log probs
    all_log_probs = compute_log_probs_simple(shift_logits, shift_tokens, temperature)

    # Extract response portion
    # Response tokens start at position (total_len - response_len) in original sequence
    # In shifted sequence, they start at (total_len - response_len - 1)
    total_len = len(tokens)
    prompt_len = total_len - response_length

    # The log prob for response token i is at shifted position (prompt_len + i - 1)
    # Because logits[prompt_len-1] predicts tokens[prompt_len] (first response token)
    response_start = prompt_len - 1 if prompt_len > 0 else 0
    response_log_probs = all_log_probs[response_start:response_start + response_length]

    return response_log_probs, all_log_probs


def load_rollout_data(path: str) -> dict:
    """Load saved rollout data from a .pt file."""
    data = torch.load(path, weights_only=False)
    return data


def analyze_rollout_logprobs(rollout_data: dict, max_samples: int = 5) -> None:
    """Analyze the logprobs in a saved rollout file."""
    samples = rollout_data.get("samples", [])

    print(f"\n{'='*60}")
    print(f"Rollout Data Analysis")
    print(f"{'='*60}")
    print(f"Number of samples: {len(samples)}")

    if not samples:
        print("No samples found!")
        return

    # Check what fields are available
    sample_keys = list(samples[0].keys()) if samples else []
    print(f"Sample keys: {sample_keys}")

    for i, sample in enumerate(samples[:max_samples]):
        print(f"\n--- Sample {i} ---")
        tokens = sample.get("tokens", [])
        response_length = sample.get("response_length", 0)
        rollout_log_probs = sample.get("rollout_log_probs", None)
        loss_mask = sample.get("loss_mask", None)

        print(f"  Total tokens: {len(tokens)}")
        print(f"  Response length: {response_length}")
        print(f"  Prompt length: {len(tokens) - response_length}")

        if rollout_log_probs is not None:
            lp = torch.tensor(rollout_log_probs) if not isinstance(rollout_log_probs, torch.Tensor) else rollout_log_probs
            print(f"  Rollout log_probs: count={len(lp)}, mean={lp.mean():.4f}, std={lp.std():.4f}")
            print(f"  Rollout log_probs range: [{lp.min():.4f}, {lp.max():.4f}]")

            # Check if length matches response_length
            if len(lp) != response_length:
                print(f"  WARNING: log_probs length ({len(lp)}) != response_length ({response_length})")
        else:
            print("  Rollout log_probs: None")

        if loss_mask is not None:
            mask = torch.tensor(loss_mask) if not isinstance(loss_mask, torch.Tensor) else loss_mask
            print(f"  Loss mask: count={len(mask)}, sum={mask.sum().item()}")


def test_sanity_check():
    """Basic sanity check for logprobs computation."""
    print("\n" + "="*60)
    print("Sanity Check: Basic Logprobs Computation")
    print("="*60)

    # Create simple test case
    vocab_size = 100
    seq_len = 10

    # Random logits
    torch.manual_seed(42)
    logits = torch.randn(seq_len, vocab_size)
    tokens = torch.randint(0, vocab_size, (seq_len,))

    # Compute log probs
    log_probs = compute_log_probs_simple(logits, tokens)

    print(f"Logits shape: {logits.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Log probs mean: {log_probs.mean():.4f}")
    print(f"Log probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")

    # Verify: log probs should be negative and less than 0
    assert (log_probs <= 0).all(), "Log probs should be <= 0"

    # Verify: manual computation matches
    log_softmax = F.log_softmax(logits.float(), dim=-1)
    expected = log_softmax[torch.arange(seq_len), tokens]
    assert torch.allclose(log_probs, expected), "Log probs mismatch!"

    print("\n✓ Sanity check passed!")
    return True


def test_hf_model_logprobs(model_name: str, device: str = "cuda"):
    """Test logprobs computation with a HuggingFace model."""
    print("\n" + "="*60)
    print(f"HuggingFace Model Test: {model_name}")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return False

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Test with a simple prompt
    prompt = "What is 2 + 2?"
    response = " The answer is 4."
    full_text = prompt + response

    # Tokenize
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    response_length = len(full_tokens) - len(prompt_tokens)

    print(f"\nPrompt: {repr(prompt)}")
    print(f"Response: {repr(response)}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Full tokens: {len(full_tokens)}")
    print(f"Response length: {response_length}")

    # Compute log probs
    response_log_probs, all_log_probs = compute_log_probs_from_hf_model(
        model, tokenizer, full_tokens, response_length, device=device
    )

    print(f"\nResponse log probs: count={len(response_log_probs)}")
    print(f"  Mean: {response_log_probs.mean():.4f}")
    print(f"  Std: {response_log_probs.std():.4f}")
    print(f"  Range: [{response_log_probs.min():.4f}, {response_log_probs.max():.4f}]")

    # Convert to perplexity
    mean_nll = -response_log_probs.mean()
    ppl = torch.exp(mean_nll)
    print(f"  Perplexity: {ppl.item():.2f}")

    # Show per-token breakdown
    print("\nPer-token log probs:")
    response_tokens = full_tokens[-response_length:]
    for i, (token_id, lp) in enumerate(zip(response_tokens, response_log_probs)):
        token_str = tokenizer.decode([token_id])
        print(f"  [{i}] {repr(token_str):15s} (id={token_id:5d}): {lp.item():.4f}")

    print("\n✓ HF model test completed!")
    return True


def test_compare_with_rollout(
    model_name: str,
    rollout_path: str,
    max_samples: int = 3,
    device: str = "cuda",
):
    """Compare HF model logprobs with saved rollout logprobs."""
    print("\n" + "="*60)
    print(f"Comparison: HF Model vs Rollout Logprobs")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return False

    # Load rollout data
    print(f"\nLoading rollout data from {rollout_path}...")
    rollout_data = load_rollout_data(rollout_path)
    samples = rollout_data.get("samples", [])

    if not samples:
        print("ERROR: No samples in rollout data!")
        return False

    # Load model
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Compare samples
    total_diff = 0.0
    total_count = 0

    for i, sample in enumerate(samples[:max_samples]):
        print(f"\n--- Sample {i} ---")

        tokens = sample.get("tokens", [])
        response_length = sample.get("response_length", 0)
        rollout_log_probs = sample.get("rollout_log_probs", None)

        if rollout_log_probs is None:
            print("  Skipping: no rollout_log_probs")
            continue

        rollout_lp = torch.tensor(rollout_log_probs, dtype=torch.float32)

        # Compute with HF model
        hf_response_lp, _ = compute_log_probs_from_hf_model(
            model, tokenizer, tokens, response_length, device=device
        )
        hf_response_lp = hf_response_lp.cpu().float()

        # Ensure same length
        min_len = min(len(rollout_lp), len(hf_response_lp))
        rollout_lp = rollout_lp[:min_len]
        hf_lp = hf_response_lp[:min_len]

        # Compare
        diff = (rollout_lp - hf_lp).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        print(f"  Token count: {min_len}")
        print(f"  Rollout log_probs: mean={rollout_lp.mean():.4f}, std={rollout_lp.std():.4f}")
        print(f"  HF log_probs:      mean={hf_lp.mean():.4f}, std={hf_lp.std():.4f}")
        print(f"  Absolute diff:     mean={mean_diff:.6f}, max={max_diff:.6f}")

        # Check for large differences
        if mean_diff > 0.1:
            print(f"  WARNING: Large mean difference!")
        if max_diff > 1.0:
            print(f"  WARNING: Large max difference!")

        total_diff += mean_diff * min_len
        total_count += min_len

        # Show worst mismatches
        worst_indices = diff.argsort(descending=True)[:5]
        print(f"\n  Top 5 mismatches:")
        for idx in worst_indices:
            token_id = tokens[-response_length + idx.item()]
            token_str = tokenizer.decode([token_id])
            print(f"    [{idx.item()}] {repr(token_str):15s}: rollout={rollout_lp[idx]:.4f}, hf={hf_lp[idx]:.4f}, diff={diff[idx]:.4f}")

    if total_count > 0:
        overall_mean_diff = total_diff / total_count
        print(f"\n{'='*60}")
        print(f"Overall mean absolute difference: {overall_mean_diff:.6f}")

        if overall_mean_diff < 0.01:
            print("✓ Excellent match!")
        elif overall_mean_diff < 0.1:
            print("✓ Good match (small numerical differences)")
        else:
            print("✗ Significant differences detected!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Minimal logprobs test script")
    parser.add_argument(
        "--mode",
        choices=["sanity", "verify", "hf", "compare"],
        default="sanity",
        help="Test mode: sanity (basic check), verify (analyze rollout), hf (test HF model), compare (HF vs rollout)",
    )
    parser.add_argument(
        "--rollout-data",
        type=str,
        default=None,
        help="Path to saved rollout data (.pt file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")

    if args.mode == "sanity":
        success = test_sanity_check()

    elif args.mode == "verify":
        if not args.rollout_data:
            print("ERROR: --rollout-data required for verify mode")
            sys.exit(1)
        rollout_data = load_rollout_data(args.rollout_data)
        analyze_rollout_logprobs(rollout_data, args.max_samples)
        success = True

    elif args.mode == "hf":
        success = test_hf_model_logprobs(args.model, args.device)

    elif args.mode == "compare":
        if not args.rollout_data:
            print("ERROR: --rollout-data required for compare mode")
            sys.exit(1)
        success = test_compare_with_rollout(
            args.model,
            args.rollout_data,
            args.max_samples,
            args.device,
        )

    else:
        print(f"Unknown mode: {args.mode}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
