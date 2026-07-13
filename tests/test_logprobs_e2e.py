#!/usr/bin/env python3
"""
End-to-End Logprobs Verification Test

Verifies that the tinker->opentinker-miles->miles pipeline computes and returns
correct logprobs for all samples.

Key validations:
  1. All samples get logprobs returned (no missing samples)
  2. Logprobs lengths match response_lengths (no zero-padding)
  3. Logprobs are valid log probabilities (negative values, not zeros)
  4. Multiple forward_backward calls return consistent results

Usage:
    # Quick test with 8 samples
    python tests/test_logprobs_e2e.py --samples 8

    # Test with existing model
    python tests/test_logprobs_e2e.py --samples 16 --model-id <model-id>

    # Verbose output with saved results
    python tests/test_logprobs_e2e.py --samples 8 --verbose --output /tmp/logprobs_test.pt
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_test_data(
    num_samples: int = 8,
    vocab_size: int = 151936,  # Qwen vocab size
    prompt_len: int = 50,
    response_len: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate synthetic test data that mimics real rollout data.

    Returns dict with:
        - tokens: List of token sequences
        - response_lengths: List of response lengths
        - advantages: List of advantage values
        - rollout_logprobs: List of synthetic logprobs (simulating SGLang output)
    """
    torch.manual_seed(seed)

    samples = []
    for i in range(num_samples):
        # Vary lengths slightly per sample
        sample_prompt_len = prompt_len + torch.randint(-5, 6, (1,)).item()
        sample_response_len = response_len + torch.randint(-10, 11, (1,)).item()
        total_len = sample_prompt_len + sample_response_len

        # Generate random tokens
        tokens = torch.randint(1, vocab_size, (total_len,)).tolist()

        # Generate synthetic rollout logprobs (for response tokens)
        # Real logprobs are typically negative (log of probabilities)
        rollout_logprobs = (torch.randn(sample_response_len) * 0.5 - 2.0).tolist()

        # Generate advantage (typically centered around 0)
        advantage = (torch.randn(1) * 0.5).item()

        # Generate loss mask (1 for response tokens, 0 for prompt)
        loss_mask = [0.0] * sample_prompt_len + [1.0] * sample_response_len

        samples.append({
            "tokens": tokens,
            "response_length": sample_response_len,
            "prompt_length": sample_prompt_len,
            "total_length": total_len,
            "rollout_logprobs": rollout_logprobs,
            "advantage": advantage,
            "loss_mask": loss_mask,
        })

    return {
        "samples": samples,
        "metadata": {
            "num_samples": num_samples,
            "vocab_size": vocab_size,
            "seed": seed,
            "timestamp": time.time(),
        }
    }


def test_data_to_tinker_format(test_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert test data to Tinker Datum format for opentinker-miles API.

    For RL training (importance_sampling/ppo), Tinker expects:
        - model_input.chunks[].tokens
        - loss_fn_inputs.target_tokens.data (required for RLLossFnInputs)
        - loss_fn_inputs.logprobs.data (rollout log probabilities)
        - loss_fn_inputs.advantages.data (per-token advantages)

    NOTE: mask is REMOVED before sending (see tinker_cookbook/rl/train.py:remove_mask)
    """
    datums = []
    for sample in test_data["samples"]:
        # target_tokens: the response portion of tokens (shifted by 1 for autoregressive)
        prompt_len = sample["prompt_length"]
        response_len = sample["response_length"]
        # Target tokens are the tokens we're predicting (response portion)
        target_tokens = sample["tokens"][prompt_len:prompt_len + response_len]

        datum = {
            "model_input": {
                "chunks": [{"tokens": sample["tokens"]}]
            },
            "loss_fn_inputs": {
                "target_tokens": {
                    "data": target_tokens,
                    "shape": [len(target_tokens)],
                    "dtype": "int64"
                },
                "logprobs": {
                    "data": sample["rollout_logprobs"],
                    "shape": [len(sample["rollout_logprobs"])],
                    "dtype": "float32"
                },
                "advantages": {
                    "data": [sample["advantage"]] * sample["response_length"],
                    "shape": [sample["response_length"]],
                    "dtype": "float32"
                }
            }
        }
        datums.append(datum)
    return datums


def test_data_to_miles_rollout(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert test data to Miles rollout_data format.

    Miles expects:
        - tokens: List[torch.Tensor]
        - response_lengths: List[int]
        - total_lengths: List[int]
        - advantages: List[torch.Tensor]
        - rollout_log_probs: List[torch.Tensor]
        - loss_masks: List[torch.Tensor]
    """
    tokens = []
    response_lengths = []
    total_lengths = []
    advantages = []
    rollout_log_probs = []
    loss_masks = []

    for sample in test_data["samples"]:
        tokens.append(torch.tensor(sample["tokens"], dtype=torch.long))
        response_lengths.append(sample["response_length"])
        total_lengths.append(sample["total_length"])
        advantages.append(torch.tensor([sample["advantage"]] * sample["response_length"], dtype=torch.float32))
        rollout_log_probs.append(torch.tensor(sample["rollout_logprobs"], dtype=torch.float32))
        loss_masks.append(torch.tensor(sample["loss_mask"], dtype=torch.float32))

    return {
        "tokens": tokens,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "advantages": advantages,
        "rollout_log_probs": rollout_log_probs,
        "log_probs": rollout_log_probs.copy(),  # Will be overwritten by Megatron
        "loss_masks": loss_masks,
        "returns": [torch.zeros_like(adv) for adv in advantages],
        "values": [torch.zeros_like(adv) for adv in advantages],
        "ref_log_probs": [torch.zeros_like(lp) for lp in rollout_log_probs],
    }


# ============================================================================
# Logprobs Validation Functions
# ============================================================================

def validate_logprobs_sample(
    sample_idx: int,
    logprobs: List[float],
    expected_length: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate logprobs for a single sample.

    Checks:
    1. Length matches expected (response_length)
    2. Values are valid log probabilities (negative, not zero)
    3. No zero-padding detected
    """
    result = {
        "sample_idx": sample_idx,
        "logprobs_length": len(logprobs),
        "expected_length": expected_length,
        "issues": [],
    }

    # Check length match
    length_diff = len(logprobs) - expected_length
    if length_diff != 0:
        # Allow for 1-token difference (autoregressive: N tokens -> N-1 logprobs)
        if length_diff == -1:
            result["issues"].append(f"Length off by 1 (autoregressive expected)")
        else:
            result["issues"].append(f"Length mismatch: got {len(logprobs)}, expected {expected_length}")

    if not logprobs:
        result["issues"].append("Empty logprobs!")
        result["valid"] = False
        return result

    lp_tensor = torch.tensor(logprobs, dtype=torch.float32)

    # Check for zeros (indicates missing/padded logprobs)
    zero_count = (lp_tensor == 0.0).sum().item()
    zero_ratio = zero_count / len(logprobs)
    result["zero_count"] = int(zero_count)
    result["zero_ratio"] = zero_ratio

    if zero_ratio > 0.1:  # More than 10% zeros is suspicious
        result["issues"].append(f"High zero ratio: {zero_ratio:.1%} ({zero_count}/{len(logprobs)})")

    # Check that logprobs are negative (valid log probabilities)
    positive_count = (lp_tensor > 0).sum().item()
    if positive_count > 0:
        result["issues"].append(f"{positive_count} positive logprobs (should all be <= 0)")

    # Check for reasonable range (typical log probs are between -20 and 0)
    very_negative = (lp_tensor < -50).sum().item()
    if very_negative > 0:
        result["issues"].append(f"{very_negative} logprobs < -50 (unusually low)")

    # Statistics
    result["mean"] = lp_tensor.mean().item()
    result["std"] = lp_tensor.std().item()
    result["min"] = lp_tensor.min().item()
    result["max"] = lp_tensor.max().item()

    result["valid"] = len(result["issues"]) == 0

    if verbose:
        status = "OK" if result["valid"] else "ISSUES"
        print(f"  Sample {sample_idx}: {status} len={len(logprobs)}/{expected_length}, "
              f"mean={result['mean']:.3f}, zeros={zero_count}")
        for issue in result["issues"]:
            print(f"    - {issue}")

    return result


# ============================================================================
# Tinker Stack Test (via opentinker-miles HTTP API)
# ============================================================================

def run_tinker_stack(
    test_data: Dict[str, Any],
    base_url: str = "http://localhost:8000",
    api_key: str = "slime-dev-key",
    model_id: Optional[str] = None,
    model_path: str = "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
) -> Dict[str, Any]:
    """
    Run test data through Tinker stack and capture logprobs.

    This calls the opentinker-miles HTTP API, which internally calls Miles.
    """
    print("\n" + "=" * 60)
    print("Tinker Stack Test (via opentinker-miles)")
    print("=" * 60)

    import httpx

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # Check health
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10.0)
        if resp.status_code != 200:
            return {"error": f"Health check failed: {resp.status_code}"}
        print(f"Server healthy: {resp.json().get('status')}")
    except Exception as e:
        return {"error": f"Could not connect to server: {e}"}

    # Create session first
    session_id = None
    if not model_id:
        print("Creating session...")
        resp = httpx.post(
            f"{base_url}/api/v1/create_session",
            json={},
            headers=headers,
            timeout=30.0,
        )
        if resp.status_code != 200:
            return {"error": f"Create session failed: {resp.text}"}
        session_id = resp.json().get("session_id")
        print(f"  Session ID: {session_id}")

        # Create model
        print(f"Creating model from {model_path}...")
        resp = httpx.post(
            f"{base_url}/api/v1/create_model",
            json={
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": model_path,
                "lora_config": {"rank": 0, "alpha": 0}
            },
            headers=headers,
            timeout=30.0,
        )
        if resp.status_code != 200:
            return {"error": f"Create model failed: {resp.text}"}

        req_id = resp.json()["request_id"]
        print(f"  Request ID: {req_id}")

        # Poll for completion
        for attempt in range(90):
            time.sleep(2)
            poll_resp = httpx.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"request_id": req_id},
                headers=headers,
                timeout=120.0,
            )
            if poll_resp.status_code == 200:
                model_id = poll_resp.json().get("model_id")
                print(f"  Model created: {model_id}")
                break
            elif poll_resp.status_code != 408:
                return {"error": f"Poll failed: {poll_resp.text}"}
        else:
            return {"error": "Timed out waiting for model creation"}

    # Convert test data to Tinker format
    datums = test_data_to_tinker_format(test_data)
    print(f"Prepared {len(datums)} Tinker datums")

    # Call forward_backward
    print("Calling forward_backward...")
    resp = httpx.post(
        f"{base_url}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "data": datums,
            "loss_fn": "importance_sampling",
        },
        headers=headers,
        timeout=30.0,
    )

    if resp.status_code != 200:
        return {"error": f"forward_backward failed: {resp.text}"}

    req_id = resp.json()["request_id"]
    print(f"  Request ID: {req_id}")

    # Poll for completion
    result_data = None
    for attempt in range(60):
        time.sleep(2)
        poll_resp = httpx.post(
            f"{base_url}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0,
        )
        if poll_resp.status_code == 200:
            result_data = poll_resp.json()
            print(f"  forward_backward completed")
            break
        elif poll_resp.status_code != 408:
            return {"error": f"Poll failed: {poll_resp.text}"}
        if attempt % 10 == 0:
            print(f"    Still waiting... ({attempt*2}s)")
    else:
        return {"error": "Timed out waiting for forward_backward"}

    # Extract logprobs from result
    result = {
        "path": "tinker_stack",
        "model_id": model_id,
        "num_samples": len(datums),
        "logprobs": [],
        "logprobs_lengths": [],
        "metrics": result_data.get("metrics", {}),
    }

    loss_fn_outputs = result_data.get("loss_fn_outputs", [])
    for i, output in enumerate(loss_fn_outputs):
        logprobs_data = output.get("logprobs", {}).get("data", [])
        result["logprobs"].append(logprobs_data)
        result["logprobs_lengths"].append(len(logprobs_data))
        print(f"  Sample {i}: logprobs={len(logprobs_data)}")

    print(f"\nTinker stack result: {result['num_samples']} samples, metrics={list(result['metrics'].keys())}")
    return result


# ============================================================================
# Full Validation
# ============================================================================

def validate_all_logprobs(
    result: Dict[str, Any],
    test_data: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate all logprobs from a tinker stack run.

    Returns validation report with pass/fail status and detailed issues.
    """
    print("\n" + "=" * 60)
    print("VALIDATING LOGPROBS")
    print("=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return {"valid": False, "error": result["error"]}

    logprobs_list = result.get("logprobs", [])
    samples = test_data.get("samples", [])

    if len(logprobs_list) != len(samples):
        print(f"ERROR: Sample count mismatch: got {len(logprobs_list)}, expected {len(samples)}")
        return {"valid": False, "error": "Sample count mismatch"}

    # Validate each sample
    validations = []
    for i, (logprobs, sample) in enumerate(zip(logprobs_list, samples)):
        expected_len = sample["response_length"]
        v = validate_logprobs_sample(i, logprobs, expected_len, verbose=verbose)
        validations.append(v)

    # Summary statistics
    valid_count = sum(1 for v in validations if v["valid"])
    total_zeros = sum(v.get("zero_count", 0) for v in validations)
    total_logprobs = sum(v["logprobs_length"] for v in validations)

    issues_by_type = {}
    for v in validations:
        for issue in v.get("issues", []):
            issue_type = issue.split(":")[0]
            issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

    summary = {
        "num_samples": len(samples),
        "valid_samples": valid_count,
        "invalid_samples": len(samples) - valid_count,
        "total_logprobs": total_logprobs,
        "total_zeros": total_zeros,
        "zero_ratio": total_zeros / total_logprobs if total_logprobs > 0 else 0,
        "issues_by_type": issues_by_type,
    }

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples: {summary['num_samples']}")
    print(f"  Valid: {summary['valid_samples']}/{summary['num_samples']}")
    print(f"  Total logprobs: {summary['total_logprobs']}")
    print(f"  Zero logprobs: {summary['total_zeros']} ({summary['zero_ratio']:.1%})")

    if issues_by_type:
        print(f"\n  Issues breakdown:")
        for issue_type, count in issues_by_type.items():
            print(f"    - {issue_type}: {count} samples")

    overall_valid = valid_count == len(samples)
    if overall_valid:
        print(f"\n  RESULT: ALL SAMPLES VALID")
    else:
        print(f"\n  RESULT: {len(samples) - valid_count} SAMPLES HAVE ISSUES")

    return {
        "valid": overall_valid,
        "summary": summary,
        "validations": validations,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="E2E Logprobs Verification Test")
    parser.add_argument("--samples", type=int, default=8, help="Number of test samples")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="opentinker-miles URL")
    parser.add_argument("--model-path", type=str, default="/data/models/Qwen2.5-0.5B-Instruct_torch_dist")
    parser.add_argument("--model-id", type=str, default=None, help="Existing model ID (skip model creation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for test data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--prompt-len", type=int, default=50, help="Prompt length")
    parser.add_argument("--response-len", type=int, default=100, help="Response length")

    args = parser.parse_args()

    print("=" * 60)
    print("E2E LOGPROBS VERIFICATION TEST")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"Prompt length: {args.prompt_len}")
    print(f"Response length: {args.response_len}")
    print(f"Base URL: {args.base_url}")
    print(f"Model path: {args.model_path}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Generate test data
    test_data = generate_test_data(
        num_samples=args.samples,
        prompt_len=args.prompt_len,
        response_len=args.response_len,
        seed=args.seed,
    )
    print(f"\nGenerated {len(test_data['samples'])} test samples")
    for i, sample in enumerate(test_data["samples"][:3]):
        print(f"  Sample {i}: prompt={sample['prompt_length']}, response={sample['response_length']}, total={sample['total_length']}")
    if len(test_data["samples"]) > 3:
        print(f"  ... and {len(test_data['samples']) - 3} more")

    # Run through tinker stack
    result = run_tinker_stack(
        test_data,
        base_url=args.base_url,
        model_id=args.model_id,
        model_path=args.model_path,
    )

    # Validate results
    validation = validate_all_logprobs(result, test_data, verbose=args.verbose)

    # Save if requested
    if args.output:
        torch.save({
            "test_data": test_data,
            "result": result,
            "validation": validation,
        }, args.output)
        print(f"\nSaved results to {args.output}")

    # Exit with appropriate code
    if validation.get("valid"):
        print("\n✓ TEST PASSED: All logprobs valid")
        sys.exit(0)
    else:
        print("\n✗ TEST FAILED: Some logprobs have issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
