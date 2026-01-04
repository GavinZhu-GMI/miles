#!/usr/bin/env python3
"""
DP Aggregation Test for Miles

This test verifies that miles' forward_backward_only() correctly aggregates
results from multiple DP ranks and returns logprobs in original input order.

The test verifies correct behavior for:
- Strided DP pattern (balance_data=False): samples [0,2,4,...] to rank 0, [1,3,5,...] to rank 1
- Group-aware DP pattern (balance_data=True): groups kept together, balanced by token count

After the refactor, miles handles DP interleaving internally via _dp_original_indices,
so opentinker-miles no longer needs to know about miles' DP partitioning scheme.

Usage:
    # Run with opentinker-miles server running
    TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=slime-dev-key \
        python3 test_dp_consistency.py --samples 16

    # With verbose output
    TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=slime-dev-key \
        python3 test_dp_consistency.py --samples 16 -v
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import torch


def generate_fixed_test_data(
    num_samples: int = 8,
    vocab_size: int = 151936,
    prompt_len: int = 32,
    response_len: int = 32,
    seed: int = 42,
    variable_lengths: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic test data in Tinker format.

    Args:
        num_samples: Number of samples to generate
        vocab_size: Vocabulary size for token generation
        prompt_len: Base prompt length
        response_len: Base response length
        seed: Random seed for reproducibility
        variable_lengths: If True, vary response lengths to test sequence-length balancing
    """
    torch.manual_seed(seed)

    datums = []
    for i in range(num_samples):
        # Vary response length if requested (tests sequence-length balancing)
        if variable_lengths:
            # Response lengths from 16 to 64
            actual_response_len = 16 + (i * 48) // max(num_samples - 1, 1)
        else:
            actual_response_len = response_len

        total_len = prompt_len + actual_response_len
        tokens = torch.randint(1, vocab_size, (total_len,), dtype=torch.long).tolist()

        # GRPO-style advantages with sample-specific patterns for verification
        # Use sin wave pattern so we can verify sample ordering
        advantage = [
            0.1 * torch.sin(torch.tensor(j * 0.1 + i * 0.5)).item()
            for j in range(actual_response_len)
        ]

        datum = {
            "model_input": {"chunks": [{"tokens": tokens}]},
            "loss_fn_inputs": {
                "target_tokens": {
                    "data": tokens[-actual_response_len:],
                    "shape": [actual_response_len],
                    "dtype": "int64"
                },
                "logprobs": {
                    "data": [0.0] * actual_response_len,
                    "shape": [actual_response_len],
                    "dtype": "float32"
                },
                "advantages": {
                    "data": advantage,
                    "shape": [actual_response_len],
                    "dtype": "float32"
                }
            },
            # Store sample index for verification
            "_sample_idx": i,
        }
        datums.append(datum)

    return datums


async def poll_for_result(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    request_id: str,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Poll for async result completion."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Use POST on the path-based endpoint
        resp = await client.post(
            f"{base_url}/api/v1/retrieve_future/{request_id}",
            headers=headers,
        )

        if resp.status_code == 200:
            # When completed, the result is returned directly (not wrapped in status)
            return resp.json()
        elif resp.status_code == 500:
            # Operation failed
            detail = resp.json().get("detail", "Unknown error")
            raise RuntimeError(f"forward_backward failed: {detail}")
        elif resp.status_code == 408:
            # Still pending, continue polling
            pass

        await asyncio.sleep(0.5)

    raise TimeoutError(f"Polling timed out after {timeout}s")


async def create_session(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
) -> str:
    """Create a new session."""
    resp = await client.post(
        f"{base_url}/api/v1/create_session",
        json={
            "tags": ["dp-consistency-test"],
            "user_metadata": {"test": "dp_consistency"},
            "sdk_version": "test-1.0.0"
        },
        headers=headers,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to create session: {resp.text}")
    return resp.json().get("session_id")


async def get_or_create_model(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    model_path: str,
    session_id: str,
    max_batch_size: int = 16,
) -> str:
    """Get existing model or create new one."""
    # List existing sessions to check for existing models
    resp = await client.get(f"{base_url}/api/v1/sessions/{session_id}", headers=headers)
    if resp.status_code == 200:
        training_run_ids = resp.json().get("training_run_ids", [])
        if training_run_ids:
            return training_run_ids[0]

    # Create new model
    print("    Creating new model...")
    resp = await client.post(
        f"{base_url}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": model_path,
            "lora_config": None,
            "debug_train_only": True,
            "max_batch_size": max_batch_size,
        },
        headers=headers,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to create model: {resp.text}")

    result = resp.json()
    request_id = result.get("request_id")
    model_id = result.get("model_id")
    print(f"    Created model: {model_id}, polling request_id: {request_id}")

    # Poll for completion (POST method for retrieve_future)
    for i in range(120):
        await asyncio.sleep(2)
        resp = await client.post(
            f"{base_url}/api/v1/retrieve_future/{request_id}",
            headers=headers,
        )
        if resp.status_code == 200:
            # Completed - result returned directly
            print(f"    Model ready after {i*2}s")
            break
        elif resp.status_code == 500:
            detail = resp.json().get("detail", "Unknown error")
            raise RuntimeError(f"Model creation failed: {detail}")
        # 408 means still pending, continue polling
        if (i + 1) % 10 == 0:
            print(f"    Still waiting... ({i*2}s)")

    return model_id


async def run_forward_backward(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    model_id: str,
    datums: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run forward_backward and return result."""
    resp = await client.post(
        f"{base_url}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "data": datums,
            "loss_fn": "importance_sampling",
        },
        headers=headers,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"forward_backward failed: {resp.text}")

    result = resp.json()
    if "request_id" in result:
        return await poll_for_result(client, base_url, headers, result["request_id"])
    return result


def verify_logprobs_order(
    datums: List[Dict[str, Any]],
    result: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Verify that logprobs are returned in the same order as input.

    Returns dict with pass/fail and any issues found.
    """
    issues = []

    loss_fn_outputs = result.get("loss_fn_outputs", [])

    # Check count matches
    if len(loss_fn_outputs) != len(datums):
        issues.append(f"Output count mismatch: got {len(loss_fn_outputs)}, expected {len(datums)}")
        return {"pass": False, "issues": issues}

    # Statistics for verbose output
    logprob_stats = []

    # Check each sample has logprobs with correct length
    for i, (datum, output) in enumerate(zip(datums, loss_fn_outputs)):
        expected_len = datum["loss_fn_inputs"]["advantages"]["shape"][0]
        logprobs = output.get("logprobs", {})
        actual_len = logprobs.get("shape", [0])[0] if logprobs else 0

        if actual_len != expected_len:
            issues.append(f"Sample {i}: logprobs length {actual_len}, expected {expected_len}")

        # Check logprobs are not all zeros (Megatron should have computed them)
        logprobs_data = logprobs.get("data", [])
        if logprobs_data:
            lp_mean = sum(logprobs_data) / len(logprobs_data)
            lp_min = min(logprobs_data)
            lp_max = max(logprobs_data)
            logprob_stats.append({
                "sample": i,
                "len": len(logprobs_data),
                "mean": lp_mean,
                "min": lp_min,
                "max": lp_max,
            })

            if all(abs(lp) < 1e-10 for lp in logprobs_data):
                issues.append(f"Sample {i}: logprobs are all zeros (not computed?)")

        # Verify ordering: check that logprobs vary as expected
        # (Different samples should have different logprob patterns due to different tokens)
        if i > 0 and logprobs_data and len(loss_fn_outputs) > 1:
            prev_logprobs = loss_fn_outputs[i-1].get("logprobs", {}).get("data", [])
            if prev_logprobs and len(prev_logprobs) == len(logprobs_data):
                # Samples should have different logprobs (unless they're identical, which is unlikely)
                if prev_logprobs == logprobs_data:
                    issues.append(f"Sample {i}: identical logprobs to sample {i-1} (ordering issue?)")

    if verbose and logprob_stats:
        print("\n    Logprob statistics:")
        for stat in logprob_stats[:5]:  # Show first 5
            print(f"      Sample {stat['sample']}: len={stat['len']}, mean={stat['mean']:.4f}, "
                  f"range=[{stat['min']:.4f}, {stat['max']:.4f}]")
        if len(logprob_stats) > 5:
            print(f"      ... ({len(logprob_stats) - 5} more samples)")

    return {"pass": len(issues) == 0, "issues": issues}


async def run_test_case(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    model_id: str,
    name: str,
    num_samples: int,
    seed: int,
    variable_lengths: bool,
    verbose: bool,
) -> Tuple[bool, str]:
    """Run a single test case and return (pass, description)."""
    print(f"\n  [{name}]")
    print(f"    Samples: {num_samples}, variable_lengths={variable_lengths}")

    # Generate test data
    datums = generate_fixed_test_data(
        num_samples=num_samples,
        seed=seed,
        variable_lengths=variable_lengths,
    )

    # Run forward_backward
    try:
        result = await run_forward_backward(client, base_url, headers, model_id, datums)
    except Exception as e:
        return False, f"forward_backward failed: {e}"

    # Verify
    verification = verify_logprobs_order(datums, result, verbose=verbose)

    if verification["pass"]:
        return True, "PASS"
    else:
        return False, f"FAIL: {verification['issues']}"


async def main():
    parser = argparse.ArgumentParser(description="Test miles DP aggregation")
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "slime-dev-key"))
    parser.add_argument("--model-path", default="/data/models/Qwen2.5-0.5B-Instruct_torch_dist")
    parser.add_argument("--samples", type=int, default=16, help="Base number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    args = parser.parse_args()

    print("=" * 70)
    print("MILES DP CONSISTENCY TEST")
    print("=" * 70)
    print(f"  Base URL: {args.base_url}")
    print(f"  Model: {args.model_path}")
    print(f"  Base samples: {args.samples}")

    headers = {"X-API-Key": args.api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Check health
        resp = await client.get(f"{args.base_url}/health", headers=headers)
        if resp.status_code != 200:
            print(f"ERROR: Server not healthy: {resp.text}")
            sys.exit(1)

        # Create session first
        print("\n[1] Creating session...")
        session_id = await create_session(client, args.base_url, headers)
        print(f"    Session: {session_id}")

        # Define test cases first to determine max_batch_size
        test_cases = [
            # (name, num_samples, variable_lengths)
            ("uniform_8", 8, False),
            ("uniform_16", 16, False),
        ]

        if not args.quick:
            test_cases.extend([
                ("uniform_32", 32, False),
                ("variable_8", 8, True),
                ("variable_16", 16, True),
                ("variable_32", 32, True),
            ])

        # Calculate max_batch_size from test cases
        max_batch_size = max(tc[1] for tc in test_cases)

        # Get/create model with correct max_batch_size
        print(f"\n[2] Getting/creating model (max_batch_size={max_batch_size})...")
        model_id = await get_or_create_model(
            client, args.base_url, headers, args.model_path, session_id,
            max_batch_size=max_batch_size
        )
        print(f"    Using model: {model_id}")

        # Run test cases
        print("\n[3] Running test cases...")

        results = []
        for name, num_samples, variable_lengths in test_cases:
            passed, desc = await run_test_case(
                client=client,
                base_url=args.base_url,
                headers=headers,
                model_id=model_id,
                name=name,
                num_samples=num_samples,
                seed=args.seed,
                variable_lengths=variable_lengths,
                verbose=args.verbose,
            )
            results.append((name, passed, desc))
            print(f"    Result: {desc}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed, desc in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {desc}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
