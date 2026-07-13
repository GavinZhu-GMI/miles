#!/usr/bin/env python3
"""
Debug script for the gradient accumulation logprobs issue.

This script helps debug the known issue where Miles only returns logprobs from
the first gradient accumulation step, losing logprobs from subsequent steps.

Root cause (from CLAUDE.md):
    ray_step.py:run_forward_backward_only() originally used num_microbatches=num_microbatches[0]
    which only processes the first gradient accumulation step.

This script helps verify the fix by:
1. Simulating the gradient accumulation scenario
2. Checking if all logprobs are returned correctly

Usage:
    python tests/test_logprobs_grad_accum.py --samples 16 --batch-size 8 --dp-size 2
"""

import argparse
import sys
from typing import List, Tuple


def simulate_gradient_accumulation(
    num_samples: int,
    global_batch_size: int,
    dp_size: int,
    num_steps_per_rollout: int = 1,
) -> Tuple[int, List[int], List[int]]:
    """
    Simulate how Miles splits samples across gradient accumulation steps.

    This mirrors the logic in:
    - miles/backends/megatron_utils/data.py:get_data_iterator()
    - miles/backends/megatron_utils/actor.py:_compute_num_local_samples()

    Returns:
        num_grad_accum_steps: Number of gradient accumulation steps
        samples_per_step: List of sample counts per grad accum step
        logprobs_per_step: List of logprob counts per step (should equal samples_per_step)
    """
    print(f"\n{'='*60}")
    print("Gradient Accumulation Simulation")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Global batch size: {global_batch_size}")
    print(f"DP size: {dp_size}")
    print(f"Num steps per rollout: {num_steps_per_rollout}")

    # Calculate local batch size (samples per DP rank)
    samples_per_dp_rank = num_samples // dp_size
    target_local_batch_size = global_batch_size // dp_size

    print(f"\nPer DP rank:")
    print(f"  Samples per rank: {samples_per_dp_rank}")
    print(f"  Target local batch size: {target_local_batch_size}")

    # Calculate gradient accumulation steps
    num_grad_accum_steps = max(1, samples_per_dp_rank // target_local_batch_size)

    print(f"\nGradient accumulation:")
    print(f"  Number of steps: {num_grad_accum_steps}")

    # Simulate samples per step
    samples_per_step = []
    remaining = samples_per_dp_rank

    for step in range(num_grad_accum_steps):
        samples_this_step = min(target_local_batch_size, remaining)
        samples_per_step.append(samples_this_step)
        remaining -= samples_this_step
        print(f"  Step {step}: {samples_this_step} samples")

    if remaining > 0:
        print(f"  WARNING: {remaining} samples would be dropped!")

    return num_grad_accum_steps, samples_per_step, samples_per_step.copy()


def check_logprobs_coverage(
    num_samples: int,
    expected_logprobs: int,
    actual_logprobs: int,
) -> bool:
    """Check if all samples have corresponding logprobs."""
    print(f"\n{'='*60}")
    print("Logprobs Coverage Check")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Expected logprobs (from all steps): {expected_logprobs}")
    print(f"Actual logprobs returned: {actual_logprobs}")

    coverage = actual_logprobs / expected_logprobs * 100 if expected_logprobs > 0 else 0
    print(f"Coverage: {coverage:.1f}%")

    if actual_logprobs == expected_logprobs:
        print("✓ All samples have logprobs!")
        return True
    else:
        missing = expected_logprobs - actual_logprobs
        print(f"✗ Missing logprobs for {missing} samples!")
        print(f"\nThis is the gradient accumulation bug!")
        print("Fix: Ensure ray_step.py loops over ALL gradient accumulation steps.")
        return False


def simulate_bug_scenario():
    """Simulate the exact bug scenario from CLAUDE.md."""
    print("\n" + "="*60)
    print("Bug Scenario: 16 samples, global_batch_size=8, dp_size=2")
    print("="*60)

    num_samples = 16
    global_batch_size = 8
    dp_size = 2

    # Each DP rank gets 8 samples
    samples_per_dp = num_samples // dp_size
    print(f"Each DP rank has {samples_per_dp} samples")

    # target_local_batch_size = 8 / 2 = 4
    target_local_batch = global_batch_size // dp_size
    print(f"Target local batch size: {target_local_batch}")

    # num_grad_accum_steps = 8 / 4 = 2
    num_grad_accum = samples_per_dp // target_local_batch
    print(f"Gradient accumulation steps: {num_grad_accum}")

    print("\nWith the bug (num_microbatches=num_microbatches[0]):")
    print(f"  Only processes step 0: {target_local_batch} samples")
    print(f"  Missing step 1: {target_local_batch} samples")
    print(f"  Per DP rank: only {target_local_batch} logprobs instead of {samples_per_dp}")
    print(f"  Total across {dp_size} ranks: {target_local_batch * dp_size} logprobs instead of {num_samples}")

    print("\nWith the fix (loop over all steps):")
    print(f"  Process step 0: {target_local_batch} samples")
    print(f"  Process step 1: {target_local_batch} samples")
    print(f"  Per DP rank: {samples_per_dp} logprobs")
    print(f"  Total across {dp_size} ranks: {num_samples} logprobs")

    return True


def check_ray_step_fix(miles_path: str = "/root/gavin/miles"):
    """Check if the ray_step.py fix is applied."""
    print("\n" + "="*60)
    print("Checking ray_step.py for gradient accumulation fix")
    print("="*60)

    import os
    ray_step_path = os.path.join(miles_path, "miles/backends/megatron_utils/ray_step.py")

    if not os.path.exists(ray_step_path):
        print(f"WARNING: File not found: {ray_step_path}")
        return None

    with open(ray_step_path, "r") as f:
        content = f.read()

    # Check for the bug pattern
    bug_pattern = "num_microbatches=num_microbatches[0]"
    if bug_pattern in content:
        print(f"✗ BUG FOUND: Found '{bug_pattern}' in ray_step.py")
        print("  This only processes the first gradient accumulation step!")

        # Find line number
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if bug_pattern in line:
                print(f"  Location: line {i+1}")
                print(f"  Code: {line.strip()}")
        return False

    # Check for the fix pattern (looping over steps)
    fix_patterns = [
        "for step_idx",
        "for grad_accum_step",
        "range(len(num_microbatches))",
    ]

    found_fix = False
    for pattern in fix_patterns:
        if pattern in content:
            found_fix = True
            print(f"✓ Fix pattern found: '{pattern}'")

    if found_fix:
        print("✓ Gradient accumulation fix appears to be applied!")
        return True
    else:
        print("? Could not determine if fix is applied.")
        print("  Please manually check run_forward_backward_only() in ray_step.py")
        return None


def main():
    parser = argparse.ArgumentParser(description="Debug gradient accumulation logprobs issue")
    parser.add_argument("--samples", type=int, default=16, help="Total number of samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Global batch size")
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--check-fix", action="store_true", help="Check if ray_step.py fix is applied")
    parser.add_argument("--miles-path", type=str, default="/root/gavin/miles", help="Path to miles repo")

    args = parser.parse_args()

    print("="*60)
    print("Gradient Accumulation Logprobs Debugger")
    print("="*60)

    # Run simulation
    num_steps, samples_per_step, logprobs_per_step = simulate_gradient_accumulation(
        args.samples,
        args.batch_size,
        args.dp_size,
    )

    # Check coverage
    total_expected = sum(samples_per_step)
    if num_steps > 1:
        # Simulate the bug: only first step's logprobs
        buggy_logprobs = samples_per_step[0] * args.dp_size
        fixed_logprobs = total_expected * args.dp_size

        print("\n" + "-"*60)
        print("Bug vs Fix comparison:")
        print("-"*60)
        print(f"With BUG:  {buggy_logprobs} logprobs (only step 0)")
        print(f"With FIX:  {fixed_logprobs} logprobs (all steps)")

        check_logprobs_coverage(args.samples, fixed_logprobs, buggy_logprobs)
    else:
        print("\n✓ No gradient accumulation needed - no bug possible.")

    # Simulate the exact bug scenario from CLAUDE.md
    simulate_bug_scenario()

    # Check if fix is applied
    if args.check_fix:
        check_ray_step_fix(args.miles_path)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("The gradient accumulation bug occurs when:")
    print("  - num_samples > global_batch_size / dp_size")
    print("  - This requires multiple gradient accumulation steps")
    print("  - The bug only returns logprobs from step 0")
    print("\nTo avoid the bug (workaround):")
    print("  - Set global_batch_size >= num_samples * dp_size")
    print("  - Or use max_batch_size in slime_builder.py (default 4096)")
    print("\nTo fix the bug permanently:")
    print("  - Modify ray_step.py:run_forward_backward_only() to loop over all steps")
    print("  - See CLAUDE.md for the exact fix")


if __name__ == "__main__":
    main()
