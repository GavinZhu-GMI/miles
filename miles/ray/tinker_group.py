# Tinker seam: RayTrainGroup extension exposing decoupled train-step
# primitives for the Tinker API (specs/005 in tinker-nemorl). Additive only —
# the frozen v1 RayTrainGroup is untouched; this subclass rides its
# _broadcast fanout. Pattern: N x forward_backward_only accumulate gradients
# on the actors, then one apply_optimizer_step applies them (per-call LR).

from miles.ray.actor_group import RayTrainGroup


def merge_dp_sample_outputs(results: list[dict], key: str = "log_probs") -> list:
    """Reassemble per-sample outputs from DP-sharded actor results into the
    client's original submission order.

    Each actor returns its shard's outputs plus ``partition_indices`` (the
    original indices assigned to its DP rank). TP/PP replicas of the same DP
    rank return duplicates or empty lists; last writer wins, which is safe
    because duplicates carry identical values.
    """
    merged = {}
    for result in results:
        if not result:
            continue
        outputs = result.get(key) or []
        indices = result.get("partition_indices") or []
        if outputs and len(outputs) == len(indices):
            for original_idx, output in zip(indices, outputs, strict=True):
                merged[original_idx] = output
    return [merged[i] for i in sorted(merged)]


class TinkerTrainGroup(RayTrainGroup):
    """RayTrainGroup + the Tinker orchestration surface."""

    async def forward_backward_only(self, rollout_id, rollout_data_ref):
        """Fan out fwd+bwd (no optimizer step) to all actors."""
        return await self._broadcast("forward_backward_only", rollout_id, rollout_data_ref)

    async def apply_optimizer_step(self, learning_rate: float | None = None):
        """Apply the optimizer over accumulated grads on all actors.

        Returns one {"success", "grad_norm"} dict per actor.
        """
        return await self._broadcast("apply_optimizer_step", learning_rate=learning_rate)

    async def apply_optimizer_step_and_sync(self, learning_rate: float | None = None, rollout_id=None):
        """Optimizer step + push updated weights to the inference engines."""
        results = await self.apply_optimizer_step(learning_rate=learning_rate)
        await self.update_weights(rollout_id)
        return results

    async def forward_logprobs(self, rollout_id, rollout_data_ref):
        """Forward-only per-sample log-probs, reassembled into client order."""
        results = await self._broadcast("forward_logprobs", rollout_id, rollout_data_ref)
        return merge_dp_sample_outputs(results, key="log_probs")

    async def load_checkpoint(self, checkpoint_path: str):
        """Full resume (model + optimizer + scheduler) on all actors."""
        return await self._broadcast("load_checkpoint", checkpoint_path)
