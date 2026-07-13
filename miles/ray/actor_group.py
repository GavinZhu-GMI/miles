import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        if self.args.use_routing_replay:
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        backend = self.args.train_backend
        if backend == "megatron":
            from miles.backends.megatron_utils import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from miles.backends.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args, role, with_ref=with_ref) for actor in self._actor_handlers]

    def async_train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def save_model(self, step_id):
        """Save actor model on rank 0."""
        return ray.get([actor.save_model.remote(step_id) for actor in self._actor_handlers])

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def update_lora_weights(self):
        """Update LoRA weights across all actors to SGLang engines."""
        return ray.get([actor.update_lora_weights.remote() for actor in self._actor_handlers])

    def onload(self):
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def connect(self, critic_group):
        return ray.get(
            [
                actor.connect_actor_critic.remote(critic)
                for actor, critic in zip(self._actor_handlers, critic_group._actor_handlers, strict=False)
            ]
        )

    def set_rollout_manager(self, rollout_manager):
        return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])

    def forward_backward_only(self, rollout_id, rollout_data_ref, zero_grads=False):
        """
        Run forward/backward without optimizer step to accumulate gradients (Tinker API support).

        Returns a single aggregated result dict with logprobs interleaved to original input order.
        """
        raw_results = ray.get(
            [
                actor.forward_backward_step_only.remote(rollout_id, rollout_data_ref, zero_grads)
                for actor in self._actor_handlers
            ]
        )
        return self._aggregate_dp_results(raw_results)

    def _aggregate_dp_results(self, results):
        """
        Aggregate per-actor results, reordering logprobs to restore original sample order.

        With DP (data parallel), each actor processes a subset of samples. The subset
        can be either:
        - Strided pattern (balance_data=False): Actor 0 gets [0, 2, 4, ...], Actor 1 gets [1, 3, 5, ...]
        - Group-aware balanced (balance_data=True): Groups kept together, balanced by token count

        Each actor returns `_dp_original_indices` indicating which global sample indices
        it processed. This method uses those indices to reorder logprobs to original order,
        making the caller DP-agnostic.

        Args:
            results: List of result dicts from each actor

        Returns:
            Single aggregated result dict with logprobs in original order
        """
        # Collect logprobs and partition indices from actors that have them (pipeline last stage only)
        dp_results_with_logprobs = []
        dp_original_indices = []
        result_with_loss = None

        for result in results:
            loss_dict = result.get("loss", {})
            if loss_dict.get("log_probs"):
                dp_results_with_logprobs.append(loss_dict["log_probs"])
                dp_original_indices.append(result.get("_dp_original_indices", []))
            if loss_dict.get("loss") is not None:
                result_with_loss = result

        # Reorder logprobs to original sample order using partition indices
        if len(dp_results_with_logprobs) > 1:
            # Check if all actors have identical indices (DP=1 case where multiple
            # actors on same DP rank return logprobs, e.g., with PP>1 or TP>1)
            indices_are_identical = all(
                dp_original_indices[i] == dp_original_indices[0]
                for i in range(1, len(dp_original_indices))
            )

            if indices_are_identical:
                # All actors processed same samples (DP=1) - use first actor's logprobs only
                # This happens when TP=2, PP=2 and both TP ranks on last PP stage return logprobs
                all_logprobs = dp_results_with_logprobs[0]
            else:
                # Different DP ranks processed different samples - reorder to original order
                total_samples = sum(len(indices) for indices in dp_original_indices)
                reordered = [None] * total_samples

                for indices, lp_list in zip(dp_original_indices, dp_results_with_logprobs):
                    if len(indices) != len(lp_list):
                        # Fallback: indices don't match logprobs, use as-is with warning
                        print(f"[WARNING] _aggregate_dp_results: indices len ({len(indices)}) != logprobs len ({len(lp_list)})", flush=True)
                        continue
                    for local_idx, (original_idx, logprob) in enumerate(zip(indices, lp_list)):
                        if original_idx < total_samples:
                            reordered[original_idx] = logprob

                # Filter out any None entries (shouldn't happen with correct indices)
                all_logprobs = [lp for lp in reordered if lp is not None]
                if len(all_logprobs) != total_samples:
                    print(f"[WARNING] _aggregate_dp_results: missing logprobs after reorder "
                          f"({len(all_logprobs)}/{total_samples})", flush=True)
        elif dp_results_with_logprobs:
            all_logprobs = dp_results_with_logprobs[0]
        else:
            all_logprobs = []

        # Build aggregated result
        if result_with_loss:
            loss_dict = dict(result_with_loss.get("loss", {}))  # Copy to avoid mutating original
            loss_dict["log_probs"] = all_logprobs
            return {
                "loss": loss_dict,
                "grad_norm": result_with_loss.get("grad_norm", 0.0),
                "valid_step": result_with_loss.get("valid_step", True),
            }
        else:
            return {
                "loss": {"log_probs": all_logprobs},
                "grad_norm": 0.0,
                "valid_step": True,
            }

    def forward_only(self, rollout_id, rollout_data_ref):
        """
        Forward-only inference to fetch per-sample log probabilities (used by DPO reference runs).
        """
        raw_results = ray.get(
            [actor.forward_only_step.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]
        )
        # Aggregate DP results (reorder logprobs to original sample order)
        return self._aggregate_dp_results(raw_results)

    def apply_optimizer_step(self, learning_rate: float = None):
        """
        Apply optimizer step after manual gradient accumulation.

        Args:
            learning_rate: Optional learning rate to override optimizer's current LR.
                          If provided, updates all param_groups['lr'] before stepping.
        """
        return ray.get([
            actor.apply_optimizer_step.remote(learning_rate=learning_rate)
            for actor in self._actor_handlers
        ])

    def apply_optimizer_step_and_sync(self, learning_rate: float = None):
        """
        Apply optimizer step and sync weights to SGLang.

        Combines apply_optimizer_step() + update_weights() for Tinker API use case
        where we always want to sync weights after training before next sample().

        Args:
            learning_rate: Optional learning rate to override optimizer's current LR.
        """
        results = self.apply_optimizer_step(learning_rate=learning_rate)
        self.update_weights()
        return results
