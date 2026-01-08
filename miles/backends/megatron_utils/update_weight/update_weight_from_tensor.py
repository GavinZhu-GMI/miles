import json
import logging
import os
import shutil
import tempfile
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.lora_weight_version = 0
        self._prev_lora_adapter_path = None

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        # create the group within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del long_lived_tensors

        dist.barrier(group=get_gloo_group())

    @torch.no_grad()
    def update_lora_weights(self) -> None:
        """Update only LoRA weights to SGLang engines via tmpfs.

        This method:
        1. Collects LoRA params from Megatron model
        2. Converts to PEFT naming format
        3. Writes to tmpfs (/dev/shm) for fast I/O
        4. Calls load_lora_adapter on all SGLang engines
        5. Unloads previous adapter version to free GPU memory
        """
        if not getattr(self.args, "lora_rank", 0) > 0:
            return

        self.lora_weight_version += 1
        rank = dist.get_rank()

        if rank == 0:
            # 1. Collect LoRA weights with PEFT naming
            lora_weights = self._collect_lora_weights()

            if not lora_weights:
                logger.warning("No LoRA weights found to sync")
                dist.barrier(group=get_gloo_group())
                return

            # 2. Write to tmpfs (/dev/shm) for fast I/O
            adapter_path = self._write_lora_to_tmpfs(lora_weights)
            adapter_name = f"lora_v{self.lora_weight_version}"

            logger.info(f"Loading LoRA adapter {adapter_name} from {adapter_path}")

            # 3. Load new adapter on all engines
            refs = []
            for engine in self.rollout_engines:
                refs.append(engine.load_lora_adapter.remote(adapter_path, adapter_name))
            ray.get(refs)

            # 4. Unload previous version to free GPU memory
            if self.lora_weight_version > 1:
                old_name = f"lora_v{self.lora_weight_version - 1}"
                refs = []
                for engine in self.rollout_engines:
                    refs.append(engine.unload_lora_adapter.remote(old_name))
                ray.get(refs)

            # 5. Cleanup previous tmpfs directory
            if self._prev_lora_adapter_path and os.path.exists(self._prev_lora_adapter_path):
                shutil.rmtree(self._prev_lora_adapter_path, ignore_errors=True)
            self._prev_lora_adapter_path = adapter_path

        dist.barrier(group=get_gloo_group())

    def _collect_lora_weights(self) -> dict[str, torch.Tensor]:
        """Extract LoRA params and convert to PEFT naming."""
        from ..lora.lora_checkpoint import _megatron_to_peft_name

        lora_weights = {}

        # Iterate through all model chunks (for pipeline parallelism)
        for model_chunk in self.model:
            for name, param in model_chunk.named_parameters():
                if "lora_" in name:
                    peft_name = _megatron_to_peft_name(name)
                    # Clone and move to CPU
                    lora_weights[peft_name] = param.data.detach().cpu().clone()

        logger.info(f"Collected {len(lora_weights)} LoRA parameters for sync")
        return lora_weights

    def _write_lora_to_tmpfs(self, lora_weights: dict[str, torch.Tensor]) -> str:
        """Write LoRA weights to tmpfs in PEFT format."""
        # Use /dev/shm for fast I/O (tmpfs on Linux)
        # Fall back to /tmp if /dev/shm doesn't exist
        tmpfs_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
        adapter_dir = tempfile.mkdtemp(prefix=f"lora_v{self.lora_weight_version}_", dir=tmpfs_dir)

        # Write weights
        weights_path = os.path.join(adapter_dir, "adapter_model.bin")
        torch.save(lora_weights, weights_path)

        # Write PEFT config
        config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": getattr(self.args, "hf_checkpoint", "unknown"),
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": getattr(self.args, "lora_alpha", self.args.lora_rank),
            "lora_dropout": getattr(self.args, "lora_dropout", 0.0),
            "modules_to_save": None,
            "r": self.args.lora_rank,
            "rank_pattern": {},
            "alpha_pattern": {},
            "revision": None,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "task_type": "CAUSAL_LM",
        }
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Wrote LoRA adapter to {adapter_dir} ({len(lora_weights)} params)")
        return adapter_dir

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # TODO improve
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
