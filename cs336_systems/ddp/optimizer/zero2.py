import math
from copy import deepcopy
from collections.abc import Callable, Iterable
from typing import Any, Type

import torch
from torch import Tensor
import torch.distributed as dist


class ZeRO2Optimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        super().__init__(params, {})
        self.handles: dist[str, Any] = {}

        world_size = dist.get_world_size()
        for group in self.param_groups:
            def hook(p: Tensor) -> None:
                n = p.shape[0]
                shard_size = int(math.ceil(n / world_size))
                buffer_size = world_size * shard_size
                buffer_shape = [buffer_size] + list(p.shape[1:])
                buffer = torch.empty(buffer_shape)
                buffer[:n] = p.grad.clone()
                p.grad = None  # Release grad memory

                dest = group["shard_params_buffer"][p]
                # reduce scatter local gradients to child optimizer shard
                handle = dist.reduce_scatter_tensor(dest, buffer, async_op=True)

                self.handles[p] = handle

            for p in group["params"]:
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(hook)

        # Copy the original parameter groups to be able to pass them to child optimizer
        shard_param_groups = [{**pg} for pg in self.param_groups]
        for group in shard_param_groups:
            # For the child optimizer, shard_params is params
            # Doing the parent and child optimizer to share memory so zero_grad just works for both
            params = group.pop("shard_params")
            group["params"] = params

        self.optimizer = optimizer_cls(shard_param_groups, **kwargs)

    def step(self, closure: Callable | None = None, **kwargs):
        world_size = dist.get_world_size()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if not p.requires_grad:
                    continue

                handle = self.handles[p]
                handle.wait()
                # p will have been all-reduced between ranks
                dest_shard = group["shard_params_buffer"][p]
                dest_shard /= world_size

                start, end = group["shard_params_indices"][i]
                # print(f"{dest_shard.shape=:}, indices: {start, end}, slice: {dest_shard[start:end].shape=:}")
                group["shard_params"][i].grad = dest_shard[start:end]

        self.handles.clear()

        # After gradients are reduce-scattered, the child optimizer
        # has them aggregated across ranks. We can now simply call .step()
        self.optimizer.step(closure, **kwargs)

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if not p.requires_grad:
                    continue
                # src now has updated parameters for its shard,
                # for the complete batch
                # We need to all_gather this to share across ranks
                src = group["shard_params"][i].data

                # Need to pad both src and dest for all_gather
                n = p.shape[0]
                shard_size = int(math.ceil(n / world_size))
                shard_buffer_shape = [shard_size] + list(src.shape[1:])
                src_buffer = torch.empty(shard_buffer_shape)
                src_buffer[:src.shape[0]] = src

                buffer_size = world_size * shard_size
                buffer_shape = [buffer_size] + list(p.shape[1:])
                buffer = torch.empty(buffer_shape)
                buffer[:n] = p.data

                dist.all_gather_into_tensor(buffer, src_buffer, async_op=False)
                p.data = buffer[:n]

    def add_param_group(self, param_group: dict[str, Any]):
        shard_params = []
        shard_params_indices = []
        shard_params_buffer = {}
        for p in param_group["params"]:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            shard_size = int(math.ceil(p.shape[0] / world_size))
            shard_buffer_shape = [shard_size] + list(p.shape[1:])
            local_start = rank * shard_size
            local_end = min(local_start+shard_size, p.shape[0])

            local_shard = p[local_start:(local_start+shard_size)].detach()
            shard_params.append(local_shard)
            shard_params_indices.append((0, local_end-local_start))
            buff = torch.empty(shard_buffer_shape)
            shard_params_buffer[p] = buff

        # These are sharded views of the original parameters
        param_group["shard_params"] = shard_params
        param_group["shard_params_indices"] = shard_params_indices
        param_group["shard_params_buffer"] = shard_params_buffer

        super().add_param_group(param_group)
