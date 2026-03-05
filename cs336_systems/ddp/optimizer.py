import math
from collections.abc import Callable, Iterable
from typing import Any, Type

import torch
import torch.distributed as dist


class ZeROOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        super().__init__(params, {})
        self.optimizer = optimizer_cls(self.param_groups, **kwargs)

    def step(self, closure, **kwargs):
        world_size = dist.get_world_size()
        for group in self.param_groups:
            for i, p in enumerate(group["original_params"]):
                if p.grad is None:
                    continue
                dest = group["params"][i].grad  # This is the parameter shard
                dist.reduce_scatter_tensor(dest, p.grad, async_op=False)
                p.grad /= world_size

        self.optimizer.step(closure, **kwargs)

        for group in self.param_groups:
            for i, p in enumerate(group["original_params"]):
                if p.grad is None:
                    continue
                src = group["params"][i]
                dist.all_gather_into_tensor(p, src, async_op=False)

    def add_param_group(self, param_group: dict[str, Any]):
        shard_params = []
        for p in param_group["params"]:
            world_size = dist.get_world_size()
            to_scatter = torch.chunk(p, world_size, dim=0)
            local_shard = torch.empty_like(to_scatter[0])
            dist.scatter(local_shard, to_scatter, src=0)
            shard_params.append(local_shard)
        # Need original params for reduce-scatter of gradients
        param_group["original_params"] = param_group["params"]
        param_group["params"] = shard_params

        super().add_param_group(param_group)
