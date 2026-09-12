import math
from copy import deepcopy
from collections.abc import Callable, Iterable
from typing import Any, Type

import torch
from torch import nn, Tensor
import torch.distributed as dist


class ZeRO3Optimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        super().__init__(params, {})

        self.handles: dist[str, Any] = {}
        self.pre_backwards_handles = {}
        self.forward_handles = {}

        world_size = dist.get_world_size()

        for g_i, group in enumerate(self.param_groups):
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
            
            def make_pre_backwards_gather_hook(p: Tensor, g_i, i):
                def pre_backwards_hook(_p: Tensor):
                    shard_params = group["shard_params"][i]

                    size = group["shard_unpadded_size"][i]
                    shard_size = int(math.ceil(size / world_size))
                    shard_buffer_shape = [shard_size] + list(shard_params.shape[1:])
                    src_buffer = torch.empty(shard_buffer_shape)
                    src_buffer[:shard_params.shape[0]] = shard_params

                    buffer_size = world_size * shard_size
                    buffer_shape = [buffer_size] + list(shard_params.shape[1:])
                    buffer = torch.empty(buffer_shape)

                    handle = dist.all_gather_into_tensor(
                        buffer, src_buffer, async_op=True
                    )

                    self.pre_backwards_handles[g_i, i] = (handle, buffer)
                
                return pre_backwards_hook
            
            def make_pre_backwards_wait_hook(p: Tensor, g_i, i):
                def pre_backwards_wait_hook(_p: Tensor):
                    handle, buffer = self.pre_backwards_handles[g_i, i]
                    handle.wait()

                    size = group["shard_unpadded_size"][i]
                    p.data = buffer[:size]

                return pre_backwards_wait_hook

            def make_start_gather_hook(p: Tensor, g_i, i):
                def pre_forward_hook(module, input) -> None:
                    shard_params = group["shard_params"][i]
                    size = group["shard_unpadded_size"][i]
                    shard_size = int(math.ceil(size / world_size))
                    shard_buffer_shape = [shard_size] + list(shard_params.shape[1:])

                    src_buffer = torch.empty(shard_buffer_shape)
                    src_buffer[:shard_params.shape[0]] = shard_params

                    buffer_size = world_size * shard_size
                    buffer_shape = [buffer_size] + list(shard_params.shape[1:])
                    buffer = torch.zeros(buffer_shape)

                    handle = dist.all_gather_into_tensor(
                        buffer, src_buffer, async_op=True
                    )

                    self.forward_handles[g_i, i] = (handle, buffer)

                return pre_forward_hook

            def make_pre_forward_hook_wait(p: Tensor, g_i, i):
                def pre_forward_hook(module, input) -> None:
                    handle, buffer = self.forward_handles[g_i, i]
                    handle.wait()

                    size = group["shard_unpadded_size"][i]
                    p.data = buffer[:size]

                return pre_forward_hook

            def make_post_forward_hook_clean_memory(p: Tensor, g_i, i):
                def post_forward_hook(module, input, output) -> None:
                    # Free parameter memory
                    p.data = group["shard_params"][i].data
                return post_forward_hook
            
            parent_modules = group["shard_params_parent"]
            next_layer = None

            for i,p in enumerate(group["params"]):
                assert next_layer is None or (next_layer == p).all()
                # wait to all-gather params for layer
                for parent_module in parent_modules[p]:
                    if i == 0:
                        parent_module.register_forward_pre_hook(make_start_gather_hook(p, g_i, i))
                    parent_module.register_forward_pre_hook(make_pre_forward_hook_wait(p, g_i, i))
                    # if first layer, there is no forward to overlap comms with
                    # just all-gather them before forward
                    # free up params post-forward
                    parent_module.register_forward_hook(make_post_forward_hook_clean_memory(p, g_i, i))

                    # start pre-fetch for next layer, if present
                    if i+1 < len(group["params"]):
                        next_layer = group["params"][i+1]
                        parent_module.register_forward_pre_hook(make_start_gather_hook(next_layer, g_i, i+1))

                if i+1 < len(group["params"]):
                    next_layer = group["params"][i+1]
                    if p.requires_grad:
                        if next_layer.requires_grad:
                            next_layer.register_hook(make_pre_backwards_gather_hook(p, g_i, i))
                        else:
                            p.register_hook(make_pre_backwards_gather_hook(p, g_i, i))
                else:
                    if p.requires_grad:
                        p.register_hook(make_pre_backwards_gather_hook(p, g_i, i))

                if p.requires_grad:
                    p.register_hook(make_pre_backwards_wait_hook(p, g_i, i))
                    p.register_post_accumulate_grad_hook(hook)

        self.optimizers = {}
        for group in self.param_groups:
            # For the child optimizer, shard_params is params
            # Doing the parent and child optimizer to share memory so zero_grad just works for both
            for i, param in enumerate(group["shard_params"]):
                shard_pg = {**group}
                shard_pg.pop("shard_params")
                shard_pg["params"] = [param]

                self.optimizers[group["params"][i]] = optimizer_cls([shard_pg], **kwargs)

    def step(self, closure: Callable | None = None, **kwargs):
        world_size = dist.get_world_size()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if not p.requires_grad:
                    continue

                handle = self.handles[p]
                handle.wait()
                # p will have been reduce-scattered now
                dest_shard = group["shard_params_buffer"][p]
                dest_shard /= world_size

                # get rid of padding
                start, end = group["shard_params_indices"][i]
                # print(f"{dest_shard.shape=:}, indices: {start, end}, slice: {dest_shard[start:end].shape=:}")
                group["shard_params"][i].grad = dest_shard[start:end]
                p.data = group["shard_params"][i]

                # After gradients are reduce-scattered, the child optimizer
                # has them aggregated across ranks. We can now simply call .step()
                self.optimizers[p].step(closure, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        shard_params = []
        shard_unpadded_size = []

        shard_params_indices = []
        shard_params_buffer = {}
        shard_params_parent = {}
        for p in param_group["params"]:
            parent = p.parent_
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            shard_size = int(math.ceil(p.shape[0] / world_size))
            shard_buffer_shape = [shard_size] + list(p.shape[1:])
            local_start = rank * shard_size
            local_end = min(local_start+shard_size, p.shape[0])
            local_shard = p[local_start:(local_start+shard_size)].detach()

            shard_params.append(local_shard)
            shard_unpadded_size.append(p.shape[0])

            shard_params_indices.append((0, local_end-local_start))
            buff = torch.empty(shard_buffer_shape)

            shard_params_buffer[p] = buff
            shard_params_parent[p] = parent

            p.data = local_shard

        # These are sharded views of the original parameters
        param_group["shard_params"] = shard_params
        param_group["shard_unpadded_size"] = shard_unpadded_size
        param_group["shard_params_indices"] = shard_params_indices
        param_group["shard_params_buffer"] = shard_params_buffer
        param_group["shard_params_parent"] = shard_params_parent

        super().add_param_group(param_group)
