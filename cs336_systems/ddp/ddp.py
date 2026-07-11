import torch
import torch.distributed as dist
from torch import nn, Tensor


class DDPModel(nn.Module):
    def __init__(
            self,
            module: nn.Module,
            world_size: int,
    ):
        super().__init__()

        self.module = module
        self.handles: list = []
        self.world_size = world_size

        def hook(p: Tensor) -> None:
            handle = dist.all_reduce(
                p.grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            self.handles.append((p, handle))

        for t in self.module.state_dict().values():
            dist.broadcast(t, 0, async_op=False)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for p, handle in self.handles:
            handle.wait()
            p.grad /= self.world_size
        self.handles.clear()


class BucketedDDPModel(nn.Module):
    def __init__(
            self,
            module: nn.Module,
            bucket_size_mb: int,
            world_size: int,
    ):
        super().__init__()

        self.module = module
        self.handles: list = []
        self.buffer: list = []
        self.estimated_buffer_size = 0
        self.world_size = world_size

        def hook(p: Tensor) -> None:
            estimated_grad_size = p.grad.numel() * p.grad.element_size()
            estimated_grad_size /= 1024 * 1024
            if self.estimated_buffer_size + estimated_grad_size > bucket_size_mb:
                self._flush_bucket()

            self.buffer.append(p)
            self.estimated_buffer_size += estimated_grad_size
            

        for t in self.module.state_dict().values():
            dist.broadcast(t, 0, async_op=False)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

    def _flush_bucket(self):
        if len(self.buffer) == 0:
            return
        flattened_grads = torch._utils._flatten_dense_tensors(
            [el.grad for el in self.buffer]
        )
        handle = dist.all_reduce(
            flattened_grads,
            op=dist.ReduceOp.SUM,
            async_op=True
        )
        self.handles.append(
            (self.buffer.copy(), flattened_grads, handle)
        )
        self.buffer.clear()
        self.estimated_buffer_size = 0

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self._flush_bucket()

        for ps, flattened_grads, handle in self.handles:
            handle.wait()
            flattened_grads /= self.world_size
            unflattened_grads = torch._utils._unflatten_dense_tensors(
                flattened_grads, [p.grad for p in ps]
            )
            for i,p in enumerate(ps):
                p.grad = unflattened_grads[i]
        self.handles.clear()
