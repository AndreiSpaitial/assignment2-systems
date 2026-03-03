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
