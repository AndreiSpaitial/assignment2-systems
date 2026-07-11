import math
from copy import deepcopy
from collections.abc import Callable, Iterable
from typing import Any, Type

import torch
from torch import Tensor
import torch.distributed as dist


class ZeRO3Optimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        super().__init__(params, {})
    
    def step(self, closure: Callable | None = None, **kwargs):
        raise NotImplementedError()
