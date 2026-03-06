import math
from copy import deepcopy
from typing import Type

import numpy
import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from .adapters import get_sharded_optimizer
from .common import (
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
)

from cs336_basics.optimizer import AdamW


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
@pytest.mark.parametrize("zero_stage", [0, 1, 2])
def test_sharded_optimizer(model_class, zero_stage):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer,
        args=(world_size, model_class, zero_stage),
        nprocs=world_size,
        join=True,
    )


def _test_sharded_optimizer(rank: int, world_size: int, model_class: Type[torch.nn.Module], zero_stage: int):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW
    # optimizer_cls = AdamW
    # Since we've seeded, model states should be the same across ranks without having to broadcast.
    non_sharded_model = model_class().to(device)

    non_sharded_optimizer = optimizer_cls(
        non_sharded_model.parameters(),
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = get_sharded_optimizer(
        sharded_model.parameters(),
        zero_stage,
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )

    for i in range(10):
        non_sharded_optimizer.zero_grad(set_to_none=False)
        sharded_optimizer.zero_grad(set_to_none=False)

        # batch size 32, 10 input features, 5 output features
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)
        non_sharded_input = deepcopy(input_)
        sharded_input = deepcopy(input_)
        non_sharded_labels = deepcopy(labels)
        sharded_labels = deepcopy(labels)

        non_sharded_model_logits = non_sharded_model(non_sharded_input)
        sharded_model_logits = sharded_model(sharded_input)

        non_sharded_model_loss = ((non_sharded_labels - non_sharded_model_logits) ** 2).sum()
        sharded_model_loss = ((sharded_labels - sharded_model_logits) ** 2).sum()

        non_sharded_model_loss.backward()
        sharded_model_loss.backward()

        non_sharded_optimizer.step()
        sharded_optimizer.step()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # for non_sharded_parameters, sharded_parameters in zip(non_sharded_optimizer.param_groups[0]["params"], sharded_optimizer.param_groups[0]["shard_params"]):
        #     if non_sharded_parameters.grad is None:
        #         continue
        #     n = non_sharded_parameters.shape[0]
        #     shard_size = int(math.ceil(n / world_size))
        #     local_start = shard_size * rank
        #     local_end = local_start + shard_size
        #     non_sharded_parameters_local_grad = non_sharded_parameters.grad[local_start:local_end]

        #     numpy.testing.assert_allclose(
        #         non_sharded_parameters_local_grad.detach().cpu().numpy(),
        #         sharded_parameters.grad.detach().cpu().numpy(),
        #         rtol=5e-5,
        #         atol=5e-8,
        #     )

        # for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        #     numpy.testing.assert_allclose(
        #         non_sharded_parameters.detach().cpu().numpy(),
        #         sharded_parameters.detach().cpu().numpy(),
        #         rtol=5e-5,
        #         atol=5e-8,
        #     )

        # print(f"Iteration {i} OK")

    # Check that the final model weights are the same regardless of if we're using
    # the sharded or non-sharded optimizer.
    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
    _cleanup_process_group()
