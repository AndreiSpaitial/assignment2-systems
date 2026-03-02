import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, Tensor

from jaxtyping import Float, Int64

import pandas as pd
import timeit
import typer
import yaml
from tqdm import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_process(
    rank: int,
    train_conf_path: str,
    world_size: int,
    warmup_steps: int = 5,
    training_steps: int = 10,
    model_config: str = "small",
    device: str = "mps",
):
    setup(rank, world_size)
    with open(train_conf_path, "r") as f:
        train_conf = yaml.safe_load(f)

    vocab_size = 10_000

    batch_size = train_conf["batch_size"]

    hyperparameter_sweep = train_conf["configs"]
    model_conf = hyperparameter_sweep[model_config]

    transformer_conf = model_conf["transformer"]
    optimizer_conf = model_conf["optimizer"]

    d_model = transformer_conf["d_model"]
    num_heads = transformer_conf["num_heads"]
    context_length = transformer_conf["context_length"]
    d_model = transformer_conf["d_model"]
    num_layers = transformer_conf["num_layers"]
    d_ff = transformer_conf["d_ff"]
    rope_theta = model_conf["rope"]["theta"]

    torch.random.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    for t in model.state_dict().values():
        dist.broadcast(t, 0, async_op=False)

    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_conf["lr"],
        betas=tuple(optimizer_conf["betas"]),
        weight_decay=optimizer_conf["weight_decay"]
    )

    local_batch_size = batch_size // world_size

    times = []
    for i_step in tqdm(range(training_steps), desc="Training"):
        optimizer.zero_grad()

        x: Int64[Tensor, "b seq_len"] = torch.randint(vocab_size, (local_batch_size, context_length), device=device)
        y = torch.randint(vocab_size, (local_batch_size, context_length), device=device)

        step_start_time = timeit.default_timer()
        y_pred: Float[Tensor, "b seq_len vocab_size"] = model(x)

        loss = cross_entropy(y_pred, y)
        loss.backward()

        flattened_grads = torch._utils._flatten_dense_tensors(
            [p.grad for p in model.parameters()]
        )
        comms_start_time = timeit.default_timer()
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=False)
        torch.mps.synchronize()
        comms_end_time = timeit.default_timer()

        flattened_grads /= world_size
        unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, [p.grad for p in model.parameters()])

        for i, p in enumerate(model.parameters()):
            p.grad = unflattened_grads[i]

        optimizer.step()
        torch.mps.synchronize()
        step_end_time = timeit.default_timer()

        if i_step < warmup_steps:
            times.append({
                "step_time": (step_end_time-step_start_time),
                "comms_time": (comms_end_time-comms_start_time),
            })

    step_times_stats = pd.Series(
        [el["step_time"] for el in times]
    ).describe().to_dict()
    comms_times_stats = pd.Series(
        [el["comms_time"] for el in times]
    ).describe().to_dict()
    comms_step_times_stats = pd.Series(
        [el["comms_time"]/el["step_time"] for el in times]
    ).describe().to_dict()

    print(f"""
    Rank {rank} stats
    Step Times: {step_times_stats}
    Comms Times: {comms_times_stats}
    Comms Times/Step Times: {comms_step_times_stats}
""")


def main(
    train_conf_path: str = "",
    world_size: int = 1,
    warmup_steps: int = 5,
    training_steps: int = 20,
    model_config: str = "small",
    device: str = "cpu",
):
    assert train_conf_path, "Need to specify train conf path"
    mp.spawn(
        fn=train_process,
        args=(
            train_conf_path,
            world_size,
            warmup_steps,
            training_steps,
            model_config,
            device,
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    typer.run(main)
