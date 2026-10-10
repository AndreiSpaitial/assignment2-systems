import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, Tensor
from torch.optim import SGD

from jaxtyping import Float
from einops import rearrange

import typer
from tqdm import tqdm

from cs336_basics.model import Linear, silu


class BasicModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 512):
        super().__init__()
        self.W1 = Linear(in_features, hidden_dim)
        self.W2 = Linear(hidden_dim, hidden_dim)
        self.W3 = Linear(hidden_dim, 1)

    def forward(
        self, x: Float[Tensor, "b in_features"]
    ) -> Float[Tensor, "b hidden_dim"]:
        x = self.W1(x)
        x = silu(x)
        x = self.W2(x)
        x = silu(x)
        x = self.W3(x)

        return x


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_single(
    train_x_file: str,
    train_y_file: str,
    output_path: str,
    batch_size: int,
):
    train_x: Float[Tensor, "data_ind in_features"] = torch.load(train_x_file)
    train_y: Float[Tensor, "data_ind 1"] = torch.load(train_y_file)

    num_datapoints, in_features = train_x.shape

    torch.random.manual_seed(0)
    model = BasicModel(in_features)

    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for b_start in tqdm(range(0, num_datapoints, batch_size), desc="Training"):
        optimizer.zero_grad()

        data_x = train_x[b_start:(b_start+batch_size)]
        data_y = train_y[b_start:(b_start+batch_size)]

        y_pred = model(data_x)
        loss = loss_fn(y_pred, data_y)
        loss.backward()

        optimizer.step()

    torch.save(model.state_dict(), output_path)


def train_process(
    rank: int,
    train_x_file: str,
    train_y_file: str,
    output_path: str,
    world_size: int,
    batch_size: int,
):
    setup(rank, world_size)
    train_x: Float[Tensor, "data_ind in_features"] = torch.load(train_x_file)
    train_y: Float[Tensor, "data_ind 1"] = torch.load(train_y_file)

    num_datapoints, in_features = train_x.shape
    if rank == 0:
        torch.random.manual_seed(0)
    model = BasicModel(in_features)

    for t in model.state_dict().values():
        dist.broadcast(t, 0, async_op=False)

    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    local_batch_size = batch_size // world_size
    for b_start in tqdm(range(0, num_datapoints, batch_size), desc="Training"):
        optimizer.zero_grad()
        local_index_start = b_start + (rank*local_batch_size)
        local_index_end = local_index_start + local_batch_size

        train_x_local = train_x[local_index_start:local_index_end]
        train_y_local = train_y[local_index_start:local_index_end]

        y_pred = model(train_x_local)
        loss = loss_fn(y_pred, train_y_local)
        loss.backward()

        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
            p.grad /= world_size

        optimizer.step()

    if rank == 0:
        torch.save(model.state_dict(), output_path)


def main(
    train_x_dir: str = "",
    train_y_dir: str = "",
    output_path: str = "",
    world_size: int = 1,
    batch_size: int = 128,
):
    assert train_x_dir, "Need to specify input data dir"
    assert train_y_dir, "Need to specify input data dir"

    if world_size > 1:
        mp.spawn(
            fn=train_process,
            args=(
                train_x_dir,
                train_y_dir,
                output_path,
                world_size,
                batch_size,
            ),
            nprocs=world_size,
            join=True
        )
    else:
        train_single(
            train_x_dir,
            train_y_dir,
            output_path,
            batch_size,
        )


if __name__ == "__main__":
    typer.run(main)
