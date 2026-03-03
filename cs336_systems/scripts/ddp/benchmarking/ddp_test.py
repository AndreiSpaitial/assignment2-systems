import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit
import typer
from tqdm import tqdm


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_benchmark(rank, world_size, data_size_bytes, warmup_steps):
    setup(rank, world_size)
    tensor_size = math.ceil(math.sqrt(data_size_bytes//4))
    for _ in tqdm(range(warmup_steps), desc="Performing warmup"):
        data = torch.randn((tensor_size, tensor_size))
        dist.all_reduce(data, async_op=False)

    print("Benchmarking")
    data = torch.randn((tensor_size, tensor_size))
    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    end_time = timeit.default_timer()
    print(f"All-reduce took {(end_time-start_time) * 1000:.3f} ms")


def main(
    world_size: int = 1,
    data_size_bytes: str = "1024",
    warmup_steps: int = 5,
):
    data_size_bytes = int(eval(data_size_bytes))
    mp.spawn(
        fn=distributed_benchmark,
        args=(world_size, data_size_bytes, warmup_steps),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    typer.run(main)
