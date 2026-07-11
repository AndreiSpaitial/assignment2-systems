import os

import pandas as pd
import timeit
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch import Tensor

from einops import einsum
from jaxtyping import Float, Int

from cs336_basics.nn_utils import softmax


DS_MODEL = [16, 32, 64, 128]
DS_SEQ = [256, 1024, 4096, 8192]


def attention(
    Q: Float[Tensor, "... N d_model"],
    K: Float[Tensor, "... M d_model"],
    V: Float[Tensor, "... M d_model"],
    mask: Int[Tensor, "... N M"] | None = None,
):
    Q_KT = einsum(
        Q,
        K,
        "... N d_model, ... M d_model -> ... N M"
    )

    if mask is not None:
        Q_KT = torch.where(mask, Q_KT, float("-inf"))

    sm = softmax(Q_KT, dim=-1)

    return einsum(
        sm,
        V,
        "... N M, ... M d_model -> ... N d_model"
    )


def benchmark_attention(
        d_model,
        N,
        stats_df_ret,
        exp_name="",
        attention_fn=attention,
):
    reps = 10
    warmup = 2

    forward_times = []
    backward_times = []
    memory_before_backwards = []
    conf_name = f"{d_model=:},{N=:}{exp_name}"

    for i in tqdm(range(warmup+reps), desc=f"Timing {conf_name}"):
        Q = torch.randn(
            (8, N, d_model),
            requires_grad=True,
            device="mps"
        )
        K = torch.randn(
            (8, N, d_model),
            requires_grad=True,
            device="mps"
        )
        V = torch.randn(
            (8, N, d_model),
            requires_grad=True,
            device="mps"
        )

        y = torch.randn((8, N, d_model), device="mps")

        start_time = timeit.default_timer()
        y_pred = attention_fn(Q, K, V)
        torch.mps.synchronize()
        end_time = timeit.default_timer()

        loss = F.mse_loss(y, y_pred)
        if i > warmup:
            forward_times.append(end_time-start_time)

        if i > warmup:
            memory_before_backwards.append(
                torch.mps.current_allocated_memory()
            )

        start_time = timeit.default_timer()
        loss.backward()
        torch.mps.synchronize()
        end_time = timeit.default_timer()

        if i > warmup:
            backward_times.append(end_time-start_time)

        stats_df_ret[(conf_name, "forward_time")] = pd.Series(
            forward_times
        ).describe().to_dict()
        stats_df_ret[(conf_name, "backward_time")] = pd.Series(
            backward_times
        ).describe().to_dict()
        stats_df_ret[(conf_name, "memory_usages")] = pd.Series(
            memory_before_backwards
        ).describe().to_dict()


def main():
    stats_df_ret = {}

    for d_model in DS_MODEL:
        for N in DS_SEQ:
            benchmark_attention(
                d_model,
                N,
                stats_df_ret,
            )
            benchmark_attention(
                d_model,
                N,
                stats_df_ret,
                exp_name="_compiled",
                attention_fn=torch.compile(attention, backend="aot_eager")
            )

    stats_df_ret_pd = pd.DataFrame.from_dict(stats_df_ret, orient="index")
    stats_df_ret_pd.index.names = ['Config', 'Metric']
    stats_df_ret_pd = stats_df_ret_pd.reset_index()
    stats_df_ret_pd.loc[stats_df_ret_pd['Config'].duplicated(), 'Config'] = ""
    print(stats_df_ret_pd.head())
    df_markdown = stats_df_ret_pd.to_markdown(index=False)
    ret = f"""
## Benchmarking Results
{df_markdown}
    """

    results_path = "cs336_systems/scripts/benchmarking/attention/results/"
    results_file = os.path.join(results_path, "results.md")
    with open(results_file, "w+") as f:
        f.write(ret)


if __name__ == "__main__":
    main()
