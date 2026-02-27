import os

import torch
from torch import Tensor

import pandas as pd
import timeit
import typer
import yaml

from jaxtyping import Float, Int64
from tqdm import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def main(
    train_conf_path: str = "",
    warmup_steps: int = 5,
    measurement_steps: int = 10,
    forward_only: bool = False,
):
    with open(train_conf_path, "r") as f:
        train_conf = yaml.safe_load(f)

    vocab_size = 10_000

    batch_size = train_conf["batch_size"]
    results_path = train_conf["results_path"]

    hyperparameter_sweep = [("default", train_conf["transformer"])]

    stats_df_ret = {}

    for conf_name, transformer_conf in hyperparameter_sweep:
        print(f"Benchmarking {conf_name=:}")

        stats_df = {}
        d_model = transformer_conf["d_model"]
        num_heads = transformer_conf["num_heads"]
        context_length = transformer_conf["context_length"]
        d_model = transformer_conf["d_model"]
        num_layers = transformer_conf["num_layers"]
        d_ff = transformer_conf["d_ff"]
        rope_theta = train_conf["rope"]["theta"]
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        )

        for _ in range(warmup_steps):
            x: Int64[Tensor, "b seq_len"] = torch.randint(vocab_size, (batch_size, context_length))
            if not forward_only:
                y = torch.randint(vocab_size, (batch_size, context_length))
            y_pred: Float[Tensor, "b seq_len vocab_size"] = model(x)

            if not forward_only:
                loss = cross_entropy(y_pred, y)
                loss.backward()

        print("Warmup finished, starting benchmark")

        forward_times = []
        backward_times = []
        for _ in tqdm(range(measurement_steps), desc="Benchmarking"):
            x = torch.randint(vocab_size, (batch_size, context_length))
            if not forward_only:
                y = torch.randint(vocab_size, (batch_size, context_length))
            start_time = timeit.default_timer()
            y_pred: Float[Tensor, "b seq_len vocab_size"] = model(x)
            torch.mps.synchronize()
            end_time = timeit.default_timer()
            forward_times.append(end_time-start_time)

            if not forward_only:
                start_time = timeit.default_timer()
                loss = cross_entropy(y_pred, y)
                loss.backward()
                torch.mps.synchronize()
                end_time = timeit.default_timer()
                backward_times.append(end_time-start_time)

        stats_df_ret[(conf_name, "forward_time")] = pd.Series(
            forward_times
        ).describe().to_dict()
        stats_df_ret[(conf_name, "backward_time")] = pd.Series(
            backward_times
        ).describe().to_dict()

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

    results_file = os.path.join(results_path, "results.md")
    with open(results_file, "w+") as f:
        f.write(ret)


if __name__ == "__main__":
    typer.run(main)
