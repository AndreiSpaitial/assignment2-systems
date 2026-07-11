import typer


def main(
    d_model: int = 16_384,
    d_ff: int = 53_248,
    num_blocks: int = 126,
    n_fsdp_devices: int = 1,
):
    def to_gb(n_bytes: int) -> float:
        return n_bytes / (1024**3)

    master_weights = (
        4 * (d_model * d_ff) +  # FF1
        4 * (d_ff * d_model)  # FF2
    ) * num_blocks
    gradients = master_weights
    optimizer_state = 2 * master_weights

    activations = (
        2 * d_ff +  # FF1 activations
        2 * d_model  # FF2 activations
    ) * num_blocks

    ret = f"""
    Total memory for optimizer and master weights: {to_gb(master_weights + gradients + optimizer_state)=:} GB
    Total memory for backwards: {to_gb(activations)=:} GB
    Would need {(to_gb(master_weights + gradients + optimizer_state) + to_gb(activations)) / 80} H100 devices

    With FSDP sharding on {n_fsdp_devices} GPUs:
    Total memory for optimizer and master weights: {to_gb(master_weights + gradients + optimizer_state)/n_fsdp_devices=:} GB per device
    Total memory for backwards: {to_gb(activations)/n_fsdp_devices=:} GB per device
    """

    print(ret)


if __name__ == "__main__":
    typer.run(main)
