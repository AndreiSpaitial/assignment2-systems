import os

import numpy
import pandas as pd
import timeit
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torch import Tensor

from einops import einsum, rearrange
from jaxtyping import Float, Int


def weighted_sum(
        X: Float[Tensor, "B ... D"],
        w: Float[Tensor, "D"],
) -> Float[Tensor, "B ..."]:

    return (w * X).sum(axis=-1)


@triton.jit
def weighted_sum_fwd(
    X_ptr,
    w_ptr,
    out_ptr,
    X_stride_row, X_stride_dim,
    w_stride_dim,
    out_stride_row,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(ROWS, D),
        strides=(X_stride_row, X_stride_dim),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, 0),
        order=(1, 0),
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(w_stride_dim,),
        block_shape=(D_TILE_SIZE,),
        offsets=(0,),
        order=(0,)
    )

    out_block_ptr = tl.make_block_ptr(
        out_ptr,
        shape=(ROWS,),
        strides=(out_stride_row,),
        block_shape=(ROWS_TILE_SIZE,),
        offsets=(row_tile_idx*ROWS_TILE_SIZE,),
        order=(0,)
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")

        output += tl.sum(x * w[None, :], axis=1)
        # print(f"{row_tile_idx=:} {i=:}, {x=:}, {w=:}, {output=:}")

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))

    tl.store(out_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    X_ptr, w_ptr,
    grad_ptr,
    out_X_grad_ptr, out_w_grad_ptr,
    X_stride_row, X_stride_dim,
    w_stride_dim,
    grad_stride_row,
    gX_stride_row, gX_stride_dim,
    gW_stride_row, gW_stride_dim,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    gW_rows = tl.num_programs(0)

    x_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(ROWS, D),
        strides=(X_stride_row, X_stride_dim),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, 0),
        order=(1, 0),
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(w_stride_dim,),
        block_shape=(D_TILE_SIZE,),
        offsets=(0,),
        order=(0,)
    )

    grad_block_ptr = tl.make_block_ptr(
        grad_ptr,
        shape=(ROWS,),
        strides=(grad_stride_row,),
        block_shape=(ROWS_TILE_SIZE,),
        offsets=(row_tile_idx*ROWS_TILE_SIZE,),
        order=(0,)
    )

    gX_block_ptr = tl.make_block_ptr(
        out_X_grad_ptr,
        shape=(ROWS, D),
        strides=(gX_stride_row, gX_stride_dim),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, 0),
        order=(1, 0),
    )

    gw_block_ptr = tl.make_block_ptr(
        out_w_grad_ptr,
        shape=(gW_rows, D),
        strides=(gW_stride_row, gW_stride_dim),
        block_shape=(1, D_TILE_SIZE),
        offsets=(row_tile_idx, 0),
        order=(1, 0)
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")
        dL = tl.load(grad_block_ptr, boundary_check=(0,), padding_option="zero")

        gx = dL[:, None] * w[None, :]
        gw = tl.sum(x * dL[:, None], axis=0, keep_dims=True)

        tl.store(gw_block_ptr, gw, boundary_check=(1,))
        tl.store(gX_block_ptr, gx, boundary_check=(0, 1))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))

        gX_block_ptr = gX_block_ptr.advance((0, D_TILE_SIZE))
        gw_block_ptr = gw_block_ptr.advance((0, D_TILE_SIZE,))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)

        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device)
        n_rows = y.numel()

        weighted_sum_fwd[(tl.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_w_grad = torch.empty((tl.cdiv(n_rows, ROWS_TILE_SIZE), D))
        x_grad = torch.empty((n_rows, D))

        weighted_sum_backward[(tl.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            x_grad, partial_w_grad,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            x_grad.stride(0), x_grad.stride(1),
            partial_w_grad.stride(0), partial_w_grad.stride(1),
            n_rows, D,
            ROWS_TILE_SIZE, D_TILE_SIZE,
        )

        w_grad = partial_w_grad.sum(axis=0)

        return x_grad, w_grad


def main():
    f_weighted_sum = WeightedSumFunc.apply

    X = torch.randn((125, 1024), requires_grad=True)
    X_triton = X.clone().detach().requires_grad_(True)

    w = torch.randn(1024, requires_grad=True)
    w_triton = w.clone().detach().requires_grad_(True)

    y_target = torch.randn(125)

    y = weighted_sum(X, w)
    y_triton = f_weighted_sum(X_triton, w_triton)

    numpy.testing.assert_allclose(
        y.detach().numpy(),
        y_triton.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    loss = F.mse_loss(y, y_target)
    loss_triton = F.mse_loss(y_triton, y_target)

    loss.backward()
    loss_triton.backward()

    numpy.testing.assert_allclose(
        X.grad.detach().numpy(),
        X_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    numpy.testing.assert_allclose(
        w.grad.detach().numpy(),
        w_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    print("All good")


if __name__ == "__main__":
    main()
