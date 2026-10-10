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


def sdpa_simple(
    Q: Float[Tensor, "B ... q_seq_len D"],
    K: Float[Tensor, "B ... k_seq_len D"],
    V: Float[Tensor, "B ... k_seq_len D"],
) -> Float[Tensor, "B ... seq_len D"]:
    Q_KT = einsum(
        Q, K,
        "... q_seq_len D, ... k_seq_len D -> ... q_seq_len k_seq_len"
    )

    O = einsum(
        Q_KT, V,
        "... q_seq_len k_seq_len, ... k_seq_len D -> ... q_seq_len D"
    )

    return O


@triton.jit
def simle_attention_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Q_stride_b, Q_stride_head, Q_stride_seq, Q_stride_d,
    K_stride_b, K_stride_head, K_stride_seq, K_stride_d,
    ROWS_Q, ROWS_K, D: tl.constexpr,
    T_Q: tl.constexpr, T_K: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)
    head_index = tl.program_id(2)

    q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * Q_stride_b + head_index * Q_stride_head,
        shape=(ROWS_Q, D),
        strides=(Q_stride_seq, Q_stride_d),
        block_shape=(T_Q, D),
        offsets=(row_tile_idx*T_Q, 0),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * K_stride_b + head_index * K_stride_head,
        shape=(ROWS_K, D),
        strides=(K_stride_seq, K_stride_d),
        block_shape=(T_K, D),
        offsets=(0, 0),
        order=(1, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * K_stride_b + head_index * K_stride_head,
        shape=(ROWS_K, D),
        strides=(K_stride_seq, K_stride_d),
        block_shape=(T_K, D),
        offsets=(0, 0),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * Q_stride_b + head_index * Q_stride_head,
        shape=(ROWS_Q, D),
        strides=(Q_stride_seq, Q_stride_d),
        block_shape=(T_Q, D),
        offsets=(row_tile_idx*T_Q, 0),
        order=(1, 0),
    )

    output = tl.zeros((T_Q, D), dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for i in range(tl.cdiv(ROWS_K, T_K)):
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # print(f"{q.shape=:}")
        # print(f"{k.shape=:}")
        # print(f"{v.shape=:}")
        # print(f"{output.shape=:}")

        # print(f"{q=:}")
        # print(f"{k=:}")
        # print(f"{v=:}")

        q_kt_v = tl.dot(tl.dot(q, tl.trans(k)), v)

        # print(f"{q_kt_v.shape=:}")
        # print(f"{q_kt_v=:}")

        output += q_kt_v

        k_block_ptr = k_block_ptr.advance((T_K, 0))
        v_block_ptr = v_block_ptr.advance((T_K, 0))
    
    tl.store(o_block_ptr, output, boundary_check=(0, 1))


class SimpleAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        assert (
            Q.is_contiguous() and
            K.is_contiguous() and
            V.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous x"

        B, H, q_seq_len, D = Q.shape
        k_seq_len = K.shape[-2]

        ctx.T_Q = 16
        ctx.T_K = 16
        ctx.D = D

        o = torch.empty_like(Q)

        simle_attention_fwd[(tl.cdiv(q_seq_len, ctx.T_Q), B, H)](
            Q,
            K,
            V,
            o,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            q_seq_len, k_seq_len, ctx.D,
            T_Q=ctx.T_Q, T_K=ctx.T_K
        )

        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError()
    

def main():
    f_simple_attention = SimpleAttention.apply

    Q = torch.randn((1, 4, 128, 32))
    K = torch.randn((1, 4, 256, 32))
    V = torch.randn((1, 4, 256, 32))

    Q_triton = Q.clone().detach().requires_grad_(True)
    K_triton = K.clone().detach().requires_grad_(True)
    V_triton = V.clone().detach().requires_grad_(True)

    O = sdpa_simple(Q, K, V)

    O_triton = f_simple_attention(Q_triton, K_triton, V_triton)

    # print(O)
    # print(O_triton)

    numpy.testing.assert_allclose(
        O.detach().numpy(),
        O_triton.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )

    print("all good")

if __name__ == "__main__":
    main()
