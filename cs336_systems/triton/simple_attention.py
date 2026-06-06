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
    Q_stride_seq, Q_stride_d,
    K_stride_seq, K_stride_d,
    ROWS_Q, ROWS_K, D,
    T_Q: tl.constexpr, T_K: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    q_block_ptr = tl.make_block_ptr(
        Q_ptr,
        shape=(ROWS_Q, D),
        strides=(Q_stride_seq, Q_stride_d),
        block_shape=(T_Q, D),
        offsets=(row_tile_idx*T_Q, 0),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        K_ptr,
        shape=(ROWS_K, D),
        strides=(K_stride_seq, K_stride_d),
        block_shape=(T_K, D),
        offsets=(0, 0),
        order=(1, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr,
        shape=(ROWS_K, D),
        strides=(K_stride_seq, K_stride_d),
        block_shape=(T_K, D),
        offsets=(0, 0),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        O_ptr,
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

        q_kt_v = (q * tl.trans(k)) * v

        output += q_kt_v

        k_block_ptr.advance((T_K, 0))
        v_block_ptr.advance((T_K, 0))
    
    tl.store(o_block_ptr, output, boundary_check=(0, 1))


class SimpleAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        B, q_seq_len, D = Q.shape[0], Q.shape[-2], Q.shape[-1]
        k_seq_len = K.shape[-2]
        
        ctx.T_Q = 16
        ctx.T_K = 16

        o = torch.empty((q_seq_len, D), device=Q.device)

        simle_attention_fwd[((tl.cdiv(q_seq_len, ctx.T_Q), B),)](
            Q,
            K,
            V,
            o,
            Q.stride(-2), Q.stride(-1),
            K.stride(-2), K.stride(-1),
            q_seq_len, k_seq_len, D,
            T_Q=ctx.T_Q, T_K=ctx.T_K
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError()
    

def main():
    f_simple_attention = SimpleAttention.apply

    Q = torch.randn((16, 64, 128))
    K = torch.randn((16, 64, 128))
    V = torch.randn((16, 64, 128))

    Q_triton = Q.clone().detach().requires_grad_(True)
    K_triton = K.clone().detach().requires_grad_(True)
    V_triton = V.clone().detach().requires_grad_(True)

    O = sdpa_simple(Q, K, V)

    O_triton = f_simple_attention(Q_triton, K_triton, V_triton)

    print(O)
    print(O_triton)

if __name__ == "__main__":
    main()
