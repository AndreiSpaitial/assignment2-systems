import math
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

from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int

from cs336_basics.nn_utils import softmax


def sdpa_simple(
    Q: Float[Tensor, "B ... q_seq_len D"],
    K: Float[Tensor, "B ... k_seq_len D"],
    V: Float[Tensor, "B ... k_seq_len D"],
) -> Float[Tensor, "B ... seq_len D"]:
    d_k = Q.shape[-1]

    Q_KT = einsum(
        Q, K,
        "... q_seq_len D, ... k_seq_len D -> ... q_seq_len k_seq_len"
    )

    Q_KT /= math.sqrt(d_k)

    sm = softmax(Q_KT, dim=-1)

    L = torch.log(
        reduce(
            torch.exp(Q_KT),
            "B ... q_seq_len k_seq_len -> B ... q_seq_len",
            "sum"
        )
    )

    O = einsum(
        sm, V,
        "... q_seq_len k_seq_len, ... k_seq_len D -> ... q_seq_len D"
    )

    return O, L


@triton.jit
def diag(v, TILE_Q):
    diag_mask = tl.arange(0, TILE_Q)[:, None] == tl.arange(0, TILE_Q)[None, :]

    return tl.where(diag_mask, v, 0.0)

@triton.jit
def simle_attention_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    scale,
    Q_stride_b, Q_stride_head, Q_stride_seq, Q_stride_d,
    K_stride_b, K_stride_head, K_stride_seq, K_stride_d,
    L_stride_b, L_stride_head, L_stride_seq,
    ROWS_Q, ROWS_K, D: tl.constexpr,
    T_Q: tl.constexpr, T_K: tl.constexpr,
    is_causal: tl.constexpr,
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

    l_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * L_stride_b + head_index * L_stride_head,
        shape=(ROWS_Q,),
        strides=(L_stride_seq,),
        block_shape=(T_Q,),
        offsets=(row_tile_idx*T_Q,),
        order=(0,),
    )

    output = tl.zeros((T_Q, D), dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l = tl.zeros((T_Q, 1), dtype=tl.float32)
    m = tl.zeros((T_Q, 1), dtype=tl.float32) + float("-inf")

    indices_T_K = tl.zeros((T_Q, T_K), dtype=tl.float16) + 1
    indices_T_K = tl.cumsum(indices_T_K, axis=1) - 1

    indices_T_Q = tl.zeros((T_Q, T_K), dtype=tl.float16) + 1
    indices_T_Q = tl.cumsum(indices_T_Q, axis=0) - 1
    indices_T_Q += row_tile_idx * T_Q

    for i in range(tl.cdiv(ROWS_K, T_K)):
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # print(f"{m.shape=:}")
        # print(f"{l.shape=:}")

        # print(f"{q.shape=:}")
        # print(f"{k.shape=:}")
        # print(f"{v.shape=:}")
        # print(f"{output.shape=:}")

        # print(f"{q=:}")
        # print(f"{k=:}")
        # print(f"{v=:}")

        q_kt = tl.dot(q, tl.trans(k))
        q_kt /= scale
        if is_causal:
            q_kt = tl.where(indices_T_K > indices_T_Q, -1e6, q_kt)

        rm = tl.max(q_kt, axis=1, keep_dims=True)
        m_new = tl.maximum(m, rm)

        # print(f"{rm.shape=:}")
        # print(f"{m_new.shape=:}")

        P = tl.exp(q_kt - m_new)
        # print(f"{P.shape=:}")
        
        m_exp = tl.exp(m - m_new)
        l = m_exp * l + tl.sum(P, axis=1, keep_dims=True)

        # print(f"{l.shape=:}")
        # print(m_exp)
        # print(l)
        
        output = tl.dot(diag(m_exp, T_Q), output) + tl.dot(P, v)

        m = m_new

        # print(f"{output.shape=:}")
        # print("---------\n"*2)

        k_block_ptr = k_block_ptr.advance((T_K, 0))
        v_block_ptr = v_block_ptr.advance((T_K, 0))

        indices_T_K += T_K
    

    # print(l)
    # print(diag(1./l, T_Q))
    # print("---------\n"*2)
    output = tl.dot(diag(1./l, T_Q), output)
    L = m + tl.log(l)
    L = tl.reshape(L, (T_Q,))

    tl.store(o_block_ptr, output, boundary_check=(0, 1))
    tl.store(l_block_ptr, L, boundary_check=(0,))


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert (
            Q.is_contiguous() and
            K.is_contiguous() and
            V.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous x"

        fake_heads = False
        if len(Q.shape) == 3:
            fake_heads = True
            Q = rearrange(Q, "B N D -> B 1 N D")
            K = rearrange(K, "B M D -> B 1 M D")
            V = rearrange(V, "B M D -> B 1 M D")
        
        B, H, q_seq_len, D = Q.shape

        k_seq_len = K.shape[-2]
        scale = math.sqrt(D)

        ctx.T_Q = 16
        ctx.T_K = 16
        ctx.D = D
        ctx.is_causal = is_causal

        o = torch.empty_like(Q)
        l = torch.empty((B, H, q_seq_len))

        simle_attention_fwd[(tl.cdiv(q_seq_len, ctx.T_Q), B, H)](
            Q,
            K,
            V,
            o,
            l,
            scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            q_seq_len, k_seq_len, ctx.D,
            T_Q=ctx.T_Q, T_K=ctx.T_K,
            is_causal=ctx.is_causal
        )

        if fake_heads:
            o = rearrange(o, "B 1 N D -> B N D")
            l = rearrange(l, "B 1 N -> B N")
            
            Q = rearrange(Q, "B 1 N D -> B N D")
            K = rearrange(K, "B 1 M D -> B M D")
            V = rearrange(V, "B 1 M D -> B M D")

        ctx.save_for_backward(Q, K, V, o, l)

        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        dO = grad_outputs[0]

        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        q_seq_len, D = Q.shape[-2:]
        k_seq_len = K.shape[-2]

        T_Q = 16
        T_K = 32
        scale = 1./math.sqrt(D)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for t_k in range(0, k_seq_len, T_K):
            for t_q in range(0, q_seq_len, T_Q):
                S_qk: Float[Tensor, "B ... T_Q T_K"] = scale * einsum(
                    Q[..., t_q:(t_q + T_Q), :],
                    K[..., t_k:(t_k + T_K), :],
                    "B ... T_Q D, B ... T_K D -> B ... T_Q T_K"
                )

                indices_T_K: Float[Tensor, "B ... T_Q T_K"] = torch.ones_like(S_qk)
                indices_T_K = torch.cumsum(indices_T_K, axis=-1) - 1 + t_k
                indices_T_Q: Float[Tensor, "B ... T_Q T_K"] = torch.ones_like(S_qk)
                indices_T_Q = torch.cumsum(indices_T_Q, axis=-2) - 1 + t_q

                if is_causal:
                    S_qk[indices_T_K > indices_T_Q] = -1e6

                L_i = repeat(L[..., t_q:(t_q + T_Q)], "B ... q_seq_len -> B ... q_seq_len T_K", T_K=T_K)
                # print(f"{S_qk.shape=:}")
                # print(f"{L_i.shape=:}")

                P_qk: Float[Tensor, "B ... T_Q T_K"] = torch.exp(
                    S_qk - L_i
                )
                # print(f"{P_qk.shape=:}")

                dO_q: Float[Tensor, "B ... T_Q D"] = dO[..., t_q:(t_q + T_Q), :]
                O_q : Float[Tensor, "B ... T_Q D"] =  O[..., t_q:(t_q + T_Q), :]

                dP_qk = einsum(
                    dO_q, V[..., t_k:(t_k + T_K), :],
                    "B ... T_Q D, B ... T_K D" 
                    "-> B ... T_Q T_K"   
                )

                # print(f"{dP_qk.shape=:}")
                # print(f"{dO_q.shape=:}")
                # print(f"{O_q.shape=:}")

                D_q = reduce(
                    dO_q * O_q,
                    "B ... T_Q D -> B ... T_Q",
                    "sum"
                )
                D_q = repeat(
                    D_q,
                    "B ... T_Q -> B ... T_Q T_K",
                    T_K=T_K,
                )
                # print(f"{D_q.shape=:}")

                dS_qk = -P_qk * D_q + P_qk * dP_qk

                # print(f"{S_qk.shape=:}")

                dV[..., t_k:(t_k + T_K), :] += einsum(
                    P_qk, dO_q,
                    "B ... T_Q T_K, B ... T_Q D -> B ... T_K D"
                )

                dK[..., t_k:(t_k + T_K), :] += scale * einsum(
                    dS_qk, Q[..., t_q:(t_q + T_Q), :],
                    "B ... T_Q T_K, B ... T_Q D"
                    "-> B ... T_K D"
                )
                dQ[..., t_q:(t_q + T_Q), :] += scale * einsum(
                    dS_qk, K[..., t_k:(t_k + T_K), :],
                    "B ... T_Q T_K,"
                    "B ... T_K D"
                    "-> B ... T_Q D"
                )

                # print("----------")

        return dQ, dK, dV, None


def main():
    f_simple_attention = TritonAttention.apply

    Q = torch.randn((1, 4, 128, 32), requires_grad=True)
    K = torch.randn((1, 4, 256, 32), requires_grad=True)
    V = torch.randn((1, 4, 256, 32), requires_grad=True)

    Q_triton = Q.clone().detach().requires_grad_(True)
    K_triton = K.clone().detach().requires_grad_(True)
    V_triton = V.clone().detach().requires_grad_(True)

    y = torch.randn((1, 4, 128, 32))

    O, L = sdpa_simple(Q, K, V)
    loss = F.mse_loss(O, y)

    O_triton = f_simple_attention(Q_triton, K_triton, V_triton)
    loss_triton = F.mse_loss(O_triton, y)

    L_triton = O_triton.grad_fn.saved_tensors[-1]

    loss.backward()
    loss_triton.backward()

    # print(f"{O.shape=:}")
    # print(f"{L.shape=:}")
    # print(O)
    # print(O_triton)

    numpy.testing.assert_allclose(
        O.detach().numpy(),
        O_triton.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    numpy.testing.assert_allclose(
        L.detach().numpy(),
        L_triton.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    numpy.testing.assert_allclose(
        V.grad.detach().numpy(),
        V_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    numpy.testing.assert_allclose(
        K.grad.detach().numpy(),
        K_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    numpy.testing.assert_allclose(
        Q.grad.detach().numpy(),
        Q_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    Q = torch.randn((1, 128, 32), requires_grad=True)
    K = torch.randn((1, 256, 32), requires_grad=True)
    V = torch.randn((1, 256, 32), requires_grad=True)

    Q_triton = Q.clone().detach().requires_grad_(True)
    K_triton = K.clone().detach().requires_grad_(True)
    V_triton = V.clone().detach().requires_grad_(True)
    y = torch.randn((1, 128, 32))

    O, L = sdpa_simple(Q, K, V)
    loss = F.mse_loss(O, y)

    O_triton = f_simple_attention(Q_triton, K_triton, V_triton)
    loss_triton = F.mse_loss(O_triton, y)

    L_triton = O_triton.grad_fn.saved_tensors[-1]

    loss.backward()
    loss_triton.backward()

    # print(O.shape)
    # print(O_triton.shape)

    numpy.testing.assert_allclose(
        O.detach().numpy(),
        O_triton.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    numpy.testing.assert_allclose(
        L.detach().numpy(),
        L_triton.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    numpy.testing.assert_allclose(
        V.grad.detach().numpy(),
        V_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    numpy.testing.assert_allclose(
        K.grad.detach().numpy(),
        K_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    numpy.testing.assert_allclose(
        Q.grad.detach().numpy(),
        Q_triton.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    print("all good")

if __name__ == "__main__":
    main()
