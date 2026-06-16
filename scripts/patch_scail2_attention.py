#!/usr/bin/env python3
import sys
from pathlib import Path


PATCH_MARKER = "SCAIL2_RUNPOD_SDPA_FALLBACK"

PATCHED_ATTENTION = r'''# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SCAIL2_RUNPOD_SDPA_FALLBACK: allow inference without external flash-attn wheels.
import math
import warnings

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
]


def _lens_to_list(lengths, default, batch):
    if lengths is None:
        return [default] * batch
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.detach().cpu().tolist()
    if len(lengths) != batch:
        raise ValueError(f'Expected {batch} sequence lengths, got {len(lengths)}.')
    return [max(0, min(int(length), default)) for length in lengths]


def _repeat_kv_for_gqa(x, query_heads):
    key_heads = x.size(1)
    if key_heads == query_heads:
        return x
    if query_heads % key_heads != 0:
        raise ValueError(f'Query heads ({query_heads}) must be divisible by key/value heads ({key_heads}).')
    return x.repeat_interleave(query_heads // key_heads, dim=1)


def _torch_sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise AssertionError('dtype must be torch.float16 or torch.bfloat16')

    if window_size != (-1, -1):
        warnings.warn(
            'Sliding-window attention is not supported by the torch SDPA fallback; '
            'using full attention instead.'
        )
    if deterministic:
        warnings.warn('deterministic=True is ignored by the torch SDPA fallback.')

    b, lq, query_heads, _ = q.shape
    lk = k.size(1)
    out_dtype = q.dtype

    q = q if q.dtype in half_dtypes else q.to(dtype)
    k = k if k.dtype in half_dtypes else k.to(dtype)
    v = v if v.dtype in half_dtypes else v.to(dtype)
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    k = _repeat_kv_for_gqa(k, query_heads)
    v = _repeat_kv_for_gqa(v, query_heads)

    q_lengths = _lens_to_list(q_lens, lq, b)
    k_lengths = _lens_to_list(k_lens, lk, b)
    output_parts = []

    for batch_idx, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        if q_len == 0:
            output_parts.append(q.new_zeros((1, query_heads, lq, v.size(-1))))
            continue
        if k_len == 0:
            raise ValueError('Key/value sequence length must be positive.')

        q_item = q[batch_idx:batch_idx + 1, :, :q_len, :]
        k_item = k[batch_idx:batch_idx + 1, :, :k_len, :]
        v_item = v[batch_idx:batch_idx + 1, :, :k_len, :]
        if softmax_scale is not None:
            q_item = q_item * (float(softmax_scale) * math.sqrt(q_item.size(-1)))

        out_item = torch.nn.functional.scaled_dot_product_attention(
            q_item,
            k_item,
            v_item,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
        )

        if q_len < lq:
            padded = q.new_zeros((1, query_heads, lq, out_item.size(-1)))
            padded[:, :, :q_len, :] = out_item
            out_item = padded
        output_parts.append(out_item)

    return torch.cat(output_parts, dim=0).transpose(1, 2).contiguous().type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if version == 2 and not FLASH_ATTN_2_AVAILABLE:
        if FLASH_ATTN_3_AVAILABLE:
            warnings.warn('Flash attention 2 is not available, use flash attention 3 instead.')
            version = 3
        else:
            return _torch_sdpa_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
            )

    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        return _torch_sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
'''


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/SCAIL-2")
    attention_path = repo / "wan" / "modules" / "attention.py"
    if not attention_path.is_file():
        raise FileNotFoundError(f"Missing SCAIL-2 attention module: {attention_path}")

    current = attention_path.read_text(encoding="utf-8")
    if PATCH_MARKER in current:
        print(f"SCAIL-2 attention fallback patch already applied: {attention_path}")
        return
    if "assert FLASH_ATTN_2_AVAILABLE" not in current:
        raise RuntimeError("Unexpected attention.py contents; refusing to patch.")

    attention_path.write_text(PATCHED_ATTENTION, encoding="utf-8")
    print(f"Applied SCAIL-2 torch SDPA fallback patch: {attention_path}")


if __name__ == "__main__":
    main()
