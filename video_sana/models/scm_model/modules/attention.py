# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import sys
import os
import pdb
current_file_path = os.path.abspath(__file__)
up_levels = 5
for _ in range(up_levels):
    current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)
print(current_file_path)

from flash_attn_scm import flash_attn_interface_scm


try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    
    
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False



import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def manual_varlen_attention(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    softmax_scale=None,
    causal=False,
    deterministic=True,
    block_size=256  # 可调整的分块大小
):
    batch_size = cu_seqlens_q.size(0) - 1
    num_heads = q.size(1)
    head_dim = q.size(2)
    
    output = torch.zeros_like(q)
    
    for b in range(batch_size):
        q_start = cu_seqlens_q[b]
        q_end = cu_seqlens_q[b+1]
        k_start = cu_seqlens_k[b]
        k_end = cu_seqlens_k[b+1]
        
        curr_q = q[q_start:q_end]  # [seq_len_q, num_heads, head_dim]
        curr_k = k[k_start:k_end]  # [seq_len_k, num_heads, head_dim]
        curr_v = v[k_start:k_end]  # [seq_len_k, num_heads, head_dim]
        
        seq_len_q, seq_len_k = curr_q.size(0), curr_k.size(0)
        
        # 分块计算注意力
        out = torch.zeros_like(curr_q)
        for i in range(0, seq_len_q, block_size):
            # 处理查询块
            q_block = curr_q[i:i+block_size]
            
            # 计算块注意力分数
            attn_scores = torch.einsum('nhd,mhd->hnm', q_block, curr_k)
            
            if softmax_scale is not None:
                attn_scores = attn_scores * softmax_scale
            
            if causal:
                # 为当前块创建因果掩码
                mask = torch.triu(
                    torch.ones(q_block.size(0), seq_len_k, 
                    device=q.device, dtype=torch.bool
                ), diagonal=1+i)
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            out_block = torch.einsum('hnm,mhd->nhd', attn_weights, curr_v)
            out[i:i+block_size] = out_block
        
        output[q_start:q_end] = out
    
    output = output.unflatten(0, (batch_size, max_seqlen_q))
    return output



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
    jvp = False,
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
    jvp:            bool. If True, apply jvp.
    """
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
    
    # print(f"flashattentionjvp:{jvp}")
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        if jvp:
            print(f"jvp:{jvp}")
            x = flash_attn_interface_scm.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
            
            
        else:
            print(f"jvp:{jvp}")
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        if jvp:
            print(f"jvp:{jvp}")
            # x = flash_attn_interface_scm.flash_attn_varlen_func(
            #     q=q,
            #     k=k,
            #     v=v,
            #     cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            #         0, dtype=torch.int32).to(q.device, non_blocking=True),
            #     cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            #         0, dtype=torch.int32).to(q.device, non_blocking=True),
            #     max_seqlen_q=lq,
            #     max_seqlen_k=lk,
            #     softmax_scale=softmax_scale,
            #     causal=causal,
            #     deterministic=deterministic).unflatten(0, (b, lq))
            x = manual_varlen_attention(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
            # pdb.set_trace()

        else:
            # print(f"jvp:{jvp}")
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
                deterministic=deterministic,
                ).unflatten(0, (b, lq)),

    # output
    return x[0].type(out_dtype)


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
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
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
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out



