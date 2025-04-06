import torch
from flash_attn import flash_attn_interface  # 需提前安装 flash-attn 库

# ==============================
# 1. 构造测试数据
# ==============================
torch.manual_seed(41)

# 定义序列长度
q_lens = torch.tensor([3, 2], dtype=torch.int32)  # 两个序列的查询长度
k_lens = torch.tensor([4, 3], dtype=torch.int32)  # 两个序列的键长度
batch_size = len(q_lens)
num_heads = 2
head_dim = 4

# 计算总 token 数
total_q = q_lens.sum().item()  # 3 + 2 = 5
total_k = k_lens.sum().item()  # 4 + 3 = 7

# 生成随机数据 (模拟实际输入)
q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(total_k, num_heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(total_k, num_heads, head_dim, dtype=torch.float16, device="cuda")

# 计算 softmax_scale
softmax_scale = 1.0 / (head_dim ** 0.5)

# 计算累积序列长度
cu_seqlens_q = torch.cat([
    torch.tensor([0], dtype=torch.int32, device="cuda"),
    q_lens.cumsum(0, dtype=torch.int32).to("cuda")
])
cu_seqlens_k = torch.cat([
    torch.tensor([0], dtype=torch.int32, device="cuda"),
    k_lens.cumsum(0, dtype=torch.int32).to("cuda")
])

max_seqlen_q = q_lens.max().item()  # 3
max_seqlen_k = k_lens.max().item()  # 4

# ==============================
# 2. FlashAttention 变长注意力
# ==============================
flash_output = flash_attn_interface.flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=softmax_scale,
    causal=False,
    deterministic=True
)  # 输出形状: [total_q, num_heads, head_dim]

# ==============================
# 3. 手动实现变长注意力
# ==============================
def manual_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale=None, causal=False):
    batch_size = cu_seqlens_q.size(0) - 1
    output = torch.zeros_like(q)  # 形状 [total_q, num_heads, head_dim]
    
    for b in range(batch_size):
        q_start = cu_seqlens_q[b]
        q_end = cu_seqlens_q[b + 1]
        k_start = cu_seqlens_k[b]
        k_end = cu_seqlens_k[b + 1]
        
        curr_q = q[q_start:q_end]  # [seq_len_q, num_heads, head_dim]
        curr_k = k[k_start:k_end]  # [seq_len_k, num_heads, head_dim]
        curr_v = v[k_start:k_end]  # [seq_len_k, num_heads, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.einsum("nhd,mhd->hnm", curr_q, curr_k)  # [num_heads, seq_len_q, seq_len_k]
        
        # 缩放
        if softmax_scale is not None:
            attn_scores = attn_scores * softmax_scale
        
        # 因果掩码
        if causal:
            mask = torch.triu(
                torch.ones(curr_q.size(0), curr_k.size(0), device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        
        # Softmax 和加权求和
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("hnm,mhd->nhd", attn_weights, curr_v)  # [seq_len_q, num_heads, head_dim]
        output[q_start:q_end] = out
    
    return output

manual_output = manual_varlen_attention(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    softmax_scale=softmax_scale,
    causal=False
)  # 输出形状: [total_q, num_heads, head_dim]

# ==============================
# 4. 比较结果
# ==============================
# 计算绝对误差
diff = (flash_output - manual_output).abs().max()
print(f"最大绝对误差: {diff.item()}")  # 期望值 < 1e-4

# 打印第一个序列的第一个头的输出
print("\nFlashAttention 输出 (序列 0, 第一个头):")
print(flash_output[:q_lens[0], 0, :].cpu().float())  # 前3个token是序列0

print("\n手动实现输出 (序列 0, 第一个头):")
print(manual_output[:q_lens[0], 0, :].cpu().float())

# ==============================
# 5. 可选：填充后对齐形状（仅调试）
# ==============================
def pad_to_fixed_shape(output, cu_seqlens, max_seqlen):
    batch_size = cu_seqlens.size(0) - 1
    padded_output = torch.zeros(
        batch_size * max_seqlen, *output.shape[1:],
        dtype=output.dtype, device=output.device
    )
    for b in range(batch_size):
        start = cu_seqlens[b]
        end = cu_seqlens[b + 1]
        padded_start = b * max_seqlen
        padded_end = padded_start + (end - start)
        padded_output[padded_start:padded_end] = output[start:end]
    return padded_output.unflatten(0, (batch_size, max_seqlen))

# 填充后的形状: [batch_size, max_seqlen_q, num_heads, head_dim]
flash_padded = pad_to_fixed_shape(flash_output, cu_seqlens_q, max_seqlen_q)
manual_padded = pad_to_fixed_shape(manual_output, cu_seqlens_q, max_seqlen_q)

print("\n填充后的形状检查:")
print("FlashAttention padded shape:", flash_padded.shape)
print("Manual padded shape:", manual_padded.shape)