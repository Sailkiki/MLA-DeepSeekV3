import torch
import torch.nn as nn
import math
from typing import Optional

# 全局配置
world_size = 1  # 假设单机单卡
attn_impl = "naive"  # 使用朴素注意力实现

# 模拟并行线性层（简化为普通线性层）
class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features)


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features)


# RMS归一化层
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


# 模型参数配置
class ModelArgs:
    def __init__(self):
        self.dim = 512
        self.n_heads = 8
        self.q_lora_rank = 0
        self.kv_lora_rank = 4
        self.qk_nope_head_dim = 64  # 合并原有两个头维度
        self.qk_rope_head_dim = 0  # 禁用旋转位置编码
        self.v_head_dim = 64
        self.max_seq_len = 1024
        self.original_seq_len = 512
        self.mscale = 1.0
        self.rope_factor = 1.0
        self.max_batch_size = 4


# 简化后的多头注意力层
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_head_dim = args.qk_nope_head_dim  # 仅使用普通注意力头
        self.v_head_dim = args.v_head_dim

        # 查询投影
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # 键值投影
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_head_dim + self.v_head_dim))

        # 输出投影
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # 注意力缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5

        # 缓存初始化
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.qk_head_dim))
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.v_head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # 查询投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)

        # 键值投影
        kv = self.wkv_b(self.kv_norm(self.wkv_a(x)))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim + self.v_head_dim)
        k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)

        # 更新缓存
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v

        # 注意力计算
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # 值聚合
        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        x = self.wo(x.flatten(2))
        return x


# 测试代码
if __name__ == "__main__":
    # 初始化配置
    args = ModelArgs()
    model = MLA(args)

    # 生成测试数据
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, args.dim)

    # 前向传播（移除了位置编码参数）
    output = model(x, start_pos=0, mask=None)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")