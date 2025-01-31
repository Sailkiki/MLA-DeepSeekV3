import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 投影矩阵定义
        self.W_Q = nn.Linear(d_model, d_model)  # 每个头独立的Q投影
        self.W_K = nn.Linear(d_model, self.d_k) # 共享的K投影（关键变化点）
        self.W_V = nn.Linear(d_model, self.d_k) # 共享的V投影
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 生成Q投影 [batch, seq_len, d_model] -> [batch, heads, seq_len, d_k]
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 生成共享的K/V投影 [batch, seq_len, d_k] -> [batch, heads, seq_len, d_k]
        K = self.W_K(K).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        V = self.W_V(V).unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 掩码处理（解码器需要）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 合并多头输出 [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(output)

def main():
    # 配置参数
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    # 生成测试数据
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # 创建MQA模块
    mqa = MultiQueryAttention(d_model, num_heads)

    # 前向传播
    output = mqa(Q, K, V)

    # 验证输出形状
    print(f"输入形状: Q/K/V={Q.shape}")
    print(f"输出形状: {output.shape}")  # 应该保持[batch, seq_len, d_model]

    # 参数数量对比
    total_params = sum(p.numel() for p in mqa.parameters())
    print(f"\n参数数量分析:")
    print(f"- W_Q参数: {mqa.W_Q.weight.shape} ({mqa.W_Q.weight.numel()}个)")
    print(f"- W_K参数: {mqa.W_K.weight.shape} ({mqa.W_K.weight.numel()}个)")
    print(f"- W_V参数: {mqa.W_V.weight.shape} ({mqa.W_V.weight.numel()}个)")
    print(f"总参数: {total_params}")

    # 对比标准多头注意力
    standard_mha = nn.MultiheadAttention(d_model, num_heads)
    mha_params = sum(p.numel() for p in standard_mha.parameters())
    print(f"\n标准多头注意力参数: {mha_params}")
    print(f"参数减少比例: {(1 - total_params/mha_params)*100:.1f}%")

if __name__ == "__main__":
    main()