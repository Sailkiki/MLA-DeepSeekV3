import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model      # 输入维度（如512）
        self.num_heads = num_heads  # 头数（如8）
        self.d_k = d_model // num_heads  # 每个头的维度（如64）

        # 定义线性投影层
        self.W_q = nn.Linear(d_model, d_model)  # Q投影
        self.W_k = nn.Linear(d_model, d_model)  # K投影
        self.W_v = nn.Linear(d_model, d_model)  # V投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

    def forward(self, x, mask=None):
        """
        输入：
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]（可选）
        输出：
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # === 步骤1：线性投影并分头 ===
        # 投影到Q/K/V [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分头操作（关键维度变换）
        # [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # === 步骤2：计算缩放点积注意力 ===
        # 计算注意力得分 [batch, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # 应用掩码（如需要）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 归一化注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和 [batch, num_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, V)

        # === 步骤3：拼接多头输出 ===
        # 合并头和序列维度 [batch, seq_len, num_heads * d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 最终线性变换 [batch, seq_len, d_model]
        output = self.W_o(attn_output)

        return output, attn_weights

# 使用示例
if __name__ == "__main__":
    # 参数设置
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 32

    # 创建多头注意力模块
    mha = MultiHeadAttention(d_model, num_heads)

    # 生成输入数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output, attn_weights = mha(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")  # [batch, num_heads, seq_len, seq_len]