import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 定义Q、K、V的线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        输入：
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]（可选）
        输出：
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # 步骤1：生成Q、K、V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]

        # 步骤2：计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
        scores = scores / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # 步骤3：应用掩码（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 步骤4：归一化注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]

        # 步骤5：加权求和
        output = torch.matmul(attn_weights, V)  # [batch, seq_len, d_model]

        return output, attn_weights

# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len = 5
    d_model = 64
    # 生成模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    # 创建自注意力模块
    self_attn = SelfAttention(d_model)
    # 前向计算
    output, attn_weights = self_attn(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(output)
    print(attn_weights)