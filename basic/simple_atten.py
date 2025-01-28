import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k  # Key的维度

    def forward(self, Q, K, V, mask=None):
        """
        参数说明：
        Q : [batch_size, seq_len_q, d_k]
        K : [batch_size, seq_len_k, d_k]
        V : [batch_size, seq_len_k, d_v]
        mask : [batch_size, seq_len_q, seq_len_k]
        """
        # 1. 计算相似度得分
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, q_len, k_len]
        # 2. 缩放处理
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # 3. 掩码处理（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 4. Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)  # [batch, q_len, k_len]

        # 5. 加权求和
        output = torch.matmul(attn_weights, V)  # [batch, q_len, d_v]

        return output, attn_weights

# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len_q = 3  # 查询序列长度
    seq_len_k = 5  # 键值序列长度
    d_k = 64  # Key维度
    d_v = 128  # Value维度
    # 生成测试数据
    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_k, d_k)
    V = torch.randn(batch_size, seq_len_k, d_v)
    # 创建注意力模块
    attention = SimpleAttention(d_k=d_k)
    # 前向计算
    output, attn_weights = attention(Q, K, V)
    # 打印结果
    print("输入Query形状:", Q.shape)
    print("输入Key形状:", K.shape)
    print("输入Value形状:", V.shape)
    print("\n输出形状:", output.shape)
    print("注意力权重形状:", attn_weights.shape)
    print("\n示例注意力权重矩阵:")
    print(attn_weights[0].detach().numpy().round(2))  # 取第一个样本，保留两位小数