import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, groups=2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        assert num_heads % groups == 0, "num_heads必须能被groups整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.groups = groups
        self.heads_per_group = num_heads // groups
        self.d_k = d_model // num_heads

        # 投影矩阵定义
        self.W_Q = nn.Linear(d_model, d_model)  # 每个头独立的Q投影
        self.W_K = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(groups)])  # 每组共享K
        self.W_V = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(groups)])  # 每组共享V
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 生成Q投影 [B, L, H, d_k]
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 分组处理K/V
        K = torch.stack([proj(x) for proj in self.W_K], dim=1)  # [B, G, L, d_k]
        V = torch.stack([proj(x) for proj in self.W_V], dim=1)  # [B, G, L, d_k]

        # 将头分配到组
        Q_groups = Q.view(batch_size, self.groups, self.heads_per_group, seq_len, self.d_k)
        K_groups = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V_groups = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)

        # 计算注意力 [B, G, H/G, L, L]
        scores = torch.matmul(Q_groups, K_groups.transpose(-1, -2)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V_groups)  # [B, G, H/G, L, d_k]

        # 合并输出
        context = context.view(batch_size, self.num_heads, seq_len, self.d_k)
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(output)

def main():
    # 配置参数
    d_model = 768
    num_heads = 12
    groups = 3
    batch_size = 2
    seq_len = 32

    # 生成测试数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建GQA模块
    gqa = GroupedQueryAttention(d_model, num_heads, groups)

    # 前向传播
    output = gqa(x)

    # 验证输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应保持[batch, seq_len, d_model]

    # 参数数量分析
    total_params = sum(p.numel() for p in gqa.parameters())
    print("\n参数数量分析:")
    print(f"- W_Q参数: {gqa.W_Q.weight.shape} ({gqa.W_Q.weight.numel()/1e6:.2f}M)")
    for i, (k, v) in enumerate(zip(gqa.W_K, gqa.W_V)):
        print(f"- 第{i+1}组 K参数: {k.weight.shape} ({k.weight.numel()/1e3:.1f}K)")
        print(f"- 第{i+1}组 V参数: {v.weight.shape} ({v.weight.numel()/1e3:.1f}K)")
    print(f"总参数: {total_params/1e6:.2f}M")
    print(torch.cuda.is_available())

    # 性能对比测试
    def speed_test(model, name):
        model.eval()
        with torch.no_grad():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(x.cuda())
            ender.record()
            torch.cuda.synchronize()
            print(f"{name}推理时间: {starter.elapsed_time(ender):.2f}ms")

    if torch.cuda.is_available():
        gqa.cuda()
        x = x.cuda()
        speed_test(gqa, "GQA")
        # 对比标准MHA
        mha = nn.MultiheadAttention(d_model, num_heads).cuda()
        speed_test(mha, "MHA")

if __name__ == "__main__":
    main()