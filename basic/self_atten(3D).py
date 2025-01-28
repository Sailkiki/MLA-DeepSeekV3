import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        # 点特征转换层
        self.W_q = nn.Linear(feat_dim, feat_dim)  # 查询向量变换
        self.W_k = nn.Linear(feat_dim, feat_dim)  # 键向量变换
        self.W_v = nn.Linear(feat_dim, feat_dim)  # 值向量变换

    def forward(self, points, mask=None):
        """
        输入：
            points: [B, N, C] 点云数据
                    B - batch大小
                    N - 点数量（如1024）
                    C - 特征维度（坐标+特征，如x,y,z + features）
            mask: [B, N, N] 可选注意力掩码
        输出：
            attended_points: [B, N, C] 增强后的点特征
            attn_weights: [B, N, N] 注意力权重矩阵
        """
        B, N, _ = points.shape

        # 生成查询、键、值
        Q = self.W_q(points)  # [B, N, C]
        K = self.W_k(points)  # [B, N, C]
        V = self.W_v(points)  # [B, N, C]

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, N, N]
        attn_scores = attn_scores / (self.feat_dim ** 0.5)

        # 可选掩码（处理动态点云时使用）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 特征聚合
        attended_points = torch.matmul(attn_weights, V)  # [B, N, C]

        return attended_points, attn_weights


# 使用示例
if __name__ == "__main__":
    # 参数设置（模拟PointNet++输入）
    batch_size = 4
    num_points = 1024  # 点云数量
    feat_dim = 6  # xyz坐标 + RGB颜色
    # 生成模拟点云（坐标+颜色）
    points = torch.randn(batch_size, num_points, feat_dim)
    # 创建注意力层
    pc_attn = PointCloudAttention(feat_dim)
    # 前向传播
    enhanced_points, attn = pc_attn(points)

    print("输入形状:", points.shape)  # [4, 1024, 6]
    print("增强特征形状:", enhanced_points.shape)  # [4, 1024, 6]
    print("注意力矩阵形状:", attn.shape)  # [4, 1024, 1024]
    print(enhanced_points)