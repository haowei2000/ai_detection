import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, d1, d2, d3, hidden_dim, output_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads
        )
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, f1, f2, f3):
        # 将所有特征拼接后输入注意力模块
        merged_features = torch.cat((f1, f2, f3), dim=-1).unsqueeze(0)  # 添加batch维度
        attn_output, _ = self.cross_attention(
            merged_features, merged_features, merged_features
        )
        return torch.relu(self.fc1(attn_output.squeeze(0)))
