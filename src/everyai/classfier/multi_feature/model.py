import torch
import torch.nn as nn


class CrossAttentionFeatureFusion(nn.Module):
    def __init__(
        self,
        proj_dim,
        output_dim,
        num_heads,
        dropout=0.1,
    ):
        """
        基于交叉注意力的特征融合模块
        :param feature_dim: 每种特征的维度
        :param num_heads: 多头注意力机制的头数
        :param dropout: Dropout 概率
        """
        super(CrossAttentionFeatureFusion, self).__init__()

        self.projection1 = nn.Linear(f1.size(-1), proj_dim)
        self.projection2 = nn.Linear(f2.size(-1), proj_dim)
        self.projection3 = nn.Linear(f3.size(-1), proj_dim)
        # 多头注意力模块
        self.cross_attention_12 = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attention_13 = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attention_23 = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=dropout
        )
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(proj_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, f1, f2, f3):
        """
        前向传播
        :param f1: 特征向量 1，形状为 (seq_len, batch_size, feature_dim)
        :param f2: 特征向量 2，形状为 (seq_len, batch_size, feature_dim)
        :param f3: 特征向量 3，形状为 (seq_len, batch_size, feature_dim)
        :return: 融合后的特征，形状为 (seq_len, batch_size, feature_dim)
        """
        # 特征1与特征2交互
        f1_projected = self.projection1(f1)
        f2_projected = self.projection2(f2)
        f3_projected = self.projection3(f3)
        attn_12, _ = self.cross_attention_12(
            query=f1_projected, key=f2_projected, value=f2_projected
        )

        # 特征1与特征3交互
        attn_13, _ = self.cross_attention_13(
            query=f1_projected, key=f3_projected, value=f3_projected
        )

        # 特征2与特征3交互
        attn_23, _ = self.cross_attention_23(
            query=f2_projected, key=f3_projected, value=f3_projected
        )

        # 拼接所有交叉注意力输出
        fused_features = torch.cat((attn_12, attn_13, attn_23), dim=-1)
        return self.fc(fused_features)


# 假设有三种特征，维度均为 128，序列长度为 10，batch_size 为 32
feature1_dim, feature2_dim, feature3_dim = 128, 64, 32
num_heads = 4
seq_len = 10
batch_size = 32
output_dim = 128
proj_dim = 64
# 初始化特征
f1 = torch.randn(seq_len, batch_size, feature1_dim)  # 特征1
f2 = torch.randn(seq_len, batch_size, feature2_dim)  # 特征2
f3 = torch.randn(seq_len, batch_size, feature3_dim)  # 特征3
print("特征1形状：", f1.shape)
print("特征2形状：", f2.shape)
print("特征3形状：", f3.shape)
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将特征和模型移动到GPU
f1 = f1.to(device)
f2 = f2.to(device)
f3 = f3.to(device)
fusion_module = CrossAttentionFeatureFusion(
    proj_dim=proj_dim,
    output_dim=output_dim,
    num_heads=num_heads,
).to(device)

# 前向传播
fused_features = fusion_module(f1, f2, f3)

print("融合后的特征形状：", fused_features.shape)


class FeatureFusionClassfier(nn.modules):
    def __init__(self, input_dim, proj_dim, classifier):
        super(FeatureFusionClassfier, self).__init__()
        self.fusion_module = CrossAttentionFeatureFusion(
            proj_dim=proj_dim, output_dim=input_dim
        )
        self.classifier = classifier

    def forward(self, f1, f2, f3):
        fused_features = self.fusion_module(f1, f2, f3)
        return self.classifier(fused_features)
