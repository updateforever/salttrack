"""
Semantic Projector for SALTTrack-SemanticAlign
将实例特征投影到 CLIP 语义空间
"""
import torch
import torch.nn as nn


class SemanticProjector(nn.Module):
    """
    将视觉实例特征投影到 CLIP 语义空间
    """
    def __init__(self, input_dim, output_dim=512, hidden_dim=None):
        """
        Args:
            input_dim: 输入特征维度（来自 RoIAlign 的特征）
            output_dim: 输出维度（CLIP 文本特征维度，ViT-B/32 是 512）
            hidden_dim: 隐藏层维度，默认为 input_dim
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        """
        Args:
            features: [B, input_dim] 实例特征
        Returns:
            projected: [B, output_dim] 投影后的特征
        """
        projected = self.projector(features)
        # L2 归一化，与 CLIP 特征对齐
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
        return projected
