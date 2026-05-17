"""
CLIP Text Encoder for Semantic Alignment
冻结的 CLIP 文本编码器，用于提取目标词的语义特征
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPTextEncoder(nn.Module):
    """
    冻结的 CLIP 文本编码器（使用 transformers 库）
    用于提取文本的语义特征，作为语义对齐的监督信号
    """
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cuda'):
        super().__init__()
        # 加载预训练的 CLIP 模型
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # 移动到指定设备
        self.clip_model = self.clip_model.to(device)

        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 设置为评估模式
        self.clip_model.eval()

        self.device = device

    def forward(self, text_inputs):
        """
        Args:
            text_inputs: 来自 processor 的 tokenized text
        Returns:
            text_features: [B, D] 归一化的文本特征向量
        """
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**text_inputs)
            # L2 归一化
            text_features = outputs / outputs.norm(dim=-1, keepdim=True)

        return text_features

    def encode_text_list(self, text_list):
        """
        便捷方法：直接从文本列表编码
        Args:
            text_list: List[str] 文本列表
        Returns:
            text_features: [B, D] 文本特征
        """
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.forward(inputs)
