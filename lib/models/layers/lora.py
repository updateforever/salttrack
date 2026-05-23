import math
from typing import Iterable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """标准 LoRA 线性层包装器。

    这个类保留原始线性层 W 的前向能力，只在旁路上学习一个低秩残差：

        y = W x + (alpha / rank) * B A x

    其中 W 被冻结，A/B 是唯一新增的可训练参数。我们保留这个标准版本，
    一方面作为最直接的 LoRA baseline，另一方面也方便和 routed LoRA 做
    等参数量消融对比。
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始 nn.Linear 作为 frozen base path。后续优化器会只打开 LoRA 参数，
        # 这里也直接冻结 base_layer，避免全量微调悄悄混进 PEFT 实验。
        self.base_layer = base_layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A: down projection, B: up projection。B 零初始化让训练开始时
        # LoRA 分支输出为 0，模型初始行为等价于原始 checkpoint。
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # 记录最近一次 LoRA residual，用于 actor 里的 semantic-guided
        # residual energy regularization。
        self.last_lora_out = None
        self.reset_parameters()

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling

        # 缓存 residual，而不是缓存 base_out。语义正则只约束低秩适配分支，
        # 不应该惩罚 frozen base path 的表示能力。
        self.last_lora_out = lora_out
        return base_out + lora_out

    def get_residual_energy(self):
        """Return per-sample LoRA residual energy for training-time regularization."""
        if self.last_lora_out is None:
            return None
        if self.last_lora_out.dim() == 0:
            return None
        batch_size = self.last_lora_out.shape[0]
        return self.last_lora_out.float().reshape(batch_size, -1).pow(2).mean(dim=1)


class RoutedLoRALinear(nn.Module):
    """MoE-style 双专家 LoRA 线性层。

    这是本文 LoRA 结构创新的核心模块。相比标准 LoRA 只有一条低秩残差，
    这里把低秩适配拆成两个 expert：

        Visual Expert:   B_v A_v x，偏向视觉外观、定位和短时序变化。
        Semantic Expert: B_s A_s x，偏向文本语义和目标身份一致性。

    Router 根据当前输入 token/sample 动态输出两个 expert 的混合权重：

        y = W x + r_v * B_v A_v x + r_s * B_s A_s x

    这里不是让文本直接生成 LoRA 参数，而是先用 tracking feature 自身做
    token-level 路由；训练时再用语义可靠性 gate 监督 r_s，让 semantic expert
    在文本-实例对齐可靠时承担更多适配，在不可靠时自动退回 visual expert。
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"RoutedLoRALinear expects nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base path，提供预训练 tracker 的稳定能力。
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base_layer = base_layer

        # 两个低秩 expert 的参数规模完全一致。若 standard LoRA 使用 rank=8，
        # routed LoRA 可设置 EXPERT_RANK=4，使两个 expert 的 A/B 参数量
        # 大致与标准 LoRA 对齐，便于做公平消融。
        self.lora_A_visual = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B_visual = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_A_semantic = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B_semantic = nn.Parameter(torch.zeros(self.out_features, rank))

        # 轻量 router 输出 [visual, semantic] 两个 logits。router 只看当前
        # 输入特征 x，不直接依赖 GT；因此训练后推理阶段仍然可以使用。
        self.lora_router = nn.Linear(self.in_features, 2)

        # last_lora_out 用于保留原来的 residual energy 正则；
        # last_router_weights 用于 actor 里对 semantic route 概率 r_s 做监督。
        self.last_lora_out = None
        self.last_router_weights = None
        self.reset_parameters()

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def reset_parameters(self):
        # A 使用 Kaiming 初始化，B 零初始化。这样两个 expert 初始 residual 都为 0，
        # router 初始即使给 0.5/0.5，也不会改变原模型输出。
        nn.init.kaiming_uniform_(self.lora_A_visual, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_visual)
        nn.init.kaiming_uniform_(self.lora_A_semantic, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_semantic)

        # router 零初始化对应 softmax 后均匀路由，避免一开始偏向某个 expert。
        # 后续通过 L_router 和主 tracking loss 学出语义可靠性相关的路由策略。
        nn.init.zeros_(self.lora_router.weight)
        nn.init.zeros_(self.lora_router.bias)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base_out = self.base_layer(x)
        x_drop = self.dropout(x)

        # Visual expert: 一条标准 LoRA residual，学习更偏视觉/定位的适配。
        visual_out = F.linear(x_drop, self.lora_A_visual)
        visual_out = F.linear(visual_out, self.lora_B_visual) * self.scaling

        # Semantic expert: 另一条标准 LoRA residual，后续通过 router supervision
        # 鼓励它在语义可靠样本上获得更高权重。
        semantic_out = F.linear(x_drop, self.lora_A_semantic)
        semantic_out = F.linear(semantic_out, self.lora_B_semantic) * self.scaling

        # router_weights[..., 0] = r_v, router_weights[..., 1] = r_s。
        # 对任意输入形状 [..., C] 都在最后一维做 expert softmax。
        router_weights = torch.softmax(self.lora_router(x_drop), dim=-1)

        # expert_out shape: [..., out_features, 2]，最后一维是两个 expert。
        # router_weights.unsqueeze(-2) 让路由权重广播到输出通道维。
        expert_out = torch.stack([visual_out, semantic_out], dim=-1)
        lora_out = (expert_out * router_weights.unsqueeze(-2)).sum(dim=-1)

        self.last_lora_out = lora_out
        self.last_router_weights = router_weights
        return base_out + lora_out

    def get_residual_energy(self):
        if self.last_lora_out is None or self.last_lora_out.dim() == 0:
            return None
        batch_size = self.last_lora_out.shape[0]
        return self.last_lora_out.float().reshape(batch_size, -1).pow(2).mean(dim=1)

    def get_semantic_route_weight(self):
        """Return per-sample semantic-expert route probability r_s.

        不同模块的 LoRA 输入可能是 [B, C]、[B, N, C] 或更高维 token
        表示。为了和样本级 semantic_gate_per_sample 对齐，这里把除 batch
        维外的 token/空间维平均，得到每个样本一个 semantic route 概率。
        """
        if self.last_router_weights is None or self.last_router_weights.dim() == 0:
            return None
        semantic_weight = self.last_router_weights[..., 1]
        if semantic_weight.dim() == 1:
            return semantic_weight.float()
        batch_size = semantic_weight.shape[0]
        return semantic_weight.float().reshape(batch_size, -1).mean(dim=1)


def _replace_linear_layers(module: nn.Module, rank: int, alpha: int, dropout: float,
                           prefix: str = "", lora_type: str = "standard") -> List[str]:
    """Recursively replace nn.Linear children with the selected LoRA wrapper."""
    replaced = []
    for child_name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            if lora_type == "routed":
                setattr(module, child_name, RoutedLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            else:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(child_prefix)
            continue
        replaced.extend(_replace_linear_layers(child, rank, alpha, dropout, child_prefix, lora_type=lora_type))
    return replaced


def apply_lora_to_modules(model: nn.Module,
                          target_module_names: Iterable[str],
                          rank: int = 8,
                          alpha: int = 16,
                          dropout: float = 0.0,
                          lora_type: str = "standard") -> List[str]:
    """把指定模块内部的 nn.Linear 替换为 LoRA 版本。

    target_module_names 控制 LoRA 的注入范围，例如 backbone、vl_fusion、
    visual_temporal_fusion、confidence_pred。这样我们可以避免直接扰动
    text_encoder/text_adj 这类语义锚点，同时也能通过 YAML 做模块级消融。

    lora_type:
        standard: 单分支 LoRA baseline。
        routed:   双专家 MoE-style routed LoRA。
    """

    replaced = []
    lora_type = lora_type.lower()
    if lora_type not in ["standard", "routed"]:
        raise ValueError(f"Unsupported LoRA type: {lora_type}")
    for module_name in target_module_names:
        target_module = dict(model.named_modules()).get(module_name, None)
        if target_module is None:
            continue
        replaced.extend(_replace_linear_layers(target_module, rank, alpha, dropout,
                                               prefix=module_name, lora_type=lora_type))
    return replaced


def collect_lora_residual_energies(model: nn.Module) -> List[torch.Tensor]:
    """Collect per-sample residual energies from all active LoRA layers.

    标准 LoRA 和 routed LoRA 都会缓存 last_lora_out，因此原来的
    semantic-guided residual energy regularization 可以继续复用。
    """
    energies = []
    for module in model.modules():
        if isinstance(module, (LoRALinear, RoutedLoRALinear)):
            energy = module.get_residual_energy()
            if energy is not None:
                energies.append(energy)
    return energies


def collect_lora_router_weights(model: nn.Module) -> List[torch.Tensor]:
    """Collect semantic-expert route probabilities from routed LoRA layers.

    actor 会把这些 r_s 与语义可靠性 gate 对齐，形成 L_router。
    对 standard LoRA 来说没有 router，因此这个函数只收集 RoutedLoRALinear。
    """
    route_weights = []
    for module in model.modules():
        if isinstance(module, RoutedLoRALinear):
            weight = module.get_semantic_route_weight()
            if weight is not None:
                route_weights.append(weight)
    return route_weights


def collect_lora_router_weights_named(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """Collect named semantic-expert route probabilities from routed LoRA layers.

    This is used only by visualization / analysis scripts. It keeps the training
    actor's lighter unnamed collection path unchanged.
    """
    route_weights = []
    for name, module in model.named_modules():
        if isinstance(module, RoutedLoRALinear):
            weight = module.get_semantic_route_weight()
            if weight is not None:
                route_weights.append((name, weight))
    return route_weights
