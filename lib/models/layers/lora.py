import math
from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """WYP: 线性层 LoRA 包装器，保留原权重并叠加低秩增量。"""

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

        self.base_layer = base_layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
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


def _replace_linear_layers(module: nn.Module, rank: int, alpha: int, dropout: float, prefix: str = "") -> List[str]:
    replaced = []
    for child_name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(child_prefix)
            continue
        replaced.extend(_replace_linear_layers(child, rank, alpha, dropout, child_prefix))
    return replaced


def apply_lora_to_modules(model: nn.Module,
                          target_module_names: Iterable[str],
                          rank: int = 8,
                          alpha: int = 16,
                          dropout: float = 0.0) -> List[str]:
    """WYP: 仅对指定模块内部的 nn.Linear 做 LoRA 替换，避免改动整网语义锚点。"""

    replaced = []
    for module_name in target_module_names:
        target_module = dict(model.named_modules()).get(module_name, None)
        if target_module is None:
            continue
        replaced.extend(_replace_linear_layers(target_module, rank, alpha, dropout, prefix=module_name))
    return replaced


def collect_lora_residual_energies(model: nn.Module) -> List[torch.Tensor]:
    """Collect per-sample residual energies from all active LoRA layers."""
    energies = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            energy = module.get_residual_energy()
            if energy is not None:
                energies.append(energy)
    return energies
