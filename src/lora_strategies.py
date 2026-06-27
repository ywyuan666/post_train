"""
LoRA (Low-Rank Adaptation) 手动实现 - 强制 FP32 参数版（支持 lora_dropout）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class LoRALinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,          # 新增 dropuout 参数
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 若 dropout > 0 则使用 Dropout，否则用 Identity
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 强制 LoRA 参数使用 FP32，避免 GradScaler 报错
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.original_linear(x)
        x_drop = self.lora_dropout(x)        # 对输入施加 dropout
        # 自动转换 dtype 以匹配混合精度
        lora_out = (x_drop @ self.lora_A.T.to(x.dtype)) @ self.lora_B.T.to(x.dtype) * self.scaling
        return result + lora_out


def _find_all_linears(model: nn.Module) -> List[str]:
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)
    return linear_names


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,              # 新增参数
    freeze_non_lora: bool = True,
) -> nn.Module:
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    if "all-linear" in target_modules:
        target_modules = _find_all_linears(model)
        print(f"🔍 自动检测到 {len(target_modules)} 个线性层")

    replaced_count = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                lora_linear = LoRALinear(
                    module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,    # 传递 dropout
                )
                setattr(parent, child_name, lora_linear)
                replaced_count += 1
                print(f"✅ 已替换: {name} -> LoRALinear(r={r})")

    if freeze_non_lora:
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    print(f"\n📊 共替换 {replaced_count} 个线性层")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    model.load_state_dict(state_dict, strict=False)