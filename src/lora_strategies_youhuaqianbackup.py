"""
LoRA 增强实现（支持 all-linear 策略和 Dropout）
基于原 lora.py，增加了对 "all-linear" 标记的支持，并优化了替换逻辑。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    手动实现 LoRA 线性层
    数学公式: h = W0 @ x + (alpha / r) * B @ A @ x
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 冻结原始权重
        self.W0 = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # 下投影 A: (r, in_features)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        # 上投影 B: (out_features, r) —— 零初始化（关键）
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持全零
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始分支
        frozen_out = F.linear(x, self.W0, self.bias)
        # LoRA 分支
        x_drop = self.lora_dropout(x)
        lora_out = (x_drop @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return frozen_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: list = None,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """
    将模型中的指定 Linear 层替换为 LoRALinear
    
    参数:
        target_modules: 包含要替换的层名关键字列表。
                        若包含 "all-linear"，则替换所有 Linear 层（排除嵌入层和输出头）。
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # 默认排除的模块（避免参数量爆炸和输出混乱）
    exclude_keywords = ["lm_head", "embed_tokens", "wte", "wpe"]
    
    apply_all = "all-linear" in target_modules
    replaced_count = 0
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        # 检查是否在排除列表中
        if any(ex in name for ex in exclude_keywords):
            continue
        
        # 决定是否替换
        if apply_all:
            should_replace = True
        else:
            should_replace = any(target in name for target in target_modules)
        
        if not should_replace:
            continue
        
        # 获取父模块
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        if parent_name == "":
            parent = model
        else:
            parent = model.get_submodule(parent_name)
        
        # 创建 LoRA 层并替换
        lora_layer = LoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        setattr(parent, child_name, lora_layer)
        replaced_count += 1
        print(f"✅ 已替换: {name} -> LoRALinear(r={r})")
    
    # 冻结所有非 LoRA 参数
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 共替换 {replaced_count} 个线性层")
    print(f"📊 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """仅提取 LoRA 参数（用于保存 adapter）"""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: dict):
    """加载 LoRA 参数到模型中"""
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
    print("✅ LoRA 权重加载完成")