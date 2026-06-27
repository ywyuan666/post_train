"""
LoRA (Low-Rank Adaptation) 手动实现
严格按照课件中的数学公式：h = W0·x + (α/r)·B·A·x
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
        
        # 获取原始线性层的参数
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 冻结原始权重（不计算梯度）
        self.W0 = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        
        # LoRA 参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # α/r 缩放因子
        
        # 下投影矩阵 A: (r, in_features) - 使用 Kaiming 初始化
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        # 上投影矩阵 B: (out_features, r) - 初始化为零（关键！）
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # Dropout（可选）
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化 A
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """初始化 A 为 Kaiming 均匀分布，B 保持为零"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 已经是零，不需要重新初始化
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：h = W0·x + (α/r)·B·A·x
        x: (..., in_features)
        """
        # 原始分支（冻结）
        frozen_out = F.linear(x, self.W0, self.bias)
        
        # LoRA 分支（可训练）
        x_drop = self.lora_dropout(x)
        # 将 LoRA 参数转换为与输入 x 相同的数据类型，避免 half vs float 错误
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_out = (x_drop @ lora_A.T) @ lora_B.T  # 先 A 后 B
        lora_out = lora_out * self.scaling
        
        return frozen_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: list = None,
) -> nn.Module:
    """
    将模型中的指定 Linear 层替换为 LoRALinear
    
    目标模块默认：q_proj, v_proj（课件推荐的最小集）
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # 递归遍历模型的所有子模块
    for name, module in model.named_modules():
        # 检查是否需要替换
        should_replace = any(target in name for target in target_modules)
        if should_replace and isinstance(module, nn.Linear):
            # 获取父模块和当前模块名
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name == "":
                parent = model
            else:
                parent = model.get_submodule(parent_name)
            
            # 创建 LoRA 层并替换
            lora_layer = LoRALinear(module, r=r, lora_alpha=lora_alpha)
            setattr(parent, child_name, lora_layer)
            
            print(f"✅ 已替换: {name} -> LoRALinear(r={r}, alpha={lora_alpha})")
    
    # 冻结所有非 LoRA 参数
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False
    
    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """仅提取 LoRA 参数（用于保存 adapter）"""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: dict):
    """加载 LoRA 参数"""
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
    print("✅ LoRA 权重加载完成")