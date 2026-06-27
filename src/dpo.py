import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,      # 模型对 chosen 回复的 log prob
    policy_rejected_logps,    # 模型对 rejected 回复的 log prob
    ref_chosen_logps,         # 参考模型对 chosen 的 log prob
    ref_rejected_logps,       # 参考模型对 rejected 的 log prob
    beta=0.1,                 # 温度系数
):
    """DPO Loss 手动实现"""
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps
    
    logits = beta * (policy_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss

# 验证单步
if __name__ == "__main__":
    # 模拟数据
    pc = torch.tensor([-2.0, -1.5])   # chosen log prob
    pr = torch.tensor([-3.0, -4.0])   # rejected log prob
    rc = torch.tensor([-2.1, -1.6])   # ref chosen
    rr = torch.tensor([-2.9, -3.8])   # ref rejected
    
    loss = dpo_loss(pc, pr, rc, rr)
    print(f"DPO Loss: {loss.item():.4f}")  # 应输出正数