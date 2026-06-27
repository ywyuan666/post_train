"""
DPO (Direct Preference Optimization) 验证 —— 使用蒸馏后的学生模型作为参考模型
策略模型：标准 LoRA 教师模型
参考模型：SFT 蒸馏得到的学生模型
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    DPO 损失函数（手动实现）
    """
    policy_log_ratio = policy_chosen_logp - policy_rejected_logp
    ref_log_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss


def compute_response_log_prob(model, tokenizer, prompt: str, response: str, device) -> float:
    """
    计算模型在给定 prompt 下生成 response 的对数概率（总和）
    """
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokens)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, prompt_len-1:-1, :]  # (response_len, vocab)
        target_ids = inputs["input_ids"][0, prompt_len:]  # response 的真实 token
    
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(len(target_ids)), target_ids]
    return token_log_probs.sum().item()


def main():
    # ========== 配置 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"
    teacher_preset = "standard"                     # 教师模型使用的 LoRA 策略
    lora_path = "./output_standard/lora_standard_final.pt"
    student_model_path = "./distill_sft/student_final.pt"   # 蒸馏后的学生模型
    beta = 0.1                                      # DPO 温度
    
    print("🚀 设备:", device)
    print("📦 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ========== 1. 加载策略模型 (policy = 微调后的 LoRA 教师) ==========
    print("👨‍🏫 加载策略模型 (教师 LoRA)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    target_modules = get_target_modules_from_preset(teacher_preset)
    policy_model = apply_lora_to_model(policy_model, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load(lora_path, map_location=device)
    load_lora_state_dict(policy_model, lora_state)
    policy_model = policy_model.to(device)   # 确保 LoRA 参数也在 GPU
    policy_model.eval()
    print("✅ 策略模型加载完成")
    
    # ========== 2. 加载参考模型 (reference = 蒸馏学生模型) ==========
    print("📚 加载参考模型 (蒸馏学生模型)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    student_state = torch.load(student_model_path, map_location=device)
    ref_model.load_state_dict(student_state)
    ref_model = ref_model.to(device)
    ref_model.eval()
    print("✅ 参考模型加载完成")
    
    # ========== 3. 构造偏好对（示例） ==========
    prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
    chosen_response = "The capital of France is Paris."
    rejected_response = "France is a country in Europe. It has many cities."
    
    print("\n📝 Prompt:")
    print(prompt)
    print(f"✅ Chosen response:   {chosen_response}")
    print(f"❌ Rejected response: {rejected_response}")
    
    # ========== 4. 计算 log prob ==========
    print("\n🧮 计算对数概率...")
    pc = compute_response_log_prob(policy_model, tokenizer, prompt, chosen_response, device)
    pr = compute_response_log_prob(policy_model, tokenizer, prompt, rejected_response, device)
    rc = compute_response_log_prob(ref_model, tokenizer, prompt, chosen_response, device)
    rr = compute_response_log_prob(ref_model, tokenizer, prompt, rejected_response, device)
    
    print(f"Policy Chosen logp:   {pc:.4f}")
    print(f"Policy Rejected logp: {pr:.4f}")
    print(f"Ref (Student) Chosen logp:   {rc:.4f}")
    print(f"Ref (Student) Rejected logp: {rr:.4f}")
    
    # ========== 5. 计算 DPO Loss (正常顺序) ==========
    pc_t = torch.tensor([pc])
    pr_t = torch.tensor([pr])
    rc_t = torch.tensor([rc])
    rr_t = torch.tensor([rr])
    
    loss_normal = dpo_loss(pc_t, pr_t, rc_t, rr_t, beta=beta)
    print(f"\n📉 正常 DPO Loss: {loss_normal.item():.4f}")
    
    # ========== 6. 交换验证 ==========
    print("\n🔄 交换 chosen/rejected 验证...")
    loss_swapped = dpo_loss(pr_t, pc_t, rr_t, rc_t, beta=beta)
    print(f"🔄 交换后 DPO Loss: {loss_swapped.item():.4f}")
    
    if loss_normal.item() != loss_swapped.item():
        print("✅ 验证通过：交换后 Loss 发生变化，DPO Loss 正确响应偏好顺序。")
    else:
        print("❌ 验证失败：Loss 未变化，请检查数据或增大 chosen/rejected 差异。")
    
    # ========== 7. 额外验证 ==========
    policy_diff = pc - pr
    ref_diff = rc - rr
    print(f"\n📊 Policy chosen - rejected: {policy_diff:.4f}")
    print(f"📊 Ref (Student) chosen - rejected: {ref_diff:.4f}")
    if policy_diff > 0:
        print("✅ 策略模型确实更偏好 chosen 回复。")
    else:
        print("⚠️ 策略模型未偏好 chosen，可能微调不充分或数据不合适。")


if __name__ == "__main__":
    main()