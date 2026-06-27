import re
import matplotlib.pyplot as plt

def extract_epoch_losses(log_file):
    train_losses = []
    val_losses = []
    with open(log_file, "r") as f:
        for line in f:
            train_match = re.search(r"Epoch\s+\d+\s+训练平均 Loss:\s*([\d.]+)", line)
            if train_match:
                train_losses.append(float(train_match.group(1)))
            val_match = re.search(r"Epoch\s+\d+\s+验证平均 Loss:\s*([\d.]+)", line)
            if val_match:
                val_losses.append(float(val_match.group(1)))
    return train_losses, val_losses

# 提取 500 样本日志
train_500, val_500 = extract_epoch_losses("train_standard.log")
# 提取全量日志（如果文件不存在则返回空列表）
train_full, val_full = extract_epoch_losses("train_standard_full.log")

# 检查是否成功读取 500 样本数据
if len(train_500) == 0:
    print("❌ 未在 train_standard.log 中找到 '训练平均 Loss'，请检查日志内容。")
    exit(1)

# 绘制对比图
plt.figure(figsize=(10, 6))

epochs_500 = range(1, len(train_500) + 1)
plt.plot(epochs_500, train_500, marker='o', label='Train Loss (500)')
if val_500:
    plt.plot(epochs_500, val_500, marker='o', linestyle='--', label='Val Loss (500)')

if train_full:
    epochs_full = range(1, len(train_full) + 1)
    plt.plot(epochs_full, train_full, marker='s', label='Train Loss (52k)')
    if val_full:
        plt.plot(epochs_full, val_full, marker='s', linestyle='--', label='Val Loss (52k)')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Standard LoRA: 500 vs 52k (Training & Validation)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_data_comparison.png", dpi=150)
print("✅ 图片已保存为 loss_data_comparison.png")