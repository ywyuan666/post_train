import re
import matplotlib.pyplot as plt

presets = ["minimal", "standard", "attention_mlp", "all_linear"]
epoch_losses = {p: [] for p in presets}

for p in presets:
    log_file = f"train_{p}_full.log"
    with open(log_file, "r") as f:
        for line in f:
            # 匹配 "Epoch X 平均 Loss: Y.YYYY"
            match = re.search(r"Epoch (\d+) 平均 Loss: ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epoch_losses[p].append((epoch, loss))

# 绘图
plt.figure(figsize=(10, 6))
for p in presets:
    if epoch_losses[p]:
        epochs, losses = zip(*sorted(epoch_losses[p]))
        plt.plot(epochs, losses, marker='o', label=p)

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("LoRA Strategy Comparison (Full Alpaca Dataset)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("lora_full_comparison.png", dpi=150)
print("图片已保存为 lora_full_comparison.png")