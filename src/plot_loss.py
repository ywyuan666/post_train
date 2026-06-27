import matplotlib.pyplot as plt

epochs = [1, 2, 3]
loss_data = {
    "Minimal": [2.345, 2.012, 1.876],
    "Standard": [1.982, 1.654, 1.432],
    "Attention_MLP": [1.876, 1.543, 1.321],
    "All_Linear": [1.953, 1.612, 1.398],
}

plt.figure(figsize=(8, 5))
for label, loss in loss_data.items():
    plt.plot(epochs, loss, marker='o', label=label)

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("LoRA Strategy Comparison (Epoch Avg Loss)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("lora_epoch_loss.png", dpi=150)
plt.show()