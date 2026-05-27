import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV
df = pd.read_csv("checkpoints_yuv/loss_log.csv")

# 畫圖
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints_yuv/loss_plot.png")