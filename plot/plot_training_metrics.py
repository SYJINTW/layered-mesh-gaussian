import json
import matplotlib.pyplot as plt

# Path to your saved JSON file
json_path = "/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/sample_exp_dynamic/hotdog/distortion_10000_occlusion/training_metrics.json"

# Load the data
with open(json_path, "r") as f:
    metrics_data = json.load(f)

# Extract the lists
history_iters = metrics_data["iteration"]
history_loss = metrics_data["loss"]
history_l1 = metrics_data["l1"]
history_ssim = metrics_data["ssim"]

# Now you can plot them exactly as before
plt.figure(figsize=(18, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(history_iters, history_loss, label='Loss', color='red', alpha=0.8)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.ylim(bottom=0.0, top=0.1) # Adjust as needed
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot L1
plt.subplot(1, 3, 2)
plt.plot(history_iters, history_l1, label='L1', color='blue', alpha=0.8)
plt.xlabel("Iteration")
plt.ylabel("L1")
plt.title("Training L1")
plt.ylim(bottom=0.0, top=0.1) # Adjust as needed
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot SSIM
plt.subplot(1, 3, 3)
plt.plot(history_iters, history_ssim, label='SSIM', color='green', alpha=0.8)
plt.xlabel("Iteration")
plt.ylabel("SSIM")
plt.title("Training SSIM")
plt.ylim(bottom=0.9, top=1.0) # Adjust as needed
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

# 儲存圖表到 model_path 底下
plot_path = f"/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/sample_exp_dynamic/hotdog/distortion_10000_occlusion/training_metrics_plot.png"
plt.savefig(plot_path, dpi=300)
plt.close()