import matplotlib.pyplot as plt
import pandas as pd

cifar_df = pd.read_csv('cifar_10_metrics.csv')

# Plot metric curves for the synthetic cifar model
plt.figure(figsize=(10, 6))
plt.plot(cifar_df["Threshold"], cifar_df["SelectiveAccuracy"], label="Selective Accuracy")
plt.plot(cifar_df["Threshold"], cifar_df["CWSA"], label="CWSA")
plt.plot(cifar_df["Threshold"], cifar_df["CWSA+"], label="CWSA+")
plt.xlabel("Threshold (τ)")
plt.ylabel("Metric Value")
plt.title("Metric Behavior of Model Trained on Cifar 10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/fig_cifar_curves.png")
plt.show()
