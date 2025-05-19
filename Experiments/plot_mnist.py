import matplotlib.pyplot as plt
import pandas as pd

mnist_df = pd.read_csv('mnist_metrics.csv')

# Plot metric curves for the synthetic mnist model
plt.figure(figsize=(10, 6))
plt.plot(mnist_df["Threshold"], mnist_df["SelectiveAccuracy"], label="Selective Accuracy")
plt.plot(mnist_df["Threshold"], mnist_df["CWSA"], label="CWSA")
plt.plot(mnist_df["Threshold"], mnist_df["CWSA+"], label="CWSA+")
plt.xlabel("Threshold (τ)")
plt.ylabel("Metric Value")
plt.title("Metric Behavior of Model Trained on MNIST")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/fig_mnist_curves.png")
plt.show()
