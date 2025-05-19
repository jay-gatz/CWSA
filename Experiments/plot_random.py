import matplotlib.pyplot as plt
import pandas as pd

random_df = pd.read_csv('results/synthetic_random.csv')

# Plot metric curves for the synthetic random model
plt.figure(figsize=(10, 6))
plt.plot(random_df["Threshold"], random_df["SelectiveAccuracy"], label="Selective Accuracy")
plt.plot(random_df["Threshold"], random_df["CWSA"], label="CWSA")
plt.plot(random_df["Threshold"], random_df["CWSA+"], label="CWSA+")
plt.xlabel("Threshold (τ)")
plt.ylabel("Metric Value")
plt.title("Metric Behavior on Synthetic Random Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/fig_random_curves.png")
plt.show()
