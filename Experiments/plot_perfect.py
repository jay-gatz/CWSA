import matplotlib.pyplot as plt
import pandas as pd

perfect_df = pd.read_csv('results/synthetic_perfect.csv')

# Plot metric curves for the synthetic Perfect model
plt.figure(figsize=(10, 6))
plt.plot(perfect_df["Threshold"], perfect_df["SelectiveAccuracy"], label="Selective Accuracy")
plt.plot(perfect_df["Threshold"], perfect_df["CWSA"], label="CWSA")
plt.plot(perfect_df["Threshold"], perfect_df["CWSA+"], label="CWSA+")
plt.xlabel("Threshold (τ)")
plt.ylabel("Metric Value")
plt.title("Metric Behavior on Synthetic Perfect Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/fig_perfect_curves.png")
plt.show()
