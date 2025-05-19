# simulate_random.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from metrics.cwsa import cwsa, cwsa_plus
from scipy.integrate import trapz
import os

def ece(y_true, y_pred, y_prob, n_bins=10):
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_bounds[i]) & (y_prob < bin_bounds[i + 1])
        if np.any(bin_mask):
            acc = np.mean(y_pred[bin_mask] == y_true[bin_mask])
            conf = np.mean(y_prob[bin_mask])
            ece_val += (np.sum(bin_mask) / len(y_true)) * abs(acc - conf)
    return ece_val

def aurc(y_true, y_pred, y_prob):
    N = len(y_true)
    sorted_idx = np.argsort(-y_prob)
    coverage, risk = [], []
    incorrect = 0
    for i in range(N):
        idx = sorted_idx[i]
        if y_pred[idx] != y_true[idx]:
            incorrect += 1
        coverage.append((i + 1) / N)
        risk.append(incorrect / (i + 1))
    return trapz(risk, coverage)

# --- Random Model Simulation
def simulate_random_model(N=1000):
    y_true = np.random.randint(0, 3, size=N)
    y_pred = np.random.randint(0, 3, size=N)
    y_conf = np.random.uniform(0.3, 1.0, size=N)
    return y_true, y_pred, y_conf

# --- Evaluate
def evaluate(y_true, y_pred, y_conf, thresholds):
    rows = []
    for tau in thresholds:
        acc_mask = y_conf >= tau
        acc = accuracy_score(y_true[acc_mask], y_pred[acc_mask]) if acc_mask.any() else float('nan')
        cwsa_score, _, _ = cwsa(y_true, y_pred, y_conf, tau, return_details=True)
        cwsa_plus_score, _, cov = cwsa_plus(y_true, y_pred, y_conf, tau, return_details=True)
        ece_score = ece(y_true, y_pred, y_conf)
        aurc_score = aurc(y_true, y_pred, y_conf)

        rows.append({
            'Threshold': tau,
            'SelectiveAccuracy': acc,
            'CWSA': cwsa_score,
            'CWSA+': cwsa_plus_score,
            'Coverage': cov,
            'ECE': ece_score,
            'AURC': aurc_score
        })

    return pd.DataFrame(rows)

# --- Main
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    thresholds = np.linspace(0.5, 0.99, 20)

    y_true, y_pred, y_conf = simulate_random_model()
    df = evaluate(y_true, y_pred, y_conf, thresholds)
    df.to_csv("results/synthetic_random.csv", index=False)
    print("Saved: results/synthetic_random.csv")
