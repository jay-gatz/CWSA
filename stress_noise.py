import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.integrate import trapz
from metrics.cwsa import cwsa, cwsa_plus

# --- Other metric helpers ---
def selective_accuracy(y_true, y_pred, y_prob, tau=0.9):
    mask = y_prob >= tau
    return accuracy_score(y_true[mask], y_pred[mask]) if np.any(mask) else np.nan

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

def ece(y_true, y_pred, y_prob, n_bins=10):
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_bounds[i]) & (y_prob < bin_bounds[i + 1])
        if np.any(bin_mask):
            acc = np.mean(y_pred[bin_mask] == y_true[bin_mask])
            conf = np.mean(y_prob[bin_mask])
            ece += (np.sum(bin_mask) / len(y_true)) * abs(acc - conf)
    return ece

# --- Add label noise to ground truth ---
def add_label_noise(y, noise_rate, num_classes=3):
    y_noisy = y.copy()
    n_noisy = int(noise_rate * len(y))
    idxs = np.random.choice(len(y), n_noisy, replace=False)
    for i in idxs:
        y_noisy[i] = np.random.choice([c for c in range(num_classes) if c != y[i]])
    return y_noisy

# --- Simulate base model ---
def simulate_calibrated_model(N=1000):
    y_true = np.random.randint(0, 3, size=N)
    y_pred = []
    y_prob = []
    for i in range(N):
        correct = np.random.rand() < 0.9
        label = y_true[i] if correct else np.random.randint(0, 3)
        conf = np.random.uniform(0.8, 1.0) if label == y_true[i] else np.random.uniform(0.5, 0.7)
        y_pred.append(label)
        y_prob.append(conf)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# --- Run evaluation across noise levels ---
def evaluate_noise_impact(noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4], tau=0.9):
    records = []
    base_y_true, base_y_pred, base_y_prob = simulate_calibrated_model()

    for noise in noise_levels:
        y_noisy = add_label_noise(base_y_true, noise_rate=noise)
        acc = accuracy_score(y_noisy, base_y_pred)
        sacc = selective_accuracy(y_noisy, base_y_pred, base_y_prob, tau)
        cwsa_score, _, _ = cwsa(y_noisy, base_y_pred, base_y_prob, tau, return_details=True)
        cwsa_plus_score, _, _ = cwsa_plus(y_noisy, base_y_pred, base_y_prob, tau, return_details=True)
        aurc_score = aurc(y_noisy, base_y_pred, base_y_prob)
        ece_score = ece(y_noisy, base_y_pred, base_y_prob)

        records.append({
            'NoiseRate': noise,
            'Accuracy': acc,
            'SelectiveAccuracy': sacc,
            'CWSA': cwsa_score,
            'CWSA+': cwsa_plus_score,
            'AURC': aurc_score,
            'ECE': ece_score
        })

    return pd.DataFrame(records)

# --- Main ---
if __name__ == "__main__":
    df = evaluate_noise_impact()
    print(df)
    df.to_csv("results/noise_stress_test.csv", index=False)
