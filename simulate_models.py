import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.integrate import trapz

# Import your metric implementations
from metrics.cwsa import cwsa, cwsa_plus

# ======= Helper Metric Functions =======

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

# ======= Synthetic Model Simulator =======

def simulate_model(N=1000, mode='calibrated', num_classes=3):
    y_true = np.random.randint(0, num_classes, size=N)
    y_pred = np.random.randint(0, num_classes, size=N)
    y_prob = np.zeros(N)

    for i in range(N):
        if mode == 'perfect':
            y_pred[i] = y_true[i]
            y_prob[i] = np.random.uniform(0.95, 1.0)
        elif mode == 'calibrated':
            correct = np.random.rand() < 0.9
            y_pred[i] = y_true[i] if correct else np.random.randint(0, num_classes)
            y_prob[i] = np.random.uniform(0.8, 1.0) if correct else np.random.uniform(0.5, 0.7)
        elif mode == 'overconfident':
            correct = np.random.rand() < 0.7
            y_pred[i] = y_true[i] if correct else np.random.randint(0, num_classes)
            y_prob[i] = np.random.uniform(0.9, 1.0)
        elif mode == 'underconfident':
            correct = np.random.rand() < 0.9
            y_pred[i] = y_true[i] if correct else np.random.randint(0, num_classes)
            y_prob[i] = np.random.uniform(0.4, 0.6)
        elif mode == 'random':
            y_pred[i] = np.random.randint(0, num_classes)
            y_prob[i] = np.random.uniform(0.3, 1.0)

    return y_true, y_pred, y_prob

# ======= Evaluation =======

def evaluate_models(tau=0.9, N=1000):
    model_types = ['perfect', 'calibrated', 'overconfident', 'underconfident', 'random']
    results = []

    for model in model_types:
        y_true, y_pred, y_prob = simulate_model(N=N, mode=model)

        acc = accuracy_score(y_true, y_pred)
        sacc = selective_accuracy(y_true, y_pred, y_prob, tau=tau)
        cwsa_score, _, _ = cwsa(y_true, y_pred, y_prob, tau=tau, return_details=True)
        cwsa_plus_score, _, _ = cwsa_plus(y_true, y_pred, y_prob, tau=tau, return_details=True)
        aurc_score = aurc(y_true, y_pred, y_prob)
        ece_score = ece(y_true, y_pred, y_prob)

        results.append({
            'Model': model,
            'Accuracy': acc,
            'SelectiveAccuracy': sacc,
            'CWSA': cwsa_score,
            'CWSA+': cwsa_plus_score,
            'AURC': aurc_score,
            'ECE': ece_score
        })

    return pd.DataFrame(results)

# ======= Run and Save =======

if __name__ == "__main__":
    df = evaluate_models()
    print(df)
    df.to_csv("results/synthetic_model_metrics.csv", index=False)
