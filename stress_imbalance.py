import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.integrate import trapz
from metrics.cwsa import cwsa, cwsa_plus

# Metric helpers
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

# Simulate imbalanced labels
def simulate_imbalanced_dataset(N=1000, imbalance_ratio=0.2, num_classes=3):
    n_minority = int(N * imbalance_ratio)
    n_majority = N - n_minority

    # Use class `0` as minority
    y = np.concatenate([
        np.full(n_minority, 0),
        np.random.randint(1, num_classes, size=n_majority)
    ])
    np.random.shuffle(y)
    return y

# Simulate a basic model
def simulate_model_from_y(y_true):
    N = len(y_true)
    y_pred = []
    y_prob = []
    for i in range(N):
        correct = np.random.rand() < 0.9
        label = y_true[i] if correct else np.random.randint(0, 3)
        conf = np.random.uniform(0.8, 1.0) if label == y_true[i] else np.random.uniform(0.5, 0.7)
        y_pred.append(label)
        y_prob.append(conf)
    return np.array(y_pred), np.array(y_prob)

# Run the imbalance experiment
def evaluate_class_imbalance(imbalance_levels=[0.0, 0.1, 0.2, 0.3, 0.4], tau=0.9):
    results = []
    for ratio in imbalance_levels:
        y_true = simulate_imbalanced_dataset(imbalance_ratio=ratio)
        y_pred, y_prob = simulate_model_from_y(y_true)

        acc = accuracy_score(y_true, y_pred)
        sacc = selective_accuracy(y_true, y_pred, y_prob, tau)
        cwsa_score, _, _ = cwsa(y_true, y_pred, y_prob, tau, return_details=True)
        cwsa_plus_score, _, _ = cwsa_plus(y_true, y_pred, y_prob, tau, return_details=True)
        aurc_score = aurc(y_true, y_pred, y_prob)
        ece_score = ece(y_true, y_pred, y_prob)

        results.append({
            'ImbalanceRatio': ratio,
            'Accuracy': acc,
            'SelectiveAccuracy': sacc,
            'CWSA': cwsa_score,
            'CWSA+': cwsa_plus_score,
            'AURC': aurc_score,
            'ECE': ece_score
        })

    return pd.DataFrame(results)

# Run and save
if __name__ == "__main__":
    df = evaluate_class_imbalance()
    print(df)
    df.to_csv("results/class_imbalance_test.csv", index=False)
