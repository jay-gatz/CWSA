# metrics/cwsa.py
import numpy as np

def cwsa(y_true, y_pred, y_prob, tau=0.9, phi=None, return_details=False):
    """
    Confidence-Weighted Selective Accuracy (CWSA)
    - Signed score: correct = +φ(p), incorrect = -φ(p)
    - Ignores samples where p < τ
    """
    assert len(y_true) == len(y_pred) == len(y_prob)
    N = len(y_true)

    if phi is None:
        def phi(p): return (p - tau) / (1 - tau) if p >= tau else 0.0

    scores = []
    accept_mask = []
    for i in range(N):
        p = y_prob[i]
        if p < tau:
            scores.append(0.0)
            accept_mask.append(False)
        elif y_pred[i] == y_true[i]:
            scores.append(+phi(p))
            accept_mask.append(True)
        else:
            scores.append(-phi(p))
            accept_mask.append(True)

    accepted = sum(accept_mask)
    if accepted == 0:
        return (0.0, scores, 0.0) if return_details else 0.0

    score = np.mean(scores)
    coverage = accepted / N
    return (score, scores, coverage) if return_details else score


def cwsa_plus(y_true, y_pred, y_prob, tau=0.9, phi=None, return_details=False):
    """
    Confidence-Weighted Selective Accuracy Plus (CWSA+)
    - Normalized: score in [0, 1]
    - Only rewards correct predictions above τ
    - Ideal for comparison across models
    """
    assert len(y_true) == len(y_pred) == len(y_prob)
    N = len(y_true)

    if phi is None:
        def phi(p): return (p - tau) / (1 - tau) if p >= tau else 0.0

    contributions = []
    accept_mask = []

    for i in range(N):
        p = y_prob[i]
        if p >= tau:
            if y_pred[i] == y_true[i]:
                contributions.append(phi(p))
            else:
                contributions.append(0.0)
            accept_mask.append(True)
        else:
            contributions.append(0.0)
            accept_mask.append(False)

    accepted = sum(accept_mask)
    if accepted == 0:
        return (0.0, contributions, 0.0) if return_details else 0.0

    score = sum(contributions) / accepted
    coverage = accepted / N
    return (score, contributions, coverage) if return_details else score
