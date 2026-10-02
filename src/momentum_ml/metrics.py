# Out-of-sample classification metrics

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Probabilities are clipped by this amount so log loss never has to take log(0)
EPS = 1e-15


# Works out the standard binary classification metrics from the 0/1 outcomes and the predicted
# probability of an up day. Class 1 is predicted when the probability is at or above threshold.
# roc_auc and balanced_accuracy are NaN if only one class is present.
# acc_pvalue_vs_majority is a one-sided binomial test of the accuracy against always predicting
# the more common class in this sample. It is optimistic, because it assumes the rows are
# independent and same-day rows across correlated ETFs are not.
def classification_metrics(y_true, prob_up, threshold=0.5):
    outcomes = np.asarray(y_true).astype(int)
    probabilities = np.clip(np.asarray(prob_up, dtype=float), EPS, 1 - EPS)
    if len(outcomes) != len(probabilities):
        raise ValueError(
            f"y_true ({len(outcomes)}) and prob_up ({len(probabilities)}) lengths differ"
        )
    if len(outcomes) == 0:
        raise ValueError("Cannot compute metrics on an empty sample")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(outcomes, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(outcomes)) == 2
    base_rate = float(outcomes.mean())

    # Testing whether the accuracy beats always guessing the majority class
    num_correct = int((predictions == outcomes).sum())
    majority_rate = max(base_rate, 1 - base_rate)
    if 0 < majority_rate < 1:
        p_value = binomtest(num_correct, len(outcomes), majority_rate, alternative="greater").pvalue
    else:
        p_value = float("nan")

    return {
        "n": len(outcomes),
        "base_rate": base_rate,
        "pred_up_rate": float(predictions.mean()),
        "accuracy": float(accuracy_score(outcomes, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(outcomes, predictions)) if both_classes else float("nan")
        ),
        "precision": float(precision_score(outcomes, predictions, zero_division=0)),
        "recall": float(recall_score(outcomes, predictions, zero_division=0)),
        "f1": float(f1_score(outcomes, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(outcomes, probabilities)) if both_classes else float("nan"),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(outcomes, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "acc_pvalue_vs_majority": float(p_value),
    }


LOWER_IS_BETTER = {"log_loss", "brier"}


# Returns True if value a beats value b for the given metric, allowing for metrics where lower
# is better. A NaN never beats a real number.
def better(metric, a, b):
    if np.isnan(b):
        return not np.isnan(a)
    if np.isnan(a):
        return False
    return a < b if metric in LOWER_IS_BETTER else a > b
