"""
metrics.py — Comprehensive classification metrics for model evaluation.

After training, we want more than just "accuracy" to understand how the model
is performing. This module computes a full suite of metrics from predictions
and probabilities, giving you a 360-degree view of model health.

Metrics computed:
  - Precision, Recall, F1 (macro avg across all 10 digits)
  - AUC-ROC and AUC-PR (how well the model ranks correct answers)
  - Per-class accuracy, FP, FN, specificity
  - MCC, Cohen's Kappa, balanced accuracy
  - Top-2 accuracy, most confused digit pairs
  - Log loss (per-sample cross-entropy)
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
)

NUM_CLASSES = 10


def compute_all_metrics(true_labels, pred_labels, probabilities):
    """
    Compute every classification metric we track.

    Args:
        true_labels:   numpy array of shape (N,) — ground truth digit labels (0-9)
        pred_labels:   numpy array of shape (N,) — model's predicted digit labels (0-9)
        probabilities: numpy array of shape (N, 10) — softmax probabilities per class

    Returns:
        dict with all metrics, ready to dump to JSON or print
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=range(NUM_CLASSES))

    return {
        "summary": _summary_metrics(true_labels, pred_labels, probabilities),
        "per_class": _per_class_metrics(cm, true_labels, pred_labels),
        "confusion_pairs": _most_confused_pairs(cm, top_n=5),
        "top_2_accuracy": _top_k_accuracy(true_labels, probabilities, k=2),
    }


def _summary_metrics(true_labels, pred_labels, probabilities):
    """Global summary metrics (single numbers)."""
    return {
        "accuracy": float((true_labels == pred_labels).mean()),
        "precision_macro": float(precision_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true_labels, pred_labels)),
        "cohen_kappa": float(cohen_kappa_score(true_labels, pred_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, pred_labels)),
        "log_loss": float(log_loss(true_labels, probabilities, labels=range(NUM_CLASSES))),
        "auc_roc_macro": float(roc_auc_score(true_labels, probabilities, multi_class="ovr", average="macro")),
        "auc_pr_macro": _macro_auc_pr(true_labels, probabilities),
    }


def _macro_auc_pr(true_labels, probabilities):
    """Compute macro-averaged AUC-PR (one-vs-rest for each class, then average)."""
    auc_scores = []
    for cls in range(NUM_CLASSES):
        binary_true = (true_labels == cls).astype(int)
        auc_scores.append(float(average_precision_score(binary_true, probabilities[:, cls])))
    return float(np.mean(auc_scores))


def _per_class_metrics(cm, true_labels, pred_labels):
    """Per-digit breakdown: accuracy, precision, recall, specificity, FP, FN."""
    per_class = {}
    total = cm.sum()

    for cls in range(NUM_CLASSES):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp           # actual=cls but predicted wrong
        fp = cm[:, cls].sum() - tp           # predicted=cls but actually wrong
        tn = total - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        per_class[str(cls)] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(float(specificity), 4),
            "f1": round(float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "support": int(cm[cls, :].sum()),  # total actual samples for this digit
        }

    return per_class


def _most_confused_pairs(cm, top_n=5):
    """Find the digit pairs that get confused most often (off-diagonal)."""
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": i,
                    "predicted": j,
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def _top_k_accuracy(true_labels, probabilities, k=2):
    """Was the correct label in the model's top-k guesses?"""
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]  # top k indices
    correct = sum(true_labels[i] in top_k_preds[i] for i in range(len(true_labels)))
    return round(float(correct / len(true_labels)), 4)


def print_metrics_report(metrics):
    """Print a clean, scannable metrics report to the terminal."""
    s = metrics["summary"]
    print(f"\n{'=' * 55}")
    print(f"  Detailed Metrics Report")
    print(f"{'=' * 55}")

    # Summary block
    print(f"\n  Overall:")
    print(f"    Accuracy:          {s['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {s['balanced_accuracy']:.4f}")
    print(f"    Precision (macro): {s['precision_macro']:.4f}")
    print(f"    Recall (macro):    {s['recall_macro']:.4f}")
    print(f"    F1 (macro):        {s['f1_macro']:.4f}")
    print(f"    MCC:               {s['mcc']:.4f}")
    print(f"    Cohen's Kappa:     {s['cohen_kappa']:.4f}")
    print(f"    Log Loss:          {s['log_loss']:.4f}")
    print(f"    AUC-ROC (macro):   {s['auc_roc_macro']:.4f}")
    print(f"    AUC-PR (macro):    {s['auc_pr_macro']:.4f}")
    print(f"    Top-2 Accuracy:    {metrics['top_2_accuracy']:.4f}")

    # Per-class table
    print(f"\n  Per-Class Breakdown:")
    print(f"    {'Digit':<7}{'Prec':>7}{'Recall':>8}{'F1':>7}{'Spec':>7}{'FP':>6}{'FN':>6}{'Support':>9}")
    print(f"    {'-' * 51}")
    for cls in range(NUM_CLASSES):
        c = metrics["per_class"][str(cls)]
        print(f"    {cls:<7}{c['precision']:>7.4f}{c['recall']:>8.4f}{c['f1']:>7.4f}"
              f"{c['specificity']:>7.4f}{c['fp']:>6}{c['fn']:>6}{c['support']:>9}")

    # Most confused pairs
    print(f"\n  Most Confused Pairs:")
    for pair in metrics["confusion_pairs"]:
        print(f"    {pair['true']} -> {pair['predicted']}:  {pair['count']} mistakes")

    print(f"{'=' * 55}\n")
