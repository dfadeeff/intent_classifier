# -*- coding: utf-8 -*-
"""
Metrics calculation for model evaluation
"""

from collections import Counter

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def calculate_basic_metrics(true_labels, predictions, confidences=None):
    """Calculate basic classification metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    avg_confidence = np.mean(confidences) if confidences else None

    return {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "total_samples": len(true_labels),
        "correct_predictions": sum(
            1 for t, p in zip(true_labels, predictions) if t == p
        ),
    }


def generate_classification_report(true_labels, predictions):
    """Generate detailed classification report"""
    return classification_report(true_labels, predictions, zero_division=0)


def find_misclassified_examples(true_labels, predictions, texts, confidences=None):
    """Find and analyze misclassified examples"""
    misclassified = []

    for i, (true_label, pred_label, text) in enumerate(
        zip(true_labels, predictions, texts)
    ):
        if true_label != pred_label:
            example = {
                "index": i,
                "text": text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidences[i] if confidences else None,
            }
            misclassified.append(example)

    return misclassified


def analyze_label_distribution(true_labels, predictions=None):
    """Analyze distribution of labels"""
    true_counts = Counter(true_labels)

    analysis = {
        "true_distribution": dict(true_counts.most_common()),
        "num_unique_labels": len(true_counts),
        "most_common_labels": true_counts.most_common(10),
    }

    if predictions:
        pred_counts = Counter(predictions)
        analysis["predicted_distribution"] = dict(pred_counts.most_common())
        analysis["prediction_bias"] = analyze_prediction_bias(true_counts, pred_counts)

    return analysis


def analyze_prediction_bias(true_counts, pred_counts):
    """Analyze if model is biased towards certain labels"""
    bias_analysis = {}

    for label in true_counts.keys():
        true_freq = true_counts[label]
        pred_freq = pred_counts.get(label, 0)

        if true_freq > 0:
            bias_ratio = pred_freq / true_freq
            bias_analysis[label] = {
                "true_count": true_freq,
                "pred_count": pred_freq,
                "bias_ratio": bias_ratio,  # >1 = overpredicted, <1 = underpredicted
                "bias_type": (
                    "overpredicted"
                    if bias_ratio > 1.2
                    else "underpredicted" if bias_ratio < 0.8 else "balanced"
                ),
            }

    return bias_analysis


def get_confusion_matrix_data(true_labels, predictions, top_k=10):
    """Get confusion matrix data for visualization"""
    # Focus on most common labels to avoid cluttered matrix
    label_counts = Counter(true_labels)
    top_labels = [label for label, _ in label_counts.most_common(top_k)]

    # Filter data to top labels
    filtered_true = [label if label in top_labels else "other" for label in true_labels]
    filtered_pred = [label if label in top_labels else "other" for label in predictions]

    # Add 'other' to labels if it exists
    if "other" in filtered_true or "other" in filtered_pred:
        top_labels.append("other")

    cm = confusion_matrix(filtered_true, filtered_pred, labels=top_labels)

    return {
        "confusion_matrix": cm,
        "labels": top_labels,
        "filtered_true": filtered_true,
        "filtered_pred": filtered_pred,
    }


def calculate_per_class_metrics(true_labels, predictions):
    """Calculate precision, recall, F1 for each class"""
    report = classification_report(
        true_labels, predictions, output_dict=True, zero_division=0
    )

    per_class_metrics = {}
    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            per_class_metrics[label] = {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": metrics["support"],
            }

    return per_class_metrics
