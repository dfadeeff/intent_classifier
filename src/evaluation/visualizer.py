# -*- coding: utf-8 -*-
"""
Visualization utilities for evaluation results
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EvaluationVisualizer:
    """Create visualizations for evaluation results"""

    def __init__(self, results):
        """Initialize with evaluation results"""
        self.results = results

    def plot_confusion_matrix(self, save_path=None, figsize=(10, 8)):
        """Plot confusion matrix heatmap"""
        cm_data = self.results["confusion_matrix_data"]
        cm = cm_data["confusion_matrix"]
        labels = cm_data["labels"]

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix (Top Classes)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_confidence_distribution(self, save_path=None, figsize=(10, 6)):
        """Plot distribution of prediction confidences"""
        confidences = self.results["raw_data"]["confidences"]
        predictions = self.results["raw_data"]["predictions"]
        true_labels = self.results["raw_data"]["test_labels"]

        # Separate correct and incorrect predictions
        correct_conf = [
            conf
            for conf, pred, true in zip(confidences, predictions, true_labels)
            if pred == true
        ]
        incorrect_conf = [
            conf
            for conf, pred, true in zip(confidences, predictions, true_labels)
            if pred != true
        ]

        plt.figure(figsize=figsize)

        plt.hist(
            correct_conf, bins=50, alpha=0.7, label="Correct Predictions", color="green"
        )
        plt.hist(
            incorrect_conf,
            bins=50,
            alpha=0.7,
            label="Incorrect Predictions",
            color="red",
        )

        plt.xlabel("Prediction Confidence")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Confidences")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics
        plt.axvline(
            np.mean(correct_conf),
            color="green",
            linestyle="--",
            alpha=0.8,
            label=f"Correct Mean: {np.mean(correct_conf):.3f}",
        )
        plt.axvline(
            np.mean(incorrect_conf),
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Incorrect Mean: {np.mean(incorrect_conf):.3f}",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Confidence distribution saved to: {save_path}")

        plt.show()

    def plot_class_performance(self, save_path=None, figsize=(12, 8)):
        """Plot per-class performance metrics"""
        per_class = self.results["per_class_metrics"]

        # Prepare data
        classes = list(per_class.keys())
        precision = [per_class[cls]["precision"] for cls in classes]
        recall = [per_class[cls]["recall"] for cls in classes]
        f1_score = [per_class[cls]["f1_score"] for cls in classes]
        support = [per_class[cls]["support"] for cls in classes]

        # Sort by support (frequency) for better visualization
        sorted_data = sorted(
            zip(classes, precision, recall, f1_score, support),
            key=lambda x: x[4],
            reverse=True,
        )

        # Take top 15 classes for readability
        top_data = sorted_data[:15]
        classes, precision, recall, f1_score, support = zip(*top_data)

        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=figsize)

        plt.bar(x - width, precision, width, label="Precision", alpha=0.8)
        plt.bar(x, recall, width, label="Recall", alpha=0.8)
        plt.bar(x + width, f1_score, width, label="F1-Score", alpha=0.8)

        plt.xlabel("Intent Classes")
        plt.ylabel("Score")
        plt.title("Per-Class Performance Metrics (Top 15 by Frequency)")
        plt.xticks(x, classes, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)

        # Add support annotations
        for i, sup in enumerate(support):
            plt.text(
                i, 0.05, f"n={sup}", ha="center", va="bottom", fontsize=8, rotation=90
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Class performance saved to: {save_path}")

        plt.show()

    def plot_label_distribution(self, save_path=None, figsize=(12, 6)):
        """Plot distribution of true vs predicted labels"""
        label_analysis = self.results["label_analysis"]

        true_dist = label_analysis["true_distribution"]
        pred_dist = label_analysis["predicted_distribution"]

        # Get top 15 labels by true frequency
        top_labels = [label for label, _ in Counter(true_dist).most_common(15)]

        true_counts = [true_dist.get(label, 0) for label in top_labels]
        pred_counts = [pred_dist.get(label, 0) for label in top_labels]

        x = np.arange(len(top_labels))
        width = 0.35

        plt.figure(figsize=figsize)

        plt.bar(x - width / 2, true_counts, width, label="True Labels", alpha=0.8)
        plt.bar(x + width / 2, pred_counts, width, label="Predicted Labels", alpha=0.8)

        plt.xlabel("Intent Classes")
        plt.ylabel("Count")
        plt.title("Distribution of True vs Predicted Labels (Top 15)")
        plt.xticks(x, top_labels, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Label distribution saved to: {save_path}")

        plt.show()

    def create_evaluation_report(self, save_dir="evaluation_plots"):
        """Create a complete visual evaluation report"""
        import os

        os.makedirs(save_dir, exist_ok=True)

        print("📊 Creating evaluation visualizations...")

        self.plot_confusion_matrix(f"{save_dir}/confusion_matrix.png")
        self.plot_confidence_distribution(f"{save_dir}/confidence_distribution.png")
        self.plot_class_performance(f"{save_dir}/class_performance.png")
        self.plot_label_distribution(f"{save_dir}/label_distribution.png")

        print(f"✅ All visualizations saved to: {save_dir}/")
