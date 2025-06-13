# -*- coding: utf-8 -*-
"""
Generic model evaluator that works with any trained model
"""

import os

import pandas as pd

from .metrics import (analyze_label_distribution, calculate_basic_metrics,
                      calculate_per_class_metrics, find_misclassified_examples,
                      generate_classification_report,
                      get_confusion_matrix_data)


class ModelEvaluator:
    """Generic evaluator for any trained model"""

    def __init__(self, model_classifier):
        """
        Initialize evaluator with a model classifier

        Args:
            model_classifier: Any classifier that has a predict() method
                             (e.g., IntentClassifier or future implementations)
        """
        self.classifier = model_classifier
        self.results = None

    def load_test_data(self, test_file):
        """Load test data from file"""
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        data = pd.read_csv(test_file, sep="\t", header=None, names=["text", "intent"])
        print(f"Loaded {len(data)} test samples")

        return data["text"].tolist(), data["intent"].tolist()

    def predict_batch(self, texts, verbose=True):
        """Make predictions for a batch of texts"""
        predictions = []
        confidences = []

        if verbose:
            print("Making predictions...")

        for i, text in enumerate(texts):
            if verbose and i % 100 == 0:
                print(f"Progress: {i}/{len(texts)}")

            try:
                results = self.classifier.predict(text)
                # Get top prediction
                top_prediction = results[0]
                predictions.append(top_prediction["label"])
                confidences.append(top_prediction["confidence"])
            except Exception as e:
                if verbose:
                    print(f"Error predicting for text '{text}': {e}")
                predictions.append("unknown")
                confidences.append(0.0)

        return predictions, confidences

    def evaluate(self, test_file, verbose=True):
        """Complete evaluation pipeline"""
        # Load test data
        test_texts, test_labels = self.load_test_data(test_file)

        # Make predictions
        predictions, confidences = self.predict_batch(test_texts, verbose)

        # Calculate all metrics
        basic_metrics = calculate_basic_metrics(test_labels, predictions, confidences)
        classification_rep = generate_classification_report(test_labels, predictions)
        misclassified = find_misclassified_examples(
            test_labels, predictions, test_texts, confidences
        )
        label_analysis = analyze_label_distribution(test_labels, predictions)
        confusion_data = get_confusion_matrix_data(test_labels, predictions)
        per_class_metrics = calculate_per_class_metrics(test_labels, predictions)

        # Store results
        self.results = {
            "basic_metrics": basic_metrics,
            "classification_report": classification_rep,
            "misclassified_examples": misclassified,
            "label_analysis": label_analysis,
            "confusion_matrix_data": confusion_data,
            "per_class_metrics": per_class_metrics,
            "raw_data": {
                "test_texts": test_texts,
                "test_labels": test_labels,
                "predictions": predictions,
                "confidences": confidences,
            },
        }

        return self.results

    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return

        metrics = self.results["basic_metrics"]
        misclassified = self.results["misclassified_examples"]
        label_analysis = self.results["label_analysis"]

        print(
            f"\n Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)"
        )

        if metrics["avg_confidence"]:
            print(f"Average Confidence: {metrics['avg_confidence']:.4f}")

        print(f"Correct: {metrics['correct_predictions']}/{metrics['total_samples']}")
        print(f"Misclassified: {len(misclassified)}/{metrics['total_samples']}")

        print("\n Classification Report:")
        print("=" * 80)
        print(self.results["classification_report"])

        print(f"\nTop 10 Most Common Intents:")
        for i, (intent, count) in enumerate(label_analysis["most_common_labels"], 1):
            print(f"{i:2d}. {intent}: {count} samples")

        print(f"\nMisclassified Examples (first 10):")
        print("=" * 80)
        for i, error in enumerate(misclassified[:10]):
            print(f"{i + 1:2d}. Text: '{error['text']}'")
            print(
                f"    True: {error['true_label']} | Predicted: {error['predicted_label']}",
                end="",
            )
            if error["confidence"]:
                print(f" (conf: {error['confidence']:.3f})")
            else:
                print()
            print()

    def save_results(self, output_file):
        """Save detailed results to file"""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return

        metrics = self.results["basic_metrics"]
        misclassified = self.results["misclassified_examples"]

        with open(output_file, "w") as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")

            # Basic metrics
            f.write(
                f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n"
            )
            if metrics["avg_confidence"]:
                f.write(f"Average Confidence: {metrics['avg_confidence']:.4f}\n")
            f.write(f"Total Samples: {metrics['total_samples']}\n")
            f.write(f"Correct Predictions: {metrics['correct_predictions']}\n")
            f.write(f"Misclassified: {len(misclassified)}\n\n")

            # Classification report
            f.write("Classification Report:\n")
            f.write("-" * 30 + "\n")
            f.write(self.results["classification_report"])
            f.write("\n\n")

            # Misclassified examples
            f.write("Misclassified Examples:\n")
            f.write("-" * 30 + "\n")
            for i, error in enumerate(misclassified[:50], 1):  # Save top 50
                f.write(f"{i:2d}. '{error['text']}'\n")
                f.write(
                    f"    True: {error['true_label']} | Predicted: {error['predicted_label']}"
                )
                if error["confidence"]:
                    f.write(f" (conf: {error['confidence']:.3f})")
                f.write("\n\n")

        print(f"Detailed results saved to: {output_file}")
