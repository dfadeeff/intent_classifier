#!/usr/bin/env python3
"""
Generic model testing script that works with any trained model
Usage: python test_model.py [--model MODEL_PATH] [--data TEST_FILE] [--visualize]
"""

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import EvaluationVisualizer

# Import the intent classifier (works with any model saved in the standard format)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from intent_classifier import IntentClassifier


def detect_model_type(model_path):
    """Detect if this is a BERT model or traditional model"""
    tokenizer_dir = os.path.join(model_path, 'tokenizer')
    vocab_file = os.path.join(model_path, 'vocab.pkl')

    if os.path.exists(tokenizer_dir):
        return 'bert'
    elif os.path.exists(vocab_file):
        return 'traditional'
    else:
        return 'unknown'


def load_bert_model(model_path):
    """Load BERT model with special handling"""
    import torch
    import pickle
    from transformers import DistilBertTokenizer
    from src.models.bert_model import BERTClassifier

    # Load components
    model_file = os.path.join(model_path, 'model.pt')
    tokenizer_dir = os.path.join(model_path, 'tokenizer')
    label_encoder_file = os.path.join(model_path, 'label_encoder.pkl')

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_dir)

    # Create BERTVocab class (same as in preprocessing.py)
    class BERTVocab:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size

        def __len__(self):
            return self.vocab_size

        def __getitem__(self, key):
            return 0

        def get(self, key, default=None):
            return default

    bert_vocab = BERTVocab(tokenizer)

    # Load label encoder
    with open(label_encoder_file, 'rb') as f:
        label_encoder = pickle.load(f)

    # Create model
    model = BERTClassifier(
        vocab_size=len(bert_vocab),
        num_classes=len(label_encoder.classes_),
        model_name='distilbert-base-uncased'
    )

    # Load state dict
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    return model, bert_vocab, label_encoder


def main():
    parser = argparse.ArgumentParser(description='Test trained intent classification model')
    parser.add_argument('--model', type=str, default='../output_models/lstm_model',
                        help='Path to model directory (default: ../output_models/lstm_model)')
    parser.add_argument('--data', type=str, default='../data/atis/test.tsv',
                        help='Path to test data file (default: ../data/atis/test.tsv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (default: auto-generated based on model name)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Convert relative paths to absolute paths for consistency
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Handle relative paths by making them relative to project root
    if not os.path.isabs(args.model):
        if args.model.startswith('../'):
            args.model = os.path.join(project_root, args.model[3:])
        else:
            args.model = os.path.join(project_root, args.model)

    if not os.path.isabs(args.data):
        if args.data.startswith('../'):
            args.data = os.path.join(project_root, args.data[3:])
        else:
            args.data = os.path.join(project_root, args.data)

    # Extract model name from path for organized output
    model_name = os.path.basename(args.model.rstrip('/'))

    # Create organized output directory structure
    results_dir = os.path.join(project_root, "evaluation_results", f"{model_name}_results")
    os.makedirs(results_dir, exist_ok=True)

    # Set output file if not specified
    if args.output is None:
        args.output = os.path.join(results_dir, "evaluation_report.txt")
    elif not os.path.isabs(args.output):
        if args.output.startswith('../'):
            args.output = os.path.join(project_root, args.output[3:])
        else:
            args.output = os.path.join(project_root, args.output)

    print("🧪 Testing trained model...")
    print(f"Model: {args.model}")
    print(f"Test data: {args.data}")
    print(f"Results will be saved to: {results_dir}")
    print("=" * 50)

    # Detect model type and load accordingly
    model_type = detect_model_type(args.model)
    print(f"🔍 Detected model type: {model_type}")

    try:
        if model_type == 'bert':
            print("🤖 Loading BERT model...")
            model, vocab, label_encoder = load_bert_model(args.model)

            # Create a custom classifier wrapper for BERT
            class BERTClassifierWrapper:
                def __init__(self, model, vocab, label_encoder):
                    self.model = model
                    self.vocab = vocab
                    self.label_encoder = label_encoder

                def predict(self, text):
                    from src.data.preprocessing import text_to_bert_indices
                    import torch

                    # Tokenize text
                    indices = text_to_bert_indices(text, self.vocab, max_len=128)
                    input_ids = torch.tensor([indices], dtype=torch.long)

                    # Get prediction
                    with torch.no_grad():
                        logits = self.model(input_ids)
                        probs = torch.softmax(logits, dim=-1)

                    # Get top 3 predictions
                    top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

                    results = []
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        label = self.label_encoder.classes_[idx.item()]
                        results.append({
                            'label': label,
                            'confidence': prob.item()
                        })

                    return results

            classifier = BERTClassifierWrapper(model, vocab, label_encoder)
            print("✅ BERT model loaded successfully!")

        else:
            # Use traditional IntentClassifier for LSTM/Transformer
            print("🔄 Loading traditional model...")
            classifier = IntentClassifier()
            classifier.load(args.model)
            print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create evaluator
    evaluator = ModelEvaluator(classifier)

    # Run evaluation
    try:
        results = evaluator.evaluate(args.data, verbose=not args.quiet)

        # Print summary
        evaluator.print_summary()

        # Save detailed results
        evaluator.save_results(args.output)

        # Create visualizations if requested
        if args.visualize:
            try:
                visualizer = EvaluationVisualizer(results)
                viz_dir = os.path.join(results_dir, "plots")
                visualizer.create_evaluation_report(viz_dir)
            except ImportError:
                print("⚠️  Visualization requires matplotlib and seaborn.")
                print("   Install with: pip install matplotlib seaborn")
            except Exception as e:
                print(f"⚠️  Visualization error: {e}")

        # Final summary
        accuracy = results['basic_metrics']['accuracy']
        misclassified_count = len(results['misclassified_examples'])
        total_samples = results['basic_metrics']['total_samples']

        print(f"\n🎉 Final Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"📁 All results saved to: {results_dir}")
        print(f"   📄 Detailed report: {os.path.basename(args.output)}")
        if args.visualize:
            print(f"   📊 Visualizations: plots/")

        # Create a quick summary file as well
        summary_file = os.path.join(results_dir, "quick_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Model Evaluation Summary\n")
            f.write(f"{'=' * 30}\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
            f.write(f"Correct Predictions: {total_samples - misclassified_count}/{total_samples}\n")
            f.write(f"Misclassified: {misclassified_count}\n")
            f.write(f"Average Confidence: {results['basic_metrics']['avg_confidence']:.4f}\n\n")
            f.write(f"Files generated:\n")
            f.write(f"- evaluation_report.txt (detailed results)\n")
            f.write(f"- quick_summary.txt (this file)\n")
            if args.visualize:
                f.write(f"- plots/ (visualization charts)\n")

        print(f"   📋 Quick summary: quick_summary.txt")

        return 0

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())