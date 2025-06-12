#!/usr/bin/env python3
"""
Transformer model training script using modular components
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Add src to path (fix the path)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import IntentDataset, collate_fn
from src.data.preprocessing import build_vocab
from src.models.transformer_model import TransformerClassifier
from src.training.trainer import TextClassifierTrainer
from src.training.utils import print_model_info


def main():
    print("Starting Transformer training...")

    # Load data (fix path to work from scripts directory)
    train_file = "../data/atis/train.tsv"
    if not os.path.exists(train_file):
        print(f"File not found: {train_file}")
        print("Make sure you have data/atis/train.tsv")
        return

    data = pd.read_csv(train_file, sep="\t", header=None, names=["text", "intent"])
    print(f"Loaded {len(data)} samples with {data['intent'].nunique()} intents")

    # Build vocab
    vocab = build_vocab(data["text"].tolist())

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["intent"])

    # Print info
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        data["text"].tolist(), labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = IntentDataset(X_train, y_train, vocab)
    val_dataset = IntentDataset(X_val, y_val, vocab)

    # Use your existing collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    # Create transformer model
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_encoder.classes_),
        embedding_dim=256,  # Increased from 128
        num_heads=8,  # 8 attention heads
        num_layers=6,  # increased from 4 transformer layers
        dim_feedforward=1024,  # increased from 512
        dropout=0.1,  # Lower dropout for transformer
        max_len=512,  # Maximum sequence length increased from 128
        use_layer_norm=True,
    )

    print_model_info(model, vocab, label_encoder)

    # Create trainer - use your existing trainer interface
    trainer = TextClassifierTrainer(
        model, learning_rate=0.0001  # Lower learning rate for transformer
    )

    # Train model with more epochs for transformer
    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,  # increased from 25 which was for a smaller transformer
        verbose=True,
    )

    # Save model - use your existing save method
    output_dir = "../output_models/transformer_model"
    trainer.save_model(output_dir, vocab, label_encoder)

    print(f"\n📋 Next steps:")
    print(f"1. python server.py --model {output_dir}")
    print("2. curl http://localhost:8080/ready")
    print(
        "3. curl -X POST http://localhost:8080/intent -H 'Content-Type: application/json' -d '{\"text\": \"find me a flight to boston\"}'"
    )


if __name__ == "__main__":
    main()
