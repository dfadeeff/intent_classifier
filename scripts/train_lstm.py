#!/usr/bin/env python3
"""
LSTM model training script using modular components
"""

import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add src to path (fix the path)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Fix imports - remove 'src.' prefix since we added the parent directory to path
from src.data.preprocessing import build_vocab
from src.data.dataset import IntentDataset, collate_fn
from src.models.lstm_model import LSTMClassifier
from src.training.trainer import TextClassifierTrainer
from src.training.utils import print_model_info


def main():
    print("🚀 Starting LSTM training...")

    # Load data (fix path to work from scripts directory)
    train_file = "../data/atis/train.tsv"
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        print("Make sure you have data/atis/train.tsv")
        return

    data = pd.read_csv(train_file, sep='\t', header=None, names=['text', 'intent'])
    print(f"📊 Loaded {len(data)} samples with {data['intent'].nunique()} intents")

    # Build vocab
    vocab = build_vocab(data['text'].tolist())

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['intent'])

    # Print info (fix - pass a dummy model first)
    print(f"📚 Vocabulary size: {len(vocab)}")
    print(f"🏷️  Classes: {list(label_encoder.classes_)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        data['text'].tolist(), labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = IntentDataset(X_train, y_train, vocab)
    val_dataset = IntentDataset(X_val, y_val, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Create model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_encoder.classes_),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )

    print_model_info(model, vocab, label_encoder)

    # Create trainer
    trainer = TextClassifierTrainer(model, learning_rate=0.001)

    # Train model
    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        verbose=True
    )

    # Save model (fix path)
    output_dir = "../output_models/lstm_model"
    trainer.save_model(output_dir, vocab, label_encoder)

    print(f"\n📋 Next steps:")
    print(f"1. python server.py --model {output_dir}")
    print("2. curl http://localhost:8080/ready")
    print("3. curl -X POST http://localhost:8080/intent -H 'Content-Type: application/json' -d '{\"text\": \"find me a flight to boston\"}'")


if __name__ == '__main__':
    main()