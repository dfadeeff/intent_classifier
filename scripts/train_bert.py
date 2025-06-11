#!/usr/bin/env python3
"""
BERT model training script using existing modular components
"""

import sys
import torch
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Reuse existing modular components!
from src.data.preprocessing import clean_text, build_bert_vocab, text_to_bert_indices
from src.data.dataset import IntentDataset, collate_fn
from src.models.bert_model import BERTClassifier
from src.training.trainer import TextClassifierTrainer
from src.training.utils import print_model_info


class BERTIntentDataset(IntentDataset):
    """Extends existing IntentDataset for BERT tokenization"""

    def __init__(self, texts, labels, vocab, max_len=128):
        # Use BERT-specific max_len, but same interface
        super().__init__(texts, labels, vocab, max_len)

    def __getitem__(self, idx):
        # Use BERT tokenization instead of word-based tokenization
        indices = text_to_bert_indices(self.texts[idx], self.vocab, self.max_len)
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'length': len([i for i in indices if i != 0])  # Count non-padding tokens
        }


def main():
    print("🤖 Starting BERT training...")

    # Load data (same as other models)
    train_file = "../data/atis/train.tsv"
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        print("Make sure you have data/atis/train.tsv")
        return

    data = pd.read_csv(train_file, sep='\t', header=None, names=['text', 'intent'])
    print(f"📊 Loaded {len(data)} samples with {data['intent'].nunique()} intents")

    # Build BERT "vocab" (actually a tokenizer wrapper)
    bert_vocab = build_bert_vocab(data['text'].tolist())

    # Encode labels (same as other models)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['intent'])

    # Print info
    print(f"📚 Tokenizer: DistilBERT")
    print(f"🏷️  Classes: {list(label_encoder.classes_)}")

    # Split data (same as other models)
    X_train, X_val, y_train, y_val = train_test_split(
        data['text'].tolist(), labels, test_size=0.2, random_state=42
    )

    # Create datasets using existing interface but with BERT tokenization
    train_dataset = BERTIntentDataset(X_train, y_train, bert_vocab)
    val_dataset = BERTIntentDataset(X_val, y_val, bert_vocab)

    # Use existing collate_fn!
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Create BERT model
    model = BERTClassifier(
        vocab_size=len(bert_vocab),  # For interface compatibility
        num_classes=len(label_encoder.classes_),
        model_name='distilbert-base-uncased',
        dropout=0.3,
        freeze_bert=False
    )

    print_model_info(model, bert_vocab, label_encoder)

    # Use existing trainer! Just with different learning rate
    trainer = TextClassifierTrainer(
        model,
        learning_rate=2e-5  # BERT-appropriate learning rate
    )

    # Train model using existing training loop
    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,  # Fewer epochs for BERT
        verbose=True
    )

    # Save model using existing save method - but save tokenizer instead of vocab
    output_dir = "../output_models/bert_model"
    os.makedirs(output_dir, exist_ok=True)

    # Custom save for BERT (need tokenizer not vocab)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    # Save tokenizer instead of vocab
    bert_vocab.tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    # Save label encoder (same as other models)
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"✅ BERT model saved to {output_dir}")
    print(f"🎯 Best accuracy: {best_accuracy:.4f}")

    print(f"\n📋 Next steps:")
    print(f"1. python server.py --model {output_dir}")
    print("2. curl http://localhost:8080/ready")
    print(
        "3. curl -X POST http://localhost:8080/intent -H 'Content-Type: application/json' -d '{\"text\": \"find me a flight to boston\"}'")


if __name__ == '__main__':
    main()
