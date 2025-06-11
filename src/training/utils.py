# -*- coding: utf-8 -*-
"""
Training utilities
"""

import os
import pickle

import torch


def save_model_components(model, vocab, label_encoder, output_dir):
    """Save model, vocab, and label encoder"""
    os.makedirs(output_dir, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # Save vocab
    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # Save label encoder
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)


def load_model_components(model_path):
    """Load vocab and label encoder from saved model"""
    # Load vocab
    with open(os.path.join(model_path, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    # Load label encoder
    with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return vocab, label_encoder


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, vocab, label_encoder):
    """Print model information"""
    print(f" Model: {model.__class__.__name__}")
    print(f" Vocabulary size: {len(vocab)}")
    print(f"️ Classes: {list(label_encoder.classes_)}")
    print(f" Trainable parameters: {count_parameters(model):,}")
