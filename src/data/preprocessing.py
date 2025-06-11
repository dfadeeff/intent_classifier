# -*- coding: utf-8 -*-
"""
Data preprocessing utilities for text classification
"""

import re
from collections import Counter


def clean_text(text):
    """Clean text same way as training"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return " ".join(text.split())


def build_vocab(texts, min_freq=2):
    """Build vocabulary from list of texts"""
    words = []
    for text in texts:
        clean_text_result = clean_text(text)
        words.extend(clean_text_result.split())

    word_counts = Counter(words)
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def text_to_indices(text, vocab, max_len=50):
    """Convert text to indices using vocabulary"""
    words = clean_text(text).split()
    indices = [vocab.get(word, vocab["<UNK>"]) for word in words]
    if len(indices) > max_len:
        indices = indices[:max_len]
    return indices


def build_bert_vocab(texts, model_name="distilbert-base-uncased"):
    """
    For BERT, we don't build a vocab from texts, but we return a tokenizer
    that acts like a vocab for consistency with the existing interface
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a vocab-like interface for compatibility
    class BERTVocab:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size

        def __len__(self):
            return self.vocab_size

        def __getitem__(self, key):
            # For compatibility, but not used in BERT
            return 0

        def get(self, key, default=None):
            # For compatibility, but not used in BERT
            return default

    return BERTVocab(tokenizer)


def text_to_bert_indices(text, vocab, max_len=128):
    """
    Convert text to BERT token indices using the tokenizer
    This maintains the same interface as text_to_indices
    """
    tokenizer = vocab.tokenizer

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Return as list to match the interface of text_to_indices
    return encoding["input_ids"].flatten().tolist()
