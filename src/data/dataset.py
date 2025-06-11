# -*- coding: utf-8 -*-
"""
Dataset classes for text classification
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .preprocessing import text_to_indices


class IntentDataset(Dataset):
    """Dataset for intent classification"""

    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = text_to_indices(self.texts[idx], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "length": len(indices),
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "lengths": lengths}
