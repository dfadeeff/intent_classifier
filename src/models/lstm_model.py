# -*- coding: utf-8 -*-
"""
LSTM model for text classification
"""

import torch
import torch.nn as nn

from .base_model import BaseTextClassifier


class LSTMClassifier(BaseTextClassifier):
    """LSTM-based text classifier with attention"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super(LSTMClassifier, self).__init__(vocab_size, num_classes)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, lengths=None):
        """Forward pass"""
        embedded = self.embedding(input_ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        # Simple attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)

        return self.classifier(attended)

    def get_model_info(self):
        """Return model information"""
        info = super().get_model_info()
        info.update(
            {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "bidirectional": self.bidirectional,
            }
        )
        return info
