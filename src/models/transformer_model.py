# -*- coding: utf-8 -*-
"""
Transformer model for text classification
"""

import torch
import torch.nn as nn
import math
from .base_model import BaseTextClassifier


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerClassifier(BaseTextClassifier):
    """Transformer-based text classifier"""

    def __init__(self, vocab_size, num_classes, embedding_dim=128,
                 num_heads=8, num_layers=4, dim_feedforward=512,
                 dropout=0.1, max_len=512):
        super(TransformerClassifier, self).__init__(vocab_size, num_classes)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)

    def _create_padding_mask(self, input_ids):
        """Create padding mask for transformer"""
        # Mask where input_ids == 0 (padding token)
        return input_ids == 0

    def forward(self, input_ids, lengths=None):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape

        # Create padding mask
        src_key_padding_mask = self._create_padding_mask(input_ids)

        # Embedding with positional encoding
        embedded = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        embedded = self.pos_encoding(embedded)
        embedded = embedded.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        embedded = self.embedding_dropout(embedded)

        # Transformer encoding
        transformer_out = self.transformer(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        # Global average pooling (ignoring padded positions)
        mask = (~src_key_padding_mask).float().unsqueeze(-1)
        masked_out = transformer_out * mask
        pooled = masked_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        # Classification
        return self.classifier(pooled)

    def get_model_info(self):
        """Return model information"""
        info = super().get_model_info()
        info.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'max_len': self.max_len
        })
        return info