# -*- coding: utf-8 -*-
"""
Transformer model for text classification
"""

import math

import torch
import torch.nn as nn

from .base_model import BaseTextClassifier


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerClassifier(BaseTextClassifier):
    """Transformer-based text classifier"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=256,  # increased from 128
        num_heads=8,
        num_layers=6,  # increased from 4
        dim_feedforward=1024,  # increased from 512
        dropout=0.1,
        max_len=512,
        use_layer_norm=True,  # additional normalization
    ):
        super(TransformerClassifier, self).__init__(vocab_size, num_classes)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.use_layer_norm = use_layer_norm

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        self.embedding_dropout = nn.Dropout(dropout)

        # Optional input layer norm
        if use_layer_norm:
            self.input_layer_norm = nn.LayerNorm(embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",  # changed from relu
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Multi-head pooling (more sophisticated than simple average)
        self.pooling_heads = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            # nn.Linear(embedding_dim, dim_feedforward // 2),
            nn.Linear(embedding_dim, dim_feedforward),  # larger hidden layer
            nn.GELU(),  # changed from ReLU()
            nn.Dropout(dropout),
            # added two more steps
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Linear(dim_feedforward // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use better initialization for deeper networks
                # nn.init.xavier_uniform_(module.weight)
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, 0, 0.02
                )  # Smaller std for larger embeddings
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _create_padding_mask(self, input_ids):
        """Create padding mask for transformer"""
        # Mask where input_ids == 0 (padding token)
        return input_ids == 0

    def _attention_pooling(self, transformer_out, padding_mask):
        """Attention-based pooling instead of simple average"""
        # Create a learnable query for pooling
        batch_size = transformer_out.size(0)

        # Use a special token representation as query
        cls_token = transformer_out[:, 0:1, :]  # Use first token as query

        # Apply attention pooling
        pooled_output, attention_weights = self.pooling_heads(
            query=cls_token,
            key=transformer_out,
            value=transformer_out,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        return pooled_output.squeeze(1)  # Remove sequence dimension

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

        # Optional input normalization
        if self.use_layer_norm:
            embedded = self.input_layer_norm(embedded)

        # Transformer encoding
        transformer_out = self.transformer(
            embedded, src_key_padding_mask=src_key_padding_mask
        )

        # # Global average pooling (ignoring padded positions)
        # mask = (~src_key_padding_mask).float().unsqueeze(-1)
        # masked_out = transformer_out * mask
        # pooled = masked_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        # Attention-based pooling instead of simple average
        pooled = self._attention_pooling(transformer_out, src_key_padding_mask)

        # Classification
        return self.classifier(pooled)

    def get_model_info(self):
        """Return model information"""
        info = super().get_model_info()

        # Calculate approximate parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "max_len": self.max_len,
                "use_layer_norm": self.use_layer_norm,
                "total_parameters": total_params,
                "architecture": "Enhanced Transformer with attention pooling",
            }
        )
        return info
