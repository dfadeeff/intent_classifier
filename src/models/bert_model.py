# -*- coding: utf-8 -*-
"""
BERT-based model for text classification
"""

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .base_model import BaseTextClassifier


class BERTClassifier(BaseTextClassifier):
    """BERT-based text classifier using pre-trained transformer"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        model_name="distilbert-base-uncased",
        dropout=0.3,
        freeze_bert=False,
    ):
        # Note: vocab_size is ignored for BERT (uses its own tokenizer)
        super(BERTClassifier, self).__init__(vocab_size, num_classes)

        self.model_name = model_name
        self.dropout = dropout
        self.freeze_bert = freeze_bert

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes),
        )

        # Store tokenizer for easy access
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, lengths=None):
        """Forward pass through BERT + classifier

        Args:
            input_ids: Either token indices (for LSTM/Transformer) or BERT token IDs
            lengths: Not used for BERT, but kept for interface compatibility
        """
        # For BERT, we expect input_ids to be BERT token IDs with attention mask info
        # Handle attention mask creation here
        attention_mask = (input_ids != 0).long()

        # Get BERT outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation for classification
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Classification
        return self.classifier(cls_output)

    def get_model_info(self):
        """Return model information"""
        info = super().get_model_info()
        info.update(
            {
                "model_name": self.model_name,
                "hidden_size": self.hidden_size,
                "freeze_bert": self.freeze_bert,
            }
        )
        return info
