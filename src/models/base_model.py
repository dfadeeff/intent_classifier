# -*- coding: utf-8 -*-
"""
Base model interface for text classification
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseTextClassifier(nn.Module, ABC):
    """Abstract base class for text classifiers"""

    def __init__(self, vocab_size, num_classes):
        super(BaseTextClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, input_ids, lengths=None):
        """Forward pass through the model"""
        pass

    def get_model_info(self):
        """Return model information"""
        return {
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "model_type": self.__class__.__name__,
        }
