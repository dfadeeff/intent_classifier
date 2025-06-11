# -*- coding: utf-8 -*-

import os
import pickle
import re
import torch
import torch.nn as nn


class IntentClassifier:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.tokenizer = None  # NEW: For BERT models
        self.model_type = None  # NEW: Track model type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ready = False

    def clean_text(self, text):
        """Clean text same way as training"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return ' '.join(text.split())

    def text_to_indices(self, text, max_len=50):
        """Convert text to indices"""
        words = self.clean_text(text).split()
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        if len(indices) > max_len:
            indices = indices[:max_len]
        return indices

    def is_ready(self):
        return self.ready

    def load(self, model_path):
        """Load model and components - auto-detects model type"""
        try:
            print(f"Loading model from {model_path}")

            # NEW: Check if this is a BERT model (has tokenizer directory, no vocab.pkl)
            has_tokenizer_dir = os.path.exists(os.path.join(model_path, 'tokenizer'))
            has_vocab_pkl = os.path.exists(os.path.join(model_path, 'vocab.pkl'))

            if has_tokenizer_dir and not has_vocab_pkl:
                # This is a BERT model
                self.model_type = 'bert'
                self._load_bert_model(model_path)
            else:
                # This is LSTM/Transformer model
                self.model_type = 'lstm_transformer'
                self._load_lstm_transformer_model(model_path)

            self.ready = True
            print(f" Model loaded! Classes: {list(self.label_encoder.classes_)}")

        except Exception as e:
            print(f" Error loading model: {e}")
            self.ready = False
            raise e

    def _load_bert_model(self, model_path):
        """Load BERT model"""
        print("Detected BERT model")
        from transformers import AutoTokenizer
        from src.models.bert_model import BERTClassifier

        # Load tokenizer
        tokenizer_path = os.path.join(model_path, 'tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load label encoder
        with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Initialize BERT model
        num_classes = len(self.label_encoder.classes_)
        self.model = BERTClassifier(
            vocab_size=None,  # Not used for BERT
            num_classes=num_classes,
            model_name='distilbert-base-uncased'
        ).to(self.device)

        # Load model weights
        model_state = torch.load(
            os.path.join(model_path, 'model.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(model_state)
        self.model.eval()

    def _load_lstm_transformer_model(self, model_path):
        """Load LSTM/Transformer model (your original code)"""
        # Load vocab
        with open(os.path.join(model_path, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)

        # Load label encoder
        with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)

        vocab_size = len(self.vocab)
        num_classes = len(self.label_encoder.classes_)

        # Load the model state dict to auto-detect type and get parameters
        model_state = torch.load(
            os.path.join(model_path, 'model.pt'),
            map_location=self.device
        )

        # Auto-detect model type based on state dict keys
        if any('transformer' in key for key in model_state.keys()):
            print("Detected Transformer model")
            from src.models.transformer_model import TransformerClassifier

            # Infer max_len from the saved positional encoding
            if 'pos_encoding.pe' in model_state:
                max_len = model_state['pos_encoding.pe'].shape[0]
            else:
                max_len = 128  # fallback

            # Infer other parameters from state dict
            embedding_dim = model_state['embedding.weight'].shape[1]

            self.model = TransformerClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                max_len=max_len
            ).to(self.device)
        else:
            print("Detected LSTM model")
            from src.models.lstm_model import LSTMClassifier
            self.model = LSTMClassifier(vocab_size, num_classes).to(self.device)

        self.model.load_state_dict(model_state)
        self.model.eval()

    def predict(self, text):
        """Predict intent - handles both BERT and LSTM/Transformer"""
        if not self.ready:
            raise Exception("Model not ready")

        with torch.no_grad():
            if self.model_type == 'bert':
                # BERT prediction
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                logits = self.model(input_ids)
                probs = torch.softmax(logits, dim=1)[0]
            else:
                # LSTM/Transformer prediction (your original code)
                indices = self.text_to_indices(text)
                input_ids = torch.tensor([indices], dtype=torch.long).to(self.device)
                logits = self.model(input_ids)
                probs = torch.softmax(logits, dim=1)[0]

        # Get top 3 predictions (same for both)
        top_probs, top_indices = torch.topk(probs, k=min(3, len(self.label_encoder.classes_)))

        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results.append({
                'label': self.label_encoder.classes_[idx],
                'confidence': float(prob)
            })

        return results
