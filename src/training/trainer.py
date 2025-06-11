# -*- coding: utf-8 -*-
"""
Generic trainer for text classification models
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import save_model_components


class TextClassifierTrainer:
    """Generic trainer for text classification models"""

    def __init__(self, model, device=None, learning_rate=0.001):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"]

            logits = self.model(input_ids, lengths)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"]

                logits = self.model(input_ids, lengths)
                _, predicted = torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        self.val_accuracies.append(accuracy)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

        return accuracy

    def train(self, train_loader, val_loader, num_epochs, verbose=True):
        """Full training loop"""
        if verbose:
            print(f" Training on {self.device}")
            print(f" Model: {self.model.__class__.__name__}")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_accuracy = self.evaluate(val_loader)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Loss={train_loss:.4f}, "
                    f"Acc={val_accuracy:.4f}, "
                    f"Best={self.best_accuracy:.4f}"
                )

        return self.best_accuracy

    def save_model(self, output_dir, vocab, label_encoder):
        """Save model and components"""
        save_model_components(self.model, vocab, label_encoder, output_dir)
        print(f" Model saved to {output_dir}")
        print(f" Best accuracy: {self.best_accuracy:.4f}")
