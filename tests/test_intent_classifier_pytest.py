#!/usr/bin/env python3
"""
Tests for IntentClassifier class
"""

import pytest
import tempfile
import os
import pickle
import sys
from unittest.mock import Mock, patch, MagicMock
import torch

# Add the parent directory to the Python path so we can import intent_classifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import IntentClassifier
from intent_classifier import IntentClassifier


class TestIntentClassifier:
    """Test IntentClassifier functionality"""

    def test_init(self):
        """Test IntentClassifier initialization"""
        classifier = IntentClassifier()

        assert classifier.model is None
        assert classifier.vocab is None
        assert classifier.label_encoder is None
        assert classifier.tokenizer is None
        assert classifier.model_type is None
        assert not classifier.is_ready()

    def test_clean_text(self):
        """Test text cleaning functionality"""
        classifier = IntentClassifier()

        # Test basic cleaning
        assert classifier.clean_text("Hello World!") == "hello world"
        assert classifier.clean_text("  Multiple   spaces  ") == "multiple spaces"
        assert classifier.clean_text("Special@#$%Characters") == "specialcharacters"
        assert classifier.clean_text("123 Numbers 456") == "123 numbers 456"
        assert classifier.clean_text("") == ""

    def test_text_to_indices_basic(self):
        """Test text to indices conversion"""
        classifier = IntentClassifier()

        # Mock vocab
        classifier.vocab = {"hello": 1, "world": 2, "<UNK>": 0}

        # Test basic conversion
        indices = classifier.text_to_indices("hello world")
        assert indices == [1, 2]

        # Test unknown word
        indices = classifier.text_to_indices("hello unknown")
        assert indices == [1, 0]  # unknown -> <UNK>

        # Test max length
        indices = classifier.text_to_indices("hello world test", max_len=2)
        assert len(indices) <= 2

    def test_text_to_indices_empty(self):
        """Test text to indices with empty input"""
        classifier = IntentClassifier()
        classifier.vocab = {"<UNK>": 0}

        indices = classifier.text_to_indices("")
        assert indices == []

    def test_is_ready_states(self):
        """Test is_ready method"""
        classifier = IntentClassifier()

        # Initially not ready
        assert not classifier.is_ready()

        # After setting ready flag
        classifier.ready = True
        assert classifier.is_ready()

    def test_predict_not_ready(self):
        """Test predict when model not ready"""
        classifier = IntentClassifier()

        with pytest.raises(Exception, match="Model not ready"):
            classifier.predict("test text")

    def test_load_invalid_path(self):
        """Test loading with invalid path"""
        classifier = IntentClassifier()

        with pytest.raises(Exception):
            classifier.load("/nonexistent/path")

    @patch("os.path.exists")
    def test_model_type_detection_lstm(self, mock_exists):
        """Test LSTM model detection"""
        classifier = IntentClassifier()

        # Mock file system for LSTM model
        def mock_exists_side_effect(path):
            if "tokenizer" in path:
                return False  # No tokenizer = LSTM
            if "vocab.pkl" in path:
                return True  # Has vocab = LSTM
            return True

        mock_exists.side_effect = mock_exists_side_effect

        with patch.object(classifier, "_load_lstm_transformer_model") as mock_load:
            mock_load.side_effect = Exception("Expected test error")

            try:
                classifier.load("/fake/path")
            except:
                pass  # Expected to fail

            # Should have detected LSTM and called the right method
            assert classifier.model_type == "lstm_transformer"
            mock_load.assert_called_once()

    @patch("os.path.exists")
    def test_model_type_detection_bert(self, mock_exists):
        """Test BERT model detection"""
        classifier = IntentClassifier()

        # Mock file system for BERT model
        def mock_exists_side_effect(path):
            if "tokenizer" in path:
                return True  # Has tokenizer = BERT
            if "vocab.pkl" in path:
                return False  # No vocab = BERT
            return True

        mock_exists.side_effect = mock_exists_side_effect

        with patch.object(classifier, "_load_bert_model") as mock_load:
            mock_load.side_effect = Exception("Expected test error")

            try:
                classifier.load("/fake/path")
            except:
                pass  # Expected to fail

            # Should have detected BERT and called the right method
            assert classifier.model_type == "bert"
            mock_load.assert_called_once()

    def test_predict_structure(self):
        """Test prediction output structure"""
        classifier = IntentClassifier()
        classifier.ready = True
        classifier.model_type = "lstm_transformer"

        # Mock required components
        classifier.vocab = {"test": 1, "<UNK>": 0}
        classifier.label_encoder = Mock()
        classifier.label_encoder.classes_ = ["intent1", "intent2", "intent3"]

        # Mock model that returns logits
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[0.1, 0.7, 0.2]])
        classifier.model = mock_model

        # Test prediction
        results = classifier.predict("test text")

        # Verify structure
        assert isinstance(results, list)
        assert len(results) == 3  # Should return top 3

        for result in results:
            assert "label" in result
            assert "confidence" in result
            assert isinstance(result["label"], str)
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0


class TestIntentClassifierIntegration:
    """Integration tests that require actual model files"""

    def test_load_with_missing_files(self):
        """Test loading when required files are missing"""
        classifier = IntentClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty directory
            with pytest.raises(Exception):
                classifier.load(tmpdir)

    def test_model_type_detection_logic(self):
        """Test the logic for detecting model types"""
        classifier = IntentClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test LSTM model (has vocab.pkl, no tokenizer dir)
            vocab_path = os.path.join(tmpdir, "vocab.pkl")
            with open(vocab_path, "wb") as f:
                pickle.dump({"test": 1}, f)

            # Should detect as LSTM model
            with patch.object(classifier, "_load_lstm_transformer_model") as mock_load:
                mock_load.side_effect = Exception("Expected test error")

                try:
                    classifier.load(tmpdir)
                except:
                    pass  # Expected to fail due to missing files
                # But detection logic should have been called
                mock_load.assert_called_once()


class TestIntentClassifierSimple:
    """Simplified tests that don't require complex mocking"""

    def test_basic_functionality(self):
        """Test basic functionality without complex mocking"""
        classifier = IntentClassifier()

        # Test initialization
        assert not classifier.is_ready()
        assert classifier.model_type is None

        # Test clean text
        cleaned = classifier.clean_text("Hello, World! 123")
        assert cleaned == "hello world 123"

        # Test ready state
        classifier.ready = True
        assert classifier.is_ready()


# Simple integration test
def test_complete_workflow():
    """Test the complete workflow"""
    classifier = IntentClassifier()

    # 1. Initially not ready
    assert not classifier.is_ready()

    # 2. Can't predict when not ready
    with pytest.raises(Exception):
        classifier.predict("test")

    # 3. Text cleaning works
    assert classifier.clean_text("Hello, World!") == "hello world"

    # 4. Ready state can be set
    classifier.ready = True
    assert classifier.is_ready()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
