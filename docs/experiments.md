# Model Comparison Experiments

## Overview
This document outlines the experimental approach and results for selecting the best model architecture for intent classification on the ATIS dataset. All results are from actual training and testing runs.

## Dataset
- **Dataset**: ATIS (Airline Travel Information System)
- **Language**: English
- **Format**: TSV files with text and intent labels
- **Classes**: 22 intent categories
- **Test Set**: 850 samples
- **Split**: 80/20 train/validation split with separate test set

## Experimental Setup


### Evaluation Framework
```bash
# Consistent testing across all models
python test_model.py --model output_models/[model_name]

# Metrics collected:
# - Test accuracy on 850 samples
# - Per-class precision, recall, F1-score
# - Average confidence scores
# - Detailed misclassification analysis
```

---

## Model 1: LSTM Classifier

### Architecture
```text
class LSTMClassifier(BaseTextClassifier):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=128,      # Compact embeddings
        hidden_dim=256,         # Bidirectional = 512 total
        num_layers=2,           # Shallow network
        dropout=0.3,            # Higher dropout
        bidirectional=True,     # Forward + backward
    ):
```

### Key Features
- **Bidirectional LSTM**: Captures context from both directions
- **Attention mechanism**: Weighted pooling of hidden states
- **Compact design**: Optimized for speed and efficiency
- **Higher regularization**: 0.3 dropout to prevent overfitting

### Training Configuration
```text
model = LSTMClassifier(
    vocab_size=len(vocab),
    num_classes=len(label_encoder.classes_),
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
)

trainer = TextClassifierTrainer(model, learning_rate=0.001)  # Higher LR
best_accuracy = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    verbose=True,
)
```

### Test Results
```
🎯 Test Accuracy: 0.9565 (95.65%)
📊 Average Confidence: 0.9911
✅ Correct: 813/850
❌ Misclassified: 37/850

Weighted Average Metrics:
- Precision: 0.95
- Recall: 0.96  
- F1-Score: 0.95
```

### Performance Analysis
- **Fast convergence**: Efficient training with higher learning rate
- **High confidence**: 99.11% average confidence shows model certainty
- **Balanced performance**: Good precision/recall across most classes
- **Efficient**: Smallest model with solid performance

---

## Model 2: Enhanced Transformer Classifier

### Architecture
```text
class TransformerClassifier(BaseTextClassifier):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=256,      # 2x larger than LSTM
        num_heads=8,            # Multi-head attention
        num_layers=6,           # 3x deeper than LSTM
        dim_feedforward=1024,   # Large feed-forward
        dropout=0.1,            # Lower dropout
        max_len=512,            # Longer sequences
        use_layer_norm=True,    # Additional normalization
    ):
```

### Key Features
- **Multi-head self-attention**: 8 attention heads for complex patterns
- **Positional encoding**: Sinusoidal position embeddings
- **Attention pooling**: Sophisticated aggregation mechanism
- **Modern architecture**: GELU activation, pre-norm, layer normalization
- **Scaled parameters**: 5.9M parameters (20x larger than LSTM)

### Training Configuration
```text
model = TransformerClassifier(
    vocab_size=len(vocab),
    num_classes=len(label_encoder.classes_),
    embedding_dim=256,
    num_heads=8,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_len=512,
    use_layer_norm=True,
)

trainer = TextClassifierTrainer(model, learning_rate=0.0001)  # Lower LR
best_accuracy = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,
    verbose=True,
)
```

### Test Results
```
🎯 Test Accuracy: 0.9647 (96.47%)
📊 Average Confidence: 0.9916
✅ Correct: 820/850
❌ Misclassified: 30/850

Weighted Average Metrics:
- Precision: 0.97
- Recall: 0.96
- F1-Score: 0.96
```

### Performance Analysis
- **Excellent accuracy**: 96.47% test accuracy (0.82% improvement over LSTM)
- **Fast convergence**: 97%+ validation accuracy by epoch 8
- **Stable training**: Consistent performance across epochs
- **High confidence**: 99.16% average confidence

---

## Model 3: BERT Classifier

### Architecture
```text
class BERTClassifier(BaseTextClassifier):
    def __init__(
        self,
        vocab_size,              # Ignored - uses BERT tokenizer
        num_classes,
        model_name="distilbert-base-uncased",
        dropout=0.3,
        freeze_bert=False,       # Fine-tune vs freeze
    ):
```

### Key Features
- **Pre-trained backbone**: DistilBERT with 66M parameters
- **Transfer learning**: Leverages pre-trained language understanding
- **Efficient variant**: DistilBERT (40% smaller than full BERT)
- **Custom classification head**: Task-specific layers on top

### Training Configuration
```text
model = BERTClassifier(
    vocab_size=len(vocab),  # Not used
    num_classes=len(label_encoder.classes_),
    model_name="distilbert-base-uncased",
    dropout=0.3,
    freeze_bert=False,      # Fine-tune entire model
)

trainer = TextClassifierTrainer(model, learning_rate=0.00002)  # Very low LR
best_accuracy = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5,           # Fewer epochs due to pre-training
    verbose=True,
)
```

### Test Results
```
🎯 Test Accuracy: 0.9729 (97.29%)
📊 Average Confidence: 0.9838
✅ Correct: 827/850
❌ Misclassified: 23/850

Weighted Average Metrics:
- Precision: 0.97
- Recall: 0.97
- F1-Score: 0.97
```

### Performance Analysis
- **Highest accuracy**: 97.29% test accuracy (best performance)
- **Pre-training advantage**: Benefits from large-scale language modeling
- **Slightly lower confidence**: 98.38% (more calibrated predictions)
- **Fewest errors**: Only 23 misclassifications

---

## Comprehensive Model Comparison

### Performance Summary

| Model | Parameters | Test Accuracy | Avg Confidence | Misclassified | Weighted F1 |
|-------|------------|---------------|----------------|---------------|-------------|
| LSTM | ~2.1m      | **95.65%** | 99.11% | 37/850 | 0.95 |
| Transformer | ~5.9M      | **96.47%** | 99.16% | 30/850 | 0.96 |
| **BERT** | **66M**    | **97.29%** | **98.38%** | **23/850** | **0.97** |

### Key Insights

#### 1. Accuracy Progression
- **LSTM → Transformer**: +0.82% accuracy gain (3x more parameters)
- **Transformer → BERT**: +0.82% accuracy gain (11x more parameters)
- **LSTM → BERT**: +1.64% total accuracy improvement

#### 2. Parameter Efficiency
- **LSTM**: Most parameter-efficient (95.65% with 2.1m params)
- **Transformer**: Good balance (96.47% with 5.9M params)
- **BERT**: Best accuracy but largest model (97.29% with 66M params)

#### 3. Confidence Calibration
- **LSTM/Transformer**: Very high confidence (99%+) but slightly overconfident
- **BERT**: More calibrated confidence (98.38%) with better accuracy

### Error Analysis

#### Common Misclassification Patterns (All Models)
1. **Multi-intent confusion**: `flight+airfare` → `flight` or `airfare`
2. **Rare class issues**: `day_name` consistently misclassified as `flight`
3. **Similar intents**: `ground_fare` vs `ground_service` confusion
4. **Quantity vs flight**: `"how many flights"` → `quantity` vs `flight`

#### BERT Advantages
- **Better context understanding**: Fewer multi-intent errors
- **Improved rare class handling**: Better performance on minority classes
- **More nuanced predictions**: Lower confidence on ambiguous cases

---

## Production Considerations

### Speed vs Accuracy Trade-offs

| Model | Inference Speed | Memory Usage | Accuracy | Use Case |
|-------|----------------|--------------|----------|----------|
| LSTM | Fastest | Lowest | Good | Real-time, high-throughput |
| Transformer | Medium | Medium | Better | Balanced production |
| BERT | Slowest | Highest | Best | High-accuracy requirements |

### Deployment Recommendations

#### Choose LSTM when:
- Need fastest inference speed
- Memory/compute constraints
- Good accuracy (95.65%) is sufficient

#### Choose Transformer when:
- Want balanced speed/accuracy
- Can afford 5.9M parameters
- Need better than LSTM performance

#### Choose BERT when:
- Accuracy is paramount
- Can afford computational cost
- 1.64% accuracy gain justifies resources

---

## Final Architecture Decision

**Selected: BERT (DistilBERT) for Production**

### Rationale
1. **Superior performance**: 97.29% accuracy (highest across all models)
2. **Fewer errors**: Only 23 misclassifications vs 30-37 for others
3. **Better calibration**: More realistic confidence scores
4. **Proven architecture**: Leverages state-of-the-art pre-training

### Accepted Trade-offs
- **Larger model size**: 66M parameters vs 5.9M (Transformer)
- **Slower inference**: Lower throughput for higher accuracy
- **Higher memory usage**: More computational resources required

### Performance Validation
- **Test accuracy**: 97.29% on 850 samples
- **Confidence**: Well-calibrated at 98.38% average
- **Error rate**: Only 2.71% misclassification rate
- **Robust performance**: Strong across multiple intent categories

---

## Reproducibility

### Training Commands
```bash
# Train LSTM
python scripts/train.py

# Train Transformer  
python scripts/train_transformer.py

# Train BERT
python scripts/train_bert.py
```

### Testing Commands
```bash
# Test any trained model
python scripts/test_model.py --model output_models/lstm_model
python scripts/test_model.py --model output_models/transformer_model  
python scripts/test_model.py --model output_models/bert_model
```

### Results Storage
- Detailed reports: `evaluation_results/[model_name]_results/`
- Training logs: Console output with epoch-by-epoch progress
- Model artifacts: Saved in `output_models/[model_name]/`

All experiments are fully reproducible with provided code and configurations.