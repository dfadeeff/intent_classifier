# Intent Classification Service

An intent classification system with multiple model architectures and HTTP API service. This project demonstrates modern ML engineering practices with comprehensive model comparison and deployment capabilities.

## 🚀 Overview

This repository implements a scalable intent classification service that can be used to provide inference via HTTP API. The system supports multiple neural network architectures (LSTM, Transformer, BERT) with easy model swapping and comprehensive evaluation framework.

## ✨ Features

- **Multiple Model Architectures**: LSTM, Enhanced Transformer, and BERT classifiers
- **API**: RESTful HTTP service with comprehensive error handling
- **Model Comparison Framework**: Systematic evaluation and benchmarking tools
- **Modular Design**: Easy integration of new model types
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and error analysis

## 🏗️ Architecture

### Model Hierarchy
| Model | Parameters | Test Accuracy | Best Use Case |
|-------|------------|---------------|---------------|
| LSTM | 2.1m       | 95.65% | Real-time, high-throughput |
| Transformer | 5.9M       | 96.47% | Balanced performance |
| **BERT** | 66M        | **97.29%** | Maximum accuracy |

### System Components
```
├── src/
│   ├── models/             # Model architectures (LSTM, Transformer, BERT)
│   ├── data/               # Data processing and loading
│   ├── training/           # Training framework and utilities
│   └── evaluation/         # Evaluation and visualization tools
├── scripts/                # Training and testing scripts
├── data/                   # ATIS dataset
├── intent_classifier.py    # Inference engine 
└── server.py               # HTTP API service
```

## 🔧 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/intent-classification-service
cd intent-classification-service
pip install -r requirements.txt
```

### Train Models
```bash
# Train LSTM (fast, efficient)
python scripts/train_lstm.py

# Train Enhanced Transformer (balanced)
python scripts/train_transformer.py

# Train BERT (highest accuracy)
python scripts/train_bert.py

# Trained models will be saved under output_models/

output_models/
  bert_model/
  lstm_model/
  transformer_model/
  
# Once the model have been trained, you can test them and check the inference service

```

### Start API Service
```bash

# Open one terminal and start service for ONE of the models:

# Use best performing model (BERT)
python server.py --model output_models/bert_model --port 8080

# Or specify different model: Transformer
python server.py --model output_models/transformer_model --port 8080

# Or specify different model: LSTM
python server.py --model output_models/lstm_model --port 8080

```

### Test API
```bash

# Open a second terminal and test the model that you have specified in the first terminal 

# Check service health
curl http://localhost:8080/ready

# Classify intent
curl -X POST http://localhost:8080/intent \
  -H 'Content-Type: application/json' \
  -d '{"text": "find me a flight to boston"}' | python -m json.tool
  
```

### Expected Response 
```bash

{
    "intents": [
        {
            "label": "flight",
            "confidence": 1.0
        },
        {
            "label": "flight+airfare",
            "confidence": 0.0
        },
        {
            "label": "airport",
            "confidence": 0.0
        }
    ]
}

```

## 🔌 API Documentation

The service provides comprehensive API documentation through Swagger/OpenAPI integration.

### Interactive Documentation
Access the interactive API documentation at: `http://localhost:8080/docs`

### Health Check
```
GET /ready
```
Returns `200 OK` when service is ready, `423` when loading.

### Intent Classification
```
POST /intent
Content-Type: application/json

{
  "text": "find me a flight that flies from Memphis to tacoma"
}
```

**Response:**
```json
{
  "intents": [{
    "label": "flight",
    "confidence": 0.973
  }, {
    "label": "aircraft", 
    "confidence": 0.018
  }, {
    "label": "airline",
    "confidence": 0.007
  }]
}
```

### Error Handling
- **400**: Empty text input
- **500**: Internal server errors
- Comprehensive error messages with proper HTTP status codes



## 📊 Model Performance

### Test Results (ATIS Dataset, 850 samples)

#### BERT Classifier (Production Model)
```
🎯 Test Accuracy: 97.29%
📊 Average Confidence: 98.38%
✅ Correct: 827/850
❌ Misclassified: 23/850
```

#### Enhanced Transformer
```
🎯 Test Accuracy: 96.47%
📊 Average Confidence: 99.16%
✅ Correct: 820/850
❌ Misclassified: 30/850
```

#### LSTM Baseline
```
🎯 Test Accuracy: 95.65%
📊 Average Confidence: 99.11%
✅ Correct: 813/850
❌ Misclassified: 37/850
```

## 🧪 Experimentation Framework

### Model Comparison
Run comprehensive model evaluation:
```bash
# Test all models
python scripts/test_model.py --model output_models/lstm_model
python scripts/test_model.py --model output_models/transformer_model
python scripts/test_model.py --model output_models/bert_model
```

### Evaluation Results
- **Detailed metrics**: Precision, recall, F1-score per class
- **Error analysis**: Misclassification examples with confidence scores
- **Performance comparison**: Speed vs accuracy trade-offs
- **Visualization**: Confusion matrices and performance plots


## 🏛️ Model Architectures

### LSTM Classifier
- **Design**: Bidirectional LSTM with attention mechanism
- **Parameters**: ~2.1m
- **Features**: Fast inference, compact model size
- **Use case**: High-throughput production environments

### Enhanced Transformer
- **Design**: Multi-head self-attention with positional encoding
- **Parameters**: ~5.9M 
- **Features**: Attention pooling, GELU activation, pre-norm architecture
- **Use case**: Balanced speed/accuracy requirements

### BERT Classifier
- **Design**: Fine-tuned DistilBERT with custom classification head
- **Parameters**: ~66M 
- **Features**: Pre-trained language understanding, transfer learning
- **Use case**: Maximum accuracy requirements

## 🔬 Experimental Results

### Key Findings

1. **Pre-training advantage**: Transfer learning provides consistent improvements
2. **Trade-off sweet spots**: Transformer offers best balance of speed/accuracy
3. **Production readiness**: All models show high confidence and stable performance

### Error Analysis
- **Multi-intent confusion**: Models struggle with compound intents (`flight+airfare`)
- **Rare class issues**: Limited training data for minority classes (`day_name`)
- **Context sensitivity**: BERT handles complex sentence structures better

## 🚀 Production Features

### Modular Design
- **Easy model swapping**: Change models with single parameter
- **Consistent interface**: All models use same API contract
- **Extensible architecture**: Add new models with minimal code changes

### Robustness
- **Error handling**: Comprehensive exception management
- **Input validation**: Text preprocessing and sanitization
- **Resource management**: Efficient memory usage and model loading

### Monitoring
- **Health checks**: Service status and model readiness
- **Confidence scores**: Model uncertainty quantification
- **Performance metrics**: Detailed evaluation framework

## 📁 Project Structure

```
intent-classification-service/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── server.py                        # HTTP API service
├── intent_classifier.py             # Main classifier interface
├── src/
│   ├── models/
│   │   ├── base_model.py            # Abstract base class
│   │   ├── lstm_model.py            # LSTM implementation
│   │   ├── transformer_model.py     # Transformer implementation
│   │   └── bert_model.py            # BERT implementation
│   ├── data/
│   │   ├── dataset.py               # Dataset classes
│   │   └── preprocessing.py         # Text preprocessing
│   ├── training/
│   │   ├── trainer.py               # Training framework
│   │   └── utils.py                 # Training utilities
│   └── evaluation/
│       ├── evaluator.py             # Model evaluation
│       └── visualizer.py            # Performance visualization
├── scripts/
│   ├── train.py                     # LSTM training
│   ├── train_transformer.py         # Transformer training
│   ├── train_bert.py                # BERT training
│   └── test_model.py                # Model testing
├── data/
│   └── atis/                        # ATIS dataset
└── docs/
    └── experiments.md               # Detailed experimental results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ATIS dataset for training and evaluation
- PyTorch community for deep learning framework

---
