# System Architecture

## Overview
This document describes the architecture of the Intent Classification service, designed for production deployment with support for multiple machine learning model types.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   HTTP Client   │────│  Flask Server   │────│ Intent Classifier│
│  (API Consumer) │    │   (server.py)   │    │     Interface    │
└─────────────────┘    └─────────────────┘    └──────────────────┘
                                │                     │
                                │                     ▼
                        ┌───────────────┐     ┌──────────────────┐
                        │ Error Handler │     │   Model Layer    │
                        │   & Validator │     │ (LSTM/Trans/BERT)│
                        └───────────────┘     └──────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────┐
                                              │ Saved Models    │
                                              │ & Artifacts     │
                                              └─────────────────┘
```

## Core Components

### 1. Flask Server (`server.py`)
**Responsibilities:**
- HTTP request handling and routing
- Input validation and sanitization
- Response formatting and error handling
- Model lifecycle management

**Key Features:**
- RESTful API endpoints (`/ready`, `/intent`)
- Comprehensive error handling with proper HTTP status codes
- JSON request/response validation
- Health check capabilities

**Design Patterns:**
- Single Responsibility: Each endpoint has one clear purpose
- Dependency Injection: Model instance passed to handlers
- Error First: Validate inputs before processing

### 2. Intent Classifier Interface (`intent_classifier.py`)
**Responsibilities:**
- Unified interface for multiple model types
- Model loading and initialization
- Prediction orchestration
- Text preprocessing and postprocessing

**Key Features:**
- **Auto-detection**: Automatically identifies model type from files
- **Polymorphism**: Same interface for LSTM, Transformer, and BERT
- **Graceful degradation**: Proper error handling and status reporting

**Design Patterns:**
- Strategy Pattern: Different prediction strategies for each model type
- Factory Pattern: Model instantiation based on detected type
- Template Method: Common prediction workflow with model-specific steps

### 3. Model Layer (`src/models/`)
**Architecture:**
```
BaseTextClassifier (Abstract Base Class)
    ├── LSTMClassifier
    ├── TransformerClassifier  
    └── BERTClassifier
```

**Common Interface:**
- `forward()`: Model prediction
- `get_model_info()`: Model metadata
- Consistent input/output format

**Model-Specific Features:**
- **LSTM**: Bidirectional processing, sequence length handling
- **Transformer**: Self-attention, positional encoding
- **BERT**: Pre-trained tokenizer, fine-tuning capabilities

## Data Flow

### Request Processing Flow
```
1. HTTP Request → Flask Router
2. Content-Type Validation
3. JSON Schema Validation  
4. Text Input Sanitization
5. Model Readiness Check
6. Prediction Generation
7. Response Formatting
8. HTTP Response
```

### Model Loading Flow
```
1. Model Path Detection
2. File Structure Analysis
3. Model Type Identification
4. Component Loading (vocab/tokenizer/weights)
5. Model Initialization
6. Device Assignment (CPU/GPU)
7. Readiness Flag Setting
```

### Prediction Flow
```
1. Text Input
2. Model-Specific Preprocessing
   ├─ LSTM/Transformer: text_to_indices()
   └─ BERT: tokenizer.encode()
3. Tensor Conversion
4. Model Forward Pass
5. Softmax Probability Calculation
6. Top-K Selection (K=3)
7. Response Formatting
```

## Design Principles

### 1. Modularity
- **Separation of Concerns**: Clear boundaries between web layer, business logic, and model layer
- **Pluggable Architecture**: Easy to add new model types
- **Interface Standardization**: Consistent APIs across components

### 2. Robustness
- **Input Validation**: Multiple layers of validation (HTTP, JSON, business logic)
- **Error Handling**: Graceful degradation with informative error messages
- **Resource Management**: Proper device handling and memory management

### 3. Maintainability
- **Auto-Detection**: Reduces configuration complexity
- **Type Safety**: Clear interfaces and error checking
- **Documentation**: Comprehensive docstrings and API documentation

### 4. Performance
- **Lazy Loading**: Models loaded only when needed
- **Device Optimization**: Automatic CPU/GPU detection
- **Efficient Inference**: Batch processing and optimized forward passes

## Configuration Management

### Model Configuration
Models are auto-detected based on file structure:
```
# BERT Model Structure
model_dir/
├── tokenizer/         # ← Indicates BERT
├── model.pt           # ← Model weights
└── label_encoder.pkl  # ← Label mapping

# LSTM/Transformer Structure  
model_dir/
├── vocab.pkl          # ← Indicates LSTM/Transformer
├── model.pt           # ← Model weights
└── label_encoder.pkl  # ← Label mapping
```

### Runtime Configuration
- **Server Port**: Configurable via command line (`--port`)
- **Model Path**: Configurable via command line (`--model`)
- **Device Selection**: Automatic CPU/GPU detection

## Error Handling Strategy

### Input Validation Errors (4xx)
- `400 TEXT_EMPTY`: Empty input text
- `400 MISSING_TEXT_FIELD`: Missing required field
- `400 INVALID_CONTENT_TYPE`: Wrong content type

### Service Errors (5xx)
- `500 INTERNAL_ERROR`: Model prediction failures
- `503 MODEL_NOT_READY`: Model loading issues

### Error Response Format
```json
{
  "label": "ERROR_CODE",
  "message": "Human-readable description"
}
```

## Security Considerations

### Input Sanitization
- Text input validation and cleaning
- JSON schema validation
- Content-type enforcement

### Resource Protection
- Model loading validation
- Memory usage monitoring
- Request size limits (implicit via Flask)

## Scalability Considerations

### Current Architecture
- Single-threaded Flask server
- In-memory model loading
- Synchronous request processing

### Production Enhancements
- **Load Balancing**: Multiple server instances
- **Model Caching**: Shared model storage
- **Async Processing**: Queue-based inference
- **Monitoring**: Health checks and metrics

## Testing Strategy

### Unit Tests
- Model interface compliance
- Prediction accuracy validation
- Error handling verification

### Integration Tests  
- End-to-end API testing
- Model loading scenarios
- Error case validation

### Performance Tests
- Inference speed benchmarking
- Memory usage profiling
- Concurrent request handling

## Deployment Architecture

### Development
```
Local Machine
├── Flask Development Server
├── Model Files (local disk)
└── In-memory processing
```

### Production (Recommended)
```
Container Environment
├── WSGI Server (Gunicorn)
├── Reverse Proxy (Nginx)  
├── Model Storage (S3/NFS)
├── Load Balancer
└── Monitoring & Logging
```

## Future Enhancements

### Technical Improvements
1. **Async Support**: FastAPI migration for better concurrency
2. **Model Versioning**: A/B testing and gradual rollouts
3. **Caching Layer**: Redis for frequently requested predictions
4. **Batch Processing**: Multiple predictions per request

### Operational Improvements
1. **Health Metrics**: Detailed health and performance monitoring
2. **Configuration Management**: External config files
3. **Logging**: Structured logging with request tracing
4. **Documentation**: Interactive API documentation with Swagger UI

This architecture provides a solid foundation for production intent classification services while maintaining flexibility for future enhancements.