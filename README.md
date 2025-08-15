# Smart Document Analyzer

A comprehensive AI-powered document processing system with machine learning, NLP capabilities, and a full REST API backend. Built with Python, spaCy, scikit-learn, and FastAPI.

## Overview

This system provides intelligent document analysis through:

- **Document Processing**: Automatic text extraction, cleaning, and preprocessing
- **Entity Recognition**: Extract people, organizations, locations, and other entities
- **Document Classification**: Automatically categorize documents by type
- **Smart Summarization**: Generate extractive summaries and bullet points
- **Semantic Search**: AI-powered document search with vector embeddings
- **Batch Processing**: Handle multiple documents efficiently
- **REST API**: Complete web API with interactive documentation

## System Architecture

```
smart-document-analyzer/
├── Step1_DocumentProcessor/     # Text preprocessing & statistics
├── Step2_EntityExtractor/       # Named Entity Recognition (spaCy)
├── Step3_DocumentClassifier/    # ML-based document classification
├── Step4_TextSummarizer/        # Extractive summarization
├── Step5_SemanticSearch/        # Vector-based search engine
├── Dataset1/ & Dataset2/        # Sample data for testing
├── integration_demo.py          # Main system orchestrator
├── api_server.py               # FastAPI REST API server
├── quick_start.py              # Interactive menu system
├── integration_test.py         # Comprehensive testing
└── requirements.txt            # Dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Start the System
```bash
# Interactive menu (recommended)
python quick_start.py

# Or directly start API server
python api_server.py
```

### 3. Access the API
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **API Documentation**: http://localhost:8000/redoc

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze/document` | Full document analysis (all features) |
| POST | `/extract/entities` | Named entity extraction only |
| POST | `/classify/document` | Document type classification |
| POST | `/summarize/text` | Text summarization |
| POST | `/search/documents` | Semantic document search |
| POST | `/batch/analyze` | Process multiple documents |
| POST | `/upload/document` | Upload and analyze file |
| GET | `/collection/insights` | Analytics and statistics |
| GET | `/health` | System health check |

## Core Components

### 1. Document Processor
- Text cleaning and normalization
- Statistical analysis (word count, readability)
- Language detection

### 2. Entity Extractor
- **Technology**: spaCy NLP library
- **Entities**: PERSON, ORG, GPE, MONEY, DATE, TIME
- **Features**: Confidence scoring, entity linking

### 3. Document Classifier
- **Algorithm**: TF-IDF + Logistic Regression
- **Categories**: News, Research, Business, Technical, Legal
- **Features**: Confidence scores, feature importance

### 4. Text Summarizer
- **Extractive Summarization**: Select key sentences
- **Bullet Point Generation**: Structured summaries
- **Configurable**: Adjustable summary length

### 5. Semantic Search Engine
- **Technology**: Sentence Transformers + FAISS
- **Features**: Vector similarity search
- **Capabilities**: Natural language queries

## Example Usage

### Analyze a Document
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze/document",
    json={
        "text": "Your document text here...",
        "doc_id": "my_document"
    }
)

result = response.json()
print(f"Classification: {result['data']['classification']}")
print(f"Entities: {result['data']['entities']}")
```

### Search Documents
```python
response = requests.post(
    "http://localhost:8000/search/documents",
    json={
        "query": "artificial intelligence research",
        "top_k": 5
    }
)
```

## Technology Stack

- **Backend Framework**: FastAPI
- **NLP Library**: spaCy
- **ML Framework**: scikit-learn
- **Vector Search**: FAISS
- **Embeddings**: Sentence Transformers
- **API Documentation**: OpenAPI/Swagger
- **Testing**: pytest

## Performance Features

- **Batch Processing**: Handle multiple documents efficiently
- **Caching**: Smart caching for repeated operations
- **Async Processing**: Non-blocking API operations
- **Vector Indexing**: Fast similarity search with FAISS
- **Memory Management**: Optimized for large document collections

## Production Ready Features

- **Error Handling**: Comprehensive exception management
- **Input Validation**: Pydantic models for request validation
- **CORS Support**: Cross-origin resource sharing
- **Health Checks**: System monitoring endpoints
- **Logging**: Structured logging throughout
- **Documentation**: Complete API documentation

## Testing

```bash
# Run integration tests
python integration_test.py

# Test individual components
python quick_start.py  # Select option 2 for demo
```

## Dependencies

- `fastapi`: Web framework for building APIs
- `uvicorn`: ASGI server for FastAPI
- `spacy`: Industrial-strength NLP
- `scikit-learn`: Machine learning library
- `sentence-transformers`: Semantic text embeddings
- `faiss-cpu`: Efficient similarity search
- `nltk`: Natural language toolkit
- `pydantic`: Data validation using Python type hints

## Use Cases

- **Document Management Systems**: Automatic categorization and search
- **Content Analysis**: Extract insights from text documents
- **Research Tools**: Summarize and search academic papers
- **Legal Tech**: Process legal documents and contracts
- **Business Intelligence**: Analyze reports and communications
- **Knowledge Bases**: Build searchable document repositories

## Future Enhancements

- PDF/DOCX file processing
- Multi-language support
- Custom model training interface
- Advanced summarization models
- Real-time document processing
- Integration with cloud storage

## Development

This project was built as a complete end-to-end document analysis system, demonstrating:
- Modular architecture design
- Machine learning pipeline development
- REST API design and implementation
- Comprehensive testing strategies
- Production-ready code practices

---

**Built using Python, FastAPI, and modern NLP libraries**
