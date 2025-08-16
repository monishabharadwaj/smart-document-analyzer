# Smart Document Analyzer

A comprehensive AI-powered document processing system with machine learning, NLP capabilities, and a full REST API backend. Built with Python, spaCy, scikit-learn, and FastAPI.

## Overview

This system provides intelligent document analysis through:

- **Document Processing**: Automatic text extraction, cleaning, and preprocessing
- **Entity Recognition**: Extract people, organizations, locations, and other entities
- **Document Classification**: Automatically categorize documents by type
- **Smart Summarization**: Generate extractive summaries and bulylet points
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

##  Core Components

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

##  Technology Stack

- **Backend Framework**: FastAPI
- **NLP Library**: spaCy
- **ML Framework**: scikit-learn
- **Vector Search**: FAISS
- **Embeddings**: Sentence Transformers
- **API Documentation**: OpenAPI/Swagger
- **Testing**: pytest

## 📈 Performance Features

- **Batch Processing**: Handle multiple documents efficiently
- **Caching**: Smart caching for repeated operations
- **Async Processing**: Non-blocking API operations
- **Vector Indexing**: Fast similarity search with FAISS
- **Memory Management**: Optimized for large document collections

## Production Ready

- ✅ **Error Handling**: Comprehensive exception management
- ✅ **Input Validation**: Pydantic models for request validation
- ✅ **CORS Support**: Cross-origin resource sharing
- ✅ **Health Checks**: System monitoring endpoints
- ✅ **Logging**: Structured logging throughout
- ✅ **Documentation**: Complete API documentation

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

##  Development

This project was built as a complete end-to-end document analysis system, demonstrating:
- Modular architecture design
- Machine learning pipeline development
- REST API design and implementation
- Comprehensive testing strategies
- Production-ready code practices

---

**The Smart Document Analyzer effectively integrates ML, NLP, and web development to provide a complete document processing solution. By leveraging modern ML models like BERT, spaCy, and transformer-based summarizers, it offers high accuracy, efficiency, and usability. Users can upload documents, receive classified results, extract key entities, get summaries, and perform semantic searches—all through a clean, responsive web interface.**

**Built with  using Python, FastAPI, and modern NLP libraries**
