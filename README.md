# Smart Document Analyzer & Knowledge Portal

A comprehensive ML/NLP backend system for intelligent document processing, analysis, and retrieval.

## Features

- **Document Classification**: Automatically classify documents into types (invoice, report, contract, research paper)
- **Named Entity Recognition**: Extract entities like Organization, Date, Money, Person from documents
- **Document Summarization**: Generate concise summaries using transformer-based models
- **Semantic Search**: Natural language queries with vector similarity search using FAISS
- **Document Preprocessing**: Extract and clean text from PDF/DOCX files

## Project Structure

```
smart-document-analyzer/
├── services/           # Core ML/NLP service modules
│   ├── __init__.py
│   ├── preprocessing.py    # Document text extraction and cleaning
│   ├── classification.py   # Document type classification
│   ├── ner.py             # Named Entity Recognition
│   ├── summarization.py   # Document summarization
│   └── semantic_search.py # Vector-based document search
├── scripts/            # Training and inference scripts
│   ├── train_classifier.py
│   ├── train_ner.py
│   ├── inference_examples.py
│   └── data_preparation.py
├── models/             # Saved model files and configurations
├── data/               # Training data and sample documents
├── config/             # Configuration files
│   ├── config.yaml
│   └── logging.yaml
├── tests/              # Unit tests
└── requirements.txt
```

