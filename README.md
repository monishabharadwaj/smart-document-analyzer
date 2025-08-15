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

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download SpaCy models:
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```
4. Download NLTK data (run once):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Quick Start

### Document Preprocessing
```python
from services.preprocessing import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text_from_pdf("document.pdf")
cleaned_text = processor.clean_text(text)
```

### Document Classification
```python
from services.classification import DocumentClassifier

classifier = DocumentClassifier()
doc_type = classifier.predict("Your document text here")
print(f"Document type: {doc_type}")
```

### Named Entity Recognition
```python
from services.ner import NERService

ner = NERService()
entities = ner.extract_entities("Apple Inc. was founded in 1976.")
print(entities)
```

### Document Summarization
```python
from services.summarization import DocumentSummarizer

summarizer = DocumentSummarizer()
summary = summarizer.summarize("Long document text here...")
print(summary)
```

### Semantic Search
```python
from services.semantic_search import SemanticSearch

search_engine = SemanticSearch()
search_engine.add_documents(["doc1 text", "doc2 text"])
results = search_engine.search("find relevant documents", top_k=5)
```

## Training Models

### Classification Model
```bash
python scripts/train_classifier.py --data_path data/classification_data.csv
```

### NER Model
```bash
python scripts/train_ner.py --data_path data/ner_training_data.json
```

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- File paths
- API settings
- Processing options

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request
