"""
Smart Document Analyzer - Services Package

This package contains the core ML/NLP services for document processing:
- preprocessing: Document text extraction and cleaning
- classification: Document type classification
- ner: Named Entity Recognition
- summarization: Document summarization
- semantic_search: Vector-based document search
"""

__version__ = "1.0.0"
__author__ = "Smart Document Analyzer Team"

from .preprocessing import DocumentProcessor
from .classification import DocumentClassifier
from .ner import NERService
from .summarization import DocumentSummarizer
from .semantic_search import SemanticSearch

__all__ = [
    'DocumentProcessor',
    'DocumentClassifier', 
    'NERService',
    'DocumentSummarizer',
    'SemanticSearch'
]
