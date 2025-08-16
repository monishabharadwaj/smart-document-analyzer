"""
Smart Document Analyzer & Knowledge Portal

A comprehensive ML/NLP backend system for intelligent document processing,
analysis, and retrieval.

Main Components:
- Document Preprocessing: Text extraction and cleaning from PDF/DOCX files
- Document Classification: Classify documents into types (invoice, report, contract, research paper)
- Named Entity Recognition: Extract entities like Organization, Date, Money, Person
- Document Summarization: Generate concise summaries using transformer models
- Semantic Search: Natural language queries with vector similarity search
"""

__version__ = "1.0.0"
__author__ = "Smart Document Analyzer Team"
__email__ = "contact@smartdocanalyzer.com"

# Import main services for easy access
from services import (
    DocumentProcessor,
    DocumentClassifier,
    NERService,
    DocumentSummarizer,
    SemanticSearch
)

__all__ = [
    'DocumentProcessor',
    'DocumentClassifier',
    'NERService', 
    'DocumentSummarizer',
    'SemanticSearch'
]
