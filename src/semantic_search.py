#!/usr/bin/env python3
"""
Semantic Search Engine
=====================

This module implements semantic search functionality using sentence transformers 
and FAISS vector database for efficient similarity search.

Features:
- Document embedding generation using sentence transformers
- FAISS index building for fast vector similarity search
- Query processing and result ranking
- Multi-document search capabilities
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """
    A semantic search engine using sentence transformers and FAISS.
    
    This class provides functionality to:
    - Build vector embeddings for documents
    - Create and maintain FAISS indices
    - Perform semantic similarity searches
    - Manage document metadata
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index and document storage
        self.index = None
        self.documents = {}
        self.document_ids = []
        self.is_built = False
        
        logger.info(f"Semantic search engine initialized with {self.dimension}D embeddings")
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the FAISS index from a list of documents.
        
        Args:
            documents: List of document dictionaries with keys like 'id', 'content', etc.
        """
        logger.info(f"Building search index for {len(documents)} documents...")
        
        # Extract document content for embedding
        texts = []
        doc_ids = []
        
        for doc in documents:
            # Combine title and content for better embedding
            text = doc.get('title', '') + ' ' + doc.get('content', '')
            texts.append(text.strip())
            doc_ids.append(doc['id'])
            
            # Store document metadata
            self.documents[doc['id']] = doc
        
        # Generate embeddings
        logger.info("Generating document embeddings...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        logger.info(f"Creating FAISS index with {self.dimension} dimensions...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        self.document_ids = doc_ids
        self.is_built = True
        
        logger.info(f"Search index built successfully! {len(documents)} documents indexed.")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (document_id, similarity_score)
        """
        if not self.is_built:
            raise ValueError("Search index not built. Call build_index() first.")
        
        # Encode the query
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_ids):  # Valid index
                doc_id = self.document_ids[idx]
                results.append((doc_id, float(score)))
        
        return results
    
    def search_with_metadata(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents and return full metadata.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing document metadata and scores
        """
        results = self.search(query, top_k)
        
        enriched_results = []
        for doc_id, score in results:
            doc = self.documents[doc_id].copy()
            doc['search_score'] = score
            enriched_results.append(doc)
        
        return enriched_results
    
    def add_documents(self, new_documents: List[Dict[str, Any]]) -> None:
        """
        Add new documents to the existing index (requires rebuilding).
        
        Args:
            new_documents: List of new document dictionaries
        """
        # Combine existing and new documents
        all_documents = list(self.documents.values()) + new_documents
        
        # Rebuild the index
        self.build_index(all_documents)
    
    def remove_documents(self, doc_ids: List[str]) -> None:
        """
        Remove documents from the index (requires rebuilding).
        
        Args:
            doc_ids: List of document IDs to remove
        """
        # Filter out removed documents
        remaining_documents = [
            doc for doc in self.documents.values() 
            if doc['id'] not in doc_ids
        ]
        
        # Rebuild the index
        self.build_index(remaining_documents)
    
    def save_index(self, filepath: str) -> None:
        """
        Save the search index and metadata to disk.
        
        Args:
            filepath: Path to save the index files
        """
        if not self.is_built:
            raise ValueError("No index to save. Build index first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'documents': self.documents,
            'document_ids': self.document_ids,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Search index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load a previously saved search index.
        
        Args:
            filepath: Path to the saved index files
        """
        filepath = Path(filepath)
        
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        self.documents = metadata['documents']
        self.document_ids = metadata['document_ids']
        self.is_built = True
        
        logger.info(f"Search index loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dictionary containing index statistics
        """
        if not self.is_built:
            return {'status': 'not_built'}
        
        return {
            'status': 'built',
            'total_documents': len(self.document_ids),
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'device': self.device,
            'categories': self._get_category_distribution()
        }
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of document categories."""
        categories = {}
        for doc in self.documents.values():
            category = doc.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories

# Standalone functions for compatibility
def create_document_embeddings(documents: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Create embeddings for a list of documents.
    
    Args:
        documents: List of document strings
        model_name: Name of the sentence transformer model
        
    Returns:
        NumPy array of embeddings
    """
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(documents, show_progress_bar=True)
    return np.array(embeddings).astype('float32')

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: NumPy array of document embeddings
        
    Returns:
        FAISS index object
    """
    dimension = embeddings.shape[1]
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create and populate index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index

def search_similar_documents(
    query: str, 
    index: faiss.Index, 
    encoder: SentenceTransformer, 
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for similar documents using a query.
    
    Args:
        query: Search query string
        index: FAISS index
        encoder: Sentence transformer model
        top_k: Number of results to return
        
    Returns:
        Tuple of (scores, indices) arrays
    """
    # Encode query
    query_embedding = encoder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Normalize query
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    return scores, indices

# Example usage and testing functions
def create_sample_documents():
    """Create sample documents for testing."""
    return [
        {
            'id': 'doc1',
            'title': 'Introduction to Machine Learning',
            'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.',
            'category': 'technology'
        },
        {
            'id': 'doc2',
            'title': 'Climate Change Effects',
            'content': 'Global warming is causing significant changes to weather patterns and ecosystem dynamics worldwide.',
            'category': 'environment'
        },
        {
            'id': 'doc3',
            'title': 'Financial Market Analysis',
            'content': 'Stock market volatility has increased due to economic uncertainty and geopolitical tensions.',
            'category': 'finance'
        }
    ]

def demo_search_engine():
    """Demonstrate the semantic search engine functionality."""
    # Create search engine
    engine = SemanticSearchEngine()
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Build index
    engine.build_index(documents)
    
    # Test searches
    test_queries = [
        "artificial intelligence algorithms",
        "environmental climate issues",
        "stock market economics"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = engine.search(query, top_k=2)
        
        for doc_id, score in results:
            doc = engine.documents[doc_id]
            print(f"  {doc['title']} (Score: {score:.3f})")

if __name__ == "__main__":
    # Run demo
    demo_search_engine()
