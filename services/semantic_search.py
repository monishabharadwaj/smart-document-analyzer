"""
Semantic Search Module

Implements natural language queries to retrieve the most relevant documents
using embeddings and vector similarity search with FAISS.
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from .preprocessing import DocumentProcessor

class SemanticSearch:
    """Semantic search using sentence embeddings and FAISS indexing."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', index_type='flat', dimension=384):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name (str): SentenceTransformer model name
            index_type (str): FAISS index type ('flat', 'ivf', 'hnsw')
            dimension (int): Embedding dimension
        """
        self.model_name = model_name
        self.index_type = index_type
        self.dimension = dimension
        
        # Initialize sentence transformer model
        self.encoder = SentenceTransformer(model_name)
        self.actual_dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = self._create_faiss_index()
        
        # Document storage
        self.documents = []  # List of original documents
        self.document_metadata = []  # List of metadata for each document
        self.embeddings = []  # List of embeddings
        
        # Text processor
        self.processor = DocumentProcessor()
        
        # Track if index is trained (for IVF indices)
        self.index_trained = False
    
    def _create_faiss_index(self):
        """Create and return a FAISS index based on the specified type."""
        if self.index_type == 'flat':
            return faiss.IndexFlatIP(self.actual_dimension)  # Inner product for cosine similarity
        elif self.index_type == 'ivf':
            # IVF index for larger datasets
            nlist = 100  # Number of cluster centroids
            quantizer = faiss.IndexFlatIP(self.actual_dimension)
            return faiss.IndexIVFFlat(quantizer, self.actual_dimension, nlist)
        elif self.index_type == 'hnsw':
            # HNSW index for very large datasets
            return faiss.IndexHNSWFlat(self.actual_dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the search index.
        
        Args:
            documents (List[str]): List of document texts
            metadata (List[Dict], optional): List of metadata dicts for each document
        """
        if metadata is None:
            metadata = [{'id': i} for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.encoder.encode(documents, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if necessary (for IVF)
        if self.index_type == 'ivf' and not self.index_trained:
            if len(embeddings) >= 100:  # Need enough data points for training
                self.index.train(embeddings)
                self.index_trained = True
            else:
                print("Not enough documents for IVF training. Consider using 'flat' index type.")
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.document_metadata.extend(metadata)
        self.embeddings.extend(embeddings.tolist())
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            threshold (float): Minimum similarity threshold
            
        Returns:
            List[Dict]: List of search results with scores and metadata
        """
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:  # -1 indicates not found
                result = {
                    'document': self.documents[idx],
                    'metadata': self.document_metadata[idx],
                    'similarity_score': float(score),
                    'index': int(idx)
                }
                results.append(result)
        
        return results
    
    def search_with_filters(self, query: str, filters: Dict, top_k: int = 5) -> List[Dict]:
        """
        Search with metadata filters.
        
        Args:
            query (str): Search query
            filters (Dict): Metadata filters to apply
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: Filtered search results
        """
        # Get all search results
        all_results = self.search(query, top_k * 10)  # Get more to account for filtering
        
        # Apply filters
        filtered_results = []
        for result in all_results:
            metadata = result['metadata']
            matches_filters = True
            
            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    matches_filters = False
                    break
            
            if matches_filters:
                filtered_results.append(result)
                
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results
    
    def add_document_chunks(self, document: str, chunk_size: int = 500, overlap: int = 50, metadata: Dict = None):
        """
        Add a document by splitting it into overlapping chunks.
        
        Args:
            document (str): Full document text
            chunk_size (int): Size of each chunk in characters
            overlap (int): Overlap between chunks in characters
            metadata (Dict): Metadata for the document
        """
        chunks = self._create_chunks(document, chunk_size, overlap)
        
        # Create metadata for each chunk
        chunk_metadata = []
        base_metadata = metadata or {}
        
        for i, chunk in enumerate(chunks):
            chunk_meta = base_metadata.copy()
            chunk_meta.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'is_chunk': True
            })
            chunk_metadata.append(chunk_meta)
        
        self.add_documents(chunks, chunk_metadata)
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from text."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size * 0.5:  # Only if we're not breaking too early
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def get_similar_documents(self, document_text: str, top_k: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document.
        
        Args:
            document_text (str): Document to find similarities for
            top_k (int): Number of similar documents to return
            
        Returns:
            List[Dict]: Similar documents with scores
        """
        return self.search(document_text, top_k)
    
    def save_index(self, filepath: str):
        """
        Save the search index and document data to disk.
        
        Args:
            filepath (str): Path to save the index
        """
        # Save FAISS index
        index_path = f"{filepath}.faiss"
        faiss.write_index(self.index, index_path)
        
        # Save document data
        data_path = f"{filepath}.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'embeddings': self.embeddings,
                'model_name': self.model_name,
                'index_type': self.index_type,
                'dimension': self.actual_dimension,
                'index_trained': self.index_trained
            }, f)
    
    def load_index(self, filepath: str):
        """
        Load a saved search index from disk.
        
        Args:
            filepath (str): Path to load the index from
        """
        # Load FAISS index
        index_path = f"{filepath}.faiss"
        self.index = faiss.read_index(index_path)
        
        # Load document data
        data_path = f"{filepath}.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_metadata = data['document_metadata']
            self.embeddings = data['embeddings']
            self.index_trained = data.get('index_trained', True)
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        return {
            'total_documents': len(self.documents),
            'index_type': self.index_type,
            'embedding_dimension': self.actual_dimension,
            'model_name': self.model_name,
            'index_trained': self.index_trained,
            'memory_usage_mb': self.index.ntotal * self.actual_dimension * 4 / (1024 * 1024)  # Rough estimate
        }
    
    def rerank_results(self, query: str, initial_results: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank initial search results using cross-encoder or other reranking methods.
        
        Args:
            query (str): Original search query
            initial_results (List[Dict]): Initial search results
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Dict]: Reranked results
        """
        # Simple reranking based on exact phrase matches and term frequency
        reranked_results = []
        query_terms = set(query.lower().split())
        
        for result in initial_results:
            document = result['document'].lower()
            
            # Calculate exact phrase matches
            phrase_score = 1.0 if query.lower() in document else 0.0
            
            # Calculate term overlap
            doc_terms = set(document.split())
            term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms)
            
            # Combine scores
            rerank_score = (
                result['similarity_score'] * 0.6 +
                phrase_score * 0.25 +
                term_overlap * 0.15
            )
            
            result['rerank_score'] = rerank_score
            reranked_results.append(result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        return reranked_results

class ChromaDBSearch:
    """Alternative semantic search implementation using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB-based semantic search.
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.processor = DocumentProcessor()
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Add documents to ChromaDB collection.
        
        Args:
            documents (List[str]): List of document texts
            metadata (List[Dict], optional): Metadata for each document
            ids (List[str], optional): IDs for each document
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """
        Search for similar documents in ChromaDB.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            filters (Dict): Metadata filters
            
        Returns:
            List[Dict]: Search results
        """
        where_clause = filters if filters else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                'id': results['ids'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by their IDs."""
        self.collection.delete(ids=ids)
    
    def update_document(self, doc_id: str, document: str, metadata: Dict = None):
        """Update a document by its ID."""
        self.collection.update(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata] if metadata else None
        )
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }

class HybridSearch:
    """Combines multiple search methods for better results."""
    
    def __init__(self, semantic_search: SemanticSearch, use_keyword_boost: bool = True):
        """
        Initialize hybrid search.
        
        Args:
            semantic_search (SemanticSearch): Semantic search instance
            use_keyword_boost (bool): Whether to boost keyword matches
        """
        self.semantic_search = semantic_search
        self.use_keyword_boost = use_keyword_boost
        self.processor = DocumentProcessor()
    
    def search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword-based methods.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            semantic_weight (float): Weight for semantic search (0-1)
            
        Returns:
            List[Dict]: Hybrid search results
        """
        # Get semantic search results
        semantic_results = self.semantic_search.search(query, top_k * 2)
        
        if not self.use_keyword_boost:
            return semantic_results[:top_k]
        
        # Apply keyword boosting
        query_terms = set(query.lower().split())
        hybrid_results = []
        
        for result in semantic_results:
            document = result['document'].lower()
            doc_terms = set(document.split())
            
            # Calculate keyword match score
            keyword_score = len(query_terms.intersection(doc_terms)) / len(query_terms)
            
            # Combine semantic and keyword scores
            hybrid_score = (
                result['similarity_score'] * semantic_weight +
                keyword_score * (1 - semantic_weight)
            )
            
            result['hybrid_score'] = hybrid_score
            result['keyword_score'] = keyword_score
            hybrid_results.append(result)
        
        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_results[:top_k]
