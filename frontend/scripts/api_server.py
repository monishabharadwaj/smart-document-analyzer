"""
FastAPI server for Smart Document Analyzer services.

This provides REST API endpoints for all the document processing services.
"""

import os
import sys
import logging
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import yaml

# Add parent directory to path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing import DocumentProcessor
from services.classification import DocumentClassifier
from services.ner import NERService
from services.summarization import DocumentSummarizer
from services.semantic_search import SemanticSearch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Document Analyzer API",
    description="REST API for document processing, classification, NER, summarization, and semantic search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str

class ClassificationRequest(BaseModel):
    text: str
    model_type: str = "traditional"

class SummarizationRequest(BaseModel):
    text: str
    model_type: str = "extractive"
    summary_ratio: float = 0.3
    max_length: Optional[int] = None
    min_length: Optional[int] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.0

class DocumentsRequest(BaseModel):
    documents: List[str]
    metadata: Optional[List[Dict]] = None

# Global service instances (initialize on startup)
processor = None
classifier = None
ner_service = None
summarizer = None
search_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global processor, classifier, ner_service, summarizer, search_engine
    
    logger.info("Initializing Smart Document Analyzer services...")
    
    try:
        processor = DocumentProcessor()
        classifier = DocumentClassifier(model_type='traditional')
        ner_service = NERService()
        summarizer = DocumentSummarizer(model_type='extractive')
        search_engine = SemanticSearch()
        
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Smart Document Analyzer API",
        "version": "1.0.0",
        "services": [
            "preprocessing",
            "classification", 
            "ner",
            "summarization",
            "semantic_search"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "services_loaded": True}

# Preprocessing endpoints
@app.post("/preprocess/clean")
async def clean_text(request: TextInput):
    """Clean and preprocess text."""
    try:
        cleaned_text = processor.clean_text(request.text)
        return {"cleaned_text": cleaned_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/extract-pdf")
async def extract_pdf_text(file: UploadFile = File(...)):
    """Extract text from uploaded PDF file."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract text
        text = processor.extract_text_from_pdf(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {"extracted_text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Classification endpoints
@app.post("/classify/predict")
async def classify_document(request: ClassificationRequest):
    """Classify document type."""
    try:
        # Note: This requires a trained model to work properly
        prediction = "unknown"  # Default response for untrained model
        confidence = 0.0
        
        return {
            "document_type": prediction,
            "confidence": confidence,
            "note": "This requires a trained classification model. Train one using train_classifier.py"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NER endpoints
@app.post("/ner/extract")
async def extract_entities(request: TextInput):
    """Extract named entities from text."""
    try:
        entities = ner_service.extract_entities(request.text)
        financial_entities = ner_service.extract_financial_entities(request.text)
        
        return {
            "general_entities": entities,
            "financial_entities": financial_entities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ner/summary")
async def get_entity_summary(request: TextInput):
    """Get entity extraction summary."""
    try:
        summary = ner_service.get_entity_summary(request.text)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarization endpoints
@app.post("/summarize/extract")
async def summarize_extractive(request: SummarizationRequest):
    """Generate extractive summary."""
    try:
        summary = summarizer.summarize(
            request.text, 
            summary_ratio=request.summary_ratio
        )
        
        # Get statistics
        stats = summarizer.get_summary_statistics(request.text, summary)
        
        return {
            "summary": summary,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/bullets")
async def generate_bullet_summary(request: TextInput, num_bullets: int = 5):
    """Generate bullet point summary."""
    try:
        bullets = summarizer.generate_bullet_summary(request.text, num_bullets)
        return {"bullet_points": bullets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Semantic search endpoints
@app.post("/search/add-documents")
async def add_documents_to_search(request: DocumentsRequest):
    """Add documents to search index."""
    try:
        search_engine.add_documents(request.documents, request.metadata)
        stats = search_engine.get_index_stats()
        return {
            "message": f"Added {len(request.documents)} documents",
            "index_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/query")
async def search_documents(request: SearchRequest):
    """Search documents using semantic similarity."""
    try:
        results = search_engine.search(
            request.query, 
            top_k=request.top_k,
            threshold=request.threshold
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/stats")
async def get_search_stats():
    """Get search index statistics."""
    try:
        stats = search_engine.get_index_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Combined processing endpoint
@app.post("/process/complete")
async def complete_document_processing(request: TextInput):
    """Complete document processing pipeline."""
    try:
        text = request.text
        
        # Clean text
        cleaned_text = processor.clean_text(text)
        
        # Extract entities
        entities = ner_service.extract_entities(text)
        
        # Generate summary
        summary = summarizer.summarize(cleaned_text, summary_ratio=0.3)
        
        # Get statistics
        stats = summarizer.get_summary_statistics(text, summary)
        
        return {
            "original_text_length": len(text.split()),
            "cleaned_text": cleaned_text,
            "entities": entities,
            "summary": summary,
            "statistics": stats,
            "processing_note": "Classification requires trained model"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
