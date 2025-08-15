#!/usr/bin/env python3
"""
Step 8: API Server Development
==============================

FastAPI-based REST API server providing endpoints for all document analysis services:
- Document upload and processing
- Entity extraction
- Document classification
- Text summarization
- Semantic search
- Batch processing

This server provides a complete web API interface for the smart document analyzer.
"""

import sys
import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
import tempfile
import asyncio

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the integrated analyzer
from integration_demo import IntegratedDocumentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Smart Document Analyzer API",
    description="AI-powered document analysis with NLP, classification, and semantic search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer = None

# Pydantic models for request/response
class DocumentAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text content to analyze")
    doc_id: Optional[str] = Field(None, description="Optional document identifier")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Optional metadata")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)

class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    ratio: float = Field(0.3, description="Summary ratio (0.1-0.9)", ge=0.1, le=0.9)
    summary_type: str = Field("extractive", description="Type of summary")

class BatchProcessRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to process")

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str
    processing_time_ms: Optional[float] = None

def create_response(success: bool, message: str, data: Any = None, processing_time: float = None) -> APIResponse:
    """Create a standardized API response."""
    return APIResponse(
        success=success,
        message=message,
        data=data,
        timestamp=datetime.now().isoformat(),
        processing_time_ms=round(processing_time * 1000, 2) if processing_time else None
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the document analyzer on startup."""
    global analyzer
    print("🚀 Starting Smart Document Analyzer API...")
    try:
        analyzer = IntegratedDocumentAnalyzer()
        print("✅ API server ready!")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down Smart Document Analyzer API...")

@app.get("/", tags=["Health"])
async def root():
    """API health check endpoint."""
    return {
        "message": "Smart Document Analyzer API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with system status."""
    global analyzer
    
    health_status = {
        "api_status": "healthy",
        "analyzer_status": "initialized" if analyzer else "not_initialized",
        "components": {
            "document_processor": "active",
            "entity_extractor": "active", 
            "classifier": "active",
            "summarizer": "active",
            "search_engine": "active" if analyzer and analyzer.search_engine.is_built else "pending"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if analyzer:
        insights = analyzer.get_collection_insights()
        health_status["collection_stats"] = insights
    
    return create_response(True, "System healthy", health_status)

@app.post("/analyze/document", tags=["Document Analysis"])
async def analyze_document(request: DocumentAnalysisRequest):
    """
    Analyze a single document with full NLP pipeline.
    
    Returns comprehensive analysis including:
    - Text statistics
    - Entity extraction
    - Document classification
    - Summary generation
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    try:
        # Generate document ID if not provided
        doc_id = request.doc_id or f"doc_{int(time.time())}"
        
        # Perform analysis
        result = analyzer.analyze_document(
            text=request.text,
            doc_id=doc_id,
            metadata=request.metadata
        )
        
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message=f"Document '{doc_id}' analyzed successfully",
            data=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/extract/entities", tags=["Entity Extraction"])
async def extract_entities(request: DocumentAnalysisRequest):
    """Extract named entities from text."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    try:
        entities = analyzer.entity_extractor.extract_entities(request.text)
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message=f"Found {len(entities)} entities",
            data={
                "entities": entities,
                "entity_count": len(entities),
                "entity_types": list(set(e["label"] for e in entities))
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")

@app.post("/classify/document", tags=["Document Classification"]) 
async def classify_document(request: DocumentAnalysisRequest):
    """Classify document type based on content."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    try:
        classification = analyzer.classifier.classify_document(request.text)
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message=f"Document classified as '{classification['predicted_class']}'",
            data=classification,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/summarize/text", tags=["Text Summarization"])
async def summarize_text(request: SummarizationRequest):
    """Generate text summary using extractive or other methods."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    try:
        if request.summary_type == "extractive":
            summary = analyzer.summarizer.extractive_summary(
                request.text, 
                ratio=request.ratio
            )
        elif request.summary_type == "bullet_points":
            summary = analyzer.summarizer.bullet_point_summary(request.text)
        else:
            raise ValueError(f"Unsupported summary type: {request.summary_type}")
        
        processing_time = time.time() - start_time
        
        # Calculate compression statistics
        original_words = len(request.text.split())
        if isinstance(summary, str):
            summary_words = len(summary.split())
            compression_ratio = 1 - (summary_words / original_words) if original_words > 0 else 0
        else:
            summary_words = sum(len(point.split()) for point in summary)
            compression_ratio = 1 - (summary_words / original_words) if original_words > 0 else 0
        
        return create_response(
            success=True,
            message="Text summarized successfully",
            data={
                "summary": summary,
                "summary_type": request.summary_type,
                "statistics": {
                    "original_words": original_words,
                    "summary_words": summary_words,
                    "compression_ratio": round(compression_ratio, 3)
                }
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/search/documents", tags=["Semantic Search"])
async def search_documents(request: SearchRequest):
    """Search documents using semantic similarity."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    try:
        results = analyzer.search_documents(request.query, top_k=request.top_k)
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message=f"Found {len(results)} results for query: '{request.query}'",
            data={
                "query": request.query,
                "results": results,
                "result_count": len(results),
                "search_time_ms": round(processing_time * 1000, 2)
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/batch/analyze", tags=["Batch Processing"])
async def batch_analyze_documents(request: BatchProcessRequest):
    """Process multiple documents in batch mode."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    try:
        results = []
        
        for i, doc in enumerate(request.documents):
            doc_id = doc.get("id", f"batch_doc_{i}")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if not text:
                results.append({
                    "doc_id": doc_id,
                    "success": False,
                    "error": "Empty text content"
                })
                continue
            
            try:
                result = analyzer.analyze_document(text, doc_id, metadata)
                results.append({
                    "doc_id": doc_id,
                    "success": True,
                    "analysis": result
                })
            except Exception as e:
                results.append({
                    "doc_id": doc_id,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        # Build search index after batch processing
        if analyzer.analyzed_documents:
            analyzer.build_search_index()
        
        success_count = sum(1 for r in results if r["success"])
        
        return create_response(
            success=True,
            message=f"Batch processing complete: {success_count}/{len(results)} documents processed successfully",
            data={
                "results": results,
                "statistics": {
                    "total_documents": len(results),
                    "successful": success_count,
                    "failed": len(results) - success_count
                }
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/upload/document", tags=["File Upload"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and analyze a document file.
    
    Supports text files for analysis. In a full implementation,
    this would handle PDF, DOCX, and other formats.
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    start_time = time.time()
    
    # Check file type
    allowed_types = ["text/plain", "application/json"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file.content_type} not supported. Allowed types: {allowed_types}"
        )
    
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # Create document metadata
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(content),
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Generate document ID from filename
        doc_id = f"upload_{file.filename}_{int(time.time())}"
        
        # Analyze the document
        result = analyzer.analyze_document(text_content, doc_id, metadata)
        
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message=f"File '{file.filename}' uploaded and analyzed successfully",
            data=result,
            processing_time=processing_time
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File contains invalid UTF-8 content")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/collection/insights", tags=["Analytics"])
async def get_collection_insights():
    """Get analytics and insights about the document collection."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        insights = analyzer.get_collection_insights()
        
        # Add additional analytics
        insights["api_stats"] = {
            "total_analyzed": len(analyzer.documents),
            "search_index_built": analyzer.search_engine.is_built,
            "last_updated": datetime.now().isoformat()
        }
        
        return create_response(
            success=True,
            message="Collection insights generated",
            data=insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@app.get("/collection/documents", tags=["Analytics"])
async def list_documents():
    """List all documents in the collection with basic information."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        documents_info = []
        
        for doc_id, doc in analyzer.documents.items():
            doc_info = {
                "id": doc_id,
                "metadata": doc["metadata"],
                "stats": doc["stats"],
                "classification": doc["classification"]["predicted_class"],
                "confidence": doc["classification"]["confidence"],
                "entities_count": len(doc["entities"]),
                "analysis_timestamp": doc["timestamp"]
            }
            documents_info.append(doc_info)
        
        return create_response(
            success=True,
            message=f"Listed {len(documents_info)} documents",
            data={
                "documents": documents_info,
                "total_count": len(documents_info)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/collection/clear", tags=["Management"])
async def clear_collection():
    """Clear all documents from the collection."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        doc_count = len(analyzer.documents)
        
        # Clear the collection
        analyzer.documents.clear()
        analyzer.analyzed_documents.clear()
        
        # Reset search index
        analyzer.search_engine.index = None
        analyzer.search_engine.documents = {}
        analyzer.search_engine.document_ids = []
        analyzer.search_engine.is_built = False
        
        return create_response(
            success=True,
            message=f"Collection cleared successfully. Removed {doc_count} documents"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=create_response(False, "Endpoint not found").dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=create_response(False, "Internal server error").dict()
    )

def main():
    """Run the API server."""
    print("🚀 Starting Smart Document Analyzer API Server...")
    print("📚 Available endpoints:")
    print("  • GET  /docs - API documentation")
    print("  • POST /analyze/document - Analyze single document")
    print("  • POST /extract/entities - Extract entities")
    print("  • POST /classify/document - Classify document")
    print("  • POST /summarize/text - Summarize text")
    print("  • POST /search/documents - Semantic search")
    print("  • POST /batch/analyze - Batch processing")
    print("  • POST /upload/document - File upload")
    print("  • GET  /collection/insights - Analytics")
    print("\n🌐 Server starting on http://localhost:8000")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    main()
