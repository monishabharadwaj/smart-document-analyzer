#!/usr/bin/env python3
"""
API Testing Script
==================

This script tests all the API endpoints of the Smart Document Analyzer API.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def print_header(title):
    """Print a formatted section header."""
    print(f"\n🚀 {title}")
    print("=" * 60)

def print_test_result(endpoint, success, data=None, error=None):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {endpoint}")
    if error:
        print(f"    Error: {error}")
    if data and success:
        processing_time = data.get('processing_time_ms', 'N/A')
        print(f"    Processing time: {processing_time}ms")

def test_health_endpoints():
    """Test health check endpoints."""
    print_header("Health Check Tests")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        success = response.status_code == 200
        print_test_result("GET /", success, response.json() if success else None, 
                         None if success else f"Status: {response.status_code}")
    except Exception as e:
        print_test_result("GET /", False, error=str(e))
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("GET /health", success, data, 
                         None if success else f"Status: {response.status_code}")
    except Exception as e:
        print_test_result("GET /health", False, error=str(e))

def test_document_analysis():
    """Test document analysis endpoint."""
    print_header("Document Analysis Tests")
    
    # Test document analysis
    test_doc = {
        "text": "INVOICE #INV-2023-001\nDate: December 15, 2023\nTotal: $1,500.00\nPayment due: January 15, 2024",
        "doc_id": "test_invoice_001",
        "metadata": {"type": "test", "source": "api_test"}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze/document", json=test_doc)
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("POST /analyze/document", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            result = data.get('data', {})
            print(f"    Classification: {result.get('classification', {}).get('predicted_class', 'N/A')}")
            print(f"    Entities found: {len(result.get('entities', []))}")
            print(f"    Word count: {result.get('stats', {}).get('word_count', 'N/A')}")
        
    except Exception as e:
        print_test_result("POST /analyze/document", False, error=str(e))

def test_entity_extraction():
    """Test entity extraction endpoint."""
    print_header("Entity Extraction Tests")
    
    test_text = {
        "text": "Contact John Smith at john.smith@company.com or call (555) 123-4567. Payment of $2,500.00 is due on January 15, 2024."
    }
    
    try:
        response = requests.post(f"{BASE_URL}/extract/entities", json=test_text)
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("POST /extract/entities", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            entities = data.get('data', {}).get('entities', [])
            entity_types = data.get('data', {}).get('entity_types', [])
            print(f"    Entities found: {len(entities)}")
            print(f"    Entity types: {', '.join(entity_types)}")
        
    except Exception as e:
        print_test_result("POST /extract/entities", False, error=str(e))

def test_document_classification():
    """Test document classification endpoint."""
    print_header("Document Classification Tests")
    
    test_docs = [
        {
            "text": "INVOICE #123 - Total amount: $500.00 - Due date: January 30, 2024",
            "expected": "invoice"
        },
        {
            "text": "This research paper investigates the effects of climate change on biodiversity. Our methodology involved analyzing 200 species across different ecosystems.",
            "expected": "research"
        },
        {
            "text": "SERVICE AGREEMENT - This agreement is between Party A and Party B for cloud hosting services.",
            "expected": "contract"
        }
    ]
    
    for i, test_doc in enumerate(test_docs):
        try:
            response = requests.post(f"{BASE_URL}/classify/document", json={"text": test_doc["text"]})
            success = response.status_code == 200
            data = response.json() if success else None
            
            endpoint = f"POST /classify/document (test {i+1})"
            print_test_result(endpoint, success, data,
                             None if success else f"Status: {response.status_code}")
            
            if success:
                predicted = data.get('data', {}).get('predicted_class', 'N/A')
                confidence = data.get('data', {}).get('confidence', 0)
                expected = test_doc['expected']
                print(f"    Predicted: {predicted} (confidence: {confidence:.3f})")
                print(f"    Expected: {expected} - {'✅ Match' if predicted == expected else '⚠️ Different'}")
            
        except Exception as e:
            print_test_result(f"POST /classify/document (test {i+1})", False, error=str(e))

def test_summarization():
    """Test text summarization endpoint."""
    print_header("Text Summarization Tests")
    
    test_text = {
        "text": "Artificial intelligence has made significant progress in recent years. Machine learning algorithms are being used in various industries including healthcare, finance, and transportation. Deep learning models have achieved remarkable results in image recognition and natural language processing. The impact of AI on society continues to grow as more companies adopt these technologies. However, there are also concerns about job displacement and ethical considerations that need to be addressed.",
        "ratio": 0.4,
        "summary_type": "extractive"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/summarize/text", json=test_text)
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("POST /summarize/text", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            result = data.get('data', {})
            stats = result.get('statistics', {})
            print(f"    Original words: {stats.get('original_words', 'N/A')}")
            print(f"    Summary words: {stats.get('summary_words', 'N/A')}")
            print(f"    Compression ratio: {stats.get('compression_ratio', 'N/A')}")
            print(f"    Summary: {result.get('summary', '')[:100]}...")
        
    except Exception as e:
        print_test_result("POST /summarize/text", False, error=str(e))

def test_semantic_search():
    """Test semantic search endpoint."""
    print_header("Semantic Search Tests")
    
    # First, add some documents to search through
    print("Setting up documents for search...")
    
    sample_docs = [
        {
            "text": "This research paper discusses artificial intelligence and machine learning applications in healthcare diagnostics.",
            "doc_id": "research_ai_health",
            "metadata": {"type": "research", "field": "AI"}
        },
        {
            "text": "Financial report showing quarterly revenue growth of 15% and improved profit margins in the technology sector.",
            "doc_id": "financial_report_q3",
            "metadata": {"type": "report", "period": "Q3"}
        }
    ]
    
    # Add documents first
    for doc in sample_docs:
        try:
            requests.post(f"{BASE_URL}/analyze/document", json=doc)
        except:
            pass  # Continue if document addition fails
    
    # Test search
    search_request = {
        "query": "artificial intelligence machine learning healthcare",
        "top_k": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/documents", json=search_request)
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("POST /search/documents", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            results = data.get('data', {}).get('results', [])
            print(f"    Results found: {len(results)}")
            for i, result in enumerate(results[:3]):  # Show top 3
                score = result.get('search_score', 0)
                title = result.get('title', 'N/A')
                print(f"    {i+1}. {title} (score: {score:.3f})")
        
    except Exception as e:
        print_test_result("POST /search/documents", False, error=str(e))

def test_batch_processing():
    """Test batch processing endpoint."""
    print_header("Batch Processing Tests")
    
    batch_docs = {
        "documents": [
            {
                "id": "batch_1",
                "text": "Invoice #001 - Amount: $750.00 - Due: January 30, 2024",
                "metadata": {"source": "batch_test"}
            },
            {
                "id": "batch_2",
                "text": "Research methodology for analyzing climate data over the past decade shows significant temperature increases.",
                "metadata": {"source": "batch_test"}
            },
            {
                "id": "batch_3",
                "text": "",  # Empty text to test error handling
                "metadata": {"source": "batch_test"}
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/batch/analyze", json=batch_docs)
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("POST /batch/analyze", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            stats = data.get('data', {}).get('statistics', {})
            print(f"    Total documents: {stats.get('total_documents', 'N/A')}")
            print(f"    Successful: {stats.get('successful', 'N/A')}")
            print(f"    Failed: {stats.get('failed', 'N/A')}")
        
    except Exception as e:
        print_test_result("POST /batch/analyze", False, error=str(e))

def test_analytics_endpoints():
    """Test analytics endpoints."""
    print_header("Analytics Tests")
    
    # Test collection insights
    try:
        response = requests.get(f"{BASE_URL}/collection/insights")
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("GET /collection/insights", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            insights = data.get('data', {})
            print(f"    Total documents: {insights.get('total_documents', 'N/A')}")
            print(f"    Total words: {insights.get('total_words', 'N/A')}")
            print(f"    Document categories: {list(insights.get('document_categories', {}).keys())}")
        
    except Exception as e:
        print_test_result("GET /collection/insights", False, error=str(e))
    
    # Test document listing
    try:
        response = requests.get(f"{BASE_URL}/collection/documents")
        success = response.status_code == 200
        data = response.json() if success else None
        print_test_result("GET /collection/documents", success, data,
                         None if success else f"Status: {response.status_code}")
        
        if success:
            docs = data.get('data', {}).get('documents', [])
            print(f"    Documents listed: {len(docs)}")
        
    except Exception as e:
        print_test_result("GET /collection/documents", False, error=str(e))

def main():
    """Run all API tests."""
    print_header("Smart Document Analyzer API Tests")
    print(f"Testing API at: {BASE_URL}")
    print("Make sure the API server is running before running these tests!")
    
    try:
        # Test basic connectivity first
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding. Please start the server first.")
            print("Run: python api_server.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API server. Please start the server first.")
        print("Run: python api_server.py")
        return
    
    print("✅ API server is reachable. Starting tests...\n")
    
    # Run all tests
    test_health_endpoints()
    test_document_analysis()
    test_entity_extraction()
    test_document_classification()
    test_summarization()
    test_semantic_search()
    test_batch_processing()
    test_analytics_endpoints()
    
    print("\n" + "="*60)
    print("✅ API Testing Complete!")
    print("🌐 Visit http://localhost:8000/docs for interactive API documentation")

if __name__ == "__main__":
    main()
