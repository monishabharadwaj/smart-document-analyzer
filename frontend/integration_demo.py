#!/usr/bin/env python3
"""
Step 7: Integration & Testing (Simplified Demo)
===============================================

This script demonstrates the integration capabilities using the components we have built:
- Semantic search engine
- Basic text processing
- Document analysis workflow

This serves as a proof-of-concept for the full integration system.
"""

import sys
import os
import time
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.semantic_search import SemanticSearchEngine

def print_header(title):
    """Print a formatted section header."""
    print(f"\n🚀 Step 7: {title}")
    print("=" * 60)

def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n=== {title} ===")

def print_results_box(title, content, emoji="📋"):
    """Print results in a formatted box."""
    print(f"\n{emoji} {title}:")
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, list) and value:
                print(f"  📌 {key.title()}:")
                for item in value[:3]:  # Show first 3 items
                    if isinstance(item, dict):
                        print(f"    • {item.get('text', item)} ({item.get('label', '')})")
                    else:
                        print(f"    • {item}")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more")
            else:
                print(f"  📌 {key.title()}: {value}")
    else:
        print(f"  {content}")

class SimpleDocumentProcessor:
    """Basic text processing functionality."""
    
    def extract_text(self, content: str) -> str:
        """Extract and clean text content."""
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        text = text.strip()
        return text
    
    def get_basic_stats(self, text: str) -> Dict[str, Any]:
        """Get basic statistics about the text."""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'character_count': len(text),
            'avg_words_per_sentence': round(len(words) / max(len(sentences), 1), 2)
        }

class SimpleEntityExtractor:
    """Basic entity extraction using regex patterns."""
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract basic entities from text."""
        entities = []
        
        # Money patterns
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        money_matches = re.finditer(money_pattern, text)
        for match in money_matches:
            entities.append({
                'text': match.group(),
                'label': 'MONEY',
                'start': match.start(),
                'end': match.end()
            })
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.finditer(email_pattern, text)
        for match in email_matches:
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end()
            })
        
        # Phone patterns
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_matches = re.finditer(phone_pattern, text)
        for match in phone_matches:
            entities.append({
                'text': match.group(),
                'label': 'PHONE',
                'start': match.start(),
                'end': match.end()
            })
        
        # Date patterns (basic)
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        date_matches = re.finditer(date_pattern, text)
        for match in date_matches:
            entities.append({
                'text': match.group(),
                'label': 'DATE',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities

class SimpleDocumentClassifier:
    """Basic document classification using keyword matching."""
    
    def __init__(self):
        self.categories = {
            'invoice': ['invoice', 'bill to', 'payment terms', 'due date', 'total:', 'amount due'],
            'contract': ['agreement', 'party', 'terms', 'conditions', 'witness', 'executed'],
            'report': ['report', 'executive summary', 'performance', 'analysis', 'quarterly'],
            'research': ['abstract', 'methodology', 'findings', 'conclusion', 'research'],
            'news': ['announced', 'sources', 'breaking', 'reported', 'according to'],
            'email': ['from:', 'to:', 'subject:', 're:', 'dear', 'sincerely']
        }
    
    def classify_document(self, text: str) -> Dict[str, Any]:
        """Classify document based on keyword matching."""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score / len(keywords)  # Normalize by number of keywords
        
        # Get the category with highest score
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_scores': scores
        }

class SimpleSummarizer:
    """Basic text summarization using sentence ranking."""
    
    def extractive_summary(self, text: str, ratio: float = 0.3) -> str:
        """Create extractive summary by selecting top sentences."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return text
        
        # Score sentences by length and keyword presence
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Basic scoring by word count
            
            # Boost score for sentences with important keywords
            important_words = ['important', 'key', 'significant', 'major', 'critical', 'main']
            for word in important_words:
                if word in sentence.lower():
                    score *= 1.2
            
            scored_sentences.append((score, i, sentence))
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:num_sentences]
        
        # Maintain original order
        selected_sentences = sorted(top_sentences, key=lambda x: x[1])
        
        return '. '.join([s[2] for s in selected_sentences]) + '.'
    
    def bullet_point_summary(self, text: str) -> List[str]:
        """Create bullet point summary."""
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s) > 10]
        
        # Select sentences that seem to be key points
        bullet_points = []
        for sentence in sentences[:5]:  # Limit to 5 bullet points
            # Look for sentences with numbers, percentages, or key phrases
            if any(indicator in sentence.lower() for indicator in 
                   ['increase', 'decrease', '%', 'million', 'billion', 'key', 'important']):
                bullet_points.append(sentence)
        
        return bullet_points[:3]  # Return top 3

class IntegratedDocumentAnalyzer:
    """
    Integrated document analysis system using simplified components.
    """
    
    def __init__(self):
        """Initialize all components."""
        print("🔧 Initializing Integrated Document Analyzer...")
        
        # Initialize components
        self.processor = SimpleDocumentProcessor()
        self.entity_extractor = SimpleEntityExtractor()
        self.classifier = SimpleDocumentClassifier()
        self.summarizer = SimpleSummarizer()
        self.search_engine = SemanticSearchEngine()
        
        # Document storage
        self.documents = {}
        self.analyzed_documents = []
        
        print("✅ All components initialized successfully!")
    
    def analyze_document(self, text: str, doc_id: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single document.
        """
        start_time = time.time()
        metadata = metadata or {}
        
        print(f"📄 Analyzing Document: {doc_id}")
        
        # Step 1: Text processing
        processed_text = self.processor.extract_text(text)
        stats = self.processor.get_basic_stats(processed_text)
        
        # Step 2: Entity extraction
        entities = self.entity_extractor.extract_entities(processed_text)
        
        # Step 3: Document classification
        classification = self.classifier.classify_document(processed_text)
        
        # Step 4: Document summarization
        summary = self.summarizer.extractive_summary(processed_text, ratio=0.3)
        bullet_points = self.summarizer.bullet_point_summary(processed_text)
        
        # Compile results
        analysis_result = {
            'id': doc_id,
            'text': processed_text,
            'metadata': metadata,
            'stats': stats,
            'entities': entities,
            'classification': classification,
            'summary': {
                'extractive': summary,
                'bullet_points': bullet_points
            },
            'analysis_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store results
        self.documents[doc_id] = analysis_result
        self.analyzed_documents.append({
            'id': doc_id,
            'title': metadata.get('title', f'Document {doc_id}'),
            'content': processed_text,
            'category': classification.get('predicted_class', 'unknown'),
            'confidence': classification.get('confidence', 0.0)
        })
        
        return analysis_result
    
    def build_search_index(self):
        """Build the semantic search index from analyzed documents."""
        print("🔍 Building semantic search index...")
        if self.analyzed_documents:
            self.search_engine.build_index(self.analyzed_documents)
            print(f"✅ Search index built with {len(self.analyzed_documents)} documents")
        else:
            print("⚠️ No documents to index")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents using semantic similarity."""
        if not self.search_engine.is_built:
            self.build_search_index()
        
        results = self.search_engine.search_with_metadata(query, top_k)
        return results
    
    def get_collection_insights(self) -> Dict[str, Any]:
        """Generate insights from all analyzed documents."""
        if not self.documents:
            return {"message": "No documents analyzed yet"}
        
        # Aggregate statistics
        total_docs = len(self.documents)
        total_words = sum(doc['stats']['word_count'] for doc in self.documents.values())
        avg_words = total_words / total_docs if total_docs > 0 else 0
        
        # Entity statistics
        all_entities = []
        for doc in self.documents.values():
            all_entities.extend(doc['entities'])
        
        entity_counts = {}
        for entity in all_entities:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        # Classification statistics
        class_counts = {}
        for doc in self.analyzed_documents:
            category = doc['category']
            class_counts[category] = class_counts.get(category, 0) + 1
        
        return {
            'total_documents': total_docs,
            'total_words': total_words,
            'average_words_per_document': round(avg_words, 1),
            'entity_distribution': dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)),
            'document_categories': dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)),
            'total_entities_found': len(all_entities)
        }

def create_test_documents():
    """Create test documents for demonstration."""
    return [
        {
            'id': 'invoice_001',
            'title': 'Software License Invoice',
            'content': """INVOICE #INV-2023-4567
Date: December 15, 2023
Bill To: TechCorp Solutions Inc.
123 Business Ave, San Francisco, CA 94105

Description: Annual Software License Renewal
Microsoft Office 365 Business Premium - 50 licenses
Unit Price: $22.00 per license per month
Total: $13,200.00 annually

Payment Terms: Net 30 days
Due Date: January 15, 2024

Contact: billing@vendor.com
Phone: (555) 123-4567""",
            'metadata': {'source': 'accounting', 'type': 'invoice'}
        },
        {
            'id': 'contract_001',
            'title': 'Service Agreement Contract',
            'content': """SERVICE AGREEMENT

This Service Agreement is entered into on January 1, 2024, between DataFlow Technologies and Global Manufacturing Corp.

1. SCOPE OF SERVICES
Service Provider shall provide cloud infrastructure management services including 24/7 system monitoring, data backup and recovery, security compliance management, and performance optimization.

2. COMPENSATION
Client agrees to pay $15,000 per month for the services described herein.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

Service Provider: John Smith, CEO DataFlow Technologies
Client: Sarah Johnson, CTO Global Manufacturing Corp""",
            'metadata': {'source': 'legal', 'type': 'contract'}
        },
        {
            'id': 'report_001',
            'title': 'Quarterly Performance Report',
            'content': """Q4 2023 QUARTERLY PERFORMANCE REPORT

EXECUTIVE SUMMARY
This report analyzes the performance metrics for Q4 2023, highlighting key achievements and areas for improvement.

FINANCIAL PERFORMANCE
• Revenue increased by 18% compared to Q4 2022, reaching $2.8 million
• Operating expenses were controlled at $1.9 million, representing a 12% improvement in efficiency
• Net profit margin improved to 32%, up from 28% in the previous quarter

KEY METRICS
• Customer retention rate: 94%
• Average order value: $1,250
• Monthly recurring revenue growth: 22%

OUTLOOK FOR Q1 2024
Based on current market conditions and pipeline analysis, we project 25% revenue growth for Q1 2024.""",
            'metadata': {'source': 'management', 'type': 'report', 'quarter': 'Q4 2023'}
        },
        {
            'id': 'research_001',
            'title': 'AI Ethics Research Paper',
            'content': """Ethical Considerations in Artificial Intelligence Development: A Framework for Responsible Innovation

ABSTRACT
As artificial intelligence systems become increasingly sophisticated, the need for comprehensive ethical frameworks has become paramount. This paper presents a structured approach to addressing ethical considerations in AI development.

METHODOLOGY
Our research methodology involved analyzing 150 AI ethics frameworks from academic institutions and technology companies. We conducted interviews with 25 AI researchers and practitioners.

KEY FINDINGS
1. Bias Prevention: Organizations implementing diverse training datasets showed 40% fewer instances of algorithmic bias
2. Transparency Requirements: Clear documentation increased user trust by 35%
3. Human Oversight: Systems with human-in-the-loop mechanisms demonstrated 60% better error detection rates

CONCLUSION
The development of ethical AI systems requires ongoing collaboration between technologists, ethicists, and policymakers.""",
            'metadata': {'source': 'academic', 'type': 'research_paper', 'field': 'AI Ethics'}
        }
    ]

def test_integrated_workflow():
    """Test the complete integrated workflow."""
    print_subheader("Complete Integration Workflow Test")
    
    # Initialize analyzer
    analyzer = IntegratedDocumentAnalyzer()
    documents = create_test_documents()
    
    print(f"📚 Processing {len(documents)} documents through complete workflow...")
    
    # Analyze all documents
    total_start = time.time()
    for doc in documents:
        result = analyzer.analyze_document(
            text=doc['content'],
            doc_id=doc['id'],
            metadata=doc['metadata']
        )
    
    # Build search index
    analyzer.build_search_index()
    
    total_time = time.time() - total_start
    
    print(f"⚡ Complete workflow finished in {total_time:.2f} seconds")
    
    # Show collection insights
    insights = analyzer.get_collection_insights()
    print_results_box("Document Collection Analysis", insights, "📊")
    
    return analyzer

def test_semantic_search_workflow():
    """Test semantic search integration."""
    print_subheader("Semantic Search Integration")
    
    # Use the analyzer from previous test or create new one
    analyzer = IntegratedDocumentAnalyzer()
    documents = create_test_documents()
    
    # Quick setup
    for doc in documents:
        analyzer.analyze_document(doc['content'], doc['id'], doc['metadata'])
    analyzer.build_search_index()
    
    # Test various search queries
    search_queries = [
        "artificial intelligence ethics research",
        "software license payment terms",
        "quarterly revenue performance metrics",
        "cloud services agreement contract"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Search Query: '{query}'")
        results = analyzer.search_documents(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (Score: {result.get('search_score', 0):.3f})")
            print(f"     📂 {result['category']} | 🎯 Confidence: {result.get('confidence', 0):.3f}")
            
            # Show document summary
            doc_analysis = analyzer.documents[result['id']]
            summary = doc_analysis['summary']['extractive'][:100] + "..."
            print(f"     💭 {summary}")

def test_component_integration():
    """Test how components work together."""
    print_subheader("Component Integration Testing")
    
    analyzer = IntegratedDocumentAnalyzer()
    
    # Test with a single document
    test_doc = create_test_documents()[0]
    result = analyzer.analyze_document(
        test_doc['content'], 
        test_doc['id'], 
        test_doc['metadata']
    )
    
    # Show detailed results
    print_results_box("Text Statistics", result['stats'], "📊")
    print_results_box("Entity Extraction", result['entities'], "🏷️")
    print_results_box("Classification Results", {
        'Predicted Class': result['classification']['predicted_class'],
        'Confidence': result['classification']['confidence'],
        'All Scores': result['classification']['all_scores']
    }, "🎯")
    print_results_box("Document Summary", {
        'Extractive Summary': result['summary']['extractive'][:150] + "...",
        'Bullet Points': result['summary']['bullet_points']
    }, "📝")

def test_performance_metrics():
    """Test performance of integrated system."""
    print_subheader("Performance Metrics")
    
    analyzer = IntegratedDocumentAnalyzer()
    documents = create_test_documents()
    
    # Measure component performance
    performance_data = {}
    
    test_text = documents[0]['content']
    
    # Text processing
    start = time.time()
    processed = analyzer.processor.extract_text(test_text)
    stats = analyzer.processor.get_basic_stats(processed)
    performance_data['text_processing'] = time.time() - start
    
    # Entity extraction
    start = time.time()
    entities = analyzer.entity_extractor.extract_entities(test_text)
    performance_data['entity_extraction'] = time.time() - start
    
    # Classification
    start = time.time()
    classification = analyzer.classifier.classify_document(test_text)
    performance_data['classification'] = time.time() - start
    
    # Summarization
    start = time.time()
    summary = analyzer.summarizer.extractive_summary(test_text)
    performance_data['summarization'] = time.time() - start
    
    # Search indexing
    analyzer.analyzed_documents.append({
        'id': 'perf_test',
        'title': 'Performance Test Doc',
        'content': test_text,
        'category': 'test'
    })
    start = time.time()
    analyzer.build_search_index()
    performance_data['search_indexing'] = time.time() - start
    
    print("⚡ Component Performance Results:")
    for component, duration in performance_data.items():
        print(f"  📊 {component.replace('_', ' ').title()}: {duration*1000:.1f}ms")
    
    total_time = sum(performance_data.values())
    print(f"\n🏆 Total Processing Time: {total_time*1000:.1f}ms")

def main():
    """Run the integration demonstration."""
    print_header("Integration & Testing Demo")
    
    try:
        # Test complete workflow
        test_integrated_workflow()
        
        # Test semantic search
        test_semantic_search_workflow()
        
        # Test component integration
        test_component_integration()
        
        # Performance metrics
        test_performance_metrics()
        
        print("\n" + "="*60)
        print("✅ Step 7 COMPLETE: Integration & Testing Demo")
        print("🎯 Integration proof-of-concept successful!")
        print("📋 Components working together:")
        print("   • Document processing & text extraction")
        print("   • Entity recognition (basic patterns)")
        print("   • Document classification (keyword-based)")
        print("   • Text summarization (extractive)")
        print("   • Semantic search (transformer-based)")
        print("\nReady for Step 8: API Server Development!")
        
    except Exception as e:
        print(f"\n❌ Error during integration demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
