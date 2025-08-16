"""
Comprehensive example script demonstrating all Smart Document Analyzer services.

This script shows how to use each service:
- Document preprocessing
- Document classification
- Named Entity Recognition (NER)
- Document summarization
- Semantic search
"""

import os
import sys
import yaml
import logging
from pathlib import Path

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

def load_config():
    """Load configuration from YAML file."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using default settings.")
        return {}

def demo_preprocessing():
    """Demonstrate document preprocessing capabilities."""
    logger.info("=" * 60)
    logger.info("DOCUMENT PREPROCESSING DEMO")
    logger.info("=" * 60)
    
    processor = DocumentProcessor()
    
    # Sample raw text with noise
    sample_text = """
    <html>
    <body>
    <h1>Annual Report 2023</h1>
    <p>This is a sample document with HTML tags, URLs like https://example.com, 
    and various formatting issues.</p>
    
    <p>The company has shown remarkable growth this year!!!   
    Performance metrics include revenue of $1.2M and customer satisfaction of 95%.</p>
    
    <p>Key achievements:
    - Increased market share
    - Improved customer service
    - Enhanced product offerings</p>
    </body>
    </html>
    """
    
    logger.info("Original text (first 200 chars):")
    logger.info(sample_text[:200] + "...")
    
    # Clean the text
    cleaned_text = processor.clean_text(sample_text)
    logger.info("\\nCleaned text:")
    logger.info(cleaned_text)
    
    # Preprocess for different tasks
    classification_text = processor.preprocess_for_classification(sample_text)
    summarization_text = processor.preprocess_for_summarization(sample_text)
    
    logger.info("\\nPreprocessed for classification:")
    logger.info(classification_text)
    
    logger.info("\\nPreprocessed for summarization:")
    logger.info(summarization_text)
    
    # Extract sentences
    sentences = processor.split_into_sentences(sample_text)
    logger.info(f"\\nExtracted {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences, 1):
        logger.info(f"{i}. {sentence}")

def demo_classification():
    """Demonstrate document classification."""
    logger.info("\\n" + "=" * 60)
    logger.info("DOCUMENT CLASSIFICATION DEMO")
    logger.info("=" * 60)
    
    # Initialize classifier (will use default untrained model for demo)
    classifier = DocumentClassifier(model_type='traditional')
    
    # Sample documents for different types
    sample_documents = [
        {
            'text': "Invoice #INV-2024-001. Bill to: ABC Corporation. Services: Consulting. Amount: $2,500.00. Due Date: 2024-02-15. Tax ID: 12-3456789.",
            'expected': 'invoice'
        },
        {
            'text': "Annual Performance Report 2023. Executive Summary: This report analyzes key performance indicators and business metrics for the fiscal year. Revenue increased by 15% compared to previous year.",
            'expected': 'report'
        },
        {
            'text': "Service Agreement. This contract is entered into between Company A and Company B. Terms and conditions: Service delivery shall commence on January 1, 2024. Payment terms: Net 30 days.",
            'expected': 'contract'
        },
        {
            'text': "Abstract: This research investigates machine learning applications in natural language processing. Introduction: Recent advances in transformer architectures have revolutionized NLP tasks. Methodology: We employed BERT-based models for text classification.",
            'expected': 'research_paper'
        }
    ]
    
    # Note: For a real demo, you would need to train the classifier first
    logger.info("Sample documents for classification:")
    for i, doc in enumerate(sample_documents, 1):
        logger.info(f"\\nDocument {i} (Expected: {doc['expected']}):")
        logger.info(doc['text'][:100] + "...")
        
        # In a real scenario with trained model:
        # prediction = classifier.predict(doc['text'])
        # logger.info(f"Predicted: {prediction}")
        
    logger.info("\\nNote: To see actual predictions, train a classification model first using:")
    logger.info("python scripts/train_classifier.py --create_sample --data_path ./data/sample_training.csv")
    logger.info("python scripts/train_classifier.py --data_path ./data/sample_training.csv")

def demo_ner():
    """Demonstrate Named Entity Recognition."""
    logger.info("\\n" + "=" * 60)
    logger.info("NAMED ENTITY RECOGNITION DEMO")
    logger.info("=" * 60)
    
    try:
        ner = NERService()
        
        # Sample text with various entities
        sample_text = """
        John Smith, CEO of TechCorp Inc., announced today that the company has secured 
        a $5.2 million contract with Microsoft Corporation. The deal, signed on December 15, 2023, 
        will run until March 2025. The project involves developing AI solutions for their 
        Seattle headquarters. For more information, contact john.smith@techcorp.com or 
        call +1-555-123-4567. The company's stock (NASDAQ: TECH) rose 12% following the announcement.
        """
        
        logger.info("Sample text:")
        logger.info(sample_text)
        
        # Extract general entities
        entities = ner.extract_entities(sample_text)
        logger.info("\\nGeneral entities found:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                logger.info(f"\\n{entity_type}:")
                for entity in entity_list:
                    logger.info(f"  - {entity['text']} (confidence: {entity.get('confidence', 'N/A')})")
        
        # Extract financial entities
        financial_entities = ner.extract_financial_entities(sample_text)
        logger.info("\\nFinancial entities:")
        for entity_type, entity_list in financial_entities.items():
            if entity_list:
                logger.info(f"\\n{entity_type}:")
                for entity in entity_list:
                    logger.info(f"  - {entity['text']}")
        
        # Get entity summary
        summary = ner.get_entity_summary(sample_text)
        logger.info(f"\\nEntity Summary:")
        logger.info(f"Total entities: {summary['total_entities']}")
        logger.info(f"Entity types: {summary['entity_types']}")
        logger.info("Entities by type:")
        for entity_type, count in summary['entities_by_type'].items():
            logger.info(f"  {entity_type}: {count}")
    
    except Exception as e:
        logger.error(f"NER demo failed: {e}")
        logger.info("Make sure you have SpaCy and the English model installed:")
        logger.info("pip install spacy")
        logger.info("python -m spacy download en_core_web_sm")

def demo_summarization():
    """Demonstrate document summarization."""
    logger.info("\\n" + "=" * 60)
    logger.info("DOCUMENT SUMMARIZATION DEMO")
    logger.info("=" * 60)
    
    try:
        # Initialize summarizers
        extractive_summarizer = DocumentSummarizer(model_type='extractive')
        
        # Sample long text
        sample_text = """
        Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
        From healthcare to finance, transportation to entertainment, AI is revolutionizing industries and changing 
        the way we live and work. Machine learning, a subset of AI, enables computers to learn and improve from 
        experience without being explicitly programmed.
        
        Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and 
        manipulate human language. NLP combines computational linguistics with statistical, machine learning, 
        and deep learning models. These technologies enable computers to process human language in the form 
        of text or voice data and 'understand' its full meaning, complete with the speaker's intent and sentiment.
        
        The applications of NLP are vast and growing. Search engines use NLP to better understand user queries 
        and provide more relevant results. Virtual assistants like Siri, Alexa, and Google Assistant use NLP 
        to understand spoken commands and respond appropriately. Social media platforms use NLP to analyze 
        sentiment in posts and comments, helping them moderate content and understand user behavior.
        
        Document processing is another area where NLP shines. Traditional document processing required manual 
        effort to extract information, categorize documents, and summarize content. With NLP, these tasks can 
        be automated, saving time and reducing errors. Document classification can automatically sort incoming 
        documents into appropriate categories. Named Entity Recognition can extract important information like 
        names, dates, and locations from documents. Text summarization can create concise summaries of long documents.
        
        The future of AI and NLP looks promising. Advances in transformer architectures, like BERT and GPT, 
        have significantly improved the performance of NLP systems. These models can understand context better 
        and generate more human-like text. As computing power increases and more data becomes available, 
        we can expect even more sophisticated AI systems that can understand and generate human language 
        with unprecedented accuracy.
        
        However, with these advances come challenges. Bias in AI systems, privacy concerns, and the need for 
        transparency in AI decision-making are important issues that need to be addressed. As AI becomes more 
        prevalent in our daily lives, it's crucial to develop these systems responsibly and ethically.
        """
        
        logger.info("Original text (first 300 chars):")
        logger.info(sample_text[:300] + "...")
        logger.info(f"\\nOriginal text length: {len(sample_text.split())} words")
        
        # Extractive summarization
        extractive_summary = extractive_summarizer.summarize(sample_text, summary_ratio=0.3)
        logger.info("\\nExtractive Summary:")
        logger.info(extractive_summary)
        
        # Generate bullet points
        bullet_points = extractive_summarizer.generate_bullet_summary(sample_text, num_bullets=5)
        logger.info("\\nKey Points (Bullet Summary):")
        for i, point in enumerate(bullet_points, 1):
            logger.info(f"{i}. {point}")
        
        # Get summary statistics
        stats = extractive_summarizer.get_summary_statistics(sample_text, extractive_summary)
        logger.info("\\nSummary Statistics:")
        logger.info(f"Original: {stats['original_stats']['words']} words, {stats['original_stats']['sentences']} sentences")
        logger.info(f"Summary: {stats['summary_stats']['words']} words, {stats['summary_stats']['sentences']} sentences")
        logger.info(f"Compression ratio: {stats['compression_ratio']:.2f}")
        logger.info(f"Reduction: {stats['reduction_percentage']:.1f}%")
        
        # Try abstractive summarization (if available)
        try:
            abstractive_summarizer = DocumentSummarizer(model_type='abstractive')
            abstractive_summary = abstractive_summarizer.summarize(sample_text, max_length=100)
            logger.info("\\nAbstractive Summary:")
            logger.info(abstractive_summary)
        except Exception as e:
            logger.info("\\nAbstractive summarization not available (requires transformer models)")
            logger.info(f"Error: {e}")
    
    except Exception as e:
        logger.error(f"Summarization demo failed: {e}")

def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    logger.info("\\n" + "=" * 60)
    logger.info("SEMANTIC SEARCH DEMO")
    logger.info("=" * 60)
    
    try:
        # Initialize search engine
        search_engine = SemanticSearch()
        
        # Sample document collection
        documents = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "Natural language processing helps computers understand and interpret human language in text and speech.",
            "Computer vision allows machines to interpret and understand visual information from the world around them.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "Cybersecurity protects computer systems, networks, and programs from digital attacks and unauthorized access.",
            "Blockchain technology creates a decentralized, distributed ledger for recording transactions across many computers.",
            "Internet of Things (IoT) connects everyday objects to the internet, enabling them to send and receive data.",
            "Big data refers to extremely large datasets that require specialized tools and techniques for processing and analysis."
        ]
        
        # Add metadata for documents
        metadata = [
            {'topic': 'AI', 'difficulty': 'beginner', 'id': 0},
            {'topic': 'AI', 'difficulty': 'intermediate', 'id': 1},
            {'topic': 'AI', 'difficulty': 'intermediate', 'id': 2},
            {'topic': 'AI', 'difficulty': 'advanced', 'id': 3},
            {'topic': 'data', 'difficulty': 'intermediate', 'id': 4},
            {'topic': 'infrastructure', 'difficulty': 'beginner', 'id': 5},
            {'topic': 'security', 'difficulty': 'intermediate', 'id': 6},
            {'topic': 'blockchain', 'difficulty': 'advanced', 'id': 7},
            {'topic': 'IoT', 'difficulty': 'beginner', 'id': 8},
            {'topic': 'data', 'difficulty': 'intermediate', 'id': 9}
        ]
        
        logger.info(f"Adding {len(documents)} documents to search index...")
        search_engine.add_documents(documents, metadata)
        
        # Perform searches
        queries = [
            "How do computers learn from data?",
            "Understanding human language with AI",
            "Visual recognition and image processing",
            "Protecting systems from cyber threats"
        ]
        
        for query in queries:
            logger.info(f"\\nQuery: {query}")
            results = search_engine.search(query, top_k=3)
            
            logger.info("Search Results:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. Score: {result['similarity_score']:.3f}")
                logger.info(f"   Text: {result['document'][:80]}...")
                logger.info(f"   Topic: {result['metadata']['topic']}, Difficulty: {result['metadata']['difficulty']}")
        
        # Filtered search
        logger.info("\\n" + "-" * 40)
        logger.info("FILTERED SEARCH DEMO")
        logger.info("-" * 40)
        
        query = "artificial intelligence and learning"
        filters = {'topic': 'AI', 'difficulty': 'beginner'}
        
        logger.info(f"\\nQuery: {query}")
        logger.info(f"Filters: {filters}")
        
        filtered_results = search_engine.search_with_filters(query, filters, top_k=5)
        logger.info("Filtered Results:")
        for i, result in enumerate(filtered_results, 1):
            logger.info(f"{i}. Score: {result['similarity_score']:.3f}")
            logger.info(f"   Text: {result['document']}")
            logger.info(f"   Metadata: {result['metadata']}")
        
        # Document similarity
        logger.info("\\n" + "-" * 40)
        logger.info("DOCUMENT SIMILARITY DEMO")
        logger.info("-" * 40)
        
        similar_doc = "Artificial neural networks mimic the human brain to process information and learn patterns."
        logger.info(f"\\nFinding documents similar to: {similar_doc}")
        
        similar_results = search_engine.get_similar_documents(similar_doc, top_k=3)
        for i, result in enumerate(similar_results, 1):
            logger.info(f"{i}. Score: {result['similarity_score']:.3f}")
            logger.info(f"   Text: {result['document']}")
        
        # Index statistics
        stats = search_engine.get_index_stats()
        logger.info(f"\\nIndex Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Semantic search demo failed: {e}")

def main():
    """Run all demonstration examples."""
    logger.info("Smart Document Analyzer - Comprehensive Demo")
    logger.info("This demo showcases all the services available in the system.")
    
    # Load configuration
    config = load_config()
    
    # Run all demos
    try:
        demo_preprocessing()
        demo_classification()
        demo_ner()
        demo_summarization()
        demo_semantic_search()
        
        logger.info("\\n" + "=" * 60)
        logger.info("DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\\nFor more advanced usage:")
        logger.info("1. Train classification models using train_classifier.py")
        logger.info("2. Train custom NER models using train_ner.py")
        logger.info("3. Build document indexes for large-scale search")
        logger.info("4. Integrate with your applications using the service APIs")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
