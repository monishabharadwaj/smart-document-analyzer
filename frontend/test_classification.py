#!/usr/bin/env python3
"""
Step 4: Document Classification Test and Training Script
"""

import os
import pandas as pd
from services.classification import DocumentClassifier, DocumentTypeFeatureExtractor
from services.preprocessing import DocumentProcessor

def create_sample_training_data():
    """Create sample training data for document classification."""
    
    # Sample texts for each document type
    sample_data = {
        'invoice': [
            "INVOICE #12345\nDate: 01/15/2024\nBill To: ABC Company\nDescription: Professional Services\nAmount: $2,500.00\nTax: $250.00\nTotal Due: $2,750.00\nPayment Terms: Net 30",
            "Invoice Number: INV-2024-001\nInvoice Date: February 1, 2024\nDue Date: March 3, 2024\nBill To: XYZ Corporation\nServices Rendered: Consulting\nSubtotal: $1,800.00\nVAT (10%): $180.00\nTotal Amount: $1,980.00",
            "BILLING STATEMENT\nAccount #: 987654\nBilling Period: Jan 1 - Jan 31, 2024\nPrevious Balance: $0.00\nCharges This Month: $450.00\nPayments Received: $0.00\nCurrent Amount Due: $450.00",
        ],
        'contract': [
            "SERVICE AGREEMENT\n\nThis Agreement is entered into between Party A and Party B.\nWHEREAS, the parties desire to enter into this contract.\nTerms and Conditions:\n1. The contractor shall provide services as described.\n2. Payment terms are Net 30 days.\n3. This agreement shall remain in effect for one year.",
            "EMPLOYMENT CONTRACT\n\nThis employment agreement hereby establishes the terms between the Company and Employee.\nThe Employee shall perform duties as assigned.\nCompensation shall be $75,000 annually.\nThis contract is subject to the terms and conditions outlined herein.",
            "LEASE AGREEMENT\n\nThe Landlord agrees to lease the premises to the Tenant.\nMonthly rent: $2,000\nLease term: 12 months\nSecurity deposit: $2,000\nThe parties agree to the following terms and conditions.",
        ],
        'report': [
            "QUARTERLY BUSINESS REPORT\n\nExecutive Summary:\nThis report analyzes the company's performance for Q1 2024.\n\nMethodology:\nData was collected from various departments.\n\nFindings:\n- Revenue increased by 15%\n- Customer satisfaction improved\n\nConclusion:\nThe quarter showed positive growth across all metrics.",
            "MARKET ANALYSIS REPORT\n\nIntroduction:\nThis analysis examines current market trends.\n\nData Analysis:\nSales figures show consistent growth.\n\nKey Findings:\n1. Market share increased by 8%\n2. New customer acquisition rose by 22%\n\nRecommendations:\nContinue current strategy with minor adjustments.",
            "PROJECT STATUS REPORT\n\nProject: Website Redesign\nStatus: On Track\n\nSummary:\nThe project is progressing according to schedule.\n\nMilestones Completed:\n- Design phase: 100%\n- Development: 60%\n\nNext Steps:\nComplete testing phase by month end.",
        ],
        'research_paper': [
            "ABSTRACT\n\nThis research investigates the effects of machine learning on document processing.\n\nINTRODUCTION\n\nNatural language processing has shown significant advances.\n\nMETHODOLOGY\n\nWe used a dataset of 10,000 documents for training.\n\nRESULTS\n\nAccuracy reached 94.5% on the test set.\n\nDISCUSSION\n\nThe results indicate promising applications.\n\nREFERENCES\n[1] Smith, J. (2023). ML in Document Processing. Journal of AI.",
            "A STUDY ON DOCUMENT CLASSIFICATION TECHNIQUES\n\nAbstract: This paper presents a comprehensive analysis of various document classification methods.\n\nIntroduction: Document classification is a fundamental task in information retrieval.\n\nRelated Work: Previous studies have explored various approaches [1][2].\n\nExperimental Results: Our method achieved 89% accuracy.\n\nConclusion: The proposed approach shows significant improvements.\n\nReferences:\n[1] Brown, A. et al. Classification Methods. ACM 2023.\n[2] Davis, L. NLP Techniques. IEEE 2024.",
            "RESEARCH ARTICLE\n\nTitle: Advanced Text Mining Techniques\n\nAbstract: We propose novel algorithms for text analysis.\n\nKeywords: text mining, NLP, machine learning\n\nIntroduction: Text mining has become increasingly important.\n\nMethodology: We employed supervised learning techniques.\n\nResults: F1-score of 0.92 was achieved.\n\nConclusion: The methodology proves effective.\n\nDOI: 10.1234/example.2024",
        ]
    }
    
    # Create training dataset
    texts = []
    labels = []
    
    for doc_type, samples in sample_data.items():
        texts.extend(samples)
        labels.extend([doc_type] * len(samples))
    
    return texts, labels

def test_document_classification():
    """Test the document classification system."""
    
    print("=== Step 4: Document Classification Test ===")
    print()
    
    # Create sample training data
    print("📊 Creating sample training data...")
    texts, labels = create_sample_training_data()
    
    print(f"Training data: {len(texts)} documents")
    print(f"Document types: {set(labels)}")
    print()
    
    # Initialize classifier
    print("🤖 Initializing Document Classifier...")
    classifier = DocumentClassifier(model_type='traditional')
    
    # Train the model
    print("🎯 Training the classification model...")
    training_results = classifier.train(texts, labels)
    
    print("Training Results:")
    print(f"  Training Accuracy: {training_results['train_accuracy']:.3f}")
    print(f"  Validation Accuracy: {training_results['validation_accuracy']:.3f}")
    print(f"  Cross-validation Mean: {training_results['cv_mean']:.3f} (±{training_results['cv_std']:.3f})")
    print()
    
    # Test with our sample document
    print("📄 Testing with sample document...")
    with open('sample_document.txt', 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    prediction = classifier.predict(sample_text)
    probabilities = classifier.predict_proba(sample_text)
    
    print(f"Sample document classification: {prediction}")
    print("Classification probabilities:")
    for doc_type, prob in probabilities.items():
        print(f"  {doc_type}: {prob:.3f}")
    print()
    
    # Test feature extraction
    print("🔍 Testing feature extraction...")
    extractor = DocumentTypeFeatureExtractor()
    
    # Test with an invoice sample
    invoice_sample = texts[0]  # First invoice sample
    invoice_features = extractor.extract_invoice_features(invoice_sample)
    print(f"Invoice features: {invoice_features}")
    
    # Test with a contract sample
    contract_sample = texts[3]  # First contract sample
    contract_features = extractor.extract_contract_features(contract_sample)
    print(f"Contract features: {contract_features}")
    
    print()
    
    # Save the trained model
    model_path = "models/document_classifier.pkl"
    os.makedirs("models", exist_ok=True)
    classifier.save_model(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    print()
    print("✅ Step 4 COMPLETE: Document Classification")
    print("Ready for Step 5: Document Summarization!")
    
    return classifier, training_results

def load_external_dataset(dataset_path):
    """
    Load external dataset for training.
    
    Args:
        dataset_path (str): Path to the dataset file (CSV, JSON, etc.)
        
    Returns:
        tuple: (texts, labels) or None if file not found
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return None
    
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            # Assuming columns are 'text' and 'label' or similar
            if 'text' in df.columns and 'label' in df.columns:
                return df['text'].tolist(), df['label'].tolist()
            elif 'document' in df.columns and 'type' in df.columns:
                return df['document'].tolist(), df['type'].tolist()
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
            if 'text' in df.columns and 'label' in df.columns:
                return df['text'].tolist(), df['label'].tolist()
        
        print(f"Could not determine text and label columns in {dataset_path}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    # Check for external datasets first
    possible_dataset_paths = [
        "data/document_classification.csv",
        "data/documents.csv", 
        "document_dataset.csv",
        "classification_data.csv"
    ]
    
    external_data = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            print(f"Found external dataset: {path}")
            external_data = load_external_dataset(path)
            break
    
    if external_data:
        print("Using external dataset for training...")
        texts, labels = external_data
    else:
        print("No external dataset found. Using sample data...")
    
    # Run the classification test
    test_document_classification()
