"""
Training script for document classification models.

This script trains classification models to categorize documents into types
(invoice, report, contract, research paper).
"""

import os
import sys
import argparse
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import logging

# Add parent directory to path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.classification import DocumentClassifier
from services.preprocessing import DocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_data(data_path):
    """
    Load training data from CSV file.
    
    Expected format:
    - 'text' column: document text
    - 'label' column: document type (invoice, report, contract, research_paper)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = ['text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df['text'].tolist(), df['label'].tolist()

def create_sample_data(output_path):
    """Create sample training data for demonstration."""
    sample_data = [
        {
            'text': "Invoice #12345 for professional services. Total amount: $1,500.00. Due date: 2024-01-15. Payment terms: Net 30.",
            'label': 'invoice'
        },
        {
            'text': "Annual Financial Report 2023. This report presents the financial performance and position of the company for the fiscal year ending December 31, 2023.",
            'label': 'report'
        },
        {
            'text': "Service Agreement between Party A and Party B. This contract shall govern the terms and conditions of service delivery.",
            'label': 'contract'
        },
        {
            'text': "Abstract: This research paper investigates the impact of machine learning on document processing. Introduction: Natural language processing has revolutionized how we handle text data.",
            'label': 'research_paper'
        },
        # Add more sample data...
        {
            'text': "Purchase Order #PO-2024-001. Vendor: ABC Corp. Items: Office supplies. Total: $350.00. Delivery date: 2024-02-01.",
            'label': 'invoice'
        },
        {
            'text': "Quarterly Sales Report Q1 2024. Sales increased by 15% compared to the previous quarter. Key metrics and analysis included.",
            'label': 'report'
        },
        {
            'text': "Employment Contract for John Doe. Position: Software Engineer. Start date: 2024-01-01. Salary and benefits outlined herein.",
            'label': 'contract'
        },
        {
            'text': "Methodology: We used a supervised learning approach with labeled training data. Results: The model achieved 92% accuracy on the test set.",
            'label': 'research_paper'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample training data created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train document classification model')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to training data CSV file')
    parser.add_argument('--model_type', type=str, default='traditional',
                        choices=['traditional', 'transformer'],
                        help='Type of model to train')
    parser.add_argument('--model_output_path', type=str, default='./models/classifier_model.pkl',
                        help='Path to save trained model')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create sample training data')
    parser.add_argument('--config_path', type=str, default='./config/config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_data(args.data_path)
        return
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using default settings.")
        config = {}
    
    try:
        # Load training data
        logger.info(f"Loading training data from: {args.data_path}")
        texts, labels = load_training_data(args.data_path)
        logger.info(f"Loaded {len(texts)} training examples")
        
        # Display label distribution
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"Label distribution:\\n{label_counts}")
        
        # Initialize classifier
        logger.info(f"Initializing {args.model_type} classifier")
        classifier = DocumentClassifier(model_type=args.model_type)
        
        # Split data for evaluation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training set size: {len(train_texts)}")
        logger.info(f"Validation set size: {len(val_texts)}")
        
        # Train model
        logger.info("Starting model training...")
        training_results = classifier.train(train_texts, train_labels, validation_split=0.0)
        
        # Log training results
        logger.info("Training completed!")
        if isinstance(training_results, dict):
            for key, value in training_results.items():
                if key != 'classification_report':  # Skip detailed report for logging
                    logger.info(f"{key}: {value}")
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        validation_results = classifier.evaluate_model(val_texts, val_labels)
        logger.info("Validation Results:")
        for key, value in validation_results.items():
            if key != 'classification_report':
                logger.info(f"{key}: {value}")
        
        # Save model
        os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)
        classifier.save_model(args.model_output_path)
        logger.info(f"Model saved to: {args.model_output_path}")
        
        # Test with sample predictions
        logger.info("\\nTesting with sample predictions:")
        sample_texts = [
            "Invoice for consulting services rendered in January 2024. Amount due: $2,500.",
            "This research study examines the effectiveness of natural language processing techniques.",
            "Service level agreement between client and vendor for cloud hosting services."
        ]
        
        for text in sample_texts:
            prediction = classifier.predict(text)
            if args.model_type == 'traditional':
                probabilities = classifier.predict_proba(text)
                logger.info(f"Text: {text[:50]}...")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Probabilities: {probabilities}")
                logger.info("-" * 50)
            else:
                logger.info(f"Text: {text[:50]}...")
                logger.info(f"Prediction: {prediction}")
                logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
