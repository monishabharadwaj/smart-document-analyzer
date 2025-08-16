"""
Document Classification Module

Classifies documents into predefined types (invoice, report, contract, research paper)
using machine learning and NLP models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from .preprocessing import DocumentProcessor

class DocumentClassifier:
    """Document classification using traditional ML and transformer models."""
    
    def __init__(self, model_type='traditional', model_path=None):
        """
        Initialize the document classifier.
        
        Args:
            model_type (str): 'traditional' for sklearn models or 'transformer' for BERT-like models
            model_path (str): Path to saved model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.processor = DocumentProcessor()
        
        # Document type labels
        self.document_types = ['invoice', 'report', 'contract', 'research_paper']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model based on type."""
        if self.model_type == 'traditional':
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                stop_words='english'
            )
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        elif self.model_type == 'transformer':
            # Use a pre-trained BERT model for sequence classification
            model_name = 'microsoft/DialoGPT-medium'  # Can be changed to other models
            self.model = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
    
    def train(self, texts, labels, validation_split=0.2):
        """
        Train the document classification model.
        
        Args:
            texts (list): List of document texts
            labels (list): List of corresponding document type labels
            validation_split (float): Proportion of data for validation
        
        Returns:
            dict: Training metrics
        """
        # Preprocess texts
        processed_texts = [self.processor.preprocess_for_classification(text) for text in texts]
        
        if self.model_type == 'traditional':
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                processed_texts, labels, test_size=validation_split, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            # Cross validation (adjust cv based on data size)
            min_samples_per_class = min([labels.count(label) for label in set(labels)])
            cv_folds = min(5, min_samples_per_class)
            if cv_folds >= 2:
                cv_scores = cross_val_score(self.model, processed_texts, labels, cv=cv_folds)
            else:
                cv_scores = [val_accuracy]  # Fallback if not enough data
            
            return {
                'train_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_val, val_pred)
            }
        else:
            # For transformer models, you would typically use Hugging Face's Trainer
            # This is a simplified version
            print("Transformer training requires more complex setup. Using pre-trained model for now.")
            return {'message': 'Using pre-trained transformer model'}
    
    def predict(self, text):
        """
        Predict the document type for a given text.
        
        Args:
            text (str): Document text to classify
            
        Returns:
            str: Predicted document type
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        processed_text = self.processor.preprocess_for_classification(text)
        
        if self.model_type == 'traditional':
            prediction = self.model.predict([processed_text])[0]
            return prediction
        elif self.model_type == 'transformer':
            result = self.model(processed_text)[0]
            return result['label']
    
    def predict_proba(self, text):
        """
        Get prediction probabilities for all document types.
        
        Args:
            text (str): Document text to classify
            
        Returns:
            dict: Dictionary with document types and their probabilities
        """
        if self.model is None or self.model_type != 'traditional':
            raise ValueError("Probability prediction only available for traditional models")
        
        processed_text = self.processor.preprocess_for_classification(text)
        probabilities = self.model.predict_proba([processed_text])[0]
        
        return dict(zip(self.model.classes_, probabilities))
    
    def save_model(self, path):
        """Save the trained model to disk."""
        if self.model_type == 'traditional':
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'document_types': self.document_types,
                    'model_type': self.model_type
                }, f)
        else:
            # For transformer models, save using Hugging Face methods
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
    
    def load_model(self, path):
        """Load a trained model from disk."""
        if self.model_type == 'traditional':
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.document_types = saved_data['document_types']
                self.model_type = saved_data['model_type']
        else:
            # Load transformer model
            self.model = pipeline("text-classification", model=path)
    
    def evaluate_model(self, test_texts, test_labels):
        """
        Evaluate the model on test data.
        
        Args:
            test_texts (list): List of test texts
            test_labels (list): List of true labels
            
        Returns:
            dict: Evaluation metrics
        """
        processed_texts = [self.processor.preprocess_for_classification(text) for text in test_texts]
        
        if self.model_type == 'traditional':
            predictions = self.model.predict(processed_texts)
            accuracy = accuracy_score(test_labels, predictions)
            report = classification_report(test_labels, predictions)
            
            return {
                'accuracy': accuracy,
                'classification_report': report
            }
        else:
            # Simplified evaluation for transformer models
            correct = 0
            for text, true_label in zip(processed_texts, test_labels):
                pred = self.predict(text)
                if pred == true_label:
                    correct += 1
            
            return {'accuracy': correct / len(test_texts)}

class DocumentTypeFeatureExtractor:
    """Extract features specific to document types for classification."""
    
    @staticmethod
    def extract_invoice_features(text):
        """Extract features specific to invoices."""
        features = {}
        text_lower = text.lower()
        
        # Common invoice keywords
        invoice_keywords = ['invoice', 'bill', 'payment', 'due', 'total', 'amount', 'tax', 'vat']
        features['invoice_keyword_count'] = sum(1 for word in invoice_keywords if word in text_lower)
        
        # Number patterns (amounts, dates)
        import re
        features['currency_patterns'] = len(re.findall(r'\$[\d,]+\.?\d*', text))
        features['date_patterns'] = len(re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text))
        
        return features
    
    @staticmethod
    def extract_contract_features(text):
        """Extract features specific to contracts."""
        features = {}
        text_lower = text.lower()
        
        # Common contract keywords
        contract_keywords = ['agreement', 'contract', 'party', 'terms', 'conditions', 'clause', 'whereas']
        features['contract_keyword_count'] = sum(1 for word in contract_keywords if word in text_lower)
        
        # Legal language patterns
        features['shall_count'] = text_lower.count('shall')
        features['hereby_count'] = text_lower.count('hereby')
        
        return features
    
    @staticmethod
    def extract_report_features(text):
        """Extract features specific to reports."""
        features = {}
        text_lower = text.lower()
        
        # Common report keywords
        report_keywords = ['report', 'analysis', 'summary', 'findings', 'conclusion', 'methodology']
        features['report_keyword_count'] = sum(1 for word in report_keywords if word in text_lower)
        
        # Structural elements
        features['section_count'] = text_lower.count('section')
        features['figure_count'] = text_lower.count('figure')
        
        return features
    
    @staticmethod
    def extract_research_paper_features(text):
        """Extract features specific to research papers."""
        features = {}
        text_lower = text.lower()
        
        # Academic keywords
        academic_keywords = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references']
        features['academic_keyword_count'] = sum(1 for word in academic_keywords if word in text_lower)
        
        # Citation patterns
        import re
        features['citation_count'] = len(re.findall(r'\[\d+\]', text))
        features['doi_count'] = len(re.findall(r'doi:', text_lower))
        
        return features
