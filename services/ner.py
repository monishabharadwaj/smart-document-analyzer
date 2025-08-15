"""
Named Entity Recognition (NER) Module

Extracts entities like Organization, Date, Money, Person from documents
using SpaCy and custom trained models.
"""

import spacy
import re
from spacy import displacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from datetime import datetime
from collections import defaultdict
import json

class NERService:
    """Named Entity Recognition service with custom entity extraction capabilities."""
    
    def __init__(self, model_name='en_core_web_sm', custom_model_path=None):
        """
        Initialize the NER service.
        
        Args:
            model_name (str): SpaCy model name to load
            custom_model_path (str): Path to custom trained model
        """
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        
        try:
            if custom_model_path:
                self.nlp = spacy.load(custom_model_path)
            else:
                self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Please install it using: python -m spacy download {model_name}")
            raise
        
        # Custom patterns for enhanced entity recognition
        self._add_custom_patterns()
    
    def _add_custom_patterns(self):
        """Add custom patterns for better entity recognition."""
        # Add entity ruler for custom patterns
        if 'entity_ruler' not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe('entity_ruler', before='ner')
            
            patterns = [
                # Email patterns
                {"label": "EMAIL", "pattern": [
                    {"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}},
                ]},
                
                # Phone number patterns
                {"label": "PHONE", "pattern": [
                    {"TEXT": {"REGEX": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"}},
                ]},
                
                # Social Security Number patterns
                {"label": "SSN", "pattern": [
                    {"TEXT": {"REGEX": r"\d{3}-\d{2}-\d{4}"}},
                ]},
                
                # Credit Card patterns
                {"label": "CREDIT_CARD", "pattern": [
                    {"TEXT": {"REGEX": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"}},
                ]},
                
                # Invoice/Reference number patterns
                {"label": "INVOICE_NUM", "pattern": [
                    {"TEXT": {"REGEX": r"(INV|REF|ORDER)[-#]?\d+"}},
                ]},
            ]
            
            ruler.add_patterns(patterns)
    
    def extract_entities(self, text, include_custom=True):
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            include_custom (bool): Whether to include custom entity types
            
        Returns:
            dict: Dictionary with entity types as keys and lists of entities as values
        """
        doc = self.nlp(text)
        
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'prob', None)
            }
            entities[ent.label_].append(entity_info)
        
        # Additional custom extractions
        if include_custom:
            custom_entities = self._extract_custom_entities(text)
            for label, ents in custom_entities.items():
                entities[label].extend(ents)
        
        return dict(entities)
    
    def _extract_custom_entities(self, text):
        """Extract custom entities using regex patterns."""
        custom_entities = defaultdict(list)
        
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        for match in re.finditer(url_pattern, text):
            custom_entities['URL'].append({
                'text': match.group(),
                'label': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        # Extract IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        for match in re.finditer(ip_pattern, text):
            custom_entities['IP_ADDRESS'].append({
                'text': match.group(),
                'label': 'IP_ADDRESS',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        # Extract postal codes (US format)
        postal_pattern = r'\b\d{5}(?:-\d{4})?\b'
        for match in re.finditer(postal_pattern, text):
            custom_entities['POSTAL_CODE'].append({
                'text': match.group(),
                'label': 'POSTAL_CODE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        return custom_entities
    
    def extract_financial_entities(self, text):
        """Extract financial-specific entities from text."""
        doc = self.nlp(text)
        financial_entities = defaultdict(list)
        
        # Money entities from spaCy
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'PERCENT', 'QUANTITY']:
                financial_entities[ent.label_].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Custom financial patterns
        patterns = {
            'CURRENCY_AMOUNT': r'\$[\d,]+\.?\d*|\d+\.\d{2}\s*(?:USD|EUR|GBP)',
            'PERCENTAGE': r'\d+\.?\d*\s*%',
            'ACCOUNT_NUMBER': r'(?:Account|Acc)\.?\s*#?\s*\d{8,20}',
            'ROUTING_NUMBER': r'(?:Routing|RT)\.?\s*#?\s*\d{9}',
            'TAX_ID': r'(?:Tax|EIN)\.?\s*ID:?\s*\d{2}-\d{7}'
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                financial_entities[label].append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        return dict(financial_entities)
    
    def extract_document_metadata(self, text):
        """Extract metadata entities specific to documents."""
        doc = self.nlp(text)
        metadata_entities = defaultdict(list)
        
        # Extract dates
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                metadata_entities['DATE'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Custom patterns for document metadata
        patterns = {
            'DOCUMENT_ID': r'(?:Document|Doc|ID)\.?\s*#?\s*[A-Z0-9-]+',
            'VERSION': r'(?:Version|Ver|v)\.?\s*\d+\.\d+',
            'PAGE_NUMBER': r'Page\s+\d+\s+of\s+\d+',
            'REFERENCE_NUMBER': r'(?:Ref|Reference)\.?\s*#?\s*[A-Z0-9-]+',
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                metadata_entities[label].append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })
        
        return dict(metadata_entities)
    
    def get_entity_summary(self, text):
        """Get a summary of all entities found in the text."""
        entities = self.extract_entities(text)
        financial_entities = self.extract_financial_entities(text)
        metadata_entities = self.extract_document_metadata(text)
        
        # Merge all entities
        all_entities = {}
        for entity_dict in [entities, financial_entities, metadata_entities]:
            for label, ents in entity_dict.items():
                if label not in all_entities:
                    all_entities[label] = []
                all_entities[label].extend(ents)
        
        # Create summary
        summary = {
            'total_entities': sum(len(ents) for ents in all_entities.values()),
            'entity_types': len(all_entities),
            'entities_by_type': {label: len(ents) for label, ents in all_entities.items()},
            'all_entities': all_entities
        }
        
        return summary
    
    def visualize_entities(self, text, style='ent'):
        """
        Visualize entities in text using spaCy's displacy.
        
        Args:
            text (str): Input text
            style (str): Visualization style ('ent' for entities, 'dep' for dependencies)
            
        Returns:
            str: HTML visualization
        """
        doc = self.nlp(text)
        return displacy.render(doc, style=style, jupyter=False)
    
    def train_custom_ner(self, training_data, model_output_path, n_iter=30):
        """
        Train a custom NER model with new entity types.
        
        Args:
            training_data (list): List of (text, entities) tuples
            model_output_path (str): Path to save the trained model
            n_iter (int): Number of training iterations
            
        Returns:
            dict: Training statistics
        """
        # Create a blank model or load existing model
        if self.nlp.meta['name'] == 'en_core_web_sm':
            nlp = spacy.load('en_core_web_sm')
        else:
            nlp = spacy.blank('en')
            nlp.add_pipe('ner')
        
        ner = nlp.get_pipe('ner')
        
        # Add new entity labels
        for text, annotations in training_data:
            for ent in annotations['entities']:
                ner.add_label(ent[2])
        
        # Train the model
        optimizer = nlp.begin_training()
        losses = {}
        
        for itn in range(n_iter):
            random.shuffle(training_data)
            batches = minibatch(training_data, size=compounding(4., 32., 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    examples.append(Example.from_dict(nlp.make_doc(text), annotations))
                
                nlp.update(examples, sgd=optimizer, losses=losses)
            
            print(f"Iteration {itn + 1}, Losses: {losses}")
        
        # Save the model
        nlp.to_disk(model_output_path)
        
        return {
            'final_losses': losses,
            'model_saved_to': model_output_path,
            'training_examples': len(training_data)
        }
    
    def evaluate_ner_model(self, test_data):
        """
        Evaluate the NER model on test data.
        
        Args:
            test_data (list): List of (text, entities) tuples for testing
            
        Returns:
            dict: Evaluation metrics
        """
        correct_entities = 0
        total_entities = 0
        precision_scores = []
        recall_scores = []
        
        for text, true_annotations in test_data:
            predicted_entities = self.extract_entities(text, include_custom=False)
            true_entities = true_annotations['entities']
            
            # Convert predicted entities to the same format as true entities
            pred_spans = []
            for label, ents in predicted_entities.items():
                for ent in ents:
                    pred_spans.append((ent['start'], ent['end'], label))
            
            true_spans = [(start, end, label) for start, end, label in true_entities]
            
            # Calculate metrics
            correct = len(set(pred_spans) & set(true_spans))
            total_pred = len(pred_spans)
            total_true = len(true_spans)
            
            correct_entities += correct
            total_entities += total_true
            
            if total_pred > 0:
                precision_scores.append(correct / total_pred)
            if total_true > 0:
                recall_scores.append(correct / total_true)
        
        precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'correct_entities': correct_entities,
            'total_entities': total_entities
        }

class EntityLinker:
    """Links extracted entities to knowledge bases or canonical forms."""
    
    def __init__(self):
        self.entity_mappings = {}
        self.load_entity_mappings()
    
    def load_entity_mappings(self):
        """Load entity mappings for normalization."""
        # This would typically load from a database or knowledge base
        self.entity_mappings = {
            'PERSON': {},  # Person name variations
            'ORG': {},     # Organization name variations
            'GPE': {},     # Geopolitical entity variations
        }
    
    def normalize_entity(self, entity_text, entity_type):
        """
        Normalize entity text to canonical form.
        
        Args:
            entity_text (str): The entity text to normalize
            entity_type (str): The type of entity
            
        Returns:
            str: Normalized entity text
        """
        # Simple normalization rules
        normalized = entity_text.strip()
        
        if entity_type == 'PERSON':
            # Capitalize names properly
            normalized = ' '.join(word.capitalize() for word in normalized.split())
        elif entity_type == 'ORG':
            # Handle company suffixes
            suffixes = ['Inc.', 'LLC', 'Corp.', 'Ltd.']
            for suffix in suffixes:
                if normalized.endswith(suffix.lower()):
                    normalized = normalized[:-len(suffix)] + suffix
        elif entity_type in ['MONEY', 'CURRENCY_AMOUNT']:
            # Normalize currency format
            import re
            match = re.search(r'[\d,]+\.?\d*', normalized)
            if match:
                amount = match.group().replace(',', '')
                normalized = f"${amount}"
        
        return normalized
