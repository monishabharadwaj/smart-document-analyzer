"""
Document Summarization Module

Generates concise summaries for long documents using transformer-based models
and traditional extractive summarization techniques.
"""

import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer
from .preprocessing import DocumentProcessor

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DocumentSummarizer:
    """Document summarization using both extractive and abstractive methods."""
    
    def __init__(self, model_type='extractive', model_name=None, max_chunk_length=1024):
        """
        Initialize the document summarizer.
        
        Args:
            model_type (str): 'extractive' or 'abstractive'
            model_name (str): Specific model name to use
            max_chunk_length (int): Maximum length for model input chunks
        """
        self.model_type = model_type
        self.max_chunk_length = max_chunk_length
        self.processor = DocumentProcessor()
        
        # Initialize models based on type
        if model_type == 'extractive':
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.stop_words = set(stopwords.words('english'))
        elif model_type == 'abstractive':
            self._initialize_abstractive_model(model_name)
        elif model_type == 'hybrid':
            # Initialize both for hybrid approach
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.stop_words = set(stopwords.words('english'))
            self._initialize_abstractive_model(model_name)
    
    def _initialize_abstractive_model(self, model_name=None):
        """Initialize abstractive summarization model."""
        if model_name is None:
            model_name = 'facebook/bart-large-cnn'  # Default model
        
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.abstractive_model = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to a smaller model
            self.abstractive_model = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=device
            )
            self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    
    def summarize(self, text, summary_ratio=0.3, max_length=None, min_length=None):
        """
        Generate a summary of the input text.
        
        Args:
            text (str): Input text to summarize
            summary_ratio (float): Ratio of sentences to include in extractive summary
            max_length (int): Maximum length for abstractive summary
            min_length (int): Minimum length for abstractive summary
            
        Returns:
            str: Generated summary
        """
        if self.model_type == 'extractive':
            return self._extractive_summarize(text, summary_ratio)
        elif self.model_type == 'abstractive':
            return self._abstractive_summarize(text, max_length, min_length)
        elif self.model_type == 'hybrid':
            return self._hybrid_summarize(text, summary_ratio, max_length, min_length)
    
    def _extractive_summarize(self, text, summary_ratio=0.3):
        """
        Generate extractive summary by selecting important sentences.
        
        Args:
            text (str): Input text
            summary_ratio (float): Ratio of sentences to include
            
        Returns:
            str: Extractive summary
        """
        # Preprocess text lightly for summarization
        cleaned_text = self.processor.preprocess_for_summarization(text)
        sentences = sent_tokenize(cleaned_text)
        
        if len(sentences) <= 3:
            return cleaned_text
        
        # Calculate sentence embeddings
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sentence_embeddings)
        
        # Calculate sentence scores based on similarity to other sentences
        sentence_scores = np.mean(similarity_matrix, axis=1)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * summary_ratio))
        top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_sentence_indices = sorted(top_sentence_indices)
        
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        return ' '.join(summary_sentences)
    
    def _abstractive_summarize(self, text, max_length=None, min_length=None):
        """
        Generate abstractive summary using transformer models.
        
        Args:
            text (str): Input text
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            
        Returns:
            str: Abstractive summary
        """
        # Handle long texts by chunking
        chunks = self._chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            chunk_tokens = len(self.tokenizer.encode(chunk))
            
            # Set default lengths based on input length
            if max_length is None:
                max_length = min(150, max(30, chunk_tokens // 4))
            if min_length is None:
                min_length = min(30, max_length // 3)
            
            try:
                summary = self.abstractive_model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                # Fallback to extractive for this chunk
                summaries.append(self._extractive_summarize(chunk, 0.3))
        
        # If multiple chunks, combine and potentially summarize again
        if len(summaries) > 1:
            combined_summary = ' '.join(summaries)
            if len(combined_summary.split()) > max_length * 2:
                return self._abstractive_summarize(combined_summary, max_length, min_length)
            return combined_summary
        else:
            return summaries[0] if summaries else ""
    
    def _hybrid_summarize(self, text, summary_ratio=0.3, max_length=None, min_length=None):
        """
        Generate hybrid summary using both extractive and abstractive methods.
        
        Args:
            text (str): Input text
            summary_ratio (float): Ratio for extractive summary
            max_length (int): Maximum length for final summary
            min_length (int): Minimum length for final summary
            
        Returns:
            str: Hybrid summary
        """
        # First, get extractive summary to reduce text length
        extractive_summary = self._extractive_summarize(text, summary_ratio)
        
        # Then apply abstractive summarization to the extractive summary
        if len(extractive_summary.split()) > 50:
            return self._abstractive_summarize(extractive_summary, max_length, min_length)
        else:
            return extractive_summary
    
    def _chunk_text(self, text):
        """
        Split text into chunks that fit within model limits.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_length + sentence_tokens <= self.max_chunk_length:
                current_chunk += sentence + " "
                current_length += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_length = sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_by_sections(self, text, section_headers=None):
        """
        Summarize text by sections for better structure preservation.
        
        Args:
            text (str): Input text
            section_headers (list): List of section headers to look for
            
        Returns:
            dict: Dictionary with section summaries
        """
        if section_headers is None:
            section_headers = [
                'introduction', 'background', 'methodology', 'methods',
                'results', 'discussion', 'conclusion', 'summary',
                'executive summary', 'abstract'
            ]
        
        sections = self._extract_sections(text, section_headers)
        section_summaries = {}
        
        for section_name, section_text in sections.items():
            if len(section_text.strip()) > 100:  # Only summarize substantial sections
                section_summaries[section_name] = self.summarize(
                    section_text, summary_ratio=0.4
                )
            else:
                section_summaries[section_name] = section_text
        
        return section_summaries
    
    def _extract_sections(self, text, section_headers):
        """Extract text sections based on headers."""
        sections = {}
        current_section = 'introduction'
        current_text = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            is_header = False
            for header in section_headers:
                if header.lower() in line_lower and len(line_lower) < 100:
                    # Save previous section
                    if current_text:
                        sections[current_section] = '\n'.join(current_text)
                    
                    # Start new section
                    current_section = header.lower()
                    current_text = []
                    is_header = True
                    break
            
            if not is_header:
                current_text.append(line)
        
        # Add the last section
        if current_text:
            sections[current_section] = '\n'.join(current_text)
        
        return sections
    
    def generate_bullet_summary(self, text, num_bullets=5):
        """
        Generate a bullet-point summary of the text.
        
        Args:
            text (str): Input text
            num_bullets (int): Number of bullet points to generate
            
        Returns:
            list: List of bullet point summaries
        """
        # Use extractive method to get key sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= num_bullets:
            return [sentence.strip() for sentence in sentences]
        
        # Get sentence embeddings and scores
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        similarity_matrix = cosine_similarity(sentence_embeddings)
        sentence_scores = np.mean(similarity_matrix, axis=1)
        
        # Select top sentences
        top_indices = np.argsort(sentence_scores)[-num_bullets:]
        top_indices = sorted(top_indices)
        
        bullet_points = []
        for i in top_indices:
            sentence = sentences[i].strip()
            # Clean up and format as bullet point
            if not sentence.endswith('.'):
                sentence += '.'
            bullet_points.append(sentence)
        
        return bullet_points
    
    def get_summary_statistics(self, original_text, summary_text):
        """
        Get statistics about the summarization.
        
        Args:
            original_text (str): Original text
            summary_text (str): Generated summary
            
        Returns:
            dict: Summary statistics
        """
        original_words = len(original_text.split())
        original_sentences = len(sent_tokenize(original_text))
        original_chars = len(original_text)
        
        summary_words = len(summary_text.split())
        summary_sentences = len(sent_tokenize(summary_text))
        summary_chars = len(summary_text)
        
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return {
            'original_stats': {
                'words': original_words,
                'sentences': original_sentences,
                'characters': original_chars
            },
            'summary_stats': {
                'words': summary_words,
                'sentences': summary_sentences,
                'characters': summary_chars
            },
            'compression_ratio': compression_ratio,
            'reduction_percentage': (1 - compression_ratio) * 100
        }

class KeyPhraseSummarizer:
    """Extractive summarization based on key phrase extraction."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.processor = DocumentProcessor()
    
    def extract_key_phrases(self, text, num_phrases=10):
        """
        Extract key phrases from text using TF-IDF.
        
        Args:
            text (str): Input text
            num_phrases (int): Number of key phrases to extract
            
        Returns:
            list: List of key phrases with scores
        """
        # Clean and preprocess text
        cleaned_text = self.processor.clean_text(text)
        sentences = sent_tokenize(text)  # Use original text for sentences
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=1000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top phrases
            top_indices = np.argsort(mean_scores)[-num_phrases:]
            key_phrases = []
            
            for idx in reversed(top_indices):
                phrase = feature_names[idx]
                score = mean_scores[idx]
                key_phrases.append({'phrase': phrase, 'score': score})
            
            return key_phrases
        
        except Exception as e:
            print(f"Error extracting key phrases: {e}")
            return []
    
    def summarize_with_key_phrases(self, text, summary_ratio=0.3):
        """
        Summarize text by selecting sentences that contain key phrases.
        
        Args:
            text (str): Input text
            summary_ratio (float): Ratio of sentences to include
            
        Returns:
            str: Summary based on key phrases
        """
        sentences = sent_tokenize(text)
        key_phrases = self.extract_key_phrases(text, num_phrases=20)
        
        if not key_phrases:
            return text[:1000] + "..." if len(text) > 1000 else text
        
        # Score sentences based on key phrase presence
        sentence_scores = []
        key_phrase_texts = [kp['phrase'] for kp in key_phrases]
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            for phrase in key_phrase_texts:
                if phrase.lower() in sentence_lower:
                    phrase_data = next((kp for kp in key_phrases if kp['phrase'] == phrase), None)
                    if phrase_data:
                        score += phrase_data['score']
            
            sentence_scores.append(score)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * summary_ratio))
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
