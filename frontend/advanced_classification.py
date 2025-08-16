#!/usr/bin/env python3
"""
Advanced Document Classification with Real Datasets
Integrates BBC News and Reuters datasets into the document classification system
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from services.classification import DocumentClassifier, DocumentTypeFeatureExtractor
from services.preprocessing import DocumentProcessor

def load_bbc_dataset():
    """Load and preprocess BBC News dataset."""
    try:
        bbc_path = r"C:\Users\Hp\OneDrive\Desktop\Dataset1\bbc_data.csv"
        df = pd.read_csv(bbc_path)
        
        print(f"✅ BBC Dataset loaded: {len(df)} articles")
        print(f"📊 Categories: {df['labels'].value_counts()}")
        
        return df['data'].tolist(), df['labels'].tolist()
    except Exception as e:
        print(f"❌ Error loading BBC dataset: {e}")
        return None, None

def load_reuters_dataset():
    """Load and preprocess Reuters dataset."""
    try:
        reuters_path = r"C:\Users\Hp\OneDrive\Desktop\Dataset2\file.txt"
        
        texts = []
        labels = []
        
        with open(reuters_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header
                    continue
                    
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    category_num = parts[0]
                    text = parts[1] if len(parts) > 1 else ""
                    
                    # Map category numbers to meaningful labels
                    category_map = {
                        '1': 'earnings',      # Earnings reports
                        '2': 'acquisitions',  # Mergers & acquisitions  
                        '3': 'trade',         # Trade news
                        '4': 'transport',     # Transportation
                        '5': 'commodities',   # Grain, oil, commodities
                        '6': 'energy',        # Oil, gas, energy
                        '7': 'interest',      # Interest rates, monetary policy
                        '8': 'money-fx'       # Money market, foreign exchange
                    }
                    
                    label = category_map.get(category_num, 'other')
                    texts.append(text)
                    labels.append(label)
        
        print(f"✅ Reuters Dataset loaded: {len(texts)} articles")
        unique_labels = list(set(labels))
        print(f"📊 Categories: {unique_labels}")
        
        return texts, labels
    except Exception as e:
        print(f"❌ Error loading Reuters dataset: {e}")
        return None, None

def combine_datasets(bbc_texts, bbc_labels, reuters_texts, reuters_labels):
    """Combine BBC and Reuters datasets."""
    if bbc_texts and reuters_texts:
        all_texts = bbc_texts + reuters_texts
        all_labels = bbc_labels + reuters_labels
        
        print(f"\n📊 Combined Dataset:")
        print(f"Total documents: {len(all_texts)}")
        
        # Show label distribution
        label_counts = pd.Series(all_labels).value_counts()
        print(f"Label distribution:\n{label_counts}")
        
        return all_texts, all_labels
    elif bbc_texts:
        return bbc_texts, bbc_labels
    elif reuters_texts:
        return reuters_texts, reuters_labels
    else:
        return None, None

def train_advanced_classifier(texts, labels):
    """Train an advanced document classifier with real datasets."""
    
    print("\n=== Advanced Document Classification Training ===")
    
    # Initialize classifier
    classifier = DocumentClassifier(model_type='traditional')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Training set: {len(X_train)} documents")
    print(f"📊 Test set: {len(X_test)} documents")
    
    # Train model
    print("\n🎯 Training classifier...")
    training_results = classifier.train(X_train, y_train, validation_split=0.2)
    
    print("\n📈 Training Results:")
    print(f"  Training Accuracy: {training_results['train_accuracy']:.4f}")
    print(f"  Validation Accuracy: {training_results['validation_accuracy']:.4f}")
    print(f"  Cross-validation Mean: {training_results['cv_mean']:.4f} (±{training_results['cv_std']:.4f})")
    
    # Evaluate on test set
    print("\n🔬 Testing on holdout set...")
    test_results = classifier.evaluate_model(X_test, y_test)
    
    print(f"\n📊 Test Results:")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"\n📋 Detailed Classification Report:")
    print(test_results['classification_report'])
    
    # Predictions on test set
    predictions = [classifier.predict(text) for text in X_test]
    
    # Confusion Matrix
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    unique_labels = sorted(list(set(labels)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Document Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    
    # Sample predictions
    print("\n🔍 Sample Predictions:")
    for i in range(min(5, len(X_test))):
        text_sample = X_test[i][:100] + "..." if len(X_test[i]) > 100 else X_test[i]
        pred = classifier.predict(X_test[i])
        actual = y_test[i]
        
        print(f"\nText: {text_sample}")
        print(f"Actual: {actual} | Predicted: {pred} {'✅' if pred == actual else '❌'}")
        
        # Show probabilities
        try:
            probabilities = classifier.predict_proba(X_test[i])
            top_3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top predictions: {', '.join([f'{label}: {prob:.3f}' for label, prob in top_3])}")
        except:
            pass
    
    # Save model
    model_path = "models/advanced_document_classifier.pkl"
    os.makedirs("models", exist_ok=True)
    classifier.save_model(model_path)
    print(f"\n💾 Advanced model saved to: {model_path}")
    
    return classifier, test_results

def analyze_feature_importance(classifier, texts, labels):
    """Analyze feature importance in the trained classifier."""
    
    print("\n🔬 Feature Analysis:")
    
    try:
        # Get TF-IDF vectorizer from the pipeline
        vectorizer = classifier.model.named_steps['tfidf']
        rf_classifier = classifier.model.named_steps['classifier']
        
        # Get feature names and importances
        feature_names = vectorizer.get_feature_names_out()
        importances = rf_classifier.feature_importances_
        
        # Get top features
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Top 20 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:20]):
            print(f"{i+1:2d}. {feature:20s} - {importance:.6f}")
            
    except Exception as e:
        print(f"❌ Could not analyze feature importance: {e}")

def test_custom_documents(classifier):
    """Test the classifier on custom documents."""
    
    print("\n🧪 Testing Custom Documents:")
    
    test_docs = [
        {
            'text': "Apple Inc. reported quarterly earnings of $3.2 billion, beating analyst expectations. The iPhone maker saw strong revenue growth in services segment.",
            'expected': 'earnings'
        },
        {
            'text': "Microsoft Corp agreed to acquire GitHub for $7.5 billion in an all-cash deal. The acquisition will help Microsoft expand its developer tools portfolio.",
            'expected': 'acquisitions'
        },
        {
            'text': "Oil prices surged 5% today after OPEC announced production cuts. Crude oil futures reached $75 per barrel amid supply concerns.",
            'expected': 'energy'
        },
        {
            'text': "The new blockbuster movie dominated box office charts this weekend. The action-packed thriller earned $50 million in its opening weekend.",
            'expected': 'entertainment'
        }
    ]
    
    for i, test_doc in enumerate(test_docs, 1):
        text = test_doc['text']
        expected = test_doc['expected']
        
        prediction = classifier.predict(text)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Expected: {expected} | Predicted: {prediction} {'✅' if prediction == expected else '❌'}")
        
        try:
            probabilities = classifier.predict_proba(text)
            top_3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Probabilities: {', '.join([f'{label}: {prob:.3f}' for label, prob in top_3])}")
        except:
            pass

def main():
    """Main function to run advanced document classification."""
    
    print("🚀 Advanced Document Classification with Real Datasets")
    print("=" * 60)
    
    # Load datasets
    print("\n📂 Loading Datasets...")
    bbc_texts, bbc_labels = load_bbc_dataset()
    reuters_texts, reuters_labels = load_reuters_dataset()
    
    # Combine datasets
    if bbc_texts or reuters_texts:
        texts, labels = combine_datasets(bbc_texts, bbc_labels, reuters_texts, reuters_labels)
        
        if texts and len(texts) > 100:  # Ensure we have enough data
            # Train classifier
            classifier, results = train_advanced_classifier(texts, labels)
            
            # Feature analysis
            analyze_feature_importance(classifier, texts, labels)
            
            # Test custom documents
            test_custom_documents(classifier)
            
            print(f"\n✅ Advanced Classification Complete!")
            print(f"📊 Final Test Accuracy: {results['accuracy']:.4f}")
            
        else:
            print("❌ Not enough data to train classifier")
            
    else:
        print("❌ Could not load any datasets")

if __name__ == "__main__":
    main()
